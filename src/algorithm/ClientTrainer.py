# Multi Modal Federated Learning algorithm in each client
from timm.models import create_model
from utils import utils
from pathlib import Path
import random
import os
import glob
import torch
import copy
import numpy as np
import torch.distributed as dist
from datasets.dataset import create_dataset_by_split, create_dataloader
from optims.optim import create_optimizer, LayerDecayValueAssigner, get_is_head_flag_for_vit
from utils.utils import NativeScalerWithGradNormCount as NativeScaler
from algorithm.BaseTrainer import train_one_epoch, VQAHandler, evaluate


class ClientTrainer:
    def __init__(self, args, model, client_id, server_path, client_path, steps_per_epoch, logger, 
                sample_num_dict, neighbour_dict, barrier, io_lock):
        self.args = args
        self.model = model
        self.client_id = client_id
        self.client_path = client_path
        self.server_path = server_path
        self.logger = logger
        self.num_training_steps_per_epoch = steps_per_epoch

        self.sample_num_dict = sample_num_dict
        self.neighbour_dict = neighbour_dict
        self.barrier = barrier
        self.io_lock = io_lock
        print(f"Neighbours of client {client_id}: {neighbour_dict[client_id]}")

        self.total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
        self.assigner = LayerDecayValueAssigner([1.0, 20.0], scale_handler=get_is_head_flag_for_vit)

        self.lr_schedule_values = utils.cosine_scheduler(
            base_value=self.args.lr, final_value=self.args.min_lr, epochs=self.args.local_epochs,
            niter_per_ep=steps_per_epoch, warmup_epochs=self.args.warmup_epochs, sched_type="linear"
        )

        self.model_ema = None
        self.update_freq = 1
        self.clip_grad = None

        self.prototypes = {"img": torch.randn([args.num_classes, args.embed_dim], dtype=torch.float32),
                           "text": torch.randn([args.num_classes, args.embed_dim], dtype=torch.float32),
                           "fusion": torch.randn([args.num_classes, args.embed_dim], dtype=torch.float32),
                           "target": torch.randn([args.num_classes, args.num_classes], dtype=torch.float32)}

    # Main algorithm for decentralized-pmcmFL
    def run(self, train_loader_img_text, train_loader_img_only, train_loader_text_only,
            val_loader, loader_for_prototype, device, n_comm_rounds, global_batch_size):

        for cur_round in range(1, n_comm_rounds + 1):

            # Set mse loss
            if self.args.prototype_as_rep_target:
                if cur_round >= self.args.regularization_start_round:
                    if not self.args.no_img_proto_target:
                        self.args.img_proto_target = True
                    if not self.args.no_text_proto_target:
                        self.args.text_proto_target = True
                    if not self.args.no_fusion_proto_target:
                        self.args.fusion_proto_target = True
            
            # Create directory for current round if it does not exist
            self.logger.write("client%d in communication round %d." % (self.client_id, cur_round))
            self.client_path = self.args.output_dir + '/client-round%d' % cur_round
            with self.io_lock:
                Path(self.client_path).mkdir(parents=True, exist_ok=True)

            # Local train
            start_global_epoch = (cur_round - 1) * self.args.local_epochs
            num_epochs = self.args.local_epochs

            self.logger.write("start local training client%d:" % (self.client_id))
            print("------- start local training client%d:" % (self.client_id))

            with self.io_lock:
                for cur_local_epoch in range(num_epochs):
                
                    self.logger.write("start training client%d img-text." % (self.client_id))
                    if self.sample_num_dict[self.client_id]["img-text"] >= global_batch_size:
                        self.local_train_one_epoch(cur_global_epoch=start_global_epoch + cur_local_epoch,
                                            cur_local_epoch=cur_local_epoch, step_offset=0,
                                            train_dataloader=train_loader_img_text, val_dataloader=val_loader,
                                            device=device, mode="vl")
                        torch.distributed.barrier()

                    if self.args.modal_missing:
                        if np.random.binomial(n=1, p=0.5, size=1)[0]:
                            self.logger.write("client%d, train img-only first." % (self.client_id))

                            offset_step = self.sample_num_dict[self.client_id]["img-text"] // global_batch_size
                            self.logger.write("offset step is %d for client%d, img-only" % (offset_step,self.client_id))
                            if self.sample_num_dict[self.client_id]["img-only"] >= global_batch_size:
                                self.local_train_one_epoch(cur_global_epoch=start_global_epoch + cur_local_epoch,
                                                    cur_local_epoch=cur_local_epoch, step_offset=offset_step,
                                                    train_dataloader=train_loader_img_only, val_dataloader=val_loader,
                                                    device=device, mode="v")
                                torch.distributed.barrier()

                            offset_step += self.sample_num_dict[self.client_id]["img-only"] // global_batch_size
                            self.logger.write("offset step is %d for client%d, text-only" % (offset_step,self.client_id))
                            if self.sample_num_dict[self.client_id]["text-only"] >= global_batch_size:
                                self.local_train_one_epoch(cur_global_epoch=start_global_epoch + cur_local_epoch,
                                                    cur_local_epoch=cur_local_epoch, step_offset=offset_step,
                                                    train_dataloader=train_loader_text_only, val_dataloader=val_loader,
                                                    device=device, mode="l")
                        else:
                            self.logger.write("client%d, train text-only first." % (self.client_id))

                            offset_step = self.sample_num_dict[self.client_id]["img-text"] // global_batch_size
                            self.logger.write("offset step is %d for client%d, text-only" % (offset_step,self.client_id))
                            if self.sample_num_dict[self.client_id]["text-only"] >= global_batch_size:
                                self.local_train_one_epoch(cur_global_epoch=start_global_epoch + cur_local_epoch,
                                                    cur_local_epoch=cur_local_epoch, step_offset=offset_step,
                                                    train_dataloader=train_loader_text_only, val_dataloader=val_loader,
                                                    device=device, mode="l")
                                torch.distributed.barrier()

                            offset_step += self.sample_num_dict[self.client_id]["text-only"] // global_batch_size
                            self.logger.write("offset step is %d for client%d, img-only" % (offset_step,self.client_id))
                            if self.sample_num_dict[self.client_id]["img-only"] >= global_batch_size:
                                self.local_train_one_epoch(cur_global_epoch=start_global_epoch + cur_local_epoch,
                                                    cur_local_epoch=cur_local_epoch, step_offset=offset_step,
                                                    train_dataloader=train_loader_img_only, val_dataloader=val_loader,
                                                    device=device, mode="v")
                        torch.distributed.barrier()
            
            self.logger.write("finish training client%d in round%d " % (self.client_id, cur_round))
            print("------- finish training client%d in round%d " % (self.client_id, cur_round))
            torch.distributed.barrier()

            # Save model
            with self.io_lock:
              self.save_model("client%d_model.pth" % self.client_id)
  
            # Compute and save local prototypes
            if self.args.prototype_as_rep_target or self.args.prototype_as_missing_modal:
                self.compute_prototypes(dataloader=loader_for_prototype, device=device)
                with self.io_lock:
                    self.save_prototypes(ckpt_name="client%d_prototypes.pth" % self.client_id)
                self.logger.write("get prototypes for client%d in round%d " % (self.client_id, cur_round))
                torch.distributed.barrier()

            print(f"------- Client {self.client_id} saved model and prototypes -------")
            # Wait for all clients to finish local training
            self.barrier.wait()

            # Agreggate models of neighbours
            self.logger.write("start fedavg in client%d." % (self.client_id))
            self.fedavg(party_list=self.neighbour_dict[self.client_id], cur_round=cur_round)
            self.logger.write("finish fedavg in client%d." % (self.client_id))
            torch.distributed.barrier()

            # Agreggate prototypes of neighbours
            if self.args.prototype_as_rep_target or self.args.prototype_as_missing_modal:
                self.logger.write("start aggregating global prototypes in client%d." % (self.client_id))
                self.aggregate_local_prototypes(party_list=self.neighbour_dict[self.client_id], cur_round=cur_round)
                self.logger.write("finish aggregating global prototypes in client%d." % (self.client_id))
                torch.distributed.barrier()

            print("--------------------- finish round%d in client%d----------------------" % (cur_round,self.client_id))     

        # We save the last model and prototypes of client 0 in the server folder
        
    def local_train_one_epoch(self, cur_local_epoch, cur_global_epoch, step_offset, device, train_dataloader, val_dataloader, mode):
        self.model.to(device)
        for key in self.prototypes.keys():
            self.prototypes[key] = self.prototypes[key].to(device)

        loss_scaler = NativeScaler()

        device_ids = [self.args.gpu] if self.args.device == 'cuda' else None # As per https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html

        model_ddp = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=device_ids,
                                                              find_unused_parameters=True)

        optimizer = create_optimizer(
            self.args, self.model, skip_list=self.model.no_weight_decay(),
            get_num_layer=self.assigner.get_layer_id, get_layer_scale=self.assigner.get_scale
        )

        task_handler = VQAHandler()

        train_dataloader.sampler.set_epoch(cur_global_epoch)
        train_one_epoch(
            self.args, model_ddp, train_dataloader, optimizer, device, task_handler, cur_global_epoch,
            cur_local_epoch, cur_local_epoch * self.num_training_steps_per_epoch + step_offset,
            self.lr_schedule_values, loss_scaler, self.prototypes, self.clip_grad, self.update_freq,
            self.model_ema, mode, self.logger
        )
        torch.distributed.barrier()

        self.model.set_mode("vl")
        self.model.cpu()
        for key in self.prototypes.keys():
            self.prototypes[key] = self.prototypes[key].cpu()
        torch.distributed.barrier()
      
    def load_model(self):
        all_global_model = glob.glob(os.path.join(self.server_path, 'global_model-*.pth'))
        latest_ckpt = -1
        for ckpt in all_global_model:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            latest_global_model = os.path.join(self.server_path, 'global_model-%d.pth' % latest_ckpt)
            self.logger.write("load global model global_model-%d.pth" % latest_ckpt)
            self.model.load_state_dict(torch.load(latest_global_model))
        else:
            self.logger.write("communication round 0, start with pretrain params")

    def save_model(self, ckpt_name):
        if utils.is_main_process():
            state_dict = self.model.state_dict()
            torch.save(state_dict, self.client_path + "/" + ckpt_name)

    def compute_prototypes(self, dataloader, device):

        self.model.set_mode("vl")
        self.model.eval()
        self.model.to(device)

        img_rep_box = None
        text_rep_box = None
        fusion_rep_box = None
        target_box = None
        label_box = None

        with torch.no_grad():
            for idx, data in enumerate(dataloader):
                image = data["image"].to(device, non_blocking=True)
                language_tokens = data["language_tokens"].to(device, non_blocking=True)
                padding_mask = data["padding_mask"].to(device, non_blocking=True)
                logits, hidden_reps = self.model(image=image, question=language_tokens, padding_mask=padding_mask)

                img_rep = hidden_reps["img_rep"].clone().detach()
                text_rep = hidden_reps["text_rep"].clone().detach()
                fusion_rep = hidden_reps["fusion_rep"].clone().detach()
                soft_logit = logits.clone().detach() / self.args.kd_temperature
                target = torch.nn.functional.softmax(soft_logit, dim=-1)
                labels = data["labels"].to(device)

                img_rep_box = img_rep if img_rep_box is None else torch.cat([img_rep_box, img_rep], dim=0)
                text_rep_box = text_rep if text_rep_box is None else torch.cat([text_rep_box, text_rep], dim=0)
                fusion_rep_box = fusion_rep if fusion_rep_box is None else torch.cat([fusion_rep_box, fusion_rep], dim=0)
                target_box = target if target_box is None else torch.cat([target_box, target], dim=0)
                label_box = labels if label_box is None else torch.cat([label_box, labels], dim=0)

            img_prototype_sum = torch.matmul(label_box.T, img_rep_box)
            text_prototype_sum = torch.matmul(label_box.T, text_rep_box)
            fusion_prototype_sum = torch.matmul(label_box.T, fusion_rep_box)
            target_prototype_sum = torch.matmul(label_box.T, target_box)
            total_weight_per_class = torch.sum(label_box.T, dim=1, keepdim=True)

            world_size = utils.get_world_size()
            img_prototypes_sum_list = [torch.zeros_like(img_prototype_sum) for _ in range(world_size)]
            text_prototypes_sum_list = [torch.zeros_like(text_prototype_sum) for _ in range(world_size)]
            fusion_prototypes_sum_list = [torch.zeros_like(fusion_prototype_sum) for _ in range(world_size)]
            target_prototype_sum_list = [torch.zeros_like(target_prototype_sum) for _ in range(world_size)]
            total_weight_per_class_list = [torch.zeros_like(total_weight_per_class) for _ in range(world_size)]
            dist.all_gather(img_prototypes_sum_list, img_prototype_sum)
            dist.all_gather(text_prototypes_sum_list, text_prototype_sum)
            dist.all_gather(fusion_prototypes_sum_list, fusion_prototype_sum)
            dist.all_gather(target_prototype_sum_list, target_prototype_sum)
            dist.all_gather(total_weight_per_class_list, total_weight_per_class)

            for i in range(world_size):
                if i == 0:
                    all_total_weight_per_class = total_weight_per_class_list[i]
                    all_img_prototype_sum = img_prototypes_sum_list[i]
                    all_text_prototype_sum = text_prototypes_sum_list[i]
                    all_fusion_prototype_sum = fusion_prototypes_sum_list[i]
                    all_target_prototype_sum = target_prototype_sum_list[i]
                else:
                    all_total_weight_per_class += total_weight_per_class_list[i]
                    all_img_prototype_sum += img_prototypes_sum_list[i]
                    all_text_prototype_sum += text_prototypes_sum_list[i]
                    all_fusion_prototype_sum += fusion_prototypes_sum_list[i]
                    all_target_prototype_sum += target_prototype_sum_list[i]

            img_prototypes = all_img_prototype_sum / all_total_weight_per_class
            text_prototypes = all_text_prototype_sum / all_total_weight_per_class
            fusion_prototypes = all_fusion_prototype_sum / all_total_weight_per_class
            target_prototypes = all_target_prototype_sum / all_total_weight_per_class

            # replace Nan
            img_prototypes = torch.where(torch.isnan(img_prototypes), torch.full_like(img_prototypes, 0.0), img_prototypes)
            text_prototypes = torch.where(torch.isnan(text_prototypes), torch.full_like(text_prototypes, 0.0), text_prototypes)
            fusion_prototypes = torch.where(torch.isnan(fusion_prototypes), torch.full_like(fusion_prototypes, 0.0), fusion_prototypes)
            target_prototypes = torch.where(torch.isnan(target_prototypes), torch.full_like(target_prototypes, 0.0), target_prototypes)

            img_prototype_new = img_prototypes.detach().cpu()
            text_prototype_new = text_prototypes.detach().cpu()
            fusion_prototype_new = fusion_prototypes.detach().cpu()
            target_prototypes_new = target_prototypes.detach().cpu()

            zero_prototype = torch.zeros([1, self.args.embed_dim], device="cpu", dtype=torch.float32)
            for clas in range(self.args.num_classes):
                if not torch.equal(img_prototype_new[clas:clas+1, :], zero_prototype):
                    self.prototypes["img"][clas:clas+1, :] = img_prototype_new[clas:clas+1, :]
                if not torch.equal(text_prototype_new[clas:clas+1, :], zero_prototype):
                    self.prototypes["text"][clas:clas+1, :] = text_prototype_new[clas:clas+1, :]
                if not torch.equal(fusion_prototype_new[clas:clas+1, :], zero_prototype):
                    self.prototypes["fusion"][clas:clas+1, :] = fusion_prototype_new[clas:clas+1, :]
            zero_target = torch.zeros([1, self.args.num_classes], device="cpu", dtype=torch.float32)
            for clas in range(self.args.num_classes):
                if not torch.equal(target_prototypes_new[clas:clas+1, :], zero_target):
                    self.prototypes["target"][clas:clas+1, :] = target_prototypes_new[clas:clas+1, :]

        self.model.cpu()
        self.model.train()
        self.logger.write("finish computing prototypes in client%d." % (self.client_id))

    def save_prototypes(self, ckpt_name):
        if utils.is_main_process():
            torch.save(self.prototypes, self.client_path + "/" + ckpt_name)

    def load_global_prototypes(self):
        all_global_proto = glob.glob(os.path.join(self.server_path, 'global_prototypes-*.pth'))
        latest_proto = -1
        for proto in all_global_proto:
            t = proto.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_proto = max(int(t), latest_proto)
        if latest_proto >= 0:
            global_prototypes_path = os.path.join(self.server_path, 'global_prototypes-%d.pth' % latest_proto)
            self.logger.write("load global prototypes global_prototypes-%d.pth" % latest_proto)
            self.prototypes = torch.load(global_prototypes_path)
        else:
            self.logger.write("no global prototypes!")

    def fedavg(self, party_list, cur_round, aggregation_factor="total"):
        if utils.is_main_process():
            global_state_dict = {}

            # get fedavg weight for each client
            sample_per_party_client = []
            for client_id in party_list:
                sample_per_party_client.append(float(self.sample_num_dict[client_id][aggregation_factor]))
            total_sample_cur_round = sum(sample_per_party_client)
            fed_avg_freqs = [i / total_sample_cur_round for i in sample_per_party_client]
            self.logger.write("fedavg weight:")
            self.logger.write(fed_avg_freqs)

            # get parameters for each client
            state_dicts = []
            all_client_model_path = []
            for client_id in party_list:
                all_client_model_path.append(os.path.join(self.client_path, 'client%d_model.pth' % client_id))
                print(f"Client {self.client_id} aggregating model {self.client_path}/client{client_id}_model.pth")
            for ckpt in all_client_model_path:
                state_dicts.append(torch.load(ckpt))
            for idx, state_dict in enumerate(state_dicts):
                if idx == 0:
                    for key in state_dict:
                        global_state_dict[key] = state_dict[key] * fed_avg_freqs[idx]
                else:
                    for key in state_dict:
                        global_state_dict[key] += state_dict[key] * fed_avg_freqs[idx]

            # Update global model
            self.model.load_state_dict(global_state_dict) # correcto?
            #torch.save(global_state_dict, self.server_path + '/' + 'global_model-%d.pth' % cur_round)
            print("finish updating global model in client%d." % (self.client_id))

    def aggregate_local_prototypes(self, party_list, cur_round):
        if utils.is_main_process():
            global_prototypes = {}

            # get aggregation weight for prototypes
            img_sample_per_party_client = []
            text_sample_per_party_client = []
            fusion_sample_per_party_client = []
            target_sample_per_party_client = []
            for client_id in party_list:
                img_sample_nums = 1 + self.sample_num_dict[client_id]["img-text"] + self.sample_num_dict[client_id]["img-only"]
                text_sample_nums = 1 + self.sample_num_dict[client_id]["img-text"] + self.sample_num_dict[client_id]["text-only"]
                fusion_sample_nums = 1 + self.sample_num_dict[client_id]["img-text"]
                target_sample_nums = 1 + self.sample_num_dict[client_id]["img-text"]
                img_sample_per_party_client.append(float(img_sample_nums))
                text_sample_per_party_client.append(float(text_sample_nums))
                fusion_sample_per_party_client.append(float(fusion_sample_nums))
                target_sample_per_party_client.append(float(target_sample_nums))
            img_proto_aggre_weight = [i / sum(img_sample_per_party_client) for i in img_sample_per_party_client]
            text_proto_aggre_weight = [i / sum(text_sample_per_party_client) for i in text_sample_per_party_client]
            fusion_proto_aggre_weight = [i / sum(fusion_sample_per_party_client) for i in fusion_sample_per_party_client]
            target_proto_aggre_weight = [i / sum(target_sample_per_party_client) for i in target_sample_per_party_client]

            # get clients' prototypes
            all_client_prototypes_box = []
            all_client_prototype_path = []
            for client_id in party_list:
                all_client_prototype_path.append(os.path.join(self.client_path, 'client%d_prototypes.pth' % client_id))
                print(f"Client {self.client_id} aggregating prototypes {self.client_path}/client{client_id}_prototypes.pth")
            for prototype_path in all_client_prototype_path:
                all_client_prototypes_box.append(torch.load(prototype_path))
            for idx, prototypes in enumerate(all_client_prototypes_box):
                if idx == 0:
                    global_prototypes["img"] = prototypes["img"] * img_proto_aggre_weight[idx]
                    global_prototypes["text"] = prototypes["text"] * text_proto_aggre_weight[idx]
                    global_prototypes["fusion"] = prototypes["fusion"] * fusion_proto_aggre_weight[idx]
                    global_prototypes["target"] = prototypes["target"] * target_proto_aggre_weight[idx]
                else:
                    global_prototypes["img"] += prototypes["img"] * img_proto_aggre_weight[idx]
                    global_prototypes["text"] += prototypes["text"] * text_proto_aggre_weight[idx]
                    global_prototypes["fusion"] += prototypes["fusion"] * fusion_proto_aggre_weight[idx]
                    global_prototypes["target"] += prototypes["target"] * target_proto_aggre_weight[idx]

            # regular (not affect by zero-prototype)
            zero_proto = torch.zeros([1, self.args.embed_dim], device="cpu", dtype=torch.float32)
            zero_target = torch.zeros([1, self.args.num_classes], device="cpu", dtype=torch.float32)
            total_img_weight_per_class = [0.0 for _ in range(self.args.num_classes)]
            total_text_weight_per_class = [0.0 for _ in range(self.args.num_classes)]
            total_fusion_weight_per_class = [0.0 for _ in range(self.args.num_classes)]
            total_target_weight_per_class = [0.0 for _ in range(self.args.num_classes)]
            for idx, prototypes in enumerate(all_client_prototypes_box):
                for clas in range(self.args.num_classes):
                    if not torch.equal(prototypes["img"][clas:clas+1, :], zero_proto):
                        total_img_weight_per_class[clas] += img_proto_aggre_weight[idx]
                    if not torch.equal(prototypes["text"][clas:clas+1, :], zero_proto):
                        total_text_weight_per_class[clas] += text_proto_aggre_weight[idx]
                    if not torch.equal(prototypes["fusion"][clas:clas+1, :], zero_proto):
                        total_fusion_weight_per_class[clas] += fusion_proto_aggre_weight[idx]
                    if not torch.equal(prototypes["target"][clas:clas+1, :], zero_target):
                        total_target_weight_per_class[clas] += target_proto_aggre_weight[idx]
            for clas in range(self.args.num_classes):
                if np.abs(total_img_weight_per_class[clas]) > 0.0001:
                    global_prototypes["img"][clas:clas+1, :] /= total_img_weight_per_class[clas]
                if np.abs(total_text_weight_per_class[clas]) > 0.0001:
                    global_prototypes["text"][clas:clas+1, :] /= total_text_weight_per_class[clas]
                if np.abs(total_fusion_weight_per_class[clas]) > 0.0001:
                    global_prototypes["fusion"][clas:clas+1, :] /= total_fusion_weight_per_class[clas]
                if np.abs(total_target_weight_per_class[clas]) > 0.0001:
                    global_prototypes["target"][clas:clas+1, :] /= total_target_weight_per_class[clas]

            # Update global prototypes
            self.prototypes = global_prototypes # correcto?
            #torch.save(global_prototypes, os.path.join(self.server_path, 'global_prototypes-%d.pth' % cur_round))
