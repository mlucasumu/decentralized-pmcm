# Multi Modal Federated Learning algorithm
from timm.models import create_model
from algorithm.ClientTrainer import ClientTrainer
from utils import utils
from pathlib import Path
import random
import os
import glob
import torch
import numpy as np
from datasets.dataset import create_dataset_by_split
import threading
import torch.multiprocessing as mp 


class MMFL:
    def __init__(self, args, client_nums, sample_num_dict, neighbour_dict, device, val_split, train_split_list_img_text,
                 train_split_list_img_only, train_split_list_text_only, logger):
        self.args = args
        self.client_nums = client_nums
        self.sample_num_dict = sample_num_dict
        self.neighbour_dict = neighbour_dict
        self.device = device
        self.val_split = val_split
        self.train_split_list_img_text = train_split_list_img_text
        self.train_split_list_img_only = train_split_list_img_only
        self.train_split_list_text_only = train_split_list_text_only
        self.logger = logger

        self.barrier = threading.Barrier(client_nums)  # Per-round synchronization
        self.io_lock = threading.Lock()                # Protection when reading/writing files

        self.client_path = None
        self.server_path = args.output_dir + "/server"
        Path(self.server_path).mkdir(parents=True, exist_ok=True)

        self.model_config = "beit3_base_patch16_480_vqav2"

    def run(self, n_comm_rounds, selected_client_nums):

        threads = []
        for client_id in range(self.client_nums):

            self.logger.write("----- initializing client%d ------" % (client_id))

            # Create model
            model = create_model(
                        self.model_config,
                        pretrained=False,
                        drop_path_rate=self.args.drop_path,
                        vocab_size=self.args.vocab_size,
                        checkpoint_activations=self.args.checkpoint_activations,
            )

            utils.load_model_and_may_interpolate(
                        ckpt_path="./init_weight/beit3_base_patch16_224.pth",
                        model=model, model_key='model|module', model_prefix=''
            )

            self.logger.write("finish create model for client%d" % (client_id))

            # Calculate batch size and steps per epoch
            global_batch_size = self.args.batch_size * utils.get_world_size()
            steps_per_epoch = self.sample_num_dict[client_id]["total"] // global_batch_size

            self.logger.write("steps_per_epoch is not more than %d for client%d" % (steps_per_epoch, client_id))

            # Generate train and val datasets
            train_loader_img_text = create_dataset_by_split(self.args,
                                                            split=self.train_split_list_img_text[client_id],
                                                            is_train=True)
            val_loader = create_dataset_by_split(self.args, split=self.val_split, is_train=True)
            loader_for_prototype = create_dataset_by_split(self.args,
                                                            split=self.train_split_list_img_text[client_id],
                                                            is_train=True, is_no_grad=True)  # randaug is used
            if self.args.modal_missing:
                train_loader_img_only = create_dataset_by_split(self.args,
                                                                split=self.train_split_list_img_only[client_id],
                                                                is_train=True)
                train_loader_text_only = create_dataset_by_split(self.args,
                                                                  split=self.train_split_list_text_only[client_id],
                                                                  is_train=True)

            self.logger.write("finish create all dataloaders for client%d" % (client_id))
            torch.distributed.barrier()

            # Create client trainer
            client_trainer = ClientTrainer(self.args, model=model, client_id=client_id,
                                            client_path=self.client_path, server_path=self.server_path,
                                            steps_per_epoch=steps_per_epoch, logger=self.logger,
                                            sample_num_dict=self.sample_num_dict, neighbour_dict=self.neighbour_dict,
                                            barrier=self.barrier, io_lock=self.io_lock)

            self.logger.write("finish creating client%d trainer" % (client_id))
            print("finish creating client%d trainer" % (client_id))

            # Start client trainer algorithm
            self.logger.write("start training client%d" % (client_id))
            t = threading.Thread(target=client_trainer.run, 
                                        args=(train_loader_img_text, train_loader_img_only, train_loader_text_only,
                                        val_loader, train_loader_img_text, self.device, 
                                        n_comm_rounds, global_batch_size,) # loader_for_prototype doesn't work??
                            )
            t.start()
            threads.append(t)

        # Wait for clients to finish
        for t in threads:
            t.join()

        self.logger.write("--------------------- Finish MMFL ----------------------")
