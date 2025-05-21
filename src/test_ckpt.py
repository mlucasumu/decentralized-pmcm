import torch
from main import get_args
from utils import utils
import numpy as np
import random
from datasets.dataset import create_dataset_by_split
from timm.models import create_model
from algorithm.BaseTrainer import evaluate, VQAHandler


if __name__ == '__main__':

    args = get_args()
    device = torch.device(args.device)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

    data_loader_val = create_dataset_by_split(args, split="val", is_train=False)

    task_handler = VQAHandler()

    model = create_model(
        "beit3_base_patch16_480_vqav2",
        pretrained=False,
        drop_path_rate=args.drop_path,
        vocab_size=args.vocab_size,
        checkpoint_activations=args.checkpoint_activations
    )

    model.to(device)

    ckpt_path = f"./save_weight/server/client{args.test_model_num}_final_model.pth"
    proto_path = f"./save_weight/server/client{args.test_model_num}_final_prototypes.pth"
    
    utils.load_model_and_may_interpolate(
        ckpt_path=ckpt_path, model=model, model_key='model|module', model_prefix=''
    )

    if args.use_test_prototypes:
        prototypes = torch.load(proto_path)
    elif args.use_random_prototypes:
        print("using random mask")
        prototypes = {"img": torch.randn([args.num_classes, args.embed_dim], dtype=torch.float32),
                        "text": torch.randn([args.num_classes, args.embed_dim], dtype=torch.float32),
                        "fusion": torch.randn([args.num_classes, args.embed_dim], dtype=torch.float32),
                        "target": torch.randn([args.num_classes, args.num_classes], dtype=torch.float32)}
    else:
        print("using zero mask")
        prototypes = {"img": torch.zeros([args.num_classes, args.embed_dim], dtype=torch.float32),
                        "text": torch.zeros([args.num_classes, args.embed_dim], dtype=torch.float32),
                        "fusion": torch.zeros([args.num_classes, args.embed_dim], dtype=torch.float32),
                        "target": torch.zeros([args.num_classes, args.num_classes], dtype=torch.float32)}
    
    for key in prototypes.keys():
        prototypes[key] = prototypes[key].to(device)

    evaluate(args, data_loader_val, model, prototypes, device, task_handler)



