import argparse
from datasets.dataset import VQAv2Dataset
from transformers import XLMRobertaTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--num_clients', type=int, default=5)
parser.add_argument('--modal_missing_rate', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--num_classes', type=int, default=310)
parser.add_argument('--scale', type=bool, default=False)
parser.add_argument('--data_path', default='./data', type=str)
args = parser.parse_args()

tokenizer = XLMRobertaTokenizer("init_weight/beit3.spm")
VQAv2Dataset.make_dataset_index(data_path=args.data_path, tokenizer=tokenizer, annotation_data_path=f"{args.data_path}/vqa", scale=args.scale)
VQAv2Dataset.make_modal_missing_index(args.data_path, modal_missing_rate=args.modal_missing_rate, alpha=args.alpha, n_clients=args.num_clients, n_classes=args.num_classes)