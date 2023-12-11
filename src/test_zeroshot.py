import argparse
import os
# os.environ["TRANSFORMERS_CACHE"] = "/local-scratch1/data/wl2787/huggingface_cache/"
os.environ["TRANSFORMERS_CACHE"] = "/local/data/wl2787/huggingface_cache/"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from munch import Munch
import torch
from utils import *
from utils.backbones import BackboneModel


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str)
cmd_args = vars(parser.parse_args())

device = torch.device("cuda")
args = Munch.fromYAML(open(cmd_args["model_dir"], "r"))
args.model.tokenizer = args.model.checkpoint

model = BackboneModel(args, device)

while True:
    output = model.infer(input("# Enter Input: "))
    print("# Output", output)