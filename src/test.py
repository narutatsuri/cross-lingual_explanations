import argparse
import os
# os.environ["TRANSFORMERS_CACHE"] = "/local-scratch1/data/wl2787/huggingface_cache/"
os.environ["TRANSFORMERS_CACHE"] = "/local/data/wl2787/huggingface_cache/"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,4,5"
import pandas as pd
from munch import Munch
import torch
from utils import *
from utils.backbones import BackboneModel
from utils.evaluation import get_metrics
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--save_dir", type=str, default="results/")
cmd_args = vars(parser.parse_args())

device = torch.device("cuda")

args = Munch.fromYAML(open(os.path.join(cmd_args["model_dir"], "model_config.yaml"), "r"))

model = BackboneModel(args, device)
data = pd.read_csv(cmd_args["data_dir"])

df_scoring = pd.DataFrame()

save_dir = os.path.join(cmd_args["save_dir"], args.model_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for index, example in tqdm(data.iterrows(), total=len(data)):
    output = model.infer(example["input"])

    scores = {}
    scores["input"] = example["input"]
    scores["output"] = output
    scores["gold_output"] = example["output"]
    scores.update(get_metrics(output, example["output"]))

    df_scoring = pd.concat([df_scoring, pd.DataFrame(scores, index=[index])])
    df_scoring.to_csv(os.path.join(save_dir, "results_all.csv"), index=False)