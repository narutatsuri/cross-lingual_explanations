import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--save_dir", type=str, default="results/")

parser.add_argument("--prompt_template", type=str)
parser.add_argument("--score_metric", type=str)

parser.add_argument("--api_keys_file", type=str, default="prompting/openai_keys.json")
parser.add_argument("--shuffle_keys", action="store_true")
parser.add_argument("--random_seed", type=int, default=42)

parser.add_argument("--device", type=str)
parser.add_argument("--huggingface_cache", type=str)

cmd_args = vars(parser.parse_args())

import json
import random
os.environ["HF_CACHE"] = cmd_args["huggingface_cache"]
os.environ["CUDA_VISIBLE_DEVICES"] = cmd_args["device"]
import pandas as pd
from munch import Munch
import torch
from utils import *
from utils.backbones import BackboneModel
from utils.evaluation import PromptEvaluator
from tqdm import tqdm


random.seed(cmd_args["random_seed"])

device = torch.device("cuda")

# Load OpenAI API Keys
keys = json.load(open(cmd_args["api_keys_file"], "r"))
assert all([key.startswith("sk-") for key in keys]), "[ERROR]: Set up keys in `openai_keys.json`."

prompt_template = json.load(open(cmd_args["prompt_template"], "r"))
prompt_evaluator = PromptEvaluator(api_key=keys, prompt_template=prompt_template)

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

    if cmd_args["score_metric"] == "comparison":
        scores["comparison"] = prompt_evaluator.get_prompt_comparison(example["output"], output)
    elif cmd_args["score_metric"] == "fluency":
        scores["fluency"] = prompt_evaluator.get_prompt_fluency(example["output"], output)
    elif cmd_args["score_metric"] == "accuracy":
        scores["accuracy"] = prompt_evaluator.get_prompt_accuracy(example["output"], output, example["input"])

    df_scoring = pd.concat([df_scoring, pd.DataFrame(scores, index=[index])])
    df_scoring.to_csv(os.path.join(save_dir, f"{cmd_args['score_metric']}_results.csv"), index=False)