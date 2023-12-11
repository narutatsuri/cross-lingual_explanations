import argparse
import os, json
# os.environ["TRANSFORMERS_CACHE"] = "/local-scratch1/data/wl2787/huggingface_cache/"
os.environ["TRANSFORMERS_CACHE"] = "/local/data/wl2787/huggingface_cache/"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,7"
import pandas as pd
from munch import Munch
import torch
from utils import *
from utils.backbones import BackboneModel
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--save_dir", type=str, default="results/emotion_check")

parser.add_argument("--trained", action="store_true")
cmd_args = vars(parser.parse_args())

device = torch.device("cuda")
if cmd_args["trained"]:
    args = Munch.fromYAML(open(os.path.join(cmd_args["model_dir"], "model_config.yaml"), "r"))
    model = BackboneModel(args, device)
else:
    args = Munch.fromYAML(open(cmd_args["model_dir"], "r"))
    args.model.tokenizer = args.model.checkpoint
    model = BackboneModel(args, device)

data = json.load(open(cmd_args["data_dir"], "r"))
df_scoring = pd.DataFrame()

for index, example in tqdm(enumerate(data)):
    text_input = f"Choose the emotion expressed by the TEXT from CHOICES. [CHOICES]: fear, joy, surprise, anticipation, anger, disgust, trust [TEXT]: {example['text']} [OUTPUT]:"
    # text_input = f"次の[文章]が表している感情を[選択肢]から選択しなさい。 [選択肢]: 喜び, 信頼, 恐れ, 悲しみ, 嫌悪, 怒り, 期待 [文章]: {example['text']} [OUTPUT]:"
    output = model.infer(text_input)

    scores = {}
    scores["input"] = example["text"]
    scores["output"] = output
    scores["gold_output"] = example["choice"]

    df_scoring = pd.concat([df_scoring, pd.DataFrame(scores, index=[index])])
    df_scoring.to_csv(os.path.join(cmd_args["save_dir"], f"{args.model.checkpoint.split('/')[-1]}-lang={args.model.language}.csv"), index=False)