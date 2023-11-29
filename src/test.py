import argparse
import json
import os
os.environ["TRANSFORMERS_CACHE"] = "/local-scratch1/data/wl2787/huggingface_cache/"
# os.environ["TRANSFORMERS_CACHE"] = "/mnt/swordfish-datastore/wl2787/huggingface_cache/"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from utils import *
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--model_tokenizer", type=str, default="google/mt5-large")
parser.add_argument("--model_checkpoint", type=str)
parser.add_argument("--data_dir", type=str)
args = vars(parser.parse_args())

data = json.load(open(args["data_dir"], "r"))
print("# LOADED DATA")

tokenizer = AutoTokenizer.from_pretrained(args["model_tokenizer"])
model = AutoModelForSeq2SeqLM.from_pretrained(args["model_checkpoint"], torch_dtype="auto", device_map="auto")

print("# LOADED MODEL")

for example in data:
    text = example["text"]
    emotion = example["choice"]
    input_format = "[TEXT]: {} [SENTIMENT]: {}".format(text, emotion)
    print("INPUT: ", input_format)

    output = model.generate(tokenizer.encode(input_format, return_tensors="pt").to(model.device), max_new_tokens=512)
    output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    print("OUTPUT: ", output)
    input()