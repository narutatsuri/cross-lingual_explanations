import json
import os
import argparse
import sys
sys.path.append('../')
from utils import *


# Command line parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/english/full_data.json")
parser.add_argument("--generated_data_dir", type=str, default="data/english/")
cmd_args = vars(parser.parse_args())

full_data = json.load(open(cmd_args["data_dir"], "r"))

new_examples = []
for folder in os.listdir(cmd_args["generated_data_dir"]):
    if "process_id" in folder:
        new_examples += json.load(open(os.path.join(cmd_args["generated_data_dir"], folder), "r"))

cleaned_new_examples = clean_generated_data(new_examples)
full_data += cleaned_new_examples
full_data = clean_generated_data(full_data)
full_data = remove_duplicates(full_data)

print(get_emotion_counts(full_data))

json.dump(full_data, open(cmd_args["data_dir"], "w"), indent=4)