import json
import os
import sys
import argparse
import random

sys.path.append("src/")
from utils import *
from utils.models import PromptHandler
from utils.multiprocessing import *
import multiprocessing

# Command line parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/")
parser.add_argument("--num_processes", type=int, default=20)

parser.add_argument("--prompt_template", type=str)
parser.add_argument("--api_keys_file", type=str, default="prompting/openai_keys.json")
parser.add_argument("--shuffle_keys", action="store_true")
parser.add_argument("--random_seed", type=int, default=42)
cmd_args = vars(parser.parse_args())

random.seed(cmd_args["random_seed"])

# Load OpenAI API Keys
keys = json.load(open(cmd_args["api_keys_file"], "r"))
assert all(
    [key.startswith("sk-") for key in keys]
), "[ERROR]: Set up keys in `openai_keys.json`."

# Create prompter
prompt_template = json.load(open(cmd_args["prompt_template"], "r"))
prompter = PromptHandler(api_key=keys, prompt_template=prompt_template)

# Load data
full_data = json.load(open(cmd_args["data_dir"], "r"))

# Get translated data and remove them from process list
save_dir = cmd_args["data_dir"].replace("lang=en", "lang=ja")

if os.path.exists(save_dir):
    output_data = json.load(open(save_dir, "r"))
    completed_ids = [item["translation_id"] for item in output_data]
else:
    output_data = []
    completed_ids = []

# Fetch remaining examples
remaining_japanese_split = [
    item for item in full_data if item["translation_id"] not in completed_ids
]
remaining_japanese_split_partition = partition(
    remaining_japanese_split, cmd_args["num_processes"]
)

print(f"{len(completed_ids)} completed, {len(remaining_japanese_split)} to translate.")

worker_results = []
lock = multiprocessing.Manager().Lock()
pool = multiprocessing.Pool(processes=cmd_args["num_processes"])

for process_id in range(cmd_args["num_processes"]):
    worker_results.append(
        pool.apply_async(
            translate_generated_data_worker_task_func,
            args=(
                prompter,
                remaining_japanese_split_partition[process_id],
                process_id,
                save_dir,
                lock,
            ),
        )
    )

pool.close()
pool.join()

for i in worker_results:
    output_data += i.get()

json.dump(output_data, fp=open(save_dir, "w"), indent=4, default=set_default)
