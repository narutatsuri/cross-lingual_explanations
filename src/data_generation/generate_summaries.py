import argparse, random, json, os
import sys

sys.path.append("../")
from utils import *
from utils.models import PromptHandler
from utils.multiprocessing import *
import itertools
import multiprocessing


# Command line parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--num_processes", type=int, default=10)

parser.add_argument("--prompt_template", type=str)
parser.add_argument("--api_keys_file", type=str, default="prompting/openai_keys.json")
parser.add_argument("--shuffle_keys", action="store_true")
parser.add_argument("--random_seed", type=int, default=42)
args = vars(parser.parse_args())

random.seed(args["random_seed"])

# Load OpenAI API Keys
keys = json.load(open(args["api_keys_file"], "r"))
assert all(
    [key.startswith("sk-") for key in keys]
), "[ERROR]: Set up keys in `openai_keys.json`."

prompt_template = json.load(open(args["prompt_template"], "r"))
prompter = PromptHandler(api_key=keys, prompt_template=prompt_template)

sentiment_data = json.load(open(args["data_dir"], "r"))

summary_dir = args["data_dir"].replace(".json", "_with_summaries.json")

if os.path.exists(summary_dir):
    summary_data = json.load(open(summary_dir, "r"))
    existing_keys = [i["id"] for i in summary_data]
else:
    summary_data = []
    existing_keys = []

extracted_files = []
for id in sentiment_data:
    if id not in existing_keys:
        example = dict()
        text = sentiment_data[id]["Reddit Post"]
        choices = [
            i["Emotion"]
            for i in list(
                itertools.chain.from_iterable(
                    list((sentiment_data[id]["Annotations"]).values())
                )
            )
        ]
        example["text"] = text
        example["choices"] = choices
        example["id"] = id

        extracted_files.append(example)


extracted_files = partition(extracted_files, args["num_processes"])
worker_results = []

lock = multiprocessing.Manager().Lock()
pool = multiprocessing.Pool(processes=args["num_processes"])

for process_id in range(len(extracted_files)):
    worker_results.append(
        pool.apply_async(
            summarize_worker_task_func,
            args=(prompter, extracted_files[process_id], process_id, summary_dir, lock),
        )
    )

pool.close()
pool.join()

summarized_examples = []
for i in worker_results:
    summarized_examples += i.get()

summary_data += summarized_examples

json.dump(summary_data, fp=open(summary_dir, "w"), indent=4, default=set_default)
