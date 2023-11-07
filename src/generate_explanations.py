import argparse, random, json, os
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
assert all([key.startswith("sk-") for key in keys]), "[ERROR]: Set up keys in `openai_keys.json`."

prompt_template = json.load(open(args["prompt_template"], "r"))
prompter = PromptHandler(api_key=keys, prompt_template=prompt_template)

summary_data = json.load( open(args["data_dir"], "r") )

assert "_with_summaries" in args["data_dir"]
explanation_dir = args["data_dir"].replace(".json", "_with_explanations.json")

if os.path.exists(explanation_dir):
    explanation_data = json.load( open(explanation_dir, "r") )
    existing_keys = [i["id"] for i in explanation_data]
else:
    explanation_data = []
    existing_keys = []

print(f"# Found {len(existing_keys)} existing explanations")

extracted_files = []
for example in summary_data:
    if example["id"] not in existing_keys:
        extracted_files.append(example)

extracted_files = partition(extracted_files, args["num_processes"])
worker_results = []

lock = multiprocessing.Manager().Lock()
pool = multiprocessing.Pool(processes=args["num_processes"])

for process_id in range(len(extracted_files)):
    worker_results.append(pool.apply_async(explain_worker_task_func, args = (prompter, 
                                                                                extracted_files[process_id], 
                                                                                process_id, 
                                                                                explanation_dir, 
                                                                                lock)))

pool.close()
pool.join()

explained_examples = []
for i in worker_results:
    explained_examples += i.get()

explanation_data += explained_examples

json.dump(explanation_data, fp=open(explanation_dir, "w"), indent=4, default=set_default)