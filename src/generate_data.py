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
parser.add_argument("--num_samples_per_emotion", type=int, default=500)

parser.add_argument("--prompt_template", type=str)
parser.add_argument("--example_retrieval", type=str, choices=[None, "diverse"])
parser.add_argument("--num_examples", type=int, default=3)
parser.add_argument("--api_keys_file", type=str, default="prompting/openai_keys.json")
parser.add_argument("--shuffle_keys", action="store_true")
parser.add_argument("--random_seed", type=int, default=42)
cmd_args = vars(parser.parse_args())

random.seed(cmd_args["random_seed"])

# Load OpenAI API Keys
keys = json.load(open(cmd_args["api_keys_file"], "r"))
assert all([key.startswith("sk-") for key in keys]), "[ERROR]: Set up keys in `openai_keys.json`."

prompt_template = json.load(open(cmd_args["prompt_template"], "r"))
prompter = PromptHandler(api_key=keys, prompt_template=prompt_template)
formatted_data = json.load( open(cmd_args["data_dir"], "r") )

# Get counts for number of instances that need generating for each emotion
emotion_counts = {}
example_by_emotion = {}
for emotion in plutchik:
    emotion_counts[emotion] = 0
    example_by_emotion[emotion] = []
for example in formatted_data:
    emotion_counts[example["choice"]] += 1
    example_by_emotion[example["choice"]].append(example)
for emotion in plutchik:
    emotion_counts[emotion] = max(cmd_args["num_samples_per_emotion"] - emotion_counts[emotion], 0)

for emotion in emotion_counts:
    # If quota for emotion is reached, skip
    if emotion_counts[emotion] == 0:
        continue

    print(f"Generating data for {emotion} ({emotion_counts[emotion]} examples)...")

    worker_results = []
    lock = multiprocessing.Manager().Lock()
    pool = multiprocessing.Pool(processes=cmd_args["num_processes"])

    count = emotion_counts[emotion]//cmd_args["num_processes"] + 1
    for process_id in range(cmd_args["num_processes"]):
        worker_results.append(pool.apply_async(generate_data_worker_task_func, args = (prompter, 
                                                                                       emotion,
                                                                                       example_by_emotion,
                                                                                       count,
                                                                                       process_id, 
                                                                                       cmd_args["data_dir"], 
                                                                                       lock, 
                                                                                       cmd_args["num_examples"],
                                                                                       cmd_args["example_retrieval"])))

    pool.close()
    pool.join()

    for i in worker_results:
        formatted_data += i.get()

    json.dump(formatted_data, fp=open(cmd_args["data_dir"], "w"), indent=4, default=set_default)