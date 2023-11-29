from tqdm import tqdm
import json
from utils import *
import random


def summarize_worker_task_func(prompter, extracted_files, process_id, summary_dir, lock):
    with lock:
        bar = tqdm(desc=f"Process {process_id+1}", total=len(extracted_files), position=process_id+1, leave=False)
    output_data = []

    for example in extracted_files:
        with lock:
            bar.update(1)

        summary = prompter.generate_summary(example["text"])
        example["summary"] = summary

        output_data.append(example)
        json.dump(output_data, fp=open(summary_dir.replace(".json", f"_process_id={process_id}.json"), "w"), indent=4, default=set_default)

    with lock:
        bar.close()
    
    return output_data

def explain_worker_task_func(prompter, extracted_files, process_id, explanation_dir, lock):
    with lock:
        bar = tqdm(desc=f"Process {process_id+1}", total=len(extracted_files), position=process_id+1, leave=False)
    output_data = []

    for example in extracted_files:

        choice, explanation, response = prompter.generate_explanation(example)
        example["choice"] = choice
        example["explanation"] = explanation
        example["raw_response"] = response

        output_data.append(example)

        json.dump(output_data, fp=open(explanation_dir.replace(".json", f"_process_id={process_id}.json"), "w"), indent=4, default=set_default)
        with lock:
            bar.update(1)

    with lock:
        bar.close()
    
    return output_data

def generate_data_worker_task_func(prompter, emotion, example_by_emotion, count, process_id, summary_dir, lock, num_examples=3):
    with lock:
        bar = tqdm(desc=f"Process {process_id+1}", total=count, position=process_id+1, leave=False)
    output_data = []

    for _ in range(count):
        with lock:
            bar.update(1)

        new_example = {}
        emotion_example = random.sample(example_by_emotion, num_examples)

        text, explanation, response = prompter.generate_data(emotion, emotion_example)
        new_example["text"] = text
        new_example["choice"] = emotion
        new_example["explanation"] = explanation
        new_example["generated_raw"] = response
        new_example["id"] = "generated"

        output_data.append(new_example)
        json.dump(output_data, fp=open(summary_dir.replace(".json", f"_process_id={process_id}_emotion={emotion}.json"), "w"), indent=4, default=set_default)

    with lock:
        bar.close()
    
    return output_data