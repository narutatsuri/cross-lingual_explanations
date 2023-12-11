from tqdm import tqdm
import json
from utils import *
import random
import itertools


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

def generate_data_worker_task_func(prompter, emotion, example_by_emotion, count, process_id, summary_dir, lock, num_examples, ic_example_retrieval):
    with lock:
        bar = tqdm(desc=f"Process {process_id+1}", total=count, position=process_id+1, leave=False)
    output_data = []

    for _ in range(count):
        with lock:
            bar.update(1)

        new_example = {}

        if ic_example_retrieval == None:
            all_examples = list(itertools.chain.from_iterable(example_by_emotion.values()))
            emotion_example = random.sample(all_examples, num_examples)
        elif ic_example_retrieval == "diverse":
            ic_example_emotions = random.sample(list(example_by_emotion.keys()), num_examples)
            emotion_example = []
            for ic_example_emotion in ic_example_emotions:
                emotion_example.append(random.sample(example_by_emotion[ic_example_emotion], 1)[0])

        text, explanation, response = prompter.generate_data(emotion, emotion_example)
        new_example["text"] = text
        new_example["choice"] = emotion
        new_example["explanation"] = explanation
        new_example["generated_raw"] = response
        new_example["id"] = "generated"

        output_data.append(new_example)
        json.dump(output_data, 
                  fp=open(summary_dir.replace(".json", f"_process_id={process_id}_emotion={emotion}.json"), "w"), 
                  indent=4, 
                  default=set_default)

    with lock:
        bar.close()
    
    return output_data

def translate_generated_data_worker_task_func(prompter, data, process_id, save_dir, lock):
    with lock:
        bar = tqdm(desc=f"Process {process_id+1}", total=len(data), position=process_id+1, leave=False)
    output_data = []

    for example in data:
        with lock:
            bar.update(1)

        new_example = dict(example)

        text = prompter.translate_generated_data(example["text"])
        explanation = prompter.translate_generated_data(example["explanation"])

        new_example["choice"] = plutchik_en_to_ja[example["choice"]]
        new_example["text"] = text
        new_example["explanation"] = explanation
        new_example["generated_raw"] = text + explanation

        output_data.append(new_example)

        json.dump(output_data, 
                  fp=open(save_dir.replace(".json", f"_process_id={process_id}.json"), "w"), 
                  indent=4, 
                  default=set_default)

    with lock:
        bar.close()
    
    return output_data


def label_emotion_worker_task_func(prompter, data, process_id, save_dir, lock):
    with lock:
        bar = tqdm(desc=f"Process {process_id+1}", total=len(data), position=process_id+1, leave=False)
    output_data = []

    for example in data:
        with lock:
            bar.update(1)

        new_example = dict()

        text = prompter.label_emotion(example["text"])

        new_example["text"] = example["text"]
        new_example["model_label"] = text
        new_example["ground_truth"] = example["choice"]

        output_data.append(new_example)

        json.dump(output_data, 
                  fp=open(save_dir.replace(".json", f"_process_id={process_id}.json"), "w"), 
                  indent=4, 
                  default=set_default)

    with lock:
        bar.close()
    
    return output_data