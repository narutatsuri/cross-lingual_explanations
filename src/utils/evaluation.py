import os
# os.environ["TRANSFORMERS_CACHE"] = "/local-scratch1/data/wl2787/huggingface_cache/" # communication
# os.environ["TRANSFORMERS_CACHE"] = "/mnt/swordfish-datastore/wl2787/huggingface_cache/" # branzino
os.environ["TRANSFORMERS_CACHE"] = "/local/data/wl2787/huggingface_cache/" # coffee
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from bleurt import score
import openai
import random


rouge = Rouge()
scorer = score.BleurtScorer()

def get_metrics(reference, prediction):
    return {
        "meteor": meteor_score([reference], prediction),
        "rouge": rouge.get_scores(prediction, reference)[0]["rouge-l"]["f"],
        "bleurt": scorer.score(references=[reference], candidates=[prediction])[0]
    }

class PromptEvaluator:
    def __init__(self, api_key, prompt_template):
        """ """
        openai.api_key = random.choice(api_key)
        self.prompt_template = prompt_template
        self.model = prompt_template["model"]

    def get_prompt_comparison(self, reference, prediction):
        """
        """
        prompt = ""
        prompt += self.prompt_template["instruction"] + "\n"
        try:
            prompt += f"[RATIONALE 1]: {prediction[prediction.index('[RATIONALE]: ') + len('[RATIONALE]: '):]}\n[RATIONALE 2]: {reference[reference.index('[RATIONALE]: ') + len('[RATIONALE]: '):]}"
        except ValueError:
            prompt += f"[RATIONALE 1]: {prediction}\n[RATIONALE 2]: {reference[reference.index('[RATIONALE]: ') + len('[RATIONALE]: '):]}"

        output = openai.ChatCompletion.create(model=self.model,messages=[{"role": "system", "content": ""},
                                                                            {"role": "user", "content": prompt},],
                                                                            temperature=0,
                                                                            top_p=1,
                                                                            frequency_penalty=0,
                                                                            presence_penalty=0,).choices[0].message.content

        return output
    

    def get_prompt_fluency(self, reference, prediction):
        """
        """
        prompt = ""
        prompt += self.prompt_template["instruction"] + "\n"
        try:
            prompt += f"[RATIONALE 1]: {prediction[prediction.index('[RATIONALE]: ') + len('[RATIONALE]: '):]}\n[RATIONALE 2]: {reference[reference.index('[RATIONALE]: ') + len('[RATIONALE]: '):]}"
        except ValueError:
            prompt += f"[RATIONALE 1]: {prediction}\n[RATIONALE 2]: {reference[reference.index('[RATIONALE]: ') + len('[RATIONALE]: '):]}"

        output = openai.ChatCompletion.create(model=self.model,messages=[{"role": "system", "content": ""},
                                                                            {"role": "user", "content": prompt},],
                                                                            temperature=0,
                                                                            top_p=1,
                                                                            frequency_penalty=0,
                                                                            presence_penalty=0,).choices[0].message.content

        return output    
    
    def get_prompt_accuracy(self, reference, prediction, text):
        """
        """
        prompt = ""
        prompt += self.prompt_template["instruction"] + "\n"
        prompt += text + "\n"
        try:
            prompt += f"[RATIONALE 1]: {prediction[prediction.index('[RATIONALE]: ') + len('[RATIONALE]: '):]}\n[RATIONALE 2]: {reference[reference.index('[RATIONALE]: ') + len('[RATIONALE]: '):]}"
        except ValueError:
            prompt += f"[RATIONALE 1]: {prediction}\n[RATIONALE 2]: {reference[reference.index('[RATIONALE]: ') + len('[RATIONALE]: '):]}"
        output = openai.ChatCompletion.create(model=self.model,messages=[{"role": "system", "content": ""},
                                                                            {"role": "user", "content": prompt},],
                                                                            temperature=0,
                                                                            top_p=1,
                                                                            frequency_penalty=0,
                                                                            presence_penalty=0,).choices[0].message.content

        return output 