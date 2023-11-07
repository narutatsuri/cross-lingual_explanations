# Define prompt handler
import openai
import random


class PromptHandler:
    def __init__(self, api_key, prompt_template):
        """ """
        openai.api_key = random.choice(api_key)
        self.prompt_template = prompt_template
        self.model = prompt_template["model"]

    def generate_summary(self, text):
        """
        """
        prompt = ""
        prompt += self.prompt_template["instruction"] + "\n"
        prompt += text

        summary = openai.ChatCompletion.create(model=self.model,messages=[{"role": "system", "content": ""},
                                                                            {"role": "user", "content": prompt},],
                                                                            temperature=0,
                                                                            top_p=1,
                                                                            frequency_penalty=0,
                                                                            presence_penalty=0,).choices[0].message.content

        return summary
    
    def generate_explanation(self, example):
        """
        """
        prompt = ""
        prompt += self.prompt_template["instruction"] + "\n"
        prompt += "[TEXT]: " + example["summary"] + "\n"
        prompt += "[CHOICES]: " + ", ".join(list(set(example["choices"]))) + "\n"
        prompt += "[CHOICE]: \n[EXPLANATION]:\n"

        response = openai.ChatCompletion.create(model=self.model,messages=[{"role": "system", "content": ""},
                                                                            {"role": "user", "content": prompt},],
                                                                            temperature=0,
                                                                            top_p=1,
                                                                            frequency_penalty=0,
                                                                            presence_penalty=0,).choices[0].message.content
        
        choice, explanation = response.split("[EXPLANATION]:")

        return choice, explanation, response
    
    def generate_data(self, emotion, emotion_example):
        topic = random.sample(self.prompt_template["topics"], 1)[0]

        prompt = ""
        prompt += self.prompt_template["instruction"].replace("[FILL]", emotion).replace("[TOPIC]", topic) + "\n"
        prompt += "[EXAMPLE]:\n"
        prompt += "[TEXT]: " + emotion_example["text"] + "\n"
        prompt += "[EXPLANATION]: " + emotion_example["explanation"] + "\n"

        response = openai.ChatCompletion.create(model=self.model,messages=[{"role": "system", "content": ""},
                                                                            {"role": "user", "content": prompt},],
                                                                            temperature=1).choices[0].message.content
        
        text, explanation = response.split("[EXPLANATION]:")

        return text, explanation, response