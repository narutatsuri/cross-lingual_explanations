from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_checkpoint", type=str, default="bigscience/bloom-560m")
args = vars(parser.parse_args())

tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint, torch_dtype="auto", device_map="auto")

inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt").to("cuda")
outputs = model.generate(inputs)

print(tokenizer.decode(outputs[0]))