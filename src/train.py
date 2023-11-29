import os
os.environ["TRANSFORMERS_CACHE"] = "/local-scratch1/data/wl2787/huggingface_cache/"
# os.environ["TRANSFORMERS_CACHE"] = "/mnt/swordfish-datastore/wl2787/huggingface_cache/"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd


class Seq2SeqDataset(Dataset):
    def __init__(self, input_texts, target_texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.max_length = max_length

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]

        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

checkpoint = "google/mt5-large"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

train_data_dir = "data/sentiment_explanation/english/train_cleaned.csv"
test_data_dir = "data/sentiment_explanation/english/test_cleaned.csv"

train = pd.read_csv(train_data_dir)
valid = pd.read_csv(test_data_dir)

train_input_texts = list(train["input"])
train_target_texts = list(train["output"])
valid_input_texts = list(valid["input"])
valid_target_texts = list(valid["output"])

# Initialize datasets
train_dataset = Seq2SeqDataset(train_input_texts, train_target_texts, tokenizer, max_length=512)
valid_dataset = Seq2SeqDataset(valid_input_texts, valid_target_texts, tokenizer, max_length=512)

# Define training arguments
training_args = TrainingArguments(
    report_to="none",
    output_dir="results",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    weight_decay=0.001,
    logging_dir='results/logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset)

trainer.train()
model.save_pretrained(f"models/{checkpoint.split('/')[-1]}-train_epochs=5")