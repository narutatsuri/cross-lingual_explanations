from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from datasets import load_dataset


class BackboneModel:
    def __init__(self, model_tokenizer, model_checkpoint, device):
        """
        """
        # Default values for Backbone model
        self.max_input_length = 256
        self.max_output_length = 256
        self.batch_size = 4
        self.weight_decay = 0.1
        self.learning_rate = 4e-5
        self.epochs = 3
        self.num_beams = 8
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_tokenizer)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, device_map = 'auto')

    def tokenizer_fn(self, batch):
        inputs = self.tokenizer(batch["input"], padding="max_length", truncation=True, max_length=self.max_input_length)
        outputs = self.tokenizer(batch["output"], padding="max_length", truncation=True, max_length=self.max_output_length)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["labels"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask

        return batch

    def train(self, train_data, test_data , output_dir):
        dataset = load_dataset("csv", data_files={'train': train_data, 'test': test_data})
        tokenized_datasets = dataset.map(self.tokenizer_fn, batched=True)
        training_args = TrainingArguments(output_dir=output_dir, evaluation_strategy="epoch", 
                                          per_device_train_batch_size=4,
                                          per_device_eval_batch_size=8,
                                          report_to=None,
                                          num_train_epochs=10)

        trainer = Trainer(model=self.model,
                          args=training_args,
                          train_dataset=tokenized_datasets['train'],
                          eval_dataset=tokenized_datasets['test'])

        trainer.train()
        
    def infer(self, sentence):
        """
        """
        input = self.tokenizer.encode(sentence, return_tensors="pt").to(self.device)
        print("# Encoded sentence")
        output = self.model.generate(input, max_new_tokens=128)

        return output
        
        inputs = self.tokenizer([sentence], max_length=self.max_input_length, truncation=True, return_tensors="pt").to(self.device)    
        # output = self.model.generate(**inputs, num_beams=self.num_beams, do_sample=True, max_length=self.max_output_length)
        output = self.model.generate(**inputs)
        print(output)
        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]

        return decoded_output

# from transformers import BloomTokenizerFast, BloomForCausalLM, TrainingArguments, Trainer
# from datasets import load_dataset
# import os


# class BackboneModel:
#     def __init__(self, model_checkpoint, device, max_output_length):
#         """
#         """
#         self.tokenizer = BloomTokenizerFast.from_pretrained(model_checkpoint)
#         self.model = BloomForCausalLM.from_pretrained(model_checkpoint).to(device)
#         self.max_output_length = max_output_length

#     def tokenizer_fn(self, batch):
#         inputs = self.tokenizer(batch["input"], padding='max_length', truncation=True, max_length=self.max_output_length)
#         outputs = self.tokenizer(batch["output"], padding='max_length', truncation=True, max_length=self.max_output_length)

#         return batch

#     def train(self, train_data, test_data, output_dir):
#         dataset = load_dataset("csv", data_files={'train': train_data, 'test': test_data})

#         tokenized_datasets = dataset.map(self.tokenizer_fn, batched=True)

#         training_arguments = TrainingArguments(
#             os.path.join(output_dir,'emotion-explanation-bloom-560m'),
#             learning_rate=2e-5,
#             per_device_train_batch_size=2,
#             num_train_epochs=2,
#             weight_decay=0.01,
#             fp16=True,
#             optim="adafactor",
#             gradient_accumulation_steps=4,
#             gradient_checkpointing=True
#         )

#         trainer = Trainer(
#             model = self.model,
#             args = training_arguments,
#             train_dataset=tokenized_datasets['train'],
#             eval_dataset=tokenized_datasets['test']
#         )

#         trainer.train()
#         trainer.save_model()
#         # training_args = TrainingArguments(output_dir=output_dir, evaluation_strategy="epoch")

#         # trainer = Trainer(model=self.model,
#         #                   args=training_args,
#         #                   train_dataset=tokenized_datasets['train'],
#         #                   eval_dataset=tokenized_datasets['test'])

#         # trainer.train()
        
#     def infer(self, sentence):
#         """
#         """
#         inputs = self.tokenizer([sentence], max_length=self.max_input_length, truncation=True, return_tensors="pt")        
#         output = self.model.generate(**inputs, max_length=self.max_output_length)
#         decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]

#         return decoded_output