import os
import yaml
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from datasets import load_dataset


class BackboneModel:
    def __init__(self, args, device=None):
        """
        """
        self.args = args
        if device != None:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(args.model.tokenizer)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(args.model.checkpoint, device_map = "auto")

    def train(self, train_data, test_data):
        """
        Train model.
        """
        # Create config file and folder
        if not os.path.exists(self.args.out_dir):
            os.makedirs(self.args.out_dir)
        # Create checkpoints folder
        if not os.path.exists(os.path.join(self.args.out_dir, "checkpoints")):
            os.makedirs(os.path.join(self.args.out_dir, "checkpoints"))
        # Dump info
        with open(os.path.join(self.args.out_dir, "model_config.yaml"), "w") as yaml_file:
            self.args.model.checkpoint = self.args.out_dir
            yaml.dump(self.args.__dict__, yaml_file, default_flow_style=False)

        def tokenizer_fn(batch):
            inputs = self.tokenizer(batch["input"], padding="max_length", truncation=True, max_length=self.args.model.max_input_length)
            outputs = self.tokenizer(batch["output"], padding="max_length", truncation=True, max_length=self.args.model.max_output_length)

            batch["input_ids"] = inputs.input_ids
            batch["attention_mask"] = inputs.attention_mask
            batch["labels"] = outputs.input_ids
            batch["decoder_attention_mask"] = outputs.attention_mask

            return batch
        
        dataset = load_dataset("csv", 
                               data_files={"train": train_data, 
                                           "test": test_data})
        tokenized_datasets = dataset.map(tokenizer_fn, batched=True)

        trainer = Trainer(model=self.model,
                          args=TrainingArguments(output_dir=os.path.join(self.args.out_dir, "checkpoints"), 
                                                 per_device_train_batch_size=self.args.training.train_batch_size,
                                                 per_device_eval_batch_size=self.args.training.val_batch_size,
                                                 save_strategy=self.args.training.save_strategy,
                                                 save_steps=self.args.training.save_steps,
                                                 num_train_epochs=self.args.training.epochs,
                                                 evaluation_strategy="epoch", 
                                                 report_to="none",
                                                 ),
                          train_dataset=tokenized_datasets["train"],
                          eval_dataset=tokenized_datasets["test"])

        trainer.train()
        self.model.save_pretrained(self.args.model.checkpoint)
        
    def infer(self, sentence):
        """
        """
        inputs = self.tokenizer([sentence], max_length=self.args.model.max_input_length, truncation=True, return_tensors="pt").to(self.model.device)    
        output = self.model.generate(**inputs, num_beams=self.args.testing.num_beams, do_sample=True, max_length=self.args.model.max_output_length)
        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]

        return decoded_output
    
    def update_parameters(self, args):
        self.args = args