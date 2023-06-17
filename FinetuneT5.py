import jsonlines
import datasets
import os
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import numpy as np
import wandb
WANDB_KEY = "00697768fad395aa33d8672046503eaba344e6ea"
wandb.login( key=WANDB_KEY)

class CreateLoadDataset:

    def __init__(self, file_path, save_path):
        self.file_path = file_path
        self.save_path = save_path
        self.objects = self.read_objects_from_jsonlines()

    def save_or_load_dataset(self):
        if os.path.exists(self.save_path):
            return datasets.load_from_disk(self.save_path)
        else:
            dataset = self.turn_to_dataset()
            self.save_dataset(dataset)
            return dataset


    def save_dataset(self, dataset):

        dataset.save_to_disk(self.save_path)

    def read_objects_from_jsonlines(self):
        #objects = []
        with jsonlines.open(self.file_path) as reader:
            return [o for o in reader]

    def turn_to_dataset(self):
        return datasets.Dataset.from_dict(self.turn_to_single_dict())

    def turn_to_single_dict(self):
        new_dict = {key:[] for key in self.objects[0].keys()}
        for obj in self.objects:
            for key in new_dict:
                try:
                    new_dict[key].append(obj[key])
                except:
                    print(key)
                    print(obj)
        return new_dict

    def gen(self):
        for object in self.objects:
            yield object




if __name__ == "__main__":

    annotated_path = "samples\\annotated_data.jsonl"
    save_dir = "datasets\\annotated_dataset.hf"

    #creating or loading datasset
    CLdataset = CreateLoadDataset(annotated_path, save_dir)
    dataset = CLdataset.save_or_load_dataset()

    #loading the model
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        inputs = [f"{examples['question'][i]} {examples['mcoptions'][i]}" for i in range(len(examples["question"]))]
        model_inputs = tokenizer(inputs, max_length = 1024, truncation = True)
        labels = tokenizer(text_target = examples["elaboration"], max_length=512, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    #setting the data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model = model_name, return_tensors="pt")

    #setting the evaluation metric (rouge, not sure if this means something here)
    rouge = evaluate.load("rouge")


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    #settings training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{model_name}_training",
        evaluation_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=4,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False
    )
    nir = 1
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    wandb.init(project="DynamicDream", name=f"elaborating_with_{model_name}")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    wandb.finish()






