import xml.sax.handler

import jsonlines
import datasets
import os
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import pipeline
import evaluate
import numpy as np
#import wandb
import argparse
# WANDB_KEY = "00697768fad395aa33d8672046503eaba344e6ea"
# wandb.login(key=WANDB_KEY)
import torch
import json
import tensor_parallel as tp


FLAN_PROMPT =  "You are given a SITUATION and possible answers for the situation in ANSWERS. Your task is to choose the best" \
               "dimensions that will help you find the correct answer on the SITUATION. After you chose the dimensions, please " \
               "elaborate the scene on these dimensions. " \
               "Inputs: OPTIONAL_DIMENSIONS, SITUATION, ANSWERS Outputs: for each dimension, output [CHOSEN_DIMENSION]: [ELABORATION]"


def load_test_data_by_name(name):
    windows_path = f"test_sets\\{name}_-1_samples_test.jsonl"
    unix_path = f"DynamicDream/test_sets/{name}_-1_samples_test.jsonl"
    try:
        dict_dataset = load_test_dataset(windows_path)
    except:
        dict_dataset = load_test_dataset(unix_path)
    return dict_dataset




def load_test_dataset(path):
    all_dicts = []
    with open(path) as f:
        for line in f.readlines():
            all_dicts.append(json.loads(line))
    return all_dicts



def get_all_questions_with_options(dict_dataset, is_flan = False):
    if is_flan:
        return [f"{FLAN_PROMPT}\n Inputs:\n SITUATION: {d['question']}\nA"
                f"NSWERS:{d['mcoptions']}" for d in dict_dataset]

        #return [f"elaborate by dimensions: {d['question']} {d['mcoptions']}" for d in dict_dataset]

    return [f"{d['question']} {d['mcoptions']}" for d in dict_dataset]

def write_elaborations_to_jsonl(dict_dataset, elaborations,save_path):
    with open(save_path, "w") as f:
        for i,dict in enumerate(dict_dataset):
            dict["context"] = elaborations[i]
            json.dump(dict, f)
            if i != len(dict_dataset) - 1:
                f.write("\n")





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

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", dest="raw_data_path", type=str, default="samples\\annotated_data.jsonl")
    parser.add_argument("--dataset_save_dir", dest="dataset_save_dir", type=str, default="datasets\\annotated_dataset.hf")
    parser.add_argument("--model_save_dir", dest="model_save_dir", type=str, default="models\\")
    parser.add_argument("--inference_test_set", dest="inference_test_set", type=str, default="piqa")
    parser.add_argument("--pretrained_model_path", dest="pretrained_model_path", type=str, default="models\\t5_small")
    parser.add_argument("--model_name", dest="model_name", type=str, default="tf-small")
    parser.add_argument("--test_save_dir", dest="test_save_dir", type=str, default="test_sets\\")
    parser.add_argument("--run_train", dest="run_train", action= "store_true")
    parser.add_argument("--run_inference", dest="run_inference", action= "store_true")
    parser.add_argument("--is_flan", dest="is_flan", action= "store_true")
    parser.add_argument("--inference_batch_size", dest="inference_batch_size", type=int, default=32)
    parser.add_argument("--train_epochs", dest="train_epochs", type=int, default=10)
    parser.add_argument("--inference_size", dest="inference_size", type=int, default=-1)
    args = parser.parse_args()

    is_flan = args.is_flan
    run_train = args.run_train
    if run_train:
        print("running training")
        raw_data_path = args.raw_data_path
        dataset_save_dir = args.dataset_save_dir
        #creating or loading dataset
        CLdataset = CreateLoadDataset(raw_data_path, dataset_save_dir)
        dataset = CLdataset.save_or_load_dataset()
        print(dataset)

        #loading the model
        model_name = args.model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def preprocess_function(examples):
            options = [examples["question"], examples["elaboration"], examples["mcoptions"]]
            inputs = []
            targets = []
            for i in range(len(examples["question"])):
                inputs.append(f"{options[0][i]} {options[2][i]}")
                targets.append(f"{options[1][i]}")
            #inputs = [f"{examples['question'][i]} {examples['mcoptions'][i]}" for i in range(len(examples["question"]))]
            model_inputs = tokenizer(inputs, max_length = 1024, truncation = True)

            labels = tokenizer(text_target = targets, max_length=1024, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        def flan_preprocess_with_prompt(examples):
            options = [examples["question"], examples["elaboration"], examples["mcoptions"]]
            inputs = []
            targets = []
            for i in range(len(examples["question"])):
                cur_input = f"{FLAN_PROMPT}\n Inputs:\n SITUATION: {options[0][i]}\nANSWERS:{options[2][i]}"

                inputs.append(cur_input)
                targets.append(f"OUTPUT: {options[1][i]}")
            model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
            # elaborated_inputs = [f"elaboration: {examples['elaboration'][i]}" for i in range(len("elaboration"))]
            labels = tokenizer(text_target=targets, max_length=1024, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs


        def flan_preprocess_function(examples):
            options = [examples["question"], examples["elaboration"], examples["mcoptions"]]
            inputs =[]
            targets = []
            for i in range(len(examples["question"])):
                inputs.append(f"elaborate by dimensions: {options[0][i]} {options[2][i]}")
                targets.append(f"elaboration: {options[1][i]}")

            #inputs = [f"elaborate by dimensions: {examples['question'][i]} {examples['mcoptions'][i]}" for i in range(len(examples["question"]))]
            model_inputs = tokenizer(inputs, max_length = 1024, truncation = True)
            #elaborated_inputs = [f"elaboration: {examples['elaboration'][i]}" for i in range(len("elaboration"))]
            labels = tokenizer(text_target = targets, max_length=1024, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        is_flan = args.is_flan
        if is_flan:
            tokenized_dataset = dataset.map(flan_preprocess_with_prompt, batched=True)
        else:
            tokenized_dataset = dataset.map(preprocess_function, batched=True)
        #setting the data collator
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model = model_name, return_tensors="pt")

        #setting the evaluation metric (rouge, not sure if this means something here)
        #rouge = evaluate.load("rouge")


        def compute_metrics(eval_pred):
            return 0

        # def compute_metrics(eval_pred):
        #     predictions, labels = eval_pred
        #     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        #     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        #
        #     result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        #
        #     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        #     result["gen_len"] = np.mean(prediction_lens)
        #
        #     return {k: round(v, 4) for k, v in result.items()}

        #settings training args
        training_args = Seq2SeqTrainingArguments(
            output_dir=f"{model_name}_training",
            evaluation_strategy="no",
            learning_rate=2e-2,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=args.train_epochs,
            predict_with_generate=True,
            fp16=True,
            push_to_hub=False,
            report_to = "none"
        )
        nir = 1
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        #print(f"the device ids are {device_ids}")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        #model - tp.tensor_parallel(model, ["cuda:0", "cuda:1"])
        # model = torch.nn.DataParallel(model, device_ids=device_ids)
       # model = model.to(device)
        if torch.cuda.is_available():
             model.to(torch.device("cuda"))
        #wandb.init(project="DynamicDream", name=f"elaborating_with_{model_name}")
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        print("finished training")
        #wandb.finish()

        #saving the model:
        model_save_dir = args.model_save_dir
        model_name = model_name.replace("/","_")
        model_save_path = f"{model_save_dir}_{model_name}_{args.train_epochs}"
        print(f"model save path: {model_save_path}")
        trainer.save_model(model_save_path)


    #inference
    run_inference = args.run_inference
    model_path  = args.pretrained_model_path
    ind = 0
    print(model_path)
    if model_path.find("s/") != -1:
        ind = model_path.find("s/") + 2
    else:
        ind = model_path.find("\\") + 2
    pretrained_model_name = model_path[ind:]
    if run_inference:
        test_dataset = load_test_data_by_name(args.inference_test_set)
        if args.inference_size != -1:
            test_dataset = test_dataset[:args.inference_size]
        text_inputs = get_all_questions_with_options(test_dataset, is_flan=is_flan)
        print(f"text_inputs are {text_inputs}")
        loaded_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        if torch.cuda.is_available():
            loaded_model.to(torch.device("cuda"))
        loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)

        #loading the required datasets for inference

        # text = "Riley forced Jan's entrance when the door wouldn't open for her. " \
        #        "What will Riley want to do next? A) check the door for locks (B) need " \
        #        "to apologize to Jan (C) run at the door "
        # text2 = "Riley forced Jan's entrance when the door wouldn't open for her. " \
        #        "What will Riley want to do next? A) check the door for locks (B) need " \
        #        "to apologize to Jan (C) run at the door "
        # texts = [text, text2]

        def generate_and_decode(texts, batch_size):
            #encoding text, and generating
            decoded_texts = []
            size = len(texts)
            #batch_size = 16
            num_iterations = size // batch_size + 1
            for i in range(num_iterations):
                if i == num_iterations - 1:
                    input_text = texts[i*batch_size:]
                else:
                    input_text = texts[i*batch_size:(i+1)*batch_size]
                print(f"the input text: {input_text}")
                inputs = loaded_tokenizer(input_text, return_tensors="pt", padding="longest").input_ids
                print(f"finished tokenizing {i/num_iterations}")
                if torch.cuda.is_available():
                    inputs = inputs.to(torch.device("cuda"))
                outputs = loaded_model.generate(inputs, max_new_tokens=400, do_sample=False)
                current_texts = [loaded_tokenizer.decode(outputs[i], skip_special_tokens=True) for i in range(outputs.shape[0])]
                decoded_texts.extend(current_texts)
                print(f"finished decoding {i / num_iterations}")
                print(f"current_texts are: {current_texts}")
            return decoded_texts


        batch_size = args.inference_batch_size
        elaborations = generate_and_decode(text_inputs, batch_size)
        print(pretrained_model_name)
        save_path = f"{args.test_save_dir}{args.inference_test_set}_{args.inference_size}_{pretrained_model_name}.jsonl"

        write_elaborations_to_jsonl(test_dataset, elaborations, save_path)
        # print()






