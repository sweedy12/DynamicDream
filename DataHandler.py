from datasets import load_dataset
import argparse
from abc import ABC, abstractmethod
import json

class DatasetGetter:
    @staticmethod
    def get_dataset(dataset, num_samples):
        if dataset == "piqa":
            loaded_set =  load_dataset("piqa")["train"].select(range(0,num_samples))
            return PIQAHandler(loaded_set, "piqa")
        if dataset == "physical_qa":
            dataset = []
            with open("datasets\\train_rand_split.jsonl") as f:
                for i,line in enumerate(f.readlines()):
                    if i == num_samples:
                        break
                    else:
                        dataset.append(json.loads(line))
                return PhysicalCSHandler(dataset, "Physical QA")

class DatasetHandler (ABC):
    def __init__(self, dataset, name):
        self.dataset = dataset
        self.name  = name

    def get_all_dicts(self):
        all_dicts = []
        for i, sample in enumerate(self.dataset):
            all_dicts.append(self.get_dict(sample,i))
        return all_dicts


    def get_dict(self, sample, i):
        jdict  = {}
        jdict["question"] = self.get_question(sample)
        jdict["id"] = f"{self.name}_train{i+1}"
        jdict["meta"] = self.get_meta(sample)
        jdict["mcoptions"] = self.get_mcoptions(sample)
        jdict["answer"] = self.get_answer(sample)
        return jdict



    @abstractmethod
    def get_question(self, sample):
        pass

    @abstractmethod
    def get_mcoptions(self, sample):
        pass

    @abstractmethod
    def get_meta(self, sample):
        pass

    @abstractmethod
    def get_answer(self, sample):
        pass


class PIQAHandler(DatasetHandler):

    def get_question(self, sample):
        return sample["goal"]

    def get_meta(self, sample):
        return {"answerKey":"sol1"} if sample["label"] == 0 else {"answerKey":"sol2"}

    def get_mcoptions(self, sample):
        return f"sol1: {sample['sol1']} sol2: {sample['sol2']}"

    def get_answer(self, sample):
        return sample["sol1"] if sample["label"] ==0 else sample["sol2"]

class PhysicalCSHandler(DatasetHandler):

    def get_question(self, sample):
        return sample["question"]["stem"]

    def get_meta(self, sample):
        #getting the answerKey
        return {"answerKey": sample["answerKey"]}

    def get_mcoptions(self, sample):
        choices = sample["question"]["choices"]
        mcoptions = ""
        for choice in choices:
            mcoptions += f"{choice['label']} {choice['text']} "
        return mcoptions

    def get_answer(self, sample):
        choices = sample["question"]["choices"]
        key = sample["answerKey"]
        for choice in choices:
            if choice['label'] == key:
                return choice["text"]
        return "none"





class WriteJson:

    def __init__(self, save_dir, num_samples, dataset_name):
        self.save_dir = save_dir
        self.dataset_name = dataset_name
        self.dataset_handler = DatasetGetter.get_dataset(dataset_name, num_samples)
        self.num_samples = num_samples
        self.save_name = self.get_save_name()

    def get_save_name(self):
        return f"{self.save_dir}\\{self.dataset_name}_{self.num_samples}_samples.jsonl"

    def write_json(self):
        #creating a dict to hold all the json information:
        all_dicts = self.dataset_handler.get_all_dicts()
        with open(self.save_name, "w") as f:
            for dict in all_dicts:
                json.dump(dict,f)
                f.write("\n")












if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", dest="num_samples", type=int, default=100)
    parser.add_argument("--save_dir", dest="save_dir", type=str, default="samples")
    parser.add_argument("--dataset_name", dest="dataset_name", type=str, default="piqa")
    args = parser.parse_args()
    dataset_name = args.dataset_name
    save_dir = args.save_dir
    try:
        os.mkdir(save_dir)
    except:
        print("save directory already exists")
    num_samples = args.num_samples
    jsonwriter = WriteJson(save_dir, num_samples, dataset_name=dataset_name)
    jsonwriter.write_json()






