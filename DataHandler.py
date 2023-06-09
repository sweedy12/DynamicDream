from datasets import load_dataset
import argparse
from abc import ABC, abstractmethod
import json

class DatasetGetter:
    @staticmethod
    def get_dataset(dataset_name, num_samples):
        if dataset_name == "piqa":
            loaded_set =  load_dataset("piqa")["train"].select(range(0,num_samples))
            return PIQAHandler(loaded_set, "piqa")
        if dataset_name == "physical_qa":
            dataset = []
            with open("datasets\\train_rand_split.jsonl") as f:
                for i,line in enumerate(f.readlines()):
                    if i == num_samples:
                        break
                    else:
                        dataset.append(json.loads(line))
                return PhysicalCSHandler(dataset, "Physical QA")
        if dataset_name == "codah":
            loaded_set = load_dataset("codah", "codah")["train"].select(range(0, num_samples))
            return CODAHHandler(loaded_set, "CODAH")
        if dataset_name == "social_iqa":
            loaded_set = load_dataset("social_i_qa")["train"].select(range(0, num_samples))
            return SocialIQAHandler(loaded_set, "social_iqa")


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
        return {"answerKey":"A"} if sample["label"] == 0 else {"answerKey":"B"}

    def get_mcoptions(self, sample):
        return f"(A) {sample['sol1']} (B) {sample['sol2']}"

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
            mcoptions += f"({choice['label']}) {choice['text']} "
        return mcoptions

    def get_answer(self, sample):
        choices = sample["question"]["choices"]
        key = sample["answerKey"]
        for choice in choices:
            if choice['label'] == key:
                return choice["text"]
        return "none"

class CODAHHandler(DatasetHandler):

    idx_to_char = {0:"A", 1:"B", 2:"C", 3:"D"}

    def get_question(self, sample):
        return sample["question_propmt"]

    def get_meta(self, sample):
        #getting the answerKey
        return {"answerKey": self.idx_to_char[sample["correct_answer_idx"]]}

    def get_mcoptions(self, sample):
        choices = sample["candidate_answers"]
        mcoptions = ""
        for i,choice in enumerate(choices):
            mcoptions += f"({self.idx_to_char[i]}) {choice} "
        return mcoptions

    def get_answer(self, sample):
        choices = sample["candidate_answers"]
        return choices[sample["correct_answer_idx"]]


class SocialIQAHandler(DatasetHandler):

    idx_to_char = {"1":"A", "2":"B", "3":"C"}

    def get_question(self, sample):
        return f"{sample['context']} {sample['question']}"

    def get_meta(self, sample):
        #getting the answerKey
        return {"answerKey": self.idx_to_char[sample["label"]]}

    def get_choices(self, sample):
        return [sample["answerA"], sample["answerB"], sample["answerC"]]

    def get_mcoptions(self, sample):
        choices = self.get_choices(sample)
        mcoptions = ""
        for i,choice in enumerate(choices):
            mcoptions += f"({self.idx_to_char[str(i+1)]}) {choice} "
        return mcoptions

    def get_answer(self, sample):
        choices = self.get_choices(sample)
        return choices[int(sample["label"]) - 1]


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






