import jsonlines
import datasets
import os

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


annotated_path = "samples\\annotated_data.jsonl"
save_dir = "datasets\\annotated_dataset.hf"
CLdataset = CreateLoadDataset(annotated_path, save_dir)
dataset = CLdataset.save_or_load_dataset()
nir = 1





