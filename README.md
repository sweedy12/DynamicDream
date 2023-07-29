#Dynamic Scene Elaboration project

This repository contains the entire code and data for the Dynamic Scene Elaboration project.



Finetuning T5 models for the dynamic elaboration task:

Done with FinetuneT5.py

#Usage
To run training, run:
``` FinetuneT5.py --model_name <model-name>--raw_data_path <data-path> --train_epochs <num-epochs> --run_train --model_save_dir <save-dir> --dataset_save_dir <dataset_path> ```


To run inference, run:
```FinetuneT5.py --pretrained_model_path <model-path> --inference_test_set <test-set> --run_inference --test_save_dir <save-dir> --inference_size <size> ```


Sampling datasets:
Done with DatasetHandelt.py
This is the code for sampling training examples from the following datasets:
1. CODAH
2. Social-IQA
3. PiQA
4. Commonsense-QA
5. ETHICS

# Usage
To sample from the databases, run:
```DataHandler.py --num_samples <n> --dataset_name <name> --save-dir <dir> --fold <f>```
Where:
```n``` is the number of examples to be collected from the dataset (default: 100).
```name``` is the name of the dataset to sample from (options: "codah", "piqa", "physical_qa", "social_iqa", default: piqa).
```dir``` is the directory to be created to store the results under (default is "samples").
```f``` is the fold to extract the data from  (options: "train", "validation", "test", default is "train").


# Output
The out is a Jsonl file.
Each line consists of the question (or situation + question, if situation exists), the multiple-choice options (mcoptions), the answerkey and the answer itself. 
Example:
{"question": "Cameron decided to have a barbecue and gathered her friends together. How would Others feel as a result?", "id": "social_iqa_train1", "meta": {"answerKey": "A"}, "mcoptions": "(A) like attending (B) like staying home (C) a good friend to have ", "answer": "like attending"}


#Evaluating Macaw QA performance


