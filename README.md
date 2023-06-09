#Dynamic Dream data curator

This is the code for sampling training examples from the following datasets:
1. CODAH
2. Social-IQA
3. PiQA
4. Physical_QA

# Usage
To sample from the databases, run:
```DataHandler.py --num_samples <n> --dataset_name <name> --save-dir <dir>```
Where:
```n``` is the number of examples to be collected from the dataset (default: 100).
```name``` is the name of the dataset to sample from (options: "codah", "piqa", "physical_qa", "social_iqa", default: piqa).
```dir``` is the directory to be created to store the results under (default is "samples").


# Output
The out is a Jsonl file.
Each line consists of the question (or situation + question, if situation exists), the multiple-choice options (mcoptions), the answerkey and the answer itself. 
Example:
{"question": "Cameron decided to have a barbecue and gathered her friends together. How would Others feel as a result?", "id": "social_iqa_train1", "meta": {"answerKey": "A"}, "mcoptions": "(A) like attending (B) like staying home (C) a good friend to have ", "answer": "like attending"}

