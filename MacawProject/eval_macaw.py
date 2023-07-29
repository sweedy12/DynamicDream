import torch
import json
import argparse
import pandas as pd
import os
import re
import string
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cache_directory = '/cs/labs/dshahaf/orens/huggingface'


def read_args():
    parser = argparse.ArgumentParser()


    parser.add_argument('--model_name', type=str, default='allenai/macaw-3b', help='choose the name of the model')

    parser.add_argument('--samples_size', type=int, default=-1, help='choose how many samples to read')

    parser.add_argument('--run_with_context', type=int, default=0)

    parser.add_argument('--dataset_file_name', type=str, default="",
                        help='choose the dataset file name')

    args = parser.parse_args()
    return args


def read_samples(filename, num_samples_to_read, run_with_context):
    data = []
    count_samples = 0
    with open(filename, 'r') as f:
        for l in f:
            count_samples += 1
            if count_samples == num_samples_to_read + 1:
                break
            line = json.loads(l)
            sample_id = line["id"]
            mcoptions = line["mcoptions"]
            question = line['question']
            context = ""
            if run_with_context:
                context = line['context']
            answer = line['answer']
            d = {"sample_id": sample_id, "mc": mcoptions, "question": question,
                 "context": context, "answer": answer}
            data.append(d)
        return data


def call_macaw(tokenizer, model, sample, with_context):
    input_string = "$answer$ ; $mcoptions$ = " + sample['mc'] + " ; $question$ = " + sample['question']
    if with_context:
        print("run macaw with context")
        input_string += " $context$ = " + sample['context']

    input_ids = tokenizer.encode(input_string, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=200)
    macaw_answer = tokenizer.batch_decode(output, skip_special_tokens=True)
    return macaw_answer

def remove_punctuation(input_string):
    translator = str.maketrans('', '', string.punctuation)
    no_punct = input_string.translate(translator)
    return no_punct




def macaw_eval_samples(tokenizer, model, samples, with_context):
    df = pd.DataFrame(columns=['gt_ans', 'macaw_ans', 'is_correct'])
    for sample in samples:
        macaw_answer = call_macaw(tokenizer, model, sample, with_context=with_context)[0].split('=')[1]
        macaw_answer = remove_punctuation(macaw_answer)
        macaw_answer = macaw_answer.strip()
        macaw_answer = re.sub(' +', ' ', macaw_answer)

        gt_answer = sample['answer']
        gt_answer = remove_punctuation(gt_answer)
        gt_answer = gt_answer.strip()
        gt_answer = re.sub(' +', ' ', gt_answer)

        is_correct = 1 if gt_answer == macaw_answer else 0
        df.loc[len(df)] = [gt_answer, macaw_answer, is_correct]
    return df


def main(args):
    num_samples_to_read = args.samples_size
    run_with_context = True if args.run_with_context == 1 else False
    dataset_file_name = args.dataset_file_name
    model_name = args.model_name

    print("num samples to read: " + str(num_samples_to_read))
    print("should run with context: " + str(run_with_context))
    print("dataset file name: " + str(dataset_file_name))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Finished to load macaw tokenizer successfully")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_directory).to(device)
    print("Finished to load macaw model successfully")

    samples = read_samples(dataset_file_name, num_samples_to_read, run_with_context)
    print("Finished to read the dataset successfully")

    print()
    print("---macaw evaluation")
    print()

    df = macaw_eval_samples(tokenizer, model, samples, with_context=run_with_context)

    output_file_name = 'dataset_' + os.path.basename(dataset_file_name).split('.')[0] + '_' + 'model_' + model_name.split('/')[1] + '_sample_size_' + str(num_samples_to_read) + \
                    '_with_context_' + str(run_with_context)

    print("saving detailed results into: " + output_file_name + '_results_details' + '.csv')
    df.to_csv(output_file_name + '_results_details' + '.csv')

    count_correct_predictions = df[df['is_correct'] == 1].shape[0]
    count_predictions = len(df)

    print("saving summary results into: " + output_file_name + '_results_summary' + '.txt')
    results_summary_output_file = open(output_file_name + '_results_summary' + '.txt', 'w')
    results_summary_output_file.write('number of correct predictions: ' +
                                      str(count_correct_predictions) + '\n' +
                                      'total number of predictions: ' + str(count_predictions) + '\n' +
                                      'accuracy: ' + str(count_correct_predictions / count_predictions))
    results_summary_output_file.close()

if __name__ == "__main__":
    args = read_args()
    main(args)

