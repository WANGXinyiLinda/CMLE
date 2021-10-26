import torch
import argparse
import csv, json
import sys, os
from datasets import Dataset
import numpy as np
import time

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

label_to_name = {'e': 'entailment', 'n': 'neutral', 'c': 'contradiction'}
label_list = ['entailment', 'neutral', 'contradiction']
label_to_id = {'e': 0, 'n': 1, 'c': 2}
labels = ['e', 'n', 'c']
num_labels = 3
is_regression = False

def predict_label(model, input_ids, attention_mask, token_type_ids):
    logits = model(input_ids, attention_mask=attention_mask,       
                    token_type_ids=token_type_ids)
    # print(logits)
    pred = torch.argmax(logits[0], -1)
    # print("predicted labels: ", pred)
    return pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fine_tuned_roberta_model",
        "-M",
        type=str,
        default=None,
        help="fine tuned roberta model name or path to local checkpoint",
    )
    parser.add_argument(
        "--out_dir",
        "-O",
        type=str,
        default=None,
        help="save to file or not",
    )
    parser.add_argument(
        "--revert",
        "-R",
        type=str,
        default=None,
        help="choose among h and p",
    )
    parser.add_argument(
        "--cache_dir",
        "-C",
        type=str,
        default=None,
        help="save to file or not",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="dataset to be augmented",
    )
    parser.add_argument(
        "--gen_file",
        type=str,
        default=None,
        help="path to file containing generated data",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument("--hypothesis_max_length", type=int, default=64)
    parser.add_argument("--premise_max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--which_gpu", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    args = parser.parse_args()

    # set the device
    device = "cuda:{}".format(args.which_gpu) if torch.cuda.is_available() and not args.no_cuda else "cpu"

    # set Random seed
    torch.manual_seed(args.seed)
    
    if args.revert == 'h':
        sentence1_key, sentence2_key = "premise", "hypothesis"
        sent1_max_length = args.premise_max_length
        sent2_max_length = args.hypothesis_max_length
    elif args.revert == 'p':
        sentence1_key, sentence2_key = "hypothesis", "premise" 
        sent1_max_length = args.hypothesis_max_length
        sent2_max_length = args.premise_max_length

    # load data
    train_data = {'premise': [], 'hypothesis': [], 'label': []}
    data_named_path = args.gen_file.split(',')
    for path in data_named_path:
        with open(path) as rf:
            for row in rf:
                d = json.loads(row.strip())
                train_data['premise'].append(d['premise'])
                train_data['hypothesis'].append(d['hypothesis'])
                train_data['label'].append(label_to_id[d['labels']])

    train_dataset = Dataset.from_dict(train_data)
    num_data = len(train_dataset)
    print("number of data: ", num_data)

    # load pretrained bert based classifier model
    classifier_config = AutoConfig.from_pretrained(
        args.fine_tuned_roberta_model,
        num_labels=num_labels,
    )
    classifier_tokenizer = AutoTokenizer.from_pretrained(
        args.fine_tuned_roberta_model, use_fast=True
    )
    classifier_model = AutoModelForSequenceClassification.from_pretrained(
        args.fine_tuned_roberta_model,
        from_tf=bool(".ckpt" in args.fine_tuned_roberta_model),
        config=classifier_config,
    )
    classifier_model.eval().to(device)
    # Freeze BERT weights
    for param in classifier_model.parameters():
        param.requires_grad = False

    if args.out_dir:
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        preds_file = os.path.join(args.out_dir, 
            "{}_aug_{}_pred.jsonl".format(args.test_file, args.revert))
        if os.path.isfile(preds_file):
            print("output file already exists. Change a directory/a new file name.")
            exit()
        pred_writer = open(preds_file, "w")
    
    num_suc = 0
    num_total = 0
    sent1_batch = []
    sent2_batch = []
    bart_decoder_input_batch = []
    pert_label_batch = []
    start_time = time.time()
    for i, example in enumerate(train_dataset): # use eval for debug only !!!
        sent2_batch.append(example[sentence2_key])
        sent1_batch.append(example[sentence1_key])
        pert_label_batch.append(example['label'])
        bart_decoder_input_batch.append('[' + labels[example['label']] + ']')
        assert len(sent1_batch) == len(sent2_batch)

        if len(sent1_batch) == args.batch_size or (i == num_data-1 and len(sent1_batch) > 0):
            passed_time = (time.time() - start_time)/60
            estimated_time = (passed_time/i)*num_data
            print("progress: {}/{}  estimated time: {}/{}".format(i, num_data, passed_time, estimated_time))
            encoded_example = classifier_tokenizer(sent1_batch, sent2_batch, 
                        padding="max_length", max_length=128, truncation=True, return_token_type_ids=True)
            input_ids = torch.tensor(encoded_example['input_ids'], device=device)
            attention_mask = torch.tensor(encoded_example['attention_mask'], device=device)
            token_type_ids = torch.tensor(encoded_example['token_type_ids'], device=device)
            pred = predict_label(classifier_model, input_ids, attention_mask, token_type_ids)
            # print("perturbing to: ", perturb_label_batch)
            num_suc += torch.sum(pred == torch.tensor(pert_label_batch, device=device)).tolist()
            num_total += len(pred)
            rate = num_suc/num_total
            print("perturb sucess rate: ", rate)
            
            for sent1, sent2, p, pl in zip(sent1_batch, sent2_batch, pred.tolist(), pert_label_batch):
                if args.out_dir:
                    pred_writer.write(json.dumps({sentence1_key: sent1, 
                        sentence2_key: sent2, "labels": labels[pl],
                        "cls_pred_label": labels[p]})+'\n')

            sent1_batch = []
            sent2_batch = []
            bart_decoder_input_batch = []
            orig_text_batch = []
            pert_label_batch = []

    with open(os.path.join(args.out_dir, "{}_{}_log.txt".format(args.test_file, args.revert)), 'w') as wf:
        wf.write("success rate: {}\n".format(rate))
        wf.write("num generated data: {}".format(num_data))        