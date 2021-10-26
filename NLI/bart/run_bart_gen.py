import torch
import argparse
import csv, json
import sys, os
from datasets import Dataset
import numpy as np

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

registered_path = {}

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
        "--pretrained_bart_model",
        "-M",
        type=str,
        default="facebook/bart-large",
        help="pretrained model name or path to local checkpoint",
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
        "--classifier_model_name_or_path",
        type=str,
        default=None,
        help="pretrained classifier",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="train data names",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of samples to generate from the modified latents",
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

    # load pretrained BART model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.pretrained_bart_model,
        cache_dir = args.cache_dir,
        output_hidden_states=True
    )
    model.eval().to(device)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_bart_model, cache_dir=args.cache_dir)
    # Freeze BART weights
    for param in model.parameters():
        param.requires_grad = False

    # load data
    train_data = {'premise': [], 'hypothesis': [], 'label': []}
    data_named_path = args.test_file.split(',')
    for path in data_named_path:
        if path in registered_path:
            with open(registered_path[path]) as rf:
                for row in rf:
                    d = json.loads(row.strip())
                    train_data['premise'].append(d['premise'])
                    train_data['hypothesis'].append(d['hypothesis'])
                    train_data['label'].append(label_to_id[d['label']])
        else:
            with open(path) as rf:
                reader = csv.DictReader(rf)
                for row in reader:
                    if int(row['label']) == int(row['perturb']):
                        train_data['premise'].append(row['premise'])
                        train_data['hypothesis'].append(row['hypothesis'])
                        train_data['label'].append(int(row['label']))
    train_dataset = Dataset.from_dict(train_data)
    num_data = len(train_dataset)

    # load pretrained bert based classifier model
    classifier_config = AutoConfig.from_pretrained(
        args.classifier_model_name_or_path,
        num_labels=num_labels,
    )
    classifier_tokenizer = AutoTokenizer.from_pretrained(
        args.classifier_model_name_or_path, use_fast=True
    )
    classifier_model = AutoModelForSequenceClassification.from_pretrained(
        args.classifier_model_name_or_path,
        from_tf=bool(".ckpt" in args.classifier_model_name_or_path),
        config=classifier_config,
    )
    classifier_model.eval().to(device)
    # Freeze BERT weights
    for param in classifier_model.parameters():
        param.requires_grad = False

    if args.out_dir:
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        output_test_preds_file = os.path.join(args.out_dir, 
            "{}_aug_{}.jsonl".format(args.test_file, args.revert))
        if os.path.isfile(output_test_preds_file):
            print("output file already exists. Change a directory/a new file name.")
            exit()
        out_writer = open(output_test_preds_file, "w")
    
    num_suc = 0
    num_total = 0
    sent1_batch = []
    sent2_batch = []
    bart_decoder_input_batch = []
    orig_text_batch = []
    true_label_batch = []
    pert_label_batch = []
    for i, example in enumerate(train_dataset): # use eval for debug only !!!
        for label in [0, 1, 2]:
            if label != example['label']:
                sent2_batch.append(example[sentence2_key])
                orig_text_batch.append(example[sentence2_key])
                sent1_batch.append(example[sentence1_key])
                true_label_batch.append(example['label'])
                pert_label_batch.append(label)
                bart_decoder_input_batch.append('[' + labels[label] + ']')
        assert len(sent1_batch) == len(orig_text_batch)

        while abs(len(orig_text_batch) - args.batch_size) < 2 or (i == num_data-1 and len(orig_text_batch) > 0):
            print("progress: {}/{}".format(i, num_data))
            input_ids = tokenizer(sent1_batch, max_length=sent1_max_length, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(device)
            decoder_input_ids = tokenizer(bart_decoder_input_batch, max_length=1, add_special_tokens=False)['input_ids']
            for i in decoder_input_ids:
                i.append(tokenizer.bos_token_id)
            decoder_input_ids = torch.tensor(decoder_input_ids, device=device)
            # print("example encoder input: ", input_ids)
            # print("example decoder input: ", decoder_input_ids)
            outputs = model.generate(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                max_length=sent2_max_length, 
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True,
                # do_sample=True,
                # top_k=25, 
                # top_p=0.95,
                num_return_sequences=1
                )
            print("Output:\n" + 100 * '-')
            print()
            gen_sent2 = []
            for true_label, pert_label, sent1, orig, output in zip(true_label_batch, pert_label_batch, sent1_batch, orig_text_batch, outputs):
                sent2 = tokenizer.decode(output, skip_special_tokens=True)
                # print(output)
                gen_sent2.append(sent2)
                print(sentence1_key + ": " + sent1)
                print("original " + label_list[true_label] + " " + sentence2_key + ": " + orig)
                print("generated " + label_list[pert_label] + " " + sentence2_key + ": " + sent2)
                print()

            encoded_example = classifier_tokenizer(sent1_batch, gen_sent2, 
                        padding="max_length", max_length=156, truncation=True, return_token_type_ids=True)
            input_ids = torch.tensor(encoded_example['input_ids'], device=device)
            attention_mask = torch.tensor(encoded_example['attention_mask'], device=device)
            token_type_ids = torch.tensor(encoded_example['token_type_ids'], device=device)
            pred = predict_label(classifier_model, input_ids, attention_mask, token_type_ids)
            # print("perturbing to: ", perturb_label_batch)
            num_suc += torch.sum(pred == torch.tensor(pert_label_batch, device=device)).tolist()
            num_total += len(pred)
            print("perturb sucess rate: ", num_suc/num_total)
            
            for sent1, sent2, new_sent2, p, pl, tl in zip(sent1_batch, sent2_batch, gen_sent2, pred.tolist(), pert_label_batch, true_label_batch):
                if sent1!=new_sent2 and sent2!=new_sent2 and len(new_sent2) > 0 and args.out_dir:
                    out_writer.write(json.dumps({sentence1_key: sent1, 
                        sentence2_key: new_sent2, "labels": labels[pl],
                        "cls_pred_label": labels[p]})+'\n')

            sent1_batch = []
            sent2_batch = []
            bart_decoder_input_batch = []
            orig_text_batch = []
            true_label_batch = []
            pert_label_batch = []