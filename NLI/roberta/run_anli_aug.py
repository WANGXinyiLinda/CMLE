# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd 
from datasets import Dataset, load_dataset, load_metric, DatasetDict
import numpy as np
import json, csv
import copy
import random, math

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

registered_path = {}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "list of train files separated by ,"}
    )
    train_weights: Optional[str] = field(
        default=None, metadata={"help": "weight if each train file separated by ,"}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "list of dev files separated by ,"}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    label_list = ['entailment', 'neutral', 'contradiction']
    label_to_id = {'e': 0, 'n': 1, 'c': 2}
    num_labels = 3
    is_regression = False
    sentence1_key, sentence2_key = "premise", "hypothesis"

    def sample_data_list(d_list, ratio):
        if ratio <= 0:
            raise ValueError("Invalid training weight ratio. Please change --train_weights.")
        upper_int = int(math.ceil(ratio))
        if upper_int == 1:
            return d_list # if ratio is 1 then we just return the data list
        else:
            sampled_d_list = []
            for _ in range(upper_int):
                sampled_d_list.extend(copy.deepcopy(d_list))
            if np.isclose(ratio, upper_int):
                return sampled_d_list
            else:
                sampled_length = int(ratio * len(d_list))
                random.shuffle(sampled_d_list)
                return sampled_d_list[:sampled_length]

    train_data = {'premise': [], 'hypothesis': [], 'label': []}
    data_named_path = data_args.train_file.split(',')
    if data_args.train_weights:
        train_weights = data_args.train_weights.split(',')
        train_weights = [float(w) for w in train_weights]
    else:
        train_weights = [1 for _ in data_named_path]
    total_num_data = 0
    for path, weight in zip(data_named_path, train_weights):
        temp_data = {'premise': [], 'hypothesis': [], 'label': []}
        if path in registered_path:
            with open(registered_path[path]) as rf:
                for row in rf:
                    d = json.loads(row.strip())
                    if "cls_pred_label" in d:
                        if d["cls_pred_label"] == d["labels"]:
                            temp_data['premise'].append(d['premise'])
                            temp_data['hypothesis'].append(d['hypothesis'])
                            temp_data['label'].append(label_to_id[d['labels']])
                    else:
                        temp_data['premise'].append(d['premise'])
                        temp_data['hypothesis'].append(d['hypothesis'])
                        temp_data['label'].append(label_to_id[d['label']])
                
        else:
            with open(path, 'r') as rf:
                rf = (line.replace('\0','') for line in rf)
                reader = csv.DictReader(rf)
                for row in reader:
                    temp_data['premise'].append(row['premise'])
                    temp_data['hypothesis'].append(row['hypothesis'])
                    temp_data['label'].append(int(row['label']))
        train_data['premise'].extend(sample_data_list(temp_data['premise'], weight))
        train_data['hypothesis'].extend(sample_data_list(temp_data['hypothesis'], weight))
        train_data['label'].extend(sample_data_list(temp_data['label'], weight))
        print("#" + path + ": ", len(train_data['label']) - total_num_data)
        total_num_data = len(train_data['label'])
    train_dataset = Dataset.from_dict(train_data)

    eval_dataset = {}
    data_named_path = data_args.validation_file.split(',')
    for path in data_named_path:
        data = {'premise': [], 'hypothesis': [], 'label': []}
        if path in registered_path:
            with open(registered_path[path]) as rf:
                for row in rf:
                    d = json.loads(row.strip())
                    if "cls_pred_label" in d:
                        if d["cls_pred_label"] == d["labels"]:
                            data['premise'].append(d['premise'])
                            data['hypothesis'].append(d['hypothesis'])
                            data['label'].append(label_to_id[d['labels']])
                    else:
                        data['premise'].append(d['premise'])
                        data['hypothesis'].append(d['hypothesis'])
                        data['label'].append(label_to_id[d['label']])
        else:
            with open(path) as rf:
                reader = csv.DictReader(rf)
                for row in reader:
                    data['premise'].append(row['premise'])
                    data['hypothesis'].append(row['hypothesis'])
                    data['label'].append(int(row['label']))
        eval_dataset[path] = Dataset.from_dict(data)
    eval_dataset = DatasetDict(eval_dataset)
        
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
        max_length = data_args.max_seq_length
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
        max_length = None

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)
        return result

    train_dataset = train_dataset.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        data_collator=default_data_collator if data_args.pad_to_max_length else None,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()  # Saves the tokenizer too for easy upload

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)

        for task in eval_dataset:
            dataset = eval_dataset[task]
            eval_result = trainer.evaluate(eval_dataset=dataset)

            output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info(f"***** Eval results {task} *****")
                    for key, value in eval_result.items():
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")

            eval_results.update(eval_result)

    return eval_results


if __name__ == "__main__":
    main()
