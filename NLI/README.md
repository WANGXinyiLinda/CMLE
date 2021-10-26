# NLI experiments

## Requirements 

* Transformers (please install my version of Transformers under this directory)
* pytorch 1.7.0

The NLI data I used is adopted from ANLI [https://github.com/facebookresearch/anli]. Please build the dataset according to their instruction.

## Usage:

To train a MLE bart that generate hypothesis conditioning on premise and label:
```
    python bart/run_bart.py \
        --model_name_or_path facebook/bart-large \
        --do_train \
        --do_eval \
        --task NLI \
        --revert h \
        --max_source_length 64 \
        --max_target_length 128 \
        --pad_to_max_length \
        --train_file snli_train,mnli_train \
        --validation_file snli_dev,snli_test,mnli_m_dev,mnli_mm_dev \
        --cache_dir /pth \
        --output_dir /out \
        --per_device_train_batch_size=32 \
        --per_device_eval_batch_size=32 \
        --gradient_accumulation_steps 4 \
        --predict_with_generate \
        --fp16 \
        --fp16_opt_level O2 \
```
To train a Implicit CMLE bart:
```
    python bart/run_bart_ipm.py \
        --model_name_or_path facebook/bart-large \
        --do_train \
        --do_eval \
        --task NLI \
        --revert h \
        --max_source_length 128 \
        --max_target_length 64 \
        --pad_to_max_length \
        --save_steps 5000 \
        --train_file mnli_train,snli_train \
        --validation_file snli_dev,snli_test,mnli_m_dev,mnli_mm_dev \
        --cache_dir /pth \
        --output_dir /out \
        --per_device_train_batch_size=32 \
        --per_device_eval_batch_size=32 \
        --gradient_accumulation_steps 4 \
        --predict_with_generate \
        --fp16 \
        --fp16_opt_level O2 \
```
To train a Explicit CMLE bart:
```
    python bart/run_bart_gumbel.py \
        --model_name_or_path facebook/bart-large \
        --clf_model_name_or_path /model \
        --do_train \
        --do_eval \
        --task NLI \
        --task_name MNLI \
        --revert h \
        --max_source_length 128 \
        --max_target_length 64 \
        --pad_to_max_length \
        --save_steps 5000 \
        --train_file mnli_train,snli_train \
        --validation_file snli_dev,snli_test,mnli_m_dev,mnli_mm_dev \
        --cache_dir /pth \
        --output_dir /out \
        --gradient_accumulation_steps 8 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 16 \
        --predict_with_generate \
        --fp16 \
        --fp16_opt_level O2 \
```
To generate augmented data using bart:
```
    export NAME=xxx

    python bart/run_bart_gen_fast.py \
        --pretrained_bart_model /model \
        --revert h \
        --test_file $NAME \
        --out_dir /out \
        --batch_size 64
        
    python bart/run_roberta_pred_fast.py \
        --fine_tuned_roberta_model /model \
        --revert h \
        --test_file $NAME \
        --gen_file /out/$NAME_aug_h.jsonl \
        --out_dir /out \
        --batch_size 128
```
To train a roberta classifier:
```
    python roberta/run_anli_aug.py \
    --model_name_or_path roberta-large \
    --task_name MNLI \
    --train_file snli_train,mnli_train,generated_data \
    --validation_file mnli_m_dev,mnli_mm_dev,snli_dev,snli_test,anli_r1_dev,anli_r1_test,anli_r2_dev,anli_r2_test,anli_r3_dev,anli_r3_test \
    --do_train \
    --do_eval \
    --max_seq_length 156 \
    --save_steps 5000 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --num_train_epochs 2.0 \
    --cache_dir /pth \
    --output_dir /out \
    --fp16 \
    --fp16_opt_level O2 \
```
