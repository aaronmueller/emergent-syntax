#!/bin/bash

#SBATCH --job-name=MT-base-finetune-passiv-en
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=1 
#SBATCH --mem=20GB 
#SBATCH --time=20:00:00 
#SBATCH --gres=gpu:rtx8000:1

source /ext3/env.sh
conda activate syntaxscaling

python ../models/run_seq2seq.py \
    --model_name_or_path 'google/t5-efficient-$1' \
    --do_train \
    --task translation_src_to_tgt \
    --train_file ../data/passiv_en_nps/passiv_en_nps.train.json \
    --validation_file ../data/passiv_en_nps/passiv_en_nps.dev.json \
    --output_dir /scratch/am12057/t5-$1-finetuning-passivization-en-nps-bs128/  \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=2 \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate \
    --logging_steps=150 \
    --eval_steps=150 \
    --num_train_epochs 10.0
