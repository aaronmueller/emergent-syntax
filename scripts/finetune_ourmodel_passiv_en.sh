#!/bin/bash

#SBATCH --job-name=MT-base-finetune
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=1 
#SBATCH --mem=20GB 
#SBATCH --time=20:00:00 
#SBATCH --gres=gpu:rtx8000:1

source /ext3/env.sh
conda activate syntaxscaling

python ../models/run_seq2seq.py \
    --model_name_or_path "$1/checkpoint-130000/" \
    --do_train \
	--from_flax \
    --task translation_src_to_tgt \
    --train_file ../data/passiv_en_nps/passiv_en_nps.train.json \
    --validation_file ../data/passiv_en_nps/passiv_en_nps.dev.json \
    --output_dir /scratch/am12057/$1-seed$2-mccoy-finetuning-passiv-en-nps-bs128/  \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=2 \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate \
    --num_train_epochs 10.0 \
	--seed $2
