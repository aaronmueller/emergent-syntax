#!/bin/bash

#SBATCH --job-name=MT-base-eval-de-canaux
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=1 
#SBATCH --mem=20GB 
#SBATCH --time=20:00:00 
#SBATCH --gres=gpu:rtx8000:1

source /ext3/env.sh
conda activate syntaxscaling

python ../models/run_seq2seq.py \
    --model_name_or_path '$1/checkpoint-130000/' \
    --do_eval \
    --do_learning_curve \
	--from_flax \
    --task translation_src_to_tgt \
    --train_file ../data/passiv_en_nps/passiv_en_nps.train.json \
    --validation_file ../data/passiv_en_nps/passiv_en_nps.$3.json \
    --output_dir $SCRATCH/$1-seed$2-mccoy-finetuning-passiv-en-nps-bs128/  \
    --per_device_train_batch_size=128 \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate \
