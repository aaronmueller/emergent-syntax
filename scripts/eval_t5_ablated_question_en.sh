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
    --model_name_or_path 'google/t5-efficient-$1' \
    --do_eval \
    --do_learning_curve \
    --task translation_src_to_tgt \
    --train_file ../data/question_have-havent_en/question_have.train.json \
    --validation_file ../data/question_have-havent_en/question_have.$2.json \
    --output_dir $SCRATCH/t5-$1-mccoy-finetuning-question-have-bs128/  \
    --per_device_train_batch_size=128 \
    --per_device_eval_batch_size=64 \
    --overwrite_output_dir \
    --predict_with_generate \
