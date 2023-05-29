#!/bin/bash

#SBATCH --job-name=train-t5
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30GB
#SBATCH --gres=gpu:v100:1
#SBATCH --time=23:59:59

size=$1

source /ext3/env.sh
conda activate syntaxscaling

mkdir childest5-$size
python run_t5_mlm_flax.py \
	--output_dir ./childest5-$size \
	--model_type t5 \
	--config_name ./childest5-$size \
	--tokenizer_name ./childest5-$size \
	--train_file $SCRATCH/BabyBERTa/data/corpora/aochildes.txt \
	--max_seq_length 128 \
	--per_device_train_batch_size='16' \
	--per_device_eval_batch_size='16' \
	--adafactor \
	--learning_rate='0.005' \
	--weight_decay='0.001' \
	--warmup_steps='2000' \
	--overwrite_output_dir \
	--logging_steps='2500' \
	--save_steps='10000' \
	--eval_steps='999999' \
	--num_train_steps='130000' \
