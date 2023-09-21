# How to Plant Trees 🌳 in Language Models
This repository provides code for our ACL 2023 paper, "How to Plant Trees 🌳 in Language Models: Data and Architectural Effects on the Emergence of Syntactic Inductive Biases."

## Requirements
We use python3.8.5.

Use `pip install -r requirements.txt` to install dependencies.

## Pre-training
We adapt pre-training code provided by HuggingFace when pre-training models from scratch. This is available in the root folder of this repository as `run_t5_mlm_flax.py`; our hyperparameters are in the `train_*_ablations.sh` scripts. We adapt the python script to allow for stopping after a specified number of training steps (as opposed to epochs).

Before running pre-training, you must first train the tokenizer and specify the architectural hyperparameters in a config file. We provide a single script for doing this:

```
python tokenizer_and_config.py --dataset <path/to/dataset> --ablation <ablation_name> --vocabsize <integer> --dataname <output_dir_name_prefix> --train_tokenizer
```
where `ablation_name` is the name of a T5-efficient config on huggingface, such as `base`, `base-nl8`, `small-nl4`, etc. These correspond to `google/t5-efficient-<ablation_name>` on the huggingface hub. The `--train_tokenizer` argument specifies that we should train and save a tokenizer; omit this if you simply want to save a config file. This will output a directory named `<dataname>-<ablation>`. For CHILDES, we use `--dataname childest5`, for example.

Now that you have a directory containing an architectural config and tokenizer, you can use our `train_*_ablations.sh` scripts to pre-train models from scratch. For example, to pre-train a model on CHILDES:

```
./train_childes_ablations.sh <ablation>
```
where `ablation` is the same as the argument you used when using the `tokenizer_and_config.py` script.

###  Data
We re-use data and preprocessing code from other sources where possible. Specifically, we use the child-directed age-ordered CHILDES corpus, as provided by the code from [Huebner et al. (2021)](https://aclanthology.org/2021.conll-1.49/): https://github.com/phueb/BabyBERTa/tree/master

We also use their code to subsample sentences from larger corpora into smaller corpora. Specifically, see the `load_wikipedia_sentences` method in [this script](https://github.com/phueb/BabyBERTa/blob/dae23f7a968158636f6143e98062e8102902eb4a/babyberta/io.py#L83), and set the `percent` argument to a value that will yield the correct word count.

For Wikipedia data, we download [Wikipedia dumps](https://dumps.wikimedia.org) and preprocess them using [witokit](https://github.com/akb89/witokit). We start by grabbing 1B words (after word tokenization with `nltk`), and then use Huebner's script to subsample this into the 100M word dataset. We use Huebner's 10M word wikipedia dataset (`wikipedia1.txt` in their repository) as the 10M word dataset, and subsample this to get the 1M word dataset. To ensure that smaller datasets are subsets of larger datasets, we concatenate the 10M word dataset to the 100M and 1B word dataset. We follow the same procedure with [Simple Wikipedia](https://dumps.wikimedia.org/simplewiki/).

For C4 data, we use the huggingface version of the dataset. We iteratively save data in streaming mode to avoid loading the entire dataset, which may not fit into memory. We save data until we have approximately 1B words (after word tokenization with `nltk`), then use Huebner's script as before to subsample this into 100M, 10M, and 1M word datasets.

## Fine-tuning
We very slightly adapt [code from Mueller et al. (2022)](https://github.com/sebschu/multilingual-transformations) to fine-tune our models, as well as existing T5 models on huggingface from [Tay et al. (2022)](https://arxiv.org/abs/2109.10686). These are provided in the `scripts` directory. For the syntactic transformations data, clone [this repository](https://github.com/sebschu/multilingual-transformations) and copy the `data` folder into the root folder of this repository.

Note that we provide separate scripts for fine-tuning the models from Tay et al. and for fine-tuning models that we pre-train from scratch. These are the same w.r.t. hyperparameters, but slightly differ because we're loading from local checkpoints, and because we're loading `flax` models.

To fine-tune the models provided by Tay et al., navigate to the `scripts` directory and run these scripts:
```
./finetune_t5_question_en.sh <ablation> <seed>
./finetune_t5_passivize_en.sh <ablation> <seed>
```

To evaluate these models after fine-tuning, run these commands:
```
./eval_t5_question_en.sh <ablation> <seed> <split>
./eval_t5_passivize_en.sh <ablation> <seed> <split>
```
where `split` is either `test` or `gen`. `test` is the in-distribution transformations (to measure whether models have learned to perform the transformations), and `gen` is the generalization set (to measure models' inductive biases).

To fine-tune the models that we pre-train from scratch using the code in the "Pre-training" section, use these scripts:
```
./finetune_ourmodel_question_en.sh <path/to/model> <seed>
./finetune_ourmodel_passivize_en.sh <path/to/model> <seed>
```

And to evaluate, run these:

```
./eval_ourmodel_question_en.sh <path/to/model> <seed> <split>
./eval_ourmodel_passivize_en.sh <path/to/model> <seed> <split>
```

## Citation
@inproceedings{mueller-linzen-2023-plant,
    title = "How to Plant Trees in Language Models: Data and Architectural Effects on the Emergence of Syntactic Inductive Biases",
    author = "Mueller, Aaron  and
      Linzen, Tal",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.629",
    doi = "10.18653/v1/2023.acl-long.629",
    pages = "11237--11252"
}

## License
This repository is made available under an MIT license.
