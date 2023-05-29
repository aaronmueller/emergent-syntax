import argparse
import datasets
import os
from transformers import T5Config
from t5_tokenizer_model import SentencePieceUnigramTokenizer

# get dataset path from argument
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", type=str, default="/scratch/am12057/BabyBERTa/data/corpora/aochildes.txt")
parser.add_argument("--ablation", "-a", type=str, default="small-nl8")
parser.add_argument("--vocabsize", "-v", type=int, default=8192)
parser.add_argument("--dataname", "-n", type=str)
parser.add_argument("--train_tokenizer", "-t", action="store_true")
args = parser.parse_args()

vocab_size = args.vocabsize
input_sentence_size = None

# Initialize a dataset
dataset_name = os.path.basename(args.dataset).split(".txt")[0]
dataset = datasets.load_dataset("text", data_files={"train": args.dataset}, split="train", name=dataset_name)

tokenizer = SentencePieceUnigramTokenizer(unk_token="<unk>", eos_token="</s>", pad_token="<pad>")

# Build an iterator over this dataset
def batch_iterator(input_sentence_size=None):
    if input_sentence_size is None:
        input_sentence_size = len(dataset)
    batch_length = 100
    for i in range(0, input_sentence_size, batch_length):
        yield dataset[i: i + batch_length]["text"]

# Train tokenizer
if args.train_tokenizer:
    tokenizer.train_from_iterator(
        iterator=batch_iterator(input_sentence_size=input_sentence_size),
        vocab_size=vocab_size,
        show_progress=True,
    )

# Save tokenizer and hyperparameter configs to a directory
ablation = args.ablation
dataname = args.dataname
if not os.path.exists(f"{dataname}-{ablation}"):
    os.mkdir(f"{dataname}-{ablation}")

if args.train_tokenizer:
    tokenizer.save(f"./{dataname}-{ablation}/tokenizer.json")

config = T5Config.from_pretrained(f"google/t5-efficient-{ablation}", vocab_size=tokenizer.get_vocab_size())
config.save_pretrained(f"./{dataname}-{ablation}")
