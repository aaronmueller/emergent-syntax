import numpy as np
import os
import re
import sys
import math
import glob
import pickle
import argparse
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
from collections import Counter, defaultdict
from transformers import T5ForConditionalGeneration
sns.set()
sns.set(font_scale=1.5)
sns.set_style("whitegrid")

from metrics import compute_metrics

MODEL_NAME_MAP = {
    "t5": "T5",
    "mt5": "mT5",
    "bart": "BART",
    "mbart": "mBART"
}

def get_dirs(prefixes, suffixes):
    dirs = []
    base_path = "/scratch/am12057/"
    for filename in os.listdir(base_path):
        for prefix in prefixes:
            for suffix in suffixes:
                if filename.startswith(prefix) and filename.endswith(suffix):
                    full_path = os.path.join(base_path, filename)
                    if os.path.isdir(full_path):
                        dirs.append(full_path)

    return dirs


ABLATION_NAME_MAP = lambda x: f"({x})"

def main():
      argparser = argparse.ArgumentParser()
      
      # argparser.add_argument("--checkpoint_dirs")
      argparser.add_argument("--gold_filename")
      argparser.add_argument("--prefix", default="t5-")
      argparser.add_argument("--suffix", default="mccoy-finetuning-question-have-bs128")
      argparser.add_argument("--metrics")
      argparser.add_argument("--move_legend", action="store_true")
      argparser.add_argument("--move_legend_left", action="store_true")
      argparser.add_argument("--show_stddev", action="store_true")
      argparser.add_argument("--label_points", action="store_true")
      argparser.add_argument("--max", action="store_true")
      argparser.add_argument("--out_dir", default="scaling")    
      args = argparser.parse_args()

      if "," in args.prefix:
          prefix = args.prefix.split(",")
      else:
          prefix = [args.prefix]
      if "," in args.suffix:
          suffix = args.suffix.split(",")
      else:
          suffix = [args.suffix]

      checkpoint_dirs = get_dirs(prefix, suffix)
      
      model_results = defaultdict(dict)
      metric_names = args.metrics.split(",")
      metrics_str = "-".join(metric_names)
      basename = os.path.basename(args.gold_filename).replace(".json", "")
      params = {}
      for checkpoint_dir in checkpoint_dirs:
        # check for ablations in name of model
        dirname = os.path.basename(checkpoint_dir)
        model_name = "t5"
        model_size = dirname.split("-")[1]
        if model_size == "large":
            continue
        second_feature = os.path.basename(checkpoint_dir).split("-")[2]
        is_ablation = False
        if second_feature[:2] in ("nl", "el", "dl", "dm", "kv", "ff", "nh"):
            hf_model = f"google/t5-efficient-{model_size}-{second_feature}"
            is_ablation = True
        else:
            hf_model = f"google/t5-efficient-{model_size}"

        # get model name for annotations
        model_name = model_name.split("/")[-1]
        if is_ablation:
            model_prefix = MODEL_NAME_MAP[model_name]
            model_ablation = ABLATION_NAME_MAP(second_feature)
            model_name = f"{model_prefix} {model_ablation}" 
        else:
            model_prefix = MODEL_NAME_MAP[model_name]
            model_name = f"{model_prefix} ({model_size})"


        # get num parameters in model
        if model_name not in params:
            print(f"Getting num params in {hf_model}...")
            model = T5ForConditionalGeneration.from_pretrained(hf_model)
            params[model_name] = sum(p.numel() for p in model.parameters() if p.requires_grad)

        for path in glob.glob(os.path.join(checkpoint_dir, "checkpoint-*", "")):
            pred_filename = os.path.join(path, basename + ".eval_preds_seq2seq.txt")
            it_res = re.match(".*checkpoint-([0-9]+)[/].*", path)
            it = int(it_res.group(1))

            # skip datapoints from before convergence (conservative heuristic)
            if it < 1000:
                continue

            model_results[model_name][it] = compute_metrics(metric_names, pred_filename, args.gold_filename) 
   
      metric_rename = {}
      for metric in metric_names:
          if metric == "exact_match":
              this_metric = {"exact_match": "sequence"}
          elif metric == "first_word":
              this_metric = {"first_word": "main aux"}
          elif metric == "second_word":
              this_metric = {"second_word": "object"}
          else:
              this_metric = {metric: metric.replace("_", " ")}
          metric_rename = {**metric_rename, **this_metric}  # combine dicts

      # convert to DataFrame
      df = pd.DataFrame.from_dict({(i,j): model_results[i][j]
                                        for i in model_results.keys()
                                        for j in model_results[i].keys()},
                                    orient='index')
      df = df.reset_index()
      index_names = {"level_0":"Model", "level_1":"Checkpoint"}
      df.rename(columns={**index_names, **metric_rename},
              inplace=True)
      df['Parameters'] = df.apply(lambda row: params[row['Model']], axis=1)

      df = df.melt(id_vars=['Model', 'Checkpoint', 'Parameters'],
                        value_vars=metric_rename.values(),
                        var_name='Metric', value_name='Accuracy')
      # Average over checkpoints for the same model and metric
      if args.max:
          df = df.groupby(['Model', 'Metric']).max().reset_index()
      else:
          df = df.groupby(['Model', 'Metric']).mean().reset_index()

      alpha = .6

      # Make scatterplot
      sns.set_palette("bright")
      if "passiv" in args.gold_filename:
        hue_order = ['sequence', 'object']
      else:
        hue_order = ['sequence', 'main aux']
      ax = sns.scatterplot(x="Parameters", y="Accuracy", hue="Metric", data=df, 
                           hue_order=hue_order)
      # Use log x-scale

      #handles, labels = ax.get_legend_handles_labels()
      if not args.move_legend and not args.move_legend_left:
        loc = "lower right"
      elif args.move_legend:
        loc = "center right"
      elif args.move_legend_left:
        loc = "center left"
      #plt.legend(handles[:2], labels[:2], loc=loc)
      plt.ylim([-0.05, 1.05])
      plt.xlim([min(df['Parameters']) - 0.05*max(df['Parameters']), 1.05 * max(df['Parameters'])])
      # label points on the plot
      if args.label_points:
          for x, y, metric, model in zip(df['Parameters'], df['Accuracy'], df['Metric'], df['Model']):
              color = 'blue' if metric == 'sequence' else 'orange'
              plt.text(x = x+(.015 * 10**8), y = y+.02, s = model, fontsize=8, color=color)

      # Automatically get title of graph
      if "passiv_en_nps/" in args.gold_filename:
        title = "English Passivization"
      elif "passiv_de_nps/" in args.gold_filename:
        title = "German Passivization"
      elif "have-havent_en" in args.gold_filename:
        title = "English Question Formation"
      elif "have-can_withquest_de" in args.gold_filename:
        title = "German Question Formation"
      # Zero-shot graph titles
      elif "passiv_en-de" in args.gold_filename:
        if "pp_o" in args.gold_filename:
          title = "Zero-shot German Passivization (PP on obj)"
        elif "pp_s" in args.gold_filename:
          title = "Zero-shot German Passivization (PP on subj)"
      elif "have-can_de" in args.gold_filename:
        if "rc_o" in args.gold_filename:
          title = "Zero-shot German Question Formation (RC on obj)"
        elif "rc_s" in args.gold_filename:
          title = "Zero-shot German Question Formation (RC on subj)"
      else:
        title = None
      if title is not None:
        plt.title(title)
    
      # Save figure
      if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
      plt.savefig(os.path.join(args.out_dir, basename + "." + metrics_str + ".scaling.pdf"),
              format='pdf', bbox_inches='tight')

if __name__ == '__main__':
  main()
