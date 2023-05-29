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

"""
METRIC_COLORMAP = {
    "first_np": "#ff7f0e",
    "second_np": "#2ca02c",
    "second_np_no_pp": "#d62728"
}
"""

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


ABLATION_NAME_MAP = lambda x: x.upper()

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

      # if "," in args.checkpoint_dirs:
      #     checkpoint_dirs = args.checkpoint_dirs.split(",")
      # else:
      #     checkpoint_dirs = [args.checkpoint_dirs]
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
      ablation_type = {}
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
            # model_prefix = MODEL_NAME_MAP[model_name]
            # make FF labels more precise
            if second_feature == "ff1000":
                second_feature = "ff1024"
            elif second_feature == "ff2000":
                second_feature = "ff2048"
            model_ablation = ABLATION_NAME_MAP(second_feature)
            model_name = model_ablation
            ablation_type[model_name] = model_name[:2]
        else:
            model_ablation = "_None"
            model_name = f"T5-{model_size}"
            ablation_type[model_name] = "_None"


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
      df['Component'] = df.apply(lambda row: ablation_type[row['Model']], axis=1)

      df = df.melt(id_vars=['Model', 'Checkpoint', 'Parameters', 'Component'],
                        value_vars=metric_rename.values(),
                        var_name='Metric', value_name='Accuracy')
      print("Pre-grouping")
      print(df)
      # Average over checkpoints for the same model and metric
      if args.max:
          df = df.groupby(['Model', 'Metric']).max().reset_index()
      else:
          df = df.groupby(['Model', 'Component', 'Metric']).mean().reset_index()

      print("Post-grouping")
      print(df)

      alpha = .6

      # Make scatterplot
      # sns.set_palette("bright")
      # f, ax = plt.subplots(figsize=(7, 7))
      # ax.set(xscale="log", yscale="log")
      # if "pass iv" in args.gold_filename:
      #   hue_order = ['sequence', 'object noun']
      # else:
      #   hue_order = ['sequence', 'main aux']
      palette = ["#33A02C", "#E31A1C", "#CAB2D6", "#1F77B4", "#BCBD22", "#000000"]
      # palette = ["#33A02C", "#E31A1C", "#CAB2D6", "#000000"]
      # palette = ["#33A02C", "#E31A1C", "#000000"]
      ax = sns.scatterplot(x="Parameters", y="Accuracy", style="Component", hue="Component",
                           palette=sns.color_palette(palette, 6), s=50, data=df)
      # Use log x-scale

      # ax.set_xticklabels(ax.get_xticks(), rotation=35)

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
              color = 'black'
              y_offset = -0.035 if model in ("NL8", "DM256", "FF1000") else 0.0175
              plt.text(x = x+(.0175 * 10**8), y = y + y_offset, s = model, fontsize=12, color=color)

      # x = df['Parameters']
      # xmin_pow = math.floor(math.log10(min(x)))
      # xmax_pow = math.ceil(math.log10(max(x)))
      # x_ticks = [10**i for i in range(xmin_pow, xmax_pow + 1)]
      # ax.set_xticks(x_ticks)
      # ax.set_xticklabels([str(i) for i in x_ticks])

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
      plt.savefig(os.path.join(args.out_dir, basename + "." + metrics_str + "all_components_shapes.pdf"),
              format='pdf', bbox_inches='tight')

if __name__ == '__main__':
  main()
