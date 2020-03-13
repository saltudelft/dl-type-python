"""
A set of utility methods for processing and analyzing results of experiments
"""

from os import listdir, remove, walk
from os.path import isfile, join, splitext
from gh_query import load_json
from statistics import mean
from shutil import copytree
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_result(result_dict: dict, title: str):

    labels = [k for k in result_dict.keys()]
    recall = [result_dict[l]['recall'] * 100 for l in labels]    
    precision = [result_dict[l]['precision'] * 100 for l in labels]
    f1 = [result_dict[l]['f1-score'] * 100 for l in labels]

    x = np.arange(len(labels))
    width = 0.2 # the width of the bars

    fig, ax = plt.subplots()
    recall_bar = ax.bar(x - width - 0.125, recall, width, label="Recall")
    precision_bar = ax.bar(x, precision, width, label="Precision")
    f1_bar = ax.bar(x + width + 0.125, f1, width, label="F1-score")

    ax.set_ylabel("Percent")
    ax.set_ylim([0, 100])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    autolabel(recall_bar, ax)
    autolabel(precision_bar, ax)
    autolabel(f1_bar, ax)

    fig.tight_layout()
    plt.show()

    print(labels, recall, precision, f1)

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize="small")


def eval_result(result_path, model_name, dataset='combined', filter="unfiltered", macro_avg=True):

    res_files = [f for f in listdir(result_path) if isfile(join(result_path, f))]
    res_files = [f for f in res_files if model_name in f and dataset == splitext(f)[0].split("_")[1] and filter == splitext(f)[0].split("_")[4]]
    #top_n = {"Top-%s" % r.split("_")[4]:[] for r in res_files}
    top_n = {}
    avg_type = 'macro avg' if macro_avg else 'weighted avg' 

    for r in res_files:
        res = load_json(join(result_path, r))[avg_type]
        if "Top-%s" % r.split("_")[3] in top_n:
            top_n["Top-%s" % r.split("_")[3]]['recall'].append(res['recall'])
            top_n["Top-%s" % r.split("_")[3]]['precision'].append(res['precision'])
            top_n["Top-%s" % r.split("_")[3]]['f1-score'].append(res['f1-score'])
        
        else:
            top_n["Top-%s" % r.split("_")[3]] = {}
            top_n["Top-%s" % r.split("_")[3]]['recall'] = []
            top_n["Top-%s" % r.split("_")[3]]['precision'] = []
            top_n["Top-%s" % r.split("_")[3]]['f1-score'] = []
            top_n["Top-%s" % r.split("_")[3]]['recall'].append(res['recall'])
            top_n["Top-%s" % r.split("_")[3]]['precision'].append(res['precision'])
            top_n["Top-%s" % r.split("_")[3]]['f1-score'].append(res['f1-score'])

    for i in top_n:
        top_n[i]['recall'] = mean(top_n[i]['recall'])
        top_n[i]['precision'] = mean(top_n[i]['precision'])
        top_n[i]['f1-score'] = mean(top_n[i]['f1-score'])

    return top_n


def copy_results(src_path, dest_path):
    """
    It copies the results to another path for cleaning output folder
    """

    copytree(src_path, join(dest_path, "report-%s" % date.today().strftime("%d-%m-%y")))


def clean_output(output_path):
    """
    Delete all the files in output folder.
    """

    for root, dirs, files in walk(output_path):
        for f in files:
            try:
                remove(join(root, f))
                print("Deleted file %s" % join(root, f))
            except PermissionError:
                print("Could not delete file %s" % join(root, f))


def plot_top_n_types(types_file, top_n=50):
    """
    Plots top n most frequent types in the dataset
    """

    # Already sorted
    df = pd.read_csv(types_file)

    types = df['type'].values[:top_n]
    counts = df['count'].values[:top_n]
    y_pos = np.arange(len(counts))

    plt.barh(y_pos, counts, align='center', alpha=0.5)
    plt.yticks(y_pos, types)
    plt.xlabel('Counts')
    plt.ylabel('Types')
    plt.title('Top %d most frequent types in the dataset' % top_n)

    plt.show()
