from __future__ import absolute_import, division, print_function

import os

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


def plot_results(labels=None):
    matplotlib.rc("text", usetex=False)

    dataset = "wordsim"
    working_dir = os.getcwd()
    results_dir = os.path.join(working_dir, "results/debug")
    results_path = os.path.join(results_dir, "%s.csv" % dataset)

    results = pd.read_csv(results_path)

    # Accuracy Plot.
    fig, ax = plt.subplots()
    results_auc = results[results.metric == "accuracy"]
    for label, auc in results_auc.groupby("model"):
        if labels is not None and label not in labels:
            continue
        ax.plot(
            auc.num_predictors.astype(np.int32), auc.value_mean, label=label
        )
        ax.fill_between(
            auc.num_predictors.astype(np.int32),
            auc.value_mean - auc.value_std,
            auc.value_mean + auc.value_std,
            alpha=0.35,
        )
    plt.legend()
    plt.savefig(os.path.join(results_dir, "%s_acc.pdf" % dataset))

    # AUC Plot.
    fig, ax = plt.subplots()
    results_auc = results[results.metric == "auc"]
    for label, auc in results_auc.groupby("model"):
        if labels is not None and label not in labels:
            continue
        ax.plot(
            auc.num_predictors.astype(np.int32), auc.value_mean, label=label
        )
        ax.fill_between(
            auc.num_predictors.astype(np.int32),
            auc.value_mean - auc.value_std,
            auc.value_mean + auc.value_std,
            alpha=0.35,
        )
    plt.legend()
    plt.savefig(os.path.join(results_dir, "%s_auc.pdf" % dataset))


if __name__ == "__main__":
    plot_results()
