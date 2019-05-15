from __future__ import absolute_import, division, print_function

import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import tensorflow as tf

from concurrent import futures
from functools import partial
from itertools import product

from noisy_ml.data.crowdsourced import *
from noisy_ml.data.rte import *
from noisy_ml.data.wordsim import *
from noisy_ml.evaluation.metrics import *
from noisy_ml.models.learners import *
from noisy_ml.models.models import *
from noisy_ml.models.amsgrad import *


logger = logging.getLogger(__name__)


def plot_results():
    dataset = "age"
    working_dir = os.getcwd()
    results_dir = os.path.join(working_dir, os.pardir, "results")
    results_path = os.path.join(results_dir, "%s.csv" % dataset)

    results = pd.read_csv(results_path)

    # Accuracy Plot.
    fig, ax = plt.subplots()
    results_auc = results[results.metric == "accuracy"]
    for label, auc in results_auc.groupby("model"):
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
