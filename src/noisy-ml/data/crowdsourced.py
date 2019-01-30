from __future__ import absolute_import, division, print_function

import logging
import os

import numpy as np
import pandas as pd

from .loaders import Dataset

__all__ = [
    "SentimentPopularityLoader",
]

logger = logging.getLogger(__name__)


class SentimentPopularityLoader(object):
  """Sentiment popularity AMT dataset.

  Source: https://eprints.soton.ac.uk/376544/1/SP_amt.csv
  """
  @staticmethod
  def load():
    # Load data.
    datapath = os.path.join(
        os.path.expandvars("$DATA_PATH"),
        "crowdsourced", "sentiment_popularity", "SP_amt.csv")
    column_names = [
        "WorkerID", "TaskID", "Label", "True label", "Judgement time"]
    df = pd.read_csv(datapath, names=column_names)

    # Get annotations
    annotations = df.pivot(index="TaskID", columns="WorkerID", values="Label")

    # Extract instances and predictors.
    instances = annotations.index.values.astype(str).tolist()
    predictors = annotations.columns.values.astype(str).tolist()

    # Extract ground truth.
    labels = [0]
    true_labels = df[["TaskID", "True label"]].drop_duplicates()
    true_labels = true_labels.drop_duplicates().set_index("TaskID")
    true_labels = true_labels.sort_index().values.flatten().tolist()
    true_labels = {0: dict(zip(range(len(true_labels)), true_labels))}

    # Extract annotations.
    annotations = annotations.fillna(-1).values
    predicted_labels = {0: dict()}
    for w_id in range(annotations.shape[1]):
        i_ids = np.nonzero(annotations[:, w_id] >= 0)[0]
        w_ans = annotations[i_ids, w_id]
        predicted_labels[0][w_id] = (i_ids.tolist(), w_ans.tolist())

    return Dataset(
        instances, predictors, labels,
        true_labels, predicted_labels)
