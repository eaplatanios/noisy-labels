# Copyright 2019, Emmanouil Antonios Platanios. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from __future__ import absolute_import, division, print_function

import logging
import os

import numpy as np
import pandas as pd

from .loaders import Dataset

__author__ = 'alshedivat'

__all__ = [
    'SentimentPopularityLoader']

logger = logging.getLogger(__name__)


class SentimentPopularityLoader(object):
  """Sentiment popularity AMT dataset.

  Source: https://eprints.soton.ac.uk/376544/1/SP_amt.csv
  """

  @staticmethod
  def load(data_dir):
    # Load data.
    datapath = os.path.join(
        data_dir,
        'crowdsourced', 'sentiment_popularity', 'SP_amt.csv')
    column_names = [
        'WorkerID', 'TaskID', 'Label', 'True label', 'Judgement time']
    df = pd.read_csv(datapath, names=column_names)

    # Get annotations
    annotations = df.pivot(index='TaskID', columns='WorkerID', values='Label')

    # Extract instances and predictors.
    instances = annotations.index.values.astype(str).tolist()
    predictors = annotations.columns.values.astype(str).tolist()

    # Extract ground truth.
    labels = [0]
    true_labels = df[['TaskID', 'True label']].drop_duplicates()
    true_labels = true_labels.drop_duplicates().set_index('TaskID')
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
