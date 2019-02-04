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

__author__ = 'eaplatanios'

__all__ = ['RTELoader']

logger = logging.getLogger(__name__)


class RTELoader(object):
  """PASCAL RTE Amazon Mechanical Turk dataset.

  Sources:
    - https://sites.google.com/site/nlpannotations/
    - https://www.kaggle.com/nltkdata/rte-corpus
  """

  @staticmethod
  def load(data_dir):
    # Load data.
    data_dir = os.path.join(data_dir, 'rte', 'original.tsv')
    df = pd.read_table(data_dir)

    # Extract instances and predictors.
    instances = df['orig_id'].unique().astype(str).tolist()
    predictors = df['!amt_worker_ids'].unique().astype(str).tolist()

    # Extract ground truth.
    labels = [0]
    true_labels = df[['orig_id', 'gold']].drop_duplicates()
    true_labels = true_labels.drop_duplicates().set_index('orig_id')
    true_labels = true_labels.sort_index().values.flatten().tolist()
    true_labels = {0: dict(zip(range(len(true_labels)), true_labels))}

    # Extract annotations.
    annotations = df.drop_duplicates(
      subset=['orig_id', '!amt_worker_ids']
    ).pivot(
      index='orig_id',
      columns='!amt_worker_ids',
      values='response')
    annotations = annotations.fillna(-1).values
    predicted_labels = {0: dict()}
    for w_id in range(annotations.shape[1]):
      i_ids = np.nonzero(annotations[:, w_id] >= 0)[0]
      w_ans = annotations[i_ids, w_id]
      predicted_labels[0][w_id] = (i_ids.tolist(), w_ans.tolist())

    return Dataset(
      instances, predictors, labels,
      true_labels, predicted_labels)
