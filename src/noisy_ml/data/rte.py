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
import numpy as np
import os
import pandas as pd

from bert_serving.client import BertClient
from nltk.corpus.reader.rte import RTECorpusReader

from .datasets import Dataset

__author__ = 'eaplatanios'

__all__ = ['convert_xml_features', 'RTELoader']

logger = logging.getLogger(__name__)


def convert_xml_features(data_dir):
  data_dir = os.path.join(data_dir, 'rte')
  features_path = os.path.join(data_dir, 'features.txt')
  reader = RTECorpusReader(data_dir, ['rte1_test.xml'])
  bc = BertClient(ip='128.2.204.114')

  with open(features_path, 'w') as f:
    for p in reader.pairs('rte1_test.xml'):
      features = bc.encode(['%s ||| %s' % (p.text, p.hyp)])[0]
      features = ' '.join(map(str, features.tolist()))
      f.write('%s\t%s\n' % (p.id, features))


class RTELoader(object):
  """PASCAL RTE Amazon Mechanical Turk dataset.

  Sources:
    - https://sites.google.com/site/nlpannotations/
    - https://www.kaggle.com/nltkdata/rte-corpus
  """

  @staticmethod
  def load(data_dir, load_features=True):
    # Load data.
    data_dir = os.path.join(data_dir, 'rte')
    df = pd.read_table(os.path.join(data_dir, 'original.tsv'))

    # Extract instances and predictors.
    instances = df['orig_id'].unique().astype(int).tolist()
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

    if load_features:
      f_file = os.path.join(data_dir, 'features.txt')
      features = dict()
      with open(f_file, 'r') as f:
        for line in f:
          line_parts = line.split('\t')
          line_id = int(line_parts[0])
          line_features = list(map(float, line_parts[1].split(' ')))
          line_features = np.array(line_features, np.float32)
          features[line_id] = line_features
      instance_features = [features[i] for i in instances]
    else:
      instance_features = None

    return Dataset(
      instances, predictors, labels,
      true_labels, predicted_labels,
      instance_features=instance_features)
