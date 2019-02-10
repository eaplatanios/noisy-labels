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

__all__ = [
  'extract_bert_features', 'extract_glove_features',
  'WordSimLoader']

logger = logging.getLogger(__name__)


def extract_bert_features(data_dir):
  dataset = WordSimLoader.load(data_dir, load_features=False)
  data_dir = os.path.join(data_dir, 'wordsim')
  features_path = os.path.join(data_dir, 'bert_features.txt')
  bc = BertClient(ip='128.2.204.114')

  with open(features_path, 'w') as f:
    for pair in dataset.instances:
      features = bc.encode(pair.split(' '))
      features = np.reshape(features, [-1])
      features = ' '.join(map(str, features.tolist()))
      f.write('%s\t%s\n' % (pair, features))


def extract_glove_features(data_dir):
  dataset = WordSimLoader.load(data_dir, load_features=False)
  data_dir = os.path.join(data_dir, 'wordsim')
  glove_path = os.path.join(data_dir, 'glove.6B.100d.txt')
  features_path = os.path.join(data_dir, 'glove_features.txt')

  embeddings = {}
  with open(glove_path, 'r') as f:
    for line in f:
      values = line.split()
      word = values[0]
      embeddings[word] = np.asarray(
        values[1:], dtype='float32')

  with open(features_path, 'w') as f:
    for pair in dataset.instances:
      words = pair.split(' ')
      features = [embeddings[words[0]], embeddings[words[1]]]
      features = np.reshape(features, [-1])
      features = ' '.join(map(str, features.tolist()))
      f.write('%s\t%s\n' % (pair, features))


class WordSimLoader(object):
  """Word similarity Amazon Mechanical Turk dataset.

  Source: https://sites.google.com/site/nlpannotations/
  """

  @staticmethod
  def load(data_dir, load_features=True):
    # Load data.
    data_dir = os.path.join(data_dir, 'wordsim')
    df = pd.read_table(os.path.join(data_dir, 'original.tsv'))

    # Extract instances and predictors.
    instances = df['orig_id'].unique().astype(str).tolist()
    predictors = df['!amt_worker_ids'].unique().astype(str).tolist()

    # Extract ground truth.
    labels = [0]
    true_labels = df[['orig_id', 'gold']].drop_duplicates()
    true_labels = true_labels.drop_duplicates().set_index('orig_id')
    true_labels = true_labels.sort_index().values.flatten()
    true_labels = (true_labels >= 2.0).astype(np.int32)
    true_labels = true_labels.tolist()
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
      w_ans = annotations[i_ids, w_id] / 10.0
      predicted_labels[0][w_id] = (i_ids.tolist(), w_ans.tolist())

    if load_features:
      f_file = os.path.join(data_dir, 'glove_features.txt')
      features = dict()
      with open(f_file, 'r') as f:
        for line in f:
          line_parts = line.split('\t')
          line_id = line_parts[0]
          line_features = list(map(float, line_parts[1].split(' ')))
          line_features = np.array(line_features, np.float32)
          features[line_id] = line_features
      instance_features = [features[i] for i in instances]
    else:
      instance_features = None


    # Single label with 2 classes.
    num_classes = [2]

    return Dataset(
      instances, predictors, labels,
      true_labels, predicted_labels,
      num_classes=num_classes,
      instance_features=instance_features)
