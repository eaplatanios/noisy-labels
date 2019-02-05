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

import json
import logging
import numpy as np
import os
import pandas as pd
import subprocess

from nltk.corpus.reader.rte import RTECorpusReader

from .datasets import Dataset

__author__ = 'eaplatanios'

__all__ = ['convert_xml_features', 'RTELoader']

logger = logging.getLogger(__name__)


def convert_xml_features(dataset, data_dir):
  data_dir = os.path.join(data_dir, 'rte')
  sentences_path = os.path.join(data_dir, 'sentences.txt')
  features_path = os.path.join(data_dir, 'features.json')
  bert_dir = os.path.join(data_dir, 'bert')
  bert_ckpt_dir = os.path.join(
    bert_dir, 'checkpoints', 'cased_L-24_H-1024_A-16')

  reader = RTECorpusReader(data_dir, ['rte1_test.xml'])
  pairs = {p.id: '%s ||| %s' % (p.text, p.hyp)
           for p in reader.pairs('rte1_test.xml')}
  pairs = [pairs[i] for i in dataset.instances]

  with open(sentences_path, 'w') as f:
    for pair in pairs:
      f.write(pair + '\n')

  bert_env = os.environ.copy()
  bert_env['BERT_BASE_DIR'] = bert_ckpt_dir
  subprocess.run(
    'python3 extract_pooled_features.py '
    '--input_file=%s --output_file=%s '
    '--vocab_file=$BERT_BASE_DIR/vocab.txt '
    '--bert_config_file=$BERT_BASE_DIR/bert_config.json '
    '--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt '
    '--max_seq_length=512 '
    '--batch_size=1'
    % (sentences_path, features_path),
    shell=True, cwd=bert_dir, env=bert_env)


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
      f_file = os.path.join(data_dir, 'features.json')
      features = list()
      with open(f_file, 'r') as f:
        for line in f:
          line = json.loads(line)
          features.append(np.array(line['features'], np.float32))
      instance_features = features
    else:
      instance_features = None

    return Dataset(
      instances, predictors, labels,
      true_labels, predicted_labels,
      instance_features=instance_features)
