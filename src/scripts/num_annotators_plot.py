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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__author__ = 'eaplatanios'

logger = logging.getLogger(__name__)


def run_experiment():
  dataset = 'rte'
  working_dir = os.getcwd()
  results_dir = os.path.join(working_dir, os.pardir, 'results')

  if not os.path.exists(results_dir):
    os.makedirs(results_dir)

  if dataset is 'bluebirds':
    results_path = os.path.join(results_dir, 'bluebirds.csv')
  elif dataset is 'rte':
    results_path = os.path.join(results_dir, 'rte.csv')
  elif dataset is 'wordsim':
    results_path = os.path.join(results_dir, 'wordsim.csv')
  else:
    raise NotImplementedError

  results = pd.read_csv(results_path)

  # Accuracy Plot.
  fig, ax = plt.subplots()
  results_auc = results[results.metric == 'accuracy']
  for label, auc in results_auc.groupby('model'):
    ax.plot(
      auc.num_predictors.astype(np.int32),
      auc.value_mean,
      label=label)
    ax.fill_between(
      auc.num_predictors.astype(np.int32),
      auc.value_mean - auc.value_std,
      auc.value_mean + auc.value_std,
      alpha=0.35)
  plt.legend()
  plt.show()

  # AUC Plot.
  fig, ax = plt.subplots()
  results_auc = results[results.metric == 'auc']
  for label, auc in results_auc.groupby('model'):
    ax.plot(
      auc.num_predictors.astype(np.int32),
      auc.value_mean,
      label=label)
    ax.fill_between(
      auc.num_predictors.astype(np.int32),
      auc.value_mean - auc.value_std,
      auc.value_mean + auc.value_std,
      alpha=0.35)
  plt.legend()
  plt.show()


if __name__ == '__main__':
  run_experiment()
