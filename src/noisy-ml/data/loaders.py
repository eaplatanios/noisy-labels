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

__author__ = 'eaplatanios'

__all__ = ['NELLLoader', 'BrainLoader']

logger = logging.getLogger(__name__)


class NELLLoader(object):
  @staticmethod
  def load_binary(working_dir, label, small_version=False):
    """Loads the NELL data for a single label.

    Args:
      working_dir: Working directory (e.g., for downloading
        and extracting data files).
      label: Label for which to load data.
      small_version: Boolean flag indicating whether to use
        the small version of the NELL dataset. This is 10%
        of the complete dataset.

    Returns:
      dict: Loaded data as a dictionary that contains:
        - `'instance_names'`: Instance names.
        - `'instances'`: Numpy array with shape [N]
          which contains the instance indices.
        - `'predictors'`: Numpy array with shape [N, M]
          which contains the predictor indices per instance.
        - `'predictor_values'`: Numpy array with shape
          [N, M, 1] which contains the predicted values
          (i.e., integers output by each predictor).
        - `'predictor_values_soft'`: Numpy array with shape
          [N, M, 1] which contains the predicted values
          (i.e., probabilities output by each predictor).
        - `'true_labels'`: List of true labels
          for each instance.
        - `'classifiers'`: List containing the
          classifier names.
    """
    data_dir = os.path.join(working_dir, 'nell', 'unconstrained')

    # Load instance names.
    filename = '{}.txt'.format(label)
    filename = os.path.join(data_dir, 'names', filename)
    with open(filename, 'r') as f:
      instance_names = [line for line in f]

    # Load predictions.
    if small_version:
      data_dir = os.path.join(data_dir, 'small')
    else:
      data_dir = os.path.join(data_dir, 'full')
    filename = '{}.csv'.format(label)
    filename = os.path.join(data_dir, filename)
    num_instances = 0
    predictor_names = []
    instances = []
    predictors = []
    true_labels = []
    train_instances = []
    train_predictors = []
    train_values = []
    train_true_labels = []
    is_header = True
    with open(filename, 'r') as f:
      for i, line in enumerate(f):
        if is_header:
          predictor_names = line.split(',')[1:]
          predictors = [j for j in range(len(predictor_names))]
          is_header = False
        else:
          line_parts = line.split(',')
          num_instances += 1
          instances.append(i - 1)
          true_labels.append(int(line_parts[0]))
          train_instances.extend([i - 1 for _ in predictors])
          train_predictors.extend(predictors)
          train_values.extend([float(v) for v in line_parts[1:]])
          train_true_labels.extend([int(line_parts[0]) for _ in predictors])
    instances = np.array(instances, dtype=np.int32)
    predictors = np.array(predictors, dtype=np.int32)
    labels = np.array([0], dtype=np.int32)
    true_labels = np.array(true_labels, dtype=np.int32)
    train_instances = np.array(train_instances, dtype=np.int32)
    train_predictors = np.array(train_predictors, dtype=np.int32)
    train_labels = np.zeros_like(train_predictors)
    train_values = np.array(train_values, dtype=np.float32)
    return {
      'num_instances': num_instances,
      'num_predictors': len(predictor_names),
      'num_labels': 1,
      'instance_names': instance_names,
      'predictor_names': predictor_names,
      'instances': instances,
      'predictors': predictors,
      'labels': labels,
      'true_labels': true_labels,
      'train_data': {
        'instances': train_instances,
        'predictors': train_predictors,
        'labels': train_labels,
        'values': train_values,
        'true_labels': train_true_labels
      }}

  @staticmethod
  def train_data_as_2d(data):
    shape_2d = [data['num_instances'], data['num_predictors']]
    return {
      'values': np.reshape(data['train_data']['values'], shape_2d),
      'true_labels': np.reshape(data['train_data']['true_labels'], shape_2d)}

  # @staticmethod
  # def load_multi_label(working_dir, labels, small_version=False):
  #   per_label = [
  #     NELLLoader.load_binary(working_dir, l, small_version)
  #     for l in labels]
  #   return {
  #     'instance_names': per_label[0]['instance_names'],
  #     'instances': np.stack([d['instances'] for d in per_label], axis=1),
  #     'predictors': np.array(predictors, dtype=np.int32),
  #     'predictor_values': np.array(predictor_values, dtype=np.int32)[:, :, None],
  #     'predictor_values_soft': np.array(predictor_values_soft, dtype=np.float32)[:, :, None],
  #     'true_labels': true_labels,
  #     'classifiers': classifiers}


class BrainLoader(object):
  @staticmethod
  def load_binary(working_dir, label, small_version=False):
    """Loads the Brain data for a single label.

    Args:
      working_dir: Working directory (e.g., for downloading
        and extracting data files).
      label: Label for which to load data.
      small_version: Boolean flag indicating whether to use
        the small version of the NELL dataset. This is 10%
        of the complete dataset.

    Returns:
      dict: Loaded data as a dictionary that contains:
        - `'instances'`: Numpy array with shape [N]
          which contains the instance indices.
        - `'predictors'`: Numpy array with shape [N, M]
          which contains the predictor indices per instance.
        - `'predictor_values'`: Numpy array with shape
          [N, M, 1] which contains the predicted values
          (i.e., integers output by each predictor).
        - `'predictor_values_soft'`: Numpy array with shape
          [N, M, 1] which contains the predicted values
          (i.e., probabilities output by each predictor).
        - `'true_labels'`: List of true labels
          for each instance.
        - `'classifiers'`: List containing the
          classifier names.
    """
    data_dir = os.path.join(working_dir, 'brain')
    if small_version:
      data_dir = os.path.join(data_dir, 'small')
    else:
      data_dir = os.path.join(data_dir, 'full')
    filename = '{}.csv'.format(label)
    filename = os.path.join(data_dir, filename)
    classifiers = []
    instances = []
    predictors = []
    predictor_values = []
    predictor_values_soft = []
    true_labels = []
    is_header = True
    with open(filename, 'r') as f:
      for i, line in enumerate(f):
        if is_header:
          classifiers = line.split(',')[1:]
          is_header = False
        else:
          values = line.split(',')
          instances.append(i - 1)
          predictors.append([j for j in range(len(classifiers))])
          predictor_values.append(
            [int(float(v) >= 0.5) for v in values[1:]])
          predictor_values_soft.append(
            [float(v) for v in values[1:]])
          true_labels.append(int(values[0]))
    return {
      'instances': np.array(instances, dtype=np.int32),
      'predictors': np.array(predictors, dtype=np.int32),
      'predictor_values': np.array(predictor_values, dtype=np.int32)[:, :, None],
      'predictor_values_soft': np.array(predictor_values_soft, dtype=np.float32)[:, :, None],
      'true_labels': true_labels,
      'classifiers': classifiers}
