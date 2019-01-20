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
        - `'instances'`: List of instances.
        - `'predictors'`: List of lists of predictors
          (i.e., predictors for each instance).
        - `'predictor_values'`: List of lists of predictor
          values (i.e., integers output by each predictor).
        - `'true_labels'`: List of true labels
          for each instance.
        - `'classifiers'`: List containing the
          classifier names.
    """
    if small_version:
      working_dir = os.path.join(working_dir, 'nell_10')
    else:
      working_dir = os.path.join(working_dir, 'nell')
    filename = '{}.csv'.format(label)
    filename = os.path.join(working_dir, filename)
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
      'instances': instances,
      'predictors': predictors,
      'predictor_values': predictor_values,
      'predictor_values_soft': predictor_values_soft,
      'true_labels': true_labels,
      'classifiers': classifiers}


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
        - `'instances'`: List of instances.
        - `'predictors'`: List of lists of predictors
          (i.e., predictors for each instance).
        - `'predictor_values'`: List of lists of predictor
          values (i.e., integers output by each predictor).
        - `'true_labels'`: List of true labels
          for each instance.
        - `'classifiers'`: List containing the
          classifier names.
    """
    if small_version:
      working_dir = os.path.join(working_dir, 'brain_10')
    else:
      working_dir = os.path.join(working_dir, 'brain')
    filename = '{}.csv'.format(label)
    filename = os.path.join(working_dir, filename)
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
      'instances': instances,
      'predictors': predictors,
      'predictor_values': predictor_values,
      'predictor_values_soft': predictor_values_soft,
      'true_labels': true_labels,
      'classifiers': classifiers}
