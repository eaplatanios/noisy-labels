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
import six

__author__ = 'eaplatanios'

__all__ = ['compute_binary_qualities', 'compute_binary_qualities_legacy']

logger = logging.getLogger(__name__)


def compute_binary_qualities(data):
  true_labels = data['train_data']['true_labels']
  predictors = data['train_data']['predictors']
  values = data['train_data']['values']
  num_correct = {}
  num_total = {}
  for (prediction, predictor, true_label) in zip(values, predictors, true_labels):
    if true_label == (prediction >= 0.5):
      num_correct[predictor] = num_correct.get(predictor, 0) + 1
    num_total[predictor] = num_total.get(predictor, 0) + 1
  for predictor in six.iterkeys(num_total):
    num_correct[predictor] /= float(num_total[predictor])
  return num_correct


def compute_binary_qualities_legacy(data):
  true_labels = data['true_labels']
  predictors = data['predictors']
  predictor_values = data['predictor_values']
  num_correct = {}
  num_total = {}
  for i, (predictions, predictors) in enumerate(zip(predictor_values, predictors)):
    for prediction, predictor in zip(predictions, predictors):
      if true_labels[i] == prediction:
        num_correct[predictor] = num_correct.get(predictor, 0) + 1
      num_total[predictor] = num_total.get(predictor, 0) + 1
  for predictor in six.iterkeys(num_correct):
    num_correct[predictor] /= float(num_total[predictor])
  return num_correct
