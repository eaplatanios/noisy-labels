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

import numpy as np

__author__ = 'eaplatanios'

__all__ = ['SyntheticBinaryGenerator']


class GeneratedDataset(object):
  def __init__(
      self, instances, predictors, predictions,
      true_labels, true_qualities):
    self.instances = instances
    self.predictors = predictors
    self.predictions = predictions
    self.true_labels = true_labels
    self.true_qualities = true_qualities


class SyntheticBinaryGenerator(object):
  def generate(self, num_instances, predictor_qualities):
    instances = list(range(num_instances))
    predictors = list(range(len(predictor_qualities)))
    true_labels = np.random.uniform(0, 1, num_instances)
    true_labels = (true_labels >= 0.5).astype(np.int32)
    true_qualities = predictor_qualities
    prediction_instances = []
    prediction_predictors = []
    prediction_predictor_values = []
    for i in instances:
      mistakes = np.random.uniform(0, 1, len(predictors))
      mistakes = mistakes > true_qualities
      mistakes = mistakes.astype(np.int32)
      true_label = np.tile(true_labels[i], len(predictors))
      predictor_values = true_label * (1 - mistakes) + (1 - true_label) * mistakes
      predictor_indices = list(range(len(predictors)))
      prediction_instances.append(i)
      prediction_predictors.append(predictor_indices)
      prediction_predictor_values.append(predictor_values)
    predictions = {
      'instances': prediction_instances,
      'predictors': prediction_predictors,
      'predictor_values': prediction_predictor_values}
    return GeneratedDataset(
      instances, predictors, predictions, true_labels, true_qualities)
