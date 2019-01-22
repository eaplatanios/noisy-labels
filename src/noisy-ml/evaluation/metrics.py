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

from sklearn import metrics

__author__ = 'eaplatanios'

__all__ = [
  'compute_mad_error_rank', 'compute_mad_error',
  'compute_auc', 'compute_accuracy']

logger = logging.getLogger(__name__)


def compute_mad_error_rank(predicted_qualities, true_qualities):
  p = np.argsort(predicted_qualities, axis=-1)
  t = np.argsort(true_qualities, axis=-1)
  return np.mean(np.abs(p - t))


def compute_mad_error(predicted_qualities, true_qualities):
  return np.mean(np.abs(predicted_qualities - true_qualities))


def compute_auc(predictions, true_labels):
  return metrics.average_precision_score(true_labels, predictions)


def compute_accuracy(predictions, true_labels):
  p = (predictions >= 0.5).astype(np.int32)
  return np.mean(p == true_labels)
