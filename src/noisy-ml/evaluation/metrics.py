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
import six

from sklearn import metrics

__author__ = 'eaplatanios'

__all__ = [
  'compute_mad_error_rank', 'compute_mad_error',
  'compute_accuracy', 'compute_auc',
  'Result', 'Evaluator']

logger = logging.getLogger(__name__)


def compute_mad_error_rank(predicted_qualities, true_qualities):
  p = np.argsort(predicted_qualities, axis=-1)
  t = np.argsort(true_qualities, axis=-1)
  return np.abs(p - t)


def compute_mad_error(predicted_qualities, true_qualities):
  return np.mean(np.abs(predicted_qualities - true_qualities))


def compute_accuracy(predictions, true_labels):
  p = (predictions >= 0.5).astype(np.int32)
  return np.mean(p == true_labels)


def compute_auc(predictions, true_labels):
  return metrics.average_precision_score(true_labels, predictions)


class Result(object):
  def __init__(
      self, mad_error_rank, mad_error,
      accuracy, auc):
    self.mad_error_rank = mad_error_rank
    self.mad_error = mad_error
    self.accuracy = accuracy
    self.auc = auc

  @staticmethod
  def merge(results):
    return Result(
      mad_error_rank=np.mean([r.mad_error_rank for r in results]),
      mad_error=np.mean([r.mad_error for r in results]),
      accuracy=np.mean([r.accuracy for r in results]),
      auc=np.mean([r.auc for r in results]))

  def log(self, prefix=None):
    message = 'MAD Error Rank: %.4f | MAD Error: %.4f | Accuracy: %.4f | AUC: %.4f' % \
              (self.mad_error_rank, self.mad_error, self.accuracy, self.auc)
    if prefix is None:
      logger.info(message)
    else:
      logger.info('%s - %s' % (prefix, message))


class Evaluator(object):
  def __init__(self, learner, dataset):
    self.learner = learner
    self.dataset = dataset

  def evaluate_per_label(self):
    instances = self.dataset.instance_indices()
    predictors = self.dataset.predictor_indices()
    labels = self.dataset.label_indices()

    # predictions shape:         [NumLabels, BatchSize]
    # predicted_qualities shape: [NumLabels, NumPredictors]
    # true_qualities shape:      [NumLabels, NumPredictors]
    predictions = self.learner.predict(instances).T
    predicted_qualities = np.mean(self.learner.qualities(
      instances, predictors, labels), axis=0)
    true_qualities = self.dataset.compute_binary_qualities()

    results = []
    for l in range(predictions.shape[0]):
      l_instances = list(six.iterkeys(self.dataset.true_labels[l]))
      tl = [self.dataset.true_labels[l][i] for i in l_instances]
      p = predictions[l, l_instances]
      pq = predicted_qualities[l]
      tq = true_qualities[l]
      results.append(Result(
        mad_error_rank=compute_mad_error_rank(pq, tq),
        mad_error=compute_mad_error(pq, tq),
        accuracy=compute_accuracy(p, tl),
        auc=compute_auc(p, tl)))

    return results

  def evaluate_maj_per_label(self, soft=False):
    labels = self.dataset.label_indices()
    true_qualities = self.dataset.compute_binary_qualities()

    results = []
    for l in range(len(labels)):
      all_predictions = dict()
      for p, indices_values in six.iteritems(self.dataset.predicted_labels[l]):
        for i, v in zip(*indices_values):
          if i not in all_predictions:
            all_predictions[i] = list()
          if not soft:
            v = int(v >= 0.5)
          all_predictions[i].append(v)
      all_predictions_mean = dict()
      true_labels = []
      predictions = []
      for i, values in six.iteritems(all_predictions):
        values_mean = np.mean(values)
        all_predictions_mean[i] = values_mean
        true_labels.append(self.dataset.true_labels[l][i])
        predictions.append(values_mean)

      predicted_qualities = dict()
      for p, indices_values in six.iteritems(self.dataset.predicted_labels[l]):
        if p not in predicted_qualities:
          predicted_qualities[p] = []
        for i, v in zip(*indices_values):
          v = int(v >= 0.5)
          maj = int(all_predictions_mean[i] >= 0.5)
          c = int(v == maj)
          predicted_qualities[p].append(c)
        predicted_qualities[p] = np.mean(predicted_qualities[p])

      tl = np.array(true_labels, np.int32)
      p = np.array(predictions, np.float32)
      pq = np.array(list(map(
        lambda kv: kv[1],
        sorted(six.iteritems(
          predicted_qualities),
          key=lambda kv: kv[0]))))
      tq = true_qualities[l]
      results.append(Result(
        mad_error_rank=compute_mad_error_rank(pq, tq),
        mad_error=compute_mad_error(pq, tq),
        accuracy=compute_accuracy(p, tl),
        auc=compute_auc(p, tl)))

    return results
