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
  return np.mean(np.abs(p - t))


def compute_mad_error(predicted_qualities, true_qualities):
  return np.mean(np.abs(predicted_qualities - true_qualities))


def compute_accuracy(predictions, true_labels):
  if len(predictions.shape) > 1:
    p = predictions.argmax(-1).astype(np.int32)
  else:
    p = (predictions >= 0.5).astype(np.int32)
  return np.mean(p == true_labels)


def compute_auc(predictions, true_labels):
  if len(predictions.shape) > 1:
    true_labels_one_hot = np.zeros(predictions.shape)
    true_labels_one_hot[np.arange(len(true_labels)), true_labels] = 1
    true_labels = true_labels_one_hot
  return metrics.average_precision_score(true_labels, predictions)


class Result(object):
  def __init__(
      self, mad_error_rank, mad_error,
      accuracy, auc):
    self.mad_error_rank = mad_error_rank
    self.mad_error = mad_error
    self.accuracy = accuracy
    self.auc = auc

  def __str__(self):
    return 'MAD Error Rank = %7.4f, MAD Error = %6.4f, Accuracy = %6.4f, AUC = %6.4f' % \
           (self.mad_error_rank, self.mad_error, self.accuracy, self.auc)

  def __repr__(self):
    return str(self)

  @staticmethod
  def merge(results):
    return Result(
      mad_error_rank=np.mean([r.mad_error_rank for r in results]),
      mad_error=np.mean([r.mad_error for r in results]),
      accuracy=np.mean([r.accuracy for r in results]),
      auc=np.mean([r.auc for r in results]))

  def log(self, prefix=None):
    message = str(self)
    if prefix is None:
      logger.info(message)
    else:
      logger.info('%s - %s' % (prefix, message))


class Evaluator(object):
  def __init__(self, dataset):
    self.dataset = dataset

  def evaluate_per_label(self, learner, batch_size=128):
    instances = self.dataset.instance_indices()
    predictors = self.dataset.predictor_indices()
    labels = self.dataset.label_indices()

    # List of <float32> [batch_size, num_classes_l] for each label l.
    predictions = learner.predict(instances, batch_size=batch_size)

    # predicted_qualities shape: [NumLabels, NumPredictors]
    # true_qualities shape:      [NumLabels, NumPredictors]
    predicted_qualities = np.mean(
      learner.qualities(
          instances, predictors, labels,
          batch_size=batch_size),
      axis=0)
    true_qualities = self.dataset.compute_binary_qualities()

    results = []
    for l in range(len(predictions)):
      l_instances = list(six.iterkeys(self.dataset.true_labels[l]))
      tl = [self.dataset.true_labels[l][i] for i in l_instances]
      p = predictions[l][l_instances]
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
      for i in range(len(self.dataset.instances)):
        if i not in all_predictions:
          all_predictions[i] = [0.5]
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
