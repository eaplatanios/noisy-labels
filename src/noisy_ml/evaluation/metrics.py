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

from collections import defaultdict
from sklearn import metrics

__author__ = "eaplatanios"

__all__ = [
    "compute_mad_error_rank",
    "compute_mad_error",
    "compute_accuracy",
    "compute_auc",
    "Result",
    "Evaluator",
]

logger = logging.getLogger(__name__)


def compute_mad_error_rank(predicted_qualities, true_qualities):
    p = np.argsort(predicted_qualities, axis=-1)
    t = np.argsort(true_qualities, axis=-1)
    return np.mean(np.abs(p - t))


def compute_mad_error(predicted_qualities, true_qualities):
    return np.mean(np.abs(predicted_qualities - true_qualities))


def compute_accuracy(predictions, true_labels):
    if len(predictions.shape) > 1:
        p = (np.random.random(predictions.shape) * 1e-3 + predictions).argmax(axis=-1).astype(np.int32)
    else:
        p = (predictions >= 0.5 + (np.random.rand() - 0.5) * 1e-3).astype(np.int32)
    return metrics.accuracy_score(true_labels, p)


def compute_auc(predictions, true_labels):
    if len(predictions.shape) > 1:
        true_labels_one_hot = np.zeros(predictions.shape)
        true_labels_one_hot[np.arange(len(true_labels)), true_labels] = 1
        true_labels = true_labels_one_hot
    return metrics.average_precision_score(true_labels, predictions)


class Result(object):
    def __init__(self, accuracy, auc, mad_error=None, mad_error_rank=None):
        self.accuracy = accuracy
        self.auc = auc
        self.mad_error = mad_error
        self.mad_error_rank = mad_error_rank

    def __str__(self):
        return (
            "MAD Error Rank = %7.4f, MAD Error = %6.4f, Accuracy = %6.4f, AUC = %6.4f"
            % (self.mad_error_rank, self.mad_error, self.accuracy, self.auc)
        )

    def __repr__(self):
        return str(self)

    @staticmethod
    def merge(results):
        return Result(
            # mad_error_rank=np.mean([r.mad_error_rank for r in results]),
            # mad_error=np.mean([r.mad_error for r in results]),
            accuracy=np.mean([r.accuracy for r in results]),
            auc=np.mean([r.auc for r in results]),
        )

    def log(self, prefix=None):
        message = str(self)
        if prefix is None:
            logger.info(message)
        else:
            logger.info("%s - %s" % (prefix, message))


class Evaluator(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def evaluate_per_label(self, learner, batch_size=128):
        instances = self.dataset.instance_indices()

        # List of <float32> [batch_size, num_classes_l] for each label l.
        predictions = learner.predict(instances, batch_size=batch_size)

        results = []
        for l in range(len(predictions)):
            if l not in self.dataset.true_labels:
                continue
            l_instances = list(six.iterkeys(self.dataset.true_labels[l]))
            tl = [self.dataset.true_labels[l][i] for i in l_instances]
            p = predictions[l][l_instances]
            results.append(
                Result(
                    accuracy=compute_accuracy(p, tl),
                    auc=compute_auc(p, tl),
                )
            )

        return results

    def evaluate_maj_per_label(self, soft=False):
        labels = self.dataset.label_indices()

        results = []
        for l in range(len(labels)):
            all_predictions = dict()
            for p, indices_values in six.iteritems(
                self.dataset.predicted_labels[l]
            ):
                for i, v in zip(*indices_values):
                    if i not in all_predictions:
                        all_predictions[i] = list()
                    if not soft:
                        v = int(v >= 0.5)
                    all_predictions[i].append(v)
            for i in range(len(self.dataset.instances)):
                if i not in all_predictions:
                    all_predictions[i] = [0.5]

            true_labels = []
            predictions = []
            for i, values in six.iteritems(all_predictions):
                true_labels.append(self.dataset.true_labels[l][i])
                predictions.append(np.mean(values))

            tl = np.array(true_labels, np.int32)
            p = np.array(predictions, np.float32)
            results.append(
                Result(
                    accuracy=compute_accuracy(p, tl),
                    auc=compute_auc(p, tl),
                )
            )

        return results

    def evaluate_maj_multi_per_label(self, soft=False, eps=1e-10):
        """Evaluates MAJ for each multi-class label."""

        results = []
        for l_id, nc in enumerate(self.dataset.num_classes):
            if l_id not in self.dataset.true_labels:
                continue

            # Get a dict of all predictions for each instance.
            all_predictions = defaultdict(list)
            for indices_values in six.itervalues(
                self.dataset.predicted_labels[l_id]
            ):
                for i, v in zip(*indices_values):
                    if isinstance(v, (float, np.float32, np.float64)):
                        if not soft:
                            v = int(v >= 0.5)
                        v = np.asarray([1 - v, v])
                    elif isinstance(v, (int, np.int32, np.int64)):
                        v_vec = np.zeros(nc)
                        v_vec[v] = 1
                        v = v_vec
                    all_predictions[i].append(v)
            for i in range(len(self.dataset.instances)):
                if i not in all_predictions:
                    v_unif = np.ones(nc) / nc
                    all_predictions[i].append(v_unif)

            # Get predictions.
            gt_indices = []
            ground_truth = []
            predictions = []
            for i in self.dataset.instance_indices():
                values = all_predictions[i]
                predictions.append(np.mean(values, axis=0))
                assert np.isclose(predictions[-1].sum(), 1.)
                if i in self.dataset.true_labels[l_id]:
                    gt = self.dataset.true_labels[l_id][i]
                    gt_indices.append(i)
                    ground_truth.append(gt)

            gt = np.array(ground_truth, np.int32)
            p = np.array(predictions, np.float32)
            results.append(
                Result(
                    accuracy=compute_accuracy(p[gt_indices], gt),
                    auc=compute_auc(p[gt_indices], gt),
                )
            )

        return results
