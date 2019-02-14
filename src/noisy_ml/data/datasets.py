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

from collections import namedtuple

__author__ = "eaplatanios"

__all__ = ["TrainData", "Dataset"]

logger = logging.getLogger(__name__)


TrainData = namedtuple("TrainData", ["instances", "predictors", "labels", "values"])


class Dataset(object):
    def __init__(
        self,
        instances,
        predictors,
        labels,
        true_labels,
        predicted_labels,
        num_classes,
        instance_features=None,
        predictor_features=None,
        label_features=None,
    ):
        """Creates a new dataset.

    Args:
      instances: List of instances.
      predictors: List of predictors.
      labels: List of labels.
      true_labels: Nested dictionary data structure that
        contains the true label value per label and
        per instance.
      num_classes: List of the number of possible classes per label.
      predicted_labels: Nested dictionary data structure
        that contains a tuple of two lists (instance
        indices and predicted values) per label and
        per predictor.
      TODO: Add the rest of the arguments.
    """
        self.instances = instances
        self.predictors = predictors
        self.labels = labels
        self.num_classes = num_classes
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.instance_features = instance_features
        self.predictor_features = predictor_features
        self.label_features = label_features

    def instance_indices(self):
        return list(range(len(self.instances)))

    def predictor_indices(self):
        return list(range(len(self.predictors)))

    def label_indices(self):
        return list(range(len(self.labels)))

    def avg_labels_per_predictor(self):
        num_labels = 0
        for l in range(len(self.labels)):
            for p, indices_values in six.iteritems(self.predicted_labels[l]):
                num_labels += len(indices_values[0])
        return float(num_labels) / len(self.predictors)

    def avg_labels_per_item(self):
        num_labels = 0
        for l in range(len(self.labels)):
            for p, indices_values in six.iteritems(self.predicted_labels[l]):
                num_labels += len(indices_values[0])
        return float(num_labels) / len(self.instances)

    @staticmethod
    def join(datasets):
        i_indices = dict()
        p_indices = dict()
        l_indices = dict()
        instances = list()
        predictors = list()
        labels = list()
        true_labels = dict()
        predicted_labels = dict()
        num_classes = list()

        has_if = True
        has_pf = True
        has_lf = True
        for d in datasets:
            has_if = has_if and d.instance_features is not None
            has_pf = has_pf and d.predictor_features is not None
            has_lf = has_lf and d.label_features is not None
        instance_features = list() if has_if else None
        predictor_features = list() if has_pf else None
        label_features = list() if has_lf else None

        for dataset in datasets:
            for l_old, label in enumerate(dataset.labels):
                l = l_indices.get(label)
                if l is None:
                    l = len(labels)
                    labels.append(label)
                    l_indices[label] = l
                    if has_lf:
                        label_features.append(dataset.label_features[l_old])

                # Process the true labels.
                if l not in true_labels:
                    true_labels[l] = dict()
                for i_old, true_label in six.iteritems(dataset.true_labels[l_old]):
                    instance = dataset.instances[i_old]
                    i = i_indices.get(instance)
                    if i is None:
                        i = len(instances)
                        instances.append(instance)
                        i_indices[instance] = i
                        if has_if:
                            instance_features.append(dataset.instance_features[i_old])
                    true_labels[l][i] = true_label

                # Process the predicted labels.
                if l not in predicted_labels:
                    predicted_labels[l] = dict()
                for p_old, indices_values in six.iteritems(
                    dataset.predicted_labels[l_old]
                ):
                    predictor = dataset.predictors[p_old]
                    p = p_indices.get(predictor)
                    if p is None:
                        p = len(predictors)
                        predictors.append(predictor)
                        p_indices[predictor] = p
                        if has_pf:
                            predictor_features.append(dataset.predictor_features[p_old])
                    if p not in predicted_labels[l]:
                        predicted_labels[l][p] = ([], [])
                    for i_old, v in zip(*indices_values):
                        instance = dataset.instances[i_old]
                        i = i_indices.get(instance)
                        if i is None:
                            i = len(instances)
                            instances.append(instance)
                            i_indices[instance] = i
                            if has_if:
                                instance_features.append(
                                    dataset.instance_features[i_old]
                                )
                        predicted_labels[l][p][0].append(i)
                        predicted_labels[l][p][1].append(v)

                # Process num_classes.
                num_classes += dataset.num_classes

        return Dataset(
            instances,
            predictors,
            labels,
            true_labels,
            predicted_labels,
            num_classes,
            instance_features,
            predictor_features,
            label_features,
        )

    def filter_labels(self, labels):
        i_indices = dict()
        p_indices = dict()
        instances = list()
        predictors = list()
        true_labels = dict()
        predicted_labels = dict()

        has_if = self.instance_features is not None
        has_pf = self.predictor_features is not None
        has_lf = self.label_features is not None
        instance_features = list() if has_if else None
        predictor_features = list() if has_pf else None
        label_features = list() if has_lf else None

        for l, label in enumerate(labels):
            l_old = self.labels.index(label)
            if has_lf:
                label_features.append(self.label_features[l_old])

            # Filter the true labels.
            true_labels[l] = dict()
            for i_old, true_label in six.iteritems(self.true_labels[label]):
                i = i_indices.get(i_old)
                if i is None:
                    i = len(instances)
                    instances.append(self.instances[i_old])
                    i_indices[i_old] = i
                    if has_if:
                        instance_features.append(self.instance_features[i_old])
                true_labels[l][i] = true_label

            # Filter the predicted labels.
            predicted_labels[l] = dict()
            for p_old, indices_values in six.iteritems(self.predicted_labels[l_old]):
                p = p_indices.get(p_old)
                if p is None:
                    p = len(predictors)
                    predictors.append(self.predictors[p_old])
                    p_indices[p_old] = p
                    if has_pf:
                        predictor_features.append(self.predictor_features[p_old])
                predicted_labels[l][p] = ([], [])
                for i_old, v in zip(*indices_values):
                    i = i_indices.get(i_old)
                    if i is None:
                        i = len(instances)
                        instances.append(self.instances[i_old])
                        i_indices[i_old] = i
                        if has_if:
                            instance_features.append(self.instance_features[i_old])
                    predicted_labels[l][p][0].append(i)
                    predicted_labels[l][p][1].append(v)

        return Dataset(
            instances,
            predictors,
            labels,
            true_labels,
            predicted_labels,
            instance_features,
            predictor_features,
            label_features,
        )

    def filter_predictors(self, predictors, keep_instances=True):
        i_indices = dict()
        p_indices = dict()
        new_instances = list()
        new_predictors = list()
        true_labels = dict()
        predicted_labels = dict()

        has_if = self.instance_features is not None
        has_pf = self.predictor_features is not None
        has_lf = self.label_features is not None
        instance_features = list() if has_if else None
        predictor_features = list() if has_pf else None
        label_features = list() if has_lf else None

        if keep_instances:
            new_instances = self.instances
            instance_features = self.instance_features

        for l, label in enumerate(self.labels):
            if has_lf:
                label_features.append(self.label_features[l])

            # Filter the predicted labels.
            predicted_labels[l] = dict()
            for p_old, indices_values in six.iteritems(self.predicted_labels[l]):
                predictor = self.predictors[p_old]
                if predictor not in predictors:
                    continue
                p = p_indices.get(p_old)
                if p is None:
                    p = len(new_predictors)
                    new_predictors.append(self.predictors[p_old])
                    p_indices[p_old] = p
                    if has_pf:
                        predictor_features.append(self.predictor_features[p_old])
                predicted_labels[l][p] = ([], [])
                for i_old, v in zip(*indices_values):
                    if keep_instances:
                        i = i_old
                    else:
                        i = i_indices.get(i_old)
                        if i is None:
                            i = len(new_instances)
                            new_instances.append(self.instances[i_old])
                            i_indices[i_old] = i
                            if has_if:
                                instance_features.append(self.instance_features[i_old])
                    predicted_labels[l][p][0].append(i)
                    predicted_labels[l][p][1].append(v)

            # Filter the true labels.
            true_labels[l] = dict()
            for i_old, true_label in six.iteritems(self.true_labels[label]):
                if keep_instances:
                    i = i_old
                else:
                    i = i_indices.get(i_old)
                    if i is None:
                        continue
                true_labels[l][i] = true_label

        return Dataset(
            new_instances,
            new_predictors,
            self.labels,
            true_labels,
            predicted_labels,
            instance_features,
            predictor_features,
            label_features,
        )

    def to_train(self, shuffle=False):
        instances = list()
        predictors = list()
        labels = list()
        values = list()
        for l, predicted_values in six.iteritems(self.predicted_labels):
            for p, (indices, l_values) in six.iteritems(predicted_values):
                instances.extend(indices)
                predictors.extend([p for _ in indices])
                labels.extend([l for _ in indices])
                values.extend(l_values)
        instances = np.array(instances, np.int32)
        predictors = np.array(predictors, np.int32)
        labels = np.array(labels, np.int32)
        values = np.array(values, np.float32)
        if shuffle:
            data = np.stack([instances, predictors, labels, values], axis=-1)
            np.random.shuffle(data)
            instances, predictors, labels, values = [
                data[..., i] for i in np.arange(data.shape[-1])
            ]
            instances = instances.astype(np.int32)
            predictors = predictors.astype(np.int32)
            labels = labels.astype(np.int32)
            values = values.astype(np.float32)
        return TrainData(instances, predictors, labels, values)

    def compute_binary_qualities(self):
        num_correct = list()
        num_total = list()
        for l, label in enumerate(self.labels):
            num_correct.append(list())
            num_total.append(list())
            true_labels = self.true_labels[l]
            for p, indices_values in six.iteritems(self.predicted_labels[l]):
                num_correct[l].append(0)
                num_total[l].append(0)
                for i, v in zip(*indices_values):
                    if true_labels[i] == (v >= 0.5):
                        num_correct[l][p] += 1
                    num_total[l][p] += 1
        num_correct = np.array(num_correct, np.float32)
        num_total = np.array(num_total, np.float32)
        return num_correct / num_total

    def compute_binary_qualities_full_confusion(self):
        num_confused = list()
        num_total = list()
        for l, label in enumerate(self.labels):
            num_confused.append(list())
            num_total.append(list())
            true_labels = self.true_labels[l]
            for p, indices_values in six.iteritems(self.predicted_labels[l]):
                num_confused[l].append(np.zeros([2, 2], dtype=np.float32))
                num_total[l].append(np.zeros([1, 1], dtype=np.float32))
                for i, v in zip(*indices_values):
                    if true_labels[i] == 0:
                        if v >= 0.5:
                            num_confused[l][p][0, 1] += 1
                        else:
                            num_confused[l][p][0, 0] += 1
                    if true_labels[i] == 1:
                        if v >= 0.5:
                            num_confused[l][p][1, 1] += 1
                        else:
                            num_confused[l][p][1, 0] += 1
                    num_total[l][p] += 1
        num_confused = np.array(num_confused, np.float32)
        num_total = np.array(num_total, np.float32)
        return num_confused / num_total
