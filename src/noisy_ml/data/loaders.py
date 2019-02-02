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
import random
import six

from collections import namedtuple

__author__ = 'eaplatanios'

__all__ = [
  'TrainData', 'Dataset', 'LegacyLoader', 'NELLLoader']

logger = logging.getLogger(__name__)


TrainData = namedtuple(
  'TrainData',
  ['instances', 'predictors', 'labels', 'values'])


class Dataset(object):
  def __init__(
      self, instances, predictors, labels,
      true_labels, predicted_labels,
      instance_features=None, predictor_features=None,
      label_features=None):
    """Creates a new dataset.

    Args:
      instances: List of instances.
      predictors: List of predictors.
      labels: List of labels.
      true_labels: Nested dictionary data structure that
        contains the true label value per label and
        per instance.
      predicted_labels: Nested dictionary data structure
        that contains a tuple of two lists (instance
        indices and predicted values) per label and
        per predictor.
      TODO: Add the rest of the arguments.
    """
    self.instances = instances
    self.predictors = predictors
    self.labels = labels
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
        for p_old, indices_values in six.iteritems(dataset.predicted_labels[l_old]):
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
                instance_features.append(dataset.instance_features[i_old])
            predicted_labels[l][p][0].append(i)
            predicted_labels[l][p][1].append(v)

    return Dataset(
      instances, predictors, labels,
      true_labels, predicted_labels,
      instance_features, predictor_features,
      label_features)

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
      instances, predictors, labels,
      true_labels, predicted_labels,
      instance_features, predictor_features,
      label_features)

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
      data = np.stack(
        [instances, predictors, labels, values],
        axis=-1)
      np.random.shuffle(data)
      instances, predictors, labels, values = [
        data[..., i] for i in np.arange(data.shape[-1])]
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


class LegacyLoader(object):
  @staticmethod
  def load(
      data_dir, dataset_type='nell',
      labels=None, small_version=False):
    """Loads a legacy dataset.

    Args:
      data_dir: Data directory (e.g., for downloading
        and extracting data files).
      dataset_type: Can be `"nell"` or `"brain"`. Defaults
        to all labels for the provided dataset type.
      labels: Labels for which to load data.
      small_version: Boolean flag indicating whether to use
        the small version of the NELL dataset. This is 10%
        of the complete dataset.

    Returns: Loaded legacy dataset for the provided labels.
    """
    if dataset_type is 'nell':
      if labels is None:
        labels = [
          'animal', 'beverage', 'bird', 'bodypart', 'city',
          'disease', 'drug', 'fish', 'food', 'fruit', 'muscle',
          'person', 'protein', 'river', 'vegetable']
    elif dataset_type is 'brain':
      if labels is None:
        labels = [
          'region_1', 'region_2', 'region_3', 'region_4',
          'region_5', 'region_6', 'region_7', 'region_8',
          'region_9', 'region_10', 'region_11']
    else:
      raise ValueError('Legacy loader type can be "nell" or "brain".')

    if len(labels) == 1:
      if dataset_type is 'nell':
        data_dir = os.path.join(data_dir, 'nell', 'unconstrained')
      elif dataset_type is 'brain':
        data_dir = os.path.join(data_dir, 'brain')
      else:
        raise ValueError('Legacy loader type can be "nell" or "brain".')

      label = labels[0]

      # Load instance names.
      if dataset_type is 'nell' and not small_version:
        filename = '{}.txt'.format(label)
        filename = os.path.join(data_dir, 'names', filename)
        with open(filename, 'r') as f:
          instances = [line for line in f]

      # Load predicted labels.
      if small_version:
        data_dir = os.path.join(data_dir, 'small')
      else:
        data_dir = os.path.join(data_dir, 'full')
      filename = '{}.csv'.format(label)
      filename = os.path.join(data_dir, filename)

      if small_version:
        instances = list()
      predictors = list()
      labels = [label]
      true_labels = {0: dict()}
      predicted_labels = {0: dict()}
      is_header = True

      with open(filename, 'r') as f:
        for i, line in enumerate(f):
          if is_header:
            predictors = [s.strip() for s in line.split(',')][1:]
            predicted_labels[0] = {
              p: ([], []) for p in range(len(predictors))}
            is_header = False
          else:
            line_parts = [s.strip() for s in line.split(',')]
            if small_version:
              instances.append('label_%d' % (i - 1))
            true_labels[0][i - 1] = int(line_parts[0])
            for p in range(len(predictors)):
              predicted_labels[0][p][0].append(i - 1)
              predicted_labels[0][p][1].append(float(line_parts[p + 1]))

      return Dataset(
        instances, predictors, labels,
        true_labels, predicted_labels)
    else:
      return Dataset.join([
        LegacyLoader.load(data_dir, dataset_type, [l], small_version)
        for l in labels])


class NELLLoader(object):
  @staticmethod
  def load(data_dir, labels, load_features=True, ground_truth_threshold=0.1):
    if load_features:
      f_filename = 'np_features.tsv'
      f_filename = os.path.join(data_dir, f_filename)
      sparse_features = list()
      num_features = 0
      with open(f_filename, 'r') as f:
        for f_line in f:
          f_line_parts = [s.strip()
                          for s in f_line.split('\t')]
          feature_indices = []
          feature_values = []
          for p in f_line_parts[1].split(','):
            pair = p.split(':')
            f_index = int(pair[0])
            feature_indices.append(f_index - 1)
            feature_values.append(float(pair[1]))
            num_features = max(num_features, f_index)
          sparse_features.append((f_line_parts[0], feature_indices, feature_values))
      features = dict()
      for instance, feature_indices, feature_values in sparse_features:
          f = np.zeros([num_features], np.float32)
          f[feature_indices] = feature_values
          features[instance] = f
    else:
      features = None

    def load_for_label(label):
      filename = '{}.extracted_instances.all_predictions.txt'.format(label)
      filename = os.path.join(data_dir, filename)

      instances = list()
      predictors = list()
      labels = [label]
      true_labels = {0: dict()}
      predicted_labels = {0: dict()}
      instance_features = list() if features is not None else None
      is_header = True
      i = 0

      with open(filename, 'r') as f:
        for line in f:
          if is_header:
            predictors = [s.strip() for s in line.split('\t')][2:]
            predicted_labels[0] = {
              p: ([], []) for p in range(len(predictors))}
            is_header = False
          else:
            line_parts = [s.strip() for s in line.split('\t')]
            instance = line_parts[0]
            if features is not None and instance not in features:
              continue
            i += 1
            instances.append(instance)
            true_labels[0][i - 1] = int(float(line_parts[1]) >= ground_truth_threshold)
            for p in range(len(predictors)):
              if line_parts[p + 2] != '-':
                predicted_labels[0][p][0].append(i - 1)
                predicted_labels[0][p][1].append(float(line_parts[p + 2]))
            if features is not None:
              instance_features.append(features[instance])

      return Dataset(
        instances, predictors, labels,
        true_labels, predicted_labels,
        instance_features=instance_features)

    return Dataset.join([load_for_label(l) for l in labels])

  @staticmethod
  def load_with_ground_truth(data_dir, labels, load_features=True, ground_truth=None):
    if ground_truth is None:
      filename = 'np_labels.tsv'
      filename = os.path.join(data_dir, filename)
      ground_truth = dict()
      with open(filename, 'r') as f:
        for line in f:
          line_parts = [s.strip() for s in line.split('\t')]
          ground_truth[line_parts[0]] = set(line_parts[1].split(','))

    if load_features:
      f_filename = 'np_features.tsv'
      f_filename = os.path.join(data_dir, f_filename)
      sparse_features = list()
      num_features = 0
      with open(f_filename, 'r') as f:
        for f_line in f:
          f_line_parts = [s.strip()
                          for s in f_line.split('\t')]
          feature_indices = []
          feature_values = []
          for p in f_line_parts[1].split(','):
            pair = p.split(':')
            f_index = int(pair[0])
            feature_indices.append(f_index - 1)
            feature_values.append(float(pair[1]))
            num_features = max(num_features, f_index)
          sparse_features.append((f_line_parts[0], feature_indices, feature_values))
      features = dict()
      for instance, feature_indices, feature_values in sparse_features:
        f = np.zeros([num_features], np.float32)
        f[feature_indices] = feature_values
        features[instance] = f
    else:
      features = None

    def load_for_label(label):
      filename = '{}.extracted_instances.all_predictions.txt'.format(label)
      filename = os.path.join(data_dir, filename)

      instances = list()
      predictors = list()
      labels = [label]
      true_labels = {0: dict()}
      predicted_labels = {0: dict()}
      instance_features = list() if features is not None else None
      is_header = True
      i = 0

      with open(filename, 'r') as f:
        for line in f:
          if is_header:
            predictors = [s.strip() for s in line.split('\t')][2:]
            predicted_labels[0] = {
              p: ([], []) for p in range(len(predictors))}
            is_header = False
          else:
            line_parts = [s.strip() for s in line.split('\t')]
            instance = line_parts[0]
            if features is not None and instance not in features:
              continue
            i += 1
            instances.append(instance)
            true_labels[0][i - 1] = int(label in ground_truth[line_parts[0]])
            for p in range(len(predictors)):
              if line_parts[p + 2] != '-':
                predicted_labels[0][p][0].append(i - 1)
                predicted_labels[0][p][1].append(float(line_parts[p + 2]))
            if features is not None:
              instance_features.append(features[instance])

      return Dataset(
        instances, predictors, labels,
        true_labels, predicted_labels,
        instance_features=instance_features)

    return Dataset.join([load_for_label(l) for l in labels])