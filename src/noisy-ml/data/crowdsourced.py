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

import numpy as np
import pandas as pd
import six
import yaml

from .loaders import Dataset

__author__ = ['alshedivat', 'eaplatanios']

__all__ = [
  'BlueBirdsLoader', 'SentimentPopularityLoader',
  'WeatherSentimentLoader']

logger = logging.getLogger(__name__)


class BlueBirdsLoader(object):
  """BlueBirds dataset.

  Source: https://github.com/welinder/cubam/tree/public/demo/bluebirds
  """

  @staticmethod
  def load(data_dir):
    data_dir = os.path.join(
      data_dir, 'crowdsourced', 'bluebirds')

    def convert_labels_to_ints(dictionary):
      return dict(map(
        lambda kv: (kv[0], int(kv[1])),
        six.iteritems(dictionary)))

    def convert_labels_to_floats(dictionary):
      return dict(map(
        lambda kv: (kv[0], float(kv[1])),
        six.iteritems(dictionary)))

    # Load the ground truth.
    gt_filename = os.path.join(data_dir, 'gt.yaml')
    with open(gt_filename, 'r') as f:
      ground_truth = yaml.safe_load(f.read())
      ground_truth = convert_labels_to_ints(ground_truth)

    # Load the predicted labels.
    predicted_labels_filename = os.path.join(
      data_dir, 'labels.yaml')
    with open(predicted_labels_filename, 'r') as f:
      predicted_labels = yaml.safe_load(f.read())
      predicted_labels = dict(map(
        lambda kv: (kv[0], convert_labels_to_floats(kv[1])),
        six.iteritems(predicted_labels)))

    # Convert to the appropriate dataset format.
    instances = list(six.iterkeys(ground_truth))
    predictors = list(six.iterkeys(predicted_labels))
    labels = [0]

    instance_ids = {
      instance: i
      for i, instance in enumerate(instances)}
    predictor_ids = {
      predictor: p
      for p, predictor in enumerate(predictors)}

    def values_to_tuple(values):
      i_ids, values = map(list, zip(*[
        (instance_ids[i], v)
        for i, v in six.iteritems(values)]))
      return i_ids, values

    true_labels = {0: {
      instance_ids[i]: l
      for i, l in six.iteritems(ground_truth)}}
    predicted_labels = {0: {
      predictor_ids[p]: values_to_tuple(values)
      for p, values in six.iteritems(predicted_labels)}}

    return Dataset(
      instances, predictors, labels,
      true_labels, predicted_labels)


class SentimentPopularityLoader(object):
  """Sentiment popularity AMT dataset.

  Source: https://eprints.soton.ac.uk/376544/1/SP_amt.csv
  """

  @staticmethod
  def load(data_dir):
    # Load data.
    datapath = os.path.join(
        data_dir,
        'crowdsourced', 'sentiment_popularity', 'SP_amt.csv')
    column_names = [
        'WorkerID', 'TaskID', 'Label', 'True label', 'Judgement time']
    df = pd.read_csv(datapath, names=column_names)

    # Get annotations
    annotations = df.pivot(index='TaskID', columns='WorkerID', values='Label')

    # Extract instances and predictors.
    instances = annotations.index.values.astype(str).tolist()
    predictors = annotations.columns.values.astype(str).tolist()

    # Extract ground truth.
    labels = [0]
    true_labels = df[['TaskID', 'True label']].drop_duplicates()
    true_labels = true_labels.drop_duplicates().set_index('TaskID')
    true_labels = true_labels.sort_index().values.flatten().tolist()
    true_labels = {0: dict(zip(range(len(true_labels)), true_labels))}

    # Extract annotations.
    annotations = annotations.fillna(-1).values
    predicted_labels = {0: dict()}
    for w_id in range(annotations.shape[1]):
        i_ids = np.nonzero(annotations[:, w_id] >= 0)[0]
        w_ans = annotations[i_ids, w_id]
        predicted_labels[0][w_id] = (i_ids.tolist(), w_ans.tolist())

    return Dataset(
        instances, predictors, labels,
        true_labels, predicted_labels)


class WeatherSentimentLoader(object):
  """Weather sentiment AMT dataset.

  Source: https://eprints.soton.ac.uk/376543/1/WeatherSentiment_amt.csv
  """

  @staticmethod
  def load(data_dir):
    # Load data.
    datapath = os.path.join(
      data_dir,
      'crowdsourced', 'weather_sentiment', 'WeatherSentiment_amt.csv')
    column_names = [
      'WorkerID', 'TaskID', 'Label', 'True label', 'Judgement time']
    df = pd.read_csv(datapath, names=column_names)

    # Get annotations
    annotations = df[['TaskID', 'WorkerID', 'Label']].drop_duplicates()\
      .pivot(index='TaskID', columns='WorkerID', values='Label')

    # Extract instances and predictors.
    instances = annotations.index.values.astype(str).tolist()
    predictors = annotations.columns.values.astype(str).tolist()

    # Extract ground truth.
    labels = [0, 1, 2, 3, 4]
    true_labels = df[['TaskID', 'True label']].drop_duplicates()
    true_labels = true_labels.drop_duplicates().set_index('TaskID')
    true_labels = true_labels.sort_index().values.flatten().tolist()
    true_labels = np.array(true_labels)
    true_labels = {
      0: dict(zip(range(len(true_labels)), (true_labels == 0).astype(np.int32))),
      1: dict(zip(range(len(true_labels)), (true_labels == 1).astype(np.int32))),
      2: dict(zip(range(len(true_labels)), (true_labels == 2).astype(np.int32))),
      3: dict(zip(range(len(true_labels)), (true_labels == 3).astype(np.int32))),
      4: dict(zip(range(len(true_labels)), (true_labels == 4).astype(np.int32)))}

    # Extract annotations.
    annotations = annotations.fillna(-1).values
    predicted_labels = {0: dict(), 1: dict(), 2: dict(), 3: dict(), 4: dict()}
    for w_id in range(annotations.shape[1]):
      i_ids = np.nonzero(annotations[:, w_id] >= 0)[0]
      w_ans = np.array(annotations[i_ids, w_id].tolist())
      i_ids = i_ids.tolist()
      for l in range(5):
        values = (w_ans == l).astype(np.float32).tolist()
        predicted_labels[l][w_id] = (i_ids, values)

    return Dataset(
      instances, predictors, labels,
      true_labels, predicted_labels)
