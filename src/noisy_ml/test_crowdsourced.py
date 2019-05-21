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

import os
import numpy as np
import tensorflow as tf

from .data.crowdsourced import *
from .evaluation.metrics import *
from .models.layers import *
from .models.learners import *
from .models.models import *

__author__ = ['alshedivat', 'eaplatanios']


class CrowdsourcedModel(Model):
  def __init__(self, dataset):
    self.dataset = dataset

  def build(self, instances, predictors, labels):
    instances_input_fn = Embedding(
      num_inputs=len(self.dataset.instances),
      emb_size=16,
      name='instance_embeddings')

    predictors_input_fn = Embedding(
      num_inputs=len(self.dataset.predictors),
      emb_size=16,
      name='predictor_embeddings')

    labels_input_fn = Embedding(
      num_inputs=len(self.dataset.labels),
      emb_size=16,
      name='label_embeddings')

    instances = instances_input_fn(instances)
    predictors = predictors_input_fn(predictors)
    labels = labels_input_fn(labels)

    # output_layer = LogSigmoid(
    #   num_labels=len(self.dataset.labels))
    # output_layer = LogSoftmax(
    #   num_labels=len(dataset.labels))
    # output_layer = HierarchicalLogSoftmax(
    #   num_labels=len(dataset.labels),
    #   hierarchy=[
    #     (0, [
    #       (2, []),
    #       (7, []),
    #       (11, [])]),
    #     (8, [
    #       (1, []),
    #       (8, []),
    #       (12, []),
    #       (13, [])]),
    #     (3, []),
    #     (4, []),
    #     (5, []),
    #     (6, []),
    #     (9, []),
    #     (10, [])])

    predictions = MLP(
      hidden_units=[],
      activation=tf.nn.selu,
      output_layer=LogSigmoid(
        num_labels=len(self.dataset.labels)),
      name='m_fn'
    )(instances)

    q_fn_args = Concatenation([1, 2])(instances, predictors, labels)

    q_params = MLP(
      hidden_units=[],
      activation=tf.nn.selu,
      output_layer=Linear(num_outputs=4),
      name='q_fn'
    )(q_fn_args)

    return BuiltModel(predictions, q_params)


def run_experiment(dataset_type):
  working_dir = os.getcwd()
  data_dir = os.path.join(working_dir, os.pardir, 'data')
  if dataset_type == 'sentiment_popularity':
    dataset = SentimentPopularityLoader.load(data_dir)
  elif dataset_type == 'weather_sentiment':
    dataset = WeatherSentimentLoader.load(data_dir)
  elif dataset_type == 'bluebirds':
    dataset = BlueBirdsLoader.load(data_dir, load_features=False)
  else:
    raise ValueError('Unknown dataset: %s', dataset_type)

  train_data = dataset.to_train()
  train_dataset = tf.data.Dataset.from_tensor_slices({
    'instances': train_data.instances,
    'predictors': train_data.predictors,
    'labels': train_data.labels,
    'values': train_data.values})

  model = CrowdsourcedModel(dataset)

  def predictions_output_fn(predictions):
    # if dataset_type == 'weather_sentiment':
    #   max_indices = predictions.argmax(1)
    #   predictions[:] = 0.0
    #   predictions[
    #     np.arange(len(predictions)),
    #     max_indices] = 1.0
    #   return predictions
    # else:
    #   return np.exp(predictions)
    return np.exp(predictions)

  learner = EMLearner(
    config=MultiLabelEMConfig(
      num_instances=len(dataset.instances),
      num_predictors=len(dataset.predictors),
      num_labels=len(dataset.labels),
      model=model,
      optimizer=tf.train.AdamOptimizer(),
      use_soft_maj=True,
      use_soft_y_hat=False),
    predictions_output_fn=predictions_output_fn)

  evaluator = Evaluator(learner, dataset)

  def em_callback(_):
    Result.merge(evaluator.evaluate_per_label(batch_size=128)).log(prefix='EM           ')
    Result.merge(evaluator.evaluate_maj_per_label()).log(prefix='Majority Vote')

  learner.train(
    dataset=train_dataset,
    batch_size=32,
    warm_start=True,
    max_m_steps=10000,
    max_em_steps=10,
    log_m_steps=1000,
    em_step_callback=em_callback)

  return {
    'em': Result.merge(evaluator.evaluate_per_label(batch_size=128)),
    'maj': Result.merge(evaluator.evaluate_maj_per_label())}


if __name__ == '__main__':
  # results = run_experiment(
  #   dataset_type='sentiment_popularity')
  # results = run_experiment(
  #   dataset_type='weather_sentiment')
  results = run_experiment(
    dataset_type='bluebirds')
  results['em'].log(prefix='EM           ')
  results['maj'].log(prefix='Majority Vote')