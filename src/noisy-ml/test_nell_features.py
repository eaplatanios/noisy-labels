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

from .data.loaders import *
from .evaluation.metrics import *
from .models.layers import *
from .models.learners import *
from .models.models import *

__author__ = 'eaplatanios'

__all__ = []


class NELLModel(Model):
  def __init__(self, dataset):
    self.dataset = dataset

  def build(self, instances, predictors, labels):
    if self.dataset.instance_features is None:
      instances_input_fn = Embedding(
        num_inputs=len(self.dataset.instances),
        emb_size=32,
        name='instance_embeddings')
    else:
      instances_input_fn = FeatureMap(
        features=np.array(self.dataset.instance_features),
        adjust_magnitude=True,
        name='instance_features')

    predictors_input_fn = Embedding(
      num_inputs=len(self.dataset.predictors),
      emb_size=32,
      name='predictor_embeddings')

    instances = instances_input_fn(instances)
    predictors = predictors_input_fn(predictors)

    q_fn_args = predictors
    # q_fn_args = Concatenation([0, 1])(instances, predictors)

    predictions = MLP(
      hidden_units=[128, 64, 32],
      activation=tf.nn.selu,
      output_layer=LogSigmoid(
        num_labels=len(self.dataset.labels)),
      name='m_fn'
    )(instances)

    q_params = MLP(
      hidden_units=[128, 64, 32],
      activation=tf.nn.selu,
      output_layer=Linear(num_outputs=4),
      name='q_fn'
    )(q_fn_args)

    return BuiltModel(predictions, q_params)


class NELLLowRankModel(Model):
  def __init__(self, dataset, q_latent_size):
    self.dataset = dataset
    self.q_latent_size = q_latent_size

  def build(self, instances, predictors, labels):
    if self.dataset.instance_features is None:
      instances_input_fn = Embedding(
        num_inputs=len(self.dataset.instances),
        emb_size=32,
        name='instance_embeddings')
    else:
      instances_input_fn = FeatureMap(
        features=np.array(self.dataset.instance_features),
        adjust_magnitude=True,
        name='instance_features')
    predictors_input_fn = Embedding(
      num_inputs=len(self.dataset.predictors),
      emb_size=32,
      name='predictor_embeddings')

    instances = instances_input_fn(instances)
    predictors = predictors_input_fn(predictors)

    predictions = MLP(
      hidden_units=[128, 64, 32],
      activation=tf.nn.selu,
      output_layer=LogSigmoid(
        num_labels=len(self.dataset.labels)),
      name='m_fn'
    )(instances)

    q_i = MLP(
      hidden_units=[],
      activation=tf.nn.selu,
      output_layer=Linear(num_outputs=4*self.q_latent_size),
      name='q_i_fn'
    )(instances)

    q_p = MLP(
      hidden_units=[],
      activation=tf.nn.selu,
      output_layer=Linear(num_outputs=4*self.q_latent_size),
      name='q_p_fn'
    )(predictors)

    q_i = tf.reshape(q_i, [-1, 4, self.q_latent_size])
    q_p = tf.reshape(q_p, [-1, 4, self.q_latent_size])

    q_params = tf.reduce_sum(q_i * q_p, axis=-1)

    return BuiltModel(predictions, q_params)


def run_experiment(labels, ground_truth_threshold):
  working_dir = os.getcwd()
  data_dir = os.path.join(working_dir, os.pardir, 'data')

  data_dir = '/Volumes/Macintosh HD/Users/eaplatanios/Development/Data/NELL/Aggregated Predictions'
  dataset = NELLLoader.load(
    data_dir=data_dir,
    labels=labels,
    load_features=False,
    ground_truth_threshold=ground_truth_threshold)
  # dataset = NELLLoader.load_with_ground_truth(
  #   data_dir=data_dir,
  #   labels=labels)

  train_data = dataset.to_train()
  train_dataset = tf.data.Dataset.from_tensor_slices({
    'instances': train_data.instances,
    'predictors': train_data.predictors,
    'labels': train_data.labels,
    'values': train_data.values})

  # model = NELLModel(dataset)
  model = NELLLowRankModel(dataset, q_latent_size=8)

  learner = EMLearner(
    config=MultiLabelFullConfusionEMConfig(
      num_instances=len(dataset.instances),
      num_predictors=len(dataset.predictors),
      num_labels=len(dataset.labels),
      model=model,
      optimizer=tf.train.AdamOptimizer(),
      use_soft_maj=True,
      use_soft_y_hat=False),
    predictions_output_fn=np.exp)

  evaluator = Evaluator(learner, dataset)

  def em_callback(_):
    Result.merge(evaluator.evaluate_per_label()).log(prefix='EM           ')
    Result.merge(evaluator.evaluate_maj_per_label()).log(prefix='Majority Vote')

  learner.train(
    dataset=train_dataset,
    batch_size=128,
    warm_start=True,
    max_m_steps=10000,
    max_em_steps=10,
    log_m_steps=1000,
    em_step_callback=em_callback)

  return {
    'em': Result.merge(evaluator.evaluate_per_label()),
    'maj': Result.merge(evaluator.evaluate_maj_per_label())}


if __name__ == '__main__':
  results = run_experiment(
    labels=['city'],
    # labels=['bird', 'city', 'country', 'fish', 'lake', 'mammal', 'river'],
    ground_truth_threshold=0.1)
  results['em'].log(prefix='EM           ')
  results['maj'].log(prefix='Majority Vote')
