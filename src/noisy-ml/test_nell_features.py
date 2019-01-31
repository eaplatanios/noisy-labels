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

__author__ = 'eaplatanios'

__all__ = []


def run_experiment(labels, ground_truth_threshold):
  working_dir = os.getcwd()
  data_dir = os.path.join(working_dir, os.pardir, 'data')

  data_dir = '/Volumes/Macintosh HD/Users/eaplatanios/Development/Data/NELL/Aggregated Predictions'
  dataset = NELLLoader.load(
    data_dir=data_dir,
    labels=labels,
    ground_truth_threshold=ground_truth_threshold)

  train_data = dataset.to_train()
  train_dataset = tf.data.Dataset.from_tensor_slices({
    'instances': train_data.instances,
    'predictors': train_data.predictors,
    'labels': train_data.labels,
    'values': train_data.values})

  if dataset.instance_features is None:
    instances_input_fn = Embedding(
      num_inputs=len(dataset.instances),
      emb_size=2,
      name='instance_embeddings')
  else:
    instances_input_fn = FeatureMap(
      features=np.array(dataset.instance_features),
      adjust_magnitude=True,
      name='instance_features')

  predictors_input_fn = Embedding(
    num_inputs=len(dataset.predictors),
    emb_size=128,
    name='predictor_embeddings')

  labels_input_fn = Embedding(
    num_inputs=len(dataset.labels),
    emb_size=1,
    name='label_embeddings')

  qualities_input_fn = Concatenation(arg_indices=[0, 1])

  output_layer = LogSigmoid(
    num_labels=len(dataset.labels))

  model_fn = MLP(
    hidden_units=[128, 64, 32],
    activation=tf.nn.selu,
    output_layer=output_layer,
    name='model_fn')

  qualities_fn = MLP(
    hidden_units=[128, 64, 32],
    activation=tf.nn.selu,
    output_layer=Linear(num_outputs=2),
    name='qualities_fn')

  learner = EMLearner(
    config=MultiLabelEMConfig(
      num_instances=len(dataset.instances),
      num_predictors=len(dataset.predictors),
      num_labels=len(dataset.labels),
      model_fn=model_fn,
      qualities_fn=qualities_fn,
      optimizer=tf.train.AdamOptimizer(1e-2),
      instances_input_fn=instances_input_fn,
      predictors_input_fn=predictors_input_fn,
      labels_input_fn=labels_input_fn,
      qualities_input_fn=qualities_input_fn,
      predictions_output_fn=np.exp,
      use_soft_maj=True,
      use_soft_y_hat=False,
      max_param_value=None))

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
    ground_truth_threshold=0.1)
  results['em'].log(prefix='EM           ')
  results['maj'].log(prefix='Majority Vote')