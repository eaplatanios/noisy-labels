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
import tensorflow as tf

from .data.loaders import *
from .evaluation.metrics import *
from .models.layers import *
from .models.learners import *
from .models.transformations import *

__author__ = 'eaplatanios'

__all__ = []


def run_experiment(dataset_type, labels, small_version):
  working_dir = os.getcwd()
  data_dir = os.path.join(working_dir, os.pardir, 'data')
  dataset = LegacyLoader.load(
    working_dir=data_dir,
    dataset_type=dataset_type,
    labels=labels,
    small_version=small_version)

  train_data = dataset.to_train()
  train_dataset = tf.data.Dataset.from_tensor_slices({
    'instances': train_data.instances,
    'predictors': train_data.predictors,
    'labels': train_data.labels,
    'values': train_data.values})

  instances_input_fn = Embedding(
    num_inputs=len(dataset.instances),
    emb_size=128,
    name='instance_embeddings')

  predictors_input_fn = Embedding(
    num_inputs=len(dataset.predictors),
    emb_size=16,
    name='predictor_embeddings')

  labels_input_fn = Embedding(
    num_inputs=len(dataset.labels),
    emb_size=16,
    name='label_embeddings')

  qualities_input_fn = Concatenation(arg_indices=[0, 1, 2])

  model_fn = MLP(
    hidden_units=[],
    num_outputs=len(dataset.labels),
    activation=tf.nn.selu,
    output_layer=LogSigmoid(),
    name='model_fn')

  qualities_fn = MLP(
    hidden_units=[],
    num_outputs=2,
    activation=tf.nn.selu,
    name='qualities_fn')

  learner = EMLearner(
    config=MultiLabelEMConfig(
      num_instances=len(dataset.instances),
      num_predictors=len(dataset.predictors),
      num_labels=len(dataset.labels),
      model_fn=model_fn,
      qualities_fn=qualities_fn,
      optimizer=tf.train.AdamOptimizer(),
      instances_input_fn=instances_input_fn,
      predictors_input_fn=predictors_input_fn,
      labels_input_fn=labels_input_fn,
      qualities_input_fn=qualities_input_fn,
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
    dataset_type='nell',
    labels=None,
    small_version=False)
  results['em'].log(prefix='EM           ')
  results['maj'].log(prefix='Majority Vote')
