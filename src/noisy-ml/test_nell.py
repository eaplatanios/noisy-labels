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

import numpy as np
import os
import tensorflow as tf
import six

from .data.loaders import NELLLoader
from .data.utilities import compute_binary_qualities
from .evaluation.metrics import *
from .models.layers import *
from .models.learners import *
from .models.transformations import *

__author__ = 'eaplatanios'

__all__ = []


def run_experiment(label):
  instances_emb_size = 128
  predictors_emb_size = 128
  batch_size = 512

  working_dir = os.getcwd()
  data_dir = os.path.join(working_dir, os.pardir, 'data')
  data = NELLLoader(
    label=label,
    small_version=False
  ).load(data_dir)

  num_instances = len(data['instances'])
  num_predictors = len(data['classifiers'])

  instances = np.array(data['instances'], dtype=np.int32)[:, None]
  predictor_indices = np.array(data['predictors'], dtype=np.int32)[:, :, None]
  predictor_values = np.array(data['predictor_values'], dtype=np.int32)[:, :, None]

  dataset = tf.data.Dataset.from_tensor_slices({
    'instances': instances,
    'predictors': predictor_indices,
    'predictor_values': predictor_values}) \
    .shuffle(20733) \
    .repeat() \
    .batch(batch_size)

  instances_input_fn = Embedding(
    num_inputs=num_instances,
    emb_size=instances_emb_size,
    name='instance_embeddings')

  predictors_input_fn = Embedding(
    num_inputs=num_predictors,
    emb_size=predictors_emb_size,
    name='predictor_embeddings')

  qualities_input_fn = InstancesPredictorsConcatenation()

  model_fn = MLP(
    hidden_units=[128, 64, 32],
    num_outputs=1,
    activation=tf.nn.leaky_relu,
    output_projection=tf.sigmoid,
    name='model_fn')

  qualities_fn = MLP(
    hidden_units=[128, 64, 32],
    num_outputs=2,
    activation=tf.nn.leaky_relu,
    name='qualities_fn')

  learner = NoisyLearner(
    instances_input_size=1,
    predictors_input_size=1,
    config=BinaryNoisyLearnerConfig(
      model_fn=model_fn,
      qualities_fn=qualities_fn,
      prior_correct=0.99,
      max_param_value=1e6,
      warm_up_steps=1000,
      eps=1e-12),
    optimizer=tf.train.AdamOptimizer(),
    instances_dtype=tf.int32,
    predictors_dtype=tf.int32,
    instances_input_fn=instances_input_fn,
    predictors_input_fn=predictors_input_fn,
    qualities_input_fn=qualities_input_fn)

  learner.train(dataset, max_steps=10000, log_steps=1000)

  predictions = learner.predict(instances)[:, 0]
  predicted_qualities = learner.qualities(instances, predictor_indices)
  predicted_qualities = np.mean(predicted_qualities, axis=0)

  true_qualities = compute_binary_qualities(data)
  true_qualities = [p[1] for p in sorted(six.iteritems(true_qualities))]

  return {
    'predictions': predictions,
    'true_labels': np.array(data['true_labels']),
    'predicted_qualities': np.array(predicted_qualities),
    'true_qualities': np.array(true_qualities)}


def evaluate(results):
  mad_error_rank = compute_mad_error_rank(
    results['predicted_qualities'],
    results['true_qualities'])
  mad_error = compute_mad_error(
    results['predicted_qualities'],
    results['true_qualities'])
  auc_target = compute_auc(
    results['predictions'],
    results['true_labels'])
  print('Current MAD_error_rank: {}'.format(mad_error_rank))
  print('Current MAD_error: {}'.format(mad_error))
  print('Current AUC_target: {}'.format(auc_target))


def main():
  labels = [
    'animal', 'beverage', 'bird', 'bodypart', 'city',
    'disease', 'drug', 'fish', 'food', 'fruit', 'muscle',
    'person', 'protein', 'river', 'vegetable']

  results = None

  for label in labels:
    with tf.Graph().as_default():
      label_results = run_experiment(label)
    if results is None:
      results = {
        'predictions': label_results['predictions'],
        'true_labels': label_results['true_labels'],
        'predicted_qualities': label_results['predicted_qualities'],
        'true_qualities': label_results['true_qualities']}
    else:
      results = {
        'predictions': np.concatenate(
          [results['predictions'], label_results['predictions']]),
        'true_labels': np.concatenate(
          [results['true_labels'], label_results['true_labels']]),
        'predicted_qualities': np.concatenate(
          [results['predicted_qualities'], label_results['predicted_qualities']]),
        'true_qualities': np.concatenate(
          [results['true_qualities'], label_results['true_qualities']])}

    print('Results so far:')
    evaluate(results)

  print('haha Christoph')


if __name__ == '__main__':
  main()
