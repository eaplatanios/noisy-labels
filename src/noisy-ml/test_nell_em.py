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

from .data.loaders import BrainLoader, NELLLoader
from .data.utilities import compute_binary_qualities
from .evaluation.metrics import *
from .models.layers import *
from .models.learners import *
from .models.transformations import *

__author__ = 'eaplatanios'

__all__ = []


def evaluate(results):
  mad_error_rank = compute_mad_error_rank(
    results['predicted_qualities'],
    results['true_qualities'])
  mad_error = compute_mad_error(
    results['predicted_qualities'],
    results['true_qualities'])
  auc = compute_auc(
    results['predictions'],
    results['true_labels'])
  accuracy = compute_accuracy(
    np.exp(results['predictions']),
    results['true_labels'])
  maj_soft_auc = compute_auc(
    results['maj_soft_predictions'],
    results['true_labels'])
  maj_soft_accuracy = compute_accuracy(
    results['maj_soft_predictions'],
    results['true_labels'])
  maj_hard_auc = compute_auc(
    results['maj_hard_predictions'],
    results['true_labels'])
  maj_hard_accuracy = compute_accuracy(
    results['maj_hard_predictions'],
    results['true_labels'])
  return {
    'mad_error_rank': mad_error_rank,
    'mad_error': mad_error,
    'auc': auc,
    'accuracy': accuracy,
    'maj_soft_auc': maj_soft_auc,
    'maj_soft_accuracy': maj_soft_accuracy,
    'maj_hard_auc': maj_hard_auc,
    'maj_hard_accuracy': maj_hard_accuracy}


def run_experiment(loader, label, small_version):
  working_dir = os.getcwd()
  data_dir = os.path.join(working_dir, os.pardir, 'data')
  data = loader.load_binary(
    working_dir=data_dir,
    label=label,
    small_version=small_version)

  num_instances = len(data['instances'])
  num_predictors = len(data['classifiers'])

  instances = np.array(data['instances'], dtype=np.int32)
  predictor_indices = np.array(data['predictors'], dtype=np.int32)
  predictor_values = np.array(data['predictor_values'], dtype=np.int32)[:, :, None]
  predictor_values_soft = np.array(data['predictor_values_soft'], dtype=np.float32)[:, :, None]

  dataset = tf.data.Dataset.from_tensor_slices({
    'instances': instances,
    'predictors': predictor_indices,
    'predictor_values': predictor_values,
    'predictor_values_soft': predictor_values_soft})

  # instances_input_fn = OneHotEncoding(
  #   num_inputs=num_instances,
  #   name='instance_embeddings')
  #
  # predictors_input_fn = OneHotEncoding(
  #   num_inputs=num_predictors,
  #   name='predictor_embeddings')

  instances_input_fn = Embedding(
    num_inputs=num_instances,
    emb_size=16,
    name='instance_embeddings')

  predictors_input_fn = Embedding(
    num_inputs=num_predictors,
    emb_size=16,
    name='predictor_embeddings')

  qualities_input_fn = InstancesPredictorsConcatenation()

  model_fn = MLP(
    hidden_units=[],
    num_outputs=1,
    activation=tf.nn.selu,
    name='model_fn')

  qualities_fn = MLP(
    hidden_units=[],
    num_outputs=2,
    activation=tf.nn.selu,
    name='qualities_fn')

  learner = EMLearner(
    config=BinaryEMConfig(
      num_instances=num_instances,
      num_predictors=num_predictors,
      model_fn=model_fn,
      qualities_fn=qualities_fn,
      optimizer=tf.train.AdamOptimizer(),
      instances_input_fn=instances_input_fn,
      predictors_input_fn=predictors_input_fn,
      qualities_input_fn=qualities_input_fn,
      use_soft_maj=True,
      use_soft_y_hat=False,
      max_param_value=None))

  def em_step_callback(learner):
    predictions = learner.predict(instances)
    predicted_qualities = learner.qualities(instances, predictor_indices)
    predicted_qualities = np.mean(predicted_qualities, axis=0)

    true_qualities = compute_binary_qualities(data)
    true_qualities = [p[1] for p in sorted(six.iteritems(true_qualities))]

    results = evaluate({
      'maj_soft_predictions': np.mean(np.array(data['predictor_values_soft'], dtype=np.float32), axis=1),
      'maj_hard_predictions': np.mean(np.array(data['predictor_values'], dtype=np.int32), axis=1),
      'predictions': predictions,
      'true_labels': np.array(data['true_labels']),
      'predicted_qualities': np.array(predicted_qualities)[None, :],
      'true_qualities': np.array(true_qualities)[None, :]})
    print('\tResults so far:')
    print('\tCurrent MAD_error_rank: {}'.format(np.mean(results['mad_error_rank'])))
    print('\tCurrent MAD_error: {}'.format(np.mean(results['mad_error'])))
    print('\tCurrent Accuracy: {}'.format(np.mean(results['accuracy'])))
    print('\tCurrent MAJ Soft Accuracy: {}'.format(np.mean(results['maj_soft_accuracy'])))
    print('\tCurrent MAJ Hard Accuracy: {}'.format(np.mean(results['maj_hard_accuracy'])))
    print('\tCurrent AUC: {}'.format(np.mean(results['auc'])))
    print('\tCurrent MAJ Soft AUC: {}'.format(np.mean(results['maj_soft_auc'])))
    print('\tCurrent MAJ Hard AUC: {}'.format(np.mean(results['maj_hard_auc'])))

  learner.train(
    dataset=dataset,
    batch_size=128,
    warm_start=False,
    max_m_steps=10000,
    max_em_steps=10,
    log_m_steps=1000,
    em_step_callback=em_step_callback)

  predictions = learner.predict(instances)
  predicted_qualities = learner.qualities(instances, predictor_indices)
  predicted_qualities = np.mean(predicted_qualities, axis=0)

  true_qualities = compute_binary_qualities(data)
  true_qualities = [p[1] for p in sorted(six.iteritems(true_qualities))]

  return {
    'predictions': predictions,
    'true_labels': np.array(data['true_labels']),
    'predicted_qualities': np.array(predicted_qualities),
    'true_qualities': np.array(true_qualities),
    'predictor_values': np.array(data['predictor_values'], dtype=np.int32),
    'predictor_values_soft': np.array(data['predictor_values_soft'], dtype=np.float32)}


def main():
  nell_labels = [
    'animal', 'beverage', 'bird', 'bodypart', 'city',
    'disease', 'drug', 'fish', 'food', 'fruit', 'muscle',
    'person', 'protein', 'river', 'vegetable']

  brain_labels = [
    'region_1', 'region_2', 'region_3', 'region_4',
    'region_5', 'region_6', 'region_7', 'region_8',
    'region_9', 'region_10', 'region_11']

  results = {
    'mad_error_rank': [],
    'mad_error': [],
    'accuracy': [],
    'maj_soft_accuracy': [],
    'maj_hard_accuracy': [],
    'auc': [],
    'maj_soft_auc': [],
    'maj_hard_auc': []}

  for label in brain_labels:
    with tf.Graph().as_default():
      label_results = run_experiment(BrainLoader, label, small_version=True)
    label_results = evaluate({
      'maj_soft_predictions': np.mean(label_results['predictor_values_soft'], axis=1),
      'maj_hard_predictions': np.mean(label_results['predictor_values'], axis=1),
      'predictions': label_results['predictions'],
      'true_labels': label_results['true_labels'],
      'predicted_qualities': label_results['predicted_qualities'][None, :],
      'true_qualities': label_results['true_qualities'][None, :]})
    results['mad_error_rank'].append(label_results['mad_error_rank'])
    results['mad_error'].append(label_results['mad_error'])
    results['accuracy'].append(label_results['accuracy'])
    results['maj_soft_accuracy'].append(label_results['maj_soft_accuracy'])
    results['maj_hard_accuracy'].append(label_results['maj_hard_accuracy'])
    results['auc'].append(label_results['auc'])
    results['maj_soft_auc'].append(label_results['maj_soft_auc'])
    results['maj_hard_auc'].append(label_results['maj_hard_auc'])

    print('Results so far:')
    print('Current MAD_error_rank: {}'.format(np.mean(results['mad_error_rank'])))
    print('Current MAD_error: {}'.format(np.mean(results['mad_error'])))
    print('Current Accuracy: {}'.format(np.mean(results['accuracy'])))
    print('Current MAJ Soft Accuracy: {}'.format(np.mean(results['maj_soft_accuracy'])))
    print('Current MAJ Hard Accuracy: {}'.format(np.mean(results['maj_hard_accuracy'])))
    print('Current AUC: {}'.format(np.mean(results['auc'])))
    print('Current MAJ Soft AUC: {}'.format(np.mean(results['maj_soft_auc'])))
    print('Current MAJ Hard AUC: {}'.format(np.mean(results['maj_hard_auc'])))

  print('haha Christoph')


if __name__ == '__main__':
  main()
