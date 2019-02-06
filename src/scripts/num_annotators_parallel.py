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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import tensorflow as tf

from concurrent import futures
from functools import partial
from itertools import product

from noisy_ml.data.crowdsourced import *
from noisy_ml.data.rte import *
from noisy_ml.data.wordsim import *
from noisy_ml.evaluation.metrics import *
from noisy_ml.models.layers import *
from noisy_ml.models.learners import *
from noisy_ml.models.models import *
from noisy_ml.models.amsgrad import *

__author__ = 'eaplatanios'

logger = logging.getLogger(__name__)


def reset_seed():
  seed = 1234567890
  np.random.seed(seed)
  tf.set_random_seed(seed)


class LNL(Model):
  def __init__(
      self, dataset, instances_emb_size=None,
      predictors_emb_size=None, labels_emb_size=None,
      instances_hidden=None, predictors_hidden=None,
      q_latent_size=None, gamma=0.25):
    self.dataset = dataset
    self.instances_emb_size = instances_emb_size
    self.predictors_emb_size = predictors_emb_size
    self.labels_emb_size = labels_emb_size
    self.instances_hidden = instances_hidden or list()
    self.predictors_hidden = predictors_hidden or list()
    self.q_latent_size = q_latent_size
    self.gamma = gamma

  def build(self, instances, predictors, labels):
    if self.dataset.instance_features is None or \
        self.instances_emb_size is not None:
      instances = Embedding(
        num_inputs=len(self.dataset.instances),
        emb_size=self.instances_emb_size,
        name='instance_embeddings'
      )(instances)
    else:
      instances = FeatureMap(
        features=np.array(self.dataset.instance_features),
        adjust_magnitude=False,
        name='instance_features/feature_map'
      )(instances)

    predictions = MLP(
      hidden_units=self.instances_hidden,
      activation=tf.nn.selu,
      output_layer=LogSigmoid(
        num_labels=len(self.dataset.labels),
        name='m_fn/log_sigmoid'),
      name='m_fn'
    )(instances)

    predictors = Embedding(
      num_inputs=len(self.dataset.predictors),
      emb_size=self.predictors_emb_size,
      name='predictor_embeddings'
    )(predictors)

    if self.q_latent_size is None:
      q_fn_args = predictors
      q_params = MLP(
        hidden_units=self.predictors_hidden,
        activation=tf.nn.selu,
        name='q_fn'
      ).and_then(Linear(
        num_outputs=4,
        name='q_fn/linear')
      )(q_fn_args)
      q_params = tf.reshape(q_params, [-1, 2, 2])
      q_params = tf.nn.log_softmax(q_params, axis=-1)

      regularization_terms = []
    else:
      q_i = MLP(
        hidden_units=self.instances_hidden,
        activation=tf.nn.selu,
        output_layer=Linear(
          num_outputs=4*self.q_latent_size,
          name='q_i_fn/linear'),
        name='q_i_fn'
      )(instances)
      q_i = tf.reshape(q_i, [-1, 2, 2, self.q_latent_size])

      q_p = MLP(
        hidden_units=self.predictors_hidden,
        activation=tf.nn.selu,
        output_layer=Linear(
          num_outputs=4*self.q_latent_size,
          name='q_p_fn/linear'),
        name='q_p_fn'
      )(predictors)
      q_p = tf.reshape(q_p, [-1, 2, 2, self.q_latent_size])

      q_params = q_i + q_p
      q_params = tf.reduce_logsumexp(q_params, axis=-1)
      q_params = tf.nn.log_softmax(q_params, axis=-1)

      num_labels_per_worker = self.dataset.avg_labels_per_predictor()
      num_labels_per_item = self.dataset.avg_labels_per_item()
      alpha = self.gamma * ((len(self.dataset.labels) * 2) ** 2)
      beta = alpha * num_labels_per_worker / num_labels_per_item
      regularization_terms = [
        beta * tf.reduce_sum(q_i * q_i) / 2,
        alpha * tf.reduce_sum(q_p * q_p) / 2]

    return BuiltModel(
      predictions=predictions,
      q_params=q_params,
      regularization_terms=regularization_terms,
      include_y_prior=True)


def sample_predictors(predictors, num_to_sample, num_sets=5):
  if len(predictors) <= num_to_sample:
    yield predictors
  else:
    for _ in range(num_sets):
      yield random.sample(predictors, num_to_sample)


def learner_fn(model, dataset):
  return EMLearner(
    config=MultiLabelEMConfig(
      num_instances=len(dataset.instances),
      num_predictors=len(dataset.predictors),
      num_labels=len(dataset.labels),
      model=model,
      optimizer=AMSGrad(1e-3),
      use_soft_maj=True,
      use_soft_y_hat=False),
    predictions_output_fn=np.exp)


def train_eval_predictors(args, dataset):
  reset_seed()

  model, model_name, num_p, num_repetitions = args
  num_p_results = []
  sampled_predictors = list(sample_predictors(
    dataset.predictors,
    num_p,
    num_sets=num_repetitions))
  for r, predictors in enumerate(sampled_predictors):
    logger.info(
      'Running repetition %d/%d for %s for %d predictors.'
      % (r + 1, len(sampled_predictors), model_name, num_p))
    data = dataset.filter_predictors(predictors)
    evaluator = Evaluator(data)

    if model == 'MAJ':
      result = evaluator.evaluate_maj_per_label()[0]
    else:
      with tf.Graph().as_default():
        train_data = data.to_train(shuffle=True)
        train_dataset = tf.data.Dataset.from_tensor_slices({
          'instances': train_data.instances,
          'predictors': train_data.predictors,
          'labels': train_data.labels,
          'values': train_data.values})

        learner = learner_fn(model, dataset)
        learner.train(
          dataset=train_dataset,
          batch_size=1024,
          warm_start=True,
          max_m_steps=1000,
          max_em_steps=10,
          max_marginal_steps=2000,
          log_m_steps=None,
          use_progress_bar=True)
        # TODO: Average results across all labels.
        result = evaluator.evaluate_per_label(
          learner=learner,
          batch_size=128)[0]
    num_p_results.append(result)

  # Collect results.
  accuracies = [r.accuracy for r in num_p_results]
  acc_result = {
    'model': model_name,
    'num_predictors': num_p,
    'metric': 'accuracy',
    'value_mean': np.mean(accuracies),
    'value_std': np.std(accuracies)}
  aucs = [r.auc for r in num_p_results]
  auc_result = {
    'model': model_name,
    'num_predictors': num_p,
    'metric': 'auc',
    'value_mean': np.mean(aucs),
    'value_std': np.std(aucs)}

  return acc_result, auc_result


def run_experiment(num_proc=1):
  reset_seed()

  dataset = 'wordsim'
  working_dir = os.getcwd()
  data_dir = os.path.join(working_dir, os.pardir, 'data')
  results_dir = os.path.join(working_dir, os.pardir, 'results')

  if not os.path.exists(results_dir):
    os.makedirs(results_dir)

  if dataset is 'bluebirds':
    dataset = BlueBirdsLoader.load(data_dir, load_features=True)
    num_predictors = [1, 10, 20, 39]
    num_repetitions = [20, 10, 5, 1]
    results_path = os.path.join(results_dir, 'bluebirds.csv')
  elif dataset is 'rte':
    dataset = RTELoader.load(data_dir, load_features=True)
    num_predictors = [1, 10, 20, 50, 100, 164]
    num_repetitions = [20, 10, 10, 5, 3, 1]
    results_path = os.path.join(results_dir, 'rte.csv')
  elif dataset is 'wordsim':
    dataset = WordSimLoader.load(data_dir, load_features=True)
    num_predictors = [1, 2, 5, 10]
    num_repetitions = [20, 10, 5, 1]
    results_path = os.path.join(results_dir, 'wordsim.csv')
  else:
    raise NotImplementedError

  models = {
    'MAJ': 'MAJ',
    'MMCE-M (γ=0.25)': MMCE_M(dataset, gamma=0.25),
    'LNL[4] (γ=0.25)': LNL(
      dataset=dataset, instances_emb_size=4,
      predictors_emb_size=4, q_latent_size=1, gamma=0.25),
    'LNL[16] (γ=0.25)': LNL(
      dataset=dataset, instances_emb_size=16,
      predictors_emb_size=16, q_latent_size=1, gamma=0.25),
    'LNL[BERT,16,16] (γ=0.25)': LNL(
      dataset=dataset, instances_emb_size=None,
      predictors_emb_size=16,
      instances_hidden=[16],
      predictors_hidden=[16],
      q_latent_size=1, gamma=0.25)
  }

  results = pd.DataFrame(
    columns=[
      'model', 'num_predictors', 'metric',
      'value_mean', 'value_std'])

  with futures.ProcessPoolExecutor(num_proc) as executor:
    func = partial(train_eval_predictors, dataset=dataset)
    inputs = [
      (model, name, num_p, num_r)
      for (name, model), (num_p, num_r) in product(
        models.items(), zip(num_predictors, num_repetitions))]
    model_results = executor.map(func, inputs)
    for n, res in enumerate(model_results, start=1):
      logger.info(
        'Finished experiment for %d/%d predictors.'
        % (n, len(num_predictors)))
      for r in res:
        results = results.append(r, ignore_index=True)
      logger.info('Results so far:\n%s' % str(results))

  logger.info('Results:\n%s' % str(results))

  results.to_csv(results_path)

  results = pd.read_csv(results_path)

  # Accuracy Plot.
  fig, ax = plt.subplots()
  results_auc = results[results.metric == 'accuracy']
  for label, auc in results_auc.groupby('model'):
    ax.plot(
      auc.num_predictors.astype(np.int32),
      auc.value_mean,
      label=label)
    ax.fill_between(
      auc.num_predictors.astype(np.int32),
      auc.value_mean - auc.value_std,
      auc.value_mean + auc.value_std,
      alpha=0.35)
  plt.legend()
  plt.show()

  # AUC Plot.
  fig, ax = plt.subplots()
  results_auc = results[results.metric == 'auc']
  for label, auc in results_auc.groupby('model'):
    ax.plot(
      auc.num_predictors.astype(np.int32),
      auc.value_mean,
      label=label)
    ax.fill_between(
      auc.num_predictors.astype(np.int32),
      auc.value_mean - auc.value_std,
      auc.value_mean + auc.value_std,
      alpha=0.35)
  plt.legend()
  plt.show()


if __name__ == '__main__':
  run_experiment(num_proc=8)
