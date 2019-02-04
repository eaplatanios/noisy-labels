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
import pprint
import six
import tensorflow as tf

from tqdm import tqdm

from noisy_ml.data.crowdsourced import *
from noisy_ml.data.rte import *
from noisy_ml.evaluation.metrics import *
from noisy_ml.models.layers import *
from noisy_ml.models.learners import *
from noisy_ml.models.models import *
from noisy_ml.models.amsgrad import *

__author__ = 'eaplatanios'

logger = logging.getLogger(__name__)


seed = 1234567890
np.random.seed(seed)
tf.random.set_random_seed(seed)


class LNL(Model):
  def __init__(
      self, dataset, instances_emb_size=None,
      predictors_emb_size=None, labels_emb_size=None,
      instances_hidden=None, predictors_hidden=None,
      q_latent_size=None, gamma=0.2):
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
      regularization_terms=regularization_terms)


def run_experiment():
  working_dir = os.getcwd()
  data_dir = os.path.join(working_dir, os.pardir, 'data')

  dataset = BlueBirdsLoader.load(data_dir, load_features=True)
  # dataset = RTELoader.load(data_dir)

  def learner_fn(model):
    return EMLearner(
      config=MultiLabelEMConfig(
        num_instances=len(dataset.instances),
        num_predictors=len(dataset.predictors),
        num_labels=len(dataset.labels),
        model=model,
        optimizer=AMSGrad(1e-3), # tf.train.AdamOptimizer(),
        use_soft_maj=True,
        use_soft_y_hat=False),
      predictions_output_fn=np.exp)

  num_annotators = [1, 10, 20, 50, 100, -1]
  models = {
    'MAJ': 'MAJ',
    'MMCE-M (γ=0.00)': MMCE_M(dataset, gamma=0.00),
    'MMCE-M (γ=0.25)': MMCE_M(dataset, gamma=0.25),
    'LNL[4] (γ=0.00)': LNL(
      dataset=dataset, instances_emb_size=4,
      predictors_emb_size=4, q_latent_size=1, gamma=0.00),
    'LNL[4] (γ=0.25)': LNL(
      dataset=dataset, instances_emb_size=4,
      predictors_emb_size=4, q_latent_size=1, gamma=0.25),
    'LNL[16] (γ=0.00)': LNL(
      dataset=dataset, instances_emb_size=16,
      predictors_emb_size=16, q_latent_size=1, gamma=0.00),
    'LNL[16] (γ=0.25)': LNL(
      dataset=dataset, instances_emb_size=16,
      predictors_emb_size=16, q_latent_size=1, gamma=0.25),
    'LNL[VGG,16,64-32-16] (γ=0.00)': LNL(
      dataset=dataset, instances_emb_size=None,
      predictors_emb_size=16,
      instances_hidden=[64, 32, 16],
      predictors_hidden=[64, 32, 16],
      q_latent_size=1, gamma=0.00),
    'LNL[VGG,16,64-32-16] (γ=0.25)': LNL(
      dataset=dataset, instances_emb_size=None,
      predictors_emb_size=16,
      instances_hidden=[64, 32, 16],
      predictors_hidden=[64, 32, 16],
      q_latent_size=1, gamma=0.25)
  }

  pp = pprint.PrettyPrinter(indent=2)

  model_results = dict()
  for name, model in tqdm(six.iteritems(models), desc='Model'):
    results = dict()
    for num_a in tqdm(num_annotators, desc='#Annotators'):
      data = dataset
      if num_a > -1:
        data = data.filter_predictors(
          dataset.predictors[:num_a])
      evaluator = Evaluator(data)

      if model is 'MAJ':
        result = evaluator.evaluate_maj_per_label()[0]
      else:
        with tf.Graph().as_default():
          train_data = data.to_train(shuffle=True)
          train_dataset = tf.data.Dataset.from_tensor_slices({
            'instances': train_data.instances,
            'predictors': train_data.predictors,
            'labels': train_data.labels,
            'values': train_data.values})

          learner = learner_fn(model)
          learner.train(
            dataset=train_dataset,
            batch_size=1024,
            warm_start=True,
            max_m_steps=1000,
            max_em_steps=10,
            log_m_steps=None,
            use_progress_bar=True)
          # TODO: Average results across all labels.
          result = evaluator.evaluate_per_label(
            learner=learner,
            batch_size=128)[0]
      results[num_a] = result
    model_results[name] = results
    logger.info('%s:\n%s' % (name, pp.pformat(results)))

  logger.info(pp.pformat(model_results))


if __name__ == '__main__':
  run_experiment()
