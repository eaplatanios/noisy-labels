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
from .networks import VGG16

__author__ = 'eaplatanios'


class BlueBirdsModel(Model):
  def __init__(self, dataset, q_latent_size=None):
    self.dataset = dataset
    self.q_latent_size = q_latent_size

  def build(self, instances, predictors, labels):
    if self.dataset.instance_features is None:
      instances = Embedding(
        num_inputs=len(self.dataset.instances),
        emb_size=16,
        name='instance_embeddings'
      )(instances)
    else:
      # <string> [batch_size]
      instances = FeatureMap(
        features=np.array(self.dataset.instance_features),
        adjust_magnitude=False,
        name='instance_features/feature_map')(instances)
      # <uint8> [batch_size, height, width, channels]
      instances = tf.map_fn(
        lambda img: tf.image.decode_jpeg(img, channels=3),
        instances,
        dtype=tf.uint8,
        back_prop=False)
      instances = tf.image.resize_bilinear(
        instances,
        size=[224, 224])

      # Apply a pre-trained VGG16 to instances.
      instances = VGG16(freeze=True, pooling='max', weights='imagenet')(instances)

    predictions = MLP(
      hidden_units=[],
      activation=tf.nn.selu,
      output_layer=LogSigmoid(
        num_labels=len(self.dataset.labels),
        name='m_fn/log_sigmoid'),
      name='m_fn'
    )(instances)

    predictors = Embedding(
      num_inputs=len(self.dataset.predictors),
      emb_size=16,
      name='predictor_embeddings'
    )(predictors)

    if self.q_latent_size is None:
      q_fn_args = predictors
      q_params = MLP(
        hidden_units=[],
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
        hidden_units=[],
        activation=tf.nn.selu,
        output_layer=Linear(
          num_outputs=4*self.q_latent_size,
          name='q_i_fn/linear'),
        name='q_i_fn'
      )(instances)
      q_i = tf.reshape(q_i, [-1, 2, 2, self.q_latent_size])

      q_p = MLP(
        hidden_units=[],
        activation=tf.nn.selu,
        output_layer=Linear(
          num_outputs=4*self.q_latent_size,
          name='q_p_fn/linear'),
        name='q_p_fn'
      )(predictors)
      q_p = tf.reshape(q_p, [-1, 2, 2, self.q_latent_size])

      q_params = tf.reduce_logsumexp(q_i + q_p, axis=-1)
      q_params = tf.nn.log_softmax(q_params, axis=-1)

      num_labels_per_worker = self.dataset.avg_labels_per_predictor()
      num_labels_per_item = self.dataset.avg_labels_per_item()
      gamma = 0.25
      alpha = gamma * (len(self.dataset.labels) ** 2)
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

  train_data = dataset.to_train()
  train_dataset = tf.data.Dataset.from_tensor_slices({
    'instances': train_data.instances,
    'predictors': train_data.predictors,
    'labels': train_data.labels,
    'values': train_data.values})

  model = BlueBirdsModel(
    dataset=dataset,
    q_latent_size=None)

  learner = EMLearner(
    config=MultiLabelFullConfusionSimpleQEMConfig(
      num_instances=len(dataset.instances),
      num_predictors=len(dataset.predictors),
      num_labels=len(dataset.labels),
      model=model,
      optimizer=tf.train.AdamOptimizer(1e-2),
      use_soft_maj=True,
      use_soft_y_hat=False),
    predictions_output_fn=np.exp)

  evaluator = Evaluator(learner, dataset)

  def em_callback(_):
    Result.merge(evaluator.evaluate_per_label(batch_size=128)).log(prefix='EM           ')
    Result.merge(evaluator.evaluate_maj_per_label()).log(prefix='Majority Vote')

  learner.train(
    dataset=train_dataset,
    batch_size=128,
    warm_start=True,
    max_m_steps=500,
    max_em_steps=100,
    log_m_steps=100,
    em_step_callback=em_callback)

  return {
    'em': Result.merge(evaluator.evaluate_per_label(batch_size=128)),
    'maj': Result.merge(evaluator.evaluate_maj_per_label())}


if __name__ == '__main__':
  results = run_experiment()
  results['em'].log(prefix='EM           ')
  results['maj'].log(prefix='Majority Vote')