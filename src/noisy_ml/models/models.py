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

import abc
import logging

import numpy as np
import tensorflow as tf

from collections import namedtuple
from six import with_metaclass

from .layers import *

__author__ = 'eaplatanios'

__all__ = ['BuiltModel', 'LNL', 'MultiClassLNL', 'Model', 'MMCE_M']

logger = logging.getLogger(__name__)


BuiltModel = namedtuple(
  'BuiltModel',
  ['predictions', 'q_params',
   'regularization_terms', 'include_y_prior'])


class Model(with_metaclass(abc.ABCMeta, object)):
  @abc.abstractmethod
  def build(self, instances, predictors, labels):
    raise NotImplementedError


class MMCE_M(Model):
  """Model proposed in 'Regularized Minimax Conditional Entropy for Crowdsourcing'.

  Source: https://arxiv.org/pdf/1503.07240.pdf
  """
  def __init__(self, dataset, gamma=0.25):
    self.dataset = dataset
    self.gamma = gamma

  def build(self, instances, predictors, labels):
    predictions = Embedding(
      num_inputs=len(self.dataset.instances),
      emb_size=len(self.dataset.labels),
      name='instance_embeddings'
    )(instances)
    predictions = tf.log_sigmoid(predictions)

    q_i = Embedding(
      num_inputs=len(self.dataset.instances),
      emb_size=4,
      name='q_i/instance_embeddings'
    )(instances)
    q_i = tf.reshape(q_i, [-1, 2, 2])

    q_p = Embedding(
      num_inputs=len(self.dataset.predictors),
      emb_size=4,
      name='q_p/predictor_embeddings'
    )(predictors)
    q_p = tf.reshape(q_p, [-1, 2, 2])

    q_params = tf.nn.log_softmax(q_i + q_p, axis=-1)

    num_labels_per_worker = self.dataset.avg_labels_per_predictor()
    num_labels_per_item = self.dataset.avg_labels_per_item()
    alpha = self.gamma * ((len(self.dataset.labels) * 2) ** 2)
    beta = alpha * num_labels_per_worker / num_labels_per_item
    regularization_terms = [
      beta * tf.reduce_sum(q_i * q_i) / 2,
      alpha * tf.reduce_sum(q_p * q_p) / 2]

    return BuiltModel(
      predictions, q_params, regularization_terms,
      include_y_prior=False)


class LNL(Model):
  def __init__(
      self,
      dataset,
      instances_emb_size=None,
      predictors_emb_size=None,
      labels_emb_size=None,
      instances_hidden=None,
      predictors_hidden=None,
      q_latent_size=None,
      gamma=0.25):
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


class MultiClassLNL(Model):
  """A multi-class version of LNL model.

  TODO: refactor LNL and merge with this to avoid code duplication.
  """

  def __init__(
      self,
      dataset,
      instances_emb_size=None,
      predictors_emb_size=None,
      labels_emb_size=None,
      instances_hidden=None,
      predictors_hidden=None,
      q_latent_size=None,
      gamma=0.25):
    self.dataset = dataset
    self.instances_emb_size = instances_emb_size
    self.predictors_emb_size = predictors_emb_size
    self.labels_emb_size = labels_emb_size
    self.instances_hidden = instances_hidden or list()
    self.predictors_hidden = predictors_hidden or list()
    self.q_latent_size = q_latent_size
    self.gamma = gamma

  def build(self, instances, predictors, labels):
    """Builds ops.

    TODO: refactor into sub-functions + factor out network creation code into `networks` module.
    """
    if (self.dataset.instance_features is None or
        self.instances_emb_size is not None):
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

    # Compute predictions.
    with tf.variable_scope('m_fn'):
      # Shared hiddens for m_fn between all labels.
      # hiddens: <float32> [batch_size, hidden_size].
      hiddens = instances
      for h_units in self.instances_hidden:
        hiddens = tf.layers.Dense(
            units=h_units,
            activation=tf.nn.selu)
      # Predictions is a list of num_labels tensors:
      # <float32> [batch_size, num_classes].
      with tf.variable_scope('log_softmax'):
        predictions = []
        for nc in self.dataset.num_classes:
          predictions.append(tf.nn.log_softmax(
              tf.layers.Dense(units=nc)(hiddens)))

    # Predictor embeddings.
    # predictors: <float32> [batch_size, predictor_emb_size]
    predictors = Embedding(
      num_inputs=len(self.dataset.predictors),
      emb_size=self.predictors_emb_size,
      name='predictor_embeddings'
    )(predictors)

    # Compute annotator quality confusion matrices (per-label).
    confusion_latent_size = self.q_latent_size or 1
    with tf.variable_scope('q_fn'):
      # Shared hiddens for q_fn between all labels.
      # hiddens: <float32> [batch_size, hidden_size].
      hiddens = predictors
      for h_units in self.instances_hidden:
        hiddens = tf.layers.Dense(
            units=h_units,
            activation=tf.nn.selu)
      # Pre-confusions is a list of num_labels tensors:
      # <float32> [batch_size, num_classes, num_classes, latent_size].
      pre_q_confusions = []
      for nc in self.dataset.num_classes:
        pre_q_confusion = tf.layers.Dense(units=(nc * nc * confusion_latent_size))(hiddens)
        pre_q_confusions.append(tf.reshape(pre_q_confusion, [-1, nc, nc, confusion_latent_size]))

    # If annotator confusions are modeled as instance-independent.
    if self.q_latent_size is None:
      with tf.variable_scope('q_fn'):
        # Confusions is a list of num_labels tensors log-normalized along the last axis:
        # <float32> [batch_size, num_classes, num_classes].
        confusions = []
        for pqc in pre_q_confusions:
          c = tf.nn.log_softmax(tf.squeeze(pqc), axis=-1)
          confusions.append(c)
        regularization_terms = []
    # If we also want to take into account per-instance difficulties.
    else:
      with tf.variable_scope('d_fn'):
        # Shared hiddens for d_fn between all labels.
        # hiddens: <float32> [batch_size, hidden_size].
        hiddens = instances
        for h_units in self.instances_hidden:
          hiddens = tf.layers.Dense(
              units=h_units,
              activation=tf.nn.selu)
        # Pre-confusions is a list of num_labels tensors:
        # <float32> [batch_size, num_classes, num_classes, latent_size].
        pre_d_confusions = []
        for nc in self.dataset.num_classes:
          pre_d_confusion = tf.layers.Dense(units=(nc * nc * confusion_latent_size))(hiddens)
          pre_d_confusions.append(tf.reshape(pre_d_confusion, [-1, nc, nc, confusion_latent_size]))

      with tf.variable_scope('q_fn_d_fn'):
        # Combine pre_q_confusions and pre_d_confusions.
        # Confusions is a list of num_labels tensors log-normalized along the last axis:
        # <float32> [batch_size, num_classes, num_classes].
        confusions = []
        for pqc, pdc in zip(pre_q_confusions, pre_d_confusions):
          # Compute confusion matrices.
          c = tf.nn.log_softmax(tf.reduce_logsumexp(pqc + pdc, axis=-1), axis=-1)
          confusions.append(c)

      with tf.variable_scope('reg_terms'):
        # Compute regularization terms.
        num_labels_per_worker = self.dataset.avg_labels_per_predictor()
        num_labels_per_item = self.dataset.avg_labels_per_item()
        # TODO: not sure how to adjust this for multi-class case...
        alpha = self.gamma * ((len(self.dataset.labels) * 2) ** 2)
        beta = alpha * num_labels_per_worker / num_labels_per_item
        regularization_terms = [
            beta * tf.reduce_sum(sum(q_i * q_i for q_i in pre_q_confusions)) / 2,
            alpha * tf.reduce_sum(sum(q_p * q_p for q_p in pre_d_confusions)) / 2]

    return BuiltModel(
      predictions=predictions,
      q_params=confusions,
      regularization_terms=regularization_terms,
      include_y_prior=True)
