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
import tensorflow as tf

from collections import namedtuple
from six import with_metaclass

from .layers import *

__author__ = 'eaplatanios'

__all__ = ['BuiltModel', 'Model', 'MMCE_M']

logger = logging.getLogger(__name__)


BuiltModel = namedtuple(
  'BuiltModel',
  ['predictions', 'q_params', 'regularization_terms'])


class Model(with_metaclass(abc.ABCMeta, object)):
  @abc.abstractmethod
  def build(self, instances, predictors, labels):
    raise NotImplementedError


class MMCE_M(Model):
  """Model proposed in 'Regularized Minimax Conditional Entropy for Crowdsourcing'.

  Source: https://arxiv.org/pdf/1503.07240.pdf
  """
  def __init__(self, dataset):
    self.dataset = dataset

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
    gamma = 0.25
    alpha = gamma * (len(self.dataset.labels) ** 2)
    beta = alpha * num_labels_per_worker / num_labels_per_item
    regularization_terms = [
      beta * tf.reduce_sum(q_i * q_i) / 2,
      alpha * tf.reduce_sum(q_p * q_p) / 2]

    return BuiltModel(predictions, q_params, regularization_terms)
