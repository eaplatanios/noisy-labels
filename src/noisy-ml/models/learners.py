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

from six import with_metaclass

__author__ = 'eaplatanios'

__all__ = [
  'NoisyLearnerConfig', 'BinaryNoisyLearnerConfig',
  'NoisyLearner', 'NoisyMLPLearner']

logger = logging.getLogger(__name__)


class NoisyLearnerConfig(with_metaclass(abc.ABCMeta, object)):
  @abc.abstractmethod
  def num_outputs(self):
    pass

  @abc.abstractmethod
  def loss_fn(
      self, predictions, qualities_prior,
      predictor_indices, predictor_values):
    pass


class BinaryNoisyLearnerConfig(NoisyLearnerConfig):
  def num_outputs(self):
    return 1

  def loss_fn(
      self, predictions, qualities_prior,
      predictor_indices, predictor_values):
    qualities_prior = tf.batch_gather(qualities_prior, predictor_indices)
    predictions = tf.squeeze(predictions, axis=-1)
    predictor_values = tf.squeeze(predictor_values, axis=-1)
    p = predictions
    a = tf.exp(qualities_prior[:, :, 0])
    b = tf.exp(qualities_prior[:, :, 1])
    temp0 = tf.cast(predictor_values, tf.float32)
    temp1 = 1 - temp0
    term0 = tf.pow(a, temp1)
    term0 *= tf.pow(b, temp0)
    term0 /= (a + b)
    term0 = tf.reduce_prod(term0, axis=1)
    term1 = tf.pow(b, temp1)
    term1 *= tf.pow(a, temp0)
    term1 /= (a + b)
    term1 = tf.reduce_prod(term1, axis=1)
    loss = p * term0 + (1 - p) * term1
    loss = tf.log(loss)
    return -tf.reduce_sum(loss)


class NoisyLearner(with_metaclass(abc.ABCMeta, object)):
  def __init__(
      self, inputs_size, predictors_size,
      config, optimizer,
      instances_input_fn=lambda x: x,
      predictors_input_fn=lambda x: x):
    self.inputs_size = inputs_size
    self.predictors_size = predictors_size
    self.config = config
    self.optimizer = optimizer
    self.instances_input_fn = instances_input_fn
    self.predictors_input_fn = predictors_input_fn
    self._build_model()
    self._session = None

  @abc.abstractmethod
  def model_fn(self, instances):
    pass

  @abc.abstractmethod
  def error_fn(self, instances, predictors):
    pass

  def _build_iterators(self):
    # The train data batches are of the form:
    #   [InstanceFeatures, PredictorFeatures, Prediction]
    self.train_iterator = tf.data.Iterator.from_structure(
      output_types={
        'instances': tf.int32,
        'predictor_indices': tf.int32,
        'predictor_values': tf.int32},
      output_shapes={
        'instances': [None, self.inputs_size],
        'predictor_indices': [None, None],
        'predictor_values': [None, None, self.config.num_outputs()]},
      shared_name='train_iterator')
    iter_next = self.train_iterator.get_next()
    self.instances = tf.placeholder_with_default(
      iter_next['instances'],
      shape=[None, self.inputs_size],
      name='instances')
    self.predictor_indices = tf.placeholder_with_default(
      iter_next['predictor_indices'],
      shape=[None, None],
      name='predictor_indices')
    self.predictor_values = tf.placeholder_with_default(
      iter_next['predictor_values'],
      shape=[None, None, self.config.num_outputs()],
      name='predictor_values')

  def _build_model(self):
    self._build_iterators()
    instances = self.instances
    predictor_indices = self.predictor_indices
    predictor_values = self.predictor_values
    instances = self.instances_input_fn(instances)
    predictors = self.predictors_input_fn(predictor_indices)
    with tf.variable_scope('model_fn'):
      self.predictions = self.model_fn(instances)
    with tf.variable_scope('error_fn'):
      self.qualities_prior = self.error_fn(instances, predictors)
    self.loss = self.config.loss_fn(
      self.predictions, self.qualities_prior, predictor_indices,
      predictor_values)
    self.train_op = self.optimizer.minimize(self.loss)
    self.init_op = tf.global_variables_initializer()

  def _init_session(self):
    if self._session is None:
      self._session = tf.Session()
      self._session.run(self.init_op)

  def train(self, dataset):
    self._init_session()
    iterator_init_op = self.train_iterator.make_initializer(dataset)
    self._session.run(iterator_init_op)
    for step in range(10000):
      loss, _ = self._session.run([self.loss, self.train_op])
      if step % 100 == 0:
        logger.info('Step: %5d | Loss: %.4f' % (step, loss))

  def predict(self, instances):
    self._init_session()
    return self._session.run(
      self.predictions,
      feed_dict={self.instances: instances})

  def qualities(self, instances, predictor_indices):
    self._init_session()
    gamma = self._session.run(
      self.qualities_prior,
      feed_dict={
        self.instances: instances,
        self.predictor_indices: predictor_indices})
    alpha = np.exp(gamma)
    beta = 1 + alpha
    return alpha / (alpha + beta)


class NoisyMLPLearner(NoisyLearner):
  def __init__(
      self, inputs_size, predictors_size,
      config, optimizer,
      model_hidden_units, error_hidden_units,
      instances_input_fn=lambda x: x,
      predictors_input_fn=lambda x: x,
      model_activation=tf.nn.leaky_relu,
      error_activation=tf.nn.leaky_relu,
      feed_inputs_to_error_fn=True):
    self.model_hidden_units = model_hidden_units
    self.error_hidden_units = error_hidden_units
    self.model_activation = model_activation
    self.error_activation = error_activation
    self.feed_inputs_to_error_fn = feed_inputs_to_error_fn
    super(NoisyMLPLearner, self).__init__(
      inputs_size=inputs_size,
      predictors_size=predictors_size,
      config=config,
      optimizer=optimizer,
      instances_input_fn=instances_input_fn,
      predictors_input_fn=predictors_input_fn)

  def model_fn(self, instances):
    w_initializer = tf.glorot_uniform_initializer(
      dtype=instances.dtype)
    b_initializer = tf.zeros_initializer(
      dtype=instances.dtype)
    hidden = instances
    layers = enumerate(self.model_hidden_units + [1])
    for i, num_units in layers:
      with tf.variable_scope('layer_{}'.format(i)):
        w = tf.get_variable(
          name='weights',
          shape=[hidden.shape[-1], num_units],
          initializer=w_initializer)
        b = tf.get_variable(
          name='bias',
          shape=[num_units],
          initializer=b_initializer)
        hidden = tf.nn.xw_plus_b(
          x=hidden, weights=w, biases=b, name='linear')
        if i < len(self.model_hidden_units):
          hidden = self.model_activation(hidden)
    return tf.sigmoid(hidden)

  def error_fn(self, instances, predictors):
    w_initializer = tf.glorot_uniform_initializer(
      dtype=instances.dtype)
    b_initializer = tf.zeros_initializer(
      dtype=instances.dtype)
    if self.feed_inputs_to_error_fn:
      instances = tf.tile(instances[:, None, :], [1, tf.shape(predictors)[1], 1])
      hidden = tf.concat([predictors, instances], axis=2)
    else:
      hidden = predictors
    layers = enumerate(self.error_hidden_units + [2])
    for i, num_units in layers:
      with tf.variable_scope('layer_{}'.format(i)):
        w = tf.get_variable(
          name='weights',
          shape=[hidden.shape[-1], num_units],
          initializer=w_initializer)
        b = tf.get_variable(
          name='bias',
          shape=[num_units],
          initializer=b_initializer)
        hidden = tf.tensordot(
          hidden, w, axes=[[2], [0]], name='linear')
        hidden += b
        if i < len(self.error_hidden_units) - 1:
          hidden = self.error_activation(hidden)
    return hidden
