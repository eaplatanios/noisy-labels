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

from .transformations import *

__author__ = 'eaplatanios'

__all__ = [
  'NoisyLearnerConfig', 'BinaryNoisyLearnerConfig',
  'NoisyLearner']

logger = logging.getLogger(__name__)


class NoisyLearnerConfig(with_metaclass(abc.ABCMeta, object)):
  def __init__(self, warm_up_steps=1000, eps=1e-12):
    self.warm_up_steps = warm_up_steps
    self.eps = eps

  @abc.abstractmethod
  def num_outputs(self):
    pass

  @abc.abstractmethod
  def qualities_mean(self, *args):
    pass

  @abc.abstractmethod
  def loss_fn(
      self, predictions, qualities_prior,
      predictor_values):
    pass


class BinaryNoisyLearnerConfig(NoisyLearnerConfig):
  def __init__(
      self, model_fn, qualities_fn, prior_correct=0.99,
      max_param_value=1e6, warm_up_steps=1000, eps=1e-12):
    super(BinaryNoisyLearnerConfig, self).__init__(
      warm_up_steps, eps)
    self.model_fn = model_fn
    self.qualities_fn = qualities_fn
    self.prior_correct = prior_correct
    self.max_param_value = max_param_value
    self.alpha = None
    self.beta = None

  def num_outputs(self):
    return 1

  def qualities_mean(self, alpha, beta):
    return alpha / (alpha + beta)

  def loss_fn(
      self, predictions, qualities_prior,
      predictor_values):
    predictions = tf.squeeze(predictions, axis=-1)
    predictor_values = tf.squeeze(predictor_values, axis=-1)
    p = predictions
    a = 1.0 + tf.exp(qualities_prior[:, :, 0])
    b = 1.0 + tf.exp(qualities_prior[:, :, 1])
    a = tf.minimum(a, self.max_param_value)
    b = tf.minimum(b, self.max_param_value)
    self.alpha, self.beta = a, b

    # Main Likelihood Terms
    y_hat = tf.reshape(predictor_values, [-1, 1])
    ab = tf.log(tf.stack([a, b], axis=-1))
    ab = tf.reshape(ab, [-1, 2])
    ab_log = tf.log(a + b)
    term1 = tf.batch_gather(ab, 1 - y_hat)
    term1 = tf.reshape(term1, tf.shape(predictor_values)) - ab_log
    term1 = tf.reduce_sum(term1, axis=1)
    term0 = tf.batch_gather(ab, y_hat)
    term0 = tf.reshape(term0, tf.shape(predictor_values)) - ab_log
    term0 = tf.reduce_sum(term0, axis=1)

    # Symmetry-Breaking Prior
    y_1 = tf.cast(predictor_values, tf.float32)
    y_0 = 1 - y_1
    global_step = tf.train.get_or_create_global_step()
    p_prior = tf.cond(global_step < self.warm_up_steps, lambda: 1.0, lambda: 0.0)
    p_correct = self.prior_correct
    prior_term1 = tf.reduce_sum(tf.log(y_1 * p_correct + y_0 * (1 - p_correct)), axis=1)
    prior_term0 = tf.reduce_sum(tf.log(y_0 * p_correct + y_1 * (1 - p_correct)), axis=1)

    # Combine all the likelihood terms
    term1 += tf.log((1 - p_prior) * p + self.eps)
    term0 += tf.log((1 - p_prior) * (1 - p) + self.eps)
    prior_term1 += tf.log(p_prior * p + self.eps)
    prior_term0 += tf.log(p_prior * (1 - p) + self.eps)
    eps = tf.zeros_like(term1) + np.log(self.eps)
    loss = tf.reduce_logsumexp(tf.stack([term1, term0, prior_term1, prior_term0, eps], axis=0), axis=0)

    return -tf.reduce_mean(loss)


class NoisyLearner(object):
  def __init__(
      self, instances_input_size, predictors_input_size,
      config, phase_one_optimizer, phase_two_optimizer,
      instances_dtype=tf.int32,
      predictors_dtype=tf.int32,
      instances_input_fn=lambda x: x,
      predictors_input_fn=lambda x: x,
      qualities_input_fn=InstancesPredictorsConcatenation()):
    self.instances_input_size = instances_input_size
    self.predictors_input_size = predictors_input_size
    self.config = config
    self.phase_one_optimizer = phase_one_optimizer
    self.phase_two_optimizer = phase_two_optimizer
    self.instances_dtype = instances_dtype
    self.predictors_dtype = predictors_dtype
    self.instances_input_fn = instances_input_fn
    self.predictors_input_fn = predictors_input_fn
    self.qualities_input_fn = qualities_input_fn
    self._build_model()
    self._session = None

  def _build_iterators(self):
    self.train_iterator = tf.data.Iterator.from_structure(
      output_types={
        'instances': self.instances_dtype,
        'predictors': self.predictors_dtype,
        'predictor_values': tf.int32},
      output_shapes={
        'instances': [None, self.instances_input_size],
        'predictors': [None, None, self.predictors_input_size],
        'predictor_values': [None, None, self.config.num_outputs()]},
      shared_name='train_iterator')
    iter_next = self.train_iterator.get_next()
    self.instances = tf.placeholder_with_default(
      iter_next['instances'],
      shape=[None, self.instances_input_size],
      name='instances')
    self.predictors = tf.placeholder_with_default(
      iter_next['predictors'],
      shape=[None, None, self.predictors_input_size],
      name='predictors')
    self.predictor_values = tf.placeholder_with_default(
      iter_next['predictor_values'],
      shape=[None, None, self.config.num_outputs()],
      name='predictor_values')

  def _build_model(self):
    self._build_iterators()
    instances = self.instances
    predictors = self.predictors
    predictor_values = self.predictor_values
    instances = self.instances_input_fn(instances)
    predictors = self.predictors_input_fn(predictors)
    self.predictions = self.config.model_fn(instances)
    self.qualities_prior = self.config.qualities_fn(
      self.qualities_input_fn(instances, predictors))
    self.loss = self.config.loss_fn(
      self.predictions, self.qualities_prior,
      predictor_values)
    global_step = tf.train.get_or_create_global_step()
    gradients, variables = zip(
      *self.phase_one_optimizer.compute_gradients(self.loss))
    # gradients, _ = tf.clip_by_global_norm(gradients, 100.0)
    self.phase_one_train_op = self.phase_one_optimizer.apply_gradients(
      zip(gradients, variables), global_step=global_step)
    self.phase_two_train_op = self.phase_two_optimizer.apply_gradients(
      zip(gradients, variables), global_step=global_step)
    self.init_op = tf.global_variables_initializer()

  def _init_session(self):
    if self._session is None:
      self._session = tf.Session()
      self._session.run(self.init_op)

  def train(
      self, dataset, max_steps=1000,
      loss_abs_threshold=None, min_steps_below_threshold=10,
      log_steps=100):
    self._init_session()
    iterator_init_op = self.train_iterator.make_initializer(dataset)
    self._session.run(iterator_init_op)
    loss_accumulator = 0.0
    accumulator_steps = 0
    steps_below_threshold = 0
    for step in range(max_steps):
      if step < self.config.warm_up_steps:
        loss, _ = self._session.run([self.loss, self.phase_one_train_op])
      else:
        loss, _ = self._session.run([self.loss, self.phase_two_train_op])
      loss_accumulator += loss
      accumulator_steps += 1
      if step % log_steps == 0 or step == max_steps - 1:
        loss = loss_accumulator / accumulator_steps
        logger.info('Step: %5d | Loss: %.8f' % (step, loss))
        loss_accumulator = 0.0
        accumulator_steps = 0
      if loss_abs_threshold is not None:
        if loss < loss_abs_threshold:
          steps_below_threshold += 1
          if steps_below_threshold >= min_steps_below_threshold:
            logger.info('Converged.')
            break
        else:
          steps_below_threshold = 0

  def predict(self, instances):
    self._init_session()
    return self._session.run(
      self.predictions,
      feed_dict={self.instances: instances})

  def qualities(self, instances, predictor_indices):
    self._init_session()
    qualities_prior = self._session.run(
      (self.config.alpha, self.config.beta),
      feed_dict={
        self.instances: instances,
        self.predictors: predictor_indices})
    return self.config.qualities_mean(*qualities_prior)
