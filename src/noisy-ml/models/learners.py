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

from six import with_metaclass

from .transformations import *

__author__ = 'eaplatanios'

__all__ = ['BinaryEMConfig', 'EMLearner']

logger = logging.getLogger(__name__)


class EMConfig(with_metaclass(abc.ABCMeta, object)):
  @abc.abstractmethod
  def num_outputs(self):
    pass

  @abc.abstractmethod
  def qualities_mean(self, *args):
    pass

  @abc.abstractmethod
  def build_ops(self, x_indices, h_1_log, q_params, y_hat, y_hat_soft):
    pass


class BinaryEMConfig(EMConfig):
  def __init__(
      self, num_instances, num_predictors,
      model_fn, qualities_fn, optimizer,
      use_soft_maj=True, use_soft_y_hat=False,
      max_param_value=None):
    super(BinaryEMConfig, self).__init__()
    self.num_instances = num_instances
    self.num_predictors = num_predictors
    self.model_fn = model_fn
    self.qualities_fn = qualities_fn
    self.optimizer = optimizer
    self.use_soft_maj = use_soft_maj
    self.use_soft_y_hat = use_soft_y_hat
    self.max_param_value = max_param_value

  def num_outputs(self):
    return 1

  def qualities_mean(self, alpha, beta):
    return alpha / (alpha + beta)

  def build_ops(self, x_indices, h_1_log, q_params, y_hat_1, y_hat_1_soft):
    # x_indices has shape [N]
    # h_1_log has shape [N, 1]
    # q_params has shape [N, M, 2]
    # y_hat has shape [N, M, 1]
    # y_hat_1_soft has shape [N, M, 1]
    h_1_log = tf.squeeze(h_1_log, axis=-1)
    y_hat_1 = tf.squeeze(y_hat_1, axis=-1)
    y_hat_1_soft = tf.squeeze(y_hat_1_soft, axis=-1)

    temp = h_1_log
    h_1_log = tf.log_sigmoid(temp)
    h_0_log = -temp + h_1_log
    y_hat_0 = 1 - y_hat_1
    y_hat_0_soft = 1 - y_hat_1_soft

    a_log = q_params[:, :, 0]
    b_log = q_params[:, :, 1]
    if self.max_param_value is not None:
      a_log = tf.minimum(a_log, tf.log(self.max_param_value))
      b_log = tf.minimum(b_log, tf.log(self.max_param_value))
    a = 1 + tf.exp(a_log)
    b = 1 + tf.exp(b_log)
    ab_log = tf.stack([a_log, b_log], axis=-1)
    a_plus_b_log = tf.log(a + b)

    # E-Step:

    # Create the accumulator variables.
    e_y_lambda_1_log = tf.get_variable(
      name='e_y_lambda_1_log',
      shape=[self.num_instances],
      initializer=tf.zeros_initializer(h_1_log.dtype),
      trainable=False)
    e_y_lambda_0_log = tf.get_variable(
      name='e_y_lambda_0_log',
      shape=[self.num_instances],
      initializer=tf.zeros_initializer(h_1_log.dtype),
      trainable=False)

    def y_lambda_log(k):
      lambda_log = h_1_log if k == 1 else h_0_log
      lambda_log = lambda_log[:, None]
      lambda_log += tf.squeeze(tf.batch_gather(
        ab_log,
        tf.expand_dims(y_hat_0 if k == 1 else y_hat_1, axis=-1)),
        axis=-1)
      # lambda_log -= a_plus_b_log
      return tf.reduce_sum(lambda_log, axis=1)

    # Boolean flag about whether or not to use majority vote
    # estimates for the E-step expectations.
    use_maj = tf.get_variable(
      name='use_maj',
      shape=[],
      dtype=tf.bool,
      initializer=tf.zeros_initializer(),
      trainable=False)
    set_use_maj = use_maj.assign(True)
    unset_use_maj = use_maj.assign(False)
    use_maj = tf.cast(use_maj, tf.float32)

    if self.use_soft_maj:
      e_y_1_maj = tf.reduce_mean(y_hat_1_soft, axis=1)
      e_y_0_maj = tf.reduce_mean(y_hat_0_soft, axis=1)
    else:
      e_y_1_maj = tf.reduce_mean(tf.cast(y_hat_1, tf.float32), axis=1)
      e_y_0_maj = tf.reduce_mean(tf.cast(y_hat_0, tf.float32), axis=1)

    y_lambda_1_log = use_maj * e_y_1_maj + (1 - use_maj) * y_lambda_log(1)
    y_lambda_0_log = use_maj * e_y_0_maj + (1 - use_maj) * y_lambda_log(0)

    # Create the accumulator variable update ops.
    e_y_lambda_1_log_update = e_y_lambda_1_log \
      .scatter_add(tf.IndexedSlices(
      indices=x_indices,
      values=y_lambda_1_log))
    e_y_lambda_0_log_update = e_y_lambda_0_log \
      .scatter_add(tf.IndexedSlices(
      indices=x_indices,
      values=y_lambda_0_log))

    e_step_init = tf.group([
      e_y_lambda_1_log.initializer,
      e_y_lambda_0_log.initializer],
      name='e_step_init')
    e_step = tf.group([
      e_y_lambda_1_log_update,
      e_y_lambda_0_log_update],
      name='e_step')

    # M-Step:

    # Compute E[I(y=1)] and E[I(y=0)] that is used
    # in the log-likelihood function.

    def expected_y(k):
      if k == 1:
        accumulated = e_y_lambda_1_log
      else:
        accumulated = e_y_lambda_0_log
      estimated = accumulated - tf.reduce_logsumexp(
        tf.stack([
          e_y_lambda_1_log,
          e_y_lambda_0_log], axis=0),
        axis=0)
      estimated = tf.exp(estimated)
      maj = accumulated
      return use_maj * maj + (1 - use_maj) * estimated

    e_y_1 = expected_y(1)
    e_y_0 = expected_y(0)
    e_y_1 = tf.gather(e_y_1, x_indices)
    e_y_0 = tf.gather(e_y_0, x_indices)

    ll_term0 = e_y_1 * h_1_log + e_y_0 * h_0_log
    ll_term0 = tf.reduce_sum(ll_term0)

    e_y_1 = tf.expand_dims(e_y_1, axis=-1)
    e_y_0 = tf.expand_dims(e_y_0, axis=-1)

    if self.use_soft_y_hat:
      e_y_y_hat_1 = e_y_1 * y_hat_1_soft + e_y_0 * y_hat_0_soft
      e_y_y_hat_0 = e_y_1 * y_hat_0_soft + e_y_0 * y_hat_1_soft
      ll_term1 = e_y_y_hat_1 * a_log + e_y_y_hat_0 * b_log
    else:
      ll_term1 = e_y_1 * tf.squeeze(tf.batch_gather(
        ab_log,
        tf.expand_dims(y_hat_0, axis=-1)),
        axis=-1)
      ll_term1 += e_y_0 * tf.squeeze(tf.batch_gather(
        ab_log,
        tf.expand_dims(y_hat_1, axis=-1)),
        axis=-1)

    ll_term1 -= a_plus_b_log
    ll_term1 = tf.reduce_sum(ll_term1)

    # We are omitting the last term because it is constant
    # with respect to the parameters of h and g.
    neg_log_likelihood = -ll_term0 - ll_term1

    # M-step:
    m_step_init = tf.variables_initializer(
      tf.trainable_variables())
    global_step = tf.train.get_or_create_global_step()
    gvs = self.optimizer.compute_gradients(neg_log_likelihood)
    gradients, variables = zip(*gvs)
    # gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    m_step = self.optimizer.apply_gradients(
      zip(gradients, variables), global_step=global_step)

    check = tf.add_check_numerics_ops()

    return {
      'h_1_log': h_1_log,
      'alpha': a,
      'beta': b,
      'set_use_maj': set_use_maj,
      'unset_use_maj': unset_use_maj,
      'e_step_accumulators': {
        'e_y_lambda_1_log': e_y_lambda_1_log,
        'e_y_lambda_0_log': e_y_lambda_0_log},
      'e_step_init': e_step_init,
      'm_step_init': m_step_init,
      'e_step': e_step,
      'm_step': m_step,
      'neg_log_likelihood': neg_log_likelihood,
      'check_numerics': check}


class EMLearner(object):
  def __init__(
      self, config,
      instances_input_fn=lambda x: x,
      predictors_input_fn=lambda x: x,
      qualities_input_fn=InstancesPredictorsConcatenation()):
    self.config = config
    self.instances_input_fn = instances_input_fn
    self.predictors_input_fn = predictors_input_fn
    self.qualities_input_fn = qualities_input_fn
    self._build_model()
    self._session = None

  def _build_iterators(self):
    self.train_iterator = tf.data.Iterator.from_structure(
      output_types={
        'instances': tf.int32,
        'predictors': tf.int32,
        'predictor_values': tf.int32,
        'predictor_values_soft': tf.float32},
      output_shapes={
        'instances': [None],
        'predictors': [None, None],
        'predictor_values': [None, None, self.config.num_outputs()],
        'predictor_values_soft': [None, None, self.config.num_outputs()]},
      shared_name='train_iterator')
    iter_next = self.train_iterator.get_next()
    self.instances = tf.placeholder_with_default(
      iter_next['instances'],
      shape=[None],
      name='instances')
    self.predictors = tf.placeholder_with_default(
      iter_next['predictors'],
      shape=[None, None],
      name='predictors')
    self.predictor_values = tf.placeholder_with_default(
      iter_next['predictor_values'],
      shape=[None, None, self.config.num_outputs()],
      name='predictor_values')
    self.predictor_values_soft = tf.placeholder_with_default(
      iter_next['predictor_values_soft'],
      shape=[None, None, self.config.num_outputs()],
      name='predictor_values_soft')

  def _build_model(self):
    self._build_iterators()
    instances = self.instances
    predictors = self.predictors
    instance_features = self.instances_input_fn(instances)
    predictors = self.predictors_input_fn(predictors)
    h_1_log = self.config.model_fn(instance_features)
    self.qualities_prior = self.config.qualities_fn(
      self.qualities_input_fn(instance_features, predictors))
    self._ops = self.config.build_ops(
      x_indices=instances,
      h_1_log=h_1_log,
      q_params=self.qualities_prior,
      y_hat_1=self.predictor_values,
      y_hat_1_soft=self.predictor_values_soft)
    self._init_op = tf.global_variables_initializer()

  def _init_session(self):
    if self._session is None:
      self._session = tf.Session()
      self._session.run(self._init_op)

  def _e_step(self, iterator_init_op, use_maj):
    self._session.run(iterator_init_op)
    self._session.run(self._ops['e_step_init'])
    if use_maj:
      self._session.run(self._ops['set_use_maj'])
    else:
      self._session.run(self._ops['unset_use_maj'])
    while True:
      try:
        self._session.run(self._ops['e_step'])
      except tf.errors.OutOfRangeError:
        break

  def _m_step(
      self, iterator_init_op, warm_start,
      max_m_steps, log_m_steps):
    if not warm_start:
      self._session.run(self._ops['m_step_init'])
    accumulator_ll = 0.0
    accumulator_steps = 0
    m_step = 0
    while m_step < max_m_steps:
      try:
        ll, _ = self._session.run([
          self._ops['neg_log_likelihood'],
          self._ops['m_step']])
        accumulator_ll += ll
        accumulator_steps += 1
        if m_step % log_m_steps == 0 or m_step == max_m_steps - 1:
          ll = accumulator_ll / accumulator_steps
          logger.info(
            'Step: %5d | Negative Log-Likelihood: %.8f'
            % (m_step, ll))
          accumulator_ll = 0.0
          accumulator_steps = 0
        m_step += 1
      except tf.errors.OutOfRangeError:
        self._session.run(iterator_init_op)
        continue

  def train(
      self, dataset, warm_start=False, max_m_steps=1000,
      max_em_steps=100, log_m_steps=100,
      em_step_callback=None):
    self._init_session()
    iterator_init_op = self.train_iterator.make_initializer(dataset)

    for em_step in range(max_em_steps):
      logger.info('Iteration %d - Running E-Step' % em_step)
      self._e_step(iterator_init_op, use_maj=em_step == 0)
      logger.info('Iteration %d - Running M-Step' % em_step)
      self._m_step(
        iterator_init_op, warm_start,
        max_m_steps, log_m_steps)
      if em_step_callback is not None:
        em_step_callback(self)

  def predict(self, instances):
    self._init_session()
    return self._session.run(
      self._ops['h_1_log'],
      feed_dict={self.instances: instances})

  def qualities(self, instances, predictors):
    self._init_session()
    qualities_prior = self._session.run(
      (self._ops['alpha'], self._ops['beta']),
      feed_dict={
        self.instances: instances,
        self.predictors: predictors})
    return self.config.qualities_mean(*qualities_prior)
