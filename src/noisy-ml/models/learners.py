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

from .utilities import log1mexp

__author__ = 'eaplatanios'

__all__ = [
  'EMLearner', 'EMConfig', 'MultiLabelEMConfig',
  'MultiLabelFullConfusionEMConfig',
  'MultiLabelFullConfusionSimpleQEMConfig']

logger = logging.getLogger(__name__)


class EMLearner(object):
  def __init__(
      self, config, predictions_output_fn=lambda x: x):
    self.config = config
    self.predictions_output_fn = predictions_output_fn
    self._build_model()
    self._session = None

  def _build_model(self):
    self._ops = self.config.build_ops()
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
    self._session.run(iterator_init_op)
    if not warm_start:
      self._session.run(self._ops['m_step_init'])
    accumulator_ll = 0.0
    accumulator_steps = 0
    for m_step in range(max_m_steps):
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

  def train(
      self, dataset, batch_size=128,
      warm_start=False, max_m_steps=1000,
      max_em_steps=100, log_m_steps=100,
      em_step_callback=None):
    e_step_dataset = dataset.batch(batch_size)
    m_step_dataset = dataset.repeat().shuffle(10000).batch(batch_size)

    self._init_session()
    e_step_iterator_init_op = self._ops['train_iterator'].make_initializer(e_step_dataset)
    m_step_iterator_init_op = self._ops['train_iterator'].make_initializer(m_step_dataset)

    for em_step in range(max_em_steps):
      logger.info('Iteration %d - Running E-Step' % em_step)
      self._e_step(e_step_iterator_init_op, use_maj=em_step == 0)
      logger.info('Iteration %d - Running M-Step' % em_step)
      self._m_step(
        m_step_iterator_init_op, warm_start,
        max_m_steps, log_m_steps)
      if em_step_callback is not None:
        em_step_callback(self)

  def predict(self, instances, batch_size=128):
    # TODO: Remove hack by having separate train and predict iterators.
    dataset = tf.data.Dataset.from_tensor_slices({
      'instances': instances,
      'predictors': np.zeros([len(instances)], np.int32),
      'labels': np.zeros([len(instances)], np.int32),
      'values': np.zeros([len(instances)], np.float32)}
    ).batch(batch_size)

    self._init_session()

    iterator_init_op = self._ops['train_iterator'].make_initializer(dataset)
    self._session.run(iterator_init_op)
    predictions = []
    while True:
      try:
        predictions.append(self._session.run(
          self._ops['predictions']))
      except tf.errors.OutOfRangeError:
        break
    return np.concatenate(predictions, axis=0)

  def qualities(self, instances, predictors, labels):
    def cartesian_transpose(arrays):
      la = len(arrays)
      dtype = np.result_type(*arrays)
      arr = np.empty([la] + [len(a) for a in arrays], dtype=dtype)
      for i, a in enumerate(np.ix_(*arrays)):
        arr[i, ...] = a
      return arr.reshape(la, -1)

    temp = cartesian_transpose([
      np.array(instances),
      np.array(labels),
      np.array(predictors)])

    self._init_session()
    qualities_mean_log = self._session.run(
      self._ops['qualities_mean_log'],
      feed_dict={
        self._ops['x_indices']: temp[0],
        self._ops['p_indices']: temp[2],
        self._ops['l_indices']: temp[1]})
    qualities = np.exp(qualities_mean_log)
    qualities = np.reshape(
      qualities,
      [len(instances), len(labels), len(predictors)])
    return qualities


class EMConfig(with_metaclass(abc.ABCMeta, object)):
  @abc.abstractmethod
  def build_ops(self):
    raise NotImplementedError


class MultiLabelEMConfig(EMConfig):
  def __init__(
      self, num_instances, num_predictors, num_labels,
      model, optimizer, use_soft_maj=True,
      use_soft_y_hat=False):
    super(MultiLabelEMConfig, self).__init__()
    self.num_instances = num_instances
    self.num_predictors = num_predictors
    self.num_labels = num_labels
    self.model = model
    self.optimizer = optimizer
    self.use_soft_maj = use_soft_maj
    self.use_soft_y_hat = use_soft_y_hat

  def build_ops(self):
    train_iterator = tf.data.Iterator.from_structure(
      output_types={
        'instances': tf.int32,
        'predictors': tf.int32,
        'labels': tf.int32,
        'values': tf.float32},
      output_shapes={
        'instances': [None],
        'predictors': [None],
        'labels': [None],
        'values': [None]},
      shared_name='train_iterator')
    iter_next = train_iterator.get_next()
    instances = tf.placeholder_with_default(
      iter_next['instances'],
      shape=[None],
      name='instances')
    predictors = tf.placeholder_with_default(
      iter_next['predictors'],
      shape=[None],
      name='predictors')
    labels = tf.placeholder_with_default(
      iter_next['labels'],
      shape=[None],
      name='labels')
    values = tf.placeholder_with_default(
      iter_next['values'],
      shape=[None],
      name='values')

    x_indices = instances
    p_indices = predictors
    l_indices = labels

    predictions, q_params = self.model.build(
      x_indices, p_indices, l_indices)

    # y_hat_1 has shape: [BatchSize]
    # y_hat_0 has shape: [BatchSize]
    y_hat_1_soft = values
    y_hat_0_soft = 1 - y_hat_1_soft
    y_hat_1 = tf.cast(tf.greater_equal(values, 0.5), tf.int32)
    y_hat_0 = 1 - y_hat_1

    # TODO: Is this necessary?
    h_1_log = tf.minimum(predictions, -1e-6) # predictions
    h_0_log = log1mexp(h_1_log)

    indices = tf.expand_dims(l_indices, axis=-1)
    h_1_log = tf.squeeze(tf.batch_gather(
      params=h_1_log,
      indices=indices), axis=-1)
    h_0_log = tf.squeeze(tf.batch_gather(
      params=h_0_log,
      indices=indices), axis=-1)

    # q_params shape: [BatchSize, 2]
    a_log = q_params[:, 0]
    b_log = q_params[:, 1]
    a = tf.exp(a_log)
    b = tf.exp(b_log)
    ab_log = tf.stack([a_log, b_log], axis=-1)
    a_plus_b_log = tf.reduce_logsumexp(ab_log, axis=-1)
    qualities_mean_log = a_log - a_plus_b_log

    # E-step:

    with tf.name_scope('e_step'):
      # Create the accumulator variables:
      e_y_lambda_1_log = tf.get_variable(
        name='e_y_lambda_1_log',
        shape=[self.num_instances, self.num_labels],
        initializer=tf.zeros_initializer(h_1_log.dtype),
        trainable=False)
      e_y_lambda_0_log = tf.get_variable(
        name='e_y_lambda_0_log',
        shape=[self.num_instances, self.num_labels],
        initializer=tf.zeros_initializer(h_1_log.dtype),
        trainable=False)

      def y_lambda_log(k):
        i = y_hat_0 if k == 1 else y_hat_1
        lambda_log = h_1_log if k == 1 else h_0_log
        lambda_log += tf.squeeze(tf.batch_gather(
          params=ab_log,
          indices=tf.expand_dims(i, axis=-1)), axis=-1)
        # lambda_log -= a_plus_b_log
        return lambda_log

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

      if self.use_soft_maj:
        e_y_1_maj = y_hat_1_soft
        e_y_0_maj = y_hat_0_soft
      else:
        e_y_1_maj = tf.cast(y_hat_1, tf.float32)
        e_y_0_maj = tf.cast(y_hat_0, tf.float32)

      y_lambda_1_log = tf.cond(use_maj, lambda: e_y_1_maj, lambda: y_lambda_log(1))
      y_lambda_0_log = tf.cond(use_maj, lambda: e_y_0_maj, lambda: y_lambda_log(0))

      # Create the accumulator variable update ops.
      xl_indices = tf.stack(
        [x_indices, l_indices], axis=-1)
      e_y_lambda_1_log_update = e_y_lambda_1_log \
        .scatter_nd_add(
          indices=xl_indices,
          updates=y_lambda_1_log)
      e_y_lambda_0_log_update = e_y_lambda_0_log \
        .scatter_nd_add(
          indices=xl_indices,
          updates=y_lambda_0_log)

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
      maj = tf.divide(
        accumulated,
        e_y_lambda_1_log + e_y_lambda_0_log)
      return tf.cond(use_maj, lambda: maj, lambda: estimated)

    with tf.name_scope('m_step'):
      e_y_1 = expected_y(1)
      e_y_0 = expected_y(0)
      e_y_1 = tf.gather_nd(e_y_1, xl_indices)
      e_y_0 = tf.gather_nd(e_y_0, xl_indices)

      ll_term0 = e_y_1 * h_1_log + e_y_0 * h_0_log
      ll_term0 = tf.reduce_sum(ll_term0)

      e_y_1 = tf.expand_dims(e_y_1, axis=-1)
      e_y_0 = tf.expand_dims(e_y_0, axis=-1)

      if self.use_soft_y_hat:
        e_y_y_hat_1 = e_y_1 * y_hat_1_soft + e_y_0 * y_hat_0_soft
        e_y_y_hat_0 = e_y_1 * y_hat_0_soft + e_y_0 * y_hat_1_soft
        ll_term1 = e_y_y_hat_1 * a_log + e_y_y_hat_0 * b_log
      else:
        ll_term1 = e_y_1 * tf.batch_gather(
          ab_log,
          tf.expand_dims(y_hat_0, axis=-1))
        ll_term1 += e_y_0 * tf.batch_gather(
          ab_log,
          tf.expand_dims(y_hat_1, axis=-1))
        ll_term1 = tf.squeeze(ll_term1, axis=-1)

      ll_term1 -= a_plus_b_log
      ll_term1 = tf.reduce_sum(ll_term1)

      # ll_term2 = -tf.exp(h_1_log) * h_1_log - tf.exp(h_0_log) * h_0_log
      # ll_term2 = tf.reduce_sum(ll_term2)

      # We are omitting the last term because it is constant
      # with respect to the parameters of h and g.
      neg_log_likelihood = -ll_term0 - ll_term1 # - ll_term2

      m_step_init = tf.variables_initializer(
        tf.trainable_variables())
      global_step = tf.train.get_or_create_global_step()
      gvs = self.optimizer.compute_gradients(neg_log_likelihood)
      gradients, variables = zip(*gvs)
      # gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
      m_step = self.optimizer.apply_gradients(
        zip(gradients, variables), global_step=global_step)

    return {
      'train_iterator': train_iterator,
      'x_indices': x_indices,
      'p_indices': p_indices,
      'l_indices': l_indices,
      'predictions': predictions,
      'alpha': a,
      'beta': b,
      'qualities_mean_log': qualities_mean_log,
      'set_use_maj': set_use_maj,
      'unset_use_maj': unset_use_maj,
      'e_step_accumulators': {
        'e_y_lambda_1_log': e_y_lambda_1_log,
        'e_y_lambda_0_log': e_y_lambda_0_log},
      'e_step_init': e_step_init,
      'm_step_init': m_step_init,
      'e_step': e_step,
      'm_step': m_step,
      'neg_log_likelihood': neg_log_likelihood}


class MultiLabelFullConfusionEMConfig(EMConfig):
  def __init__(
      self, num_instances, num_predictors, num_labels,
      model, optimizer, use_soft_maj=True,
      use_soft_y_hat=False):
    super(MultiLabelFullConfusionEMConfig, self).__init__()
    self.num_instances = num_instances
    self.num_predictors = num_predictors
    self.num_labels = num_labels
    self.model = model
    self.optimizer = optimizer
    self.use_soft_maj = use_soft_maj
    self.use_soft_y_hat = use_soft_y_hat

  def build_ops(self):
    train_iterator = tf.data.Iterator.from_structure(
      output_types={
        'instances': tf.int32,
        'predictors': tf.int32,
        'labels': tf.int32,
        'values': tf.float32},
      output_shapes={
        'instances': [None],
        'predictors': [None],
        'labels': [None],
        'values': [None]},
      shared_name='train_iterator')
    iter_next = train_iterator.get_next()
    instances = tf.placeholder_with_default(
      iter_next['instances'],
      shape=[None],
      name='instances')
    predictors = tf.placeholder_with_default(
      iter_next['predictors'],
      shape=[None],
      name='predictors')
    labels = tf.placeholder_with_default(
      iter_next['labels'],
      shape=[None],
      name='labels')
    values = tf.placeholder_with_default(
      iter_next['values'],
      shape=[None],
      name='values')

    x_indices = instances
    p_indices = predictors
    l_indices = labels

    predictions, q_params = self.model.build(
      x_indices, p_indices, l_indices)

    # y_hat_1 has shape: [BatchSize]
    # y_hat_0 has shape: [BatchSize]
    y_hat_1_soft = values
    y_hat_0_soft = 1 - y_hat_1_soft
    y_hat_1 = tf.cast(tf.greater_equal(values, 0.5), tf.int32)
    y_hat_0 = 1 - y_hat_1

    # TODO: Is this necessary?
    h_1_log = tf.minimum(predictions, -1e-6) # predictions
    h_0_log = log1mexp(h_1_log)

    indices = tf.expand_dims(l_indices, axis=-1)
    h_1_log = tf.squeeze(tf.batch_gather(
      params=h_1_log,
      indices=indices), axis=-1)
    h_0_log = tf.squeeze(tf.batch_gather(
      params=h_0_log,
      indices=indices), axis=-1)

    # q_params shape: [BatchSize, 4]
    a_0_log = q_params[:, 0]
    b_0_log = q_params[:, 1]
    a_1_log = q_params[:, 2]
    b_1_log = q_params[:, 3]
    a_0 = tf.exp(a_0_log)
    b_0 = tf.exp(b_0_log)
    a_1 = tf.exp(a_1_log)
    b_1 = tf.exp(b_1_log)
    ab_0_log = tf.stack([a_0_log, b_0_log], axis=-1)
    ab_1_log = tf.stack([a_1_log, b_1_log], axis=-1)
    a_0_plus_b_0_log = tf.reduce_logsumexp(ab_0_log, axis=-1)
    a_1_plus_b_1_log = tf.reduce_logsumexp(ab_1_log, axis=-1)
    qualities_0_mean_log = a_0_log - a_0_plus_b_0_log
    qualities_1_mean_log = a_1_log - a_1_plus_b_1_log
    qualities_mean_log = tf.stack([qualities_0_mean_log, qualities_1_mean_log], axis=-1)
    qualities_mean_log = tf.reduce_logsumexp(qualities_mean_log, axis=-1)
    qualities_mean_log -= np.log(2)

    # E-step:

    with tf.name_scope('e_step'):
      # Create the accumulator variables:
      e_y_lambda_1_log = tf.get_variable(
        name='e_y_lambda_1_log',
        shape=[self.num_instances, self.num_labels],
        initializer=tf.zeros_initializer(h_1_log.dtype),
        trainable=False)
      e_y_lambda_0_log = tf.get_variable(
        name='e_y_lambda_0_log',
        shape=[self.num_instances, self.num_labels],
        initializer=tf.zeros_initializer(h_1_log.dtype),
        trainable=False)

      def y_lambda_log(k):
        i = y_hat_0 if k == 1 else y_hat_1
        ab_log = ab_1_log if k == 1 else ab_0_log
        lambda_log = h_1_log if k == 1 else h_0_log
        lambda_log += tf.squeeze(tf.batch_gather(
          params=ab_log,
          indices=tf.expand_dims(i, axis=-1)), axis=-1)
        return lambda_log

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

      if self.use_soft_maj:
        e_y_1_maj = y_hat_1_soft
        e_y_0_maj = y_hat_0_soft
      else:
        e_y_1_maj = tf.cast(y_hat_1, tf.float32)
        e_y_0_maj = tf.cast(y_hat_0, tf.float32)

      y_lambda_1_log = tf.cond(use_maj, lambda: e_y_1_maj, lambda: y_lambda_log(1))
      y_lambda_0_log = tf.cond(use_maj, lambda: e_y_0_maj, lambda: y_lambda_log(0))

      # Create the accumulator variable update ops.
      xl_indices = tf.stack(
        [x_indices, l_indices], axis=-1)
      e_y_lambda_1_log_update = e_y_lambda_1_log \
        .scatter_nd_add(
        indices=xl_indices,
        updates=y_lambda_1_log)
      e_y_lambda_0_log_update = e_y_lambda_0_log \
        .scatter_nd_add(
        indices=xl_indices,
        updates=y_lambda_0_log)

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
      maj = tf.divide(
        accumulated,
        e_y_lambda_1_log + e_y_lambda_0_log)
      return tf.cond(use_maj, lambda: maj, lambda: estimated)

    with tf.name_scope('m_step'):
      e_y_1 = expected_y(1)
      e_y_0 = expected_y(0)
      e_y_1 = tf.gather_nd(e_y_1, xl_indices)
      e_y_0 = tf.gather_nd(e_y_0, xl_indices)

      ll_term0 = e_y_1 * h_1_log + e_y_0 * h_0_log
      ll_term0 = tf.reduce_sum(ll_term0)

      e_y_1 = tf.expand_dims(e_y_1, axis=-1)
      e_y_0 = tf.expand_dims(e_y_0, axis=-1)

      if self.use_soft_y_hat:
        y_1_term = y_hat_1_soft * a_1_log + y_hat_0_soft * b_1_log
        y_1_term -= tf.expand_dims(
          a_1_plus_b_1_log, axis=-1)
        y_0_term = y_hat_0_soft * a_0_log + y_hat_1_soft * b_0_log
        y_0_term -= tf.expand_dims(
          a_0_plus_b_0_log, axis=-1)
        ll_term1 = e_y_1 * y_1_term + e_y_0 * y_0_term
      else:
        y_1_term = tf.batch_gather(
          ab_1_log,
          tf.expand_dims(y_hat_0, axis=-1))
        y_1_term -= tf.expand_dims(
          a_1_plus_b_1_log, axis=-1)
        y_0_term = tf.batch_gather(
          ab_0_log,
          tf.expand_dims(y_hat_1, axis=-1))
        y_0_term -= tf.expand_dims(
          a_0_plus_b_0_log, axis=-1)
        ll_term1 = e_y_1 * y_1_term + e_y_0 * y_0_term

      ll_term1 = tf.reduce_sum(ll_term1)

      # We are omitting the last term because it is constant
      # with respect to the parameters of h and g.
      neg_log_likelihood = -ll_term0 - ll_term1

      m_step_init = tf.variables_initializer(
        tf.trainable_variables())
      global_step = tf.train.get_or_create_global_step()
      gvs = self.optimizer.compute_gradients(neg_log_likelihood)
      gradients, variables = zip(*gvs)
      # gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
      m_step = self.optimizer.apply_gradients(
        zip(gradients, variables), global_step=global_step)

    return {
      'train_iterator': train_iterator,
      'x_indices': x_indices,
      'p_indices': p_indices,
      'l_indices': l_indices,
      'predictions': predictions,
      'alpha_0': a_0,
      'beta_0': b_0,
      'alpha_1': a_1,
      'beta_1': b_1,
      'qualities_mean_log': qualities_mean_log,
      'set_use_maj': set_use_maj,
      'unset_use_maj': unset_use_maj,
      'e_step_accumulators': {
        'e_y_lambda_1_log': e_y_lambda_1_log,
        'e_y_lambda_0_log': e_y_lambda_0_log},
      'e_step_init': e_step_init,
      'm_step_init': m_step_init,
      'e_step': e_step,
      'm_step': m_step,
      'neg_log_likelihood': neg_log_likelihood}


class MultiLabelFullConfusionSimpleQEMConfig(EMConfig):
  def __init__(
      self, num_instances, num_predictors, num_labels,
      model, optimizer, use_soft_maj=True,
      use_soft_y_hat=False):
    super(MultiLabelFullConfusionSimpleQEMConfig, self).__init__()
    self.num_instances = num_instances
    self.num_predictors = num_predictors
    self.num_labels = num_labels
    self.model = model
    self.optimizer = optimizer
    self.use_soft_maj = use_soft_maj
    self.use_soft_y_hat = use_soft_y_hat

  def build_ops(self):
    train_iterator = tf.data.Iterator.from_structure(
      output_types={
        'instances': tf.int32,
        'predictors': tf.int32,
        'labels': tf.int32,
        'values': tf.float32},
      output_shapes={
        'instances': [None],
        'predictors': [None],
        'labels': [None],
        'values': [None]},
      shared_name='train_iterator')
    iter_next = train_iterator.get_next()
    instances = tf.placeholder_with_default(
      iter_next['instances'],
      shape=[None],
      name='instances')
    predictors = tf.placeholder_with_default(
      iter_next['predictors'],
      shape=[None],
      name='predictors')
    labels = tf.placeholder_with_default(
      iter_next['labels'],
      shape=[None],
      name='labels')
    values = tf.placeholder_with_default(
      iter_next['values'],
      shape=[None],
      name='values')

    x_indices = instances
    p_indices = predictors
    l_indices = labels

    predictions, q_params, regularization_terms = self.model.build(
      x_indices, p_indices, l_indices)

    # y_hat_1 has shape: [BatchSize]
    # y_hat_0 has shape: [BatchSize]
    y_hat_1_soft = values
    y_hat_0_soft = 1 - y_hat_1_soft
    y_hat_1 = tf.cast(tf.greater_equal(values, 0.5), tf.int32)
    y_hat_0 = 1 - y_hat_1

    # TODO: Is this necessary?
    h_1_log = tf.minimum(predictions, -1e-6) # predictions
    h_0_log = log1mexp(h_1_log)

    indices = tf.expand_dims(l_indices, axis=-1)
    h_1_log = tf.squeeze(tf.batch_gather(
      params=h_1_log,
      indices=indices), axis=-1)
    h_0_log = tf.squeeze(tf.batch_gather(
      params=h_0_log,
      indices=indices), axis=-1)

    # q_params shape: [BatchSize, 2, 2]
    q_log = q_params
    qualities_mean_log = tf.stack([
      h_1_log + q_log[:, 1, 1],
      h_0_log + q_log[:, 0, 0]], axis=-1)
    qualities_mean_log = tf.reduce_logsumexp(qualities_mean_log, axis=-1)

    # E-Step:

    with tf.name_scope('e_step'):
      # Create the accumulator variables:
      e_y_lambda_1_log = tf.get_variable(
        name='e_y_lambda_1_log',
        shape=[self.num_instances, self.num_labels],
        initializer=tf.zeros_initializer(h_1_log.dtype),
        trainable=False)
      e_y_lambda_0_log = tf.get_variable(
        name='e_y_lambda_0_log',
        shape=[self.num_instances, self.num_labels],
        initializer=tf.zeros_initializer(h_1_log.dtype),
        trainable=False)

      def y_lambda_log(k):
        i = y_hat_1 if k == 1 else y_hat_0
        lambda_log = h_1_log if k == 1 else h_0_log
        lambda_log += tf.squeeze(tf.batch_gather(
          params=q_log[:, k, :],
          indices=tf.expand_dims(i, axis=-1)), axis=-1)
        return lambda_log

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

      if self.use_soft_maj:
        e_y_1_maj = y_hat_1_soft
        e_y_0_maj = y_hat_0_soft
      else:
        e_y_1_maj = tf.cast(y_hat_1, tf.float32)
        e_y_0_maj = tf.cast(y_hat_0, tf.float32)

      y_lambda_1_log = tf.cond(use_maj, lambda: e_y_1_maj, lambda: y_lambda_log(1))
      y_lambda_0_log = tf.cond(use_maj, lambda: e_y_0_maj, lambda: y_lambda_log(0))

      # Create the accumulator variable update ops.
      xl_indices = tf.stack(
        [x_indices, l_indices], axis=-1)
      e_y_lambda_1_log_update = e_y_lambda_1_log \
        .scatter_nd_add(
        indices=xl_indices,
        updates=y_lambda_1_log)
      e_y_lambda_0_log_update = e_y_lambda_0_log \
        .scatter_nd_add(
        indices=xl_indices,
        updates=y_lambda_0_log)

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
      maj = tf.divide(
        accumulated,
        e_y_lambda_1_log + e_y_lambda_0_log)
      return tf.cond(use_maj, lambda: maj, lambda: estimated)

    with tf.name_scope('m_step'):
      e_y_1 = expected_y(1)
      e_y_0 = expected_y(0)
      e_y_1 = tf.gather_nd(e_y_1, xl_indices)
      e_y_0 = tf.gather_nd(e_y_0, xl_indices)

      ll_term0 = e_y_1 * h_1_log + e_y_0 * h_0_log
      ll_term0 = tf.reduce_sum(ll_term0)

      e_y_1 = tf.expand_dims(e_y_1, axis=-1)
      e_y_0 = tf.expand_dims(e_y_0, axis=-1)

      if self.use_soft_y_hat:
        y_1_term = y_hat_1_soft * q_log[:, 1, 1] + y_hat_0_soft * q_log[:, 1, 0]
        y_0_term = y_hat_0_soft * q_log[:, 0, 0] + y_hat_1_soft * q_log[:, 0, 1]
        ll_term1 = e_y_1 * y_1_term + e_y_0 * y_0_term
      else:
        y_hat_1 = tf.expand_dims(y_hat_1, axis=-1)
        y_1_term = e_y_1 * tf.batch_gather(
          q_log[:, 1, :], y_hat_1)
        y_0_term = e_y_0 * tf.batch_gather(
          q_log[:, 0, :], y_hat_1)
        ll_term1 = y_1_term + y_0_term

      ll_term1 = tf.reduce_sum(ll_term1)

      # We are omitting the last term because it is constant
      # with respect to the parameters of h and g.
      neg_log_likelihood = -ll_term0 - ll_term1
      if len(regularization_terms) > 0:
        neg_log_likelihood += tf.add_n(regularization_terms)

      m_step_init = tf.variables_initializer(
        tf.trainable_variables())
      global_step = tf.train.get_or_create_global_step()
      gvs = self.optimizer.compute_gradients(neg_log_likelihood)
      gradients, variables = zip(*gvs)
      # gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
      m_step = self.optimizer.apply_gradients(
        zip(gradients, variables), global_step=global_step)

    return {
      'train_iterator': train_iterator,
      'x_indices': x_indices,
      'p_indices': p_indices,
      'l_indices': l_indices,
      'predictions': predictions,
      'qualities_mean_log': qualities_mean_log,
      'set_use_maj': set_use_maj,
      'unset_use_maj': unset_use_maj,
      'e_step_accumulators': {
        'e_y_lambda_1_log': e_y_lambda_1_log,
        'e_y_lambda_0_log': e_y_lambda_0_log},
      'e_step_init': e_step_init,
      'm_step_init': m_step_init,
      'e_step': e_step,
      'm_step': m_step,
      'neg_log_likelihood': neg_log_likelihood}
