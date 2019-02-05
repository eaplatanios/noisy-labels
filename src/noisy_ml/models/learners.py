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
from sklearn.model_selection import KFold
from tqdm import tqdm

from .utilities import log1mexp

__author__ = 'eaplatanios'

__all__ = [
  'k_fold_cv', 'EMLearner', 'EMConfig',
  'MultiLabelEMConfig']

logger = logging.getLogger(__name__)


def k_fold_cv(
    kw_args, learner_fn, dataset, num_folds=5,
    batch_size=128, warm_start=True, max_m_steps=10000,
    max_em_steps=5, log_m_steps=1000,
    em_step_callback=lambda l: l, seed=None):
  """Selects the best argument list out of `args`, based
  on performance measured using `eval_fn`.

  Args:
    kw_args: List of argument dicts to pass to the
      `learner_fn`.
    dataset: Dataset to use when performing
      cross-validation.

  Returns:
    Instantiated (but not trained) learner using the
    best config out of the provided argument lists.
  """
  logger.info('K-Fold CV - Initializing.')

  if seed is not None:
    np.random.set_state(seed)

  train_data = dataset.to_train(shuffle=False)
  kf = KFold(n_splits=num_folds)

  learner_scores = []
  for learner_kw_args in kw_args:
    logger.info(
      'K-Fold CV - Evaluating kw_args: %s'
      % learner_kw_args)

    with tf.Graph().as_default():
      if seed is not None:
        tf.random.set_random_seed(seed)

      learner = learner_fn(**learner_kw_args)
      scores = []
      folds = kf.split(train_data[0])
      for f, (train_ids, test_ids) in enumerate(folds):
        logger.info(
          'K-Fold CV - Fold %d / %d' % (f, num_folds))

        learner.reset()

        # Train the learner.
        train_dataset = tf.data.Dataset.from_tensor_slices({
          'instances': train_data.instances[train_ids],
          'predictors': train_data.predictors[train_ids],
          'labels': train_data.labels[train_ids],
          'values': train_data.values[train_ids]})
        learner.train(
          dataset=train_dataset, batch_size=batch_size,
          warm_start=warm_start, max_m_steps=max_m_steps,
          max_em_steps=max_em_steps,
          log_m_steps=log_m_steps,
          em_step_callback=em_step_callback)

        # Test the learner.
        test_dataset = tf.data.Dataset.from_tensor_slices({
          'instances': train_data.instances[test_ids],
          'predictors': train_data.predictors[test_ids],
          'labels': train_data.labels[test_ids],
          'values': train_data.values[test_ids]})
        score = -learner.neg_log_likelihood(
          dataset=test_dataset, batch_size=batch_size)
        scores.append(score)

        logger.info(
          'K-Fold CV - Fold %d / %d score %10.4f for kw_args: %s'
          % (f, num_folds, score, learner_kw_args))

      score = float(np.mean(scores))
      learner_scores.append(score)
      logger.info(
        'K-Fold CV - Mean score %10.4f for kw_args: %s'
        % (score, learner_kw_args))
  best_l = int(np.argmax(learner_scores))
  best_kw_args = kw_args[best_l]
  logger.info(
    'K-Fold CV - Best score was %10.4f for kw_args: %s'
    % (learner_scores[best_l], best_kw_args))
  return learner_fn(**best_kw_args)


class EMLearner(object):
  def __init__(
      self, config, predictions_output_fn=lambda x: x):
    self.config = config
    self.predictions_output_fn = predictions_output_fn
    self._build_model()
    self._session = None

  def reset(self):
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
      max_m_steps, log_m_steps=None):
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
      if log_m_steps is not None and \
          (m_step % log_m_steps == 0 or
           m_step == max_m_steps - 1):
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
      max_marginal_steps=0, em_step_callback=None,
      use_progress_bar=False):
    e_step_dataset = dataset.batch(batch_size)
    m_step_dataset = dataset.repeat().shuffle(10000).batch(batch_size)

    self._init_session()
    e_step_iterator_init_op = self._ops['train_iterator'].make_initializer(e_step_dataset)
    m_step_iterator_init_op = self._ops['train_iterator'].make_initializer(m_step_dataset)

    em_steps_range = range(max_em_steps)
    if use_progress_bar:
      em_steps_range = tqdm(em_steps_range, 'EM Step')

    for em_step in em_steps_range:
      if not use_progress_bar:
        logger.info('Iteration %d - Running E-Step' % em_step)
      self._e_step(e_step_iterator_init_op, use_maj=em_step == 0)
      if not use_progress_bar:
        logger.info('Iteration %d - Running M-Step' % em_step)
      self._m_step(
        m_step_iterator_init_op, warm_start,
        max_m_steps, log_m_steps)
      if em_step_callback is not None:
        em_step_callback(self)

    if max_marginal_steps > 0:
      self._session.run(self._ops['set_opt_marginal'])
      logger.info('Optimizing marginal log-likelihood.')
      self._m_step(
        m_step_iterator_init_op, warm_start,
        max_marginal_steps, log_m_steps)
      self._session.run(self._ops['unset_opt_marginal'])

  def neg_log_likelihood(self, dataset, batch_size=128):
    dataset = dataset.batch(batch_size)
    iterator_init_op = self._ops['train_iterator'].make_initializer(dataset)

    self._init_session()
    self._session.run(iterator_init_op)
    neg_log_likelihood = 0.0
    while True:
      try:
        neg_log_likelihood += self._session.run(
          self._ops['neg_log_likelihood'])
      except tf.errors.OutOfRangeError:
        break
    return neg_log_likelihood

  def predict(self, instances, batch_size=128):
    # TODO: Remove hack by having separate train and predict iterators.
    dataset = tf.data.Dataset.from_tensor_slices({
      'instances': instances,
      'predictors': np.zeros([len(instances)], np.int32),
      'labels': np.zeros([len(instances)], np.int32),
      'values': np.zeros([len(instances)], np.float32)}
    ).batch(batch_size)
    iterator_init_op = self._ops['train_iterator'].make_initializer(dataset)

    self._init_session()
    self._session.run(iterator_init_op)
    predictions = []
    while True:
      try:
        p = self._session.run(self._ops['predictions'])
        if self.predictions_output_fn is not None:
          p = self.predictions_output_fn(p)
        predictions.append(p)
      except tf.errors.OutOfRangeError:
        break
    return np.concatenate(predictions, axis=0)

  def qualities(self, instances, predictors, labels, batch_size=128):
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

    # TODO: Remove hack by having separate train and predict iterators.
    dataset = tf.data.Dataset.from_tensor_slices({
      'instances': temp[0].astype(np.int32),
      'predictors': temp[2].astype(np.int32),
      'labels': temp[1].astype(np.int32),
      'values': np.zeros([len(temp[0])], np.float32)}
    ).batch(batch_size)
    iterator_init_op = self._ops['train_iterator'].make_initializer(dataset)

    self._init_session()

    self._session.run(iterator_init_op)
    qualities_mean_log = []
    while True:
      try:
        qualities_mean_log.append(self._session.run(
          self._ops['qualities_mean_log']))
      except tf.errors.OutOfRangeError:
        break

    qualities_mean_log = np.concatenate(
      qualities_mean_log, axis=0)
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

    predictions, q_params, regularization_terms, include_y_prior = self.model.build(
      x_indices, p_indices, l_indices)

    # y_hat has shape: [BatchSize, 2]
    # y_hat[i, 0] is the probability that instance i has label 0.
    y_hat = tf.stack([1.0 - values, values], axis=-1)
    if not self.use_soft_maj:
      y_hat = tf.cast(tf.greater_equal(y_hat, 0.5), tf.int32)

    # TODO: Is this necessary?
    h_log = tf.minimum(predictions, -1e-6)
    h_log = tf.squeeze(tf.batch_gather(
      params=h_log,
      indices=tf.expand_dims(l_indices, axis=-1)), axis=-1)
    h_log = tf.stack([log1mexp(h_log), h_log], axis=-1)

    # q_params shape: [BatchSize, 2, 2]
    q_log = q_params
    h_log_q_log = q_log + tf.expand_dims(h_log, axis=-1)
    qualities_mean_log = tf.stack([
      h_log_q_log[:, 1, 1],
      h_log_q_log[:, 0, 0]], axis=-1)
    qualities_mean_log = tf.reduce_logsumexp(
      qualities_mean_log, axis=-1)

    # The following represent the qualities that correspond
    # to the provided predictor values.
    q_log_y_hat = tf.einsum(
      'ijk,ik->ij', q_log, tf.cast(y_hat, tf.float32))

    # E-Step:

    with tf.name_scope('e_step'):
      # Create the accumulator variables:
      e_y_acc = tf.get_variable(
        name='e_y_acc',
        shape=[self.num_instances, self.num_labels, 2],
        initializer=tf.zeros_initializer(h_log.dtype),
        trainable=False)

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

      e_y_acc_update = tf.cond(
        use_maj,
        lambda: tf.cast(y_hat, tf.float32),
        lambda: q_log_y_hat)

      # Create the accumulator variable update ops.
      xl_indices = tf.stack(
        [x_indices, l_indices], axis=-1)
      e_y_acc_update = e_y_acc \
        .scatter_nd_add(
        indices=xl_indices,
        updates=e_y_acc_update)

      e_y_a = tf.gather_nd(e_y_acc, xl_indices)
      e_y_a_h_log = e_y_a
      if include_y_prior:
        e_y_a_h_log += h_log
      else:
        e_y_a_h_log += np.log(0.5)
      e_y_log = tf.stop_gradient(tf.cond(
        use_maj,
        lambda: tf.log(e_y_a) - tf.log(tf.reduce_sum(e_y_a, axis=-1, keepdims=True)),
        lambda: e_y_a_h_log - tf.reduce_logsumexp(e_y_a_h_log, axis=-1, keepdims=True)))
      e_y = tf.exp(e_y_log)

      e_step_init = e_y_acc.initializer
      e_step = e_y_acc_update

    # M-Step:

    with tf.name_scope('m_step'):
      # Boolean flag about whether or not to optimize the
      # marginal log-likelihood.
      opt_marginal = tf.get_variable(
        name='opt_marginal',
        shape=[],
        dtype=tf.bool,
        initializer=tf.zeros_initializer(),
        trainable=False)
      set_opt_marginal = opt_marginal.assign(True)
      unset_opt_marginal = opt_marginal.assign(False)

      em_ll_term0 = tf.reduce_sum(e_y * h_log)
      em_ll_term1 = tf.reduce_sum(
        tf.einsum('ij,ij->i', q_log_y_hat, e_y))
      em_nll = -em_ll_term0 - em_ll_term1

      marginal_ll = q_log_y_hat + h_log
      marginal_ll = tf.reduce_sum(marginal_ll, axis=-1)
      marginal_ll = -tf.reduce_logsumexp(marginal_ll)

      neg_log_likelihood = tf.cond(
        opt_marginal,
        lambda: marginal_ll,
        lambda : em_nll)
      if len(regularization_terms) > 0:
        neg_log_likelihood += tf.add_n(regularization_terms)

      m_step_init = tf.variables_initializer(
        tf.trainable_variables())
      global_step = tf.train.get_or_create_global_step()
      gvs = self.optimizer.compute_gradients(neg_log_likelihood, tf.trainable_variables())
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
      'set_opt_marginal': set_opt_marginal,
      'unset_opt_marginal': unset_opt_marginal,
      'e_step_init': e_step_init,
      'm_step_init': m_step_init,
      'e_step': e_step,
      'm_step': m_step,
      'neg_log_likelihood': neg_log_likelihood}
