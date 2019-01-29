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

__author__ = 'eaplatanios'

__all__ = [
  'Layer', 'Linear', 'LogSigmoid', 'LogSoftmax',
  'HierarchicalLogSoftmax', 'MLP']

logger = logging.getLogger(__name__)


class Layer(with_metaclass(abc.ABCMeta, object)):
  @abc.abstractmethod
  def apply(self, *args, **kwargs):
    pass

  def __call__(self, *args, **kwargs):
    return self.apply(*args, **kwargs)


class Linear(Layer):
  def __init__(self, num_outputs, name='Linear'):
    self.num_outputs = num_outputs
    self.name = name

  def apply(self, *args, **kwargs):
    inputs = args[0]
    w_initializer = tf.glorot_uniform_initializer(
      dtype=inputs.dtype)
    b_initializer = tf.zeros_initializer(
      dtype=inputs.dtype)
    with tf.variable_scope(self.name):
      w = tf.get_variable(
        name='weights',
        shape=[inputs.shape[-1], self.num_outputs],
        initializer=w_initializer)
      b = tf.get_variable(
        name='bias',
        shape=[self.num_outputs],
        initializer=b_initializer)
      outputs = tf.nn.xw_plus_b(
        x=inputs, weights=w, biases=b, name='linear')
    return outputs


class LogSigmoid(Layer):
  def __init__(self, num_labels, name='LogSigmoid'):
    self.num_labels = num_labels
    self.name = name

  def apply(self, *args, **kwargs):
    inputs = args[0]
    w_initializer = tf.glorot_uniform_initializer(
      dtype=inputs.dtype)
    b_initializer = tf.zeros_initializer(
      dtype=inputs.dtype)
    with tf.variable_scope(self.name):
      w = tf.get_variable(
        name='weights',
        shape=[inputs.shape[-1], self.num_labels],
        initializer=w_initializer)
      b = tf.get_variable(
        name='bias',
        shape=[self.num_labels],
        initializer=b_initializer)
      outputs = tf.nn.xw_plus_b(
        x=inputs, weights=w, biases=b, name='linear')
    return tf.log_sigmoid(outputs)


class LogSoftmax(Layer):
  def __init__(self, num_labels, name='LogSoftmax'):
    self.num_labels = num_labels
    self.name = name

  def apply(self, *args, **kwargs):
    inputs = args[0]
    w_initializer = tf.glorot_uniform_initializer(
      dtype=inputs.dtype)
    b_initializer = tf.zeros_initializer(
      dtype=inputs.dtype)
    with tf.variable_scope(self.name):
      w = tf.get_variable(
        name='weights',
        shape=[inputs.shape[-1], self.num_labels],
        initializer=w_initializer)
      b = tf.get_variable(
        name='bias',
        shape=[self.num_labels],
        initializer=b_initializer)
      outputs = tf.nn.xw_plus_b(
        x=inputs, weights=w, biases=b, name='linear')
    return tf.nn.log_softmax(outputs)


class HierarchicalLogSoftmax(Layer):
  def __init__(
      self, num_labels, hierarchy,
      name='HierarchicalLogSoftmax'):
    self.num_labels = num_labels
    self.hierarchy = hierarchy
    self.name = name

  def apply(self, *args, **kwargs):
    return self._compute(
      args[0], self.hierarchy, parent=0, level=0)

  def _compute(self, inputs, hierarchy, level, parent):
    w_initializer = tf.glorot_uniform_initializer(
      dtype=inputs.dtype)
    b_initializer = tf.zeros_initializer(
      dtype=inputs.dtype)
    parents = list(map(lambda n: n[0], hierarchy))
    num_parents = len(parents)
    with tf.variable_scope('%s_%d_%d' % (self.name, level, parent)):
      w = tf.get_variable(
        name='weights',
        shape=[inputs.shape[-1], num_parents],
        initializer=w_initializer)
      b = tf.get_variable(
        name='bias',
        shape=[num_parents],
        initializer=b_initializer)
      outputs = tf.nn.xw_plus_b(
        x=inputs, weights=w, biases=b, name='linear')

      bs = tf.shape(inputs)[0]

      total_output = None
      for i, p in enumerate(parents):
        output = tf.gather(outputs, i, axis=-1)
        output = tf.nn.softmax(output)
        p_hierarchy = hierarchy[i][1]
        if len(p_hierarchy) > 0:
          p_output = self._compute(
            inputs, p_hierarchy, level + 1, p)
          p_output = tf.exp(p_output)
          temp = tf.convert_to_tensor(
            tf.IndexedSlices(
              indices=p,
              values=1.0,
              dense_shape=tf.constant([self.num_labels])))
          temp = tf.expand_dims(temp, 0)
          temp = tf.tile(temp, [bs, 1])
          p_output += temp
          p_output *= tf.expand_dims(output, axis=-1)
        else:
          p_output = tf.sparse_to_dense(
            sparse_indices=tf.stack([
              tf.range(bs), tf.tile([p], [bs])], axis=-1),
            sparse_values=output,
            output_shape=tf.stack([bs, self.num_labels]))
          # p_output = tf.sparse.to_dense(
          #   tf.SparseTensor(
          #     indices=tf.stack([
          #       tf.range(bs), tf.tile([p], [bs])], axis=-1),
          #     values=output,
          #     dense_shape=tf.stack([bs, self.num_labels])))
        if total_output is None:
          total_output = p_output
        else:
          total_output += p_output

    return tf.log(total_output + 1e-12)


class MLP(Layer):
  def __init__(
      self,
      hidden_units,
      activation=tf.nn.leaky_relu,
      output_layer=lambda x: x,
      name='MLP'):
    self.hidden_units = hidden_units
    self.activation = activation
    self.output_projection = output_layer
    self.name = name

  def apply(self, *args, **kwargs):
    with tf.variable_scope(self.name):
      hidden = args[0]
      w_initializer = tf.glorot_uniform_initializer(
        dtype=hidden.dtype)
      b_initializer = tf.zeros_initializer(
        dtype=hidden.dtype)
      layers = enumerate(self.hidden_units)
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
          if hidden.shape.rank == 3:
            hidden = tf.tensordot(
              hidden, w, axes=[[2], [0]], name='linear')
          else:
            hidden = tf.nn.xw_plus_b(
              x=hidden, weights=w, biases=b, name='linear')
          hidden = self.activation(hidden)
      return self.output_projection(hidden)
