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

__all__ = ['Layer', 'MLP']

logger = logging.getLogger(__name__)


class Layer(with_metaclass(abc.ABCMeta, object)):
  @abc.abstractmethod
  def apply(self, *args, **kwargs):
    pass

  def __call__(self, *args, **kwargs):
    return self.apply(*args, **kwargs)


class MLP(Layer):
  def __init__(
      self,
      hidden_units,
      num_outputs,
      activation=tf.nn.leaky_relu,
      output_projection=lambda x: x,
      name='mlp'):
    self.hidden_units = hidden_units
    self.num_outputs = num_outputs
    self.activation = activation
    self.output_projection = output_projection
    self.name = name

  def apply(self, *args, **kwargs):
    with tf.variable_scope(self.name):
      hidden = args[0]
      w_initializer = tf.glorot_uniform_initializer(
        dtype=hidden.dtype)
      b_initializer = tf.zeros_initializer(
        dtype=hidden.dtype)
      layers = enumerate(
        self.hidden_units + [self.num_outputs])
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
          if i < len(self.hidden_units):
            hidden = self.activation(hidden)
      return self.output_projection(hidden)
