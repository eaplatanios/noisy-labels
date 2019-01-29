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
  'Transformation', 'OneHotEncoding', 'Embedding',
  'Selection', 'Concatenation']

logger = logging.getLogger(__name__)


class Transformation(with_metaclass(abc.ABCMeta, object)):
  @abc.abstractmethod
  def apply(self, *args, **kwargs):
    pass

  def __call__(self, *args, **kwargs):
    return self.apply(*args, **kwargs)


class OneHotEncoding(Transformation):
  def __init__(
      self, num_inputs, dtype=tf.float32,
      name='OneHotEncoding'):
    self.num_inputs = num_inputs
    self.dtype = dtype
    self.name = name

  def apply(self, *args, **kwargs):
    with tf.name_scope(self.name):
      inputs = args[0]
      if inputs.shape[-1] == 1:
        inputs = tf.squeeze(inputs, axis=-1)
      return tf.one_hot(inputs, depth=self.num_inputs)


class Embedding(Transformation):
  def __init__(
      self, num_inputs, emb_size, dtype=tf.float32,
      name='Embedding'):
    self.num_inputs = num_inputs
    self.emb_size = emb_size
    self.dtype = dtype
    self.name = name

  def apply(self, *args, **kwargs):
    with tf.variable_scope(self.name):
      emb_matrix = tf.get_variable(
        name='emb_matrix',
        shape=[self.num_inputs, self.emb_size],
        initializer=tf.random_normal_initializer(
          dtype=self.dtype))
      inputs = args[0]
      if inputs.shape[-1] == 1:
        inputs = tf.squeeze(inputs, axis=-1)
      return tf.gather(emb_matrix, inputs)


class Selection(Transformation):
  """Selects one of the arguments."""

  def __init__(self, arg_index):
    self.arg_index = arg_index

  def apply(self, *args, **kwargs):
    return args[self.arg_index]


class Concatenation(Transformation):
  """Concatenates selected arguments."""

  def __init__(self, arg_indices):
    self.arg_indices = arg_indices

  def apply(self, *args, **kwargs):
    values = [args[i] for i in self.arg_indices]
    return tf.concat(values, axis=-1)
