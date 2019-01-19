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

import numpy as np
import tensorflow as tf

from .data.synthetic import SyntheticBinaryGenerator
from .models.learners import *
from .models.transformations import *

__author__ = 'eaplatanios'

__all__ = []

np.set_printoptions(linewidth=120)


def main():
  num_instances = 128
  num_predictors = 8
  instances_emb_size = 32
  predictors_emb_size = 32

  data = SyntheticBinaryGenerator().generate(
    num_instances=num_instances,
    predictor_qualities=[0.95, 0.60, 0.45, 0.85, 0.59, 0.98, 0.94, 0.76])

  instances = np.array(data.predictions['instances'], dtype=np.int32)[:, None]
  predictor_indices = np.array(data.predictions['predictor_indices'], dtype=np.int32)
  predictor_values = np.array(data.predictions['predictor_values'], dtype=np.int32)[:, :, None]

  dataset = tf.data.Dataset.from_tensor_slices({
    'instances': instances,
    'predictor_indices': predictor_indices,
    'predictor_values': predictor_values}) \
    .shuffle(128) \
    .repeat() \
    .batch(128)

  learner = NoisyMLPLearner(
    inputs_size=1,
    predictors_size=1,
    config=BinaryNoisyLearnerConfig(),
    optimizer=tf.train.AdamOptimizer(),
    instances_input_fn=Embedding(
      num_inputs=num_instances,
      emb_size=instances_emb_size,
      name='instance_embeddings'),
    predictors_input_fn=Embedding(
      num_inputs=num_predictors,
      emb_size=predictors_emb_size,
      name='predictor_embeddings'),
    model_hidden_units=[128, 64, 32],
    error_hidden_units=[128, 64, 32])

  learner.train(dataset)

  predictions = learner.predict(instances)
  qualities = learner.qualities(instances, predictor_indices)
  qualities_a = np.exp(qualities[:, :, 0])
  qualities_b = np.exp(qualities[:, :, 1])
  qualities_mean = np.mean(qualities_a / (qualities_a + qualities_b), axis=0)
  qualities_mode = np.mean((qualities_a - 1) / (qualities_a + qualities_b - 2), axis=0)

  print('Accuracy: %.4f' % np.mean((predictions[:, 0] >= 0.5) == data.true_labels))
  print('Qualities mean: {}'.format(qualities_mean))
  print('Qualities mode: {}'.format(qualities_mode))

  print('haha Christoph')

  



if __name__ == '__main__':
  main()
