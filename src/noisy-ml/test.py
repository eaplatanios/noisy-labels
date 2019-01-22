# # Copyright 2019, Emmanouil Antonios Platanios. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License"); you may not
# # use this file except in compliance with the License. You may obtain a copy of
# # the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# # WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# # License for the specific language governing permissions and limitations under
# # the License.
#
# from __future__ import absolute_import, division, print_function
#
# import numpy as np
# import tensorflow as tf
#
# from .data.synthetic import SyntheticBinaryGenerator
# from .models.layers import *
# from .models.learners import *
# from .models.transformations import *
#
# __author__ = 'eaplatanios'
#
# __all__ = []
#
# np.set_printoptions(linewidth=120)
#
#
# def main():
#   num_instances = 32
#   num_predictors = 8
#   instances_emb_size = 128
#   predictors_emb_size = 128
#
#   data = SyntheticBinaryGenerator().generate(
#     num_instances=num_instances,
#     predictor_qualities=[0.90, 0.75, 0.45, 0.85, 0.59, 0.90, 0.90, 0.76])
#
#   instances = np.array(data.predictions['instances'], dtype=np.int32)[:, None]
#   predictor_indices = np.array(data.predictions['predictors'], dtype=np.int32)[:, :, None]
#   predictor_values = np.array(data.predictions['predictor_values'], dtype=np.int32)[:, :, None]
#
#   dataset = tf.data.Dataset.from_tensor_slices({
#     'instances': instances,
#     'predictors': predictor_indices,
#     'predictor_values': predictor_values}) \
#     .shuffle(128) \
#     .repeat() \
#     .batch(128)
#
#   instances_input_fn = Embedding(
#     num_inputs=num_instances,
#     emb_size=instances_emb_size,
#     name='instance_embeddings')
#
#   predictors_input_fn = Embedding(
#     num_inputs=num_predictors,
#     emb_size=predictors_emb_size,
#     name='predictor_embeddings')
#
#   qualities_input_fn = InstancesPredictorsConcatenation()
#
#   model_fn = MLP(
#     hidden_units=[64, 32],
#     num_outputs=1,
#     activation=tf.nn.leaky_relu,
#     output_projection=tf.sigmoid,
#     name='model_fn')
#
#   qualities_fn = MLP(
#     hidden_units=[64, 32],
#     num_outputs=2,
#     activation=tf.nn.leaky_relu,
#     name='qualities_fn')
#
#   learner = NoisyLearner(
#     instances_input_size=1,
#     predictors_input_size=1,
#     config=BinaryNoisyLearnerConfig(
#       model_fn=model_fn,
#       qualities_fn=qualities_fn,
#       warm_up_steps=1000,
#       prior_correct=0.99,
#       max_param_value=1e6),
#     optimizer=tf.train.AdamOptimizer(),
#     instances_dtype=tf.int32,
#     predictors_dtype=tf.int32,
#     instances_input_fn=instances_input_fn,
#     predictors_input_fn=predictors_input_fn,
#     qualities_input_fn=qualities_input_fn)
#
#   learner.train(dataset, max_steps=10000)
#
#   predictions = learner.predict(instances)
#   qualities = learner.qualities(instances, predictor_indices)
#   qualities_mean = np.mean(qualities, axis=0)
#
#   print('Accuracy: %.4f' % np.mean((predictions[:, 0] >= 0.5) == data.true_labels))
#   print('Qualities mean: {}'.format(qualities_mean))
#
#   print('haha Christoph')
#
#
# if __name__ == '__main__':
#   main()
