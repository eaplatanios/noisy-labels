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
# import os
# import tensorflow as tf
# import six
#
# from .data.loaders import BrainLoader, NELLLoader
# from .data.utilities import compute_binary_qualities
# from .evaluation.metrics import *
# from .models.layers import *
# from .models.learners import *
# from .models.transformations import *
#
# __author__ = 'eaplatanios'
#
# __all__ = []
#
#
# def run_experiment(loader, label):
#   working_dir = os.getcwd()
#   data_dir = os.path.join(working_dir, os.pardir, 'data')
#   data = loader.load_binary(
#     data_dir=data_dir,
#     label=label,
#     small_version=True)
#
#   num_instances = len(data['instances'])
#   num_predictors = len(data['classifiers'])
#
#   instances = np.array(data['instances'], dtype=np.int32)[:, None]
#   predictor_indices = np.array(data['predictors'], dtype=np.int32)[:, :, None]
#   predictor_values = np.array(data['predictor_values'], dtype=np.int32)[:, :, None]
#   predictor_values_soft = np.array(data['predictor_values_soft'], dtype=np.float32)[:, :, None]
#
#   dataset = tf.data.Dataset.from_tensor_slices({
#     'instances': instances,
#     'predictors': predictor_indices,
#     'predictor_values': predictor_values,
#     'predictor_values_soft': predictor_values_soft}) \
#     .shuffle(100) \
#     .repeat() \
#     .batch(128)
#
#   # instances_input_fn = OneHotEncoding(
#   #   num_inputs=num_instances,
#   #   name='instance_embeddings')
#   #
#   # predictors_input_fn = OneHotEncoding(
#   #   num_inputs=num_predictors,
#   #   name='predictor_embeddings')
#
#   instances_input_fn = Embedding(
#     num_inputs=num_instances,
#     emb_size=32,
#     name='instance_embeddings')
#
#   predictors_input_fn = Embedding(
#     num_inputs=num_predictors,
#     emb_size=32,
#     name='predictor_embeddings')
#
#   qualities_input_fn = InstancesPredictorsConcatenation()
#
#   model_fn = MLP(
#     hidden_units=[128, 64, 32],
#     num_outputs=1,
#     activation=tf.nn.selu,
#     output_projection=tf.sigmoid,
#     name='model_fn')
#
#   qualities_fn = MLP(
#     hidden_units=[128, 64, 32],
#     num_outputs=2,
#     activation=tf.nn.selu,
#     name='qualities_fn')
#
#   learner = NoisyLearner(
#     instances_input_size=1,
#     predictors_input_size=1,
#     config=BinaryNoisyLearnerConfig(
#       model_fn=model_fn,
#       qualities_fn=qualities_fn,
#       prior_probability=0.5,
#       prior_correct=0.9,
#       max_param_value=1e6,
#       warm_up_steps=10000000,
#       eps=1e-12),
#     phase_one_optimizer=tf.train.AdamOptimizer(),
#     phase_two_optimizer=tf.train.AdamOptimizer(),
#     instances_dtype=tf.int32,
#     predictors_dtype=tf.int32,
#     instances_input_fn=instances_input_fn,
#     predictors_input_fn=predictors_input_fn,
#     qualities_input_fn=qualities_input_fn)
#
#   learner.train(
#     dataset=dataset,
#     max_steps=10000,
#     loss_abs_threshold=-np.log(0.99),
#     min_steps_below_threshold=20,
#     log_steps=1000)
#
#   # predictions, predicted_qualities = learner.run_gibbs_sampler(predictor_values[:, :, 0])
#
#   predictions = learner.predict(instances)[:, 0]
#   predicted_qualities = learner.qualities(instances, predictor_indices)
#   predicted_qualities = np.mean(predicted_qualities, axis=0)
#
#   true_qualities = compute_binary_qualities(data)
#   true_qualities = [p[1] for p in sorted(six.iteritems(true_qualities))]
#
#   return {
#     'predictions': predictions,
#     'true_labels': np.array(data['true_labels']),
#     'predicted_qualities': np.array(predicted_qualities),
#     'true_qualities': np.array(true_qualities),
#     'predictor_values': np.array(data['predictor_values'], dtype=np.int32),
#     'predictor_values_soft': np.array(data['predictor_values_soft'], dtype=np.float32)}
#
#
# def evaluate(results):
#   mad_error_rank = compute_mad_error_rank(
#     results['predicted_qualities'],
#     results['true_qualities'])
#   mad_error = compute_mad_error(
#     results['predicted_qualities'],
#     results['true_qualities'])
#   auc_target = compute_auc(
#     results['predictions'],
#     results['true_labels'])
#   maj_soft_auc_target = compute_auc(
#     results['maj_soft_predictions'],
#     results['true_labels'])
#   maj_hard_auc_target = compute_auc(
#     results['maj_hard_predictions'],
#     results['true_labels'])
#   return {
#     'mad_error_rank': mad_error_rank,
#     'mad_error': mad_error,
#     'auc_target': auc_target,
#     'maj_soft_auc_target': maj_soft_auc_target,
#     'maj_hard_auc_target': maj_hard_auc_target}
#
#
# def main():
#   nell_labels = [
#     'animal', 'beverage', 'bird', 'bodypart', 'city',
#     'disease', 'drug', 'fish', 'food', 'fruit', 'muscle',
#     'person', 'protein', 'river', 'vegetable']
#
#   brain_labels = [
#     'region_1', 'region_2', 'region_3', 'region_4',
#     'region_5', 'region_6', 'region_7', 'region_8',
#     'region_9', 'region_10', 'region_11']
#
#   results = {
#     'mad_error_rank': [],
#     'mad_error': [],
#     'auc_target': [],
#     'maj_soft_auc_target': [],
#     'maj_hard_auc_target': []}
#
#   for label in brain_labels:
#     with tf.Graph().as_default():
#       label_results = run_experiment(BrainLoader, label)
#     label_results = evaluate({
#       'maj_soft_predictions': np.mean(label_results['predictor_values_soft'], axis=1),
#       'maj_hard_predictions': np.mean(label_results['predictor_values'], axis=1),
#       'predictions': label_results['predictions'],
#       'true_labels': label_results['true_labels'],
#       'predicted_qualities': label_results['predicted_qualities'][None, :],
#       'true_qualities': label_results['true_qualities'][None, :]})
#     results['mad_error_rank'].append(label_results['mad_error_rank'])
#     results['mad_error'].append(label_results['mad_error'])
#     results['auc_target'].append(label_results['auc_target'])
#     results['maj_soft_auc_target'].append(label_results['maj_soft_auc_target'])
#     results['maj_hard_auc_target'].append(label_results['maj_hard_auc_target'])
#
#     print('Results so far:')
#     print('Current MAD_error_rank: {}'.format(np.mean(results['mad_error_rank'])))
#     print('Current MAD_error: {}'.format(np.mean(results['mad_error'])))
#     print('Current AUC_target: {}'.format(np.mean(results['auc_target'])))
#     print('Current MAJ Soft AUC_target: {}'.format(np.mean(results['maj_soft_auc_target'])))
#     print('Current MAJ Hard AUC_target: {}'.format(np.mean(results['maj_hard_auc_target'])))
#
#   print('haha Christoph')
#
#
# if __name__ == '__main__':
#   main()
