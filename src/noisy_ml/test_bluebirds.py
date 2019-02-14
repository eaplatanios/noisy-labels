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

import os
import numpy as np
import tensorflow as tf

from .data.crowdsourced import *
from .data.rte import *
from .data.wordsim import *
from .evaluation.metrics import *
from .training.layers import *
from .training.learners import *
from .training.models import *
from .training.amsgrad import *

__author__ = "eaplatanios"


seed = 1234567890
np.random.seed(seed)
tf.random.set_random_seed(seed)


class BlueBirdsModel(Model):
    def __init__(self, dataset, q_latent_size=None, gamma=0.25):
        self.dataset = dataset
        self.q_latent_size = q_latent_size
        self.gamma = gamma

    def build(self, instances, predictors, labels):
        if self.dataset.instance_features is None:
            instances = Embedding(
                num_inputs=len(self.dataset.instances),
                emb_size=16,
                name="instance_embeddings",
            )(instances)
        else:
            instances = FeatureMap(
                features=np.array(self.dataset.instance_features),
                adjust_magnitude=False,
                name="instance_features/feature_map",
            )(instances)

        predictions = MLP(
            hidden_units=[],
            activation=tf.nn.selu,
            output_layer=LogSigmoid(
                num_labels=len(self.dataset.labels), name="m_fn/log_sigmoid"
            ),
            name="m_fn",
        )(instances)

        predictors = Embedding(
            num_inputs=len(self.dataset.predictors),
            emb_size=16,
            name="predictor_embeddings",
        )(predictors)

        if self.q_latent_size is None:
            q_fn_args = predictors
            q_params = MLP(
                hidden_units=[], activation=tf.nn.selu, name="q_fn"
            ).and_then(Linear(num_outputs=4, name="q_fn/linear"))(q_fn_args)
            q_params = tf.reshape(q_params, [-1, 2, 2])
            q_params = tf.nn.log_softmax(q_params, axis=-1)

            regularization_terms = []
        else:
            q_i = MLP(
                hidden_units=[],
                activation=tf.nn.selu,
                output_layer=Linear(
                    num_outputs=4 * self.q_latent_size, name="q_i_fn/linear"
                ),
                name="q_i_fn",
            )(instances)
            q_i = tf.reshape(q_i, [-1, 2, 2, self.q_latent_size])

            q_p = MLP(
                hidden_units=[],
                activation=tf.nn.selu,
                output_layer=Linear(
                    num_outputs=4 * self.q_latent_size, name="q_p_fn/linear"
                ),
                name="q_p_fn",
            )(predictors)
            q_p = tf.reshape(q_p, [-1, 2, 2, self.q_latent_size])

            q_params = q_i + q_p
            q_params = tf.reduce_logsumexp(q_params, axis=-1)
            q_params = tf.nn.log_softmax(q_params, axis=-1)

            num_labels_per_worker = self.dataset.avg_labels_per_predictor()
            num_labels_per_item = self.dataset.avg_labels_per_item()
            alpha = self.gamma * ((len(self.dataset.labels) * 2) ** 2)
            beta = alpha * num_labels_per_worker / num_labels_per_item
            regularization_terms = [
                beta * tf.reduce_sum(q_i * q_i) / 2,
                alpha * tf.reduce_sum(q_p * q_p) / 2,
            ]

        return BuiltModel(
            predictions=predictions,
            q_params=q_params,
            regularization_terms=regularization_terms,
            include_y_prior=True,
        )


def run_experiment():
    working_dir = os.getcwd()
    data_dir = os.path.join(working_dir, os.pardir, "data")
    # dataset = BlueBirdsLoader.load(data_dir, load_features=True)
    # dataset = RTELoader.load(data_dir, load_features=True)
    dataset = WordSimLoader.load(data_dir, load_features=True)

    def learner_fn(q_latent_size, gamma):
        model = BlueBirdsModel(
            dataset=dataset, q_latent_size=q_latent_size, gamma=gamma
        )
        # model = MMCE_M(
        #   dataset=dataset,
        #   gamma=gamma)
        return EMLearner(
            config=MultiLabelEMConfig(
                num_instances=len(dataset.instances),
                num_predictors=len(dataset.predictors),
                num_labels=len(dataset.labels),
                model=model,
                optimizer=AMSGrad(1e-3),  # tf.train.AdamOptimizer(),
                use_soft_maj=True,
                use_soft_y_hat=False,
            ),
            predictions_output_fn=np.exp,
        )

    evaluator = Evaluator(dataset)

    def em_callback(learner):
        Result.merge(evaluator.evaluate_per_label(learner, batch_size=128)).log(
            prefix="EM           "
        )
        Result.merge(evaluator.evaluate_maj_per_label()).log(prefix="Majority Vote")

    learner_cv_kwargs = [
        {"q_latent_size": 1, "gamma": 0.25},
        {"q_latent_size": 1, "gamma": 2 ** -3},
        {"q_latent_size": 1, "gamma": 2 ** -2},
        {"q_latent_size": 1, "gamma": 2 ** -1},
        {"q_latent_size": 1, "gamma": 2 ** 0},
        {"q_latent_size": 1, "gamma": 2 ** 1},
        {"q_latent_size": 1, "gamma": 2 ** 2},
        {"q_latent_size": 1, "gamma": 2 ** 3},
    ]

    # learner = k_fold_cv(
    #   kw_args=cv_kw_args, learner_fn=learner_fn,
    #   dataset=dataset, num_folds=5, batch_size=128,
    #   warm_start=True, max_m_steps=10000, max_em_steps=5,
    #   log_m_steps=1000, em_step_callback=em_callback,
    #   seed=seed)
    learner = learner_fn(**learner_cv_kwargs[0])

    train_data = dataset.to_train(shuffle=True)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        {
            "instances": train_data.instances,
            "predictors": train_data.predictors,
            "labels": train_data.labels,
            "values": train_data.values,
        }
    )
    learner.train(
        dataset=train_dataset,
        batch_size=1024,
        warm_start=True,
        max_m_steps=1000,
        max_em_steps=5,
        log_m_steps=100,
        max_marginal_steps=1000,
        em_step_callback=em_callback,
    )

    return {
        "em": Result.merge(evaluator.evaluate_per_label(learner, batch_size=128)),
        "maj": Result.merge(evaluator.evaluate_maj_per_label()),
    }


if __name__ == "__main__":
    results = run_experiment()
    results["em"].log(prefix="EM           ")
    results["maj"].log(prefix="Majority Vote")
