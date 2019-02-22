"""A collection of models for noisy data."""

import abc
import logging

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from collections import namedtuple

from ..networks.layers import *

__all__ = ["BuiltModel", "LNL", "MultiClassLNL", "Model", "MMCE_M", "VariationalNoisyModel"]

logger = logging.getLogger(__name__)

tfd = tfp.distributions


class BuiltModel(
    namedtuple(
        "BuiltModel", (
            "predictions",
            "q_params",
            "regularization_terms",
            "include_y_prior"
        )
    )
):
    """Type of the instance returned by Model.build."""
    pass


class BuiltVariationalModel(
    namedtuple(
        "BuiltModel", (
            "regularization_terms",
            "predictor_embeddings",
            "approx_posteriors",
            "predictions",
            "confusions"
        )
    )
):
    """Type of the instance returned by Model.build."""
    pass


class Model(abc.ABC):
    @abc.abstractmethod
    def build(self, instances, predictors, labels):
        raise NotImplementedError


class MMCE_M(Model):
    """Model proposed in 'Regularized Minimax Conditional Entropy for Crowdsourcing'.

    Source: https://arxiv.org/pdf/1503.07240.pdf
    """

    def __init__(self, dataset, gamma=0.25):
        self.dataset = dataset
        self.gamma = gamma

    def build(self, instances, predictors, labels):
        predictions = Embedding(
            num_inputs=len(self.dataset.instances),
            emb_size=len(self.dataset.labels),
            name="instance_embeddings",
        )(instances)
        predictions = tf.log_sigmoid(predictions)

        q_i = Embedding(
            num_inputs=len(self.dataset.instances),
            emb_size=4,
            name="q_i/instance_embeddings",
        )(instances)
        q_i = tf.reshape(q_i, [-1, 2, 2])

        q_p = Embedding(
            num_inputs=len(self.dataset.predictors),
            emb_size=4,
            name="q_p/predictor_embeddings",
        )(predictors)
        q_p = tf.reshape(q_p, [-1, 2, 2])

        q_params = tf.nn.log_softmax(q_i + q_p, axis=-1)

        num_labels_per_worker = self.dataset.avg_labels_per_predictor()
        num_labels_per_item = self.dataset.avg_labels_per_item()
        alpha = self.gamma * ((len(self.dataset.labels) * 2) ** 2)
        beta = alpha * num_labels_per_worker / num_labels_per_item
        regularization_terms = [
            beta * tf.reduce_sum(q_i * q_i) / 2,
            alpha * tf.reduce_sum(q_p * q_p) / 2,
        ]

        return BuiltModel(
            predictions, q_params, regularization_terms, include_y_prior=False
        )


class LNL(Model):
    def __init__(
        self,
        dataset,
        instances_emb_size=None,
        predictors_emb_size=None,
        labels_emb_size=None,
        instances_hidden=None,
        predictors_hidden=None,
        q_latent_size=None,
        gamma=0.25,
    ):
        self.dataset = dataset
        self.instances_emb_size = instances_emb_size
        self.predictors_emb_size = predictors_emb_size
        self.labels_emb_size = labels_emb_size
        self.instances_hidden = instances_hidden or list()
        self.predictors_hidden = predictors_hidden or list()
        self.q_latent_size = q_latent_size
        self.gamma = gamma

    def build(self, instances, predictors, labels):
        if (
            self.dataset.instance_features is None
            or self.instances_emb_size is not None
        ):
            instances = Embedding(
                num_inputs=len(self.dataset.instances),
                emb_size=self.instances_emb_size,
                name="instance_embeddings",
            )(instances)
        else:
            instances = FeatureMap(
                features=np.array(self.dataset.instance_features),
                adjust_magnitude=False,
                name="instance_features/feature_map",
            )(instances)

        predictions = MLP(
            hidden_units=self.instances_hidden,
            activation=tf.nn.selu,
            output_layer=LogSigmoid(
                num_labels=len(self.dataset.labels), name="m_fn/log_sigmoid"
            ),
            name="m_fn",
        )(instances)

        predictors = Embedding(
            num_inputs=len(self.dataset.predictors),
            emb_size=self.predictors_emb_size,
            name="predictor_embeddings",
        )(predictors)

        if self.q_latent_size is None:
            q_fn_args = predictors
            q_params = MLP(
                hidden_units=self.predictors_hidden, activation=tf.nn.selu, name="q_fn"
            ).and_then(Linear(num_outputs=4, name="q_fn/linear"))(q_fn_args)
            q_params = tf.reshape(q_params, [-1, 2, 2])
            q_params = tf.nn.log_softmax(q_params, axis=-1)

            regularization_terms = []
        else:
            q_i = MLP(
                hidden_units=self.instances_hidden,
                activation=tf.nn.selu,
                output_layer=Linear(
                    num_outputs=4 * self.q_latent_size, name="q_i_fn/linear"
                ),
                name="q_i_fn",
            )(instances)
            q_i = tf.reshape(q_i, [-1, 2, 2, self.q_latent_size])

            q_p = MLP(
                hidden_units=self.predictors_hidden,
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


class MultiClassLNL(Model):
    """A multi-class version of LNL model.

    TODO: refactor LNL and merge with this to avoid code duplication.
    """

    def __init__(
        self,
        dataset,
        instances_emb_size=None,
        predictors_emb_size=None,
        labels_emb_size=None,
        instances_hidden=None,
        predictors_hidden=None,
        q_latent_size=None,
        gamma=0.25,
    ):
        self.dataset = dataset
        self.instances_emb_size = instances_emb_size
        self.predictors_emb_size = predictors_emb_size
        self.labels_emb_size = labels_emb_size
        self.instances_hidden = instances_hidden or list()
        self.predictors_hidden = predictors_hidden or list()
        self.q_latent_size = q_latent_size
        self.gamma = gamma

    def build(self, instances, predictors, labels):
        """Builds ops.

        TODO: refactor into sub-functions + factor out network creation code into `networks` module.
        """
        if (
            self.dataset.instance_features is None
            or self.instances_emb_size is not None
        ):
            instances = Embedding(
                num_inputs=len(self.dataset.instances),
                emb_size=self.instances_emb_size,
                name="instance_embeddings",
            )(instances)
        else:
            instances = FeatureMap(
                features=np.array(self.dataset.instance_features),
                adjust_magnitude=False,
                name="instance_features/feature_map",
            )(instances)

        # Compute predictions.
        with tf.variable_scope("m_fn"):
            # Shared hiddens for m_fn between all labels.
            # hiddens: <float32> [batch_size, hidden_size].
            hiddens = instances
            for h_units in self.instances_hidden:
                hiddens = tf.layers.Dense(units=h_units, activation=tf.nn.selu)(hiddens)
            # Predictions is a list of num_labels tensors:
            # <float32> [batch_size, num_classes].
            with tf.variable_scope("log_softmax"):
                predictions = []
                for nc in self.dataset.num_classes:
                    predictions.append(
                        tf.nn.log_softmax(tf.layers.Dense(units=nc)(hiddens))
                    )

        # Predictor embeddings.
        # predictors: <float32> [batch_size, predictor_emb_size]
        predictors = Embedding(
            num_inputs=len(self.dataset.predictors),
            emb_size=self.predictors_emb_size,
            name="predictor_embeddings",
        )(predictors)

        # Compute annotator quality confusion matrices (per-label).
        confusion_latent_size = self.q_latent_size or 1
        with tf.variable_scope("q_fn"):
            # Shared hiddens for q_fn between all labels.
            # hiddens: <float32> [batch_size, hidden_size].
            hiddens = predictors
            for h_units in self.predictors_hidden:
                hiddens = tf.layers.Dense(units=h_units, activation=tf.nn.selu)(hiddens)
            # Pre-confusions is a list of num_labels tensors:
            # <float32> [batch_size, num_classes, num_classes, latent_size].
            pre_q_confusions = []
            for nc in self.dataset.num_classes:
                pre_q_confusion = tf.layers.Dense(
                    units=(nc * nc * confusion_latent_size)
                )(hiddens)
                pre_q_confusions.append(
                    tf.reshape(pre_q_confusion, [-1, nc, nc, confusion_latent_size])
                )

        # If annotator confusions are modeled as instance-independent.
        if self.q_latent_size is None:
            with tf.variable_scope("q_fn"):
                # Confusions is a list of num_labels tensors log-normalized along the last axis:
                # <float32> [batch_size, num_classes, num_classes].
                confusions = []
                for pqc in pre_q_confusions:
                    c = tf.nn.log_softmax(tf.squeeze(pqc, axis=-1), axis=-1)
                    confusions.append(c)
                regularization_terms = []
        # If we also want to take into account per-instance difficulties.
        else:
            with tf.variable_scope("d_fn"):
                # Shared hiddens for d_fn between all labels.
                # hiddens: <float32> [batch_size, hidden_size].
                hiddens = instances
                for h_units in self.predictors_hidden:
                    hiddens = tf.layers.Dense(units=h_units, activation=tf.nn.selu)(
                        hiddens
                    )
                # Pre-confusions is a list of num_labels tensors:
                # <float32> [batch_size, num_classes, num_classes, latent_size].
                pre_d_confusions = []
                for nc in self.dataset.num_classes:
                    pre_d_confusion = tf.layers.Dense(
                        units=(nc * nc * confusion_latent_size)
                    )(hiddens)
                    pre_d_confusions.append(
                        tf.reshape(pre_d_confusion, [-1, nc, nc, confusion_latent_size])
                    )

            with tf.variable_scope("q_fn_d_fn"):
                # Combine pre_q_confusions and pre_d_confusions.
                # Confusions is a list of num_labels tensors log-normalized along the last axis:
                # <float32> [batch_size, num_classes, num_classes].
                confusions = []
                for pqc, pdc in zip(pre_q_confusions, pre_d_confusions):
                    # Compute confusion matrices.
                    c = tf.nn.log_softmax(pqc + pdc, axis=-2)
                    c = tf.reduce_logsumexp(c, axis=-1)
                    confusions.append(c)

            with tf.variable_scope("reg_terms"):
                # Compute regularization terms.
                num_labels_per_worker = self.dataset.avg_labels_per_predictor()
                num_labels_per_item = self.dataset.avg_labels_per_item()
                # TODO: not sure how to adjust this for multi-class case...
                alpha = self.gamma * ((len(self.dataset.labels) * 2) ** 2)
                beta = alpha * num_labels_per_worker / num_labels_per_item
                regularization_terms = [
                    beta
                    * tf.reduce_sum(sum(q_i * q_i for q_i in pre_q_confusions))
                    / 2,
                    alpha
                    * tf.reduce_sum(sum(q_p * q_p for q_p in pre_d_confusions))
                    / 2,
                ]

        return BuiltModel(
            predictions=predictions,
            q_params=confusions,
            regularization_terms=regularization_terms,
            include_y_prior=True,
        )


class VariationalNoisyModel(Model):
    """A model for noisy learning with a variational posterior distribution."""

    def __init__(
        self,
        dataset,
        num_labels,
        num_classes,
        num_predictors,
        num_predictor_samples=1,
        instances_emb_size=None,
        predictors_emb_size=None,
        instances_hidden=None,
        predictors_hidden=None,
        q_latent_size=None,
        gamma=1.0,
    ):
        self.dataset = dataset
        self.num_labels = num_labels
        self.num_classes = num_classes
        self.num_predictors = num_predictors
        self.num_predictor_samples = num_predictor_samples
        self.instances_emb_size = instances_emb_size
        self.predictors_emb_size = predictors_emb_size
        self.instances_hidden = instances_hidden or list()
        self.predictors_hidden = predictors_hidden or list()
        self.q_latent_size = q_latent_size
        self.gamma = gamma

    def _build_predictor_embeddings(self, instances, labels, values, predictors):
        """Builds predictor embeddings as re-parametrized samples
        from the variational distribution."""
        predictor_approx_posteriors = []
        predictor_embeddings = []

        # Convert predictor indices into one-hot masks.
        # predictors_onehot: list of <int32> [batch_size] for each predictor.
        predictors_onehot = tf.unstack(
            tf.one_hot(predictors, self.num_predictors, dtype=tf.int32),
            axis=-1
        )

        # For each predictor, select the corresponding, instances, labels, and values.
        for p_onehot in predictors_onehot:
            # <int32> [batch_size_p].
            p_range = tf.range(tf.reduce_sum(p_onehot))
            # <float32> [batch_size_p, instance_feature_size].
            p_instances = tf.boolean_mask(instances, p_onehot)
            # <int32> [batch_size_p].
            p_labels = tf.to_int32(tf.boolean_mask(labels, p_onehot))
            # <int32> [batch_size_p].
            p_values = tf.to_int32(tf.round(tf.boolean_mask(values, p_onehot)))

            # Build encoder input out of (p_instance, p_labels, p_values).
            p_values_reprs = []
            # list of <int32> [batch_size_p] for each label.
            p_labels_onehot = tf.unstack(tf.one_hot(p_labels, self.num_labels), axis=-1)
            for nc, pl_onehot in zip(self.num_classes, p_labels_onehot):
                # <int32> [batch_size_p_l].
                pl_indices = tf.boolean_mask(p_range, pl_onehot)
                pl_values = tf.boolean_mask(p_values, pl_onehot)
                # <float32> [batch_size_p, nc].
                pl_values_repr = tf.sparse_to_dense(
                    sparse_indices=tf.stack([pl_indices, pl_values], axis=1),
                    output_shape=(tf.reduce_sum(p_onehot), nc),
                    sparse_values=1.0
                )
                p_values_reprs.append(pl_values_repr)
            # <float32> [batch_size_p, feature_size + total_num_classes_for_all_labels].
            encoder_input = tf.concat([p_instances] + p_values_reprs, axis=-1)

            # Build an encoder.
            with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
                hiddens = encoder_input
                for h_units in self.predictors_hidden:
                    hiddens = tf.layers.Dense(
                        units=h_units,
                        activation=tf.nn.selu
                    )(hiddens)
                hiddens = tf.reduce_mean(hiddens, axis=0, keepdims=True)

                # Build an approximate posterior distribution.
                p_embedding_loc_scale = tf.layers.Dense(
                    units=(2 * self.predictors_emb_size)
                )(hiddens)
                p_embedding_loc = p_embedding_loc_scale[:, self.predictors_emb_size]
                p_embedding_scale = tf.nn.softplus(
                    p_embedding_loc_scale[:, self.predictors_emb_size:] +
                    tf.math.log(tf.math.expm1(1.0))
                )
                p_approx_posterior = tfd.MultivariateNormalDiag(
                    loc=p_embedding_loc,
                    scale_diag=p_embedding_scale,
                    name="p_emb_dist"
                )
                predictor_approx_posteriors.append(p_approx_posterior)

                # Sample predictor embeddings from the variational distribution.
                # <float32> [num_predictor_samples, predictor_emb_size].
                p_embeddings = p_approx_posterior.sample(self.num_predictor_samples)
                predictor_embeddings.append(p_embeddings)

        # Construct predictor embeddings for each predictor in the batch.
        # <float32> [num_predictors, num_predictor_samples, predictor_emb_size].
        predictor_embeddings = tf.stack(predictor_embeddings)

        return predictor_embeddings, predictor_approx_posteriors

    def build(self, instances, predictors, labels, values):
        """Builds ops."""
        if (
            self.dataset.instance_features is None
            or self.instances_emb_size is not None
        ):
            instances = Embedding(
                num_inputs=len(self.dataset.instances),
                emb_size=self.instances_emb_size,
                name="instance_embeddings",
            )(instances)
        else:
            instances = FeatureMap(
                features=np.array(self.dataset.instance_features),
                adjust_magnitude=False,
                name="instance_features/feature_map",
            )(instances)

        # Compute predictions.
        with tf.variable_scope("m_fn"):
            # Shared hiddens for m_fn between all labels.
            # hiddens: <float32> [batch_size, hidden_size].
            hiddens = instances
            for h_units in self.instances_hidden:
                hiddens = tf.layers.Dense(units=h_units, activation=tf.nn.selu)(hiddens)
            # Predictions is a list of num_labels tensors:
            # <float32> [batch_size, num_classes].
            with tf.variable_scope("log_softmax"):
                predictions = []
                for nc in self.dataset.num_classes:
                    predictions.append(
                        tf.nn.log_softmax(tf.layers.Dense(units=nc)(hiddens))
                    )

        # Predictor embeddings.
        # predictors: <float32> [num_predictors, num_predictor_samples, predictor_emb_size].
        predictor_embeddings, approx_posteriors = self._build_predictor_embeddings(
            instances, labels, values, predictors
        )
        # <float32> [batch_size, num_predictor_samples, predictor_emb_size].
        predictors = tf.gather(predictor_embeddings, predictors)

        # Compute annotator quality confusion matrices (per-label).
        confusion_latent_size = self.q_latent_size or 1
        with tf.variable_scope("q_fn"):
            # Shared hiddens for q_fn between all labels.
            # hiddens: <float32> [batch_size * num_predictor_samples, hidden_size].
            hiddens = tf.reshape(predictors, [-1, self.predictors_emb_size])
            for h_units in self.predictors_hidden:
                hiddens = tf.layers.Dense(units=h_units, activation=tf.nn.selu)(hiddens)
            # Pre-confusions is a list of num_labels tensors:
            # <float32> [batch_size * num_predictor_samples, num_classes, num_classes, latent_size].
            pre_q_confusions = []
            for nc in self.dataset.num_classes:
                pre_q_confusion = tf.layers.Dense(
                    units=(nc * nc * confusion_latent_size)
                )(hiddens)
                pre_q_confusions.append(
                    tf.reshape(pre_q_confusion, [-1, nc, nc, confusion_latent_size])
                )

            # Confusions is a list of num_labels tensors log-normalized along the last axis:
            # <float32> [batch_size, num_predictor_samples, num_classes, num_classes].
            confusions = []
            for pqc, nc in zip(pre_q_confusions, self.num_classes):
                c = tf.nn.log_softmax(tf.squeeze(pqc, axis=-1), axis=-1)
                c = tf.reshape(c, [-1, self.num_predictor_samples, nc, nc])
                confusions.append(c)

        # Compute KL between priors and posteriors (regularization terms).
        prior = tfd.MultivariateNormalDiag(
            loc=tf.zeros(self.predictors_emb_size),
            scale_diag=tf.ones(self.predictors_emb_size)
        )
        regularization_terms = []
        for p_approx_posterior in approx_posteriors:
            p_kl = tfd.kl_divergence(p_approx_posterior, prior)
            regularization_terms.append(self.gamma * p_kl)

        return BuiltVariationalModel(
            regularization_terms=regularization_terms,
            predictor_embeddings=predictor_embeddings,
            approx_posteriors=approx_posteriors,
            predictions=predictions,
            confusions=confusions
        )
