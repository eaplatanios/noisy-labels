"""A collection of learners from noisy data."""

import abc
import logging
import os

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tqdm import tqdm

from .models import MMCE_M
from .models import VariationalNoisyModel
from .utilities import log1mexp

SHUFFLE_BUFFER_SIZE = 10000

__all__ = [
    "EMLearner",
    "GeneralizedStochasticEMLearner",
    "EMConfig",
    "MultiLabelEMConfig",
    "MultiLabelMultiClassEMConfig",
    "MultiLabelMultiClassGEMConfig",
]

logger = logging.getLogger(__name__)

tfd = tfp.distributions


class Learner(abc.ABC):
    """Defines the interface for Learners."""

    @abc.abstractmethod
    def train(self, dataset, **kwargs):
        raise NotImplementedError("Abstract method")


class EMLearner(Learner):
    def __init__(self, config, predictions_output_fn=lambda x: x):
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
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self._session = tf.Session(config=config)
            self._session.run(self._init_op)

    def _e_step(self, iterator_init_op, use_maj):
        self._session.run(iterator_init_op)
        self._session.run(self._ops["e_step_init"])
        if use_maj:
            self._session.run(self._ops["set_use_maj"])
        else:
            self._session.run(self._ops["unset_use_maj"])
        while True:
            try:
                self._session.run(self._ops["e_step"])
            except tf.errors.OutOfRangeError:
                break

    def _m_step(self, iterator_init_op, warm_start, max_m_steps, log_m_steps=None):
        self._session.run(iterator_init_op)
        if not warm_start:
            self._session.run(self._ops["m_step_init"])
        accumulator_nll = 0.0
        accumulator_steps = 0
        for m_step in range(max_m_steps):
            nll, _ = self._session.run(
                [self._ops["neg_log_likelihood"], self._ops["m_step"]]
            )
            accumulator_nll += nll
            accumulator_steps += 1
            if log_m_steps is not None and (
                m_step % log_m_steps == 0 or m_step == max_m_steps - 1
            ):
                nll = accumulator_nll / accumulator_steps
                logger.info("Step: %5d | Negative Log-Likelihood: %.8f" % (m_step, nll))
                accumulator_nll = 0.0
                accumulator_steps = 0

    def train(
        self,
        dataset,
        batch_size=128,
        warm_start=False,
        max_m_steps=1000,
        max_em_steps=100,
        log_m_steps=100,
        max_marginal_steps=0,
        em_step_callback=None,
        use_progress_bar=False,
    ):
        e_step_dataset = dataset.batch(batch_size)
        m_step_dataset = dataset.repeat().shuffle(10000).batch(batch_size)

        self._init_session()
        e_step_iterator_init_op = self._ops["train_iterator"].make_initializer(
            e_step_dataset
        )
        m_step_iterator_init_op = self._ops["train_iterator"].make_initializer(
            m_step_dataset
        )

        em_steps_range = range(max_em_steps)
        if use_progress_bar:
            em_steps_range = tqdm(em_steps_range, "EM Step (%s)" % os.getpid())

        for em_step in em_steps_range:
            if not use_progress_bar:
                logger.info("Iteration %d - Running E-Step" % em_step)
            self._e_step(e_step_iterator_init_op, use_maj=(em_step == 0))
            if not use_progress_bar:
                logger.info("Iteration %d - Running M-Step" % em_step)
            self._m_step(m_step_iterator_init_op, warm_start, max_m_steps, log_m_steps)
            if em_step_callback is not None:
                em_step_callback(self)

        if max_marginal_steps > 0:
            self._session.run(self._ops["set_opt_marginal"])
            logger.info("Optimizing marginal log-likelihood.")
            self._m_step(
                m_step_iterator_init_op, warm_start, max_marginal_steps, log_m_steps
            )
            self._session.run(self._ops["unset_opt_marginal"])

    def neg_log_likelihood(self, dataset, batch_size=128):
        dataset = dataset.batch(batch_size)
        iterator_init_op = self._ops["train_iterator"].make_initializer(dataset)

        self._init_session()
        self._session.run(iterator_init_op)
        neg_log_likelihood = 0.0
        while True:
            try:
                neg_log_likelihood += self._session.run(self._ops["neg_log_likelihood"])
            except tf.errors.OutOfRangeError:
                break
        return neg_log_likelihood

    def predict(self, instances, batch_size=128):
        # TODO: Remove hack by having separate train and predict iterators.
        dataset = tf.data.Dataset.from_tensor_slices(
            {
                "instances": instances,
                "predictors": np.zeros([len(instances)], np.int32),
                "labels": np.zeros([len(instances)], np.int32),
                "values": np.zeros([len(instances)], np.float32),
            }
        ).batch(batch_size)
        iterator_init_op = self._ops["train_iterator"].make_initializer(dataset)

        self._init_session()
        self._session.run(iterator_init_op)
        predictions = []
        while True:
            try:
                p = self._session.run(self._ops["predictions"])
                if self.predictions_output_fn is not None:
                    p = self.predictions_output_fn(p)
                predictions.append(p)
            except tf.errors.OutOfRangeError:
                break
        if isinstance(predictions[0], list):
            predictions = [np.concatenate(p, axis=0) for p in zip(*predictions)]
        else:
            predictions = np.concatenate(predictions, axis=0)
        return predictions

    def qualities(self, instances, predictors, labels, batch_size=128):
        # TODO: Move this function to utils and document.
        def cartesian_transpose(arrays):
            la = len(arrays)
            dtype = np.result_type(*arrays)
            arr = np.empty([la] + [len(a) for a in arrays], dtype=dtype)
            for i, a in enumerate(np.ix_(*arrays)):
                arr[i, ...] = a
            return arr.reshape(la, -1)

        temp = cartesian_transpose(
            [np.array(instances), np.array(labels), np.array(predictors)]
        )

        # TODO: Remove hack by having separate train and predict iterators.
        dataset = tf.data.Dataset.from_tensor_slices(
            {
                "instances": temp[0].astype(np.int32),
                "predictors": temp[2].astype(np.int32),
                "labels": temp[1].astype(np.int32),
                "values": np.zeros([len(temp[0])], np.float32),
            }
        ).batch(batch_size)
        iterator_init_op = self._ops["train_iterator"].make_initializer(dataset)

        self._init_session()

        self._session.run(iterator_init_op)
        qualities_mean_log = []
        while True:
            try:
                qualities_mean_log.append(
                    self._session.run(self._ops["qualities_mean_log"])
                )
            except tf.errors.OutOfRangeError:
                break

        if isinstance(qualities_mean_log[0], list):
            # Manually scatter computed qualities...
            qualities = np.zeros([len(instances), len(labels), len(predictors)])
            for l, q in zip(range(self.config.num_labels), zip(*qualities_mean_log)):
                indices_l = np.where(temp[1].astype(np.int32) == l)[0]
                qualities_l = np.exp(np.concatenate(q, axis=0))
                for j, i in enumerate(indices_l):
                    qualities[temp[0][i], l, temp[2][i]] = qualities_l[j]
        else:
            qualities = np.reshape(
                np.exp(np.concatenate(qualities_mean_log, axis=0)),
                [len(instances), len(labels), len(predictors)],
            )
        return qualities


class GeneralizedStochasticEMLearner(Learner):
    """Generalized EM that alternates between partial E-steps and
    M-steps based on a single (stochastic) gradient update.
    """

    def __init__(self, config, predictions_output_fn=lambda x: x):
        self.config = config
        self.predictions_output_fn = predictions_output_fn
        self._build_ops()
        self._session = None

    def _build_ops(self):
        self._ops = self.config.build_ops()
        self._init_op = tf.global_variables_initializer()

    def _init_session(self):
        if self._session is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self._session = tf.Session(config=config)
            self._session.run(self._init_op)

    def _update(self):
        nll, em_nll_term0, em_nll_term1, kl_reg, _ = self._session.run([
            self._ops["neg_log_likelihood"],
            self._ops["em_nll_term0"],
            self._ops["em_nll_term1"],
            self._ops["kl_reg"],
            self._ops["m_step"]])
        return nll, em_nll_term0, em_nll_term1, kl_reg

    def reset(self):
        self._session = None

    def train(
        self,
        dataset,
        batch_size=128,
        max_em_steps=1000,
        log_em_steps=10,
        em_step_callback=None,
        use_progress_bar=False,
    ):
        # Init.
        self._init_session()

        # Batch dataset.
        dataset = dataset.repeat().shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)
        iterator_init_op = self._ops["train_iterator"].make_initializer(dataset)
        self._session.run(iterator_init_op)

        logger.info("Running GEM for %d steps..." % max_em_steps)

        # Run GEM steps.
        accumulator_nll = 0.0
        accumulator_em_nll_term0 = 0.0
        accumulator_em_nll_term1 = 0.0
        accumulator_kl_reg = [0.0 for _ in range(10)]
        accumulator_steps = 0
        for step in range(max_em_steps):
            nll, em_nll_term0, em_nll_term1, kl_reg = self._update()
            accumulator_nll += nll
            accumulator_em_nll_term0 += em_nll_term0
            accumulator_em_nll_term1 += em_nll_term1
            accumulator_kl_reg = [v + u for v, u in zip(accumulator_kl_reg, kl_reg)]
            accumulator_steps += 1
            if (step % log_em_steps == 0) or (step == max_em_steps - 1):
                nll = accumulator_nll / accumulator_steps
                em_nll_term0 = accumulator_em_nll_term0 / accumulator_steps
                em_nll_term1 = accumulator_em_nll_term1 / accumulator_steps
                kl_reg = [v / accumulator_steps for v in accumulator_kl_reg]
                logger.info("Step: %5d | Negative Log-Likelihood: %.8f" % (step, nll))
                # logger.info("Step: %5d | EM NLL term0: %.8f" % (step, em_nll_term0))
                # logger.info("Step: %5d | EM NLL term1: %.8f" % (step, em_nll_term1))
                # logger.info("Step: %5d | KL terms: %s" % (step, kl_reg))
                accumulator_nll = 0.0
                accumulator_steps = 0

            if em_step_callback is not None:
                em_step_callback(self)

        logger.info("Done.")

    def neg_log_likelihood(self, dataset, batch_size=128):
        dataset = dataset.batch(batch_size)
        iterator_init_op = self._ops["train_iterator"].make_initializer(dataset)

        self._init_session()
        self._session.run(iterator_init_op)
        neg_log_likelihood = 0.0
        while True:
            try:
                neg_log_likelihood += self._session.run(self._ops["neg_log_likelihood"])
            except tf.errors.OutOfRangeError:
                break
        return neg_log_likelihood

    def predict(self, instances, batch_size=128):
        # TODO: Remove hack by having separate train and predict iterators.
        dataset = tf.data.Dataset.from_tensor_slices(
            {
                "instances": instances,
                "predictors": np.zeros([len(instances)], np.int32),
                "labels": np.zeros([len(instances)], np.int32),
                "values": np.zeros([len(instances)], np.float32),
            }
        ).batch(batch_size)
        iterator_init_op = self._ops["train_iterator"].make_initializer(dataset)

        self._init_session()
        self._session.run(iterator_init_op)
        predictions = []
        while True:
            try:
                p = self._session.run(self._ops["predictions"])
                if self.predictions_output_fn is not None:
                    p = self.predictions_output_fn(p)
                predictions.append(p)
            except tf.errors.OutOfRangeError:
                break
        if isinstance(predictions[0], list):
            predictions = [np.concatenate(p, axis=0) for p in zip(*predictions)]
        else:
            predictions = np.concatenate(predictions, axis=0)
        return predictions

    def qualities(self, instances, predictors, labels, batch_size=128):
        # TODO: Move this function to utils and document.
        def cartesian_transpose(arrays):
            la = len(arrays)
            dtype = np.result_type(*arrays)
            arr = np.empty([la] + [len(a) for a in arrays], dtype=dtype)
            for i, a in enumerate(np.ix_(*arrays)):
                arr[i, ...] = a
            return arr.reshape(la, -1)

        temp = cartesian_transpose(
            [np.array(instances), np.array(labels), np.array(predictors)]
        )

        # TODO: Remove hack by having separate train and predict iterators.
        dataset = tf.data.Dataset.from_tensor_slices(
            {
                "instances": temp[0].astype(np.int32),
                "predictors": temp[2].astype(np.int32),
                "labels": temp[1].astype(np.int32),
                "values": np.zeros([len(temp[0])], np.float32),
            }
        ).batch(batch_size)
        iterator_init_op = self._ops["train_iterator"].make_initializer(dataset)

        self._init_session()

        self._session.run(iterator_init_op)
        qualities_mean_log = []
        while True:
            try:
                qualities_mean_log.append(
                    self._session.run(self._ops["qualities_mean_log"])
                )
            except tf.errors.OutOfRangeError:
                break

        if isinstance(qualities_mean_log[0], list):
            # Manually scatter computed qualities...
            qualities = np.zeros([len(instances), len(labels), len(predictors)])
            for l, q in zip(range(self.config.num_labels), zip(*qualities_mean_log)):
                indices_l = np.where(temp[1].astype(np.int32) == l)[0]
                qualities_l = np.exp(np.concatenate(q, axis=0))
                for j, i in enumerate(indices_l):
                    qualities[temp[0][i], l, temp[2][i]] = qualities_l[j]
        else:
            qualities = np.reshape(
                np.exp(np.concatenate(qualities_mean_log, axis=0)),
                [len(instances), len(labels), len(predictors)],
            )
        return qualities


class EMConfig(abc.ABC):
    """Defines an interface for EM configurations that specify the ops being built."""

    @abc.abstractmethod
    def build_ops(self):
        raise NotImplementedError


class MultiLabelEMConfig(EMConfig):
    def __init__(
        self,
        num_instances,
        num_predictors,
        num_labels,
        model,
        optimizer,
        lambda_entropy=0.0,
        use_soft_maj=True,
        use_soft_y_hat=False,
    ):
        super(MultiLabelEMConfig, self).__init__()
        self.num_instances = num_instances
        self.num_predictors = num_predictors
        self.num_labels = num_labels
        self.model = model
        self.optimizer = optimizer
        self.lambda_entropy = lambda_entropy
        self.use_soft_maj = use_soft_maj
        self.use_soft_y_hat = use_soft_y_hat

    def build_ops(self):
        train_iterator = tf.data.Iterator.from_structure(
            output_types={
                "instances": tf.int32,
                "predictors": tf.int32,
                "labels": tf.int32,
                "values": tf.float32,
            },
            output_shapes={
                "instances": [None],
                "predictors": [None],
                "labels": [None],
                "values": [None],
            },
            shared_name="train_iterator",
        )
        iter_next = train_iterator.get_next()
        instances = tf.placeholder_with_default(
            iter_next["instances"], shape=[None], name="instances"
        )
        predictors = tf.placeholder_with_default(
            iter_next["predictors"], shape=[None], name="predictors"
        )
        labels = tf.placeholder_with_default(
            iter_next["labels"], shape=[None], name="labels"
        )
        values = tf.placeholder_with_default(
            iter_next["values"], shape=[None], name="values"
        )

        x_indices = instances
        p_indices = predictors
        l_indices = labels

        predictions, q_params, regularization_terms, include_y_prior = self.model.build(
            x_indices, p_indices, l_indices
        )

        # y_hat has shape: [BatchSize, 2]
        # y_hat[i, 0] is the probability that instance i has label 0.
        y_hat = tf.stack([1.0 - values, values], axis=-1)
        if not self.use_soft_maj:
            y_hat = tf.cast(tf.greater_equal(y_hat, 0.5), tf.int32)

        # TODO: Is this necessary?
        h_log = tf.minimum(predictions, -1e-6)
        h_log = tf.squeeze(
            tf.batch_gather(params=h_log, indices=tf.expand_dims(l_indices, axis=-1)),
            axis=-1,
        )
        h_log = tf.stack([log1mexp(h_log), h_log], axis=-1)

        # q_params shape: [BatchSize, 2, 2]
        q_log = q_params
        h_log_q_log = q_log + tf.expand_dims(h_log, axis=-1)
        qualities_mean_log = tf.stack(
            [h_log_q_log[:, 1, 1], h_log_q_log[:, 0, 0]], axis=-1
        )
        qualities_mean_log = tf.reduce_logsumexp(qualities_mean_log, axis=-1)

        # The following represent the qualities that correspond
        # to the provided predictor values.
        q_log_y_hat = tf.einsum("ijk,ik->ij", q_log, tf.cast(y_hat, tf.float32))

        # E-Step:

        with tf.name_scope("e_step"):
            # Create the accumulator variables:
            e_y_acc = tf.get_variable(
                name="e_y_acc",
                shape=[self.num_instances, self.num_labels, 2],
                initializer=tf.zeros_initializer(h_log.dtype),
                trainable=False,
            )

            # Boolean flag about whether or not to use majority vote
            # estimates for the E-step expectations.
            use_maj = tf.get_variable(
                name="use_maj",
                shape=[],
                dtype=tf.bool,
                initializer=tf.zeros_initializer(),
                trainable=False,
            )
            set_use_maj = use_maj.assign(True)
            unset_use_maj = use_maj.assign(False)

            e_y_acc_update = tf.cond(
                use_maj, lambda: tf.cast(y_hat, tf.float32), lambda: q_log_y_hat
            )

            # Create the accumulator variable update ops.
            xl_indices = tf.stack([x_indices, l_indices], axis=-1)
            e_y_acc_update = e_y_acc.scatter_nd_add(
                indices=xl_indices, updates=e_y_acc_update
            )

            e_y_a = tf.gather_nd(e_y_acc, xl_indices)
            e_y_a_h_log = e_y_a
            # if include_y_prior:
            #   e_y_a_h_log += h_log
            # else:
            #   e_y_a_h_log += np.log(0.5)
            e_y_log = tf.stop_gradient(
                tf.cond(
                    use_maj,
                    lambda: tf.log(e_y_a)
                    - tf.log(tf.reduce_sum(e_y_a, axis=-1, keepdims=True)),
                    lambda: e_y_a_h_log
                    - tf.reduce_logsumexp(e_y_a_h_log, axis=-1, keepdims=True),
                )
            )
            if isinstance(self.model, MMCE_M):
                e_y = tf.exp(e_y_log)
            else:
                e_y = tf.cast(tf.greater_equal(tf.exp(e_y_log), 0.5), e_y_log.dtype)

            e_step_init = e_y_acc.initializer
            e_step = e_y_acc_update

        # M-Step:

        with tf.name_scope("m_step"):
            # Boolean flag about whether or not to optimize the
            # marginal log-likelihood.
            opt_marginal = tf.get_variable(
                name="opt_marginal",
                shape=[],
                dtype=tf.bool,
                initializer=tf.zeros_initializer(),
                trainable=False,
            )
            set_opt_marginal = opt_marginal.assign(True)
            unset_opt_marginal = opt_marginal.assign(False)

            em_ll_term0 = tf.reduce_sum(e_y * h_log)
            # TODO: isn't the below just a plain reduce_sum?
            em_ll_term1 = tf.reduce_sum(tf.einsum("ij,ij->i", q_log_y_hat, e_y))
            em_nll = -em_ll_term0 - em_ll_term1

            marginal_ll = q_log_y_hat + h_log
            marginal_ll = tf.reduce_logsumexp(marginal_ll, axis=-1)
            marginal_ll = -tf.reduce_sum(marginal_ll)

            neg_log_likelihood = tf.cond(
                opt_marginal, lambda: marginal_ll, lambda: em_nll
            )
            if len(regularization_terms) > 0:
                neg_log_likelihood += tf.add_n(regularization_terms)

            # Entropy regularization.
            h_entropy = tf.reduce_sum(tf.exp(h_log) * h_log)
            neg_log_likelihood += self.lambda_entropy * h_entropy

            m_step_init = tf.variables_initializer(tf.trainable_variables())
            global_step = tf.train.get_or_create_global_step()
            gvs = self.optimizer.compute_gradients(
                neg_log_likelihood, tf.trainable_variables()
            )
            gradients, variables = zip(*gvs)
            # gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            m_step = self.optimizer.apply_gradients(
                zip(gradients, variables), global_step=global_step
            )

        return {
            "train_iterator": train_iterator,
            "x_indices": x_indices,
            "p_indices": p_indices,
            "l_indices": l_indices,
            "predictions": predictions,
            "qualities_mean_log": qualities_mean_log,
            "set_use_maj": set_use_maj,
            "unset_use_maj": unset_use_maj,
            "set_opt_marginal": set_opt_marginal,
            "unset_opt_marginal": unset_opt_marginal,
            "e_step_init": e_step_init,
            "m_step_init": m_step_init,
            "e_step": e_step,
            "m_step": m_step,
            "neg_log_likelihood": neg_log_likelihood,
        }


class MultiLabelMultiClassEMConfig(EMConfig):
    """EMConfig for multi-label, multi-class problems."""

    def __init__(
        self,
        num_instances,
        num_predictors,
        num_labels,
        num_classes,
        model,
        optimizer,
        lambda_entropy=0.0,
        use_soft_maj=True,
        use_soft_y_hat=False,
    ):
        super(MultiLabelMultiClassEMConfig, self).__init__()
        self.num_instances = num_instances
        self.num_predictors = num_predictors
        self.num_labels = num_labels
        self.num_classes = num_classes
        self.model = model
        self.optimizer = optimizer
        self.lambda_entropy = lambda_entropy
        self.use_soft_maj = use_soft_maj
        self.use_soft_y_hat = use_soft_y_hat

    def build_ops(self):
        # TODO: refactor to eliminate code duplication.
        train_iterator = tf.data.Iterator.from_structure(
            output_types={
                "instances": tf.int32,
                "predictors": tf.int32,
                "labels": tf.int32,
                "values": tf.float32,
            },
            output_shapes={
                "instances": [None],
                "predictors": [None],
                "labels": [None],
                "values": [None],
            },
            shared_name="train_iterator",
        )
        iter_next = train_iterator.get_next()
        instances = tf.placeholder_with_default(
            iter_next["instances"], shape=[None], name="instances"
        )
        predictors = tf.placeholder_with_default(
            iter_next["predictors"], shape=[None], name="predictors"
        )
        labels = tf.placeholder_with_default(
            iter_next["labels"], shape=[None], name="labels"
        )
        values = tf.placeholder_with_default(
            iter_next["values"], shape=[None], name="values"
        )

        x_indices = instances
        p_indices = predictors
        l_indices = labels

        # Build model ops.
        # predictions: list of <float32> [batch_size, num_classes_l] for each label l.
        # q_params: list of <float32> [batch_size, num_classes_l, num_classes_l] for each label l.
        predictions, q_params, regularization_terms, include_y_prior = self.model.build(
            x_indices, p_indices, l_indices
        )

        # Convert label indices into one-hot masks.
        # l_indices_onehot: list of <int32> [batch_size_l] for each label.
        l_indices_onehot = tf.unstack(tf.one_hot(l_indices, self.num_labels), axis=-1)

        # Slice instance indices for each label.
        # xl_indices: list of <int32> [batch_size_l] for each label l.
        xl_indices = []
        for l in l_indices_onehot:
            xl_indices.append(tf.boolean_mask(x_indices, l))

        # Slice predictions and confusions for each label.
        for j, (p, c, l) in enumerate(zip(predictions, q_params, l_indices_onehot)):
            predictions[j] = tf.boolean_mask(p, l)
            q_params[j] = tf.boolean_mask(c, l)

        # Convert values in y_hats.
        # y_hats: list of  <float32> [batch_size, num_classes] for each label.
        # y_hats[l][i, k] is the probability/indicator that instance i has value k for label l.
        y_hats = []
        for l_mask, nc in zip(l_indices_onehot, self.num_classes):
            values_l = tf.boolean_mask(values, l_mask)
            if self.use_soft_y_hat:
                y_hat = tf.stack([1.0 - values_l, values_l], axis=-1)
            else:
                y_hat = tf.one_hot(tf.to_int32(values_l), nc)
            y_hats.append(y_hat)

        # Compute mean qualities.
        # qualities_mean_logs: list of <float32> [batch_size_l] for each label l.
        qualities_mean_log = []
        for h_log, q_log in zip(predictions, q_params):
            h_log_q_log = q_log + tf.expand_dims(h_log, axis=-1)
            qualities_mean_log.append(
                # <float32> [batch_size_l].
                tf.reduce_logsumexp(
                    # <float32> [batch_size_l, num_classes_l].
                    tf.matrix_diag_part(h_log_q_log),
                    axis=-1,
                )
            )

        # Compute confusion values for each y_hat.
        # q_log_y_hats: list of <float32> [batch_size_l, num_classes_l] for each label l.
        q_log_y_hats = []
        for y_hat, q_log in zip(y_hats, q_params):
            q_log_y_hats.append(tf.einsum("ijk,ik->ij", q_log, tf.to_float(y_hat)))

        # E-Step:

        with tf.name_scope("e_step"):
            # Create the accumulator variables.
            # e_y_accs: list of <float32> [num_instances, num_classes_l] for each label l.
            e_y_accs = [
                tf.get_variable(
                    name=("e_y_acc_%d" % l),
                    shape=[self.num_instances, nc],
                    initializer=tf.zeros_initializer(h_log.dtype),
                    trainable=False,
                )
                for l, (nc, h_log) in enumerate(zip(self.num_classes, predictions))
            ]

            # Boolean flag for whether or not to use majority vote
            # estimates for the E-step expectations.
            use_maj = tf.get_variable(
                name="use_maj",
                shape=[],
                dtype=tf.bool,
                initializer=tf.zeros_initializer(),
                trainable=False,
            )
            set_use_maj = use_maj.assign(True)
            unset_use_maj = use_maj.assign(False)

            # Create the accumulator variable update ops.
            # list of <float32> [batch_size_l, num_classes_l] for each label l.
            e_y_acc_updates = [
                tf.cond(use_maj, lambda: tf.to_float(y_hat), lambda: q_log_y_hat)
                for y_hat, q_log_y_hat in zip(y_hats, q_log_y_hats)
            ]
            # list of <float32> [num_instances, num_classes_l] for each label l.
            e_y_accs_updated = [
                tf.scatter_add(ref=e_y_acc, indices=xli, updates=e_y_acc_update)
                for xli, e_y_acc, e_y_acc_update in zip(
                    xl_indices, e_y_accs, e_y_acc_updates
                )
            ]

            # TODO: what the hell is this mess? :P (I mean variable names.)
            # list of <float32> [batch_size_l, num_classes_l] for each label l.
            e_y_as = [
                tf.gather(e_y_acc, xli) for xli, e_y_acc in zip(xl_indices, e_y_accs)
            ]
            # list of <float32> [batch_size_l, num_classes_l] for each label l.
            e_y_a_h_logs = e_y_as
            # if include_y_prior:
            #   e_y_a_h_logs = [e_y_a + h_log for e_y_a, h_log in zip(e_y_as, predictions)]
            # else:
            #   e_y_a_h_log = [e_y_a + np.log(0.5) for e_y_a in e_y_as]
            # list of <float32> [batch_size_l, num_classes_l] for each label l.
            e_y_logs = [
                tf.stop_gradient(
                    tf.cond(
                        use_maj,
                        lambda: tf.log(e_y_a)
                        - tf.log(tf.reduce_sum(e_y_a, axis=-1, keepdims=True)),
                        lambda: e_y_a_h_log
                        - tf.reduce_logsumexp(e_y_a_h_log, axis=-1, keepdims=True),
                    )
                )
                for e_y_a, e_y_a_h_log in zip(e_y_as, e_y_a_h_logs)
            ]
            # e_ys: list of <float32> [batch_size_l, num_classes_l] for each label l.
            if isinstance(self.model, MMCE_M):
                e_ys = [tf.exp(e_y_log) for e_y_log in e_y_logs]
            else:
                e_ys = [
                    tf.cast(tf.greater_equal(tf.exp(e_y_log), 0.5), e_y_log.dtype)
                    for e_y_log in e_y_logs
                ]

            # Build E-step.
            e_step_init = [e_y_acc.initializer for e_y_acc in e_y_accs]
            e_step = e_y_accs_updated

        # M-Step:

        with tf.name_scope("m_step"):
            # Boolean flag about whether or not to optimize the
            # marginal log-likelihood.
            opt_marginal = tf.get_variable(
                name="opt_marginal",
                shape=[],
                dtype=tf.bool,
                initializer=tf.zeros_initializer(),
                trainable=False,
            )
            set_opt_marginal = opt_marginal.assign(True)
            unset_opt_marginal = opt_marginal.assign(False)

            # Compute EM NLL: <float32> [].
            em_ll_term0 = sum(
                tf.reduce_sum(e_y * h_log) for e_y, h_log in zip(e_ys, predictions)
            )
            em_ll_term1 = sum(
                tf.reduce_sum(q_log_y_hat * e_y)
                for q_log_y_hat, e_y in zip(q_log_y_hats, e_ys)
            )
            em_nll = -(em_ll_term0 + em_ll_term1)

            # Compute marginal NLL: <float32> [].
            marginal_ll_list = [
                q_log_y_hat + h_log
                for q_log_y_hat, h_log in zip(q_log_y_hats, predictions)
            ]
            marginal_nll = -sum(
                [
                    tf.reduce_sum(tf.reduce_logsumexp(mll, axis=-1))
                    for mll in marginal_ll_list
                ]
            )

            # Compute the final NLL objective: <float32> [].
            neg_log_likelihood = tf.cond(
                opt_marginal, lambda: marginal_nll, lambda: em_nll
            )
            if len(regularization_terms) > 0:
                neg_log_likelihood += tf.add_n(regularization_terms)

            # Entropy regularization.
            h_entropy = sum(
                [tf.reduce_sum(tf.exp(h_log) * h_log) for h_log in predictions]
            )
            neg_log_likelihood += self.lambda_entropy * h_entropy

            # Build M-step.
            m_step_init = tf.variables_initializer(tf.trainable_variables())
            global_step = tf.train.get_or_create_global_step()
            gvs = self.optimizer.compute_gradients(
                neg_log_likelihood, tf.trainable_variables()
            )
            gradients, variables = zip(*gvs)
            # gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            m_step = self.optimizer.apply_gradients(
                zip(gradients, variables), global_step=global_step
            )

        return {
            "train_iterator": train_iterator,
            "x_indices": x_indices,
            "p_indices": p_indices,
            "l_indices": l_indices,
            "predictions": predictions,
            "qualities_mean_log": qualities_mean_log,
            "set_use_maj": set_use_maj,
            "unset_use_maj": unset_use_maj,
            "set_opt_marginal": set_opt_marginal,
            "unset_opt_marginal": unset_opt_marginal,
            "e_step_init": e_step_init,
            "m_step_init": m_step_init,
            "e_step": e_step,
            "m_step": m_step,
            "neg_log_likelihood": neg_log_likelihood,
        }


class MultiLabelMultiClassGEMConfig(EMConfig):
    """Constructs ops for multi-label multi-class GEM."""

    def __init__(
        self,
        num_instances,
        num_predictors,
        num_labels,
        num_classes,
        model,
        optimizer,
        lambda_entropy=0.0,
    ):
        super(MultiLabelMultiClassGEMConfig, self).__init__()
        self.num_instances = num_instances
        self.num_predictors = num_predictors
        self.num_labels = num_labels
        self.num_classes = num_classes
        self.model = model
        self.optimizer = optimizer
        self.lambda_entropy = lambda_entropy

    def build_ops(self):
        # TODO: refactor to eliminate code duplication.
        train_iterator = tf.data.Iterator.from_structure(
            output_types={
                "instances": tf.int32,
                "predictors": tf.int32,
                "labels": tf.int32,
                "values": tf.float32,
            },
            output_shapes={
                "instances": [None],
                "predictors": [None],
                "labels": [None],
                "values": [None],
            },
            shared_name="train_iterator",
        )
        iter_next = train_iterator.get_next()
        instances = tf.placeholder_with_default(
            iter_next["instances"], shape=[None], name="instances"
        )
        predictors = tf.placeholder_with_default(
            iter_next["predictors"], shape=[None], name="predictors"
        )
        labels = tf.placeholder_with_default(
            iter_next["labels"], shape=[None], name="labels"
        )
        values = tf.placeholder_with_default(
            iter_next["values"], shape=[None], name="values"
        )

        x_indices = instances
        p_indices = predictors
        l_indices = labels

        # Build model ops.
        assert isinstance(self.model, VariationalNoisyModel)
        # approx_posteriors: list of tfd.MultivariateNormalDiag for each predictor.
        # predictions: list of <float32> [batch_size, num_classes_l] for each label l.
        # confusions: list of <float32>
        #   [batch_size, num_predictor_samples, num_classes_l, num_classes_l] for each label l.
        (
            regularization_terms,
            predictor_embeddings,
            approx_posteriors,
            predictions,
            confusions
        ) = self.model.build(
            x_indices, p_indices, l_indices, values
        )

        # Convert label indices into one-hot masks.
        # l_indices_onehot: list of <int32> [batch_size] for each label.
        l_indices_onehot = tf.unstack(tf.one_hot(l_indices, self.num_labels), axis=-1)

        # Slice instance indices for each label.
        # xl_indices: list of <int32> [batch_size_l] for each label l.
        xl_indices = []
        for l in l_indices_onehot:
            xl_indices.append(tf.boolean_mask(x_indices, l))

        # Slice predictions and confusions for each label.
        for j, (p, c, l) in enumerate(zip(predictions, confusions, l_indices_onehot)):
            predictions[j] = tf.boolean_mask(p, l)
            confusions[j] = tf.boolean_mask(c, l)

        # Convert values in y_hats.
        # y_hats: list of  <float32> [batch_size, num_classes] for each label.
        # y_hats[l][i, k] is the probability/indicator that instance i has value k for label l.
        y_hats = []
        for l_mask, nc in zip(l_indices_onehot, self.num_classes):
            values_l = tf.boolean_mask(values, l_mask)
            y_hat = tf.one_hot(tf.to_int32(tf.round(values_l)), nc)
            y_hats.append(y_hat)

        # Compute mean qualities.
        # qualities_mean_logs: list of <float32> [batch_size_l] for each label l.
        qualities_mean_log = []
        for h_log, q_log in zip(predictions, confusions):
            h_log_expanded = tf.expand_dims(tf.expand_dims(h_log, axis=-1), axis=1)
            h_log_q_log = q_log + h_log_expanded
            qualities_mean_log.append(
                # <float32> [batch_size_l].
                tf.reduce_mean(
                    tf.reduce_logsumexp(
                        # <float32> [batch_size_l, num_predictor_samples, num_classes_l].
                        tf.matrix_diag_part(h_log_q_log),
                        axis=-1,
                    ),
                    axis=1,
                )
            )

        # Compute confusion values for each y_hat.
        # q_log_y_hats: list of
        #   <float32> [batch_size_l, num_predictor_samples, num_classes_l] for each label l.
        q_log_y_hats = []
        for y_hat, q_log in zip(y_hats, confusions):
            q_log_y_hats.append(tf.einsum("ijlk,ik->ijl", q_log, tf.to_float(y_hat)))

        # E-Step:

        with tf.name_scope("e_step"):
            # Compute masks that select identical instances.
            # <int32> [batch_size, num_instances].
            unique_instances_onehot = tf.one_hot(instances, self.num_instances)
            # <bool> [num_instances].
            batch_unique_instances_onehot = tf.reduce_any(
                tf.not_equal(unique_instances_onehot, 0.),
                axis=0
            )
            # <int32> [batch_size, num_unique_instances].
            unique_instances_onehot = tf.boolean_mask(
                unique_instances_onehot,
                batch_unique_instances_onehot,
                axis=1
            )

            # For each instance, compute the posterior p(y_i | x_i, {l_ij}, {r_j}).
            p_y_given_x_l_r = []
            for q_log_y_hat, h_log in zip(q_log_y_hats, predictions):
                # Compute log P({l}, y | x, {r}).
                # TODO: explain what's happening here...
                log_p = tf.einsum("ilk,ij->ijlk", q_log_y_hat, unique_instances_onehot)
                log_p = tf.einsum("ijlk->jlk", log_p)
                log_p = tf.einsum("jlk,ij->ijlk", log_p, unique_instances_onehot)
                log_p = tf.einsum("ijlk->ilk", log_p)
                log_p = log_p + tf.expand_dims(log_p, axis=1)
                # Compute P(y | x, {l}, {r}).
                p = tf.exp(log_p - tf.reduce_logsumexp(log_p, axis=-1, keepdims=True))
                p_y_given_x_l_r.append(p)

        # M-Step:

        with tf.name_scope("m_step"):
            # Compute EM NLL: <float32> [].
            em_ll_term0 = sum(
                tf.reduce_mean(tf.reduce_sum(p_y_post * tf.expand_dims(h_log, axis=1), axis=-1))
                for p_y_post, h_log in zip(p_y_given_x_l_r, predictions)
            )
            em_ll_term1 = sum(
                tf.reduce_mean(tf.reduce_sum(p_y_post * q_log_y_hat, axis=-1))
                for p_y_post, q_log_y_hat in zip(p_y_given_x_l_r, q_log_y_hats)
            )
            em_nll = -(em_ll_term0 + em_ll_term1)

            # Compute the final NLL objective: <float32> [].
            neg_log_likelihood = em_nll
            if len(regularization_terms) > 0:
                neg_log_likelihood += tf.add_n(regularization_terms)

            # Entropy regularization.
            if self.lambda_entropy > 0:
                h_entropy = sum(
                    [tf.reduce_sum(tf.exp(h_log) * h_log) for h_log in predictions]
                )
                neg_log_likelihood += self.lambda_entropy * h_entropy

            # Build M-step.
            m_step_init = tf.variables_initializer(tf.trainable_variables())
            global_step = tf.train.get_or_create_global_step()
            gvs = self.optimizer.compute_gradients(
                neg_log_likelihood, tf.trainable_variables()
            )
            gradients, variables = zip(*gvs)
            # gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            m_step = self.optimizer.apply_gradients(
                zip(gradients, variables), global_step=global_step
            )

        return {
            "train_iterator": train_iterator,
            "x_indices": x_indices,
            "p_indices": p_indices,
            "l_indices": l_indices,
            "predictions": predictions,
            "qualities_mean_log": qualities_mean_log,
            "m_step_init": m_step_init,
            "m_step": m_step,
            "neg_log_likelihood": neg_log_likelihood,
            "em_nll_term0": -em_ll_term0,
            "em_nll_term1": -em_ll_term1,
            "kl_reg": regularization_terms,
        }
