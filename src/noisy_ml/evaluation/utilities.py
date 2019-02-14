"""Evaluation utility functions."""

import logging
import numpy as np
import tensorflow as tf

from sklearn.model_selection import KFold

__all__ = ["k_fold_cv"]

logger = logging.getLogger(__name__)


def k_fold_cv(
    kw_args,
    learner_fn,
    dataset,
    num_folds=5,
    batch_size=128,
    warm_start=True,
    max_m_steps=10000,
    max_em_steps=5,
    log_m_steps=1000,
    em_step_callback=lambda l: l,
    seed=None,
):
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
    logger.info("K-Fold CV - Initializing.")

    if seed is not None:
        np.random.set_state(seed)

    train_data = dataset.to_train(shuffle=False)
    kf = KFold(n_splits=num_folds)

    learner_scores = []
    for learner_kw_args in kw_args:
        logger.info("K-Fold CV - Evaluating kw_args: %s" % learner_kw_args)

        with tf.Graph().as_default():
            if seed is not None:
                tf.random.set_random_seed(seed)

            learner = learner_fn(**learner_kw_args)
            scores = []
            folds = kf.split(train_data[0])
            for f, (train_ids, test_ids) in enumerate(folds):
                logger.info("K-Fold CV - Fold %d / %d" % (f, num_folds))

                learner.reset()

                # Train the learner.
                train_dataset = tf.data.Dataset.from_tensor_slices(
                    {
                        "instances": train_data.instances[train_ids],
                        "predictors": train_data.predictors[train_ids],
                        "labels": train_data.labels[train_ids],
                        "values": train_data.values[train_ids],
                    }
                )
                learner.train(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    warm_start=warm_start,
                    max_m_steps=max_m_steps,
                    max_em_steps=max_em_steps,
                    log_m_steps=log_m_steps,
                    em_step_callback=em_step_callback,
                )

                # Test the learner.
                test_dataset = tf.data.Dataset.from_tensor_slices(
                    {
                        "instances": train_data.instances[test_ids],
                        "predictors": train_data.predictors[test_ids],
                        "labels": train_data.labels[test_ids],
                        "values": train_data.values[test_ids],
                    }
                )
                score = -learner.neg_log_likelihood(
                    dataset=test_dataset, batch_size=batch_size
                )
                scores.append(score)

                logger.info(
                    "K-Fold CV - Fold %d / %d score %10.4f for kw_args: %s"
                    % (f, num_folds, score, learner_kw_args)
                )

            score = float(np.mean(scores))
            learner_scores.append(score)
            logger.info(
                "K-Fold CV - Mean score %10.4f for kw_args: %s"
                % (score, learner_kw_args)
            )
    best_l = int(np.argmax(learner_scores))
    best_kw_args = kw_args[best_l]
    logger.info(
        "K-Fold CV - Best score was %10.4f for kw_args: %s"
        % (learner_scores[best_l], best_kw_args)
    )
    return learner_fn(**best_kw_args)
