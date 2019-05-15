"""Utils for running experiments."""

import itertools
import logging
import os
import six

import random
import numpy as np
import tensorflow as tf

from ..data.crowdsourced import AgeLoader
from ..data.crowdsourced import BlueBirdsLoader
from ..data.medical import RelationExtractionLoader
from ..data.rte import RTELoader
from ..data.wordsim import WordSimLoader

from ..evaluation.metrics import Evaluator

from ..models.amsgrad import AMSGrad
from ..models.learners import EMLearner
from ..models.learners import MultiLabelEMConfig
from ..models.learners import MultiLabelMultiClassEMConfig
from ..models.models import MultiClassLNL

__all__ = [
    "reset_seed",
    "sample_predictors",
    "learner_fn",
    "get_dataset_setup",
    "get_models",
    "train_eval_predictors",
]

logger = logging.getLogger(__name__)


def reset_seed(seed=None):
    seed = seed or 1234567890
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def sample_predictors(predictors, num_to_sample, num_sets=5):
    """A generator that sub-samples sets of predictors from the provided."""
    if len(predictors) <= num_to_sample:
        yield predictors
    else:
        for _ in range(num_sets):
            yield random.sample(predictors, num_to_sample)


def learner_fn(
    model,
    dataset,
    learner_cls=EMLearner,
    config_cls=MultiLabelMultiClassEMConfig,
    optimizer="amsgrad",
    optimizer_kwargs=("lr", 1e-3),
):
    if optimizer == "amsgrad":
        optimizer = AMSGrad(**dict(optimizer_kwargs))
    else:
        optimizer = tf.train.AdamOptimizer(**dict(optimizer_kwargs))
    return learner_cls(
        config=config_cls(
            num_instances=len(dataset.instances),
            num_predictors=len(dataset.predictors),
            num_labels=len(dataset.labels),
            num_classes=dataset.num_classes,
            model=model,
            optimizer=optimizer,
            lambda_entropy=0.0,
            use_soft_maj=True,
            use_soft_y_hat=False,
        ),
        predictions_output_fn=np.exp,
    )


def get_dataset_setup(dataset, data_dir, results_dir):
    """Returns experimental setup parameters for the given dataset."""
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if dataset is "bluebirds":
        dataset = BlueBirdsLoader.load(data_dir, load_features=True)
        num_predictors = [1, 10, 20, 39]
        num_repetitions = [20, 10, 5, 1]
        results_path = os.path.join(results_dir, "bluebirds.csv")
    elif dataset is "rte":
        dataset = RTELoader.load(data_dir, load_features=True)
        num_predictors = [1, 10, 20, 50, 100, 164]
        num_repetitions = [20, 10, 10, 5, 3, 1]
        results_path = os.path.join(results_dir, "rte.csv")
    elif dataset is "wordsim":
        dataset = WordSimLoader.load(data_dir, load_features=True)
        num_predictors = [1, 2, 5, 10]
        num_repetitions = [20, 10, 5, 1]
        results_path = os.path.join(results_dir, "wordsim.csv")
    elif dataset is "age":
        dataset = AgeLoader.load(data_dir, load_features=True)
        num_predictors = [1, 2, 5, 10, 20, 50, 100, 165]
        num_repetitions = [20, 20, 20, 20, 10, 10, 3, 1]
        results_path = os.path.join(results_dir, "age.csv")
    elif dataset is "medical-causes":
        relations = ("CAUSES",)
        dataset = RelationExtractionLoader.load(
            data_dir, relations, load_features=True
        )
        num_predictors = [1, 10, 20, 50, 100, 200, 400, 467]
        num_repetitions = [20, 20, 20, 20, 10, 5, 3, 1]
        results_path = os.path.join(results_dir, "medical-causes.csv")
    elif dataset is "medical-treats":
        relations = ("TREATS",)
        dataset = RelationExtractionLoader.load(
            data_dir, relations, load_features=True
        )
        num_predictors = [1, 10, 20, 50, 100, 200, 400, 467]
        num_repetitions = [20, 20, 20, 20, 10, 5, 3, 1]
        results_path = os.path.join(results_dir, "medical-treats.csv")
    elif dataset is "medical-causes-treats":
        relations = ("CAUSES", "TREATS")
        dataset = RelationExtractionLoader.load(
            data_dir, relations, load_features=True
        )
        num_predictors = [1, 10, 20, 50, 100, 200, 400, 467]
        num_repetitions = [20, 20, 20, 20, 10, 5, 3, 1]
        results_path = os.path.join(results_dir, "medical-causes-treats.csv")
    else:
        raise NotImplementedError

    return dataset, num_predictors, num_repetitions, results_path


def get_models(
    dataset,
    instances_emb_size=(4, None),
    instances_hidden=([], [16, 16]),
    predictors_emb_size=(4, 16),
    predictors_hidden=([],),
    q_latent_size=(1,),
    gamma=(0.50, 0.75, 1.00),
):
    """Generates a dict of models for the specified parameters."""
    # Generate configurations.
    configuration_values = list(itertools.product(
        instances_emb_size,
        instances_hidden,
        predictors_emb_size,
        predictors_hidden,
        q_latent_size,
        gamma,
    ))
    configuration_names = len(configuration_values) * [[
        "instances_emb_size",
        "instances_hidden",
        "predictors_emb_size",
        "predictors_hidden",
        "q_latent_size",
        "gamma",
    ]]
    configuration_dicts = map(
        lambda x: dict(zip(*x)),
        zip(configuration_names, configuration_values)
    )

    models = {"MAJ": "MAJ"}
    for config in configuration_dicts:
        name = "LNL"
        name += "-F" if config["instances_emb_size"] is None else ""
        name += "%s" % config["instances_hidden"]
        name += " (γ=%.2f)" % config["gamma"]
        models[name] = MultiClassLNL(dataset, **config)

    return models


def train_eval_predictors(
    model,
    num_predictors,
    num_repetitions,
    dataset,
    batch_size=256,
    max_m_steps=2000,
    max_em_steps=20,
    max_marginal_steps=0,
    log_m_steps=None,
    warm_start=True,
    use_progress_bar=True,
    time_stamp=None,
    seed=None,
):
    """Runs train-eval loop for the given predictors."""
    reset_seed(seed)

    results = []
    predictor_sets = list(
        sample_predictors(dataset.predictors, num_predictors, num_repetitions)
    )

    # Train and evaluate for each set of sampled predictors.
    for r, predictors in enumerate(predictor_sets, 1):
        logger.info(
            "Running repetition %d/%d for %s for %d predictors."
            % (r, len(predictor_sets), model.name, num_predictors)
        )
        data = dataset.filter_predictors(predictors, keep_instances=True)
        evaluator = Evaluator(data)

        if model == "MAJ":
            result = evaluator.evaluate_maj_multi_per_label(soft=False)[0]
        elif model == "MAJ-S":
            result = evaluator.evaluate_maj_multi_per_label(soft=True)[0]
        else:
            with tf.Graph().as_default():
                train_data = data.to_train(shuffle=True)
                train_dataset = tf.data.Dataset.from_tensor_slices(
                    {
                        "instances": train_data.instances,
                        "predictors": train_data.predictors,
                        "labels": train_data.labels,
                        "values": train_data.values,
                    }
                )

                learner = learner_fn(model, dataset)
                learner.train(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    warm_start=warm_start,
                    max_m_steps=max_m_steps,
                    max_em_steps=max_em_steps,
                    max_marginal_steps=max_marginal_steps,
                    log_m_steps=log_m_steps,
                    use_progress_bar=use_progress_bar,
                )
                # TODO: Average results across all labels.
                result = evaluator.evaluate_per_label(
                    learner=learner, batch_size=batch_size
                )[0]
        results.append(result)

    # Collect results.
    accuracies = [r.accuracy for r in results]
    acc_result = {
        "time": time_stamp,
        "model": model.name,
        "num_predictors": num_predictors,
        "metric": "accuracy",
        "value_mean": np.mean(accuracies),
        "value_std": np.std(accuracies),
    }

    aucs = [r.auc for r in results]
    auc_result = {
        "time": time_stamp,
        "model": model.name,
        "num_predictors": num_predictors,
        "metric": "auc",
        "value_mean": np.mean(aucs),
        "value_std": np.std(aucs),
    }

    return acc_result, auc_result
