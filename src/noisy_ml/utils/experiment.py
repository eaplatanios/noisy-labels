"""Utils for running experiments."""

import itertools
import logging
import os

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
from ..models.models import MultiClassMMCE_M

__all__ = [
    "reset_seed",
    "sample_predictors",
    "learner_fn",
    "get_dataset_setup",
    "get_models",
    "gen_exp_configs",
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
    for _ in range(num_sets):
        yield random.sample(predictors, num_to_sample)


def learner_fn(
    dataset,
    model,
    learner_cls=EMLearner,
    config_cls=MultiLabelMultiClassEMConfig,
    optimizer="amsgrad",
    optimizer_kwargs=(("learning_rate", 1e-3),),
    lambda_entropy=0.,
    use_soft_y_hat=False,
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
            lambda_entropy=lambda_entropy,
            use_soft_y_hat=use_soft_y_hat,
        ),
        predictions_output_fn=np.exp,
    )


def get_dataset_setup(dataset, data_dir, results_dir, enforce_redundancy_limit=False):
    """Returns experimental setup parameters for the given dataset."""
    if dataset == "bluebirds":
        results_path = os.path.join(results_dir, "bluebirds.csv")
        dataset = BlueBirdsLoader.load(data_dir, load_features=True)
        if not enforce_redundancy_limit:
            num_predictors = [1, 2, 5, 10, 20, 39]
            num_repetitions = [10] * len(num_predictors)
            max_redundancy = None
        else:
            num_predictors = None
            max_redundancy = [1, 2, 5, 10, 20, 39]
            num_repetitions = [10] * len(max_redundancy)
    elif dataset == "rte":
        results_path = os.path.join(results_dir, "rte.csv")
        dataset = RTELoader.load(data_dir, load_features=True)
        if not enforce_redundancy_limit:
            num_predictors = [1, 10, 20, 50, 100, 164]
            num_repetitions = [50, 50, 50, 50, 20, 10]
            max_redundancy = None
        else:
            num_predictors = None
            max_redundancy = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            num_repetitions = [10] * len(max_redundancy)
    elif dataset == "wordsim":
        results_path = os.path.join(results_dir, "wordsim.csv")
        dataset = WordSimLoader.load(data_dir, load_features=True)
        if not enforce_redundancy_limit:
            num_predictors = [1, 2, 5, 10]
            num_repetitions = [50, 50, 50, 10]
            max_redundancy = None
        else:
            num_predictors = None
            max_redundancy = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            num_repetitions = [10] * len(max_redundancy)
    elif dataset == "age":
        results_path = os.path.join(results_dir, "age.csv")
        dataset = AgeLoader.load(data_dir, load_features=True)
        if not enforce_redundancy_limit:
            num_predictors = [1, 2, 5, 10, 20, 50, 100, 165]
            num_repetitions = [20, 20, 20, 20, 10, 10, 3, 3]
            max_redundancy = None
        else:
            num_predictors = None
            max_redundancy = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            num_repetitions = [10, 10, 10, 5, 5, 5, 3, 3, 3, 3]
    elif dataset == "medical-causes":
        results_path = os.path.join(results_dir, "medical-causes.csv")
        relations = ("CAUSES",)
        dataset = RelationExtractionLoader.load(
            data_dir, relations, load_features=True
        )
        if not enforce_redundancy_limit:
            num_predictors = [1, 10, 20, 50, 100, 200, 400, 467]
            num_repetitions = [20, 20, 20, 20, 10, 5, 3, 3]
            max_redundancy = None
        else:
            num_predictors = None
            max_redundancy = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            num_repetitions = [10, 10, 10, 5, 5, 5, 3, 3, 3, 3]
    elif dataset == "medical-treats":
        results_path = os.path.join(results_dir, "medical-treats.csv")
        relations = ("TREATS",)
        dataset = RelationExtractionLoader.load(
            data_dir, relations, load_features=True
        )
        if not enforce_redundancy_limit:
            num_predictors = [1, 10, 20, 50, 100, 200, 400, 467]
            num_repetitions = [20, 20, 20, 20, 10, 5, 3, 3]
            max_redundancy = None
        else:
            num_predictors = None
            max_redundancy = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            num_repetitions = [10, 10, 10, 5, 5, 5, 3, 3, 3, 3]
    elif dataset == "medical-causes-treats":
        results_path = os.path.join(results_dir, "medical-causes-treats.csv")
        relations = ("CAUSES", "TREATS")
        dataset = RelationExtractionLoader.load(
            data_dir, relations, load_features=True
        )
        if not enforce_redundancy_limit:
            num_predictors = [1, 10, 20, 50, 100, 200, 400, 467]
            num_repetitions = [20, 20, 20, 20, 10, 5, 3, 3]
            max_redundancy = None
        else:
            num_predictors = None
            max_redundancy = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            num_repetitions = [10, 10, 10, 5, 5, 5, 3, 3, 3, 3]
    else:
        raise NotImplementedError

    return dataset, num_predictors, num_repetitions, max_redundancy, results_path


def get_models(
    dataset,
    instances_emb_size=(4, 0),
    instances_hidden=([], [16, 16]),
    predictors_emb_size=(4, 16),
    predictors_hidden=([],),
    q_latent_size=(1,),
    gamma=(0.50, 0.75, 1.00),
):
    """Generates a dict of models for the specified parameters."""
    # Generate MMCE configurations.
    mmce_config_dicts = [{"gamma": 0.25}]

    # Generate LNL configurations.
    lnl_config_values = list(
        itertools.product(
            instances_emb_size,
            instances_hidden,
            predictors_emb_size,
            predictors_hidden,
            q_latent_size,
            gamma,
        )
    )
    lnl_config_names = len(lnl_config_values) * [
        [
            "instances_emb_size",
            "instances_hidden",
            "predictors_emb_size",
            "predictors_hidden",
            "q_latent_size",
            "gamma",
        ]
    ]
    lnl_config_dicts = map(
        lambda x: dict(zip(*x)), zip(lnl_config_names, lnl_config_values)
    )

    models = {"MAJ": "MAJ"}
    # Add MMCE models.
    for config in mmce_config_dicts:
        name = "MMCE-M"
        name += " (γ=%.2f)" % config["gamma"]
        models[name] = MultiClassMMCE_M(dataset, **config)

    # Add LNL models.
    for config in lnl_config_dicts:
        # If we use embeddings, no hidden layers.
        if (
            (config["instances_emb_size"] == 0 and not config["instances_hidden"])
            or (config["predictors_emb_size"] == 0 and not config["predictors_hidden"])
        ):
            continue
        name = "LNL"
        name += "-IE-%d" % config["instances_emb_size"]
        name += "-IH-%s" % config["instances_hidden"]
        name += "-PE-%d" % config["predictors_emb_size"]
        name += "-PH-%s" % config["predictors_hidden"]
        name += " (gam=%.2f)" % config["gamma"]
        models[name] = MultiClassLNL(dataset, **config)

    return models


def gen_exp_configs(models, num_predictors, num_repetitions, max_redundancy, results=None):
    if num_predictors is not None:
        inputs = [
            (
                ("model", model),
                ("model_name", name),
                ("num_predictors", num_p),
                ("num_repetitions", num_r),
            )
            for (name, model), (num_p, num_r) in itertools.product(
                models.items(), zip(num_predictors, num_repetitions)
            )
        ]
    else:
        assert max_redundancy is not None
        inputs = [
            (
                ("model", model),
                ("model_name", name),
                ("max_redundancy", max_r),
                ("num_repetitions", num_r),
            )
            for (name, model), (max_r, num_r) in itertools.product(
                models.items(), zip(max_redundancy, num_repetitions)
            )
        ]
    print("Total configs: %d" % len(inputs))

    # Filter out configurations for which we have results.
    excludes = set()
    if results is not None:
        if num_predictors is not None:
            res_exclude = results[["model", "num_predictors"]].values.tolist()
        else:
            res_exclude = results[["model", "max_redundancy"]].values.tolist()
        excludes.update(map(tuple, res_exclude))
    inputs = [i for i in inputs if (i[1][1], i[2][1]) not in excludes]
    print("Total configs after filtering: %d" % len(inputs))

    # Generate unique seed for each config and form input dicts.
    seeds = [random.randint(0, 2 ** 20) for _ in range(len(inputs))]
    configs = [dict(x + (("seed", s),)) for x, s in zip(inputs, seeds)]

    return configs


def train_eval_predictors(
    dataset,
    model,
    model_name,
    num_repetitions,
    num_predictors=None,
    max_redundancy=None,
    batch_size=128,
    max_em_iters=10,
    max_m_steps=1000,
    max_marginal_steps=1000,
    optimizer="amsgrad",
    optimizer_kwargs=(("learning_rate", 1e-3),),
    log_m_steps=None,
    warm_start=True,
    lambda_entropy=0.,
    use_soft_maj=False,
    use_soft_y_hat=False,
    use_progress_bar=False,
    time_stamp=None,
    seed=None,
):
    """Runs train-eval loop for the given predictors."""
    reset_seed(seed)

    results = []

    if num_predictors is not None:
        predictor_sets = list(
            sample_predictors(
                dataset.predictors, num_predictors, num_repetitions
            )
        )
    elif max_redundancy is None:
        raise ValueError("num_predictors or max_redundancy must be given.")

    # Train and evaluate for each set of sampled predictors.
    for r in range(num_repetitions):
        if num_predictors is not None:
            predictors = predictor_sets[r]
            logger.info(
                "Running repetition %d/%d for %s for %d predictors."
                % (r + 1, num_repetitions, model_name, num_predictors)
            )
            data = dataset.filter_predictors(predictors, keep_instances=True)
        else:
            logger.info(
                "Running repetition %d/%d for %s for %d maximum redundancy."
                % (r + 1, num_repetitions, model_name, max_redundancy)
            )
            data = dataset.enforce_redundancy_limit(max_redundancy)

        evaluator = Evaluator(data)

        if model == "MAJ":
            result = evaluator.evaluate_maj_multi_per_label(soft=use_soft_maj)[0]
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
                learner = learner_fn(
                    dataset=dataset,
                    model=model,
                    optimizer=optimizer,
                    optimizer_kwargs=optimizer_kwargs,
                    lambda_entropy=lambda_entropy,
                    use_soft_y_hat=use_soft_y_hat,
                )
                learner.train(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    warm_start=warm_start,
                    max_em_iters=max_em_iters,
                    max_m_steps=max_m_steps,
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
        "model": model_name,
        "metric": "accuracy",
        "value_mean": np.mean(accuracies),
        "value_std": np.std(accuracies),
    }

    aucs = [r.auc for r in results]
    auc_result = {
        "time": time_stamp,
        "model": model_name,
        "metric": "auc",
        "value_mean": np.mean(aucs),
        "value_std": np.std(aucs),
    }

    if num_predictors is not None:
        acc_result["num_predictors"] = num_predictors
        auc_result["num_predictors"] = num_predictors
    else:
        acc_result["max_redundancy"] = max_redundancy
        auc_result["max_redundancy"] = max_redundancy

    return acc_result, auc_result
