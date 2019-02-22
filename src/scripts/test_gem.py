"""Runs training and evaluation of different methods in parallel."""

import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

import tensorflow as tf

from functools import partial
from itertools import product

from noisy_ml.data.crowdsourced import *
from noisy_ml.data.rte import *
from noisy_ml.data.wordsim import *
from noisy_ml.evaluation.metrics import *
from noisy_ml.training.learners import *
from noisy_ml.training.models import *
from noisy_ml.training.amsgrad import *

logger = logging.getLogger(__name__)


def reset_seed():
    seed = 1234567890
    np.random.seed(seed)
    tf.set_random_seed(seed)


def sample_predictors(predictors, num_to_sample, num_sets=5):
    if len(predictors) <= num_to_sample:
        yield predictors
    else:
        for _ in range(num_sets):
            yield random.sample(predictors, num_to_sample)


def learner_fn(model, dataset):
    return GeneralizedStochasticEMLearner(
        config=MultiLabelMultiClassGEMConfig(
            num_instances=len(dataset.instances),
            num_predictors=len(dataset.predictors),
            num_labels=len(dataset.labels),
            num_classes=dataset.num_classes,
            model=model,
            optimizer=AMSGrad(1e-3),
            lambda_entropy=0.0,
        )
    )


def train_eval_predictors(args, dataset, time_stamp):
    reset_seed()

    model, model_name, num_p, num_repetitions = args
    num_p_results = []
    sampled_predictors = list(
        sample_predictors(dataset.predictors, num_p, num_sets=num_repetitions)
    )
    for r, predictors in enumerate(sampled_predictors):
        logger.info(
            "Running repetition %d/%d for %s for %d predictors."
            % (r + 1, len(sampled_predictors), model_name, num_p)
        )
        data = dataset.filter_predictors(predictors, keep_instances=True)
        evaluator = Evaluator(data)

        if model == "MAJ":
            result = evaluator.evaluate_maj_per_label(soft=False)[0]
        elif model == "MAJ-S":
            result = evaluator.evaluate_maj_per_label(soft=True)[0]
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
                    batch_size=128,
                    max_em_steps=1000,
                    use_progress_bar=True,
                )
                # TODO: Average results across all labels.
                result = evaluator.evaluate_per_label(learner=learner, batch_size=128)[
                    0
                ]
        num_p_results.append(result)

    # Collect results.
    accuracies = [r.accuracy for r in num_p_results]
    acc_result = {
        "time": time_stamp,
        "model": model_name,
        "num_predictors": num_p,
        "metric": "accuracy",
        "value_mean": np.mean(accuracies),
        "value_std": np.std(accuracies),
    }
    aucs = [r.auc for r in num_p_results]
    auc_result = {
        "time": time_stamp,
        "model": model_name,
        "num_predictors": num_p,
        "metric": "auc",
        "value_mean": np.mean(aucs),
        "value_std": np.std(aucs),
    }

    return acc_result, auc_result


def run_experiment(num_proc=1):
    reset_seed()

    dataset = "wordsim"
    working_dir = os.getcwd()
    data_dir = os.path.join(working_dir, os.pardir, "data")
    results_dir = os.path.join(working_dir, os.pardir, "results")

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
        num_predictors = [10]
        num_repetitions = [1]
        results_path = os.path.join(results_dir, "wordsim.csv")
    else:
        raise NotImplementedError

    models = {
        "VNM[4]": VariationalNoisyModel(
            dataset=dataset,
            num_labels=len(dataset.labels),
            num_classes=dataset.num_classes,
            num_predictors=len(dataset.predictors),
            num_predictor_samples=1,
            instances_emb_size=4,
            predictors_emb_size=4,
            instances_hidden=[4],
            predictors_hidden=[4],
            q_latent_size=None,
            gamma=1.0,
        ),
    }

    results = pd.DataFrame(
        columns=["model", "num_predictors", "metric", "value_mean", "value_std"]
    )
    time_stamp = pd.Timestamp.now()

    func = partial(train_eval_predictors, dataset=dataset, time_stamp=time_stamp)
    inputs = [
        (model, name, num_p, num_r)
        for (name, model), (num_p, num_r) in product(
            models.items(), zip(num_predictors, num_repetitions)
        )
    ]
    for n, inp in enumerate(inputs, start=1):
        res = func(inp)
        logger.info(
            "Finished experiment for %d/%d predictors." % (n, len(num_predictors))
        )
        for r in res:
            results = results.append(r, ignore_index=True)
            results.to_csv(results_path)
        logger.info("Results so far:\n%s" % str(results))

    logger.info("Results:\n%s" % str(results))

    results.to_csv(results_path)

    results = pd.read_csv(results_path)

    # Accuracy Plot.
    fig, ax = plt.subplots()
    results_auc = results[results.metric == "accuracy"]
    for label, auc in results_auc.groupby("model"):
        ax.plot(auc.num_predictors.astype(np.int32), auc.value_mean, label=label)
        ax.fill_between(
            auc.num_predictors.astype(np.int32),
            auc.value_mean - auc.value_std,
            auc.value_mean + auc.value_std,
            alpha=0.35,
        )
    plt.legend()
    plt.show()

    # AUC Plot.
    fig, ax = plt.subplots()
    results_auc = results[results.metric == "auc"]
    for label, auc in results_auc.groupby("model"):
        ax.plot(auc.num_predictors.astype(np.int32), auc.value_mean, label=label)
        ax.fill_between(
            auc.num_predictors.astype(np.int32),
            auc.value_mean - auc.value_std,
            auc.value_mean + auc.value_std,
            alpha=0.35,
        )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_experiment(num_proc=16)
