import click
import logging
import os
import random

import pandas as pd

from concurrent import futures
from functools import partial
from itertools import product

from noisy_ml.utils.experiment import *

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--dataset-name",
    type=str,
    default="wordsim",
    help="Path to the data directory.",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True),
    default="data/",
    help="Path to the data directory.",
)
@click.option(
    "--results-dir",
    type=click.Path(exists=True),
    default="results/",
    help="Path to the results directory.",
)
@click.option(
    "--num-proc",
    type=int,
    default=4,
    help="Number of processes to run in parallel.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Seed used for all random number generators.",
)
def main(dataset_name, data_dir, results_dir, num_proc, seed):
    reset_seed(seed)

    # Get dataset setup.
    dataset, num_predictors, num_repetitions, results_path = get_dataset_setup(
        dataset=dataset_name,
        data_dir=data_dir,
        results_dir=results_dir,
    )

    # Get models.
    models = get_models(
        dataset,
        instances_emb_size=(4, None),
        instances_hidden=([], [16, 16]),
        predictors_emb_size=(4, 16),
        predictors_hidden=([],),
        q_latent_size=(1,),
        gamma=(0.50, 0.75, 1.00),
    )

    results = pd.DataFrame(
        columns=["model", "num_predictors", "metric", "value_mean", "value_std"]
    )
    time_stamp = pd.Timestamp.now()

    with futures.ProcessPoolExecutor(num_proc) as executor:
        func = partial(
            train_eval_predictors, dataset=dataset, time_stamp=time_stamp
        )
        seeds = [random.randint()]
        inputs = [
            (model, name, num_p, num_r, seed)
            for (name, model), (num_p, num_r, seed) in product(
                models.items(), zip(num_predictors, num_repetitions, seeds)
            )
        ]
        model_results = executor.map(func, inputs)
        for n, res in enumerate(model_results, start=1):
            logger.info(
                "Finished experiment for %d/%d predictors."
                % (n, len(num_predictors))
            )
            for r in res:
                results = results.append(r, ignore_index=True)
                results.to_csv(results_path)
            logger.info("Results so far:\n%s" % str(results))

    logger.info("Results:\n%s" % str(results))

    results.to_csv(results_path)

    results = pd.read_csv(results_path)


if __name__ == "__main__":
    main()
