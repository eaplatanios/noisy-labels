import click
import json
import logging
import os
import random

import pandas as pd

from concurrent import futures
from functools import partial
from itertools import product

from noisy_ml.utils.experiment import *

logger = logging.getLogger(__name__)


def _parse_int_list(list_str):
    str_list = list_str[1:-1].split(",")
    int_list = [int(s.strip()) for s in str_list if s]
    return int_list


def _unpack(kwargs, f):
    return f(**kwargs)


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
    "--instances-emb-size",
    type=int,
    multiple=True,
    default=[4, 0],
    help="Instance embedding sizes to try.",
)
@click.option(
    "--instances-hidden",
    type=str,
    multiple=True,
    default=["[]", "[16, 16]"],
    help="Instance hidden sizes to try.",
)
@click.option(
    "--predictors-emb-size",
    type=int,
    multiple=True,
    default=[4, 16],
    help="Predictor embedding sizes to try.",
)
@click.option(
    "--predictors-hidden",
    type=str,
    multiple=True,
    default=["[]"],
    help="Predictor hidden sizes to try.",
)
@click.option(
    "--q-latent-size",
    type=int,
    multiple=True,
    default=[1],
    help="Latent sizes of the qualities to try.",
)
@click.option(
    "--gamma",
    type=float,
    multiple=True,
    default=[0.5, 0.75, 1.0],
    help="Values of gamma to try.",
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
@click.pass_context
def main(
    ctx,
    dataset_name,
    data_dir,
    results_dir,
    instances_emb_size,
    instances_hidden,
    predictors_emb_size,
    predictors_hidden,
    q_latent_size,
    gamma,
    num_proc,
    seed,
):
    reset_seed(seed)

    # Process hiddens.
    instances_hidden = list(map(_parse_int_list, instances_hidden))
    predictors_hidden = list(map(_parse_int_list, predictors_hidden))

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save experiment config.
    config_path = os.path.join(results_dir, "config.json")
    with open(config_path, "w") as fp:
        json.dump(ctx.params, fp, indent=4, sort_keys=True)

    # Get dataset setup.
    dataset, num_predictors, num_repetitions, results_path = get_dataset_setup(
        dataset=dataset_name, data_dir=data_dir, results_dir=results_dir
    )

    # Get models.
    models = get_models(
        dataset,
        instances_emb_size=instances_emb_size,
        instances_hidden=instances_hidden,
        predictors_emb_size=predictors_emb_size,
        predictors_hidden=predictors_hidden,
        q_latent_size=q_latent_size,
        gamma=gamma,
    )

    # Setup a DataFrame for results.
    res_cols = ["model", "num_predictors", "metric", "value_mean", "value_std"]
    results = pd.DataFrame(columns=res_cols)
    time_stamp = pd.Timestamp.now()

    with futures.ProcessPoolExecutor(num_proc) as executor:
        # Generate experiment configurations.
        inputs = [
            (
                ("model", model),
                ("model_name", name),
                ("num_predictors", num_p),
                ("num_repetitions", num_r),
            )
            for (name, model), (num_p, num_r) in product(
                models.items(), zip(num_predictors, num_repetitions)
            )
        ]
        seeds = [random.randint(0, 2 ** 20) for _ in range(len(inputs))]
        input_dicts = [dict(x + (("seed", s),)) for x, s in zip(inputs, seeds)]

        # Run experiments for each configuration (in parallel).
        logger.info("Running %d experiments..." % len(input_dicts))
        func = partial(train_eval_predictors, dataset, time_stamp=time_stamp)
        model_results = executor.map(partial(_unpack, f=func), input_dicts)
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


if __name__ == "__main__":
    main()
