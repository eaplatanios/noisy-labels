import logging.config
import os
import yaml

from . import data
from . import evaluation
from . import training

__all__ = ["data", "evaluation", "training"]

__LOGGING_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "assets", "logging.yaml"
)
if os.path.exists(__LOGGING_CONFIG_PATH):
    with open(__LOGGING_CONFIG_PATH, "rt") as f:
        __CONFIG = yaml.safe_load(f.read())
    logging.config.dictConfig(__CONFIG)
else:
    logging.getLogger("").addHandler(logging.NullHandler())
