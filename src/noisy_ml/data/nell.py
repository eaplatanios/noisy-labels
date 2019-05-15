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

import logging
import numpy as np
import os

from .datasets import Dataset

__author__ = "eaplatanios"

__all__ = ["LegacyLoader", "NELLLoader"]

logger = logging.getLogger(__name__)


class LegacyLoader(object):
    @staticmethod
    def load(data_dir, dataset_type="nell", labels=None, small_version=False):
        """Loads a legacy dataset.

    Args:
      data_dir: Data directory (e.g., for downloading
        and extracting data files).
      dataset_type: Can be `"nell"` or `"brain"`. Defaults
        to all labels for the provided dataset type.
      labels: Labels for which to load data.
      small_version: Boolean flag indicating whether to use
        the small version of the NELL dataset. This is 10%
        of the complete dataset.

    Returns: Loaded legacy dataset for the provided labels.
    """
        if dataset_type is "nell":
            if labels is None:
                labels = [
                    "animal",
                    "beverage",
                    "bird",
                    "bodypart",
                    "city",
                    "disease",
                    "drug",
                    "fish",
                    "food",
                    "fruit",
                    "muscle",
                    "person",
                    "protein",
                    "river",
                    "vegetable",
                ]
        elif dataset_type is "brain":
            if labels is None:
                labels = [
                    "region_1",
                    "region_2",
                    "region_3",
                    "region_4",
                    "region_5",
                    "region_6",
                    "region_7",
                    "region_8",
                    "region_9",
                    "region_10",
                    "region_11",
                ]
        else:
            raise ValueError('Legacy loader type can be "nell" or "brain".')

        if len(labels) == 1:
            if dataset_type is "nell":
                data_dir = os.path.join(data_dir, "nell", "unconstrained")
            elif dataset_type is "brain":
                data_dir = os.path.join(data_dir, "brain")
            else:
                raise ValueError('Legacy loader type can be "nell" or "brain".')

            label = labels[0]

            # Load instance names.
            if dataset_type is "nell" and not small_version:
                filename = "{}.txt".format(label)
                filename = os.path.join(data_dir, "names", filename)
                with open(filename, "r") as f:
                    instances = [line for line in f]

            # Load predicted labels.
            if small_version:
                data_dir = os.path.join(data_dir, "small")
            else:
                data_dir = os.path.join(data_dir, "full")
            filename = "{}.csv".format(label)
            filename = os.path.join(data_dir, filename)

            if small_version:
                instances = list()
            predictors = list()
            labels = [label]
            true_labels = {0: dict()}
            predicted_labels = {0: dict()}
            is_header = True

            with open(filename, "r") as f:
                for i, line in enumerate(f):
                    if is_header:
                        predictors = [s.strip() for s in line.split(",")][1:]
                        predicted_labels[0] = {
                            p: ([], []) for p in range(len(predictors))
                        }
                        is_header = False
                    else:
                        line_parts = [s.strip() for s in line.split(",")]
                        if small_version:
                            instances.append("label_%d" % (i - 1))
                        true_labels[0][i - 1] = int(line_parts[0])
                        for p in range(len(predictors)):
                            predicted_labels[0][p][0].append(i - 1)
                            predicted_labels[0][p][1].append(
                                float(line_parts[p + 1])
                            )

            return Dataset(
                instances, predictors, labels, true_labels, predicted_labels
            )
        else:
            return Dataset.join(
                [
                    LegacyLoader.load(
                        data_dir, dataset_type, [l], small_version
                    )
                    for l in labels
                ]
            )


class NELLLoader(object):
    @staticmethod
    def load(data_dir, labels, load_features=True, ground_truth_threshold=0.1):
        if load_features:
            f_filename = "np_features.tsv"
            f_filename = os.path.join(data_dir, f_filename)
            sparse_features = list()
            num_features = 0
            with open(f_filename, "r") as f:
                for f_line in f:
                    f_line_parts = [s.strip() for s in f_line.split("\t")]
                    feature_indices = []
                    feature_values = []
                    for p in f_line_parts[1].split(","):
                        pair = p.split(":")
                        f_index = int(pair[0])
                        feature_indices.append(f_index - 1)
                        feature_values.append(float(pair[1]))
                        num_features = max(num_features, f_index)
                    sparse_features.append(
                        (f_line_parts[0], feature_indices, feature_values)
                    )
            features = dict()
            for instance, feature_indices, feature_values in sparse_features:
                f = np.zeros([num_features], np.float32)
                f[feature_indices] = feature_values
                features[instance] = f
        else:
            features = None

        def load_for_label(label):
            filename = "{}.extracted_instances.all_predictions.txt".format(
                label
            )
            filename = os.path.join(data_dir, filename)

            instances = list()
            predictors = list()
            labels = [label]
            true_labels = {0: dict()}
            predicted_labels = {0: dict()}
            instance_features = list() if features is not None else None
            is_header = True
            i = 0

            with open(filename, "r") as f:
                for line in f:
                    if is_header:
                        predictors = [s.strip() for s in line.split("\t")][2:]
                        predicted_labels[0] = {
                            p: ([], []) for p in range(len(predictors))
                        }
                        is_header = False
                    else:
                        line_parts = [s.strip() for s in line.split("\t")]
                        instance = line_parts[0]
                        if features is not None and instance not in features:
                            continue
                        i += 1
                        instances.append(instance)
                        true_labels[0][i - 1] = int(
                            float(line_parts[1]) >= ground_truth_threshold
                        )
                        for p in range(len(predictors)):
                            if line_parts[p + 2] != "-":
                                predicted_labels[0][p][0].append(i - 1)
                                predicted_labels[0][p][1].append(
                                    float(line_parts[p + 2])
                                )
                        if features is not None:
                            instance_features.append(features[instance])

            # Single label with 2 classes.
            num_classes = [2]

            return Dataset(
                instances,
                predictors,
                labels,
                true_labels,
                predicted_labels,
                num_classes=num_classes,
                instance_features=instance_features,
            )

        return Dataset.join([load_for_label(l) for l in labels])

    @staticmethod
    def load_with_ground_truth(
        data_dir, labels, load_features=True, ground_truth=None
    ):
        if ground_truth is None:
            filename = "np_labels.tsv"
            filename = os.path.join(data_dir, filename)
            ground_truth = dict()
            with open(filename, "r") as f:
                for line in f:
                    line_parts = [s.strip() for s in line.split("\t")]
                    ground_truth[line_parts[0]] = set(line_parts[1].split(","))

        if load_features:
            f_filename = "np_features.tsv"
            f_filename = os.path.join(data_dir, f_filename)
            sparse_features = list()
            num_features = 0
            with open(f_filename, "r") as f:
                for f_line in f:
                    f_line_parts = [s.strip() for s in f_line.split("\t")]
                    feature_indices = []
                    feature_values = []
                    for p in f_line_parts[1].split(","):
                        pair = p.split(":")
                        f_index = int(pair[0])
                        feature_indices.append(f_index - 1)
                        feature_values.append(float(pair[1]))
                        num_features = max(num_features, f_index)
                    sparse_features.append(
                        (f_line_parts[0], feature_indices, feature_values)
                    )
            features = dict()
            for instance, feature_indices, feature_values in sparse_features:
                f = np.zeros([num_features], np.float32)
                f[feature_indices] = feature_values
                features[instance] = f
        else:
            features = None

        def load_for_label(label):
            filename = "{}.extracted_instances.all_predictions.txt".format(
                label
            )
            filename = os.path.join(data_dir, filename)

            instances = list()
            predictors = list()
            labels = [label]
            true_labels = {0: dict()}
            predicted_labels = {0: dict()}
            instance_features = list() if features is not None else None
            is_header = True
            i = 0

            with open(filename, "r") as f:
                for line in f:
                    if is_header:
                        predictors = [s.strip() for s in line.split("\t")][2:]
                        predicted_labels[0] = {
                            p: ([], []) for p in range(len(predictors))
                        }
                        is_header = False
                    else:
                        line_parts = [s.strip() for s in line.split("\t")]
                        instance = line_parts[0]
                        if features is not None and instance not in features:
                            continue
                        i += 1
                        instances.append(instance)
                        true_labels[0][i - 1] = int(
                            label in ground_truth[line_parts[0]]
                        )
                        for p in range(len(predictors)):
                            if line_parts[p + 2] != "-":
                                predicted_labels[0][p][0].append(i - 1)
                                predicted_labels[0][p][1].append(
                                    float(line_parts[p + 2])
                                )
                        if features is not None:
                            instance_features.append(features[instance])

            return Dataset(
                instances,
                predictors,
                labels,
                true_labels,
                predicted_labels,
                instance_features=instance_features,
            )

        return Dataset.join([load_for_label(l) for l in labels])
