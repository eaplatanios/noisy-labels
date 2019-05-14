"""
Data loader and preprocessor for Medical Relation Extraction dataset.
Source: https://github.com/CrowdTruth/Medical-Relation-Extraction
"""

import glob
import logging
import os

import numpy as np
import pandas as pd

from collections import defaultdict
from bert_serving.client import BertClient

from .datasets import Dataset

logger = logging.getLogger(__name__)


class RelationExtractionLoader(object):
    """Relation extraction task.

    Data loader and preprocessor for Medical Relation Extraction dataset.
    Source: https://github.com/CrowdTruth/Medical-Relation-Extraction

    The task is given a sentence with 2 medical terms, assign potentially
    multiple binary labels that indicate relations between the two terms.

    Notes:
        - We ignore directionality of the relations.
        - There are multiple relations that crowdworkers could pick from.
           However, we have ground truth only for a subset of relations:
           ["CAUSES", "TREATS"].
    """

    RELATIONS = (
        "ASSOCIATED_WITH",
        "CAUSES",
        "CONTRAINDICATES",
        "DIAGNOSE_BY_TEST_OR_DRUG",
        "IS_A",
        "LOCATION",
        "MANIFESTATION",
        "NONE",
        "PART_OF",
        "PREVENTS",
        "SIDE_EFFECT",
        "SYMPTOM",
        "TREATS",
        "OTHER",
    )

    @staticmethod
    def load_ground_truth(data_dir):
        names = {"cause": "CAUSES", "treat": "TREATS"}

        ground_truth = {}
        for relation in ["cause", "treat"]:
            gt_path = os.path.join(
                data_dir, "ground_truth", "ground_truth_%s.xlsx" % relation
            )
            gt_df = pd.read_excel(gt_path)

            # Select only the instances for which we have expert labels.
            index = (gt_df["expert"] * gt_df["baseline"]).dropna().index
            sentence_ids = gt_df.ix[index]["SID"].values
            baseline = gt_df.ix[index]["baseline"].values
            expert = gt_df.ix[index]["expert"].values
            gt_labels = ((baseline * expert) > 0).astype(np.int32)
            ground_truth[names[relation]] = (sentence_ids, gt_labels)

        return ground_truth

    @staticmethod
    def load_crowdsourced(data_dir):
        crowdsourced_dir = os.path.join(data_dir, "raw", "RelEx")
        work_feature_names = ["_channel", "_trust", "_country"]
        sent_feature_names = ["sentence", "term1", "term2"]

        sentence_ids, sentence_features = [], []
        worker_ids, worker_features = [], []
        relations = []

        for batch_path in glob.glob(os.path.join(crowdsourced_dir, "*.csv")):
            batch_df = pd.read_csv(batch_path)
            # Sentence ids and features.
            batch_sids = batch_df["sent_id"].values.tolist()
            batch_sfeats = batch_df[sent_feature_names].values.tolist()
            sentence_ids.extend(batch_sids)
            sentence_features.extend(batch_sfeats)

            # Worker ids and features.
            batch_wids = batch_df["_worker_id"].values.tolist()
            batch_wfeats = batch_df[work_feature_names].values.tolist()
            worker_ids.extend(batch_wids)
            worker_features.extend(batch_wfeats)

            # Crowdsourced relations.
            batch_relations = list(map(
                lambda x: [s[1:-1] for s in x.split()],
                batch_df["relations"].values
            ))
            relations.extend(batch_relations)

        # Clean up sentence ids (make all of them ints).
        sentence_ids = [
            int(sid.split("-")[0]) if isinstance(sid, str) else sid
            for sid in sentence_ids
        ]

        # Determine unique relations.
        unique_relations = set([r for r_list in relations for r in r_list])
        unique_relations = list(unique_relations)

        # Index relations.
        relation_ids = [
            [unique_relations.index(r) for r in r_list]
            for r_list in relations
        ]

        crowdsourced = {
            "sentences": (sentence_ids, sentence_features),
            "workers": (worker_ids, worker_features),
            "relations": (relations, relation_ids, unique_relations),
        }

        return crowdsourced

    @staticmethod
    def load_bert_features(data_dir, descriptions):
        features_dir = os.path.join(data_dir, "bert")
        features_path = os.path.join(features_dir, "sentence_features.txt")

        # Compute features, if do not exist.
        if not os.path.exists(features_path):
            os.makedirs(features_dir, exist_ok=True)
            bc = BertClient(ip="localhost")
            with open(features_path, "w") as f:
                for sentence, term1, term2 in descriptions:
                    terms12 = "%s and %s" % (term1, term2)
                    line_features = bc.encode(["%s ||| %s" % (sentence, terms12)])[0]
                    line_features_str = " ".join(map(str, line_features.tolist()))
                    f.write(line_features_str + "\n")

        # Load features.
        features = []
        with open(features_path) as fp:
            for line in fp:
                line_features = list(map(float, line.split()))
                features.append(np.asarray(line_features, dtype=np.float32))

        return features

    @staticmethod
    def load(data_dir, load_relations=("CAUSES", "TREATS"), load_features=True):
        data_dir = os.path.join(data_dir, "crowdsourced", "medical_relations")

        """Loads data."""
        ground_truth = RelationExtractionLoader.load_ground_truth(data_dir)
        crowdsourced = RelationExtractionLoader.load_crowdsourced(data_dir)
        unique_relations = crowdsourced["relations"][-1]

        # Convert everything to the required format.
        instances = sorted(set(crowdsourced["sentences"][0]))
        predictors = sorted(set(crowdsourced["workers"][0]))
        labels = [unique_relations.index(r) for r in load_relations]
        num_classes = [2 for _ in labels]

        true_labels = {}
        for relation in load_relations:
            label_id = labels.index(unique_relations.index(relation))
            true_labels[label_id] = dict([
                (instances.index(sid), gt_label)
                for sid, gt_label in zip(*ground_truth[relation])
            ])

        # Extract annotations.
        predicted_labels = dict()
        crowdsourced_data = (
            crowdsourced["sentences"][:1] +
            crowdsourced["workers"][:1] +
            crowdsourced["relations"][1:2]
        )
        for lid, l in enumerate(labels):
            predicted_labels[lid] = defaultdict(list)
            for sid, wid, rlist in zip(*crowdsourced_data):
                iid = instances.index(sid)
                pid = predictors.index(wid)
                predicted_labels[lid][pid].append((iid, int(l in rlist)))
            for pid in range(len(predictors)):
                predicted_labels[lid][pid] = list(zip(*predicted_labels[lid][pid]))

        # Load features.
        if load_features:
            sentence_ids = crowdsourced["sentences"][0]
            sentence_descriptions = crowdsourced["sentences"][1]
            instance_descriptions = [
                sentence_descriptions[sentence_ids.index(i)]
                for i in instances
            ]
            instance_features = RelationExtractionLoader.load_bert_features(
                data_dir, instance_descriptions
            )
        else:
            instance_features = None

        return Dataset(
            instances, predictors, labels,
            true_labels, predicted_labels,
            num_classes=num_classes,
            instance_features=instance_features)
