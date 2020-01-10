#!/usr/bin/env python3

import argparse
import json
import os

from collections import defaultdict

from fever.scorer import fever_score
from prettytable import PrettyTable

def main(evidence_file, golden_file):
    path = os.getcwd()
    evidence_file = os.path.join(path, evidence_file)
    golden_file = os.path.join(path, golden_file)

    actual = []
    with open(golden_file, "r") as f:
        for line in f:
            actual.append(json.loads(line))

    predictions = []
    with open(evidence_file, "r") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            line["predicted_label"] = actual[i]["label"]
            line["predicted_evidence"] = list(map(lambda e: e[1][:2], line["predicted_sentences"]))
            predictions.append(line)

    assert len(predictions) == len(actual), "The two file provided does not have the same number of lines"

    score,_,precision,recall,f1 = fever_score(predictions, actual)

    tab = PrettyTable()
    tab.field_names = ["OFEVER Score", "Evidence Precision", "Evidence Recall", "Evidence F1"]
    tab.add_row((round(score,4),round(precision,4),round(recall,4),round(f1,4)))
    print(tab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-file", type=str, help="input dataset")
    parser.add_argument("--golden-file", type=str, help="original input dataset with gold sentences")
    args = parser.parse_args()
    main(args.evidence_file, args.golden_file)
