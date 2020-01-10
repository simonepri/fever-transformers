#!/usr/bin/env python3

import argparse
import bisect
import csv
import json
import os

from collections import defaultdict
from functools import reduce

from tqdm import tqdm


def get_classified_sentences(labels_file):
    claim_labels = defaultdict(lambda: [])
    label_map = ["REFUTES", "SUPPORTS", "NOT ENOUGH INFO"]
    with open(labels_file, "r") as f:
        nlines = reduce(lambda a, b: a + b, map(lambda x: 1, f.readlines()), 0)
        f.seek(0)
        lines = csv.reader(f, delimiter="\t")
        for line in tqdm(lines, desc="Label", total=nlines):
            claim_id, claim, page, sent_id, sent, label = line
            claim_id, sent_id, label = int(claim_id), int(sent_id), label_map[int(label)]
            evid = (page, sent_id, sent)
            claim_labels[claim_id].append((label, evid))
    return claim_labels


def predict_claim(classified_sentences):
    prediction = ("NOT ENOUGH INFO", [])
    for label, (page, sent_id, _) in classified_sentences:
        if label == "NOT ENOUGH INFO":
            continue
        elif label == "SUPPORTS":
            if prediction[0] != label:
                prediction = (label, [])
            prediction[1].append((page, sent_id))
        elif label == "REFUTES":
            if prediction[0] == "SUPPORTS":
                continue
            if prediction[0] != label:
                prediction = (label, [])
            prediction[1].append((page, sent_id))
        else:
            raise KeyError(label)
    return {"predicted_label": prediction[0], "predicted_evidence": prediction[1]}


def main(labels_file, in_file, out_file):
    path = os.getcwd()
    labels_file = os.path.join(path, labels_file)
    in_file = os.path.join(path, in_file)
    out_file = os.path.join(path, out_file)

    classified_sentences = get_classified_sentences(labels_file)

    with open(out_file, "w+") as fout:
        with open(in_file, "r") as fin:
            nlines = reduce(lambda a, b: a + b, map(lambda x: 1, fin.readlines()), 0)
            fin.seek(0)
            lines = map(json.loads, fin.readlines())
            for line in tqdm(lines, desc="Claim", total=nlines):
                claim_id = line["id"]
                line["classified_sentences"] = classified_sentences[claim_id]
                line.update(predict_claim(classified_sentences[claim_id]))
                fout.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-file", type=str)
    parser.add_argument("--in-file", type=str, help="input dataset")
    parser.add_argument("--out-file", type=str,
                        help="path to save output dataset")
    args = parser.parse_args()
    main(args.labels_file, args.in_file, args.out_file)
