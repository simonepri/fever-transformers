#!/usr/bin/env python3

import argparse
import bisect
import csv
import json
import os
from collections import defaultdict
from functools import reduce

from tqdm import tqdm


def get_best_evidence(scores_file, max_sentences_per_claim):
    weighted_claim_evidence = defaultdict(lambda: [])
    with open(scores_file, "r") as f:
        nlines = reduce(lambda a, b: a + b, map(lambda x: 1, f.readlines()), 0)
        f.seek(0)
        lines = csv.reader(f, delimiter="\t")
        for line in tqdm(lines, desc="Score", total=nlines):
            claim_id, claim, page, sent_id, sent, score = line
            claim_id, sent_id, score = int(claim_id), int(sent_id), float(score)
            evid = (page, sent_id, sent)
            bisect.insort(weighted_claim_evidence[claim_id], (-score, evid))
            if len(weighted_claim_evidence[claim_id]) > max_sentences_per_claim:
                weighted_claim_evidence[claim_id].pop()
    for claim_id in weighted_claim_evidence:
        for i, (score, evid) in enumerate(weighted_claim_evidence[claim_id]):
            weighted_claim_evidence[claim_id][i] = (-score, evid)
    return weighted_claim_evidence


def main(scores_file, in_file, out_file, max_sentences_per_claim=None):
    path = os.getcwd()
    scores_file = os.path.join(path, scores_file)
    in_file = os.path.join(path, in_file)
    out_file = os.path.join(path, out_file)

    best_evidence = get_best_evidence(scores_file, max_sentences_per_claim)

    with open(out_file, "w+") as fout:
        with open(in_file, "r") as fin:
            nlines = reduce(lambda a, b: a + b, map(lambda x: 1, fin.readlines()), 0)
            fin.seek(0)
            lines = map(json.loads, fin.readlines())
            for line in tqdm(lines, desc="Claim", total=nlines):
                claim_id = line["id"]
                line["predicted_sentences"] = best_evidence[claim_id]
                fout.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-file", type=str)
    parser.add_argument("--in-file", type=str, help="input dataset")
    parser.add_argument("--out-file", type=str,
                        help="path to save output dataset")
    parser.add_argument("--max-sentences-per-claim", type=int,
                        help="number of top sentences to return for each claim")
    args = parser.parse_args()
    main(args.scores_file, args.in_file, args.out_file, max_sentences_per_claim=args.max_sentences_per_claim)
