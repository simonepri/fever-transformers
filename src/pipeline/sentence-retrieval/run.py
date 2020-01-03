#!/usr/bin/env python3

import argparse
import bisect
import csv
import json
import os

from collections import defaultdict
from functools import reduce

from tqdm import tqdm


def get_best_evidences(scores_file, max_sent_per_claim):
    claim_evidences = defaultdict(lambda: [])
    with open(scores_file, "r") as f:
        nlines = reduce(lambda a, b: a + b, map(lambda x: 1, f.readlines()), 0)
        f.seek(0)
        lines = csv.reader(f, delimiter="\t")
        for line in tqdm(lines, desc="Score", total=nlines):
            claim_id, claim, page, sent_id, sent, score = line
            claim_id, sent_id, score = int(claim_id), int(sent_id), float(score)
            evid = (page, sent_id, sent)
            bisect.insort(claim_evidences[claim_id], (-score, evid))
            if len(claim_evidences[claim_id]) > max_sent_per_claim:
                claim_evidences[claim_id].pop()
    for claim_id in claim_evidences:
        for i, (score, evid) in enumerate(claim_evidences[claim_id]):
            claim_evidences[claim_id][i] = (-score, evid)
    return claim_evidences


def main(scores_file, in_file, out_file, max_sent_per_claim=None):
    path = os.getcwd()
    scores_file = os.path.join(path, scores_file)
    in_file = os.path.join(path, in_file)
    out_file = os.path.join(path, out_file)

    best_evidences = get_best_evidences(scores_file, max_sent_per_claim)

    with open(out_file, "w+") as fout:
        with open(in_file, "r") as fin:
            nlines = reduce(lambda a, b: a + b, map(lambda x: 1, fin.readlines()), 0)
            fin.seek(0)
            lines = map(json.loads, fin.readlines())
            for line in tqdm(lines, desc="Claim", total=nlines):
                claim_id = line["id"]
                line["best_evidences"] = best_evidences[claim_id]
                fout.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-file", type=str)
    parser.add_argument("--in-file", type=str, help="input dataset")
    parser.add_argument("--out-file", type=str,
                        help="path to save output dataset")
    parser.add_argument("--max-sent-per-claim", type=int,
                        help="number of top sentences to return for each claim")
    args = parser.parse_args()
    main(args.scores_file, args.in_file, args.out_file, max_sent_per_claim=args.max_sent_per_claim)
