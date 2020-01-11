#!/usr/bin/env python3

import argparse
import bisect
import csv
import json
import os

from collections import defaultdict
from functools import reduce

from tqdm import tqdm

def main(in_file, out_file):
    path = os.getcwd()
    in_file = os.path.join(path, in_file)
    out_file = os.path.join(path, out_file)

    with open(out_file, "w+") as fout:
        with open(in_file, "r") as fin:
            nlines = reduce(lambda a, b: a + b, map(lambda x: 1, fin.readlines()), 0)
            fin.seek(0)
            lines = map(json.loads, fin.readlines())
            for line in tqdm(lines, desc="Claim", total=nlines):
                prediction = {
                    "id": line["id"],
                    "predicted_label": line["predicted_label"],
                    "predicted_evidence": line["predicted_evidence"]
                }
                json.dump(prediction, fout)
                fout.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, help="input dataset")
    parser.add_argument("--out-file", type=str,
                        help="path to save output dataset")
    args = parser.parse_args()
    main(args.in_file, args.out_file)
