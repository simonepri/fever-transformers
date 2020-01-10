#!/usr/bin/env python3

import argparse
import re
import json
import os
import unicodedata
import random
from collections import defaultdict
from functools import reduce

import numpy as np
from tqdm import tqdm

from common.fever_doc_db import FeverDocDB


def get_all_sentences(docs, weighted_sentences):
    for (score, sentence) in weighted_sentences:
        yield sentence


def get_evidence_sentences(docs, evid_sets):
    evidence = set()
    for evid_set in evid_sets:
        for item in evid_set:
            Annotation_ID, Evidence_ID, Wikipedia_URL, sentence_ID = item
            if Wikipedia_URL is not None:
                sent = docs[Wikipedia_URL][sentence_ID].split("\t")[1]
                evidence.add((Wikipedia_URL, sentence_ID, sent))
    return evidence


def get_non_evidence_sentences(docs, evid_sets, weighted_sentences):
    positive_sentences = {}
    for evid_set in evid_sets:
        for item in evid_set:
            Annotation_ID, Evidence_ID, Wikipedia_URL, sentence_ID = item
            if Wikipedia_URL is not None:
                if not Wikipedia_URL in positive_sentences:
                    positive_sentences[Wikipedia_URL] = set()
                positive_sentences[Wikipedia_URL].add(sentence_ID)

    # sample negative examples from other sentences that are not useful evidence
    for (score, sentence) in weighted_sentences:
        page, sent_id, sent = sentence
        if page in positive_sentences and sent_id in positive_sentences[page]:
            continue
        yield sentence


def fetch_documents(db, evid_sets):
    pages = set()
    for evid_set in evid_sets:
        for item in evid_set:
            _, _, page, _ = item
            if page is not None:
                pages.add(page)

    docs = defaultdict(lambda: [])
    for page, lines in db.get_all_doc_lines(pages):
        docs[page] = re.split("\n(?=\d+)", lines)
    return docs


def main(db_file, in_file, out_file, prediction=None):
    path = os.getcwd()
    outfile = open(os.path.join(path, out_file), "w+")

    db = FeverDocDB(db_file)

    with open(os.path.join(path, in_file), "r") as f:
        nlines = reduce(lambda a, b: a + b, map(lambda x: 1, f.readlines()), 0)
        f.seek(0)
        lines = map(json.loads, f.readlines())
        for line in tqdm(lines, total=nlines):
            id = line["id"]
            claim = line["claim"]
            evid_sets = line.get("evidence", [])
            weighted_sentences = line["predicted_sentences"]

            docs = fetch_documents(db, evid_sets)

            if prediction:
                # extract all the sentences predicted for this claim
                for page, sent_id, sentence in get_all_sentences(docs, weighted_sentences):
                    outfile.write("\t".join([str(id), claim, page, str(sent_id), sentence]) + "\n")
            else:
                label = line["label"]
                # write positive and negative evidence to file
                for page, sent_id, sentence in get_evidence_sentences(docs, evid_sets):
                    outfile.write("\t".join([str(id), claim, page, str(sent_id), sentence, label[0]]) + "\n")
                for page, sent_id, sentence in get_non_evidence_sentences(docs, evid_sets, weighted_sentences):
                    outfile.write("\t".join([str(id), claim, page, str(sent_id), sentence, "NOT ENOUGH INFO"[0]]) + "\n")
    outfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-file", type=str,
                        help="database file which contains wiki pages")
    parser.add_argument("--in-file", type=str, help="input dataset")
    parser.add_argument("--out-file", type=str,
                        help="path to save output dataset")
    parser.add_argument("--prediction", action='store_true',
                        help="when set it generate all the sentences of the prediceted documents")
    args = parser.parse_args()
    main(args.db_file, args.in_file, args.out_file, prediction=args.prediction)
