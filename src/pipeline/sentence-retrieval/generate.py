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


def get_all_sentences(docs, pages):
    for page in pages:
        for evidence in sample_evidences(docs, page, num_samples=None):
            yield evidence


def get_evidence_sentences(docs, evid_sets):
    evidences = set()
    for evid_set in evid_sets:
        for item in evid_set:
            Annotation_ID, Evidence_ID, Wikipedia_URL, sentence_ID = item
            sent = docs[Wikipedia_URL][sentence_ID].split("\t")[1]
            evidences.add((Wikipedia_URL, sentence_ID, sent))
    return evidences


def get_non_evidence_sentences(docs, evid_sets, pred_pages, max_non_evidence_per_page=None):
    positive_sentences = {}
    for evid_set in evid_sets:
        for item in evid_set:
            Annotation_ID, Evidence_ID, Wikipedia_URL, sentence_ID = item
            if not Wikipedia_URL in positive_sentences:
                positive_sentences[Wikipedia_URL] = set()
            positive_sentences[Wikipedia_URL].add(sentence_ID)

    # sample negative examples from pages where good evidences are, by
    # avoiding to select the good evidences themself
    for page, positives in positive_sentences.items():
        for evidence in sample_evidences(docs, page, positives, num_samples=max_non_evidence_per_page):
            yield evidence

    # sample negative examples from other predicted pages in which there
    # are not good evidences.
    for page in pred_pages:
        if page in positive_sentences:
            continue
        for evidence in sample_evidences(docs, page, num_samples=max_non_evidence_per_page):
            yield evidence


def sample_evidences(docs, page, to_ignore=set(), num_samples=1):
    evidences = []
    for sent in docs[page]:
        sent = sent.split("\t")
        if len(sent) < 2:
            continue
        sent_id, sent_text = sent[0], sent[1]
        if len(sent_text.strip()) == 0:
            continue
        if sent_id in to_ignore:
            continue
        evidences.append((page, sent_id, sent_text))
    return evidences if num_samples is None else random.sample(evidences, min(len(evidences), num_samples))


def fetch_documents(db, evid_sets, pred_pages):
    pages = set(pred_pages)
    for evid_set in evid_sets:
        for item in evid_set:
            _, _, page, _ = item
            if page is not None:
                pages.add(page)

    docs = defaultdict(lambda: [])
    for page, lines in db.get_all_doc_lines(pages):
        docs[page] = re.split("\n(?=\d+)", lines)
    return docs


def main(db_file, in_file, out_file, max_non_evidence_per_page=None, prediction=None):
    path = os.getcwd()
    outfile = open(os.path.join(path, out_file), "w+")

    db = FeverDocDB(db_file)

    with open(os.path.join(path, in_file), "r") as f:
        nlines = reduce(lambda a, b: a + b, map(lambda x: 1, f.readlines()), 0)
        f.seek(0)
        lines = map(json.loads, f.readlines())
        for line in tqdm(lines, total=nlines):
            # if not verifiable, we don't have evidence and just continue
            if not prediction and line["verifiable"] == "NOT VERIFIABLE":
                continue

            id = line["id"]
            claim = line["claim"]
            evid_sets = line.get("evidence", [])
            pred_pages = line["predicted_pages"]

            docs = fetch_documents(db, evid_sets, pred_pages)

            if prediction:
                # extract all the sentences for the documents predicted for this claim
                for page, sent_id, sentence in get_all_sentences(docs, pred_pages):
                    outfile.write("\t".join([str(id), claim, page, str(sent_id), sentence]) + "\n")
            else:
                # write positive and negative evidence examples to file
                for page, sent_id, sentence in get_evidence_sentences(docs, evid_sets):
                    outfile.write("\t".join([str(id), claim, page, str(sent_id), sentence, "1"]) + "\n")
                for page, sent_id, sentence in get_non_evidence_sentences(docs, evid_sets, pred_pages, max_non_evidence_per_page=max_non_evidence_per_page):
                    outfile.write("\t".join([str(id), claim, page, str(sent_id), sentence, "0"]) + "\n")
    outfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-file", type=str,
                        help="database file which contains wiki pages")
    parser.add_argument("--in-file", type=str, help="input dataset")
    parser.add_argument("--out-file", type=str,
                        help="path to save output dataset")
    parser.add_argument("--max-non-evidence-per-page", type=int,
                        help="number of negative evidance to in each of the page that are relevant for a claim")
    parser.add_argument("--prediction", action='store_true',
                        help="when set it generate all the sentences of the prediceted documents")
    args = parser.parse_args()
    main(args.db_file, args.in_file, args.out_file, max_non_evidence_per_page=args.max_non_evidence_per_page, prediction=args.prediction)
