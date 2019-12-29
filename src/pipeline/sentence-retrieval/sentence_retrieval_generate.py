import argparse
import re
import json
import os
import unicodedata
import random

from collections import defaultdict

import numpy as np

from tqdm import tqdm

from common.fever_doc_db import FeverDocDB
from common.file_reader import JSONLineReader

def get_positive_claim_evidences(docs, evid_sets):
    evidences = set()
    for evid_set in evid_sets:
        for item in evid_set:
            Annotation_ID, Evidence_ID, Wikipedia_URL, sentence_ID = item
            page = Wikipedia_URL
            sent = docs[Wikipedia_URL][sentence_ID].split("\t")[1]
            evidences.add((page, sent))
    return evidences

def sample_negative_evidences(docs, page, positives=set(), num_samples=1):
    evidences = set()
    for sent in docs[page]:
        sent = sent.split("\t")
        if len(sent) < 2:
            continue
        sent_id, sent_text = sent[0], sent[1]
        if len(sent_text.strip()) == 0:
            continue
        if sent_id in positives:
            continue
        evidences.add((page, sent_text))
    return evidences if num_samples is None else random.sample(evidences, min(len(evidences), num_samples))

def get_negative_claim_evidences(docs, evid_sets, pred_pages, max_evidences_per_page=None):
    positive_sentences = {}
    for evid_set in evid_sets:
        for item in evid_set:
            Annotation_ID, Evidence_ID, Wikipedia_URL, sentence_ID = item
            if not Wikipedia_URL in positive_sentences:
                positive_sentences[Wikipedia_URL] = set()
            positive_sentences[Wikipedia_URL].add(sentence_ID)

    # sample negative examples from pages where good evidences are, by
    # avoiding to select the good evidences themself
    for page,positives in positive_sentences.items():
        for evidence in sample_negative_evidences(docs, page, positives, num_samples=max_evidences_per_page):
            yield evidence

    # sample negative examples from other predicted pages in which there
    # are not good evidences.
    for page in pred_pages:
        if page in positive_sentences:
            continue
        for evidence in sample_negative_evidences(docs, page, num_samples=max_evidences_per_page):
            yield evidence

def fetch_documents(db, evid_sets, pred_pages):
    pages = set(pred_pages)
    for evid_set in evid_sets:
        for item in evid_set:
            _, _, page, _ = item
            pages.add(page)

    docs = defaultdict(lambda:[])
    for page,lines in db.get_all_doc_lines(pages):
        docs[page] = lines.split("\n")
    return docs

def main(db_file, in_file, out_file, max_evidences_per_page):
    path = os.getcwd()
    outfile = open(os.path.join(path, out_file), "w+")

    db = FeverDocDB(db_file)
    jlr = JSONLineReader()

    with open(os.path.join(path, in_file), "r") as f:
        lines = jlr.process(f)
        for line in tqdm(lines):
            # if not verifiable, we don't have evidence and just continue
            if line["verifiable"] ==  "NOT VERIFIABLE":
                continue

            claim = line["claim"]
            label = line["label"]
            evid_sets = line["evidence"]
            pred_pages = line["predicted_pages"]

            docs = fetch_documents(db, evid_sets, pred_pages)

            # write positive and negative evidence to file
            for page,evidence in get_positive_claim_evidences(docs, evid_sets):
                outfile.write('\t'.join([page, claim, evidence, '1']) + '\n')
            for page,evidence in get_negative_claim_evidences(docs, evid_sets, pred_pages, max_evidences_per_page=max_evidences_per_page):
                outfile.write('\t'.join([page, claim, evidence, '0']) + '\n')
    outfile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-file", type=str,
                        help="database file which contains wiki pages")
    parser.add_argument("--in-file", type=str, help="input dataset")
    parser.add_argument("--out-file", type=str,
                        help="path to save output dataset")
    parser.add_argument("--max-neg-evidences-per-page", type=int,
                        help="number of negative evidance to in each of the page that are relevant for a claim")
    args = parser.parse_args()
    main(args.db_file, args.in_file, args.out_file, args.max_neg_evidences_per_page)
