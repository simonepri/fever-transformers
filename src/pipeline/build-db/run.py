#!/usr/bin/env python3

# Adapted from https://github.com/sheffieldnlp/fever-naacl-2018/blob/5ebeaf4/src/scripts/build_db.py
# Originally taken from https://github.com/facebookresearch/DrQA/blob/a1082db/scripts/retriever/build_db.py
#
# Additional license and copyright information for this source code are available at:
# https://github.com/facebookresearch/DrQA/blob/master/LICENSE
# https://github.com/sheffieldnlp/fever-naacl-2018/blob/master/LICENSE
"""A script to read in and store documents in a sqlite database."""

import argparse
import json
import importlib.util
import logging
import os
import sqlite3
import unicodedata
from multiprocessing import Pool as ProcessPool

from tqdm import tqdm


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s: [ %(message)s ]", "%m/%d/%Y %H:%M:%S")
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


# ------------------------------------------------------------------------------
# Import helper
# ------------------------------------------------------------------------------


PREPROCESS_FN = None


def init(filename):
    global PREPROCESS_FN
    if filename:
        PREPROCESS_FN = import_module(filename).preprocess


def import_module(filename):
    """Import a module given a full path to the file."""
    spec = importlib.util.spec_from_file_location("doc_filter", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------
# Store corpus.
# ------------------------------------------------------------------------------


def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError("Path %s is invalid" % path)


def get_contents(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    global PREPROCESS_FN
    documents = []
    with open(filename) as f:
        for line in f:
            # Parse document
            doc = json.loads(line)
            # Maybe preprocess the document with custom function
            if PREPROCESS_FN:
                doc = PREPROCESS_FN(doc)
            # Skip if it is empty or None
            if not doc:
                continue
            # Add the document
            documents.append(
                (unicodedata.normalize(
                    "NFD", doc["id"]), doc["lines"])
            )
    return documents


def store_contents(data_path, save_path, preprocess, num_workers=None):
    """Preprocess and store a corpus of documents in sqlite.

    Args:
        data_path: Root path to directory (or directory of directories) of files
          containing json encoded documents (must have `id` and `lines` fields).
        save_path: Path to output sqlite db.
        preprocess: Path to file defining a custom `preprocess` function. Takes
          in and outputs a structured doc.
        num_workers: Number of parallel processes to use when reading docs.
    """
    if os.path.isfile(save_path):
        raise RuntimeError("%s already exists! Not overwriting." % save_path)

    logger.info("Reading into database...")
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("CREATE TABLE documents (id PRIMARY KEY, lines);")

    workers = ProcessPool(num_workers, initializer=init,
                          initargs=(preprocess,))
    files = [f for f in iter_files(data_path)]
    count = 0
    for pairs in tqdm(
            workers.imap_unordered(
                get_contents,
                files),
            total=len(files)):
        count += len(pairs)
        c.executemany("INSERT INTO documents VALUES (?,?)", pairs)
    logger.info("Read %d docs." % count)
    logger.info("Committing...")
    conn.commit()
    conn.close()


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, help="/path/to/data")
    parser.add_argument("--save-path", type=str, help="/path/to/saved/db.db")
    parser.add_argument(
        "--preprocess",
        type=str,
        default=None,
        help=(
            "File path to a python module that defines "
            "a `preprocess` function"),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of CPU processes (for tokenizing, etc)",
    )
    args = parser.parse_args()

    save_dir = os.path.dirname(args.save_path)
    if not os.path.exists(save_dir):
        logger.info(
            "Save directory doesn't exist. Making {0}".format(save_dir))
        os.makedirs(save_dir)

    store_contents(args.data_path, args.save_path,
                   args.preprocess, args.num_workers)
