# Adapted from https://github.com/facebookresearch/DrQA/blob/master/drqa/retriever/doc_db.py
#
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file at https://github.com/facebookresearch/DrQA/blob/master/LICENSE
"""Documents, in a sqlite database."""

import sqlite3
import unicodedata


class FeverDocDB(object):
    """Sqlite backed document storage."""

    def __init__(self, db_path):
        self.path = db_path
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def get_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_doc_text(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT text FROM documents WHERE id = ?",
            (unicodedata.normalize("NFD", doc_id),),
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]

    def get_doc_lines(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT lines FROM documents WHERE id = ?",
            (unicodedata.normalize("NFD", doc_id),),
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]

    def get_all_doc_text(self, doc_ids):
        """Fetch the raw text of the docs in 'doc_ids'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT id,text FROM documents WHERE id IN ({})".format(','.join(['?']*len(doc_ids))),
            [unicodedata.normalize("NFD", doc_id) for doc_id in doc_ids]
        )
        results = cursor.fetchall()
        cursor.close()
        return results

    def get_all_doc_lines(self, doc_ids):
        """Fetch the raw text of the docs in 'doc_ids'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT id,lines FROM documents WHERE id IN ({})".format(','.join(['?']*len(doc_ids))),
            [unicodedata.normalize("NFD", doc_id) for doc_id in doc_ids]
        )
        results = cursor.fetchall()
        cursor.close()
        return results
