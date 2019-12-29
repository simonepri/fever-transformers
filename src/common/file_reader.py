# Adapted from https://github.com/UKPLab/fever-2018-team-athene/blob/master/src/retrieval/reader.py
#
# Copyright 2019-present, UKP TU-Darsmtadt
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file at
# https://github.com/UKPLab/fever-2018-team-athene/blob/master/LICENSE.txt
"""Read different types of files"""

import csv
import json


class Reader:
    def __init__(self, encoding="utf-8"):
        self.enc = encoding

    def read(self, file):
        with open(file, "r", encoding=self.enc) as f:
            return self.process(f)

    def process(self, f):
        pass


class CSVReader(Reader):
    def process(self, fp):
        r = csv.DictReader(fp)
        return [line for line in r]


class JSONReader(Reader):
    def process(self, fp):
        return json.load(fp)


class JSONLineReader(Reader):
    def process(self, fp):
        data = []
        for line in fp.readlines():
            data.append(json.loads(line.strip()))
        return data
