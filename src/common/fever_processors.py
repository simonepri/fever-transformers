# Adapted from https://github.com/huggingface/transformers/blob/v2.3.0/transformers/data/processors/glue.py
#
# Additional license and copyright information for this source code are available at:
# https://github.com/huggingface/transformers/blob/master/LICENSE
""" FEVER processors and helpers """

import csv
import logging
import os
import re

import numpy as np

from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from transformers.file_utils import is_tf_available


if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def fever_convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: FEVER task
        label_list: List of labels. Can be obtained from the processor using the
            ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or
            ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left
            rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is
            usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be
            filled by ``1`` for actual values and by ``0`` for padded values. If
            set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        A list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    if task is not None:
        processor = fever_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = fever_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    for (ex_index, example) in enumerate(examples):
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        if output_mode == "classification":
            label_map = {label: i for i, label in enumerate(label_list)}
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        yield InputFeatures(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            label=label,
        )


def fever_compute_metrics(task_name, preds, labels):
    def mse(preds, labels):
        return np.mean((labels - preds) ** 2)

    def accuracy(preds, labels):
        return (preds == labels).mean()

    assert len(preds) == len(labels)
    if task_name == "sentence_retrieval":
        return {"mse": mse(preds, labels)}
    if task_name == "claim_verification":
        return {"acc": accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def process_sent(sentence):
    sentence = convert_to_unicode(sentence)
    sentence = re.sub(" \-LSB\-.*?\-RSB\-", "", sentence)
    sentence = re.sub("\-LRB\- \-RRB\- ", "", sentence)
    sentence = re.sub(" -LRB-", " ( ", sentence)
    sentence = re.sub("-RRB-", " )", sentence)
    sentence = re.sub("--", "-", sentence)
    sentence = re.sub("``", '"', sentence)
    sentence = re.sub("''", '"', sentence)
    return sentence


def process_title(title):
    title = convert_to_unicode(title)
    title = re.sub("_", " ", title)
    title = re.sub(" -LRB-", " ( ", title)
    title = re.sub("-RRB-", " )", title)
    title = re.sub("-COLON-", ":", title)
    return title


def process_evid(sentence):
    sentence = convert_to_unicode(sentence)
    sentence = re.sub(" -LSB-.*-RSB-", " ", sentence)
    sentence = re.sub(" -LRB- -RRB- ", " ", sentence)
    sentence = re.sub("-LRB-", "(", sentence)
    sentence = re.sub("-RRB-", ")", sentence)
    sentence = re.sub("-COLON-", ":", sentence)
    sentence = re.sub("_", " ", sentence)
    sentence = re.sub("\( *\,? *\)", "", sentence)
    sentence = re.sub("\( *[;,]", "(", sentence)
    sentence = re.sub("--", "-", sentence)
    sentence = re.sub("``", '"', sentence)
    sentence = re.sub("''", '"', sentence)
    return sentence


def process_label(label):
    label = convert_to_unicode(label)
    return label


class SentenceRetrievalProcessor(DataProcessor):
    """Processor for the sentence retrieval data set."""

    def get_examples(self, file_path, purpose):
        """See base class."""
        with open(file_path, "r", encoding="utf-8-sig") as f:
            lines = csv.reader(f, delimiter="\t")
            for (i, line) in enumerate(lines):
                guid = "%s-%d" % (purpose, i)
                title = process_title(line[2])
                text_a = process_sent(line[1])
                text_b = process_evid(line[4])
                text_b = title + " : " + text_b
                label = process_label(line[5]) if purpose != "predict" else self.get_dummy_label()
                yield InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)

    def get_length(self, file_path):
        """Return the number of examples."""
        return sum(1 for line in open(file_path, "r", encoding="utf-8-sig"))

    def get_labels(self):
        """See base class."""
        return [None]

    def get_dummy_label(self):
        return "-1"


class ClaimVerificationProcessor(SentenceRetrievalProcessor):
    """Processor for the claim verification data set."""

    def get_labels(self):
        """See base class."""
        return ["R", "S", "N"]  # REFUTES, SUPPORTS, NOT ENOUGH INFO

    def get_dummy_label(self):
        return "N"


fever_processors = {
    "sentence_retrieval": SentenceRetrievalProcessor,
    "claim_verification": ClaimVerificationProcessor,
}

fever_tasks_num_labels = {
    "sentence_retrieval": 1,
    "claim_verification": 3,
}
fever_output_modes = {
    "sentence_retrieval": "regression",
    "claim_verification": "classification",
}
