# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function
import logging
import re
from nltk.corpus import wordnet

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(
        self,
        guid,
        text_a,
        text_b=None,
        label=None,
        POS=None,
        FGPOS=None,
        text_a_2=None,
        text_b_2=None,
    ):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.POS = POS
        self.FGPOS = FGPOS
        self.text_a_2 = text_a_2
        self.text_b_2 = text_b_2


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids,
        input_mask,
        segment_ids,
        label_id,
        guid=None,
        input_ids_2=None,
        input_mask_2=None,
        segment_ids_2=None,
        input_ids_3=None,
        input_mask_3=None
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.guid = guid
        self.input_ids_2 = input_ids_2
        self.input_mask_2 = input_mask_2
        self.segment_ids_2 = segment_ids_2
        self.input_ids_3 = input_ids_3
        self.input_mask_3 = input_mask_3


def convert_examples_to_two_features(
    examples, label_list, max_seq_length, tokenizer, output_task, args
):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)  # tokenize the sentence
        tokens_b = None
        text_b = None

        try:
            text_b = int(example.text_b)  # index of target word
            tokens_b = text_b

            # truncate the sentence to max_seq_len
            if len(tokens_a) > max_seq_length - 6:
                tokens_a = tokens_a[: (max_seq_length - 6)]

            # Find the target word index
            for i, w in enumerate(example.text_a.split()):
                # If w is a target word, tokenize the word and save to text_b
                if i == text_b:
                    # consider the index due to models that use a byte-level BPE as a tokenizer (e.g., GPT2, RoBERTa)
                    text_b = tokenizer.tokenize(w) if i == 0 else tokenizer.tokenize(" " + w)
                    break

                w_tok = tokenizer.tokenize(w) if i == 0 else tokenizer.tokenize(" " + w)

                # Count number of tokens before the target word to get the target word index
                if w_tok:
                    tokens_b += len(w_tok) - 1

            if tokens_b + len(text_b) > max_seq_length - 6:
                continue

        except TypeError:
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[: (max_seq_length - 2)]

        tokens = [tokenizer.cls_token] + tokens_a + [tokenizer.sep_token]

        # POS tag tokens
        if args.use_pos:
            POS_token = tokenizer.tokenize(example.POS)
            tokens += POS_token + [tokenizer.sep_token]

        # Local context
        if args.use_local_context:
            local_start = 1
            local_end = local_start + len(tokens_a)
            comma1 = tokenizer.tokenize(",")[0]
            comma2 = tokenizer.tokenize(" ,")[0]
            for i, w in enumerate(tokens):
                if i < tokens_b + 1 and (w in [comma1, comma2]):
                    local_start = i
                if i > tokens_b + 1 and (w in [comma1, comma2]):
                    local_end = i
                    break
            segment_ids = [
                2 if i >= local_start and i <= local_end else 0 for i in range(len(tokens))
            ]
        else:
            segment_ids = [0] * len(tokens)

        # POS tag encoding
        after_token_a = False
        for i, t in enumerate(tokens):
            if t == tokenizer.sep_token:
                after_token_a = True
            if after_token_a and t != tokenizer.sep_token:
                segment_ids[i] = 3

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        try:
            tokens_b += 1  # add 1 to the target word index considering [CLS]
            for i in range(len(text_b)):
                segment_ids[tokens_b + i] = 1
        except TypeError:
            pass

        input_mask = [1] * len(input_ids)
        padding = [tokenizer.convert_tokens_to_ids(tokenizer.pad_token)] * (
            max_seq_length - len(input_ids)
        )
        input_ids += padding
        input_mask += [0] * len(padding)
        segment_ids += [0] * len(padding)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_task == "classification":
            label_id = label_map[example.label]
        else:
            raise KeyError(output_task)

        # Second features (Target word)
        tokens = [tokenizer.cls_token] + text_b + [tokenizer.sep_token]
        segment_ids_2 = [0] * len(tokens)
        try:
            tokens_b = 1  # add 1 to the target word index considering [CLS]
            for i in range(len(text_b)):
                segment_ids_2[tokens_b + i] = 1
        except TypeError:
            pass

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_ids_2 = tokenizer.convert_tokens_to_ids(tokens)
        input_mask_2 = [1] * len(input_ids_2)

        padding = [tokenizer.convert_tokens_to_ids(tokenizer.pad_token)] * (
            max_seq_length - len(input_ids_2)
        )
        input_ids_2 += padding
        input_mask_2 += [0] * len(padding)
        segment_ids_2 += [0] * len(padding)

        assert len(input_ids_2) == max_seq_length
        assert len(input_mask_2) == max_seq_length
        assert len(segment_ids_2) == max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                guid=example.guid + " " + str(example.text_b),
                input_ids_2=input_ids_2,
                input_mask_2=input_mask_2,
                segment_ids_2=segment_ids_2,
            )
        )

    return features
    
def convert_examples_to_three_features(
    examples, label_list, max_seq_length, tokenizer, output_task, args
):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)  # tokenize the sentence
        tokens_b = None
        text_b = None
        text_c = None # target word meaning

        try:
            text_b = int(example.text_b)  # index of target word
            tokens_b = text_b

            # truncate the sentence to max_seq_len
            if len(tokens_a) > max_seq_length - 6:
                tokens_a = tokens_a[: (max_seq_length - 6)]

            # Find the target word index
            for i, w in enumerate(example.text_a.split()):
                # If w is a target word, tokenize the word and save to text_b
                if i == text_b:
                    # consider the index due to models that use a byte-level BPE as a tokenizer (e.g., GPT2, RoBERTa)
                    text_b = tokenizer.tokenize(w) if i == 0 else tokenizer.tokenize(" " + w)
                    text_c = get_basic_meaning(w.replace(r'\W+', ''))
                    text_c = tokenizer.tokenize(text_c)
                    break

                w_tok = tokenizer.tokenize(w) if i == 0 else tokenizer.tokenize(" " + w)

                # Count number of tokens before the target word to get the target word index
                if w_tok:
                    tokens_b += len(w_tok) - 1

            if tokens_b + len(text_b) > max_seq_length - 6:
                continue

        except TypeError as e:
            print(f"some errors!!!! {e}")
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[: (max_seq_length - 2)]

        tokens = [tokenizer.cls_token] + tokens_a + [tokenizer.sep_token]

        # POS tag tokens
        if args.use_pos:
            POS_token = tokenizer.tokenize(example.POS)
            tokens += POS_token + [tokenizer.sep_token]

        # Local context
        if args.use_local_context:
            local_start = 1
            local_end = local_start + len(tokens_a)
            comma1 = tokenizer.tokenize(",")[0]
            comma2 = tokenizer.tokenize(" ,")[0]
            for i, w in enumerate(tokens):
                if i < tokens_b + 1 and (w in [comma1, comma2]):
                    local_start = i
                if i > tokens_b + 1 and (w in [comma1, comma2]):
                    local_end = i
                    break
            segment_ids = [
                2 if i >= local_start and i <= local_end else 0 for i in range(len(tokens))
            ]
        else:
            segment_ids = [0] * len(tokens)

        # POS tag encoding
        after_token_a = False
        for i, t in enumerate(tokens):
            if t == tokenizer.sep_token:
                after_token_a = True
            if after_token_a and t != tokenizer.sep_token:
                segment_ids[i] = 3

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        try:
            tokens_b += 1  # add 1 to the target word index considering [CLS]
            for i in range(len(text_b)):
                segment_ids[tokens_b + i] = 1
        except TypeError:
            pass

        input_mask = [1] * len(input_ids)
        padding = [tokenizer.convert_tokens_to_ids(tokenizer.pad_token)] * (
            max_seq_length - len(input_ids)
        )
        input_ids += padding
        input_mask += [0] * len(padding)
        segment_ids += [0] * len(padding)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_task == "classification":
            label_id = label_map[example.label]
        else:
            raise KeyError(output_task)

        # Second features (Target word)
        tokens = [tokenizer.cls_token] + text_b + [tokenizer.sep_token]
        segment_ids_2 = [0] * len(tokens)
        try:
            tokens_b = 1  # add 1 to the target word index considering [CLS]
            for i in range(len(text_b)):
                segment_ids_2[tokens_b + i] = 1
        except TypeError:
            pass

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_ids_2 = tokenizer.convert_tokens_to_ids(tokens)
        input_mask_2 = [1] * len(input_ids_2)

        padding = [tokenizer.convert_tokens_to_ids(tokenizer.pad_token)] * (
            max_seq_length - len(input_ids_2)
        )
        input_ids_2 += padding
        input_mask_2 += [0] * len(padding)
        segment_ids_2 += [0] * len(padding)

        assert len(input_ids_2) == max_seq_length
        assert len(input_mask_2) == max_seq_length
        assert len(segment_ids_2) == max_seq_length

        # Third feature basic meaning
        tokens = [tokenizer.cls_token] + text_c + [tokenizer.sep_token]
        input_ids_3 = tokenizer.convert_tokens_to_ids(tokens)
        input_mask_3 = [1] * len(input_ids_3)

        padding = [tokenizer.convert_tokens_to_ids(tokenizer.pad_token)] * (
            max_seq_length - len(input_ids_3)
        )
        input_ids_3 += padding
        input_mask_3 += [0] * len(padding)

        assert len(input_ids_3) == max_seq_length
        assert len(input_mask_3) == max_seq_length        
        
        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                guid=example.guid + " " + str(example.text_b),
                input_ids_2=input_ids_2,
                input_mask_2=input_mask_2,
                segment_ids_2=segment_ids_2,
                input_ids_3=input_ids_3,
                input_mask_3=input_mask_3,
            )
        )

    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def get_basic_meaning(word):
    """Returns the basic definition of a word using WordNet."""
    # regex to remove punctuations
    word = re.sub(r'[^\w\s]', '', word)
    word = word.lower().strip()
    synsets = wordnet.synsets(word)
    if synsets:
        return synsets[0].definition()
    else:
        print(f"Word {word} not found in WordNet")
        return ''
