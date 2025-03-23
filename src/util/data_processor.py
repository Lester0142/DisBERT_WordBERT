import os
import sys
import csv
from util.data_helper import InputExample

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, "utf-8") for cell in line)
                lines.append(line)
            return lines


class TrofiProcessor(DataProcessor):
    """Processor for the TroFi and MOH-X data set."""

    def get_train_examples(self, data_dir, k=None):
        """See base class."""
        if k is not None or k >= 0:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train" + str(k) + ".tsv")), "train"
            )
        else:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
            )

    def get_test_examples(self, data_dir, k=None):
        """See base class."""
        if k is not None  or k >= 0:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test" + str(k) + ".tsv")), "test"
            )
        else:
            return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_dev_examples(self, data_dir, k=None):
        """See base class."""
        if k is not None:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev" + str(k) + ".tsv")), "dev"
            )
        else:
            return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[2]
            label = line[1]
            POS = line[3]
            FGPOS = line[4]
            index = line[-1]
            examples.append(
                InputExample(
                    guid=guid, text_a=text_a, text_b=index, label=label, POS=POS, FGPOS=FGPOS
                )
            )
        return examples


class VUAProcessor(DataProcessor):
    """Processor for the VUA data set."""

    def get_train_examples(self, data_dir, k=None):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_test_examples(self, data_dir, k=None):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_dev_examples(self, data_dir, k=None):
        """See base class."""
        try:
            return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
        except FileNotFoundError:
            return self.get_test_examples(data_dir, k)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[2]
            label = line[1]
            POS = line[3]
            FGPOS = line[4]
            if len(line) == 8:
                index = line[5]
                text_a_2 = line[6]
                index_2 = line[7]
                examples.append(
                    InputExample(
                        guid=guid,
                        text_a=text_a,
                        text_b=index,
                        label=label,
                        POS=POS,
                        FGPOS=FGPOS,
                        text_a_2=text_a_2,
                        text_b_2=index_2,
                    )
                )
            else:
                index = line[-1]
                examples.append(
                    InputExample(
                        guid=guid, text_a=text_a, text_b=index, label=label, POS=POS, FGPOS=FGPOS
                    )
                )
        return examples

PROCESSORS = {
    "vua": VUAProcessor,
    "trofi": TrofiProcessor,
}

OUTPUT_TASKS = {
    "vua": "classification",
    "trofi": "classification",
}