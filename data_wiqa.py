import torch
from torch.utils.data.dataset import Dataset
from typing import List, Dict, Optional
from transformers import PreTrainedTokenizer
from filelock import FileLock
import logging
from enum import Enum
from dataclasses import dataclass
import os
import jsonlines
from itertools import chain
from tqdm import tqdm

logger = logging.getLogger(__name__)



question_types = {
    "INPARA_EFFECT": 1,  # in_para
    "EXOGENOUS_EFFECT": 2,  # out_of_para
    "OUTOFPARA_DISTRACTOR": 3  # no_effect
}


class Split(Enum):
    train = "train"
    dev = "eval"
    test = "test"


@dataclass(frozen=True)
class InputFeatures:
    question_type: str
    question_id: int
    input_ids: List[List[int]]
    input_mask: Optional[List[int]]
    segment_ids: Optional[List[int]]
    doc_len: int
    ques_len: int
    label: Optional[int]


class WIQAData(Dataset):
    features: List[InputFeatures]

    def __init__(
            self,
            task: str,
            data_dir: str,
            evid_dir: str,
            tokenizer: PreTrainedTokenizer,
            max_seq_length: Optional[int],
            mode: Split = Split.train,
            overwrite_cache=False,
            demo=False
            ):

        self.evid_dir = evid_dir
        self.mode = mode
        if mode == Split.dev:
            self.evids = torch.load(os.path.join(self.evid_dir, "dev_ante_cons_feats.pt"))
        elif mode == Split.test:
            self.evids = torch.load(os.path.join(self.evid_dir, "test_ante_cons_feats.pt"))

        processor = processors[task]()

        cached_features_file = os.path.join(
            data_dir,
            "cached_data",
            "cached_{}_{}_{}_{}".format(
                mode.value,
                tokenizer.__class__.__name__,
                str(max_seq_length),
                task,
            )
        )
        if demo: cached_features_file += "_demo"

        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")
                label_list = processor.get_labels()
                if mode == Split.dev:
                    if demo:
                        examples = processor.get_dev_demo(data_dir)
                    else:
                        examples = processor.get_dev_examples(data_dir)
                elif mode == Split.test:
                    examples = processor.get_test_examples(data_dir)
                elif mode == Split.train:
                    if demo:
                        examples = processor.get_train_demo(data_dir)
                    else:
                        examples = processor.get_train_examples(data_dir)
                else:
                    raise Exception()
                logger.info("Training examples: %s", len(examples))

                self.features = convert_examples_to_features(
                    examples, tokenizer, max_seq_length
                )
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        item =  self.features[i]
        qid = item.question_id

        if self.mode == Split.train:
            try:
                evid = torch.load(os.path.join(self.evid_dir,
                                               'train_ante_cons_feats',
                                               '{}_ante_cons_feats.pt'.format(qid)
                                               ))
            except:
                print('evid_dir: {} is not available'.format(
                    os.path.join(self.evid_dir, 'train_ante_cons_feats', '{}_ante_cons_feats.pt'.format(qid))
                ))
                assert 1 == 0
        else:
            evid = self.evids[qid]

        ante_feats, cons_feats = evid  # np.array

        return item.input_ids, item.input_mask, item.segment_ids, item.label, ante_feats, cons_feats, question_types[item.question_type]



class WIQAExample(object):
    def __init__(self,
                 question_type,
                 question_id,
                 ante_evidence,
                 cons_evidence,
                 context_sentence,
                 context_list,
                 start_ending,
                 label=None):
        self.question_type = question_type
        self.question_id = question_id
        self.ante_evidence = ante_evidence
        self.cons_evidence = cons_evidence
        self.context_sentence = context_sentence
        self.context_list = context_list,
        self.start_ending = start_ending
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"question_type: {self.question_type}",
            f"question_id: {self.question_id}",
            f"ante_evidence: {self.ante_evidence}",
            f"cons_evidence: {self.cons_evidence}",
            f"context_sentence: {self.context_sentence}",
            f"context_list: {self.context_list}",
            f"start_ending: {self.start_ending}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)



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
    def _read_json(cls, input_file):
        reader = jsonlines.Reader(open(input_file, "r"))
        lines = [each for each in reader]
        return lines


class WIQAProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.jsonl")))

    def get_train_demo(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train_10.jsonl")))

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.jsonl")))

    def get_dev_demo(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev_10.jsonl")))

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.jsonl")))

    def get_examples_from_file(self, input_file):
        return self._create_examples(
            self._read_json(input_file))

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _create_examples(self, records):
        examples = []
        for (i, record) in enumerate(records):
            question_type = record["metadata"]["question_type"]
            question_id = record['metadata']['ques_id']
            context_list = record["question"]["para_steps"]
            context = '. '.join(context_list)
            if "" in context_list:
                context_list.remove("")

            ante_evidence = []
            for k, v in record['question']['evidence1'].items():
                ante_evidence += v
            cons_evidence = []
            for k, v in record['question']['evidence2'].items():
                cons_evidence += v

            question = record["question"]["stem"]
            label = ord(record["question"]["answer_label_as_choice"]) - ord("A")
            examples.append(
                WIQAExample(
                    question_type=question_type,
                    question_id=question_id,
                    ante_evidence=ante_evidence,
                    cons_evidence=cons_evidence,
                    context_sentence=context,
                    context_list=context_list,
                    start_ending=question,
                    label=int(label))
            )
        return examples

    def label_field(self):
        return "AnswerRightEnding"


def convert_examples_to_features(examples, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for example_index, example in tqdm(enumerate(examples), desc="converting examples to features"):
        CLS = tokenizer.cls_token
        SEP = tokenizer.sep_token
        PAD_id = tokenizer.pad_token_id

        ante_evidence = example.ante_evidence  # list of strings. len=155
        cons_evidence = example.cons_evidence  # list of strings. len=155
        evidence = list(chain.from_iterable(zip(ante_evidence, cons_evidence)))
        assert len(evidence) == len(ante_evidence) + len(cons_evidence)
        evidence = ' '.join(evidence)
        evid_tokens = tokenizer.tokenize(evidence)

        context_tokens = tokenizer.tokenize(example.context_sentence)
        start_ending_tokens = tokenizer.tokenize(example.start_ending)
        context_tokens += evid_tokens  # added by Eleanor.
        _truncate_seq_pair(context_tokens, start_ending_tokens, max_seq_length - 3)
        doc_len = len(context_tokens)
        ques_len = len(start_ending_tokens)

        tokens = [CLS] + context_tokens + [SEP] + start_ending_tokens + [SEP]
        segment_ids = [0] * (len(context_tokens) + 2) + [1] * (len(start_ending_tokens) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        padding_ids = [PAD_id] * (max_seq_length - len(input_ids))
        input_ids += padding_ids
        input_mask += padding
        segment_ids += padding

        label = int(example.label)
        question_type = example.question_type
        question_id = example.question_id

        features.append(
            InputFeatures(
                question_type=question_type,
                question_id=question_id,
                label=label,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                doc_len=doc_len,
                ques_len=ques_len,
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

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


def _truncate_sequences(max_length, inputs):
    idx = 0
    for ta, tb in zip(inputs[0], inputs[1]):
        _truncate_seq_pair(ta, tb, max_length)


processors = {
    "wiqa": WIQAProcessor,
}