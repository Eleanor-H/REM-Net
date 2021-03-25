import os
import collections
import numpy as np
import json
from transformers import RobertaTokenizer
import torch
from torch.utils.data import Dataset, DataLoader


class CosmosQAExample(object):
    def __init__(self,
                 qid,
                 evidence,
                 para,
                 question,
                 answers,
                 label=None):
        self.qid = qid
        self.evidence = evidence
        self.para = para
        self.question = question
        self.answers = answers
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"qid: {self.qid}",
            f"evidence: {self.evidence}",
            f"para: {self.para}",
            f"question: {self.question}",
            f"answers: {self.answers}",
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

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_json(cls, input_file):
      """Reads a JSON file."""
      with open(input_file, 'rb') as f:
          return json.load(f)

  @classmethod
  def _read_jsonl(cls, input_file):
      """Reads a JSON Lines file."""
      with open(input_file, 'rb') as f:
          return [json.loads(ln) for ln in f]


class CosmosQAProcessor(DataProcessor):
  """Processor for the cosmosqa data set."""
  def __init__(self):

      self.LABELS = ['A', 'B', 'C', 'D']

      self.TRAIN_FILE_NAME = 'train.jsonl'
      self.DEV_FILE_NAME = 'dev.jsonl'
      self.TEST_FILE_NAME = 'test.jsonl'

  def get_train_examples(self, data_dir):
    train_file_name = self.TRAIN_FILE_NAME

    return self._create_examples(
      self._read_jsonl(os.path.join(data_dir, train_file_name)))

  def get_dev_examples(self, data_dir):
      dev_file_name = self.DEV_FILE_NAME

      return self._create_examples(
          self._read_jsonl(os.path.join(data_dir, dev_file_name)))

  def get_test_examples(self, data_dir):
      test_file_name = self.TEST_FILE_NAME

      return self._create_examples(
          self._read_jsonl(os.path.join(data_dir, test_file_name)))

  def get_labels(self):
    return [0, 1, 2, 3]

  def _create_examples(self, records):
      examples = []
      for (i, record) in enumerate(records):
          question_id = record["id"]
          context = record["context"]
          evidence = record["evidence"]
          question = record["question"]
          answers = record["choices"]
          answer_0 = answers[0]["text"]
          answer_1 = answers[1]["text"]
          answer_2 = answers[2]["text"]
          answer_3 = answers[3]["text"]
          label = record["label"]
          if label is not None: label = int(label)
          else: label = int(0)

          examples.append(
              CosmosQAExample(
                  qid=question_id,
                  evidence=evidence,
                  para=context,
                  question=question,
                  answers=[answer_0, answer_1, answer_2, answer_3],
                  label=label)
          )

      return examples


def _truncate_seq_pair(tokens_a, tokens_b, tokens_c, max_length):
    while True:
        total_length=len(tokens_a)+len(tokens_b)+len(tokens_c)+3
        if total_length<=max_length:
            break
        if len(tokens_a)>len(tokens_b)+len(tokens_c):
            tokens_a.pop(-1)
        else:
            tokens_b.pop(0)
    return tokens_a, tokens_b, tokens_c


def _sentence_pointers(para, tokenizer):
    pointers = []
    para_tokens = []
    start, end = int(-1), int(-1)
    for sent in para:
        sent_tokens = tokenizer(sent)
        para_tokens += sent_tokens
        start = int(end + 1)
        end = int(start + len(sent_tokens) - 1)
        pointers.append((start, end))
    return pointers, para_tokens


def _truncate_sentence_pointers(sentence_pointers, para_tokens):
    len_para_tokens = len(para_tokens)
    for i, (start, end) in enumerate(reversed(sentence_pointers)):
        if len_para_tokens < start:
            sentence_pointers.pop(-1)
        elif start <= len_para_tokens < end:
            sentence_pointers.pop(-1)
            sentence_pointers.append((start, len_para_tokens))
            break
        elif len_para_tokens == end:
            break
    return sentence_pointers

def example_to_token_ids_segment_ids_label_ids(
    ex_index,
    example,
    max_seq_length,
    max_evid_length,
    tokenizer):
  """
  Converts an ``InputExample`` to token ids and segment ids.
  two-fold inputs

  1. [CLS] para [SEP] q [SEP] a
     0     0    0     1 1     1

  2. [CLS] evidence [SEP]
     0     0         0
  """
  if ex_index < 5:
    print('*** Example {} ***'.format(ex_index))
    print('qid: {}'.format(example.qid))

  qid = example.qid

  # evidence
  ante_evidence = example.evidence  # list of strings.
  ante_token_ids, ante_segment_ids = [], []
  for ante_sent in ante_evidence:
      ante_sent_tokens = tokenizer.tokenize(ante_sent)
      ante_sent_tokens = ante_sent_tokens[:max_evid_length-2]
      sent_tokens, sent_segment_ids = [], []
      sent_tokens.append("<s>")
      sent_segment_ids.append(0)
      sent_tokens += ante_sent_tokens + ["</s>"]
      sent_segment_ids += [0] * len(ante_sent_tokens) + [0]

      assert len(sent_tokens) <= max_evid_length
      sent_tokens_ids = tokenizer.convert_tokens_to_ids(sent_tokens)

      assert len(sent_tokens_ids) == len(sent_segment_ids)
      ante_token_ids.append(sent_tokens_ids)
      ante_segment_ids.append(sent_segment_ids)
  assert len(ante_evidence) == len(ante_token_ids)
  assert len(ante_evidence) == len(ante_segment_ids)
  l_ante = min(len(ante_evidence), max_evid_length)

  para_tokens = tokenizer.tokenize(example.para)
  question_tokens = tokenizer.tokenize(example.question)
  answers_tokens = map(tokenizer.tokenize, example.answers)

  token_ids = []
  segment_ids = []
  for choice_idx, answer_tokens in enumerate(answers_tokens):

      truncated_para_tokens, \
      truncated_question_tokens,\
      truncated_answer_tokens = _truncate_seq_pair(para_tokens, question_tokens, answer_tokens, max_seq_length)

      choice_tokens = []
      choice_segment_ids = []
      choice_tokens.append("<s>")
      choice_segment_ids.append(0)
      choice_tokens += truncated_para_tokens + ["</s>"]
      choice_segment_ids += [0] * len(truncated_para_tokens) + [0]
      choice_tokens += truncated_question_tokens + ["</s>"]
      choice_segment_ids += [1] * len(truncated_question_tokens) + [1]
      choice_tokens += truncated_answer_tokens
      choice_segment_ids += [1] * len(truncated_answer_tokens)

      choice_token_ids = tokenizer.convert_tokens_to_ids(choice_tokens)

      token_ids.append(choice_token_ids)
      segment_ids.append(choice_segment_ids)

  label_ids = [example.label]

  return ante_token_ids, ante_segment_ids, l_ante, \
         token_ids, segment_ids, label_ids, qid


def segment_question_and_answers(opt, mode):
    processor = CosmosQAProcessor()

    tokenizer = RobertaTokenizer.from_pretrained(opt.model_name_or_path,
                                                 do_lower_case=opt.do_lower_case)

    set_examples = None
    if mode == "train":
        set_examples = processor.get_train_examples(opt.data_dir)
    elif mode == "eval":
        set_examples = processor.get_dev_examples(opt.data_dir)
    elif mode == "test":
        set_examples = processor.get_test_examples(opt.data_dir)

    token_ids_segment_ids_label_ids = [
        example_to_token_ids_segment_ids_label_ids(
            ex_index,
            example,
            opt.max_seq_length,
            opt.max_evid_length,
            tokenizer)
        for ex_index, example in enumerate(set_examples)
    ]

    len_ante, len_cons = [], []
    for ante_token_ids, *_ in token_ids_segment_ids_label_ids:
        len_ante.append(len(ante_token_ids))

    return token_ids_segment_ids_label_ids


class BERTData(Dataset):

    def __init__(self, token_ids_segment_ids_label_ids, max_seq_length, max_evid_length,
                 n_para_sents, n_evid_sents):
        self.examples = token_ids_segment_ids_label_ids
        self.max_seq_length = max_seq_length
        self.max_evid_length = max_evid_length
        self.n_para_sents = n_para_sents
        self.n_evid_sents = n_evid_sents

    def __getitem__(self, index):
        sample = self.examples[index]

        ante_token_ids, ante_segment, l_ante, \
            token_ids, segment_ids, label_ids, qid = sample

        features = collections.OrderedDict()
        features['qid'] = qid

        for i, (choice_token_ids, choice_segment_ids) in enumerate(zip(token_ids, segment_ids)):

            input_ids = np.zeros(self.max_seq_length)
            input_ids[:len(choice_token_ids)] = np.array(choice_token_ids)

            input_mask = np.zeros(self.max_seq_length)
            input_mask[:len(choice_token_ids)] = 1

            segment_ids = np.zeros(self.max_seq_length)
            segment_ids[:len(choice_segment_ids)] = np.array(choice_segment_ids)

            features[f'input_ids{i}'] = torch.from_numpy(input_ids).long()
            features[f'input_mask{i}'] = torch.from_numpy(input_mask).long()
            features[f'segment_ids{i}'] = torch.from_numpy(segment_ids).long()

        features[f'label_ids'] = torch.Tensor(label_ids).long()

        # evidence features
        ante_evid_ids = np.zeros((self.n_evid_sents, self.max_evid_length))
        ante_evid_mask = np.zeros((self.n_evid_sents, self.max_evid_length))
        ante_segment_ids = np.zeros((self.n_evid_sents, self.max_evid_length))
        ante_keypadding_mask = np.zeros(self.n_evid_sents)

        n = min(self.n_evid_sents, len(ante_token_ids))
        for i in range(n):
            ante_evid_ids[i, :len(ante_token_ids[i])] = np.array(ante_token_ids[i])
            ante_evid_mask[i, :len(ante_token_ids[i])] = 1
            ante_segment_ids[i, :len(ante_segment[i])] = np.array(ante_segment[i])
            ante_keypadding_mask[i] = 1

        features['ante_evid_ids'] = torch.from_numpy(ante_evid_ids).long()  # n_evid_sents x max_seq_length
        features['ante_evid_mask'] = torch.from_numpy(ante_evid_mask).long()
        features['ante_segment_ids'] = torch.from_numpy(ante_segment_ids).long()
        features['ante_keypadding_mask'] = torch.from_numpy(ante_keypadding_mask).bool()

        return features

    def __len__(self):
        return len(self.examples)


def bertLoader(token_ids_segment_ids_label_ids, opt):

    dataset = BERTData(token_ids_segment_ids_label_ids,
                       max_seq_length=opt.max_seq_length,
                       max_evid_length=opt.max_evid_length,
                       n_para_sents=opt.num_paragraph_sents,
                       n_evid_sents=opt.num_evidence_sents)

    loader = DataLoader(dataset=dataset, batch_size=opt.batch_size,
                        shuffle=False, num_workers=16)

    return loader


def data_cosmosqa(opt, mode):
    token_ids_segment_ids_label_ids = segment_question_and_answers(opt, mode)
    loader = bertLoader(token_ids_segment_ids_label_ids, opt)
    return loader

