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
""" Finetuning the library models for multiple choice (Bert, Roberta, XLNet)."""


import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
import torch

from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from mytrainer import Trainer


logger = logging.getLogger(__name__)



def simple_accuracy(preds, labels):
    return (preds == labels).mean()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    recursive_step: int = field(
        default=2,
        metadata={"help": "REM-Net recursive step"}
    )
    erasure_k: int = field(
        default=2,
        metadata={"help": "REM-Net number of erased evidence per recursive step"}
    )
    attention_drop: float = field(
        default=0.1,
        metadata={"help": "huggingface RoBERTa config.attention_probs_dropout_prob"}
    )
    hidden_drop: float = field(
        default=0.1,
        metadata={"help": "huggingface RoBERTa config.hidden_dropout_prob"}
    )

    # training
    roberta_lr: float = field(
        default=5e-6,
        metadata={"help": "learning rate for updating roberta parameters"}
    )
    gcn_lr: float = field(
        default=5e-6,
        metadata={"help": "learning rate for updating gcn parameters"}
    )
    proj_lr: float = field(
        default=5e-6,
        metadata={"help": "learning rate for updating fc parameters"}
    )




@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": ""})
    data_dir: str = field(metadata={"help": "Should contain the data files for the task."})
    evid_dir: str = field(metadata={"help": ""})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        }
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    demo_data: bool = field(
        default=False,
        metadata={"help": "demo data sets with 100 samples."}
    )


def collate_fn(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_ids, input_mask, segment_ids, labels, ante_feats, cons_feats, question_types = zip(*batch)

    input_ids = torch.tensor(input_ids)
    input_mask = torch.tensor(input_mask)
    segment_ids = torch.tensor(segment_ids)
    labels = torch.tensor(labels)

    ante_feats = tuple(np.expand_dims(item, axis=0) for item in ante_feats)
    ante_feats = np.concatenate(ante_feats, axis=0)
    ante_feats = torch.from_numpy(ante_feats)

    if cons_feats[0] is not None:
        cons_feats = tuple(np.expand_dims(item, axis=0) for item in cons_feats)
        cons_feats = np.concatenate(cons_feats, axis=0)
        cons_feats = torch.from_numpy(cons_feats)

        evid_feats = [ante_feats.to(device), cons_feats.to(device)]

    else:
        evid_feats = ante_feats.to(device)

    if question_types[0] is not None:
        question_types = torch.tensor(question_types)
    else:
        question_types = None

    # print('collate: {}'.format(evid_feats.size()))

    batch_output = {
        "input_ids": input_ids.to(device),
        "attention_mask": input_mask.to(device),
        "token_type_ids": segment_ids.to(device),
        "labels": labels.to(device),
        "evid_feats": evid_feats,
        "question_types": question_types
    }

    return batch_output




def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)


    from data_wiqa import WIQAProcessor, WIQAData
    from data_cosmosqa import CosmosQAProcessor, CosmosQAData
    processors = {
        "wiqa": WIQAProcessor,
        "cosmosqa": CosmosQAProcessor
    }

    datasets = {
        "wiqa": WIQAData,
        "cosmosqa": CosmosQAData
    }
    if data_args.task_name == "wiqa":
        from data_wiqa import Split
    elif data_args.task_name == "cosmosqa":
        from data_cosmosqa import Split
    else:
        raise NotImplementedError

    try:
        processor = processors[data_args.task_name]()
        label_list = processor.get_labels()
        num_labels = len(label_list)
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    if "roberta-large" in model_args.model_name_or_path:
        from model_remnet_roberta import REMNet  # to remove
    elif "bert-large" in model_args.model_name_or_path or "bert-base" in model_args.model_name_or_path:
        from model_remnet_bert import REMNet
    else:
        raise Exception()

    model = REMNet.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        num_choices=num_labels,
        recursive_step=model_args.recursive_step,
        erasure_k=model_args.erasure_k
    )


    train_dataset = (
        datasets[data_args.task_name](
            task=data_args.task_name,
            data_dir=data_args.data_dir,
            evid_dir=data_args.evid_dir,
            mode=Split.train,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            demo=data_args.demo_data
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        datasets[data_args.task_name](
            task=data_args.task_name,
            data_dir=data_args.data_dir,
            evid_dir=data_args.evid_dir,
            mode=Split.dev,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            demo=data_args.demo_data
        )
        if training_args.do_eval
        else None
    )
    test_dataset = (
        datasets[data_args.task_name](
            task=data_args.task_name,
            data_dir=data_args.data_dir,
            evid_dir=data_args.evid_dir,
            mode=Split.test,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
        )
        if training_args.do_predict
        else None
    )


    def compute_metrics(p: EvalPrediction) -> Dict:
        if hasattr(p, 'question_types'):
            preds = np.argmax(p.predictions, axis=1)
            score_in, n_in = 0.0, 0
            score_out, n_out = 0.0, 0
            score_no, n_no = 0.0, 0
            result_writer = open(os.path.join(training_args.output_dir, "test_preds.txt"), 'a')
            result_writer.write("id, qtype, pred, label\n")
            for i, (pred, qtype, label_id) in enumerate(zip(preds, p.question_types, p.label_ids)):
                result_writer.write('{}, {}, {}, {}\n'.format(i, qtype, pred, label_id))
                if qtype == 1:  # question_type: in_para
                    if pred == label_id: score_in += 1
                    n_in += 1
                elif qtype == 2:  # question_type: out_of_para
                    if pred == label_id: score_out += 1
                    n_out += 1
                elif qtype == 3:  # question_type: no_effect
                    if pred == label_id: score_no += 1
                    n_no += 1
                else:
                    raise Exception()

            result_writer.close()

            score_in = score_in / n_in
            score_out = score_out / n_out
            score_no = score_no / n_no

            return {"acc": simple_accuracy(preds, p.label_ids),
                    "acc_in": score_in,
                    "acc_out": score_out,
                    "acc_no": score_no
                    }

        else:
            preds = np.argmax(p.predictions, axis=1)
            return {"acc": simple_accuracy(preds, p.label_ids)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn
    )


    # Save model_args and data_args
    torch.save(model_args, os.path.join(training_args.output_dir, "model_args.bin"))
    torch.save(data_args, os.path.join(training_args.output_dir, "data_args.bin"))


    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    ckpt_id = model_args.model_name_or_path.split("-")[-1]

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate(task_name=data_args.task_name)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_{}.txt".format(ckpt_id))
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

                results.update(result)

    # Test
    if training_args.do_predict:
        if data_args.task_name == "wiqa":
            logger.info("*** Test ***")

            test_result = trainer.predict(task_name=data_args.task_name,
                                          test_dataset=test_dataset)


            output_test_file = os.path.join(training_args.output_dir, "test_results_{}.txt".format(ckpt_id))
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results *****")
                    for key, value in test_result.metrics.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

                    results.update(test_result.metrics)




def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()