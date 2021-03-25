#!/usr/bin/env bash
export ROOT_DIR=/dockerdata/wengeliu
export TASK_NAME=cosmosqa
export BACKBONE=roberta
export BACKBONE_SIZE=large
export MODEL_DIR=$ROOT_DIR/BERT_MODELS/${BACKBONE}-${BACKBONE_SIZE}-uncased
export DATA_DIR=$ROOT_DIR/datasets/cosmosqa/cosmosqa_cometgenerated

CUDA_VISIBLE_DEVICES=7 python save_evidence.py \
    --task_name $TASK_NAME \
    --model_name_or_path $MODEL_DIR \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir $DATA_DIR \
    --batch_size 1 \
    --max_seq_length 128 \
    --max_evid_length 20 \
    --num_paragraph_sents 5 \
    --num_evidence_sents 25