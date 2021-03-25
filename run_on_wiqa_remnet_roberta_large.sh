#!/usr/bin/env bash
export ROOT_DIR=/data2/yinyahuang
export TASK_NAME=wiqa
export BACKBONE=roberta
export MODEL_DIR=$ROOT_DIR/BERT_MODELS/${BACKBONE}-large-uncased
export DATA_DIR=$ROOT_DIR/datasets/wiqa/wiqa_cometgenerated_conceptnet
export EVID_DIR=$ROOT_DIR/datasets/wiqa/wiqa_cometgenerated_conceptnet/feats_${BACKBONE}_large
export SAVE_DIR=remnet_roberta_large


CUDA_VISIBLE_DEVICES=7 python run_multiple_choice.py \
    --disable_tqdm \
    --task_name $TASK_NAME \
    --model_name_or_path $MODEL_DIR \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir $DATA_DIR \
    --evid_dir $EVID_DIR \
    --recursive_step 2 \
    --erasure_k 50 \
    --max_seq_length 128 \
    --per_device_eval_batch_size 8 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 20 \
    --learning_rate 5e-6 \
    --seed 42 \
    --save_steps 0 \
    --output_dir Checkpoints/$TASK_NAME/$SAVE_DIR