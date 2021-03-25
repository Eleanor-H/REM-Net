#!/usr/bin/env bash
export ROOT_DIR=/data2/yinyahuang
export TASK_NAME=cosmosqa
export BACKBONE=roberta
export MODEL_DIR=$ROOT_DIR/BERT_MODELS/${BACKBONE}-large-uncased
export DATA_DIR=$ROOT_DIR/datasets/cosmosqa/cosmosqa_cometgenerated
export EVID_DIR=$ROOT_DIR/datasets/cosmosqa/cosmosqa_cometgenerated/feats_${BACKBONE}_large
export SAVE_DIR=remnet_roberta_large


CUDA_VISIBLE_DEVICES=5 python run_multiple_choice.py \
    --overwrite_output_dir \
    --disable_tqdm \
    --task_name $TASK_NAME \
    --model_name_or_path $MODEL_DIR \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir $DATA_DIR \
    --evid_dir $EVID_DIR \
    --recursive_step 2 \
    --erasure_k 3 \
    --max_seq_length 128 \
    --per_device_eval_batch_size 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 10 \
    --learning_rate 1e-5 \
    --seed 42 \
    --save_steps 0 \
    --output_dir Checkpoints/$TASK_NAME/$SAVE_DIR