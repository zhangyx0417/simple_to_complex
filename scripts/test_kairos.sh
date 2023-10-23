#!/usr/bin/env bash
set -e 
set -x 
CKPT_1=test
CKPT_2=test

rm -rf checkpoints/${CKPT_1}-pred
python train.py \
    --model=constrained-gen \
    --ckpt_name=${CKPT_1}-pred \
    --load_ckpt=checkpoints/${CKPT_1}/epoch=4-v1.ckpt \
    --train_file=data/wikievents/train_info.jsonl \
    --val_file=data/wikievents/dev_info.jsonl \
    --test_file=data/wikievents/test_info.jsonl \
    --train_batch_size=2 \
    --eval_batch_size=1 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=4 \
    --num_train_epochs=5 \
    --coref_dir=data/wikievents/coref \
    --eval_only \
    --retrieval_augmented

python src/genie/order.py --ckpt_name=${CKPT_1}-pred

rm -rf checkpoints/${CKPT_2}-pred
python train.py \
    --model=constrained-gen \
    --ckpt_name=${CKPT_2}-pred \
    --load_ckpt=checkpoints/${CKPT_2}/epoch=4-v1.ckpt \
    --train_file=data/wikievents/train_info.jsonl \
    --val_file=data/wikievents/dev_info.jsonl \
    --test_file=data/wikievents/test_info.jsonl \
    --train_batch_size=2 \
    --eval_batch_size=1 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=4 \
    --num_train_epochs=5 \
    --coref_dir=data/wikievents/coref \
    --eval_only \
    --ordering \
    --retrieval_augmented \
    --constrained_decoding \
    --revised_constraints

python src/genie/scorer.py \
    --ckpt_name=${CKPT_2}-pred \
    --test_file=data/wikievents/test_info.jsonl \
    --coref_file=data/wikievents/coref/test.jsonlines \
    --head_only \
    --coref \
    --ordering