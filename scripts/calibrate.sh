#!/usr/bin/env bash
set -e
set -x
CKPT=test

rm -rf checkpoints/${CKPT}-calibrated

python train.py --model=constrained-gen --ckpt_name=${CKPT}-calibrated \
    --load_ckpt=checkpoints/${CKPT}/epoch=4-v1.ckpt \
    --train_file=data/wikievents/train_info.jsonl \
    --val_file=data/wikievents/dev_info.jsonl \
    --test_file=data/wikievents/test_info.jsonl \
    --train_batch_size=2 \
    --eval_batch_size=1 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=4 \
    --num_train_epochs=5 \
    --coref_dir=data/wikievents/coref \
    --calibrate \
    --retrieval_augmented

rm -rf checkpoints/${CKPT}-calibrated