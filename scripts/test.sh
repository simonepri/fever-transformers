#!/usr/bin/env bash

PATH_SRC='src'
PATH_DATA='data'

PATH_S_PIPELINE=$PATH_SRC'/pipeline'

PATH_D_FEVER=$PATH_DATA'/fever'
PATH_D_PIPELINE=$PATH_DATA'/pipeline'
PATH_D_CACHE=$PATH_DATA'/cache'

PATH_D_NLTK=$PATH_D_CACHE'/nltk'
PATH_D_ALLEN=$PATH_D_CACHE'/allen'

PYTHON_ENVS='PYTHONPATH='$PATH_SRC' NLTK_DATA='$PATH_D_NLTK' ALLENNLP_CACHE_ROOT='$PATH_D_ALLEN

mkdir -p $PATH_D_PIPELINE'/sentence-retrieval/out'
mkdir -p $PATH_D_CACHE'/transformers'

env $PYTHON_ENVS pipenv run python3 $PATH_S_PIPELINE'/sentence-retrieval/sentence_retrieval_train.py' \
    --model_type bert \
    --model_name_or_path bert-base-cased \
    --task_name 'sentence_retrieval' \
    --do_train \
    --do_eval \
    --data_dir $PATH_D_PIPELINE'/sentence-retrieval' \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --output_dir $PATH_D_PIPELINE'/sentence-retrieval/out' \
    --cache_dir $PATH_D_CACHE'/transformers'
