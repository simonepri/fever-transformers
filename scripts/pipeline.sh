#!/usr/bin/env bash


# Install the dependencies needed to run the python scripts.
function install_deps() {
  local fever_path=$1
  local pipeline_path=$2
  local cache_path=$3
  local force=$4

  local venv_path=".venv"

  if [ ! -d "$venv_path" ] || [ ! -z "$force" ]; then
    rm -rf "$venv_path"
    mkdir -p "$venv_path"

    if ! hash pipenv; then
      echo 'You need to install pipenv to run this.'
      echo 'Check it out at https://github.com/pypa/pipenv .'
      exit 1
    fi
    echo 'Install dependencies...'
    env "PIPENV_VENV_IN_PROJECT=1" \
    pipenv install --skip-lock
  fi
}


# Download the FEVER dataset and the pre-processed Wikipedia articles from the
# website of the FEVER share task.
function download_fever() {
  local fever_path=$1
  local pipeline_path=$2
  local cache_path=$3
  local force=$4

  local dataset_path="$fever_path/dataset"
  local wikipedia_path="$fever_path/wikipedia"

  if [ ! -d "$dataset_path" ] || [ ! -z "$force" ]; then
    rm -rf "$dataset_path"
    mkdir -p "$dataset_path"

    echo 'Downloading the FEVER dataset...'
    wget -q --show-progress -O "$dataset_path/train.jsonl" 'https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl'
    wget -q --show-progress -O "$dataset_path/dev.jsonl" 'https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl'
    wget -q --show-progress -O "$dataset_path/test.jsonl" 'https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl'
  fi

  if [ ! -d "$wikipedia_path" ] || [ ! -z "$force" ]; then
    rm -rf "$wikipedia_path"
    mkdir -p "$wikipedia_path"

    echo 'Download the FEVER pre-processed Wikipedia articles...'
    wget -q --show-progress -O "$fever_path/wikipedia.zip" 'https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip'
    unzip -j "$fever_path/wikipedia.zip" 'wiki-pages/*' -d "$wikipedia_path"
    rm "$fever_path/wikipedia.zip"
  fi
}


# Download the output of the build db step instead of computing it
function download_build_db() {
  local fever_path=$1
  local pipeline_path=$2
  local cache_path=$3
  local force=$4

  local db_path="$pipeline_path/build-db"

  if [ ! -d "$db_path" ] || [ ! -z "$force" ]; then
    rm -rf "$db_path"
    mkdir -p "$db_path"

    echo 'Downloading the output of the build db step instead of computing it...'
    wget -q --show-progress -O "$pipeline_path/build-db.zip" 'https://s3-eu-west-1.amazonaws.com/fever.public/build-db.zip'
    unzip -j "$pipeline_path/build-db.zip" -d "$db_path"
    rm "$pipeline_path/build-db.zip"
  fi
}


# Construct an SQLite Database from the pre-processed Wikipedia articles.
function pipeline_build_db() {
  local fever_path=$1
  local pipeline_path=$2
  local cache_path=$3
  local force=$4

  local db_path="$pipeline_path/build-db"
  local wikipedia_path="$fever_path/wikipedia"

  if [ ! -d "$db_path" ] || [ ! -z "$force" ]; then
    rm -rf "$db_path"
    mkdir -p "$db_path"

    echo 'Constructing an SQLite Database from the pre-processed Wikipedia articles...'
    env "PYTHONPATH=src" \
    pipenv run python3 'src/pipeline/build-db/build_db.py' \
        "$wikipedia_path" \
        "$db_path/wikipedia.db"
  fi
}


# Download the output of the document retrieval step instead of computing it
function download_document_retrieval() {
  local fever_path=$1
  local pipeline_path=$2
  local cache_path=$3
  local force=$4

  local doc_ret_path="$pipeline_path/document-retrieval"

  if [ ! -d "$doc_ret_path" ] || [ ! -z "$force" ]; then
    rm -rf "$doc_ret_path"
    mkdir -p "$doc_ret_path"

    echo 'Downloading the output of the document retrieval step instead of computing it...'
    wget -q --show-progress -O "$pipeline_path/document-retrieval.zip" 'https://s3-eu-west-1.amazonaws.com/fever.public/document-retrieval.zip'
    unzip -j "$pipeline_path/document-retrieval.zip" -d "$db_path"
    rm "$pipeline_path/document-retrieval.zip"
  fi
}


# Execute the document retrieval step
function pipeline_document_retrieval() {
  local fever_path=$1
  local pipeline_path=$2
  local cache_path=$3
  local force=$4

  local doc_ret_path="$pipeline_path/document-retrieval"
  local db_path="$pipeline_path/build-db"
  local dataset_path="$fever_path/dataset"

  if [ ! -d "$doc_ret_path" ] || [ ! -z "$force" ]; then
    rm -rf "$doc_ret_path"
    mkdir -p "$doc_ret_path"

    K=7
    echo "Retrieving the top $K documents for each claim..."
    for file in "$dataset_path/"{train,dev,test}'.jsonl'; do
      filename=${file##*/}
      echo "Processing claims in $file..."
      env "PYTHONPATH=src NLTK_DATA=$cache_path/nltk ALLENNLP_CACHE_ROOT=$cache_path/allen" \
      pipenv run python3 'src/pipeline/document-retrieval/document_retrieval.py' \
          --db-file "$db_path/wikipedia.db" \
          --in-file "$dataset_path/$filename" \
          --out-file "$doc_ret_path/$filename" \
          --max-results $K
    done
  fi
}


# Download the output of the sentence retrieval step instead of computing it
function download_sentence_retrieval() {
  local fever_path=$1
  local pipeline_path=$2
  local cache_path=$3
  local force=$4

  local doc_sent_path="$pipeline_path/sentence-retrieval"

  if [ ! -d "$doc_sent_path" ] || [ ! -z "$force" ]; then
    rm -rf "$doc_sent_path"
    mkdir -p "$doc_sent_path"

    echo 'Downloading the output of the sentence retrieval step instead of computing it...'
    wget -q --show-progress -O "$pipeline_path/sentence-retrieval.zip" 'https://s3-eu-west-1.amazonaws.com/fever.public/sentence-retrieval.zip'
    unzip -j "$pipeline_path/sentence-retrieval.zip" -d "$db_path"
    rm "$pipeline_path/sentence-retrieval.zip"
  fi
}


# Execute the sentence retrieval step
function pipeline_sentence_retrieval() {
  local fever_path=$1
  local pipeline_path=$2
  local cache_path=$3
  local force=$4

  local doc_ret_path="$pipeline_path/document-retrieval"
  local doc_sent_path="$pipeline_path/sentence-retrieval"
  local db_path="$pipeline_path/build-db"

  if [ ! -d "$doc_sent_path" ] || [ ! -z "$force" ]; then
    rm -rf "$doc_sent_path"
    mkdir -p "$doc_sent_path"

    K=5
    echo 'Generating the training set for the sentence retrieval model...'
    for file in "$doc_ret_path/"{train,dev}'.jsonl'; do
      filename=${file##*/}
      echo "Processing claims in $file..."
      env "PYTHONPATH=src" \
      pipenv run python3 'src/pipeline/sentence-retrieval/sentence_retrieval_generate.py' \
          --db-file "$db_path/wikipedia.db" \
          --in-file "$doc_ret_path/$filename" \
          --out-file "$doc_sent_path/$filename.tsv" \
          --max-neg-evidences-per-page $K
    done

    echo 'Finetuning the transformer model...'
    env "PYTHONPATH=src" \
    pipenv run python3 'src/pipeline/sentence-retrieval/sentence_retrieval_train.py' \
        --model_type bert \
        --model_name_or_path bert-base-cased \
        --task_name 'sentence_retrieval' \
        --do_train \
        --do_eval \
        --data_dir "$doc_sent_path" \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1 \
        --output_dir "$doc_sent_path/model" \
        --cache_dir "$cache_path/transformers"

    # TODO
  fi
}


# Download the output of the claim verification step instead of computing it
function download_claim_verification() {
  local fever_path=$1
  local pipeline_path=$2
  local cache_path=$3
  local force=$4

  local doc_claim_path="$pipeline_path/claim-verification"

  if [ ! -d "$doc_claim_path" ] || [ ! -z "$force" ]; then
    rm -rf "$doc_claim_path"
    mkdir -p "$doc_claim_path"

    echo 'Downloading the output of the claim verification step instead of computing it...'
    wget -q --show-progress -O "$pipeline_path/claim-verification.zip" 'https://s3-eu-west-1.amazonaws.com/fever.public/claim-verification.zip'
    unzip -j "$pipeline_path/claim-verification.zip" -d "$db_path"
    rm "$pipeline_path/claim-verification.zip"
  fi
}


# Execute the claim verification step
function pipeline_claim_verification() {
  local fever_path=$1
  local pipeline_path=$2
  local cache_path=$3
  local force=$4

  # TODO
}


# Run the pipeline
function run() {
  # Read all the recognized flags and expected arguments.
  local -a pargs
  while [[ $1 != "" ]]; do
    case "$1" in
      -force ) flag_force=1; shift;;
      -quick ) flag_quick=1; shift;;
      * ) pargs+=("$1"); shift;;
    esac
  done
  local parg_task="${pargs[0]:-'all'}"
  unset pargs

  # Create the necessary folders
  local PATH_DATA='data'
  local PATH_D_FEVER="$PATH_DATA/fever"
  local PATH_D_PIPELINE="$PATH_DATA/pipeline"
  local PATH_D_CACHE="$PATH_DATA/cache"
  mkdir -p $PATH_D_FEVER
  mkdir -p $PATH_D_PIPELINE
  mkdir -p $PATH_D_CACHE

  # Execute the tasks
  if [[ $parg_task == "all" ]] || [[ $parg_task == "install_deps" ]]; then
    install_deps "$PATH_D_FEVER" "$PATH_D_PIPELINE" "$PATH_D_CACHE" $flag_force
  fi
  if [[ $parg_task == "all" ]] || [[ $parg_task == "download_fever" ]]; then
    download_fever "$PATH_D_FEVER" "$PATH_D_PIPELINE" "$PATH_D_CACHE" $flag_force
  fi

  if [ ! -z $flag_quick ]; then
    download_build_db "$PATH_D_FEVER" "$PATH_D_PIPELINE" "$PATH_D_CACHE" $flag_force
    download_document_retrieval "$PATH_D_FEVER" "$PATH_D_PIPELINE" "$PATH_D_CACHE" $flag_force
    download_sentence_retrieval "$PATH_D_FEVER" "$PATH_D_PIPELINE" "$PATH_D_CACHE" $flag_force
    download_claim_verification "$PATH_D_FEVER" "$PATH_D_PIPELINE" "$PATH_D_CACHE" $flag_force
  else
    if [[ $parg_task == "all" ]] || [[ $parg_task == "pipeline_build_db" ]]; then
      pipeline_build_db "$PATH_D_FEVER" "$PATH_D_PIPELINE" "$PATH_D_CACHE" $flag_force
    fi
    if [[ $parg_task == "all" ]] || [[ $parg_task == "pipeline_document_retrieval" ]]; then
      pipeline_document_retrieval "$PATH_D_FEVER" "$PATH_D_PIPELINE" "$PATH_D_CACHE" $flag_force
    fi
    if [[ $parg_task == "all" ]] || [[ $parg_task == "pipeline_sentence_retrieval" ]]; then
      pipeline_sentence_retrieval "$PATH_D_FEVER" "$PATH_D_PIPELINE" "$PATH_D_CACHE" $flag_force
    fi
    if [[ $parg_task == "all" ]] || [[ $parg_task == "pipeline_claim_verification" ]]; then
      pipeline_claim_verification "$PATH_D_FEVER" "$PATH_D_PIPELINE" "$PATH_D_CACHE" $flag_force
    fi
  fi
}


run "$@"
