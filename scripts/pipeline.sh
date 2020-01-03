#!/usr/bin/env bash

# Install the dependencies needed to run the python scripts.
function install_deps() {
  local fever_path=$1
  local pipeline_path=$2
  local cache_path=$3
  local force=$4
  local download=$5

  local venv_path=".venv"

  if (( $force != 0 )); then
    rm -rf "$venv_path"
  fi

  if [ ! -d "$venv_path" ]; then
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
  local download=$5

  local dataset_path="$fever_path/dataset"
  local wikipedia_path="$fever_path/wikipedia"

  if (( $force != 0 )); then
    rm -rf "$dataset_path"
    rm -rf "$wikipedia_path"
  fi

  if [ ! -d "$dataset_path" ]; then
    mkdir -p "$dataset_path"

    echo '● Downloading the FEVER dataset...'
    wget -q --show-progress --progress=bar:force -O "$dataset_path/train.jsonl" \
    'https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl'
    wget -q --show-progress --progress=bar:force -O "$dataset_path/dev.jsonl" \
    'https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl'
    wget -q --show-progress --progress=bar:force -O "$dataset_path/test.jsonl" \
    'https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl'
  fi

  if [ ! -d "$wikipedia_path" ]; then
    mkdir -p "$wikipedia_path"

    echo '● Downloading the FEVER pre-processed Wikipedia articles...'
    wget -q --show-progress --progress=bar:force -O "$fever_path/wikipedia.zip" \
    'https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip'
    unzip -o -j "$fever_path/wikipedia.zip" 'wiki-pages/*' -d "$wikipedia_path"
    rm "$fever_path/wikipedia.zip"
  fi
}


# Construct an SQLite Database from the pre-processed Wikipedia articles.
function build_db() {
  local fever_path=$1
  local pipeline_path=$2
  local cache_path=$3
  local force=$4
  local download=$5

  local db_path="$pipeline_path/build-db"
  local wikipedia_path="$fever_path/wikipedia"

  if (( $force != 0 )); then
    rm -rf "$db_path"
  fi

  local db_file="$db_path/wikipedia.db"

  if [ ! -f "$db_file" ]; then
    mkdir -p "$db_path"

    if (( $download != 0)); then
      local zip_file="$pipeline_path/build-db.zip"

      echo '● Downloading the output of the build db step instead of computing it...'
      wget -q --show-progress --progress=bar:force -O "$zip_file" \
      'https://github.com/simonepri/fever-transformers/releases/download/0.0.1/build-db.zip'
      if [ $? -eq 0 ]; then
        unzip -o -j "$zip_file" -d "$db_path"
        rm "$zip_file"
        return
      else
        rm "$zip_file"
        echo 'Download failed...'
      fi
    fi

    echo '● Constructing an SQLite Database from the pre-processed Wikipedia articles...'
    env "PYTHONPATH=src" \
    pipenv run python3 'src/pipeline/build-db/run.py' \
        "$wikipedia_path" \
        "$db_file"
  fi
}


# Execute the document retrieval step
function document_retrieval() {
  local fever_path=$1
  local pipeline_path=$2
  local cache_path=$3
  local force=$4
  local download=$5

  local doc_ret_path="$pipeline_path/document-retrieval"
  local db_path="$pipeline_path/build-db"
  local dataset_path="$fever_path/dataset"

  local allen_cache_path="$cache_path/allen"
  local nltk_cache_path="$cache_path/nltk"

  local db_file="$db_path/wikipedia.db"

  local max_docs_per_claim=7

  if (( $force != 0 )); then
    rm -rf "$doc_ret_path"
  fi

  if [ ! -d "$doc_ret_path" ]; then
    mkdir -p "$doc_ret_path"

    if (( $download != 0)); then
      local zip_file="$pipeline_path/document-retrieval.zip"

      echo '● Downloading the output of the document retrieval step instead of computing it...'
      wget -q --show-progress --progress=bar:force -O "$zip_file" \
      'https://github.com/simonepri/fever-transformers/releases/download/0.0.1/document-retrieval.zip'
      if [ $? -eq 0 ]; then
        unzip -o -j "$zip_file" -d "$doc_ret_path"
        rm "$zip_file"
        return
      else
        rm "$zip_file"
        echo 'Download failed...'
      fi
    fi
  fi

  for filetype in {dev,test,train}; do
    local dataset_file="$dataset_path/$filetype.jsonl"
    local doc_ret_file="$doc_ret_path/documents.predicted.$filetype.jsonl"

    if [ ! -f "$doc_ret_file" ]; then
      echo "● Retrieving the top $max_docs_per_claim documents for each claim in $dataset_file..."
      env "PYTHONPATH=src NLTK_DATA=$nltk_cache_path ALLENNLP_CACHE_ROOT=$allen_cache_path" \
      pipenv run python3 'src/pipeline/document-retrieval/run.py' \
          --db-file "$db_file" \
          --in-file "$dataset_file" \
          --out-file "$doc_ret_file" \
          --max-docs-per-claim $max_docs_per_claim
    fi
  done
}


# Run the pipeline
function run() {
  # Read all the recognized flags and expected arguments.
  local -a pargs
  local flag_force=0
  local flag_download=0
  local flag_data='data'
  while [[ $1 != "" ]]; do
    case "$1" in
      --force ) flag_force=1; shift;;
      --download ) flag_download=1; shift;;
      --data ) flag_data=$2; shift 2;;
      -* ) shift;;
      * ) pargs+=("$1"); shift;;
    esac
  done
  local parg_task=${pargs[0]}
  unset pargs

  # Create the necessary folders
  local PATH_DATA="$flag_data"
  local PATH_D_FEVER="$PATH_DATA/fever"
  local PATH_D_PIPELINE="$PATH_DATA/pipeline"
  local PATH_D_CACHE="$PATH_DATA/cache"
  local PATH_D_LOGS="$PATH_DATA/logs"
  mkdir -p "$PATH_D_FEVER"
  mkdir -p "$PATH_D_PIPELINE"
  mkdir -p "$PATH_D_CACHE"
  mkdir -p "$PATH_D_LOGS"

  # Execute the tasks
  if [ -z $parg_task ] || [[ $parg_task == "install_deps" ]]; then
    install_deps "$PATH_D_FEVER" "$PATH_D_PIPELINE" "$PATH_D_CACHE" $flag_force $flag_download > >(tee -a "$PATH_D_LOGS/install_deps.log") 2>&1
  fi
  if [ -z $parg_task ] || [[ $parg_task == "download_fever" ]]; then
    download_fever "$PATH_D_FEVER" "$PATH_D_PIPELINE" "$PATH_D_CACHE" $flag_force $flag_download > >(tee -a "$PATH_D_LOGS/download_fever.log") 2>&1
  fi
  if [ -z $parg_task ] || [[ $parg_task == "build_db" ]]; then
    build_db "$PATH_D_FEVER" "$PATH_D_PIPELINE" "$PATH_D_CACHE" $flag_force $flag_download > >(tee -a "$PATH_D_LOGS/build_db.log") 2>&1
  fi
  if [ -z $parg_task ] || [[ $parg_task == "document_retrieval" ]]; then
    document_retrieval "$PATH_D_FEVER" "$PATH_D_PIPELINE" "$PATH_D_CACHE" $flag_force $flag_download > >(tee -a "$PATH_D_LOGS/document_retrieval.log") 2>&1
  fi
}

# Kill ourself with SIGINT upon receiving SIGINT (i.e. CTRL + C)
trap '
  trap - INT # restore default INT handler
  kill -s INT "$$"
' INT

run "$@"
