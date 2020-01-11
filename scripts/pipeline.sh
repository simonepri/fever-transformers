#!/usr/bin/env bash

# Install the dependencies needed to run the python scripts.
function install_deps() {
  local fever_path=$1
  local pipeline_path=$2
  local cache_path=$3
  local force=$4

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

  local db_path="$pipeline_path/build-db"
  local wikipedia_path="$fever_path/wikipedia"

  if (( $force != 0 )); then
    rm -rf "$db_path"
  fi

  local db_file="$db_path/wikipedia.db"

  if [ ! -f "$db_file" ]; then
    mkdir -p "$db_path"

    echo '● Constructing an SQLite Database from the pre-processed Wikipedia articles...'
    env "PYTHONPATH=src" \
    pipenv run python3 'src/pipeline/build-db/run.py' \
        --data-path "$wikipedia_path" \
        --save-path "$db_file"
  fi
}


# Execute the document retrieval step
function document_retrieval() {
  local fever_path=$1
  local pipeline_path=$2
  local cache_path=$3
  local force=$4

  local doc_ret_path="$pipeline_path/document-retrieval"
  local db_path="$pipeline_path/build-db"
  local dataset_path="$fever_path/dataset"

  local allen_cache_path="$cache_path/allen"
  local nltk_cache_path="$cache_path/nltk"

  local db_file="$db_path/wikipedia.db"

  local max_pages_per_query=7

  if (( $force != 0 )); then
    rm -rf "$doc_ret_path"
  fi

  if [ ! -d "$doc_ret_path" ]; then
    mkdir -p "$doc_ret_path"
  fi

  for filetype in {dev,test,train}; do
    local dataset_file="$dataset_path/$filetype.jsonl"
    local doc_ret_file="$doc_ret_path/documents.predicted.$filetype.jsonl"

    if [ ! -f "$doc_ret_file" ]; then
      echo "● Retrieving the top documents for each claim in $dataset_file..."
      env "PYTHONPATH=src" "NLTK_DATA=$nltk_cache_path" "ALLENNLP_CACHE_ROOT=$allen_cache_path" \
      pipenv run python3 'src/pipeline/document-retrieval/run.py' \
          --db-file "$db_file" \
          --in-file "$dataset_file" \
          --out-file "$doc_ret_file" \
          --max-pages-per-query $max_pages_per_query
    fi
  done
}


# Execute the sentence retrieval step
function sentence_retrieval() {
  local fever_path=$1
  local pipeline_path=$2
  local cache_path=$3
  local force=$4
  local model_type=$5
  local model_name=$6

  local doc_ret_path="$pipeline_path/document-retrieval"
  local sent_ret_path="$pipeline_path/sentence-retrieval"
  local db_path="$pipeline_path/build-db"
  local dataset_path="$fever_path/dataset"

  local transformers_cache_path="$cache_path/transformers"

  local model_path="$sent_ret_path/model"
  local db_file="$db_path/wikipedia.db"

  local max_non_evidence_per_page=2
  local max_sentences_per_claim=5

  if (( $force != 0 )); then
    rm -rf "$sent_ret_path"
  fi

  if [ ! -d "$sent_ret_path" ]; then
    mkdir -p "$sent_ret_path"
  fi

  if [ ! -f "$model_path/config.json" ]; then
    local tuning_file="$sent_ret_path/sentences.golden.train.tsv"
    local doc_ret_file="$doc_ret_path/documents.predicted.train.jsonl"

    if [ ! -f "$tuning_file" ]; then
      echo "● Generating tuning examples from claims in $doc_ret_file..."
      env "PYTHONPATH=src" \
      pipenv run python3 'src/pipeline/sentence-retrieval/generate.py' \
          --db-file "$db_file" \
          --in-file "$doc_ret_file" \
          --out-file "$tuning_file" \
          --max-non-evidence-per-page $max_non_evidence_per_page
    fi

    echo '● Finetuning the transformer model...'
    env "PYTHONPATH=src" \
    pipenv run python3 'src/pipeline/sentence-retrieval/model.py' \
        --model_type "$model_type" \
        --model_name_or_path "$model_name" \
        --max_seq_length 128 \
        --task_name 'sentence_retrieval' \
        --output_dir "$model_path" \
        --cache_dir "$transformers_cache_path" \
        --do_train \
        --train_in_file "$tuning_file" \
        --per_gpu_train_batch_size=32 \
        --learning_rate 2e-5 \
        --num_train_epochs 2 \
        --logging_steps 1000 \
        --save_steps 10000
  fi

  if [ ! -f "$model_path/eval_results.txt" ]; then
    local eval_file="$sent_ret_path/sentences.golden.dev.tsv"
    local doc_ret_file="$doc_ret_path/documents.predicted.dev.jsonl"

    if [ ! -f "$eval_file" ]; then
      echo "● Generating evaluation examples from claims in $doc_ret_file..."
      env "PYTHONPATH=src" \
      pipenv run python3 'src/pipeline/sentence-retrieval/generate.py' \
          --db-file "$db_file" \
          --in-file "$doc_ret_file" \
          --out-file "$eval_file" \
          --max-non-evidence-per-page $max_non_evidence_per_page
    fi

    echo '● Evaluating the finetuned transformer model...'
    env "PYTHONPATH=src" \
    pipenv run python3 'src/pipeline/sentence-retrieval/model.py' \
        --model_type "$model_type" \
        --model_name_or_path "$model_name" \
        --max_seq_length 128 \
        --task_name 'sentence_retrieval' \
        --output_dir "$model_path" \
        --cache_dir "$transformers_cache_path" \
        --do_eval \
        --eval_in_file "$eval_file" \
        --per_gpu_eval_batch_size=32
  fi

  for filetype in {dev,test,train}; do
    local dataset_file="$dataset_path/$filetype.jsonl"
    local sent_ret_file="$sent_ret_path/sentences.predicted.$filetype.jsonl"
    local doc_ret_file="$doc_ret_path/documents.predicted.$filetype.jsonl"

    local sent_score_file="$sent_ret_path/sentences.scored.$filetype.tsv"
    local sent_file="$sent_ret_path/sentences.all.$filetype.tsv"
    local score_file="$sent_ret_path/sentences.score.$filetype.tsv"

    if [ ! -f "$sent_ret_file" ]; then
      if [ ! -f "$sent_score_file" ]; then
        if [ ! -f "$sent_file" ]; then
          echo "● Generating sentences to score from retrieved documents for claims in $doc_ret_file..."
          env "PYTHONPATH=src" \
          pipenv run python3 'src/pipeline/sentence-retrieval/generate.py' \
              --prediction \
              --db-file "$db_file" \
              --in-file "$doc_ret_file" \
              --out-file "$sent_file"
        fi

        if [ ! -f "$score_file" ]; then
          echo "● Scoring sentences from retrieved documents for claims in $sent_file..."
          env "PYTHONPATH=src" \
          pipenv run python3 'src/pipeline/sentence-retrieval/model.py' \
              --model_type "$model_type" \
              --model_name_or_path "$model_name" \
              --max_seq_length 128 \
              --task_name 'sentence_retrieval' \
              --output_dir "$model_path" \
              --cache_dir "$transformers_cache_path" \
              --do_predict \
              --predict_in_file "$sent_file" \
              --predict_out_file "$score_file" \
              --per_gpu_predict_batch_size=32
        fi

        echo "● Combining $sent_file and $score_file in $sent_score_file..."
        paste -d'\t' "$sent_file" "$score_file" > "$sent_score_file"
      fi

      echo "● Retrieving the top $max_sentences_per_claim evidence sentences for each claim in $dataset_file..."
      env "PYTHONPATH=src" \
      pipenv run python3 'src/pipeline/sentence-retrieval/run.py' \
          --scores-file "$sent_score_file" \
          --in-file "$dataset_file" \
          --out-file "$sent_ret_file" \
          --max-sentences-per-claim $max_sentences_per_claim
    fi
  done

  local sent_ret_dev_eval_file="$sent_ret_path/eval.dev.txt"
  local sent_ret_dev_file="$sent_ret_path/sentences.predicted.dev.jsonl"
  echo "● Evaluating predictions in $sent_ret_dev_file..."
  env "PYTHONPATH=src" \
  pipenv run python3 'src/pipeline/sentence-retrieval/evaluate.py' \
      --golden-file "$sent_ret_dev_file" \
      --evidence-file "$sent_ret_dev_file" \
  | tee "$sent_ret_dev_eval_file"
}


# Execute the claim verification step
function claim_verification() {
  local fever_path=$1
  local pipeline_path=$2
  local cache_path=$3
  local force=$4
  local model_type=$5
  local model_name=$6

  local doc_ret_path="$pipeline_path/document-retrieval"
  local sent_ret_path="$pipeline_path/sentence-retrieval"
  local claim_ver_path="$pipeline_path/claim-verification"
  local db_path="$pipeline_path/build-db"
  local dataset_path="$fever_path/dataset"

  local transformers_cache_path="$cache_path/transformers"

  local model_path="$claim_ver_path/model"
  local db_file="$db_path/wikipedia.db"

  if (( $force != 0 )); then
    rm -rf "$claim_ver_path"
  fi

  if [ ! -d "$claim_ver_path" ]; then
    mkdir -p "$claim_ver_path"
  fi

  if [ ! -f "$model_path/config.json" ]; then
    local tuning_file="$claim_ver_path/claims.golden.train.tsv"
    local sent_ret_file="$sent_ret_path/sentences.predicted.train.jsonl"

    if [ ! -f "$tuning_file" ]; then
      echo "● Generating tuning examples from claims in $sent_ret_file..."
      env "PYTHONPATH=src" \
      pipenv run python3 'src/pipeline/claim-verification/generate.py' \
          --db-file "$db_file" \
          --in-file "$sent_ret_file" \
          --out-file "$tuning_file"
    fi

    echo '● Finetuning the transformer model...'
    env "PYTHONPATH=src" \
    pipenv run python3 'src/pipeline/claim-verification/model.py' \
        --model_type "$model_type" \
        --model_name_or_path "$model_name" \
        --max_seq_length 128 \
        --task_name 'claim_verification' \
        --output_dir "$model_path" \
        --cache_dir "$transformers_cache_path" \
        --do_train \
        --train_in_file "$tuning_file" \
        --per_gpu_train_batch_size=32 \
        --learning_rate 2e-5 \
        --num_train_epochs 2 \
        --logging_steps 1000 \
        --save_steps 10000
  fi

  if [ ! -f "$model_path/eval_results.txt" ]; then
    local eval_file="$claim_ver_path/claims.golden.dev.tsv"
    local sent_ret_file="$sent_ret_path/sentences.predicted.dev.jsonl"

    if [ ! -f "$eval_file" ]; then
      echo "● Generating evaluation examples from claims in $sent_ret_file..."
      env "PYTHONPATH=src" \
      pipenv run python3 'src/pipeline/claim-verification/generate.py' \
          --db-file "$db_file" \
          --in-file "$sent_ret_file" \
          --out-file "$eval_file"
    fi

    echo '● Evaluating the finetuned transformer model...'
    env "PYTHONPATH=src" \
    pipenv run python3 'src/pipeline/claim-verification/model.py' \
        --model_type "$model_type" \
        --model_name_or_path "$model_name" \
        --max_seq_length 128 \
        --task_name 'claim_verification' \
        --output_dir "$model_path" \
        --cache_dir "$transformers_cache_path" \
        --do_eval \
        --eval_in_file "$eval_file" \
        --per_gpu_eval_batch_size=32
  fi

  for filetype in {dev,test,train}; do
    local dataset_file="$dataset_path/$filetype.jsonl"
    local sent_ret_file="$sent_ret_path/sentences.predicted.$filetype.jsonl"
    local claim_ver_file="$claim_ver_path/claims.predicted.$filetype.jsonl"

    local claim_label_file="$claim_ver_path/claims.labelled.$filetype.tsv"
    local claim_file="$claim_ver_path/claims.all.$filetype.tsv"
    local label_file="$claim_ver_path/claims.label.$filetype.tsv"

    if [ ! -f "$claim_ver_file" ]; then
      if [ ! -f "$claim_label_file" ]; then
        if [ ! -f "$claim_file" ]; then
          echo "● Generating claims to label from retrieved sentences for claims in $sent_ret_file..."
          env "PYTHONPATH=src" \
          pipenv run python3 'src/pipeline/claim-verification/generate.py' \
              --prediction \
              --db-file "$db_file" \
              --in-file "$sent_ret_file" \
              --out-file "$claim_file"
        fi

        if [ ! -f "$label_file" ]; then
          echo "● Labelling claims from retrieved sentences for claims in $claim_file..."
          env "PYTHONPATH=src" \
          pipenv run python3 'src/pipeline/claim-verification/model.py' \
              --model_type "$model_type" \
              --model_name_or_path "$model_name" \
              --max_seq_length 128 \
              --task_name 'claim_verification' \
              --output_dir "$model_path" \
              --cache_dir "$transformers_cache_path" \
              --do_predict \
              --predict_in_file "$claim_file" \
              --predict_out_file "$label_file" \
              --per_gpu_predict_batch_size=32
        fi

        echo "● Combining $claim_file and $label_file in $claim_label_file..."
        paste -d'\t' "$claim_file" "$label_file" > "$claim_label_file"
      fi

      echo "● Verifying each claim in $dataset_file..."
      env "PYTHONPATH=src" \
      pipenv run python3 'src/pipeline/claim-verification/run.py' \
          --labels-file "$claim_label_file" \
          --in-file "$dataset_file" \
          --out-file "$claim_ver_file"
    fi
  done

  local claim_ver_dev_eval_file="$claim_ver_path/eval.dev.txt"
  local claim_ver_dev_file="$claim_ver_path/claims.predicted.dev.jsonl"
  echo "● Evaluating predictions in $claim_ver_dev_file..."
  env "PYTHONPATH=src" \
  pipenv run python3 'src/pipeline/claim-verification/evaluate.py' \
      --golden-file "$claim_ver_dev_file" \
      --prediction-file "$claim_ver_dev_file" \
  | tee "$claim_ver_dev_eval_file"
}


# Put all the pieces together and generate the file to submit
function generate_submission() {
  local fever_path=$1
  local pipeline_path=$2
  local cache_path=$3
  local force=$4

  local dataset_path="$fever_path/dataset"
  local sub_path="$pipeline_path/generate-submission"
  local claim_ver_path="$pipeline_path/claim-verification"

  if (( $force != 0 )); then
    rm -rf "$sub_path"
  fi

  if [ ! -f "$sub_path" ]; then
    mkdir -p "$sub_path"

    for filetype in {dev,test,train}; do
      local dataset_file="$dataset_path/$filetype.jsonl"
      local claim_ver_file="$claim_ver_path/claims.predicted.$filetype.jsonl"
      local sub_file="$sub_path/submission.$filetype.jsonl"
      if [ ! -f "$sub_file" ]; then
        echo "● Building the submission files for claims in $dataset_file..."
        env "PYTHONPATH=src" \
        pipenv run python3 'src/pipeline/generate-submission/run.py' \
            --in-file "$claim_ver_file" \
            --out-file "$sub_file"
      fi
    done
  fi

  local gold_dev_file="$dataset_path/dev.jsonl"
  local sub_dev_file="$sub_path/submission.dev.jsonl"
  local sub_dev_eval_file="$sub_path/eval.dev.txt"
  echo "● Evaluating predictions in $sub_dev_file..."
  env "PYTHONPATH=src" \
  pipenv run python3 'src/pipeline/generate-submission/evaluate.py' \
      --golden-file "$gold_dev_file" \
      --prediction-file "$sub_dev_file" \
  | tee "$sub_dev_eval_file"
}


# Run the pipeline
function run() {
  # Read all the recognized flags and expected arguments.
  local -a pargs
  local flag_force=0
  local flag_data='data'
  local flag_model_type='bert'
  local flag_model_name='bert-base-cased'
  while [[ $1 != "" ]]; do
    case "$1" in
      --force ) flag_force=1; shift;;
      --data ) flag_data=$2; shift 2;;
      --model-type) flag_model_type=$2; shift 2;;
      --model-name) flag_model_name=$2; shift 2;;
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
    install_deps "$PATH_D_FEVER" "$PATH_D_PIPELINE" "$PATH_D_CACHE" $flag_force \
    > >(tee -a "$PATH_D_LOGS/install_deps.log") 2>&1
  fi
  if [ -z $parg_task ] || [[ $parg_task == "download_fever" ]]; then
    download_fever "$PATH_D_FEVER" "$PATH_D_PIPELINE" "$PATH_D_CACHE" $flag_force \
    > >(tee -a "$PATH_D_LOGS/download_fever.log") 2>&1
  fi
  if [ -z $parg_task ] || [[ $parg_task == "build_db" ]]; then
    build_db "$PATH_D_FEVER" "$PATH_D_PIPELINE" "$PATH_D_CACHE" $flag_force \
    > >(tee -a "$PATH_D_LOGS/build_db.log") 2>&1
  fi
  if [ -z $parg_task ] || [[ $parg_task == "document_retrieval" ]]; then
    document_retrieval "$PATH_D_FEVER" "$PATH_D_PIPELINE" "$PATH_D_CACHE" $flag_force \
    > >(tee -a "$PATH_D_LOGS/document_retrieval.log") 2>&1
  fi
  if [ -z $parg_task ] || [[ $parg_task == "sentence_retrieval" ]]; then
    sentence_retrieval "$PATH_D_FEVER" "$PATH_D_PIPELINE" "$PATH_D_CACHE" $flag_force "$flag_model_type" "$flag_model_name" \
    > >(tee -a "$PATH_D_LOGS/sentence_retrieval.log") 2>&1
  fi
  if [ -z $parg_task ] || [[ $parg_task == "claim_verification" ]]; then
    claim_verification "$PATH_D_FEVER" "$PATH_D_PIPELINE" "$PATH_D_CACHE" $flag_force "$flag_model_type" "$flag_model_name" \
    > >(tee -a "$PATH_D_LOGS/claim_verification.log") 2>&1
  fi
  if [ -z $parg_task ] || [[ $parg_task == "generate_submission" ]]; then
    generate_submission "$PATH_D_FEVER" "$PATH_D_PIPELINE" "$PATH_D_CACHE" $flag_force \
    > >(tee -a "$PATH_D_LOGS/generate_submission.log") 2>&1
  fi
}

# Kill ourself with SIGINT upon receiving SIGINT (i.e. CTRL + C)
trap '
  trap - INT # restore default INT handler
  kill -s INT "$$"
' INT

run "$@"
