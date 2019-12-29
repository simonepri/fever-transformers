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

if ! hash pipenv; then
  echo 'You need to install pipenv to run this.'
  echo 'Check it out at https://github.com/pypa/pipenv.'
  exit 1
fi

# Install the dependencies needed to run the python scripts.
echo 'Install dependencies...'
pipenv install --skip-lock

# Create necessary directories.
mkdir -p $PATH_D_NLTK
mkdir -p $PATH_D_ALLEN

# Construct an SQLite Database from the pre-processed Wikipedia articles into the data folder.
if [ ! -f $PATH_D_PIPELINE'/build-db/wikipedia.db' ]; then
  echo 'Constructing an SQLite Database from the pre-processed Wikipedia articles...'
  mkdir -p $PATH_D_PIPELINE'/build-db'
  env $PYTHON_ENVS pipenv run python3 $PATH_S_PIPELINE'/build-db/build_db.py' $PATH_D_FEVER'/wikipedia' $PATH_D_PIPELINE'/build-db/wikipedia.db'

  if [ ! "$(ls -A $PATH_D_PIPELINE'/build-db')" ]; then
    rm -r $PATH_D_PIPELINE'/build-db'
  fi
fi

# Execute the document retrieval step
if [ ! -d $PATH_D_PIPELINE'/document-retrieval' ]; then
  K=7
  echo 'Retrieving the top '$K' documents for each claim...'
  mkdir -p $PATH_D_PIPELINE'/document-retrieval'
  for file in $PATH_D_FEVER'/dataset/'{train,dev,test}'.jsonl'; do
    filename=${file##*/}
    echo 'Processing claims in '$file'...'
    env $PYTHON_ENVS pipenv run python3 $PATH_S_PIPELINE'/document-retrieval/document_retrieval.py' --db-file $PATH_D_PIPELINE'/build-db/wikipedia.db' --in-file $PATH_D_FEVER'/dataset/'$filename --out-file $PATH_D_PIPELINE'/document-retrieval/'$filename --max-results=$K
  done

  if [ ! "$(ls -A $PATH_D_PIPELINE'/document-retrieval')" ]; then
    rm -r $PATH_D_PIPELINE'/document-retrieval'
  fi
fi

# Execute the sentence retrieval step
if [ ! -d $PATH_D_PIPELINE'/sentence-retrieval' ]; then
  K=5
  echo 'Generating the training set for the sentence retrieval model...'
  mkdir -p $PATH_D_PIPELINE'/sentence-retrieval'
  for file in $PATH_D_FEVER'/dataset/'{train,dev}'.jsonl'; do
    filename=${file##*/}
    echo 'Processing claims in '$file'...'
    env $PYTHON_ENVS pipenv run python3 $PATH_S_PIPELINE'/sentence-retrieval/sentence_retrieval_generate.py' --db-file $PATH_D_PIPELINE'/build-db/wikipedia.db' --in-file $PATH_D_PIPELINE'/document-retrieval/'$filename --out-file $PATH_D_PIPELINE'/sentence-retrieval/'$filename'.tsv' --max-neg-evidences-per-page $K
  done

  if [ ! "$(ls -A $PATH_D_PIPELINE'/sentence-retrieval')" ]; then
    rm -r $PATH_D_PIPELINE'/sentence-retrieval'
  fi
fi
