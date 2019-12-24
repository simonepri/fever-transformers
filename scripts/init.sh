#!/usr/bin/env bash
PATH_DATA='data'

PATH_D_FEVER=$PATH_DATA'/fever'

# Create necessary directories.
mkdir -p $PATH_D_FEVER

# Download the FEVER dataset from the website of the FEVER share task into the data folder.
if [ ! -d $PATH_D_FEVER'/dataset' ]; then
  echo 'Downloading the FEVER dataset...'
  mkdir -p $PATH_D_FEVER'/dataset'
  wget -q --show-progress -O $PATH_D_FEVER'/dataset/train.jsonl' 'https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl'
  wget -q --show-progress -O $PATH_D_FEVER'/dataset/dev.jsonl' 'https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl'
  wget -q --show-progress -O $PATH_D_FEVER'/dataset/test.jsonl' 'https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl'
fi

# Download the FEVER pre-processed Wikipedia articles and unzip it into the data folder.
if [ ! -d $PATH_D_FEVER'/wikipedia' ]; then
  echo 'Download the FEVER pre-processed Wikipedia articles...'
  mkdir -p $PATH_D_FEVER'/wikipedia'
  if [ ! -f $PATH_D_FEVER'/wikipedia.zip' ]; then
    wget -q --show-progress -O $PATH_D_FEVER'/wikipedia.zip' 'https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip'
  fi
  unzip -j $PATH_D_FEVER'/wikipedia.zip' 'wiki-pages/*' -d $PATH_D_FEVER'/wikipedia'
  rm $PATH_D_FEVER'/wikipedia.zip'
fi
