#!/usr/bin/env bash

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

}

# Kill ourself with SIGINT upon receiving SIGINT (i.e. CTRL + C)
trap '
  trap - INT # restore default INT handler
  kill -s INT "$$"
' INT

run "$@"
