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
}

# Kill ourself with SIGINT upon receiving SIGINT (i.e. CTRL + C)
trap '
  trap - INT # restore default INT handler
  kill -s INT "$$"
' INT

run "$@"
