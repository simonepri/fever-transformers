#!/usr/bin/env bash


GITHUB_PAGE="https://github.com/simonepri/fever-transformers"
DOWNLOAD_RELEASE="0.0.1"

# Download preprocessed output from GitHub release
function download_from_release() {
  local pipeline_path=$1
  local download_type=$2
  local download_name=$3

  download_url="$GITHUB_PAGE/releases/download/$DOWNLOAD_RELEASE/$download_type.$download_name.zip"
  zip_file="$pipeline_path/$download_type.$download_name.zip"

  echo "● Downloading $download_url ..."
  wget -q --show-progress --progress=bar:force -O "$zip_file" "$download_url"
  if [ $? -eq 0 ]; then
    unzip -o "$zip_file" -d "$pipeline_path"
    rm "$zip_file"
    return
  else
    rm "$zip_file"
    echo 'Download failed...'
  fi
}


# Run the release script
function run() {
  # Read all the recognized flags and expected arguments.
  local -a pargs
  local flag_download_type=''
  local flag_download_name=''
  local flag_data='data'
  while [[ $1 != "" ]]; do
    case "$1" in
      --data ) flag_data=$2; shift 2;;
      --download ) flag_download_type=$2; flag_download_name=$3; shift 3;;
      -* ) shift;;
      * ) pargs+=("$1"); shift;;
    esac
  done
  unset pargs

  # Create the necessary folders
  local PATH_DATA="$flag_data"
  local PATH_D_PIPELINE="$PATH_DATA/pipeline"
  local PATH_D_LOGS="$PATH_DATA/logs"
  mkdir -p "$PATH_D_PIPELINE"
  mkdir -p "$PATH_D_LOGS"

  # Execute the tasks
  if [[ $flag_download_type == "model" ]] || [[ $flag_download_type == "output" ]]; then
    download_from_release "$PATH_D_PIPELINE" "$flag_download_type" "$flag_download_name" \
    > >(tee -a "$PATH_D_LOGS/release.log") 2>&1
  fi
}

# Kill ourself with SIGINT upon receiving SIGINT (i.e. CTRL + C)
trap '
  trap - INT # restore default INT handler
  kill -s INT "$$"
' INT

run "$@"
