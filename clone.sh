#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dirname() {
  echo "$1" | sed 's/git@github.com://' | sed 's/.git//'
}

print_line() {
  printf "\n--------------------------------------------------\n" 
}

clone_repo() {
  local repo="$1"
  local fail_log="$2"
  local dir="$(dirname $repo)"

  print_line
  echo git clone --recurse-submodules "$repo" "$dir"

  if ! git clone --recurse-submodules "$repo" "$dir"; then
    echo "$repo" >> "$fail_log"
  fi

  print_line
}

main() {
  if [[ $# != 2 ]]; then
    echo "Usage $0: repo-list output_dir"
    exit 1
  fi

  local repos="$(realpath $1)"
  local output_dir="$2"
  local fail_log="${repos%%.*}-failed_to_clone.log"

  mkdir -p "$output_dir"
  cd "$output_dir"

  while read -r line; do
    if [[ ! $line =~ ^\# ]]; then
      clone_repo "$line" "$fail_log"
    fi
  done < "$repos"

  if [[ -e "$fail_log" ]]; then
    echo "There were repos that failed to clone. Check $fail_log"
  fi
}

main "$@"
