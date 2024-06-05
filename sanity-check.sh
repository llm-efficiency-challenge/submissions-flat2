#!/usr/bin/env bash

dirname() {
  echo "$1" | sed 's/git@github.com://' | sed 's/.git//'
}

while read -r line; do
  if [[ ! $line =~ ^\# ]]; then
    if [[ ! -d dirname $line ]]; then
      echo "$line should exist as $(dirname $line) but is missing"
    fi
  fi
done < "repos.txt"
