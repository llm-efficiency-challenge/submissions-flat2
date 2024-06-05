#!/usr/bin/env bash

dirname() {
  echo "$1" | sed 's/git@github.com://' | sed 's/.git//'
}

while read -r line; do
  if [[ ! $line =~ ^\# ]]; then
    if [[ -d dirname $line ]]; then
      rm -rfv "./$dirname"
    fi
  fi
done < "repos.txt"
