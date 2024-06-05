#! /bin/bash

SUITE=$1
CONF=$2

helm-run --conf-paths $CONF   --suite $SUITE   -m 10000

helm-summarize --suite $SUITE

helm-server