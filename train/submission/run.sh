#!/usr/bin/env bash
# The evaluator runs `bash run.sh` with cwd = this directory.
# -u is mandatory: the stdio protocol needs unbuffered output.
exec python3 -u main.py
