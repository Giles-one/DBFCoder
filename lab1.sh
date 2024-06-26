#!/usr/bin/env bash

if [[ $# -ne 1 ]]; then
  echo Usage: lab.sh path/to/model
  exit 1
fi

MODEL=$1

if [[ ! -f $MODEL ]]; then
  echo $MODEL not exits.
  exit 1
fi

#exit 0

python lab1.py \
  --model $MODEL \
  --anchorCompiler clang-4.0 \
  --anchorOptimization O0 \
  --positiveCompiler clang-4.0 \
  --positiveOptimization O1

python lab1.py \
  --model $MODEL \
  --anchorCompiler clang-4.0 \
  --anchorOptimization O0 \
  --positiveCompiler clang-4.0 \
  --positiveOptimization O2

python lab1.py \
  --model $MODEL \
  --anchorCompiler clang-4.0 \
  --anchorOptimization O0 \
  --positiveCompiler clang-4.0 \
  --positiveOptimization O3

python lab1.py \
  --model $MODEL \
  --anchorCompiler clang-4.0 \
  --anchorOptimization O1 \
  --positiveCompiler clang-4.0 \
  --positiveOptimization O2

python lab1.py \
  --model $MODEL \
  --anchorCompiler clang-4.0 \
  --anchorOptimization O1 \
  --positiveCompiler clang-4.0 \
  --positiveOptimization O3

python lab1.py \
  --model $MODEL \
  --anchorCompiler clang-4.0 \
  --anchorOptimization O2 \
  --positiveCompiler clang-4.0 \
  --positiveOptimization O3

python lab1.py \
  --model $MODEL \
  --anchorCompiler clang-4.0 \
  --anchorOptimization O0 \
  --positiveCompiler clang-obfus-sub \
  --positiveOptimization O0

python lab1.py \
  --model $MODEL \
  --anchorCompiler clang-4.0 \
  --anchorOptimization O0 \
  --positiveCompiler clang-obfus-bcf \
  --positiveOptimization O0

python lab1.py \
  --model $MODEL \
  --anchorCompiler clang-4.0 \
  --anchorOptimization O0 \
  --positiveCompiler clang-obfus-fla \
  --positiveOptimization O0

python lab1.py \
  --model $MODEL \
  --anchorCompiler clang-4.0 \
  --anchorOptimization O0 \
  --positiveCompiler clang-obfus-all \
  --positiveOptimization O0
