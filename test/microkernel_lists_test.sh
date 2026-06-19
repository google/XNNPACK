#!/bin/bash
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Checks that the generated *_microkernel.bzl and *_microkernel.cmake files are
# up to date.

ERROR_MESSAGE='

Generated bzl/cmake files do not match the source files.
Please run the following command in your source directory to update them:

    ./tools/update-microkernels.py
'

function test_microkernel_lists {
  BASE_DIR=${RUNFILES_DIR}/xnnpack

  # Check generated .bzl files.
  for filename in $(ls ${BASE_DIR}/testdata/gen); do
    echo "Checking ${filename}..."
    cmp "${BASE_DIR}/testdata/gen/${filename}" \
        "${BASE_DIR}/gen/${filename}"
    if [ "$?" == "1" ]; then
      echo "${ERROR_MESSAGE}"
      exit 1
    fi
  done

  # Check generated .cmake files.
  for filename in $(ls ${BASE_DIR}/testdata/cmake/gen); do
    echo "Checking ${filename}..."
    cmp "${BASE_DIR}/testdata/cmake/gen/${filename}" \
        "${BASE_DIR}/cmake/gen/${filename}"
    if [ "$?" == "1" ]; then
      echo "${ERROR_MESSAGE}"
      exit 1
    fi
  done
}

test_microkernel_lists
