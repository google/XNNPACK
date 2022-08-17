#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## Unit tests #################################
tools/generate-filterbank-subtract-test.py --spec test/u32-filterbank-subtract.yaml --output test/u32-filterbank-subtract.cc &

wait
