#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## Unit tests #################################
tools/generate-gavgpool-cw-test.py --spec test/f16-gavgpool-cw.yaml --output test/f16-gavgpool-cw.cc &

wait
