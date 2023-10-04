#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## Unit tests #################################
tools/generate-fill-test.py --spec test/xx-fill.yaml --output test/xx-fill.cc &

wait
