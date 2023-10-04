#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## Unit tests #################################
tools/generate-reduce-test.py --tester ReduceMicrokernelTester --spec test/u8-rmax.yaml --output test/u8-rmax.cc &

wait
