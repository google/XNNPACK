#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## Unit tests #################################
tools/generate-vunary-test.py --spec test/u64-u32-vsqrtshift.yaml --output test/u64-u32-vsqrtshift.cc &

wait
