#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

tools/generate-conv-hwc2chw-test.py --spec test/f16-conv-hwc2chw.yaml --output test/f16-conv-hwc2chw.cc &

wait
