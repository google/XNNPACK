#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

### Generate tests.
scripts/generate-tests.sh &

### Generate benchmarks.
scripts/generate-benchmarks.sh &

wait
