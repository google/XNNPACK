#!/bin/bash
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run a few configurations of schedule_bench to make sure they don't crash.
set -e

# The path to the binary from the first argument.
BINARY="$1"

# Run a simple configuration.
"${BINARY}" dot_fp32_1x128x1_1x1x1 64x128x256

# Run a configuration with loop specifiers.
"${BINARY}" dot_fp32_1x128x1_1x1x1 100x300x200 m16 n128 k32
