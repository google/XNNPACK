#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

############################### AArch64 assembly ##############################
### LD64 micro-kernels
tools/xngen src/f32-gemm/4x8-aarch64-neonfma-ld64.S.in -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x8-minmax-asm-aarch64-neonfma-ld64.S &

################################## Unit tests #################################
#tools/generate-gemm-test.py --spec test/f32-qc8w-gemm-minmax.yaml --output test/f32-qc8w-gemm-minmax.cc &

wait
