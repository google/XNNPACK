#!/bin/sh
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

mkdir -p src/qs8-qc2w-packw/gen

### QD8_QC2W and QS8_QC2W Scalar packing (C8 and C4)
tools/xngen src/x8-packw/kr-qc2w-scalar.c.in -D NR=2  -D KR=8 -D DATATYPE=QD8_QC2W -D IZP=0   -o src/qs8-qc2w-packw/gen/qd8-qc2w-packw-x2c8-gemm-goi-scalar.c &
tools/xngen src/x8-packw/kr-qc2w-scalar.c.in -D NR=4  -D KR=8 -D DATATYPE=QD8_QC2W -D IZP=0   -o src/qs8-qc2w-packw/gen/qd8-qc2w-packw-x4c8-gemm-goi-scalar.c &
tools/xngen src/x8-packw/kr-qc2w-scalar.c.in -D NR=8  -D KR=8 -D DATATYPE=QD8_QC2W -D IZP=0   -o src/qs8-qc2w-packw/gen/qd8-qc2w-packw-x8c8-gemm-goi-scalar.c &
tools/xngen src/x8-packw/kr-qc2w-scalar.c.in -D NR=16 -D KR=8 -D DATATYPE=QD8_QC2W -D IZP=0   -o src/qs8-qc2w-packw/gen/qd8-qc2w-packw-x16c8-gemm-goi-scalar.c &

tools/xngen src/x8-packw/kr-qc2w-scalar.c.in -D NR=2  -D KR=4 -D DATATYPE=QD8_QC2W -D IZP=0   -o src/qs8-qc2w-packw/gen/qd8-qc2w-packw-x2c4-gemm-goi-scalar.c &
tools/xngen src/x8-packw/kr-qc2w-scalar.c.in -D NR=4  -D KR=4 -D DATATYPE=QD8_QC2W -D IZP=0   -o src/qs8-qc2w-packw/gen/qd8-qc2w-packw-x4c4-gemm-goi-scalar.c &
tools/xngen src/x8-packw/kr-qc2w-scalar.c.in -D NR=8  -D KR=4 -D DATATYPE=QD8_QC2W -D IZP=0   -o src/qs8-qc2w-packw/gen/qd8-qc2w-packw-x8c4-gemm-goi-scalar.c &
tools/xngen src/x8-packw/kr-qc2w-scalar.c.in -D NR=16 -D KR=4 -D DATATYPE=QD8_QC2W -D IZP=0   -o src/qs8-qc2w-packw/gen/qd8-qc2w-packw-x16c4-gemm-goi-scalar.c &

tools/xngen src/x8-packw/kr-qc2w-scalar.c.in -D NR=2  -D KR=8 -D DATATYPE=QS2 -D IZP=128 -o src/qs8-qc2w-packw/gen/qs8-to-qu8-qc2w-packw-x2c8-gemm-goi-scalar.c &
tools/xngen src/x8-packw/kr-qc2w-scalar.c.in -D NR=4  -D KR=8 -D DATATYPE=QS2 -D IZP=128 -o src/qs8-qc2w-packw/gen/qs8-to-qu8-qc2w-packw-x4c8-gemm-goi-scalar.c &
tools/xngen src/x8-packw/kr-qc2w-scalar.c.in -D NR=8  -D KR=8 -D DATATYPE=QS2 -D IZP=128 -o src/qs8-qc2w-packw/gen/qs8-to-qu8-qc2w-packw-x8c8-gemm-goi-scalar.c &
tools/xngen src/x8-packw/kr-qc2w-scalar.c.in -D NR=16 -D KR=8 -D DATATYPE=QS2 -D IZP=128 -o src/qs8-qc2w-packw/gen/qs8-to-qu8-qc2w-packw-x16c8-gemm-goi-scalar.c &

tools/xngen src/x8-packw/kr-qc2w-scalar.c.in -D NR=2  -D KR=4 -D DATATYPE=QS2 -D IZP=128 -o src/qs8-qc2w-packw/gen/qs8-to-qu8-qc2w-packw-x2c4-gemm-goi-scalar.c &
tools/xngen src/x8-packw/kr-qc2w-scalar.c.in -D NR=4  -D KR=4 -D DATATYPE=QS2 -D IZP=128 -o src/qs8-qc2w-packw/gen/qs8-to-qu8-qc2w-packw-x4c4-gemm-goi-scalar.c &
tools/xngen src/x8-packw/kr-qc2w-scalar.c.in -D NR=8  -D KR=4 -D DATATYPE=QS2 -D IZP=128 -o src/qs8-qc2w-packw/gen/qs8-to-qu8-qc2w-packw-x8c4-gemm-goi-scalar.c &
tools/xngen src/x8-packw/kr-qc2w-scalar.c.in -D NR=16 -D KR=4 -D DATATYPE=QS2 -D IZP=128 -o src/qs8-qc2w-packw/gen/qs8-to-qu8-qc2w-packw-x16c4-gemm-goi-scalar.c &

tools/xngen src/x8-packw/kr-qc2w-scalar.c.in -D NR=2  -D KR=8 -D DATATYPE=QS2 -D IZP=0   -o src/qs8-qc2w-packw/gen/qs8-qc2w-packw-x2c8-gemm-goi-scalar.c &
tools/xngen src/x8-packw/kr-qc2w-scalar.c.in -D NR=4  -D KR=8 -D DATATYPE=QS2 -D IZP=0   -o src/qs8-qc2w-packw/gen/qs8-qc2w-packw-x4c8-gemm-goi-scalar.c &
tools/xngen src/x8-packw/kr-qc2w-scalar.c.in -D NR=8  -D KR=8 -D DATATYPE=QS2 -D IZP=0   -o src/qs8-qc2w-packw/gen/qs8-qc2w-packw-x8c8-gemm-goi-scalar.c &
tools/xngen src/x8-packw/kr-qc2w-scalar.c.in -D NR=16 -D KR=8 -D DATATYPE=QS2 -D IZP=0   -o src/qs8-qc2w-packw/gen/qs8-qc2w-packw-x16c8-gemm-goi-scalar.c &

tools/xngen src/x8-packw/kr-qc2w-scalar.c.in -D NR=2  -D KR=4 -D DATATYPE=QS2 -D IZP=0   -o src/qs8-qc2w-packw/gen/qs8-qc2w-packw-x2c4-gemm-goi-scalar.c &
tools/xngen src/x8-packw/kr-qc2w-scalar.c.in -D NR=4  -D KR=4 -D DATATYPE=QS2 -D IZP=0   -o src/qs8-qc2w-packw/gen/qs8-qc2w-packw-x4c4-gemm-goi-scalar.c &
tools/xngen src/x8-packw/kr-qc2w-scalar.c.in -D NR=8  -D KR=4 -D DATATYPE=QS2 -D IZP=0   -o src/qs8-qc2w-packw/gen/qs8-qc2w-packw-x8c4-gemm-goi-scalar.c &
tools/xngen src/x8-packw/kr-qc2w-scalar.c.in -D NR=16 -D KR=4 -D DATATYPE=QS2 -D IZP=0   -o src/qs8-qc2w-packw/gen/qs8-qc2w-packw-x16c4-gemm-goi-scalar.c &

### SSE2/SSSE3 C8 packing
tools/xngen src/x8-packw/kr-qc2w-sse.c.in -D NR=4 -D KR=8 -D DATATYPE=QD8_QC2W -D IZP=0 -o src/qs8-qc2w-packw/gen/qd8-qc2w-packw-x4c8-gemm-goi-ssse3.c &
tools/xngen src/x8-packw/kr-qc2w-sse.c.in -D NR=8 -D KR=8 -D DATATYPE=QD8_QC2W -D IZP=0 -o src/qs8-qc2w-packw/gen/qd8-qc2w-packw-x8c8-gemm-goi-ssse3.c &
tools/xngen src/x8-packw/kr-qc2w-sse.c.in -D NR=4 -D KR=8 -D DATATYPE=QS2 -D IZP=128 -o src/qs8-qc2w-packw/gen/qs8-to-qu8-qc2w-packw-x4c8-gemm-goi-ssse3.c &
tools/xngen src/x8-packw/kr-qc2w-sse.c.in -D NR=8 -D KR=8 -D DATATYPE=QS2 -D IZP=128 -o src/qs8-qc2w-packw/gen/qs8-to-qu8-qc2w-packw-x8c8-gemm-goi-ssse3.c &

### AVX2 C8 packing
tools/xngen src/x8-packw/kr-qc2w-avx.c.in -D NR=8  -D KR=8 -D DATATYPE=QD8_QC2W -D IZP=0   -o src/qs8-qc2w-packw/gen/qd8-qc2w-packw-x8c8-gemm-goi-avx2.c &
tools/xngen src/x8-packw/kr-qc2w-avx.c.in -D NR=16 -D KR=8 -D DATATYPE=QD8_QC2W -D IZP=0   -o src/qs8-qc2w-packw/gen/qd8-qc2w-packw-x16c8-gemm-goi-avx2.c &
tools/xngen src/x8-packw/kr-qc2w-avx.c.in -D NR=8  -D KR=8 -D DATATYPE=QS2 -D IZP=128 -o src/qs8-qc2w-packw/gen/qs8-to-qu8-qc2w-packw-x8c8-gemm-goi-avx2.c &
tools/xngen src/x8-packw/kr-qc2w-avx.c.in -D NR=16 -D KR=8 -D DATATYPE=QS2 -D IZP=128 -o src/qs8-qc2w-packw/gen/qs8-to-qu8-qc2w-packw-x16c8-gemm-goi-avx2.c &

### NEONDOT C4 packing
tools/xngen src/x8-packw/c4-qc2w-neon.c.in -D NR=8  -D KR=4 -D DATATYPE=QD8_QC2W -D IZP=0   -o src/qs8-qc2w-packw/gen/qd8-qc2w-packw-x8c4-gemm-goi-neondot.c &
tools/xngen src/x8-packw/c4-qc2w-neon.c.in -D NR=16 -D KR=4 -D DATATYPE=QD8_QC2W -D IZP=0   -o src/qs8-qc2w-packw/gen/qd8-qc2w-packw-x16c4-gemm-goi-neondot.c &
tools/xngen src/x8-packw/c4-qc2w-neon.c.in -D NR=8  -D KR=4 -D DATATYPE=QS2 -D IZP=128 -o src/qs8-qc2w-packw/gen/qs8-to-qu8-qc2w-packw-x8c4-gemm-goi-neondot.c &
tools/xngen src/x8-packw/c4-qc2w-neon.c.in -D NR=16 -D KR=4 -D DATATYPE=QS2 -D IZP=128 -o src/qs8-qc2w-packw/gen/qs8-to-qu8-qc2w-packw-x16c4-gemm-goi-neondot.c &
tools/xngen src/x8-packw/c4-qc2w-neon.c.in -D NR=8  -D KR=4 -D DATATYPE=QS2 -D IZP=0   -o src/qs8-qc2w-packw/gen/qs8-qc2w-packw-x8c4-gemm-goi-neondot.c &
tools/xngen src/x8-packw/c4-qc2w-neon.c.in -D NR=16 -D KR=4 -D DATATYPE=QS2 -D IZP=0   -o src/qs8-qc2w-packw/gen/qs8-qc2w-packw-x16c4-gemm-goi-neondot.c &

wait
