#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/x8-packw/scalar.c.in -D NR=2  -D KBLOCK=2 -D TYPE=int8_t -o src/x8-packw/gen/x8-packw-x2-gemm-goi-scalar-int-u2.c &
tools/xngen src/x8-packw/scalar.c.in -D NR=4  -D KBLOCK=2 -D TYPE=int8_t -o src/x8-packw/gen/x8-packw-x4-gemm-goi-scalar-int-u2.c &
tools/xngen src/x8-packw/scalar.c.in -D NR=8  -D KBLOCK=2 -D TYPE=int8_t -o src/x8-packw/gen/x8-packw-x8-gemm-goi-scalar-int-u2.c &
tools/xngen src/x8-packw/scalar.c.in -D NR=16 -D KBLOCK=2 -D TYPE=int8_t -o src/x8-packw/gen/x8-packw-x16-gemm-goi-scalar-int-u2.c &
tools/xngen src/x8-packw/scalar.c.in -D NR=32 -D KBLOCK=2 -D TYPE=int8_t -o src/x8-packw/gen/x8-packw-x32-gemm-goi-scalar-int-u2.c &

tools/xngen src/x8-packw/scalar.c.in -D NR=2  -D KBLOCK=4 -D TYPE=int8_t -o src/x8-packw/gen/x8-packw-x2-gemm-goi-scalar-int-u4.c &
tools/xngen src/x8-packw/scalar.c.in -D NR=4  -D KBLOCK=4 -D TYPE=int8_t -o src/x8-packw/gen/x8-packw-x4-gemm-goi-scalar-int-u4.c &
tools/xngen src/x8-packw/scalar.c.in -D NR=8  -D KBLOCK=4 -D TYPE=int8_t -o src/x8-packw/gen/x8-packw-x8-gemm-goi-scalar-int-u4.c &
tools/xngen src/x8-packw/scalar.c.in -D NR=16 -D KBLOCK=4 -D TYPE=int8_t -o src/x8-packw/gen/x8-packw-x16-gemm-goi-scalar-int-u4.c &
tools/xngen src/x8-packw/scalar.c.in -D NR=32 -D KBLOCK=4 -D TYPE=int8_t -o src/x8-packw/gen/x8-packw-x32-gemm-goi-scalar-int-u4.c &


################################### x86 AVX512 ##################################
### NR multiple of 16
tools/xngen src/x8-packw/c4.c.in -D NR=16 -D KBLOCK=16 -D TYPE=int8_t -D X8KERNEL=avx512f_u16_prfm -D X32KERNEL=avx512f_u4_prfm -o src/x8-packw/gen/x8-packw-x16c4-gemm-goi-avx512f-u16-prfm.c &
tools/xngen src/x8-packw/c4.c.in -D NR=32 -D KBLOCK=16 -D TYPE=int8_t -D X8KERNEL=avx512f_u16_prfm -D X32KERNEL=avx512f_u4_prfm -o src/x8-packw/gen/x8-packw-x32c4-gemm-goi-avx512f-u16-prfm.c &
tools/xngen src/x8-packw/c4.c.in -D NR=64 -D KBLOCK=16 -D TYPE=int8_t -D X8KERNEL=avx512f_u16_prfm -D X32KERNEL=avx512f_u4_prfm -o src/x8-packw/gen/x8-packw-x64c4-gemm-goi-avx512f-u16-prfm.c &

wait
