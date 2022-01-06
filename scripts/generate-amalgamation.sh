#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

tools/amalgamate-microkernels.py -s PROD_SSE_MICROKERNEL_SRCS -o src/amalgam/sse.c &
tools/amalgamate-microkernels.py -s PROD_SSE2_MICROKERNEL_SRCS -o src/amalgam/sse2.c &
tools/amalgamate-microkernels.py -s PROD_SSSE3_MICROKERNEL_SRCS -o src/amalgam/ssse3.c &
tools/amalgamate-microkernels.py -s PROD_SSE41_MICROKERNEL_SRCS -o src/amalgam/sse41.c &
tools/amalgamate-microkernels.py -s PROD_AVX_MICROKERNEL_SRCS -o src/amalgam/avx.c &
tools/amalgamate-microkernels.py -s PROD_FMA3_MICROKERNEL_SRCS -o src/amalgam/fma3.c &
tools/amalgamate-microkernels.py -s PROD_AVX2_MICROKERNEL_SRCS -o src/amalgam/avx2.c &
tools/amalgamate-microkernels.py -s PROD_F16C_MICROKERNEL_SRCS -o src/amalgam/f16c.c &
tools/amalgamate-microkernels.py -s PROD_AVX512F_MICROKERNEL_SRCS -o src/amalgam/avx512f.c &
tools/amalgamate-microkernels.py -s PROD_AVX512SKX_MICROKERNEL_SRCS -o src/amalgam/avx512skx.c &

wait
