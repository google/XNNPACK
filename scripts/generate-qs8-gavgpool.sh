#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### Scalar ####################################
tools/xngen src/qs8-gavgpool/unipass-scalar.c.in -D ROW_TILE=7 -D CHANNEL_TILE=1 -D ACCUMULATORS=1 -o src/qs8-gavgpool/gen/7x-minmax-scalar-c1.c
tools/xngen src/qs8-gavgpool/unipass-scalar.c.in -D ROW_TILE=7 -D CHANNEL_TILE=2 -D ACCUMULATORS=1 -o src/qs8-gavgpool/gen/7x-minmax-scalar-c2.c
tools/xngen src/qs8-gavgpool/unipass-scalar.c.in -D ROW_TILE=7 -D CHANNEL_TILE=4 -D ACCUMULATORS=1 -o src/qs8-gavgpool/gen/7x-minmax-scalar-c4.c

tools/xngen src/qs8-gavgpool/multipass-scalar.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=1 -D ACCUMULATORS=1 -o src/qs8-gavgpool/gen/7p7x-minmax-scalar-c1.c
tools/xngen src/qs8-gavgpool/multipass-scalar.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=2 -D ACCUMULATORS=1 -o src/qs8-gavgpool/gen/7p7x-minmax-scalar-c2.c
tools/xngen src/qs8-gavgpool/multipass-scalar.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=4 -D ACCUMULATORS=1 -o src/qs8-gavgpool/gen/7p7x-minmax-scalar-c4.c

################################## ARM NEON ###################################
tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=8  -D ACCUMULATORS=2 -o src/qs8-gavgpool/gen/7x-minmax-neon-c8-acc2.c
tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=16 -D ACCUMULATORS=2 -o src/qs8-gavgpool/gen/7x-minmax-neon-c16-acc2.c
tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=24 -D ACCUMULATORS=2 -o src/qs8-gavgpool/gen/7x-minmax-neon-c24-acc2.c
tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=32 -D ACCUMULATORS=2 -o src/qs8-gavgpool/gen/7x-minmax-neon-c32-acc2.c

tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=8  -D ACCUMULATORS=2 -o src/qs8-gavgpool/gen/7p7x-minmax-neon-c8-acc2.c
tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=16 -D ACCUMULATORS=2 -o src/qs8-gavgpool/gen/7p7x-minmax-neon-c16-acc2.c
tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=24 -D ACCUMULATORS=2 -o src/qs8-gavgpool/gen/7p7x-minmax-neon-c24-acc2.c
tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=32 -D ACCUMULATORS=2 -o src/qs8-gavgpool/gen/7p7x-minmax-neon-c32-acc2.c

################################## WAsm SIMD ##################################
tools/xngen src/qs8-gavgpool/unipass-wasmsimd.c.in -D ROW_TILE=7 -D CHANNEL_TILE=8  -D ACCUMULATORS=2 -o src/qs8-gavgpool/gen/7x-minmax-wasmsimd-c8-acc2.c
tools/xngen src/qs8-gavgpool/unipass-wasmsimd.c.in -D ROW_TILE=7 -D CHANNEL_TILE=16 -D ACCUMULATORS=2 -o src/qs8-gavgpool/gen/7x-minmax-wasmsimd-c16-acc2.c
tools/xngen src/qs8-gavgpool/unipass-wasmsimd.c.in -D ROW_TILE=7 -D CHANNEL_TILE=24 -D ACCUMULATORS=2 -o src/qs8-gavgpool/gen/7x-minmax-wasmsimd-c24-acc2.c

tools/xngen src/qs8-gavgpool/multipass-wasmsimd.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=8  -D ACCUMULATORS=2 -o src/qs8-gavgpool/gen/7p7x-minmax-wasmsimd-c8-acc2.c
tools/xngen src/qs8-gavgpool/multipass-wasmsimd.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=16 -D ACCUMULATORS=2 -o src/qs8-gavgpool/gen/7p7x-minmax-wasmsimd-c16-acc2.c
tools/xngen src/qs8-gavgpool/multipass-wasmsimd.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=24 -D ACCUMULATORS=2 -o src/qs8-gavgpool/gen/7p7x-minmax-wasmsimd-c24-acc2.c

################################### x86 SSE ###################################
tools/xngen src/qs8-gavgpool/unipass-sse.c.in -D ROW_TILE=7 -D CHANNEL_TILE=8  -D ACCUMULATORS=2 -D SSE=2 -o src/qs8-gavgpool/gen/7x-minmax-sse2-c8-acc2.c
tools/xngen src/qs8-gavgpool/unipass-sse.c.in -D ROW_TILE=7 -D CHANNEL_TILE=16 -D ACCUMULATORS=2 -D SSE=2 -o src/qs8-gavgpool/gen/7x-minmax-sse2-c16-acc2.c
tools/xngen src/qs8-gavgpool/unipass-sse.c.in -D ROW_TILE=7 -D CHANNEL_TILE=24 -D ACCUMULATORS=2 -D SSE=2 -o src/qs8-gavgpool/gen/7x-minmax-sse2-c24-acc2.c

tools/xngen src/qs8-gavgpool/unipass-sse.c.in -D ROW_TILE=7 -D CHANNEL_TILE=8  -D ACCUMULATORS=2 -D SSE=3 -o src/qs8-gavgpool/gen/7x-minmax-ssse3-c8-acc2.c
tools/xngen src/qs8-gavgpool/unipass-sse.c.in -D ROW_TILE=7 -D CHANNEL_TILE=16 -D ACCUMULATORS=2 -D SSE=3 -o src/qs8-gavgpool/gen/7x-minmax-ssse3-c16-acc2.c
tools/xngen src/qs8-gavgpool/unipass-sse.c.in -D ROW_TILE=7 -D CHANNEL_TILE=24 -D ACCUMULATORS=2 -D SSE=3 -o src/qs8-gavgpool/gen/7x-minmax-ssse3-c24-acc2.c

tools/xngen src/qs8-gavgpool/unipass-sse.c.in -D ROW_TILE=7 -D CHANNEL_TILE=8  -D ACCUMULATORS=2 -D SSE=4 -o src/qs8-gavgpool/gen/7x-minmax-sse41-c8-acc2.c
tools/xngen src/qs8-gavgpool/unipass-sse.c.in -D ROW_TILE=7 -D CHANNEL_TILE=16 -D ACCUMULATORS=2 -D SSE=4 -o src/qs8-gavgpool/gen/7x-minmax-sse41-c16-acc2.c
tools/xngen src/qs8-gavgpool/unipass-sse.c.in -D ROW_TILE=7 -D CHANNEL_TILE=24 -D ACCUMULATORS=2 -D SSE=4 -o src/qs8-gavgpool/gen/7x-minmax-sse41-c24-acc2.c

tools/xngen src/qs8-gavgpool/multipass-sse.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=8  -D ACCUMULATORS=2 -D SSE=2 -o src/qs8-gavgpool/gen/7p7x-minmax-sse2-c8-acc2.c
tools/xngen src/qs8-gavgpool/multipass-sse.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=16 -D ACCUMULATORS=2 -D SSE=2 -o src/qs8-gavgpool/gen/7p7x-minmax-sse2-c16-acc2.c
tools/xngen src/qs8-gavgpool/multipass-sse.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=24 -D ACCUMULATORS=2 -D SSE=2 -o src/qs8-gavgpool/gen/7p7x-minmax-sse2-c24-acc2.c

tools/xngen src/qs8-gavgpool/multipass-sse.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=8  -D ACCUMULATORS=2 -D SSE=3 -o src/qs8-gavgpool/gen/7p7x-minmax-ssse3-c8-acc2.c
tools/xngen src/qs8-gavgpool/multipass-sse.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=16 -D ACCUMULATORS=2 -D SSE=3 -o src/qs8-gavgpool/gen/7p7x-minmax-ssse3-c16-acc2.c
tools/xngen src/qs8-gavgpool/multipass-sse.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=24 -D ACCUMULATORS=2 -D SSE=3 -o src/qs8-gavgpool/gen/7p7x-minmax-ssse3-c24-acc2.c

tools/xngen src/qs8-gavgpool/multipass-sse.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=8  -D ACCUMULATORS=2 -D SSE=4 -o src/qs8-gavgpool/gen/7p7x-minmax-sse41-c8-acc2.c
tools/xngen src/qs8-gavgpool/multipass-sse.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=16 -D ACCUMULATORS=2 -D SSE=4 -o src/qs8-gavgpool/gen/7p7x-minmax-sse41-c16-acc2.c
tools/xngen src/qs8-gavgpool/multipass-sse.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=24 -D ACCUMULATORS=2 -D SSE=4 -o src/qs8-gavgpool/gen/7p7x-minmax-sse41-c24-acc2.c

################################## Unit tests #################################
tools/generate-gavgpool-test.py --spec test/qs8-gavgpool-minmax.yaml --output test/qs8-gavgpool-minmax.cc
