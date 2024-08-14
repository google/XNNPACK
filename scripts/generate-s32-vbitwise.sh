#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD VAND #####################################
tools/xngen src/s32-vbitwise/s32-vbitwise.c.in -D ARCH=scalar -D OP=AND -D BATCH_TILES=1,2,4,8  -o src/s32-vbitwise/gen/s32-vand-scalar.c &
tools/xngen src/s32-vbitwise/s32-vbitwise.c.in -D ARCH=sse41  -D OP=AND -D BATCH_TILES=4,8,12,16  -o src/s32-vbitwise/gen/s32-vand-sse41.c &
tools/xngen src/s32-vbitwise/s32-vbitwise.c.in -D ARCH=wasmsimd -D OP=AND  -D BATCH_TILES=4,8,12,16  -o src/s32-vbitwise/gen/s32-vand-wasmsimd.c &
tools/xngen src/s32-vbitwise/s32-vbitwise.c.in -D ARCH=neon  -D OP=AND  -D BATCH_TILES=4,8,12,16  -o src/s32-vbitwise/gen/s32-vand-neon.c &
tools/xngen src/s32-vbitwise/s32-vbitwise.c.in -D ARCH=avx2  -D OP=AND  -D BATCH_TILES=8,16,24,32  -o src/s32-vbitwise/gen/s32-vand-avx2.c &
tools/xngen src/s32-vbitwise/s32-vbitwise.c.in -D ARCH=avx512f  -D OP=AND -D BATCH_TILES=16,32,48,64  -o src/s32-vbitwise/gen/s32-vand-avx512f.c &

##################################### SIMD VANDC #####################################
tools/xngen src/s32-vbitwise/s32-vbitwisec.c.in -D ARCH=scalar -D OP=AND -D BATCH_TILES=1,2,4,8  -o src/s32-vbitwise/gen/s32-vandc-scalar.c &
tools/xngen src/s32-vbitwise/s32-vbitwisec.c.in -D ARCH=sse41 -D OP=AND -D BATCH_TILES=4,8,12,16  -o src/s32-vbitwise/gen/s32-vandc-sse41.c &
tools/xngen src/s32-vbitwise/s32-vbitwisec.c.in -D ARCH=wasmsimd  -D OP=AND -D BATCH_TILES=4,8,12,16  -o src/s32-vbitwise/gen/s32-vandc-wasmsimd.c &
tools/xngen src/s32-vbitwise/s32-vbitwisec.c.in -D ARCH=neon -D OP=AND  -D BATCH_TILES=4,8,12,16  -o src/s32-vbitwise/gen/s32-vandc-neon.c &
tools/xngen src/s32-vbitwise/s32-vbitwisec.c.in -D ARCH=avx2  -D OP=AND -D BATCH_TILES=8,16,24,32  -o src/s32-vbitwise/gen/s32-vandc-avx2.c &
tools/xngen src/s32-vbitwise/s32-vbitwisec.c.in -D ARCH=avx512f  -D OP=AND -D BATCH_TILES=16,32,48,64  -o src/s32-vbitwise/gen/s32-vandc-avx512f.c &

##################################### SIMD VOR #####################################
tools/xngen src/s32-vbitwise/s32-vbitwise.c.in -D ARCH=scalar -D OP=OR -D BATCH_TILES=1,2,4,8  -o src/s32-vbitwise/gen/s32-vor-scalar.c &
tools/xngen src/s32-vbitwise/s32-vbitwise.c.in -D ARCH=sse41  -D OP=OR -D BATCH_TILES=4,8,12,16  -o src/s32-vbitwise/gen/s32-vor-sse41.c &
tools/xngen src/s32-vbitwise/s32-vbitwise.c.in -D ARCH=wasmsimd -D OP=OR  -D BATCH_TILES=4,8,12,16  -o src/s32-vbitwise/gen/s32-vor-wasmsimd.c &
tools/xngen src/s32-vbitwise/s32-vbitwise.c.in -D ARCH=neon  -D OP=OR  -D BATCH_TILES=4,8,12,16  -o src/s32-vbitwise/gen/s32-vor-neon.c &
tools/xngen src/s32-vbitwise/s32-vbitwise.c.in -D ARCH=avx2  -D OP=OR  -D BATCH_TILES=8,16,24,32  -o src/s32-vbitwise/gen/s32-vor-avx2.c &
tools/xngen src/s32-vbitwise/s32-vbitwise.c.in -D ARCH=avx512f  -D OP=OR -D BATCH_TILES=16,32,48,64  -o src/s32-vbitwise/gen/s32-vor-avx512f.c &

##################################### SIMD VORC #####################################
tools/xngen src/s32-vbitwise/s32-vbitwisec.c.in -D ARCH=scalar -D OP=OR -D BATCH_TILES=1,2,4,8  -o src/s32-vbitwise/gen/s32-vorc-scalar.c &
tools/xngen src/s32-vbitwise/s32-vbitwisec.c.in -D ARCH=sse41 -D OP=OR -D BATCH_TILES=4,8,12,16  -o src/s32-vbitwise/gen/s32-vorc-sse41.c &
tools/xngen src/s32-vbitwise/s32-vbitwisec.c.in -D ARCH=wasmsimd  -D OP=OR -D BATCH_TILES=4,8,12,16  -o src/s32-vbitwise/gen/s32-vorc-wasmsimd.c &
tools/xngen src/s32-vbitwise/s32-vbitwisec.c.in -D ARCH=neon -D OP=OR  -D BATCH_TILES=4,8,12,16  -o src/s32-vbitwise/gen/s32-vorc-neon.c &
tools/xngen src/s32-vbitwise/s32-vbitwisec.c.in -D ARCH=avx2  -D OP=OR -D BATCH_TILES=8,16,24,32  -o src/s32-vbitwise/gen/s32-vorc-avx2.c &
tools/xngen src/s32-vbitwise/s32-vbitwisec.c.in -D ARCH=avx512f  -D OP=OR -D BATCH_TILES=16,32,48,64  -o src/s32-vbitwise/gen/s32-vorc-avx512f.c &

##################################### SIMD VXOR #####################################
tools/xngen src/s32-vbitwise/s32-vbitwise.c.in -D ARCH=scalar -D OP=XOR -D BATCH_TILES=1,2,4,8  -o src/s32-vbitwise/gen/s32-vxor-scalar.c &
tools/xngen src/s32-vbitwise/s32-vbitwise.c.in -D ARCH=sse41  -D OP=XOR -D BATCH_TILES=4,8,12,16  -o src/s32-vbitwise/gen/s32-vxor-sse41.c &
tools/xngen src/s32-vbitwise/s32-vbitwise.c.in -D ARCH=wasmsimd -D OP=XOR  -D BATCH_TILES=4,8,12,16  -o src/s32-vbitwise/gen/s32-vxor-wasmsimd.c &
tools/xngen src/s32-vbitwise/s32-vbitwise.c.in -D ARCH=neon  -D OP=XOR  -D BATCH_TILES=4,8,12,16  -o src/s32-vbitwise/gen/s32-vxor-neon.c &
tools/xngen src/s32-vbitwise/s32-vbitwise.c.in -D ARCH=avx2  -D OP=XOR  -D BATCH_TILES=8,16,24,32  -o src/s32-vbitwise/gen/s32-vxor-avx2.c &
tools/xngen src/s32-vbitwise/s32-vbitwise.c.in -D ARCH=avx512f  -D OP=XOR -D BATCH_TILES=16,32,48,64  -o src/s32-vbitwise/gen/s32-vxor-avx512f.c &

##################################### SIMD VXORC #####################################
tools/xngen src/s32-vbitwise/s32-vbitwisec.c.in -D ARCH=scalar -D OP=XOR -D BATCH_TILES=1,2,4,8  -o src/s32-vbitwise/gen/s32-vxorc-scalar.c &
tools/xngen src/s32-vbitwise/s32-vbitwisec.c.in -D ARCH=sse41 -D OP=XOR -D BATCH_TILES=4,8,12,16  -o src/s32-vbitwise/gen/s32-vxorc-sse41.c &
tools/xngen src/s32-vbitwise/s32-vbitwisec.c.in -D ARCH=wasmsimd  -D OP=XOR -D BATCH_TILES=4,8,12,16  -o src/s32-vbitwise/gen/s32-vxorc-wasmsimd.c &
tools/xngen src/s32-vbitwise/s32-vbitwisec.c.in -D ARCH=neon -D OP=XOR  -D BATCH_TILES=4,8,12,16  -o src/s32-vbitwise/gen/s32-vxorc-neon.c &
tools/xngen src/s32-vbitwise/s32-vbitwisec.c.in -D ARCH=avx2  -D OP=XOR -D BATCH_TILES=8,16,24,32  -o src/s32-vbitwise/gen/s32-vxorc-avx2.c &
tools/xngen src/s32-vbitwise/s32-vbitwisec.c.in -D ARCH=avx512f  -D OP=XOR -D BATCH_TILES=16,32,48,64  -o src/s32-vbitwise/gen/s32-vxorc-avx512f.c &

wait
