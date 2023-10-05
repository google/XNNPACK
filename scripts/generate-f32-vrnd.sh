#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDNE -D BATCH_TILE=1 -o src/f32-vrnd/gen/f32-vrndne-scalar-libm-u1.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDNE -D BATCH_TILE=2 -o src/f32-vrnd/gen/f32-vrndne-scalar-libm-u2.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDNE -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndne-scalar-libm-u4.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDZ  -D BATCH_TILE=1 -o src/f32-vrnd/gen/f32-vrndz-scalar-libm-u1.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDZ  -D BATCH_TILE=2 -o src/f32-vrnd/gen/f32-vrndz-scalar-libm-u2.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDZ  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndz-scalar-libm-u4.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDU  -D BATCH_TILE=1 -o src/f32-vrnd/gen/f32-vrndu-scalar-libm-u1.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDU  -D BATCH_TILE=2 -o src/f32-vrnd/gen/f32-vrndu-scalar-libm-u2.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDU  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndu-scalar-libm-u4.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDD  -D BATCH_TILE=1 -o src/f32-vrnd/gen/f32-vrndd-scalar-libm-u1.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDD  -D BATCH_TILE=2 -o src/f32-vrnd/gen/f32-vrndd-scalar-libm-u2.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDD  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndd-scalar-libm-u4.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-vrnd/wasmsimd.c.in -D BATCH_TILE=4 -D OP=RNDNE -o src/f32-vrnd/gen/f32-vrndne-wasmsimd-u4.c &
tools/xngen src/f32-vrnd/wasmsimd.c.in -D BATCH_TILE=8 -D OP=RNDNE -o src/f32-vrnd/gen/f32-vrndne-wasmsimd-u8.c &
tools/xngen src/f32-vrnd/wasmsimd.c.in -D BATCH_TILE=4 -D OP=RNDZ  -o src/f32-vrnd/gen/f32-vrndz-wasmsimd-u4.c &
tools/xngen src/f32-vrnd/wasmsimd.c.in -D BATCH_TILE=8 -D OP=RNDZ  -o src/f32-vrnd/gen/f32-vrndz-wasmsimd-u8.c &
tools/xngen src/f32-vrnd/wasmsimd.c.in -D BATCH_TILE=4 -D OP=RNDU  -o src/f32-vrnd/gen/f32-vrndu-wasmsimd-u4.c &
tools/xngen src/f32-vrnd/wasmsimd.c.in -D BATCH_TILE=8 -D OP=RNDU  -o src/f32-vrnd/gen/f32-vrndu-wasmsimd-u8.c &
tools/xngen src/f32-vrnd/wasmsimd.c.in -D BATCH_TILE=4 -D OP=RNDD  -o src/f32-vrnd/gen/f32-vrndd-wasmsimd-u4.c &
tools/xngen src/f32-vrnd/wasmsimd.c.in -D BATCH_TILE=8 -D OP=RNDD  -o src/f32-vrnd/gen/f32-vrndd-wasmsimd-u8.c &

################################### ARM NEON ##################################
tools/xngen src/f32-vrnd/vrndne-neon.c.in -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndne-neon-u4.c &
tools/xngen src/f32-vrnd/vrndne-neon.c.in -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndne-neon-u8.c &
tools/xngen src/f32-vrnd/vrndz-neon.c.in  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndz-neon-u4.c &
tools/xngen src/f32-vrnd/vrndz-neon.c.in  -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndz-neon-u8.c &
tools/xngen src/f32-vrnd/vrndu-neon.c.in  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndu-neon-u4.c &
tools/xngen src/f32-vrnd/vrndu-neon.c.in  -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndu-neon-u8.c &
tools/xngen src/f32-vrnd/vrndd-neon.c.in  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndd-neon-u4.c &
tools/xngen src/f32-vrnd/vrndd-neon.c.in  -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndd-neon-u8.c &

tools/xngen src/f32-vrnd/neonv8.c.in -D OP=RNDNE -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndne-neonv8-u4.c &
tools/xngen src/f32-vrnd/neonv8.c.in -D OP=RNDNE -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndne-neonv8-u8.c &
tools/xngen src/f32-vrnd/neonv8.c.in -D OP=RNDZ  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndz-neonv8-u4.c &
tools/xngen src/f32-vrnd/neonv8.c.in -D OP=RNDZ  -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndz-neonv8-u8.c &
tools/xngen src/f32-vrnd/neonv8.c.in -D OP=RNDU  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndu-neonv8-u4.c &
tools/xngen src/f32-vrnd/neonv8.c.in -D OP=RNDU  -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndu-neonv8-u8.c &
tools/xngen src/f32-vrnd/neonv8.c.in -D OP=RNDD  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndd-neonv8-u4.c &
tools/xngen src/f32-vrnd/neonv8.c.in -D OP=RNDD  -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndd-neonv8-u8.c &

################################# x86 128-bit #################################
tools/xngen src/f32-vrnd/vrndne-sse2.c.in -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndne-sse2-u4.c &
tools/xngen src/f32-vrnd/vrndne-sse2.c.in -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndne-sse2-u8.c &
tools/xngen src/f32-vrnd/vrndz-sse2.c.in  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndz-sse2-u4.c &
tools/xngen src/f32-vrnd/vrndz-sse2.c.in  -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndz-sse2-u8.c &
tools/xngen src/f32-vrnd/vrndu-sse2.c.in  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndu-sse2-u4.c &
tools/xngen src/f32-vrnd/vrndu-sse2.c.in  -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndu-sse2-u8.c &
tools/xngen src/f32-vrnd/vrndd-sse2.c.in  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndd-sse2-u4.c &
tools/xngen src/f32-vrnd/vrndd-sse2.c.in  -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndd-sse2-u8.c &

tools/xngen src/f32-vrnd/sse41.c.in -D OP=RNDNE -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndne-sse41-u4.c &
tools/xngen src/f32-vrnd/sse41.c.in -D OP=RNDNE -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndne-sse41-u8.c &
tools/xngen src/f32-vrnd/sse41.c.in -D OP=RNDZ  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndz-sse41-u4.c &
tools/xngen src/f32-vrnd/sse41.c.in -D OP=RNDZ  -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndz-sse41-u8.c &
tools/xngen src/f32-vrnd/sse41.c.in -D OP=RNDU  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndu-sse41-u4.c &
tools/xngen src/f32-vrnd/sse41.c.in -D OP=RNDU  -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndu-sse41-u8.c &
tools/xngen src/f32-vrnd/sse41.c.in -D OP=RNDD  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndd-sse41-u4.c &
tools/xngen src/f32-vrnd/sse41.c.in -D OP=RNDD  -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndd-sse41-u8.c &

################################# x86 256-bit #################################
tools/xngen src/f32-vrnd/avx.c.in -D OP=RNDNE -D BATCH_TILE=8  -o src/f32-vrnd/gen/f32-vrndne-avx-u8.c &
tools/xngen src/f32-vrnd/avx.c.in -D OP=RNDNE -D BATCH_TILE=16 -o src/f32-vrnd/gen/f32-vrndne-avx-u16.c &
tools/xngen src/f32-vrnd/avx.c.in -D OP=RNDZ  -D BATCH_TILE=8  -o src/f32-vrnd/gen/f32-vrndz-avx-u8.c &
tools/xngen src/f32-vrnd/avx.c.in -D OP=RNDZ  -D BATCH_TILE=16 -o src/f32-vrnd/gen/f32-vrndz-avx-u16.c &
tools/xngen src/f32-vrnd/avx.c.in -D OP=RNDU  -D BATCH_TILE=8  -o src/f32-vrnd/gen/f32-vrndu-avx-u8.c &
tools/xngen src/f32-vrnd/avx.c.in -D OP=RNDU  -D BATCH_TILE=16 -o src/f32-vrnd/gen/f32-vrndu-avx-u16.c &
tools/xngen src/f32-vrnd/avx.c.in -D OP=RNDD  -D BATCH_TILE=8  -o src/f32-vrnd/gen/f32-vrndd-avx-u8.c &
tools/xngen src/f32-vrnd/avx.c.in -D OP=RNDD  -D BATCH_TILE=16 -o src/f32-vrnd/gen/f32-vrndd-avx-u16.c &

################################# x86 512-bit #################################
tools/xngen src/f32-vrnd/avx512f.c.in -D OP=RNDNE -D BATCH_TILE=16 -o src/f32-vrnd/gen/f32-vrndne-avx512f-u16.c &
tools/xngen src/f32-vrnd/avx512f.c.in -D OP=RNDNE -D BATCH_TILE=32 -o src/f32-vrnd/gen/f32-vrndne-avx512f-u32.c &
tools/xngen src/f32-vrnd/avx512f.c.in -D OP=RNDZ  -D BATCH_TILE=16 -o src/f32-vrnd/gen/f32-vrndz-avx512f-u16.c &
tools/xngen src/f32-vrnd/avx512f.c.in -D OP=RNDZ  -D BATCH_TILE=32 -o src/f32-vrnd/gen/f32-vrndz-avx512f-u32.c &
tools/xngen src/f32-vrnd/avx512f.c.in -D OP=RNDU  -D BATCH_TILE=16 -o src/f32-vrnd/gen/f32-vrndu-avx512f-u16.c &
tools/xngen src/f32-vrnd/avx512f.c.in -D OP=RNDU  -D BATCH_TILE=32 -o src/f32-vrnd/gen/f32-vrndu-avx512f-u32.c &
tools/xngen src/f32-vrnd/avx512f.c.in -D OP=RNDD  -D BATCH_TILE=16 -o src/f32-vrnd/gen/f32-vrndd-avx512f-u16.c &
tools/xngen src/f32-vrnd/avx512f.c.in -D OP=RNDD  -D BATCH_TILE=32 -o src/f32-vrnd/gen/f32-vrndd-avx512f-u32.c &

wait
