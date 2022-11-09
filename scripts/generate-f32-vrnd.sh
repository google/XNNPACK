#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDNE -D BATCH_TILE=1 -o src/f32-vrnd/gen/f32-vrndne-scalar-libm-x1.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDNE -D BATCH_TILE=2 -o src/f32-vrnd/gen/f32-vrndne-scalar-libm-x2.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDNE -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndne-scalar-libm-x4.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDZ  -D BATCH_TILE=1 -o src/f32-vrnd/gen/f32-vrndz-scalar-libm-x1.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDZ  -D BATCH_TILE=2 -o src/f32-vrnd/gen/f32-vrndz-scalar-libm-x2.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDZ  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndz-scalar-libm-x4.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDU  -D BATCH_TILE=1 -o src/f32-vrnd/gen/f32-vrndu-scalar-libm-x1.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDU  -D BATCH_TILE=2 -o src/f32-vrnd/gen/f32-vrndu-scalar-libm-x2.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDU  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndu-scalar-libm-x4.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDD  -D BATCH_TILE=1 -o src/f32-vrnd/gen/f32-vrndd-scalar-libm-x1.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDD  -D BATCH_TILE=2 -o src/f32-vrnd/gen/f32-vrndd-scalar-libm-x2.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDD  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndd-scalar-libm-x4.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-vrnd/wasmsimd.c.in -D BATCH_TILE=4 -D OP=RNDNE -o src/f32-vrnd/gen/f32-vrndne-wasmsimd-x4.c &
tools/xngen src/f32-vrnd/wasmsimd.c.in -D BATCH_TILE=8 -D OP=RNDNE -o src/f32-vrnd/gen/f32-vrndne-wasmsimd-x8.c &
tools/xngen src/f32-vrnd/wasmsimd.c.in -D BATCH_TILE=4 -D OP=RNDZ  -o src/f32-vrnd/gen/f32-vrndz-wasmsimd-x4.c &
tools/xngen src/f32-vrnd/wasmsimd.c.in -D BATCH_TILE=8 -D OP=RNDZ  -o src/f32-vrnd/gen/f32-vrndz-wasmsimd-x8.c &
tools/xngen src/f32-vrnd/wasmsimd.c.in -D BATCH_TILE=4 -D OP=RNDU  -o src/f32-vrnd/gen/f32-vrndu-wasmsimd-x4.c &
tools/xngen src/f32-vrnd/wasmsimd.c.in -D BATCH_TILE=8 -D OP=RNDU  -o src/f32-vrnd/gen/f32-vrndu-wasmsimd-x8.c &
tools/xngen src/f32-vrnd/wasmsimd.c.in -D BATCH_TILE=4 -D OP=RNDD  -o src/f32-vrnd/gen/f32-vrndd-wasmsimd-x4.c &
tools/xngen src/f32-vrnd/wasmsimd.c.in -D BATCH_TILE=8 -D OP=RNDD  -o src/f32-vrnd/gen/f32-vrndd-wasmsimd-x8.c &

################################### ARM NEON ##################################
tools/xngen src/f32-vrnd/vrndne-neon.c.in -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndne-neon-x4.c &
tools/xngen src/f32-vrnd/vrndne-neon.c.in -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndne-neon-x8.c &
tools/xngen src/f32-vrnd/vrndz-neon.c.in  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndz-neon-x4.c &
tools/xngen src/f32-vrnd/vrndz-neon.c.in  -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndz-neon-x8.c &
tools/xngen src/f32-vrnd/vrndu-neon.c.in  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndu-neon-x4.c &
tools/xngen src/f32-vrnd/vrndu-neon.c.in  -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndu-neon-x8.c &
tools/xngen src/f32-vrnd/vrndd-neon.c.in  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndd-neon-x4.c &
tools/xngen src/f32-vrnd/vrndd-neon.c.in  -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndd-neon-x8.c &

tools/xngen src/f32-vrnd/neonv8.c.in -D OP=RNDNE -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndne-neonv8-x4.c &
tools/xngen src/f32-vrnd/neonv8.c.in -D OP=RNDNE -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndne-neonv8-x8.c &
tools/xngen src/f32-vrnd/neonv8.c.in -D OP=RNDZ  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndz-neonv8-x4.c &
tools/xngen src/f32-vrnd/neonv8.c.in -D OP=RNDZ  -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndz-neonv8-x8.c &
tools/xngen src/f32-vrnd/neonv8.c.in -D OP=RNDU  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndu-neonv8-x4.c &
tools/xngen src/f32-vrnd/neonv8.c.in -D OP=RNDU  -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndu-neonv8-x8.c &
tools/xngen src/f32-vrnd/neonv8.c.in -D OP=RNDD  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndd-neonv8-x4.c &
tools/xngen src/f32-vrnd/neonv8.c.in -D OP=RNDD  -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndd-neonv8-x8.c &

################################# x86 128-bit #################################
tools/xngen src/f32-vrnd/vrndne-sse2.c.in -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndne-sse2-x4.c &
tools/xngen src/f32-vrnd/vrndne-sse2.c.in -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndne-sse2-x8.c &
tools/xngen src/f32-vrnd/vrndz-sse2.c.in  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndz-sse2-x4.c &
tools/xngen src/f32-vrnd/vrndz-sse2.c.in  -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndz-sse2-x8.c &
tools/xngen src/f32-vrnd/vrndu-sse2.c.in  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndu-sse2-x4.c &
tools/xngen src/f32-vrnd/vrndu-sse2.c.in  -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndu-sse2-x8.c &
tools/xngen src/f32-vrnd/vrndd-sse2.c.in  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndd-sse2-x4.c &
tools/xngen src/f32-vrnd/vrndd-sse2.c.in  -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndd-sse2-x8.c &

tools/xngen src/f32-vrnd/sse41.c.in -D OP=RNDNE -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndne-sse41-x4.c &
tools/xngen src/f32-vrnd/sse41.c.in -D OP=RNDNE -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndne-sse41-x8.c &
tools/xngen src/f32-vrnd/sse41.c.in -D OP=RNDZ  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndz-sse41-x4.c &
tools/xngen src/f32-vrnd/sse41.c.in -D OP=RNDZ  -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndz-sse41-x8.c &
tools/xngen src/f32-vrnd/sse41.c.in -D OP=RNDU  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndu-sse41-x4.c &
tools/xngen src/f32-vrnd/sse41.c.in -D OP=RNDU  -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndu-sse41-x8.c &
tools/xngen src/f32-vrnd/sse41.c.in -D OP=RNDD  -D BATCH_TILE=4 -o src/f32-vrnd/gen/f32-vrndd-sse41-x4.c &
tools/xngen src/f32-vrnd/sse41.c.in -D OP=RNDD  -D BATCH_TILE=8 -o src/f32-vrnd/gen/f32-vrndd-sse41-x8.c &

################################# x86 256-bit #################################
tools/xngen src/f32-vrnd/avx.c.in -D OP=RNDNE -D BATCH_TILE=8  -o src/f32-vrnd/gen/f32-vrndne-avx-x8.c &
tools/xngen src/f32-vrnd/avx.c.in -D OP=RNDNE -D BATCH_TILE=16 -o src/f32-vrnd/gen/f32-vrndne-avx-x16.c &
tools/xngen src/f32-vrnd/avx.c.in -D OP=RNDZ  -D BATCH_TILE=8  -o src/f32-vrnd/gen/f32-vrndz-avx-x8.c &
tools/xngen src/f32-vrnd/avx.c.in -D OP=RNDZ  -D BATCH_TILE=16 -o src/f32-vrnd/gen/f32-vrndz-avx-x16.c &
tools/xngen src/f32-vrnd/avx.c.in -D OP=RNDU  -D BATCH_TILE=8  -o src/f32-vrnd/gen/f32-vrndu-avx-x8.c &
tools/xngen src/f32-vrnd/avx.c.in -D OP=RNDU  -D BATCH_TILE=16 -o src/f32-vrnd/gen/f32-vrndu-avx-x16.c &
tools/xngen src/f32-vrnd/avx.c.in -D OP=RNDD  -D BATCH_TILE=8  -o src/f32-vrnd/gen/f32-vrndd-avx-x8.c &
tools/xngen src/f32-vrnd/avx.c.in -D OP=RNDD  -D BATCH_TILE=16 -o src/f32-vrnd/gen/f32-vrndd-avx-x16.c &

################################# x86 512-bit #################################
tools/xngen src/f32-vrnd/avx512f.c.in -D OP=RNDNE -D BATCH_TILE=16 -o src/f32-vrnd/gen/f32-vrndne-avx512f-x16.c &
tools/xngen src/f32-vrnd/avx512f.c.in -D OP=RNDNE -D BATCH_TILE=32 -o src/f32-vrnd/gen/f32-vrndne-avx512f-x32.c &
tools/xngen src/f32-vrnd/avx512f.c.in -D OP=RNDZ  -D BATCH_TILE=16 -o src/f32-vrnd/gen/f32-vrndz-avx512f-x16.c &
tools/xngen src/f32-vrnd/avx512f.c.in -D OP=RNDZ  -D BATCH_TILE=32 -o src/f32-vrnd/gen/f32-vrndz-avx512f-x32.c &
tools/xngen src/f32-vrnd/avx512f.c.in -D OP=RNDU  -D BATCH_TILE=16 -o src/f32-vrnd/gen/f32-vrndu-avx512f-x16.c &
tools/xngen src/f32-vrnd/avx512f.c.in -D OP=RNDU  -D BATCH_TILE=32 -o src/f32-vrnd/gen/f32-vrndu-avx512f-x32.c &
tools/xngen src/f32-vrnd/avx512f.c.in -D OP=RNDD  -D BATCH_TILE=16 -o src/f32-vrnd/gen/f32-vrndd-avx512f-x16.c &
tools/xngen src/f32-vrnd/avx512f.c.in -D OP=RNDD  -D BATCH_TILE=32 -o src/f32-vrnd/gen/f32-vrndd-avx512f-x32.c &

################################## Unit tests #################################
tools/generate-vunary-test.py --spec test/f32-vrndne.yaml --output test/f32-vrndne.cc &
tools/generate-vunary-test.py --spec test/f32-vrndz.yaml  --output test/f32-vrndz.cc &
tools/generate-vunary-test.py --spec test/f32-vrndu.yaml  --output test/f32-vrndu.cc &
tools/generate-vunary-test.py --spec test/f32-vrndd.yaml  --output test/f32-vrndd.cc &

wait
