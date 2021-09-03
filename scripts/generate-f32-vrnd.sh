#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDNE -D BATCH_TILE=1 -o src/f32-vrnd/gen/vrndne-scalar-libm-x1.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDNE -D BATCH_TILE=2 -o src/f32-vrnd/gen/vrndne-scalar-libm-x2.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDNE -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndne-scalar-libm-x4.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDZ  -D BATCH_TILE=1 -o src/f32-vrnd/gen/vrndz-scalar-libm-x1.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDZ  -D BATCH_TILE=2 -o src/f32-vrnd/gen/vrndz-scalar-libm-x2.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDZ  -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndz-scalar-libm-x4.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDU  -D BATCH_TILE=1 -o src/f32-vrnd/gen/vrndu-scalar-libm-x1.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDU  -D BATCH_TILE=2 -o src/f32-vrnd/gen/vrndu-scalar-libm-x2.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDU  -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndu-scalar-libm-x4.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDD  -D BATCH_TILE=1 -o src/f32-vrnd/gen/vrndd-scalar-libm-x1.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDD  -D BATCH_TILE=2 -o src/f32-vrnd/gen/vrndd-scalar-libm-x2.c &
tools/xngen src/f32-vrnd/scalar-libm.c.in -D OP=RNDD  -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndd-scalar-libm-x4.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-vrnd/vrndne-wasmsimd-addsub.c.in -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndne-wasmsimd-addsub-x4.c &
tools/xngen src/f32-vrnd/vrndne-wasmsimd-addsub.c.in -D BATCH_TILE=8 -o src/f32-vrnd/gen/vrndne-wasmsimd-addsub-x8.c &
tools/xngen src/f32-vrnd/vrndz-wasmsimd-addsub.c.in  -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndz-wasmsimd-addsub-x4.c &
tools/xngen src/f32-vrnd/vrndz-wasmsimd-addsub.c.in  -D BATCH_TILE=8 -o src/f32-vrnd/gen/vrndz-wasmsimd-addsub-x8.c &
tools/xngen src/f32-vrnd/vrndu-wasmsimd-addsub.c.in  -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndu-wasmsimd-addsub-x4.c &
tools/xngen src/f32-vrnd/vrndu-wasmsimd-addsub.c.in  -D BATCH_TILE=8 -o src/f32-vrnd/gen/vrndu-wasmsimd-addsub-x8.c &
tools/xngen src/f32-vrnd/vrndd-wasmsimd-addsub.c.in  -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndd-wasmsimd-addsub-x4.c &
tools/xngen src/f32-vrnd/vrndd-wasmsimd-addsub.c.in  -D BATCH_TILE=8 -o src/f32-vrnd/gen/vrndd-wasmsimd-addsub-x8.c &

tools/xngen src/f32-vrnd/vrndz-wasmsimd-cvt.c.in -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndz-wasmsimd-cvt-x4.c &
tools/xngen src/f32-vrnd/vrndz-wasmsimd-cvt.c.in -D BATCH_TILE=8 -o src/f32-vrnd/gen/vrndz-wasmsimd-cvt-x8.c &
tools/xngen src/f32-vrnd/vrndu-wasmsimd-cvt.c.in -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndu-wasmsimd-cvt-x4.c &
tools/xngen src/f32-vrnd/vrndu-wasmsimd-cvt.c.in -D BATCH_TILE=8 -o src/f32-vrnd/gen/vrndu-wasmsimd-cvt-x8.c &
tools/xngen src/f32-vrnd/vrndd-wasmsimd-cvt.c.in -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndd-wasmsimd-cvt-x4.c &
tools/xngen src/f32-vrnd/vrndd-wasmsimd-cvt.c.in -D BATCH_TILE=8 -o src/f32-vrnd/gen/vrndd-wasmsimd-cvt-x8.c &

tools/xngen src/f32-vrnd/wasmsimd-native.c.in -D BATCH_TILE=4 -D OP=RNDNE -o src/f32-vrnd/gen/vrndne-wasmsimd-native-x4.c &
tools/xngen src/f32-vrnd/wasmsimd-native.c.in -D BATCH_TILE=8 -D OP=RNDNE -o src/f32-vrnd/gen/vrndne-wasmsimd-native-x8.c &
tools/xngen src/f32-vrnd/wasmsimd-native.c.in -D BATCH_TILE=4 -D OP=RNDZ  -o src/f32-vrnd/gen/vrndz-wasmsimd-native-x4.c &
tools/xngen src/f32-vrnd/wasmsimd-native.c.in -D BATCH_TILE=8 -D OP=RNDZ  -o src/f32-vrnd/gen/vrndz-wasmsimd-native-x8.c &
tools/xngen src/f32-vrnd/wasmsimd-native.c.in -D BATCH_TILE=4 -D OP=RNDU  -o src/f32-vrnd/gen/vrndu-wasmsimd-native-x4.c &
tools/xngen src/f32-vrnd/wasmsimd-native.c.in -D BATCH_TILE=8 -D OP=RNDU  -o src/f32-vrnd/gen/vrndu-wasmsimd-native-x8.c &
tools/xngen src/f32-vrnd/wasmsimd-native.c.in -D BATCH_TILE=4 -D OP=RNDD  -o src/f32-vrnd/gen/vrndd-wasmsimd-native-x4.c &
tools/xngen src/f32-vrnd/wasmsimd-native.c.in -D BATCH_TILE=8 -D OP=RNDD  -o src/f32-vrnd/gen/vrndd-wasmsimd-native-x8.c &

################################### ARM NEON ##################################
tools/xngen src/f32-vrnd/vrndne-neon.c.in -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndne-neon-x4.c &
tools/xngen src/f32-vrnd/vrndne-neon.c.in -D BATCH_TILE=8 -o src/f32-vrnd/gen/vrndne-neon-x8.c &
tools/xngen src/f32-vrnd/vrndz-neon.c.in  -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndz-neon-x4.c &
tools/xngen src/f32-vrnd/vrndz-neon.c.in  -D BATCH_TILE=8 -o src/f32-vrnd/gen/vrndz-neon-x8.c &
tools/xngen src/f32-vrnd/vrndu-neon.c.in  -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndu-neon-x4.c &
tools/xngen src/f32-vrnd/vrndu-neon.c.in  -D BATCH_TILE=8 -o src/f32-vrnd/gen/vrndu-neon-x8.c &
tools/xngen src/f32-vrnd/vrndd-neon.c.in  -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndd-neon-x4.c &
tools/xngen src/f32-vrnd/vrndd-neon.c.in  -D BATCH_TILE=8 -o src/f32-vrnd/gen/vrndd-neon-x8.c &

tools/xngen src/f32-vrnd/neonv8.c.in -D OP=RNDNE -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndne-neonv8-x4.c &
tools/xngen src/f32-vrnd/neonv8.c.in -D OP=RNDNE -D BATCH_TILE=8 -o src/f32-vrnd/gen/vrndne-neonv8-x8.c &
tools/xngen src/f32-vrnd/neonv8.c.in -D OP=RNDZ  -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndz-neonv8-x4.c &
tools/xngen src/f32-vrnd/neonv8.c.in -D OP=RNDZ  -D BATCH_TILE=8 -o src/f32-vrnd/gen/vrndz-neonv8-x8.c &
tools/xngen src/f32-vrnd/neonv8.c.in -D OP=RNDU  -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndu-neonv8-x4.c &
tools/xngen src/f32-vrnd/neonv8.c.in -D OP=RNDU  -D BATCH_TILE=8 -o src/f32-vrnd/gen/vrndu-neonv8-x8.c &
tools/xngen src/f32-vrnd/neonv8.c.in -D OP=RNDD  -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndd-neonv8-x4.c &
tools/xngen src/f32-vrnd/neonv8.c.in -D OP=RNDD  -D BATCH_TILE=8 -o src/f32-vrnd/gen/vrndd-neonv8-x8.c &

################################# x86 128-bit #################################
tools/xngen src/f32-vrnd/vrndne-sse2.c.in -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndne-sse2-x4.c &
tools/xngen src/f32-vrnd/vrndne-sse2.c.in -D BATCH_TILE=8 -o src/f32-vrnd/gen/vrndne-sse2-x8.c &
tools/xngen src/f32-vrnd/vrndz-sse2.c.in  -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndz-sse2-x4.c &
tools/xngen src/f32-vrnd/vrndz-sse2.c.in  -D BATCH_TILE=8 -o src/f32-vrnd/gen/vrndz-sse2-x8.c &
tools/xngen src/f32-vrnd/vrndu-sse2.c.in  -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndu-sse2-x4.c &
tools/xngen src/f32-vrnd/vrndu-sse2.c.in  -D BATCH_TILE=8 -o src/f32-vrnd/gen/vrndu-sse2-x8.c &
tools/xngen src/f32-vrnd/vrndd-sse2.c.in  -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndd-sse2-x4.c &
tools/xngen src/f32-vrnd/vrndd-sse2.c.in  -D BATCH_TILE=8 -o src/f32-vrnd/gen/vrndd-sse2-x8.c &

tools/xngen src/f32-vrnd/sse41.c.in -D OP=RNDNE -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndne-sse41-x4.c &
tools/xngen src/f32-vrnd/sse41.c.in -D OP=RNDNE -D BATCH_TILE=8 -o src/f32-vrnd/gen/vrndne-sse41-x8.c &
tools/xngen src/f32-vrnd/sse41.c.in -D OP=RNDZ  -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndz-sse41-x4.c &
tools/xngen src/f32-vrnd/sse41.c.in -D OP=RNDZ  -D BATCH_TILE=8 -o src/f32-vrnd/gen/vrndz-sse41-x8.c &
tools/xngen src/f32-vrnd/sse41.c.in -D OP=RNDU  -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndu-sse41-x4.c &
tools/xngen src/f32-vrnd/sse41.c.in -D OP=RNDU  -D BATCH_TILE=8 -o src/f32-vrnd/gen/vrndu-sse41-x8.c &
tools/xngen src/f32-vrnd/sse41.c.in -D OP=RNDD  -D BATCH_TILE=4 -o src/f32-vrnd/gen/vrndd-sse41-x4.c &
tools/xngen src/f32-vrnd/sse41.c.in -D OP=RNDD  -D BATCH_TILE=8 -o src/f32-vrnd/gen/vrndd-sse41-x8.c &

################################# x86 256-bit #################################
tools/xngen src/f32-vrnd/avx.c.in -D OP=RNDNE -D BATCH_TILE=8  -o src/f32-vrnd/gen/vrndne-avx-x8.c &
tools/xngen src/f32-vrnd/avx.c.in -D OP=RNDNE -D BATCH_TILE=16 -o src/f32-vrnd/gen/vrndne-avx-x16.c &
tools/xngen src/f32-vrnd/avx.c.in -D OP=RNDZ  -D BATCH_TILE=8  -o src/f32-vrnd/gen/vrndz-avx-x8.c &
tools/xngen src/f32-vrnd/avx.c.in -D OP=RNDZ  -D BATCH_TILE=16 -o src/f32-vrnd/gen/vrndz-avx-x16.c &
tools/xngen src/f32-vrnd/avx.c.in -D OP=RNDU  -D BATCH_TILE=8  -o src/f32-vrnd/gen/vrndu-avx-x8.c &
tools/xngen src/f32-vrnd/avx.c.in -D OP=RNDU  -D BATCH_TILE=16 -o src/f32-vrnd/gen/vrndu-avx-x16.c &
tools/xngen src/f32-vrnd/avx.c.in -D OP=RNDD  -D BATCH_TILE=8  -o src/f32-vrnd/gen/vrndd-avx-x8.c &
tools/xngen src/f32-vrnd/avx.c.in -D OP=RNDD  -D BATCH_TILE=16 -o src/f32-vrnd/gen/vrndd-avx-x16.c &

################################# x86 512-bit #################################
tools/xngen src/f32-vrnd/avx512f.c.in -D OP=RNDNE -D BATCH_TILE=16 -o src/f32-vrnd/gen/vrndne-avx512f-x16.c &
tools/xngen src/f32-vrnd/avx512f.c.in -D OP=RNDNE -D BATCH_TILE=32 -o src/f32-vrnd/gen/vrndne-avx512f-x32.c &
tools/xngen src/f32-vrnd/avx512f.c.in -D OP=RNDZ  -D BATCH_TILE=16 -o src/f32-vrnd/gen/vrndz-avx512f-x16.c &
tools/xngen src/f32-vrnd/avx512f.c.in -D OP=RNDZ  -D BATCH_TILE=32 -o src/f32-vrnd/gen/vrndz-avx512f-x32.c &
tools/xngen src/f32-vrnd/avx512f.c.in -D OP=RNDU  -D BATCH_TILE=16 -o src/f32-vrnd/gen/vrndu-avx512f-x16.c &
tools/xngen src/f32-vrnd/avx512f.c.in -D OP=RNDU  -D BATCH_TILE=32 -o src/f32-vrnd/gen/vrndu-avx512f-x32.c &
tools/xngen src/f32-vrnd/avx512f.c.in -D OP=RNDD  -D BATCH_TILE=16 -o src/f32-vrnd/gen/vrndd-avx512f-x16.c &
tools/xngen src/f32-vrnd/avx512f.c.in -D OP=RNDD  -D BATCH_TILE=32 -o src/f32-vrnd/gen/vrndd-avx512f-x32.c &

################################## Unit tests #################################
tools/generate-vunary-test.py --spec test/f32-vrndne.yaml --output test/f32-vrndne.cc &
tools/generate-vunary-test.py --spec test/f32-vrndz.yaml  --output test/f32-vrndz.cc &
tools/generate-vunary-test.py --spec test/f32-vrndu.yaml  --output test/f32-vrndu.cc &
tools/generate-vunary-test.py --spec test/f32-vrndd.yaml  --output test/f32-vrndd.cc &

wait
