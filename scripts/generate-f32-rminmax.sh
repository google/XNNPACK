#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-neon-x4.c &
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-neon-x8-acc2.c &
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-neon-x12-acc3.c &
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-neon-x16-acc2.c &
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-neon-x16-acc4.c &

tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-neon-x4.c &
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-neon-x8-acc2.c &
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-neon-x12-acc3.c &
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-neon-x16-acc2.c &
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-neon-x16-acc4.c &

tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-neon-x4.c &
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-neon-x8-acc2.c &
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-neon-x12-acc3.c &
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-neon-x16-acc2.c &
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-neon-x16-acc4.c &

################################### x86 SSE ###################################
tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-sse-x4.c &
tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-sse-x8-acc2.c &
tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-sse-x12-acc3.c &
tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-sse-x16-acc2.c &
tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-sse-x16-acc4.c &

tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-sse-x4.c &
tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-sse-x8-acc2.c &
tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-sse-x12-acc3.c &
tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-sse-x16-acc2.c &
tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-sse-x16-acc4.c &

################################## Wasm SIMD ##################################
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D OP=MAX -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rmax-wasmsimd-minmax-x4.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D OP=MAX -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rmax-wasmsimd-minmax-x8-acc2.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D OP=MAX -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rmax-wasmsimd-minmax-x12-acc3.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MAX -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rmax-wasmsimd-minmax-x16-acc2.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D OP=MAX -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rmax-wasmsimd-minmax-x16-acc4.c &

tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D OP=MIN -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rmin-wasmsimd-minmax-x4.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D OP=MIN -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rmin-wasmsimd-minmax-x8-acc2.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D OP=MIN -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rmin-wasmsimd-minmax-x12-acc3.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MIN -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rmin-wasmsimd-minmax-x16-acc2.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D OP=MIN -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rmin-wasmsimd-minmax-x16-acc4.c &

tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D OP=MAX -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rmax-wasmsimd-pminmax-x4.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D OP=MAX -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rmax-wasmsimd-pminmax-x8-acc2.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D OP=MAX -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rmax-wasmsimd-pminmax-x12-acc3.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MAX -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rmax-wasmsimd-pminmax-x16-acc2.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D OP=MAX -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rmax-wasmsimd-pminmax-x16-acc4.c &

tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D OP=MIN -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rmin-wasmsimd-pminmax-x4.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D OP=MIN -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rmin-wasmsimd-pminmax-x8-acc2.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D OP=MIN -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rmin-wasmsimd-pminmax-x12-acc3.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MIN -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rmin-wasmsimd-pminmax-x16-acc2.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D OP=MIN -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rmin-wasmsimd-pminmax-x16-acc4.c &

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=1 -D ACCUMULATORS=1 -D OP=MINMAX -D WASM=0 -o src/f32-rminmax/gen/f32-rminmax-scalar-x1.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=2 -D ACCUMULATORS=2 -D OP=MINMAX -D WASM=0 -o src/f32-rminmax/gen/f32-rminmax-scalar-x2-acc2.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=3 -D ACCUMULATORS=3 -D OP=MINMAX -D WASM=0 -o src/f32-rminmax/gen/f32-rminmax-scalar-x3-acc3.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=2 -D OP=MINMAX -D WASM=0 -o src/f32-rminmax/gen/f32-rminmax-scalar-x4-acc2.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=4 -D OP=MINMAX -D WASM=0 -o src/f32-rminmax/gen/f32-rminmax-scalar-x4-acc4.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=1 -D ACCUMULATORS=1 -D OP=MAX -D WASM=0 -o src/f32-rminmax/gen/f32-rmax-scalar-x1.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=2 -D ACCUMULATORS=2 -D OP=MAX -D WASM=0 -o src/f32-rminmax/gen/f32-rmax-scalar-x2-acc2.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=3 -D ACCUMULATORS=3 -D OP=MAX -D WASM=0 -o src/f32-rminmax/gen/f32-rmax-scalar-x3-acc3.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=2 -D OP=MAX -D WASM=0 -o src/f32-rminmax/gen/f32-rmax-scalar-x4-acc2.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=4 -D OP=MAX -D WASM=0 -o src/f32-rminmax/gen/f32-rmax-scalar-x4-acc4.c &

tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=1 -D ACCUMULATORS=1 -D OP=MIN -D WASM=0 -o src/f32-rminmax/gen/f32-rmin-scalar-x1.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=2 -D ACCUMULATORS=2 -D OP=MIN -D WASM=0 -o src/f32-rminmax/gen/f32-rmin-scalar-x2-acc2.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=3 -D ACCUMULATORS=3 -D OP=MIN -D WASM=0 -o src/f32-rminmax/gen/f32-rmin-scalar-x3-acc3.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=2 -D OP=MIN -D WASM=0 -o src/f32-rminmax/gen/f32-rmin-scalar-x4-acc2.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=4 -D OP=MIN -D WASM=0 -o src/f32-rminmax/gen/f32-rmin-scalar-x4-acc4.c &

### Wasm-specific micro-kernels
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=1 -D ACCUMULATORS=1 -D OP=MAX -D WASM=1 -o src/f32-rminmax/gen/f32-rmax-wasm-x1.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=2 -D ACCUMULATORS=2 -D OP=MAX -D WASM=1 -o src/f32-rminmax/gen/f32-rmax-wasm-x2-acc2.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=3 -D ACCUMULATORS=3 -D OP=MAX -D WASM=1 -o src/f32-rminmax/gen/f32-rmax-wasm-x3-acc3.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=2 -D OP=MAX -D WASM=1 -o src/f32-rminmax/gen/f32-rmax-wasm-x4-acc2.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=4 -D OP=MAX -D WASM=1 -o src/f32-rminmax/gen/f32-rmax-wasm-x4-acc4.c &

tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=1 -D ACCUMULATORS=1 -D OP=MIN -D WASM=1 -o src/f32-rminmax/gen/f32-rmin-wasm-x1.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=2 -D ACCUMULATORS=2 -D OP=MIN -D WASM=1 -o src/f32-rminmax/gen/f32-rmin-wasm-x2-acc2.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=3 -D ACCUMULATORS=3 -D OP=MIN -D WASM=1 -o src/f32-rminmax/gen/f32-rmin-wasm-x3-acc3.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=2 -D OP=MIN -D WASM=1 -o src/f32-rminmax/gen/f32-rmin-wasm-x4-acc2.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=4 -D OP=MIN -D WASM=1 -o src/f32-rminmax/gen/f32-rmin-wasm-x4-acc4.c &

################################## Unit tests #################################
tools/generate-reduce-test.py --tester ReduceMicrokernelTester --spec test/f32-rmax.yaml --output test/f32-rmax.cc &
tools/generate-reduce-test.py --tester ReduceMicrokernelTester --spec test/f32-rmin.yaml --output test/f32-rmin.cc &
tools/generate-reduce-test.py --tester ReduceMicrokernelTester --spec test/f32-rminmax.yaml --output test/f32-rminmax.cc &

wait
