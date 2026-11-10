#!/bin/sh
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### Scalar ###################################
tools/xngen src/qs8-avgpool/scalar.c.in -D DATATYPE=QS8 -o src/qs8-avgpool/gen/qs8-avgpool-9p-minmax-fp32-scalar-fmagic-u1.c &

################################## ARM NEON ##################################
tools/xngen src/qs8-avgpool/neon.c.in -D DATATYPE=QS8 -o src/qs8-avgpool/gen/qs8-avgpool-9p-minmax-fp32-neon-u16.c &

#################################### x86 #####################################
tools/xngen src/qs8-avgpool/sse41.c.in -D DATATYPE=QS8 -o src/qs8-avgpool/gen/qs8-avgpool-9p-minmax-fp32-sse41-u16.c &
tools/xngen src/qs8-avgpool/avx2.c.in -D DATATYPE=QS8 -o src/qs8-avgpool/gen/qs8-avgpool-9p-minmax-fp32-avx2-u32.c &
tools/xngen src/qs8-avgpool/avx512skx.c.in -D DATATYPE=QS8 -o src/qs8-avgpool/gen/qs8-avgpool-9p-minmax-fp32-avx512skx-u32.c &

################################## Wasm SIMD #################################
tools/xngen src/qs8-avgpool/wasmsimd.c.in -D DATATYPE=QS8 -o src/qs8-avgpool/gen/qs8-avgpool-9p-minmax-fp32-wasmsimd-u16.c &

################################ RISC-V Vector ###############################
tools/xngen src/qs8-avgpool/rvv.c.in -D DATATYPE=QS8 -o src/qs8-avgpool/gen/qs8-avgpool-9p-minmax-fp32-rvv-u2v.c &

################################# Hexagon HVX ################################
tools/xngen src/qs8-avgpool/hvx.c.in -D DATATYPE=QS8 -o src/qs8-avgpool/gen/qs8-avgpool-9p-minmax-fp32-hvx-u32.c &

wait
