#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### Scalar ####################################
tools/xngen src/f16-igemm/scalar.c.in -D MR=1 -D NR=4 -D ACTIVATION=MINMAX -o src/f16-igemm/gen/f16-igemm-1x4-minmax-scalar.c &
tools/xngen src/f16-igemm/scalar.c.in -D MR=4 -D NR=4 -D ACTIVATION=MINMAX -o src/f16-igemm/gen/f16-igemm-4x4-minmax-scalar.c &

########################## ARM NEON with FP16 compute #########################
### LD64 micro-kernels
tools/xngen src/f16-igemm/neonfp16arith-ld64.c.in -D MR=1 -D NR=8  -o src/f16-igemm/gen/f16-igemm-1x8-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-igemm/neonfp16arith-ld64.c.in -D MR=4 -D NR=8  -o src/f16-igemm/gen/f16-igemm-4x8-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-igemm/neonfp16arith-ld64.c.in -D MR=6 -D NR=8  -o src/f16-igemm/gen/f16-igemm-6x8-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-igemm/neonfp16arith-ld64.c.in -D MR=8 -D NR=8  -o src/f16-igemm/gen/f16-igemm-8x8-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-igemm/neonfp16arith-ld64.c.in -D MR=1 -D NR=16 -o src/f16-igemm/gen/f16-igemm-1x16-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-igemm/neonfp16arith-ld64.c.in -D MR=4 -D NR=16 -o src/f16-igemm/gen/f16-igemm-4x16-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-igemm/neonfp16arith-ld64.c.in -D MR=6 -D NR=16 -o src/f16-igemm/gen/f16-igemm-6x16-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-igemm/neonfp16arith-ld64.c.in -D MR=8 -D NR=16 -o src/f16-igemm/gen/f16-igemm-8x16-minmax-neonfp16arith-ld64.c &

################################### x86 AVX2 ###################################
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=1 -D NR=8  -D ISA=avx2 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-1x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=4 -D NR=8  -D ISA=avx2 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-4x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=5 -D NR=8  -D ISA=avx2 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-5x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=6 -D NR=8  -D ISA=avx2 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-6x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=7 -D NR=8  -D ISA=avx2 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-7x8-minmax-avx2-broadcast.c &

tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=1 -D NR=16 -D ISA=avx2 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-1x16-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=3 -D NR=16 -D ISA=avx2 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-3x16-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=4 -D NR=16 -D ISA=avx2 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-4x16-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=5 -D NR=16 -D ISA=avx2 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-5x16-minmax-avx2-broadcast.c &

tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=1 -D NR=8  -D ISA=avx2 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-1x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=4 -D NR=8  -D ISA=avx2 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-4x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=5 -D NR=8  -D ISA=avx2 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-5x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=6 -D NR=8  -D ISA=avx2 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-6x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=7 -D NR=8  -D ISA=avx2 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-7x8-minmax-avx2-broadcast.c &

tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=1 -D NR=16 -D ISA=avx2 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-1x16-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=3 -D NR=16 -D ISA=avx2 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-3x16-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=4 -D NR=16 -D ISA=avx2 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-4x16-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=5 -D NR=16 -D ISA=avx2 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-5x16-minmax-avx2-broadcast.c &

################################### x86 FMA3 ###################################
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=1 -D NR=8  -D ISA=fma3 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-1x8-minmax-fma3-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=4 -D NR=8  -D ISA=fma3 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-4x8-minmax-fma3-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=5 -D NR=8  -D ISA=fma3 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-5x8-minmax-fma3-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=6 -D NR=8  -D ISA=fma3 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-6x8-minmax-fma3-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=7 -D NR=8  -D ISA=fma3 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-7x8-minmax-fma3-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=1 -D NR=16 -D ISA=fma3 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-1x16-minmax-fma3-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=3 -D NR=16 -D ISA=fma3 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-3x16-minmax-fma3-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=4 -D NR=16 -D ISA=fma3 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-4x16-minmax-fma3-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=5 -D NR=16 -D ISA=fma3 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-5x16-minmax-fma3-broadcast.c &

################################### x86 F16C ###################################
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=1 -D NR=8  -D ISA=f16c -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-1x8-minmax-f16c-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=4 -D NR=8  -D ISA=f16c -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-4x8-minmax-f16c-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=5 -D NR=8  -D ISA=f16c -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-5x8-minmax-f16c-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=6 -D NR=8  -D ISA=f16c -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-6x8-minmax-f16c-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=7 -D NR=8  -D ISA=f16c -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-7x8-minmax-f16c-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=1 -D NR=16 -D ISA=f16c -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-1x16-minmax-f16c-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=3 -D NR=16 -D ISA=f16c -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-3x16-minmax-f16c-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=4 -D NR=16 -D ISA=f16c -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-4x16-minmax-f16c-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=5 -D NR=16 -D ISA=f16c -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-5x16-minmax-f16c-broadcast.c &

################################# x86 AVX-512 FP16 #################################
### AVX512FP16+BROADCAST micro-kernels
tools/xngen src/f16-igemm/avx512fp16-broadcast.c.in -D MR=1 -D NR=32 -D DATATYPE=F16 -o src/f16-igemm/gen/f16-igemm-1x32-minmax-avx512fp16-broadcast.c &
tools/xngen src/f16-igemm/avx512fp16-broadcast.c.in -D MR=4 -D NR=32 -D DATATYPE=F16 -o src/f16-igemm/gen/f16-igemm-4x32-minmax-avx512fp16-broadcast.c &
tools/xngen src/f16-igemm/avx512fp16-broadcast.c.in -D MR=5 -D NR=32 -D DATATYPE=F16 -o src/f16-igemm/gen/f16-igemm-5x32-minmax-avx512fp16-broadcast.c &
tools/xngen src/f16-igemm/avx512fp16-broadcast.c.in -D MR=6 -D NR=32 -D DATATYPE=F16 -o src/f16-igemm/gen/f16-igemm-6x32-minmax-avx512fp16-broadcast.c &
tools/xngen src/f16-igemm/avx512fp16-broadcast.c.in -D MR=7 -D NR=32 -D DATATYPE=F16 -o src/f16-igemm/gen/f16-igemm-7x32-minmax-avx512fp16-broadcast.c &
tools/xngen src/f16-igemm/avx512fp16-broadcast.c.in -D MR=8 -D NR=32 -D DATATYPE=F16 -o src/f16-igemm/gen/f16-igemm-8x32-minmax-avx512fp16-broadcast.c &

tools/xngen src/f16-igemm/avx512fp16-broadcast.c.in -D MR=1 -D NR=64 -D DATATYPE=F16 -o src/f16-igemm/gen/f16-igemm-1x64-minmax-avx512fp16-broadcast.c &
tools/xngen src/f16-igemm/avx512fp16-broadcast.c.in -D MR=4 -D NR=64 -D DATATYPE=F16 -o src/f16-igemm/gen/f16-igemm-4x64-minmax-avx512fp16-broadcast.c &
tools/xngen src/f16-igemm/avx512fp16-broadcast.c.in -D MR=5 -D NR=64 -D DATATYPE=F16 -o src/f16-igemm/gen/f16-igemm-5x64-minmax-avx512fp16-broadcast.c &
tools/xngen src/f16-igemm/avx512fp16-broadcast.c.in -D MR=6 -D NR=64 -D DATATYPE=F16 -o src/f16-igemm/gen/f16-igemm-6x64-minmax-avx512fp16-broadcast.c &
tools/xngen src/f16-igemm/avx512fp16-broadcast.c.in -D MR=7 -D NR=64 -D DATATYPE=F16 -o src/f16-igemm/gen/f16-igemm-7x64-minmax-avx512fp16-broadcast.c &
tools/xngen src/f16-igemm/avx512fp16-broadcast.c.in -D MR=8 -D NR=64 -D DATATYPE=F16 -o src/f16-igemm/gen/f16-igemm-8x64-minmax-avx512fp16-broadcast.c &

################################# x86 AVX512 AMX FP16 ################################
### C2 micro-kernels
tools/xngen src/f16-igemm/c2-avx512amxfp16.c.in -D PREFETCH=0 -D MR=1  -D NR=16 -D DATATYPE=f16 -o src/f16-igemm/gen/f16-igemm-1x16c2-minmax-avx512amxfp16.c &
tools/xngen src/f16-igemm/c2-avx512amxfp16.c.in -D PREFETCH=0 -D MR=16 -D NR=16 -D DATATYPE=f16 -o src/f16-igemm/gen/f16-igemm-16x16c2-minmax-avx512amxfp16.c &
tools/xngen src/f16-igemm/c2-avx512amxfp16.c.in -D PREFETCH=1 -D MR=16 -D NR=16 -D DATATYPE=f16 -o src/f16-igemm/gen/f16-igemm-16x16c2-minmax-avx512amxfp16-prfm.c &

tools/xngen src/f16-igemm/c2-avx512amxfp16.c.in -D PREFETCH=0 -D MR=1  -D NR=32 -D DATATYPE=f16 -o src/f16-igemm/gen/f16-igemm-1x32c2-minmax-avx512amxfp16.c &
tools/xngen src/f16-igemm/c2-avx512amxfp16.c.in -D PREFETCH=0 -D MR=16 -D NR=32 -D DATATYPE=f16 -o src/f16-igemm/gen/f16-igemm-16x32c2-minmax-avx512amxfp16.c &
tools/xngen src/f16-igemm/c2-avx512amxfp16.c.in -D PREFETCH=1 -D MR=16 -D NR=32 -D DATATYPE=f16 -o src/f16-igemm/gen/f16-igemm-16x32c2-minmax-avx512amxfp16-prfm.c &

tools/xngen src/f16-igemm/c2-avx512amxfp16.c.in -D PREFETCH=0 -D MR=1  -D NR=64 -D DATATYPE=f16 -o src/f16-igemm/gen/f16-igemm-1x64c2-minmax-avx512amxfp16.c &
tools/xngen src/f16-igemm/c2-avx512amxfp16.c.in -D PREFETCH=0 -D MR=16 -D NR=64 -D DATATYPE=f16 -o src/f16-igemm/gen/f16-igemm-16x64c2-minmax-avx512amxfp16.c &
tools/xngen src/f16-igemm/c2-avx512amxfp16.c.in -D PREFETCH=1 -D MR=16 -D NR=64 -D DATATYPE=f16 -o src/f16-igemm/gen/f16-igemm-16x64c2-minmax-avx512amxfp16-prfm.c &

################################# x86 AVX-512 SKX #################################
tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=1 -D NR=16 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-1x16-minmax-avx512skx-broadcast.c &
tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=4 -D NR=16 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-4x16-minmax-avx512skx-broadcast.c &
tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=5 -D NR=16 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-5x16-minmax-avx512skx-broadcast.c &
tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=6 -D NR=16 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-6x16-minmax-avx512skx-broadcast.c &
tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=7 -D NR=16 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-7x16-minmax-avx512skx-broadcast.c &
tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=8 -D NR=16 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-8x16-minmax-avx512skx-broadcast.c &

tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=1 -D NR=32 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-1x32-minmax-avx512skx-broadcast.c &
tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=4 -D NR=32 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-4x32-minmax-avx512skx-broadcast.c &
tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=5 -D NR=32 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-5x32-minmax-avx512skx-broadcast.c &
tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=6 -D NR=32 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-6x32-minmax-avx512skx-broadcast.c &
tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=7 -D NR=32 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-7x32-minmax-avx512skx-broadcast.c &
tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=8 -D NR=32 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-8x32-minmax-avx512skx-broadcast.c &

tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=1 -D NR=16 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-1x16-minmax-avx512skx-broadcast.c &
tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=4 -D NR=16 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-4x16-minmax-avx512skx-broadcast.c &
tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=5 -D NR=16 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-5x16-minmax-avx512skx-broadcast.c &
tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=6 -D NR=16 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-6x16-minmax-avx512skx-broadcast.c &
tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=7 -D NR=16 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-7x16-minmax-avx512skx-broadcast.c &
tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=8 -D NR=16 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-8x16-minmax-avx512skx-broadcast.c &

tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=1 -D NR=32 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-1x32-minmax-avx512skx-broadcast.c &
tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=4 -D NR=32 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-4x32-minmax-avx512skx-broadcast.c &
tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=5 -D NR=32 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-5x32-minmax-avx512skx-broadcast.c &
tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=6 -D NR=32 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-6x32-minmax-avx512skx-broadcast.c &
tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=7 -D NR=32 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-7x32-minmax-avx512skx-broadcast.c &
tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=8 -D NR=32 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-8x32-minmax-avx512skx-broadcast.c &

tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=1 -D NR=64 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-1x64-minmax-avx512skx-broadcast.c &
tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=4 -D NR=64 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-4x64-minmax-avx512skx-broadcast.c &
tools/xngen src/f16-igemm/avx512skx-broadcast.c.in -D MR=5 -D NR=64 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-5x64-minmax-avx512skx-broadcast.c &

################################ RISC-V Vector #####################################
tools/xngen src/f16-igemm/MRxNRv-rvv.c.in -D MR=1 -D NR=m4 -o src/f16-igemm/gen/f16-igemm-1x4v-minmax-rvvfp16arith.c &
tools/xngen src/f16-igemm/MRxNRv-rvv.c.in -D MR=4 -D NR=m4 -o src/f16-igemm/gen/f16-igemm-4x4v-minmax-rvvfp16arith.c &
tools/xngen src/f16-igemm/MRxNRv-rvv.c.in -D MR=7 -D NR=m4 -o src/f16-igemm/gen/f16-igemm-7x4v-minmax-rvvfp16arith.c &

################################# WASM FP16 #################################
### WASM FP16X8 micro-kernels
tools/xngen src/f16-igemm/wasmrelaxedsimdfp16-splat.c.in -D MR=1 -D NR=8 -o src/f16-igemm/gen/f16-igemm-1x8-minmax-wasmrelaxedsimdfp16-splat.c &
tools/xngen src/f16-igemm/wasmrelaxedsimdfp16-splat.c.in -D MR=4 -D NR=8 -o src/f16-igemm/gen/f16-igemm-4x8-minmax-wasmrelaxedsimdfp16-splat.c &
tools/xngen src/f16-igemm/wasmrelaxedsimdfp16-splat.c.in -D MR=6 -D NR=8 -o src/f16-igemm/gen/f16-igemm-6x8-minmax-wasmrelaxedsimdfp16-splat.c &
tools/xngen src/f16-igemm/wasmrelaxedsimdfp16-splat.c.in -D MR=8 -D NR=8 -o src/f16-igemm/gen/f16-igemm-8x8-minmax-wasmrelaxedsimdfp16-splat.c &

tools/xngen src/f16-igemm/wasmrelaxedsimdfp16-splat.c.in -D MR=1 -D NR=16 -o src/f16-igemm/gen/f16-igemm-1x16-minmax-wasmrelaxedsimdfp16-splat.c &
tools/xngen src/f16-igemm/wasmrelaxedsimdfp16-splat.c.in -D MR=4 -D NR=16 -o src/f16-igemm/gen/f16-igemm-4x16-minmax-wasmrelaxedsimdfp16-splat.c &
tools/xngen src/f16-igemm/wasmrelaxedsimdfp16-splat.c.in -D MR=6 -D NR=16 -o src/f16-igemm/gen/f16-igemm-6x16-minmax-wasmrelaxedsimdfp16-splat.c &
tools/xngen src/f16-igemm/wasmrelaxedsimdfp16-splat.c.in -D MR=8 -D NR=16 -o src/f16-igemm/gen/f16-igemm-8x16-minmax-wasmrelaxedsimdfp16-splat.c &

wait
