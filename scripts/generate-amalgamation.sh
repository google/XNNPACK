#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Scalar microkernels
tools/amalgamate-microkernels.py -s PROD_SCALAR_MICROKERNEL_SRCS -o src/amalgam/scalar.c &
tools/amalgamate-microkernels.py -s PROD_SCALAR_AARCH32_MICROKERNEL_SRCS -o src/amalgam/scalar-aarch32.c &
tools/amalgamate-microkernels.py -s PROD_SCALAR_WASM_MICROKERNEL_SRCS -o src/amalgam/scalar-wasm.c &
tools/amalgamate-microkernels.py -s PROD_SCALAR_RISCV_MICROKERNEL_SRCS -o src/amalgam/scalar-riscv.c &

# Wasm microkernels
tools/amalgamate-microkernels.py -s PROD_WASM_MICROKERNEL_SRCS -o src/amalgam/wasm.c &
tools/amalgamate-microkernels.py -i wasm_simd128.h -s PROD_WASMSIMD_MICROKERNEL_SRCS -o src/amalgam/wasmsimd.c &
tools/amalgamate-microkernels.py -i wasm_simd128.h -s PROD_WASMRELAXEDSIMD_MICROKERNEL_SRCS -o src/amalgam/wasmrelaxedsimd.c &

# x86/x86-64 microkernels
tools/amalgamate-microkernels.py -i immintrin.h -s PROD_SSE_MICROKERNEL_SRCS -o src/amalgam/sse.c &
tools/amalgamate-microkernels.py -i immintrin.h -s PROD_SSE2_MICROKERNEL_SRCS -o src/amalgam/sse2.c &
tools/amalgamate-microkernels.py -i immintrin.h -s PROD_SSSE3_MICROKERNEL_SRCS -o src/amalgam/ssse3.c &
tools/amalgamate-microkernels.py -i immintrin.h -s PROD_SSE41_MICROKERNEL_SRCS -o src/amalgam/sse41.c &
tools/amalgamate-microkernels.py -i immintrin.h -s PROD_AVX_MICROKERNEL_SRCS -o src/amalgam/avx.c &
tools/amalgamate-microkernels.py -i xopintrin.h -s PROD_XOP_MICROKERNEL_SRCS -o src/amalgam/xop.c &
tools/amalgamate-microkernels.py -i immintrin.h -s PROD_FMA3_MICROKERNEL_SRCS -o src/amalgam/fma3.c &
tools/amalgamate-microkernels.py -i immintrin.h -s PROD_AVX2_MICROKERNEL_SRCS -o src/amalgam/avx2.c &
tools/amalgamate-microkernels.py -i immintrin.h -s PROD_F16C_MICROKERNEL_SRCS -o src/amalgam/f16c.c &
tools/amalgamate-microkernels.py -i immintrin.h -s PROD_AVX512F_MICROKERNEL_SRCS -o src/amalgam/avx512f.c &
tools/amalgamate-microkernels.py -i immintrin.h -s PROD_AVX512SKX_MICROKERNEL_SRCS -o src/amalgam/avx512skx.c &
tools/amalgamate-microkernels.py -i immintrin.h -s PROD_AVX512VBMI_MICROKERNEL_SRCS -o src/amalgam/avx512vbmi.c &

# ARM/ARM64 microkernels
tools/amalgamate-microkernels.py -i arm_acle.h -s PROD_ARMSIMD32_MICROKERNEL_SRCS -o src/amalgam/armsimd32.c &
tools/amalgamate-microkernels.py -i arm_fp16.h -s PROD_FP16ARITH_MICROKERNEL_SRCS -o src/amalgam/fp16arith.c &
tools/amalgamate-microkernels.py -i arm_neon.h -s PROD_NEON_MICROKERNEL_SRCS -o src/amalgam/neon.c &
tools/amalgamate-microkernels.py -i arm_neon.h -s PROD_NEONFP16_MICROKERNEL_SRCS -o src/amalgam/neonfp16.c &
tools/amalgamate-microkernels.py -i arm_neon.h -s PROD_NEONFMA_MICROKERNEL_SRCS -o src/amalgam/neonfma.c &
tools/amalgamate-microkernels.py -i arm_neon.h -s PROD_NEON_AARCH64_MICROKERNEL_SRCS -o src/amalgam/neon-aarch64.c &
tools/amalgamate-microkernels.py -i arm_neon.h -s PROD_NEONV8_MICROKERNEL_SRCS -o src/amalgam/neonv8.c &
tools/amalgamate-microkernels.py -i arm_neon.h -s PROD_NEONFP16ARITH_MICROKERNEL_SRCS -o src/amalgam/neonfp16arith.c &
tools/amalgamate-microkernels.py -i arm_neon.h -s PROD_NEONFP16ARITH_AARCH64_MICROKERNEL_SRCS -o src/amalgam/neonfp16arith-aarch64.c &
tools/amalgamate-microkernels.py -i arm_neon.h -s PROD_NEONDOT_MICROKERNEL_SRCS -o src/amalgam/neondot.c &

wait
