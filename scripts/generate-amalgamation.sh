#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Scalar microkernels
tools/amalgamate-microkernels.py -s PROD_SCALAR_MICROKERNEL_SRCS -o src/amalgam/gen/scalar.c &
tools/amalgamate-microkernels.py -s PROD_SCALAR_AARCH32_MICROKERNEL_SRCS -o src/amalgam/gen/scalar-aarch32.c &
tools/amalgamate-microkernels.py -s PROD_SCALAR_WASM_MICROKERNEL_SRCS -o src/amalgam/gen/scalar-wasm.c &
tools/amalgamate-microkernels.py -s PROD_SCALAR_RISCV_MICROKERNEL_SRCS -o src/amalgam/gen/scalar-riscv.c &

# Wasm microkernels
tools/amalgamate-microkernels.py -s PROD_WASM_MICROKERNEL_SRCS -o src/amalgam/gen/wasm.c &
tools/amalgamate-microkernels.py -i wasm_simd128.h -s PROD_WASMSIMD_MICROKERNEL_SRCS -o src/amalgam/gen/wasmsimd.c &
tools/amalgamate-microkernels.py -i wasm_simd128.h -s PROD_WASMRELAXEDSIMD_MICROKERNEL_SRCS -o src/amalgam/gen/wasmrelaxedsimd.c &

# x86/x86-64 microkernels
tools/amalgamate-microkernels.py -i immintrin.h -s PROD_SSE_MICROKERNEL_SRCS -o src/amalgam/gen/sse.c &
tools/amalgamate-microkernels.py -i immintrin.h -s PROD_SSE2_MICROKERNEL_SRCS -o src/amalgam/gen/sse2.c &
tools/amalgamate-microkernels.py -i immintrin.h -s PROD_SSSE3_MICROKERNEL_SRCS -o src/amalgam/gen/ssse3.c &
tools/amalgamate-microkernels.py -i immintrin.h -s PROD_SSE41_MICROKERNEL_SRCS -o src/amalgam/gen/sse41.c &
tools/amalgamate-microkernels.py -i immintrin.h -s PROD_AVX_MICROKERNEL_SRCS -o src/amalgam/gen/avx.c &
tools/amalgamate-microkernels.py -i xopintrin.h -s PROD_XOP_MICROKERNEL_SRCS -o src/amalgam/gen/xop.c &
tools/amalgamate-microkernels.py -i immintrin.h -s PROD_FMA3_MICROKERNEL_SRCS -o src/amalgam/gen/fma3.c &
tools/amalgamate-microkernels.py -i immintrin.h -s PROD_AVX2_MICROKERNEL_SRCS -o src/amalgam/gen/avx2.c &
tools/amalgamate-microkernels.py -i immintrin.h -s PROD_F16C_MICROKERNEL_SRCS -o src/amalgam/gen/f16c.c &
tools/amalgamate-microkernels.py -i immintrin.h -s PROD_AVX512F_MICROKERNEL_SRCS -o src/amalgam/gen/avx512f.c &
tools/amalgamate-microkernels.py -i immintrin.h -s PROD_AVX512SKX_MICROKERNEL_SRCS -o src/amalgam/gen/avx512skx.c &
tools/amalgamate-microkernels.py -i immintrin.h -s PROD_AVX512VBMI_MICROKERNEL_SRCS -o src/amalgam/gen/avx512vbmi.c &

# ARM/ARM64 microkernels
tools/amalgamate-microkernels.py -i arm_acle.h -s PROD_ARMSIMD32_MICROKERNEL_SRCS -o src/amalgam/gen/armsimd32.c &
tools/amalgamate-microkernels.py -i arm_fp16.h -s PROD_FP16ARITH_MICROKERNEL_SRCS -o src/amalgam/gen/fp16arith.c &
tools/amalgamate-microkernels.py -i arm_neon.h -s PROD_NEON_MICROKERNEL_SRCS -o src/amalgam/gen/neon.c &
tools/amalgamate-microkernels.py -i arm_neon.h -s PROD_NEONFP16_MICROKERNEL_SRCS -o src/amalgam/gen/neonfp16.c &
tools/amalgamate-microkernels.py -i arm_neon.h -s PROD_NEONFMA_MICROKERNEL_SRCS -o src/amalgam/gen/neonfma.c &
tools/amalgamate-microkernels.py -i arm_neon.h -s PROD_NEON_AARCH64_MICROKERNEL_SRCS -o src/amalgam/gen/neon-aarch64.c &
tools/amalgamate-microkernels.py -i arm_neon.h -s PROD_NEONV8_MICROKERNEL_SRCS -o src/amalgam/gen/neonv8.c &
tools/amalgamate-microkernels.py -i arm_neon.h -s PROD_NEONFP16ARITH_MICROKERNEL_SRCS -o src/amalgam/gen/neonfp16arith.c &
tools/amalgamate-microkernels.py -i arm_neon.h -s PROD_NEONFP16ARITH_AARCH64_MICROKERNEL_SRCS -o src/amalgam/gen/neonfp16arith-aarch64.c &
tools/amalgamate-microkernels.py -i arm_neon.h -s PROD_NEONDOT_MICROKERNEL_SRCS -o src/amalgam/gen/neondot.c &

wait
