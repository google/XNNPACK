// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <xnnpack/common.h>
#include <xnnpack/config.h>


#if XNN_ARCH_X86
  #define TEST_REQUIRES_X86_SSE \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_X86_SSE
#endif

#if XNN_ARCH_X86
  #define TEST_REQUIRES_X86_SSE2 \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_X86_SSE2
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_SSSE3 \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_x86_ssse3) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_X86_SSSE3
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_SSE41 \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_x86_sse4_1) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_X86_SSE41
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_AVX \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_x86_avx) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_X86_AVX
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_F16C \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_x86_f16c) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_X86_F16C
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_XOP \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_x86_xop) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_X86_XOP
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_FMA3 \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_x86_fma3) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_X86_FMA3
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_AVX2 \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_x86_avx2) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_X86_AVX2
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_AVX512F \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_x86_avx512f) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_X86_AVX512F
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_AVX512SKX \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_x86_avx512skx) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_X86_AVX512SKX
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_AVX512VBMI \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_x86_avx512vbmi) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_X86_AVX512VBMI
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_AVX512VNNI \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_x86_avx512vnni) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_X86_AVX512VNNI
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_AVX512AMX \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_x86_avx512amx) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_X86_AVX512AMX
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_AVX512VNNIGFNI \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_x86_avx512vnnigfni) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_X86_AVX512VNNIGFNI
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_AVXVNNI \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_x86_avxvnni) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_X86_AVXVNNI
#endif

#if XNN_ARCH_ARM
  #define TEST_REQUIRES_ARM_SIMD32 \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_arm_v6) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_ARM_SIMD32
#endif

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  #define TEST_REQUIRES_ARM_FP16_ARITH \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_arm_fp16_arith) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_ARM_FP16_ARITH
#endif

#if XNN_ARCH_ARM
  #define TEST_REQUIRES_ARM_NEON \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_arm_neon) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_ARM_NEON
#endif

#if XNN_ARCH_ARM
  #define TEST_REQUIRES_ARM_NEON_FP16 \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_arm_neon_fp16) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_ARM_NEON_FP16
#endif

#if XNN_ARCH_ARM
  #define TEST_REQUIRES_ARM_NEON_FMA \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_arm_neon_fma) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_ARM_NEON_FMA
#endif

#if XNN_ARCH_ARM
  #define TEST_REQUIRES_ARM_NEON_V8 \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_arm_neon_v8) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_ARM_NEON_V8
#endif

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  #define TEST_REQUIRES_ARM_NEON_FP16_ARITH \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_arm_neon_fp16_arith) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_ARM_NEON_FP16_ARITH
#endif

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  #define TEST_REQUIRES_ARM_NEON_BF16 \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_arm_neon_bf16) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_ARM_NEON_BF16
#endif

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  #define TEST_REQUIRES_ARM_NEON_DOT \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_arm_neon_dot) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_ARM_NEON_DOT
#endif

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  #define TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !(hardware_config->use_arm_neon_dot && hardware_config->use_arm_neon_fp16_arith)) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_ARM_NEON_DOT
#endif

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  #define TEST_REQUIRES_ARM_NEON_I8MM \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_arm_neon_i8mm) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_ARM_NEON_I8MM
#endif

#if XNN_ARCH_RISCV
  #define TEST_REQUIRES_RISCV_VECTOR \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_riscv_vector) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_RISCV_VECTOR
#endif

#if XNN_ARCH_RISCV
  #define TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_riscv_vector_fp16_arith) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH
#endif

#if XNN_ARCH_WASMRELAXEDSIMD
  #define TEST_REQUIRES_WASM_PSHUFB \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_wasm_pshufb) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_WASM_PSHUFB
#endif

#if XNN_ARCH_WASMRELAXEDSIMD
  #define TEST_REQUIRES_WASM_SDOT \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_wasm_sdot) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_WASM_SDOT
#endif

#if XNN_ARCH_WASMRELAXEDSIMD
  #define TEST_REQUIRES_WASM_BLENDVPS \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !hardware_config->use_wasm_blendvps) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_WASM_BLENDVPS
#endif
