// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "xnnpack/common.h"
#include "xnnpack/hardware-config.h"


#define XNN_TEST_HWCONFIG_FLAG_NONE() []() -> bool { \
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
    return hardware_config != nullptr; \
}()

#define XNN_TEST_HWCONFIG_FLAG(FLAG) []() -> bool { \
    const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
    return hardware_config != nullptr && hardware_config->FLAG; \
}()

inline size_t get_batch_scale(size_t element_size) {
#if XNN_ARCH_RISCV
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  return hardware_config ? std::max<size_t>(1, hardware_config->vlenb / element_size) : 1;
#else
  return 1;
#endif
}

template <typename T>
size_t get_batch_scale() {
  return get_batch_scale(sizeof(T));
}

#define TEST_REQUIRES_HWCONFIG_FLAG_NONE() \
  do { \
    if (!XNN_TEST_HWCONFIG_FLAG_NONE()) { \
      GTEST_SKIP(); \
    } \
  } while (0)

#define TEST_REQUIRES_HWCONFIG_FLAG(FLAG) \
  do {                                    \
    if (!XNN_TEST_HWCONFIG_FLAG(FLAG)) {  \
      GTEST_SKIP();                       \
    }                                     \
  } while (0)

#define TEST_REQUIRES_ARCH_FLAGS(FLAGS)                       \
  do {                                                        \
    const struct xnn_hardware_config* hardware_config =       \
        xnn_init_hardware_config();                           \
    if (hardware_config == nullptr ||                         \
        (hardware_config->arch_flags & (FLAGS)) != (FLAGS)) { \
      GTEST_SKIP();                                           \
    }                                                         \
  } while (0)

#if XNN_ARCH_X86
  #define TEST_REQUIRES_X86_SSE_VALUE XNN_TEST_HWCONFIG_FLAG_NONE()
#else
  #define TEST_REQUIRES_X86_SSE_VALUE (false)
#endif
#define TEST_REQUIRES_X86_SSE TEST_REQUIRES_HWCONFIG_FLAG_NONE()

#if XNN_ARCH_X86
  #define TEST_REQUIRES_X86_SSE2_VALUE XNN_TEST_HWCONFIG_FLAG_NONE()
#else
  #define TEST_REQUIRES_X86_SSE2_VALUE (false)
#endif
#define TEST_REQUIRES_X86_SSE2 TEST_REQUIRES_HWCONFIG_FLAG_NONE()

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_SSSE3_VALUE XNN_TEST_HWCONFIG_FLAG(use_x86_ssse3)
  #define TEST_REQUIRES_X86_SSSE3 TEST_REQUIRES_HWCONFIG_FLAG(use_x86_ssse3)
#else
  #define TEST_REQUIRES_X86_SSSE3_VALUE (false)
  #define TEST_REQUIRES_X86_SSSE3 do {} while (0)
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_SSE41_VALUE XNN_TEST_HWCONFIG_FLAG(use_x86_sse4_1)
  #define TEST_REQUIRES_X86_SSE41 TEST_REQUIRES_HWCONFIG_FLAG(use_x86_sse4_1)
#else
  #define TEST_REQUIRES_X86_SSE41_VALUE (false)
  #define TEST_REQUIRES_X86_SSE41 do {} while (0)
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_AVX_VALUE XNN_TEST_HWCONFIG_FLAG(use_x86_avx)
  #define TEST_REQUIRES_X86_AVX TEST_REQUIRES_HWCONFIG_FLAG(use_x86_avx)
#else
  #define TEST_REQUIRES_X86_AVX_VALUE (false)
  #define TEST_REQUIRES_X86_AVX do {} while (0)
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_F16C_VALUE XNN_TEST_HWCONFIG_FLAG(use_x86_f16c)
  #define TEST_REQUIRES_X86_F16C TEST_REQUIRES_HWCONFIG_FLAG(use_x86_f16c)
#else
  #define TEST_REQUIRES_X86_F16C_VALUE (false)
  #define TEST_REQUIRES_X86_F16C do {} while (0)
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_FMA3_VALUE XNN_TEST_HWCONFIG_FLAG(use_x86_fma3)
  #define TEST_REQUIRES_X86_FMA3 TEST_REQUIRES_HWCONFIG_FLAG(use_x86_fma3)
#else
  #define TEST_REQUIRES_X86_FMA3_VALUE (false)
  #define TEST_REQUIRES_X86_FMA3 do {} while (0)
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_AVX2_VALUE XNN_TEST_HWCONFIG_FLAG(use_x86_avx2)
  #define TEST_REQUIRES_X86_AVX2 TEST_REQUIRES_HWCONFIG_FLAG(use_x86_avx2)
#else
  #define TEST_REQUIRES_X86_AVX2_VALUE (false)
  #define TEST_REQUIRES_X86_AVX2 do {} while (0)
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_AVX512F_VALUE XNN_TEST_HWCONFIG_FLAG(use_x86_avx512f)
  #define TEST_REQUIRES_X86_AVX512F TEST_REQUIRES_HWCONFIG_FLAG(use_x86_avx512f)
#else
  #define TEST_REQUIRES_X86_AVX512F_VALUE (false)
  #define TEST_REQUIRES_X86_AVX512F do {} while (0)
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_AVX512SKX_VALUE XNN_TEST_HWCONFIG_FLAG(use_x86_avx512skx)
  #define TEST_REQUIRES_X86_AVX512SKX TEST_REQUIRES_HWCONFIG_FLAG(use_x86_avx512skx)
#else
  #define TEST_REQUIRES_X86_AVX512SKX_VALUE (false)
  #define TEST_REQUIRES_X86_AVX512SKX do {} while (0)
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_AVX512VBMI_VALUE XNN_TEST_HWCONFIG_FLAG(use_x86_avx512vbmi)
  #define TEST_REQUIRES_X86_AVX512VBMI TEST_REQUIRES_HWCONFIG_FLAG(use_x86_avx512vbmi)
#else
  #define TEST_REQUIRES_X86_AVX512VBMI_VALUE (false)
  #define TEST_REQUIRES_X86_AVX512VBMI do {} while (0)
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_AVX512VNNI_VALUE XNN_TEST_HWCONFIG_FLAG(use_x86_avx512vnni)
  #define TEST_REQUIRES_X86_AVX512VNNI TEST_REQUIRES_HWCONFIG_FLAG(use_x86_avx512vnni)
#else
  #define TEST_REQUIRES_X86_AVX512VNNI_VALUE (false)
  #define TEST_REQUIRES_X86_AVX512VNNI do {} while (0)
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_AVX512AMX_VALUE XNN_TEST_HWCONFIG_FLAG(use_x86_avx512amx)
  #define TEST_REQUIRES_X86_AVX512AMX TEST_REQUIRES_HWCONFIG_FLAG(use_x86_avx512amx)
#else
  #define TEST_REQUIRES_X86_AVX512AMX_VALUE (false)
  #define TEST_REQUIRES_X86_AVX512AMX do {} while (0)
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_AVX512FP16_VALUE XNN_TEST_HWCONFIG_FLAG(use_x86_avx512fp16)
  #define TEST_REQUIRES_X86_AVX512FP16 TEST_REQUIRES_HWCONFIG_FLAG(use_x86_avx512fp16)
#else
  #define TEST_REQUIRES_X86_AVX512FP16_VALUE (false)
  #define TEST_REQUIRES_X86_AVX512FP16 do {} while (0)
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_AVX512VNNIGFNI_VALUE XNN_TEST_HWCONFIG_FLAG(use_x86_avx512vnnigfni)
  #define TEST_REQUIRES_X86_AVX512VNNIGFNI TEST_REQUIRES_HWCONFIG_FLAG(use_x86_avx512vnnigfni)
#else
  #define TEST_REQUIRES_X86_AVX512VNNIGFNI_VALUE (false)
  #define TEST_REQUIRES_X86_AVX512VNNIGFNI do {} while (0)
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_AVXVNNI_VALUE XNN_TEST_HWCONFIG_FLAG(use_x86_avxvnni)
  #define TEST_REQUIRES_X86_AVXVNNI TEST_REQUIRES_HWCONFIG_FLAG(use_x86_avxvnni)
#else
  #define TEST_REQUIRES_X86_AVXVNNI_VALUE (false)
  #define TEST_REQUIRES_X86_AVXVNNI do {} while (0)
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_AVXVNNIINT8_VALUE XNN_TEST_HWCONFIG_FLAG(use_x86_avxvnniint8)
  #define TEST_REQUIRES_X86_AVXVNNIINT8 TEST_REQUIRES_HWCONFIG_FLAG(use_x86_avxvnni)
#else
  #define TEST_REQUIRES_X86_AVXVNNIINT8_VALUE (false)
  #define TEST_REQUIRES_X86_AVXVNNIINT8 do {} while (0)
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_AVX256SKX_VALUE XNN_TEST_HWCONFIG_FLAG(use_x86_avx256skx)
  #define TEST_REQUIRES_X86_AVX256SKX TEST_REQUIRES_HWCONFIG_FLAG(use_x86_avx256skx)
#else
  #define TEST_REQUIRES_X86_AVX256SKX_VALUE (false)
  #define TEST_REQUIRES_X86_AVX256SKX do {} while (0)
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_AVX256VNNI_VALUE XNN_TEST_HWCONFIG_FLAG(use_x86_avx256vnni)
  #define TEST_REQUIRES_X86_AVX256VNNI TEST_REQUIRES_HWCONFIG_FLAG(use_x86_avx256vnni)
#else
  #define TEST_REQUIRES_X86_AVX256VNNI_VALUE (false)
  #define TEST_REQUIRES_X86_AVX256VNNI do {} while (0)
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  #define TEST_REQUIRES_X86_AVX256VNNIGFNI_VALUE XNN_TEST_HWCONFIG_FLAG(use_x86_avx256vnnigfni)
  #define TEST_REQUIRES_X86_AVX256VNNIGFNI TEST_REQUIRES_HWCONFIG_FLAG(use_x86_avx256vnnigfni)
#else
  #define TEST_REQUIRES_X86_AVX256VNNIGFNI_VALUE (false)
  #define TEST_REQUIRES_X86_AVX256VNNIGFNI do {} while (0)
#endif

#if XNN_ARCH_HEXAGON
  #define TEST_REQUIRES_HVX_VALUE XNN_TEST_HWCONFIG_FLAG(use_hvx)
  #define TEST_REQUIRES_HVX TEST_REQUIRES_HWCONFIG_FLAG(use_hvx)
#else
  #define TEST_REQUIRES_HVX_VALUE (false)
  #define TEST_REQUIRES_HVX do {} while (0)
#endif

#if XNN_ARCH_ARM
  #define TEST_REQUIRES_ARM_SIMD32_VALUE XNN_TEST_HWCONFIG_FLAG(use_arm_v6)
  #define TEST_REQUIRES_ARM_SIMD32 TEST_REQUIRES_HWCONFIG_FLAG(use_arm_v6)
#else
  #define TEST_REQUIRES_ARM_SIMD32_VALUE (false)
  #define TEST_REQUIRES_ARM_SIMD32 do {} while (0)
#endif

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  #define TEST_REQUIRES_ARM_FP16_ARITH_VALUE XNN_TEST_HWCONFIG_FLAG(use_arm_fp16_arith)
  #define TEST_REQUIRES_ARM_FP16_ARITH TEST_REQUIRES_HWCONFIG_FLAG(use_arm_fp16_arith)
#else
  #define TEST_REQUIRES_ARM_FP16_ARITH_VALUE (false)
  #define TEST_REQUIRES_ARM_FP16_ARITH do {} while (0)
#endif

#if XNN_ARCH_ARM
  #define TEST_REQUIRES_ARM_NEON_VALUE XNN_TEST_HWCONFIG_FLAG(use_arm_neon)
  #define TEST_REQUIRES_ARM_NEON TEST_REQUIRES_HWCONFIG_FLAG(use_arm_neon)
#else
  #define TEST_REQUIRES_ARM_NEON_VALUE (false)
  #define TEST_REQUIRES_ARM_NEON do {} while (0)
#endif

#if XNN_ARCH_ARM
  #define TEST_REQUIRES_ARM_NEON_FP16_VALUE XNN_TEST_HWCONFIG_FLAG(use_arm_neon_fp16)
  #define TEST_REQUIRES_ARM_NEON_FP16 TEST_REQUIRES_HWCONFIG_FLAG(use_arm_neon_fp16)
#else
  #define TEST_REQUIRES_ARM_NEON_FP16_VALUE (false)
  #define TEST_REQUIRES_ARM_NEON_FP16 do {} while (0)
#endif

#if XNN_ARCH_ARM
  #define TEST_REQUIRES_ARM_NEON_FMA_VALUE XNN_TEST_HWCONFIG_FLAG(use_arm_neon_fma)
  #define TEST_REQUIRES_ARM_NEON_FMA TEST_REQUIRES_HWCONFIG_FLAG(use_arm_neon_fma)
#else
  #define TEST_REQUIRES_ARM_NEON_FMA_VALUE (false)
  #define TEST_REQUIRES_ARM_NEON_FMA do {} while (0)
#endif

#if XNN_ARCH_ARM
  #define TEST_REQUIRES_ARM_NEON_V8_VALUE XNN_TEST_HWCONFIG_FLAG(use_arm_neon_v8)
  #define TEST_REQUIRES_ARM_NEON_V8 TEST_REQUIRES_HWCONFIG_FLAG(use_arm_neon_v8)
#else
  #define TEST_REQUIRES_ARM_NEON_V8_VALUE (false)
  #define TEST_REQUIRES_ARM_NEON_V8 do {} while (0)
#endif

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  #define TEST_REQUIRES_ARM_NEON_FP16_ARITH_VALUE XNN_TEST_HWCONFIG_FLAG(use_arm_neon_fp16_arith)
  #define TEST_REQUIRES_ARM_NEON_FP16_ARITH TEST_REQUIRES_HWCONFIG_FLAG(use_arm_neon_fp16_arith)
#else
  #define TEST_REQUIRES_ARM_NEON_FP16_ARITH_VALUE (false)
  #define TEST_REQUIRES_ARM_NEON_FP16_ARITH do {} while (0)
#endif

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  #define TEST_REQUIRES_ARM_NEON_BF16_VALUE XNN_TEST_HWCONFIG_FLAG(use_arm_neon_bf16)
  #define TEST_REQUIRES_ARM_NEON_BF16 TEST_REQUIRES_HWCONFIG_FLAG(use_arm_neon_bf16)
#else
  #define TEST_REQUIRES_ARM_NEON_BF16_VALUE (false)
  #define TEST_REQUIRES_ARM_NEON_BF16 do {} while (0)
#endif

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  #define TEST_REQUIRES_ARM_NEON_DOT_VALUE XNN_TEST_HWCONFIG_FLAG(use_arm_neon_dot)
  #define TEST_REQUIRES_ARM_NEON_DOT TEST_REQUIRES_HWCONFIG_FLAG(use_arm_neon_dot)
#else
  #define TEST_REQUIRES_ARM_NEON_DOT_VALUE (false)
  #define TEST_REQUIRES_ARM_NEON_DOT do {} while (0)
#endif

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  #define TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH_VALUE \
    []() -> bool { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      return hardware_config != nullptr && hardware_config->use_arm_neon_dot && hardware_config->use_arm_neon_fp16_arith; \
  }()
  #define TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH \
    do { \
      const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config(); \
      if (hardware_config == nullptr || !(hardware_config->use_arm_neon_dot && hardware_config->use_arm_neon_fp16_arith)) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#else
  #define TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH_VALUE (false)
  #define TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH do {} while (0)
#endif

#if XNN_ARCH_ARM64
  #define TEST_REQUIRES_ARM_NEON_I8MM_VALUE XNN_TEST_HWCONFIG_FLAG(use_arm_neon_i8mm)
  #define TEST_REQUIRES_ARM_NEON_I8MM TEST_REQUIRES_HWCONFIG_FLAG(use_arm_neon_i8mm)
#else
  #define TEST_REQUIRES_ARM_NEON_I8MM_VALUE (false)
  #define TEST_REQUIRES_ARM_NEON_I8MM do {} while (0)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_RISCV
  #define TEST_REQUIRES_RISCV_VECTOR_VALUE XNN_TEST_HWCONFIG_FLAG(use_riscv_vector)
  #define TEST_REQUIRES_RISCV_VECTOR TEST_REQUIRES_HWCONFIG_FLAG(use_riscv_vector)
#else
  #define TEST_REQUIRES_RISCV_VECTOR_VALUE (false)
  #define TEST_REQUIRES_RISCV_VECTOR do {} while (0)
#endif

#if XNN_ARCH_RISCV
  #define TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH_VALUE XNN_TEST_HWCONFIG_FLAG(use_riscv_vector_fp16_arith)
  #define TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH TEST_REQUIRES_HWCONFIG_FLAG(use_riscv_vector_fp16_arith)
#else
  #define TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH_VALUE (false)
  #define TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH do {} while (0)
#endif

#if XNN_ARCH_WASMRELAXEDSIMD
  #define TEST_REQUIRES_WASM_PSHUFB_VALUE XNN_TEST_HWCONFIG_FLAG(use_wasm_pshufb)
  #define TEST_REQUIRES_WASM_PSHUFB TEST_REQUIRES_HWCONFIG_FLAG(use_wasm_pshufb)
#else
  #define TEST_REQUIRES_WASM_PSHUFB_VALUE (false)
  #define TEST_REQUIRES_WASM_PSHUFB do {} while (0)
#endif

#if XNN_ARCH_WASMRELAXEDSIMD
  #define TEST_REQUIRES_WASM_SDOT_VALUE XNN_TEST_HWCONFIG_FLAG(use_wasm_sdot)
  #define TEST_REQUIRES_WASM_SDOT TEST_REQUIRES_HWCONFIG_FLAG(use_wasm_sdot)
#else
  #define TEST_REQUIRES_WASM_SDOT_VALUE (false)
  #define TEST_REQUIRES_WASM_SDOT do {} while (0)
#endif

#if XNN_ARCH_WASMRELAXEDSIMD
  #define TEST_REQUIRES_WASM_USDOT_VALUE XNN_TEST_HWCONFIG_FLAG(use_wasm_usdot)
  #define TEST_REQUIRES_WASM_USDOT TEST_REQUIRES_HWCONFIG_FLAG(use_wasm_usdot)
#else
  #define TEST_REQUIRES_WASM_USDOT_VALUE (false)
  #define TEST_REQUIRES_WASM_USDOT do {} while (0)
#endif

#if XNN_ARCH_WASMRELAXEDSIMD
  #define TEST_REQUIRES_WASM_BLENDVPS_VALUE XNN_TEST_HWCONFIG_FLAG(use_wasm_blendvps)
  #define TEST_REQUIRES_WASM_BLENDVPS TEST_REQUIRES_HWCONFIG_FLAG(use_wasm_blendvps)
#else
  #define TEST_REQUIRES_WASM_BLENDVPS_VALUE (false)
  #define TEST_REQUIRES_WASM_BLENDVPS do {} while (0)
#endif
