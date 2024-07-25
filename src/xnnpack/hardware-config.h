// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"

#ifdef __cplusplus
extern "C" {
#endif

struct xnn_hardware_config {
  char _; // Dummy member variable to comply with the C standard
#if XNN_ARCH_ARM
  bool use_arm_v6;
  bool use_arm_vfpv2;
  bool use_arm_vfpv3;
  bool use_arm_neon;
  bool use_arm_neon_fp16;
  bool use_arm_neon_fma;
  bool use_arm_neon_v8;
#endif  // XNN_ARCH_ARM
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  bool use_arm_fp16_arith;
  bool use_arm_neon_fp16_arith;
  bool use_arm_neon_bf16;
  bool use_arm_neon_dot;
  bool use_arm_neon_i8mm;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  bool use_x86_ssse3;
  bool use_x86_sse4_1;
  bool use_x86_avx;
  bool use_x86_f16c;
  bool use_x86_fma3;
  bool use_x86_avx2;
  bool use_x86_avx512f;
  bool use_x86_avx512vbmi;
  bool use_x86_avx512skx;
  bool use_x86_avx512vnni;
  bool use_x86_avx512vnnigfni;
  bool use_x86_avx512amx;
  bool use_x86_avx512fp16;
  bool use_x86_avxvnni;
  bool use_x86_avx256skx;
  bool use_x86_avx256vnni;
  bool use_x86_avx256vnnigfni;
#endif
#if XNN_ARCH_RISCV
  bool use_riscv_vector;
  bool use_riscv_vector_fp16_arith;
  // vlenb CSR (VLEN/8). 0 if vector extension is unsupported.
  uint32_t vlenb;
#endif
#if XNN_ARCH_PPC64
  bool use_vsx;
  bool use_vsx3;
  bool use_mma;
#endif
#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  bool is_x86;
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
#if XNN_ARCH_WASMRELAXEDSIMD
  bool use_wasm_blendvps;
  bool use_wasm_pshufb;
  bool use_wasm_sdot;
  bool use_wasm_usdot;
  bool use_wasm_fma;
#endif  // XNN_ARCH_WASMRELAXEDSIMD
#if XNN_ARCH_HEXAGON
  bool use_hvx;
#endif  // XNN_ARCH_HEXAGON
};

XNN_INTERNAL const struct xnn_hardware_config* xnn_init_hardware_config();

static inline bool xnn_is_f16_compatible_config(const struct xnn_hardware_config hardware_config[XNN_MIN_ELEMENTS(1)]) {
  #if (XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR) || (XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR)
    return hardware_config->use_arm_neon_fp16_arith;
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    return hardware_config->use_x86_avx2;
  #else
    return false;
  #endif
}

static inline bool xnn_is_f16_chw_compatible_config(const struct xnn_hardware_config hardware_config[XNN_MIN_ELEMENTS(1)]) {
  #if (XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR) || (XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR)
    return hardware_config->use_arm_neon_fp16_arith;
  #else
    return false;
  #endif
}

static inline bool xnn_is_chw_compatible_config(const struct xnn_hardware_config hardware_config[XNN_MIN_ELEMENTS(1)]) {
  #if (XNN_ARCH_X86 || XNN_ARCH_X86_64)
    // Sparse microkernels on x86 currently target only SSE, and on processors
    // with AVX ISA dense inference is expected to be faster than sparse.
    return (!hardware_config->use_x86_avx);
  #else
    return true;
  #endif
}

static inline bool xnn_is_f16_supported_natively(const struct xnn_hardware_config hardware_config[XNN_MIN_ELEMENTS(1)]) {
  #if (XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR) || (XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR)
    return hardware_config->use_arm_neon_fp16_arith;
  #else
    return false;
  #endif
}

#ifdef __cplusplus
}  // extern "C"
#endif
