// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_HARDWARE_CONFIG_H_
#define XNNPACK_SRC_XNNPACK_HARDWARE_CONFIG_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// These flags should be sorted by preference (a < b ==> a slower than b).
enum xnn_arch_flags {
  xnn_arch_none = 0,
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  xnn_arch_arm_v6 = 1 << 0,
  xnn_arch_arm_vfpv2 = 1 << 1,
  xnn_arch_arm_vfpv3 = 1 << 2,
  xnn_arch_arm_neon = 1 << 3,
  xnn_arch_arm_neon_fma = 1 << 4,
  xnn_arch_arm_neon_v8 = 1 << 5,
  xnn_arch_arm_fp16_arith = 1 << 6,
  xnn_arch_arm_neon_fp16_arith = 1 << 7,
  xnn_arch_arm_neon_fp16 = 1 << 8,
  xnn_arch_arm_neon_bf16 = 1 << 9,
  xnn_arch_arm_neon_dot = 1 << 10,
  xnn_arch_arm_neon_i8mm = 1 << 11,
  xnn_arch_arm_sve = 1 << 12,
  xnn_arch_arm_sve2 = 1 << 13,
  xnn_arch_arm_sme = 1 << 14,
  xnn_arch_arm_sme2 = 1 << 15,
#elif XNN_ARCH_X86 || XNN_ARCH_X86_64
  xnn_arch_x86_sse = 1 << 0,
  xnn_arch_x86_sse2 = 1 << 1,
  xnn_arch_x86_ssse3 = 1 << 2,
  xnn_arch_x86_sse4_1 = 1 << 3,
  xnn_arch_x86_avx = 1 << 4,
  xnn_arch_x86_f16c = 1 << 5,
  xnn_arch_x86_fma3 = 1 << 6,
  xnn_arch_x86_avx2 = 1 << 7,
  xnn_arch_x86_avx256skx = 1 << 8,
  xnn_arch_x86_avx512f = 1 << 9,
  xnn_arch_x86_avx512skx = 1 << 10,
  xnn_arch_x86_avx512vbmi = 1 << 11,
  xnn_arch_x86_avx512fp16 = 1 << 12,
  xnn_arch_x86_avx512bf16 = 1 << 13,
  xnn_arch_x86_avxvnni = 1 << 14,
  xnn_arch_x86_avxvnniint8 = 1 << 15,
  xnn_arch_x86_avx256vnni = 1 << 16,
  xnn_arch_x86_avx256vnnigfni = 1 << 17,
  xnn_arch_x86_avx512vnni = 1 << 18,
  xnn_arch_x86_avx512vnnigfni = 1 << 19,
  xnn_arch_x86_avx512amx = 1 << 20,
#elif XNN_ARCH_RISCV
  xnn_arch_riscv_vector = 1 << 0,
  xnn_arch_riscv_vector_fp16_arith = 1 << 1,
#elif XNN_ARCH_PPC64
  xnn_arch_vsx = 1 << 0,
  xnn_arch_vsx3 = 1 << 1,
  xnn_arch_mma = 1 << 2,
#elif XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  xnn_arch_wasm_is_x86 = 1 << 0,
#if XNN_ARCH_WASMRELAXEDSIMD
  xnn_arch_wasm_blendvps = 1 << 1,
  xnn_arch_wasm_pshufb = 1 << 2,
  xnn_arch_wasm_sdot = 1 << 3,
  xnn_arch_wasm_usdot = 1 << 4,
  xnn_arch_wasm_fma = 1 << 5,
#endif // XNN_ARCH_WASMRELAXEDSIMD
#elif XNN_ARCH_HEXAGON
  xnn_arch_hvx = 1 << 0,
#endif
};

enum xnn_uarch {
  xnn_uarch_unknown = 0,

  xnn_uarch_dhyana,
  xnn_uarch_zen,
  xnn_uarch_zen4,
  xnn_uarch_zen5,

  xnn_uarch_cortex_a5,
  xnn_uarch_cortex_a7,
  xnn_uarch_cortex_a32,
  xnn_uarch_cortex_a35,
  xnn_uarch_cortex_a53,
  xnn_uarch_cortex_a55,
  xnn_uarch_cortex_a55r0,
  xnn_uarch_cortex_a57,
  xnn_uarch_cortex_a72,
  xnn_uarch_cortex_a73,
  xnn_uarch_cortex_a75,
  xnn_uarch_cortex_a76,
  xnn_uarch_cortex_a77,
  xnn_uarch_cortex_a78,
  xnn_uarch_cortex_a510,
  xnn_uarch_cortex_a710,
  xnn_uarch_cortex_a715,
  xnn_uarch_cortex_x1,
  xnn_uarch_cortex_x2,
  xnn_uarch_cortex_x3,
  xnn_uarch_cortex_x4,
  xnn_uarch_exynos_m1,
  xnn_uarch_exynos_m2,
  xnn_uarch_exynos_m3,
  xnn_uarch_exynos_m4,
  xnn_uarch_exynos_m5,
  xnn_uarch_krait,
  xnn_uarch_kryo,
  xnn_uarch_neoverse_n1,
  xnn_uarch_neoverse_n2,
  xnn_uarch_neoverse_v1,
  xnn_uarch_neoverse_v2,
  xnn_uarch_oryon,
};

struct xnn_hardware_config {
  enum xnn_uarch uarch[XNN_MAX_UARCH_TYPES];
  uint64_t arch_flags;
#if XNN_ARCH_RISCV
  // vlenb CSR (VLEN/8). 0 if vector extension is unsupported.
  uint32_t vlenb;
#endif
#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  bool is_x86;
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  // Size in bytes of the L1 data cache.
  size_t l1_data_cache_bytes;
  size_t l1_data_cache_line_size;
  size_t l1_data_cache_associativity;
  size_t l1_data_cache_num_sets;

  // Size in bytes of the L2 data cache.
  size_t l2_data_cache_bytes;
  size_t l2_data_cache_line_size;
  size_t l2_data_cache_associativity;
  size_t l2_data_cache_num_sets;
};

XNN_INTERNAL const struct xnn_hardware_config* xnn_init_hardware_config();

static inline bool xnn_is_bf16_compatible_config(
    const struct xnn_hardware_config* hardware_config) {
#if XNN_ARCH_X86_64
  return (hardware_config->arch_flags & xnn_arch_x86_avx512bf16);
#else
  return false;
#endif
}

static inline bool xnn_is_f16_compatible_config(
    const struct xnn_hardware_config* hardware_config) {
#if (XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && \
     XNN_ENABLE_ARM_FP16_SCALAR) ||                \
    (XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR)
  return (hardware_config->arch_flags & xnn_arch_arm_neon_fp16_arith);
#elif XNN_ENABLE_AVX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  return (hardware_config->arch_flags & xnn_arch_x86_f16c) && (hardware_config->arch_flags & xnn_arch_x86_avx);
#elif (XNN_ARCH_RISCV && XNN_ENABLE_RISCV_FP16_VECTOR)
  return (hardware_config->arch_flags & xnn_arch_riscv_vector_fp16_arith);
#else
  return false;
#endif
}

static inline bool xnn_is_f16_chw_compatible_config(
    const struct xnn_hardware_config* hardware_config) {
#if (XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && \
     XNN_ENABLE_ARM_FP16_SCALAR) ||                \
    (XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR)
  return (hardware_config->arch_flags & xnn_arch_arm_neon_fp16_arith);
#else
  return false;
#endif
}

static inline bool xnn_is_chw_compatible_config(
    const struct xnn_hardware_config* hardware_config) {
#if (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  // Sparse microkernels on x86 currently target only SSE, and on processors
  // with AVX ISA dense inference is expected to be faster than sparse.
  return (!(hardware_config->arch_flags & xnn_arch_x86_avx));
#else
  return true;
#endif
}

static inline bool xnn_is_f16_supported_natively(
    const struct xnn_hardware_config* hardware_config) {
#if (XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && \
     XNN_ENABLE_ARM_FP16_SCALAR) ||                \
    (XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR)
  return (hardware_config->arch_flags & xnn_arch_arm_neon_fp16_arith);
#elif (XNN_ARCH_RISCV && XNN_ENABLE_RISCV_FP16_VECTOR)
  return (hardware_config->arch_flags & xnn_arch_riscv_vector_fp16_arith);
#else
  return false;
#endif
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_SRC_XNNPACK_HARDWARE_CONFIG_H_
