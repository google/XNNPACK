// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_F32_WASMRELAXEDSIMD_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_F32_WASMRELAXEDSIMD_H_

#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/simd/f32-wasmsimd-base.h"  // IWYU pragma: export

// Whether or not this architecture has native fused multiply-add support.
#define XNN_SIMD_HAS_NATIVE_FMA 1

static XNN_INLINE xnn_simd_f32_t xnn_fmadd_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b,
                                               xnn_simd_f32_t c) {
  return wasm_f32x4_relaxed_madd(a, b, c);
}

static XNN_INLINE xnn_simd_f32_t xnn_fnmadd_f32(xnn_simd_f32_t a,
                                                xnn_simd_f32_t b,
                                                xnn_simd_f32_t c) {
  return wasm_f32x4_relaxed_nmadd(a, b, c);
}

static XNN_INLINE xnn_simd_f32_t xnn_fmsub_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b,
                                               xnn_simd_f32_t c) {
  return wasm_f32x4_relaxed_madd(a, b, wasm_f32x4_neg(c));
}

static XNN_INLINE xnn_simd_f32_t xnn_max_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return wasm_f32x4_relaxed_max(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_min_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return wasm_f32x4_relaxed_min(a, b);
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_F32_WASMRELAXEDSIMD_H_
