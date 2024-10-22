// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_U32_WASMSIMD_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_U32_WASMSIMD_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <wasm_simd128.h>

#include "xnnpack/common.h"
#include "xnnpack/unaligned.h"

// SIMD vector type for u32 using WASMSIMD.
typedef v128_t xnn_simd_u32_t;
#define xnn_simd_size_u32 4
#define xnn_simd_log2_size_u32 2
#define xnn_simd_bytes_u32 (xnn_simd_size_u32 * sizeof(uint32_t))

#define XNN_SIMD_CONST_U32(var, val) \
  const xnn_simd_u32_t var = wasm_u32x4_splat(val);

// Arithmetic operations.

static XNN_INLINE xnn_simd_u32_t xnn_mul_u32(xnn_simd_u32_t a,
                                             xnn_simd_u32_t b) {
  return wasm_i32x4_mul(a, b);
}

static XNN_INLINE xnn_simd_u32_t xnn_max_u32(xnn_simd_u32_t a,
                                             xnn_simd_u32_t b) {
  return wasm_u32x4_max(a, b);
}

static XNN_INLINE xnn_simd_u32_t xnn_min_u32(xnn_simd_u32_t a,
                                             xnn_simd_u32_t b) {
  return wasm_u32x4_min(a, b);
}

static XNN_INLINE xnn_simd_u32_t xnn_sub_u32(xnn_simd_u32_t a,
                                             xnn_simd_u32_t b) {
  return wasm_i32x4_sub(a, b);
}

static XNN_INLINE v128_t xnn_subw_f32_u32(xnn_simd_u32_t a,
                                          xnn_simd_u32_t b) {
  const v128_t mask = wasm_u32x4_gt(a, b);
  const v128_t variant1 = wasm_f32x4_convert_u32x4(wasm_i32x4_sub(a, b));
  const v128_t variant2 = wasm_f32x4_convert_u32x4(wasm_i32x4_sub(b, a));
  const v128_t sign = wasm_v128_bitselect(wasm_f32x4_splat(1),
                                          wasm_f32x4_splat(-1), mask);
  return wasm_f32x4_mul(wasm_v128_bitselect(variant1, variant2, mask), sign);
}

// Load/store operations.

static XNN_INLINE xnn_simd_u32_t xnn_loadu_u32(const uint32_t* ptr) {
  return wasm_v128_load(ptr);
}

static XNN_INLINE xnn_simd_u32_t xnn_load_u32(const uint32_t* ptr) {
  return wasm_v128_load(ptr);
}

static XNN_INLINE void xnn_storeu_u32(uint32_t* ptr, xnn_simd_u32_t v) {
  wasm_v128_store(ptr, v);
}

static XNN_INLINE void xnn_store_u32(float* ptr, xnn_simd_u32_t v) {
  wasm_v128_store(ptr, v);
}

static XNN_INLINE xnn_simd_u32_t xnn_set1_u32(uint32_t v) {
  return wasm_u32x4_splat(v);
}

static XNN_INLINE xnn_simd_u32_t xnn_set1_or_load_u32(const uint32_t* v) {
  return wasm_u32x4_splat(*v);
}

// Tail load/store operations.

static XNN_INLINE xnn_simd_u32_t
xnn_load_tail_u32(const uint32_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_u32);
  return wasm_v128_load(input);
}

static XNN_INLINE void xnn_store_tail_u32(uint32_t* output, xnn_simd_u32_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_u32);

  if (num_elements & 2) {
    wasm_v128_store64_lane(output, v, 0);
    v = wasm_i64x2_shuffle(v, v, 1, 1);
    output += 2;
  }
  if (num_elements & 1) {
    wasm_v128_store32_lane(output, v, 0);
  }
}

// Conversion operations.

static XNN_INLINE v128_t xnn_cvt_f32_u32(xnn_simd_u32_t a) {
  return wasm_f32x4_convert_u32x4(a);
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_U32_WASMSIMD_H_
