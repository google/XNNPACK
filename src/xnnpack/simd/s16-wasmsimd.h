// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_S16_WASMSIMD_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_S16_WASMSIMD_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <wasm_simd128.h>

#include "xnnpack/common.h"
#include "xnnpack/unaligned.h"

// SIMD vector type for s16 using WASMSIMD.
typedef v128_t xnn_simd_s16_t;
#define xnn_simd_size_s16 8
#define xnn_simd_log2_size_s16 3
#define xnn_simd_bytes_s16 (xnn_simd_size_s16 * sizeof(int16_t))

#define XNN_SIMD_CONST_S16(var, val) \
  const xnn_simd_s16_t var = wasm_i16x8_splat(val);

// Arithmetic operations.

// Load/store operations.

static XNN_INLINE xnn_simd_s16_t xnn_loadu_s16(const int16_t* ptr) {
  return wasm_v128_load(ptr);
}

static XNN_INLINE xnn_simd_s16_t xnn_load_s16(const int16_t* ptr) {
  return wasm_v128_load(ptr);
}

static XNN_INLINE void xnn_storeu_s16(int16_t* ptr, xnn_simd_s16_t v) {
  wasm_v128_store(ptr, v);
}

static XNN_INLINE void xnn_store_s16(float* ptr, xnn_simd_s16_t v) {
  wasm_v128_store(ptr, v);
}

static XNN_INLINE xnn_simd_s16_t xnn_set1_s16(int16_t v) {
  return wasm_i16x8_splat(v);
}

static XNN_INLINE xnn_simd_s16_t xnn_set1_or_load_s16(const int16_t* v) {
  return wasm_i16x8_splat(*v);
}

// Tail load/store operations.

static XNN_INLINE xnn_simd_s16_t
xnn_load_tail_s16(const int16_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s16);
  return wasm_v128_load(input);
}

static XNN_INLINE void xnn_store_tail_s16(int16_t* output, xnn_simd_s16_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s16);

  if (num_elements & 4) {
    wasm_v128_store64_lane(output, v, 0);
    v = wasm_i64x2_shuffle(v, v, 1, 1);
    output += 4;
  }
  if (num_elements & 2) {
    wasm_v128_store32_lane(output, v, 0);
    v = wasm_i64x2_shr(v, 32);
    output += 2;
  }
  if (num_elements & 1) {
    wasm_v128_store16_lane(output, v, 0);
  }
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_S16_WASMSIMD_H_
