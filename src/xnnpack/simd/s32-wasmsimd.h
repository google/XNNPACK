// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef XNNPACK_SRC_XNNPACK_SIMD_S32_WASMSIMD_H_
#define XNNPACK_SRC_XNNPACK_SIMD_S32_WASMSIMD_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <wasm_simd128.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/unaligned.h"

// SIMD vector type for s32 using WASMSIMD.
typedef v128_t xnn_simd_s32_t;
#define xnn_simd_size_s32 4
#define xnn_simd_log2_size_s32 2
#define xnn_simd_bytes_s32 (xnn_simd_size_s32 * sizeof(int32_t))

#define XNN_SIMD_CONST_S32(var, val) \
  const xnn_simd_s32_t var = wasm_i32x4_splat(val);

// Arithmetic operations.

static XNN_INLINE xnn_simd_s32_t xnn_mul_s32(xnn_simd_s32_t a,
                                             xnn_simd_s32_t b) {
  return wasm_i32x4_mul(a, b);
}

static XNN_INLINE xnn_simd_s32_t xnn_max_s32(xnn_simd_s32_t a,
                                             xnn_simd_s32_t b) {
  return wasm_i32x4_max(a, b);
}

static XNN_INLINE xnn_simd_s32_t xnn_min_s32(xnn_simd_s32_t a,
                                             xnn_simd_s32_t b) {
  return wasm_i32x4_min(a, b);
}

static XNN_INLINE xnn_simd_s32_t xnn_sub_s32(xnn_simd_s32_t a,
                                             xnn_simd_s32_t b) {
  return wasm_i32x4_sub(a, b);
}

// Load/store operations.

static XNN_INLINE xnn_simd_s32_t xnn_loadu_s32(const int32_t* ptr) {
  return wasm_v128_load(ptr);
}

static XNN_INLINE xnn_simd_s32_t xnn_load_s32(const int32_t* ptr) {
  return wasm_v128_load(ptr);
}

static XNN_INLINE void xnn_storeu_s32(int32_t* ptr, xnn_simd_s32_t v) {
  wasm_v128_store(ptr, v);
}

static XNN_INLINE void xnn_store_s32(int32_t* ptr, xnn_simd_s32_t v) {
  wasm_v128_store(ptr, v);
}

static XNN_INLINE xnn_simd_s32_t xnn_set1_s32(int32_t v) {
  return wasm_i32x4_splat(v);
}

// Tail load/store operations.

static XNN_INLINE xnn_simd_s32_t
xnn_load_tail_s32(const int32_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements <= xnn_simd_size_s32);
  return wasm_v128_load(input);
}

static XNN_INLINE xnn_simd_s32_t xnn_load_tail_safe_s32(const int32_t* input,
                                                        size_t num_elements) {
  assert(num_elements <= xnn_simd_size_s32);

  XNN_ALIGN(16) int32_t padded[4];
  int32_t* dst = padded;
  switch (num_elements) {
    case 4:
      *dst++ = *input++;
    case 3:
      *dst++ = *input++;
    case 2:
      *dst++ = *input++;
    case 1:
      *dst++ = *input++;
    default: ;
  }
  return wasm_v128_load(padded);
}

static XNN_INLINE void xnn_store_tail_s32(int32_t* output, xnn_simd_s32_t v,
                                          size_t num_elements) {
  assert(num_elements <= xnn_simd_size_s32);

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

static XNN_INLINE v128_t xnn_cvt_f32_s32(xnn_simd_s32_t a) {
  return wasm_f32x4_convert_i32x4(a);
}

#endif  // XNNPACK_SRC_XNNPACK_SIMD_S32_WASMSIMD_H_
