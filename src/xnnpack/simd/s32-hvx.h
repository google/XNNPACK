// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_S32_HVX_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_S32_HVX_H_

#include <assert.h>
#include <stddef.h>

#include <hvx_hexagon_protos.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"

// SIMD vector type for s32 using NEON.
typedef HVX_Vector xnn_simd_s32_t;
#define xnn_simd_size_s32 32
#define xnn_simd_log2_size_s32 5
#define xnn_simd_bytes_s32 (xnn_simd_size_s32 * sizeof(int32_t))

#define XNN_SIMD_CONST_S32(var, val) const HVX_Vector var = Q6_V_vsplat_R(val);

// Arithmetic operations.
static XNN_INLINE xnn_simd_s32_t xnn_mul_s32(xnn_simd_s32_t a,
                                             xnn_simd_s32_t b) {
  return Q6_V_lo_W(Q6_W_vmpyoacc_WVwVh(Q6_W_vmpye_VwVuh(a, b), a, b));
}

static XNN_INLINE xnn_simd_s32_t xnn_max_s32(xnn_simd_s32_t a,
                                             xnn_simd_s32_t b) {
  return Q6_Vw_vmax_VwVw(a, b);
}

static XNN_INLINE xnn_simd_s32_t xnn_min_s32(xnn_simd_s32_t a,
                                             xnn_simd_s32_t b) {
  return Q6_Vw_vmin_VwVw(a, b);
}

static XNN_INLINE xnn_simd_s32_t xnn_sub_s32(xnn_simd_s32_t a,
                                             xnn_simd_s32_t b) {
  return Q6_Vw_vsub_VwVw(a, b);
}

// Load/store operations.
static XNN_INLINE xnn_simd_s32_t xnn_loadu_s32(const int32_t* ptr) {
  return *((HVX_UVector*) ptr);
}

static XNN_INLINE xnn_simd_s32_t xnn_load_s32(const int32_t* ptr) {
  return *((HVX_UVector*) ptr);
}

static XNN_INLINE void xnn_storeu_s32(int32_t* ptr, xnn_simd_s32_t v) {
  *((HVX_UVector*) ptr) = v;
}

static XNN_INLINE void xnn_store_s32(int32_t* ptr, xnn_simd_s32_t v) {
  *((HVX_UVector*) ptr) = v;
}

static XNN_INLINE xnn_simd_s32_t xnn_set1_s32(int32_t v) {
  return Q6_V_vsplat_R(*(uint32_t *)&v);
}

static XNN_INLINE xnn_simd_s32_t xnn_set1_or_load_s32(const int32_t* v) {
  return *((HVX_UVector*) v);
}

// Tail load/store operations.
static XNN_INLINE xnn_simd_s32_t
xnn_load_tail_s32(const int32_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s32);

  return *((HVX_UVector*) input);
}

static XNN_INLINE void xnn_store_tail_s32(int32_t* output, xnn_simd_s32_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s32);

  return Q6_V_vstu_variable(output, num_elements << XNN_LOG2_SIZEOF_INT32_T, v);
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_S32_HVX_H_
