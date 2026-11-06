// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef XNNPACK_SRC_XNNPACK_SIMD_S16_HVX_H_
#define XNNPACK_SRC_XNNPACK_SIMD_S16_HVX_H_

#include <assert.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>  // for memcpy

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"

// SIMD vector type for s16 using HVX.
typedef HVX_Vector xnn_simd_s16_t;
#define xnn_simd_size_s16 64
#define xnn_simd_log2_size_s16 6
#define xnn_simd_bytes_s16 (xnn_simd_size_s16 * sizeof(int16_t))

#define XNN_SIMD_CONST_S16(var, val) const HVX_Vector var = Q6_Vh_vsplat_R(val);

// Arithmetic operations.
static XNN_INLINE xnn_simd_s16_t xnn_min_s16(xnn_simd_s16_t a,
                                             xnn_simd_s16_t b) {
  return Q6_Vh_vmin_VhVh(a, b);
}

static XNN_INLINE xnn_simd_s16_t xnn_max_s16(xnn_simd_s16_t a,
                                             xnn_simd_s16_t b) {
  return Q6_Vh_vmax_VhVh(a, b);
}

static XNN_INLINE xnn_simd_s16_t xnn_signcomplement_s16(xnn_simd_s16_t x) {
  XNN_SIMD_CONST_S16(nonsign_mask, 0x7FFF);
  return Q6_V_vxor_VV(Q6_V_vand_VV(x, nonsign_mask), Q6_Vh_vasr_VhR(x, 15));
}

// Load/store operations.
static XNN_INLINE xnn_simd_s16_t xnn_loadu_s16(const int16_t* ptr) {
  return *((HVX_UVector*)ptr);
}

static XNN_INLINE xnn_simd_s16_t xnn_load_s16(const int16_t* ptr) {
  return *((HVX_Vector*)ptr);
}

static XNN_INLINE void xnn_storeu_s16(int16_t* ptr, xnn_simd_s16_t v) {
  *((HVX_UVector*)ptr) = v;
}

static XNN_INLINE void xnn_store_s16(int16_t* ptr, xnn_simd_s16_t v) {
  *((HVX_Vector*)ptr) = v;
}

static XNN_INLINE xnn_simd_s16_t xnn_set1_s16(int16_t v) {
  return Q6_Vh_vsplat_R(*(uint16_t*)&v);
}

// Tail load/store operations.
static XNN_INLINE xnn_simd_s16_t
xnn_load_tail_s16(const int16_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements <= xnn_simd_size_s16);

  return *((HVX_UVector*)input);
}

static XNN_INLINE xnn_simd_s16_t xnn_load_tail_safe_s16(const int16_t* input,
                                                        size_t num_elements) {
  assert(num_elements <= xnn_simd_size_s16);

  xnn_simd_s16_t padded;
  memcpy(&padded, input, num_elements * sizeof(int16_t));
  return xnn_load_s16((const int16_t*)&padded);
}

static XNN_INLINE void xnn_store_tail_s16(int16_t* output, xnn_simd_s16_t v,
                                          size_t num_elements) {
  assert(num_elements <= xnn_simd_size_s16);

  return Q6_V_vstu_variable(output, num_elements << XNN_LOG2_SIZEOF_INT16_T, v);
}

#endif  // XNNPACK_SRC_XNNPACK_SIMD_S16_HVX_H_
