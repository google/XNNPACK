// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_U8_HVX_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_U8_HVX_H_

#include <assert.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stddef.h>
#include <string.h>  // for memcpy

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"

// SIMD vector type for u8 using HVX.
typedef HVX_Vector xnn_simd_u8_t;
#define xnn_simd_size_u8 128
#define xnn_simd_bytes_u8 128

#define XNN_SIMD_CONST_U8(var, val) const HVX_Vector var = Q6_Vb_vsplat_R(val);

static XNN_INLINE xnn_simd_u8_t xnn_add_u8(xnn_simd_u8_t a, xnn_simd_u8_t b) {
  return Q6_Vb_vadd_VbVb(a, b);
}

static XNN_INLINE xnn_simd_u8_t xnn_max_u8(xnn_simd_u8_t a, xnn_simd_u8_t b) {
  return Q6_Vub_vmax_VubVub(a, b);
}

static XNN_INLINE xnn_simd_u8_t xnn_min_u8(xnn_simd_u8_t a, xnn_simd_u8_t b) {
  return Q6_Vub_vmin_VubVub(a, b);
}

static XNN_INLINE float xnn_reduce_max_u8(xnn_simd_u8_t v) {
  v = Q6_Vub_vmax_VubVub(v, Q6_V_vror_VR(v, 64));
  v = Q6_Vub_vmax_VubVub(v, Q6_V_vror_VR(v, 32));
  v = Q6_Vub_vmax_VubVub(v, Q6_V_vror_VR(v, 16));
  v = Q6_Vub_vmax_VubVub(v, Q6_V_vror_VR(v, 8));
  v = Q6_Vub_vmax_VubVub(v, Q6_V_vror_VR(v, 4));
  v = Q6_Vub_vmax_VubVub(v, Q6_V_vror_VR(v, 2));
  v = Q6_Vub_vmax_VubVub(v, Q6_V_vror_VR(v, 1));
  return *((uint8_t*)&v);
}

static XNN_INLINE float xnn_reduce_min_u8(xnn_simd_u8_t v) {
  v = Q6_Vub_vmin_VubVub(v, Q6_V_vror_VR(v, 64));
  v = Q6_Vub_vmin_VubVub(v, Q6_V_vror_VR(v, 32));
  v = Q6_Vub_vmin_VubVub(v, Q6_V_vror_VR(v, 16));
  v = Q6_Vub_vmin_VubVub(v, Q6_V_vror_VR(v, 8));
  v = Q6_Vub_vmin_VubVub(v, Q6_V_vror_VR(v, 4));
  v = Q6_Vub_vmin_VubVub(v, Q6_V_vror_VR(v, 2));
  v = Q6_Vub_vmin_VubVub(v, Q6_V_vror_VR(v, 1));
  return *((uint8_t*)&v);
}

static XNN_INLINE xnn_simd_u8_t xnn_xor_u8(xnn_simd_u8_t a, xnn_simd_u8_t b) {
  return Q6_V_vxor_VV(a, b);
}

// Load/store operations.
static XNN_INLINE xnn_simd_u8_t xnn_loadu_u8(const uint8_t* ptr) {
  return *((HVX_UVector*)ptr);
}

static XNN_INLINE xnn_simd_u8_t xnn_load_u8(const uint8_t* ptr) {
  return *((HVX_Vector*)ptr);
}

static XNN_INLINE void xnn_storeu_u8(uint8_t* ptr, xnn_simd_u8_t v) {
  *((HVX_UVector*)ptr) = v;
}

static XNN_INLINE void xnn_store_u8(uint8_t* ptr, xnn_simd_u8_t v) {
  *((HVX_Vector*)ptr) = v;
}

static XNN_INLINE xnn_simd_u8_t xnn_set1_u8(uint8_t v) {
  return Q6_Vb_vsplat_R(*(uint8_t*)&v);
}

// Tail load/store operations.
static XNN_INLINE xnn_simd_u8_t
xnn_load_tail_u8(const uint8_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements <= xnn_simd_size_u8);

  return *((HVX_UVector*)input);
}

static XNN_INLINE xnn_simd_u8_t xnn_load_tail_safe_u8(const uint8_t* input,
                                                      size_t num_elements) {
  assert(num_elements <= xnn_simd_size_u8);

  xnn_simd_u8_t padded;
  memcpy(&padded, input, num_elements * sizeof(uint8_t));
  return xnn_load_u8((const uint8_t*)&padded);
}

static XNN_INLINE void xnn_store_tail_u8(uint8_t* output, xnn_simd_u8_t v,
                                         size_t num_elements) {
  assert(num_elements <= xnn_simd_size_u8);

  return Q6_V_vstu_variable(output, num_elements, v);
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_U8_HVX_H_
