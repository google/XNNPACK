// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_F32_HVX_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_F32_HVX_H_

#include <assert.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <math.h>  // for roundf
#include <stddef.h>
#include <string.h>  // for memcpy

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"

// SIMD vector type for f32 using HVX.
typedef HVX_Vector xnn_simd_f32_t;
#define xnn_simd_size_f32 32
#define xnn_simd_log2_size_f32 5
#define xnn_simd_bytes_f32 (xnn_simd_size_f32 * sizeof(float))

#define XNN_SIMD_CONST_F32(var, val) \
  const xnn_simd_f32_t var = Q6_V_vsplat_R(val);

#define XNN_SIMD_CONST_F32_FROM_INT32(var, val) \
  const HVX_Vector var = Q6_V_vsplat_R(val);

// Whether or not this architecture has native fused multiply-add support.
#define XNN_SIMD_HAS_NATIVE_FMA 0

// Arithmetic operations.

static XNN_INLINE xnn_simd_f32_t xnn_zero_f32() { return Q6_V_vzero(); }

static XNN_INLINE xnn_simd_f32_t xnn_add_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}

static XNN_INLINE xnn_simd_f32_t xnn_mul_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
}

static XNN_INLINE xnn_simd_f32_t xnn_rcp_f32(xnn_simd_f32_t a) {
  return fast_inverse__vsf(a);
}

static XNN_INLINE xnn_simd_f32_t xnn_div_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, xnn_rcp_f32(b)));
}

static XNN_INLINE xnn_simd_f32_t xnn_fmadd_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b,
                                               xnn_simd_f32_t c) {
  return Q6_Vsf_equals_Vqf32(
      Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(a, b), c));
}

// c - a*b -> c + -(a*b)
static XNN_INLINE xnn_simd_f32_t xnn_fnmadd_f32(xnn_simd_f32_t a,
                                                xnn_simd_f32_t b,
                                                xnn_simd_f32_t c) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(
      Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_vzero(), Q6_Vqf32_vmpy_VsfVsf(a, b)), c));
}

static XNN_INLINE xnn_simd_f32_t xnn_fmsub_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b,
                                               xnn_simd_f32_t c) {
  return Q6_Vsf_equals_Vqf32(
      Q6_Vqf32_vsub_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(a, b), c));
}

static XNN_INLINE xnn_simd_f32_t xnn_sub_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b));
}

static XNN_INLINE xnn_simd_f32_t xnn_max_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return Q6_Vsf_vmax_VsfVsf(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_min_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return Q6_Vsf_vmin_VsfVsf(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_abs_f32(xnn_simd_f32_t a) {
  return Q6_V_vand_VV(a, Q6_V_vsplat_R(0x7FFFFFFF));
}

static XNN_INLINE xnn_simd_f32_t xnn_neg_f32(xnn_simd_f32_t a) {
  return Q6_V_vxor_VV(a, Q6_V_vsplat_R(0x80000000));
}

// TODO: Implement hvx code sequence.
// - compare exp to smallest exp that is integer (23)
// - convert to int and back to float
// - use compare result to select rounding int or original float
// - large exp includes NaN and inf and negative versions of these
static XNN_INLINE xnn_simd_f32_t xnn_round_f32(xnn_simd_f32_t a) {
  XNN_ALIGN(128) float input[xnn_simd_size_f32];
  XNN_ALIGN(128) float output[xnn_simd_size_f32];
  *((HVX_Vector*)input) = a;
  for (size_t k = 0; k < xnn_simd_size_f32; ++k) {
    output[k] = roundf(input[k]);
  }
  return *((HVX_Vector*)output);
}

// Logical operations.

static XNN_INLINE xnn_simd_f32_t xnn_and_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return Q6_V_vand_VV(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_or_f32(xnn_simd_f32_t a,
                                            xnn_simd_f32_t b) {
  return Q6_V_vor_VV(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_xor_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return Q6_V_vxor_VV(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_sll_f32(xnn_simd_f32_t a, uint8_t bits) {
  return Q6_Vw_vasl_VwR(a, (uint32_t)bits);
}

static XNN_INLINE xnn_simd_f32_t xnn_srl_f32(xnn_simd_f32_t a, uint8_t bits) {
  return Q6_Vuw_vlsr_VuwR(a, (uint32_t)bits);
}

static XNN_INLINE xnn_simd_f32_t xnn_sra_f32(xnn_simd_f32_t a, uint8_t bits) {
  return Q6_Vw_vasr_VwR(a, (uint32_t)bits);
}

static XNN_INLINE xnn_simd_f32_t xnn_cmpeq_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b) {
  return Q6_V_vand_QR(Q6_Q_vcmp_eq_VwVw(a, b), 0xFFFFFFFF);
}

// Special functions.
#define XNN_SIMD_HAVE_RCP_F32 0
#define XNN_SIMD_HAVE_RSQRT_F32 0

// Load/store operations.

static XNN_INLINE xnn_simd_f32_t xnn_loadu_f32(const float* ptr) {
  return *((HVX_UVector*)ptr);
}

static XNN_INLINE xnn_simd_f32_t xnn_load_f32(const float* ptr) {
  return *((HVX_Vector*)ptr);
}

static XNN_INLINE void xnn_storeu_f32(float* ptr, xnn_simd_f32_t v) {
  *((HVX_UVector*)ptr) = v;
}

static XNN_INLINE void xnn_store_f32(float* ptr, xnn_simd_f32_t v) {
  *((HVX_Vector*)ptr) = v;
}

static XNN_INLINE xnn_simd_f32_t xnn_set1_f32(float v) {
  return Q6_V_vsplat_R(*(uint32_t*)&v);
}

// Tail load/store operations.

static XNN_INLINE xnn_simd_f32_t
xnn_load_tail_f32(const float* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_f32);

  return *((HVX_UVector*)input);
}

static XNN_INLINE xnn_simd_f32_t xnn_load_tail_safe_f32(const float* input,
                                                        size_t num_elements) {
  assert(num_elements <= xnn_simd_size_f32);

  XNN_ALIGN(128) float padded[xnn_simd_size_f32];
  memcpy(padded, input, num_elements * sizeof(float));
  return *(HVX_Vector*)padded;
}

static XNN_INLINE void xnn_store_tail_f32(float* output, xnn_simd_f32_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_f32);

  return Q6_V_vstu_variable(output, num_elements << XNN_LOG2_SIZEOF_FLOAT, v);
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_F32_HVX_H_
