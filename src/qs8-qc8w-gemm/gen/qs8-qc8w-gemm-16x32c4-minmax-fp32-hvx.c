// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx32c4-hvx.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/unaligned.h"



void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_16x32c4__hvx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 16);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const int8_t* a4 = (const int8_t*) ((uintptr_t) a3 + a_stride);
  int8_t* c4 = (int8_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const int8_t* a5 = (const int8_t*) ((uintptr_t) a4 + a_stride);
  int8_t* c5 = (int8_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const int8_t* a6 = (const int8_t*) ((uintptr_t) a5 + a_stride);
  int8_t* c6 = (int8_t*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }
  const int8_t* a7 = (const int8_t*) ((uintptr_t) a6 + a_stride);
  int8_t* c7 = (int8_t*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 8) {
    a7 = a6;
    c7 = c6;
  }
  const int8_t* a8 = (const int8_t*) ((uintptr_t) a7 + a_stride);
  int8_t* c8 = (int8_t*) ((uintptr_t) c7 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 8) {
    a8 = a7;
    c8 = c7;
  }
  const int8_t* a9 = (const int8_t*) ((uintptr_t) a8 + a_stride);
  int8_t* c9 = (int8_t*) ((uintptr_t) c8 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 10) {
    a9 = a8;
    c9 = c8;
  }
  const int8_t* a10 = (const int8_t*) ((uintptr_t) a9 + a_stride);
  int8_t* c10 = (int8_t*) ((uintptr_t) c9 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 10) {
    a10 = a9;
    c10 = c9;
  }
  const int8_t* a11 = (const int8_t*) ((uintptr_t) a10 + a_stride);
  int8_t* c11 = (int8_t*) ((uintptr_t) c10 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 12) {
    a11 = a10;
    c11 = c10;
  }
  const int8_t* a12 = (const int8_t*) ((uintptr_t) a11 + a_stride);
  int8_t* c12 = (int8_t*) ((uintptr_t) c11 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 12) {
    a12 = a11;
    c12 = c11;
  }
  const int8_t* a13 = (const int8_t*) ((uintptr_t) a12 + a_stride);
  int8_t* c13 = (int8_t*) ((uintptr_t) c12 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 14) {
    a13 = a12;
    c13 = c12;
  }
  const int8_t* a14 = (const int8_t*) ((uintptr_t) a13 + a_stride);
  int8_t* c14 = (int8_t*) ((uintptr_t) c13 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 14) {
    a14 = a13;
    c14 = c13;
  }
  const int8_t* a15 = (const int8_t*) ((uintptr_t) a14 + a_stride);
  int8_t* c15 = (int8_t*) ((uintptr_t) c14 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 16) {
    a15 = a14;
    c15 = c14;
  }

  const float output_max_less_zero_point = (float)((int32_t) params->fp32_scalar.output_max - (int32_t)params->fp32_scalar.output_zero_point);
  const HVX_Vector voutput_max_less_zero_point = Q6_V_vsplat_R(*(uint32_t *)(&output_max_less_zero_point));
  const HVX_Vector voutput_zero_point = Q6_V_vsplat_R(params->fp32_scalar.output_zero_point);
  const HVX_Vector voutput_min = Q6_Vb_vsplat_R(params->fp32_scalar.output_min);

  do {
    HVX_Vector vacc0x32 = *((HVX_Vector*)w);
    HVX_Vector vacc1x0x32 = Q6_V_vsplat_R(0);
    HVX_Vector vacc1x32 = *((HVX_Vector*)w);
    HVX_Vector vacc1x1x32 = Q6_V_vsplat_R(0);
    HVX_Vector vacc2x32 = *((HVX_Vector*)w);
    HVX_Vector vacc1x2x32 = Q6_V_vsplat_R(0);
    HVX_Vector vacc3x32 = *((HVX_Vector*)w);
    HVX_Vector vacc1x3x32 = Q6_V_vsplat_R(0);
    HVX_Vector vacc4x32 = *((HVX_Vector*)w);
    HVX_Vector vacc1x4x32 = Q6_V_vsplat_R(0);
    HVX_Vector vacc5x32 = *((HVX_Vector*)w);
    HVX_Vector vacc1x5x32 = Q6_V_vsplat_R(0);
    HVX_Vector vacc6x32 = *((HVX_Vector*)w);
    HVX_Vector vacc1x6x32 = Q6_V_vsplat_R(0);
    HVX_Vector vacc7x32 = *((HVX_Vector*)w);
    HVX_Vector vacc1x7x32 = Q6_V_vsplat_R(0);
    HVX_Vector vacc8x32 = *((HVX_Vector*)w);
    HVX_Vector vacc1x8x32 = Q6_V_vsplat_R(0);
    HVX_Vector vacc9x32 = *((HVX_Vector*)w);
    HVX_Vector vacc1x9x32 = Q6_V_vsplat_R(0);
    HVX_Vector vacc10x32 = *((HVX_Vector*)w);
    HVX_Vector vacc1x10x32 = Q6_V_vsplat_R(0);
    HVX_Vector vacc11x32 = *((HVX_Vector*)w);
    HVX_Vector vacc1x11x32 = Q6_V_vsplat_R(0);
    HVX_Vector vacc12x32 = *((HVX_Vector*)w);
    HVX_Vector vacc1x12x32 = Q6_V_vsplat_R(0);
    HVX_Vector vacc13x32 = *((HVX_Vector*)w);
    HVX_Vector vacc1x13x32 = Q6_V_vsplat_R(0);
    HVX_Vector vacc14x32 = *((HVX_Vector*)w);
    HVX_Vector vacc1x14x32 = Q6_V_vsplat_R(0);
    HVX_Vector vacc15x32 = *((HVX_Vector*)w);
    HVX_Vector vacc1x15x32 = Q6_V_vsplat_R(0);

    w = (const int32_t*) w + 32;

    size_t k = kc;
    while (k >= 8 * sizeof(int8_t)) {
      const HVX_Vector va0x0123 = Q6_V_vsplat_R(unaligned_load_s32(a0));
      const HVX_Vector va0x4567 = Q6_V_vsplat_R(unaligned_load_s32(a0+4));
      a0 += 8;
      const HVX_Vector va1x0123 = Q6_V_vsplat_R(unaligned_load_s32(a1));
      const HVX_Vector va1x4567 = Q6_V_vsplat_R(unaligned_load_s32(a1+4));
      a1 += 8;
      const HVX_Vector va2x0123 = Q6_V_vsplat_R(unaligned_load_s32(a2));
      const HVX_Vector va2x4567 = Q6_V_vsplat_R(unaligned_load_s32(a2+4));
      a2 += 8;
      const HVX_Vector va3x0123 = Q6_V_vsplat_R(unaligned_load_s32(a3));
      const HVX_Vector va3x4567 = Q6_V_vsplat_R(unaligned_load_s32(a3+4));
      a3 += 8;
      const HVX_Vector va4x0123 = Q6_V_vsplat_R(unaligned_load_s32(a4));
      const HVX_Vector va4x4567 = Q6_V_vsplat_R(unaligned_load_s32(a4+4));
      a4 += 8;
      const HVX_Vector va5x0123 = Q6_V_vsplat_R(unaligned_load_s32(a5));
      const HVX_Vector va5x4567 = Q6_V_vsplat_R(unaligned_load_s32(a5+4));
      a5 += 8;
      const HVX_Vector va6x0123 = Q6_V_vsplat_R(unaligned_load_s32(a6));
      const HVX_Vector va6x4567 = Q6_V_vsplat_R(unaligned_load_s32(a6+4));
      a6 += 8;
      const HVX_Vector va7x0123 = Q6_V_vsplat_R(unaligned_load_s32(a7));
      const HVX_Vector va7x4567 = Q6_V_vsplat_R(unaligned_load_s32(a7+4));
      a7 += 8;
      const HVX_Vector va8x0123 = Q6_V_vsplat_R(unaligned_load_s32(a8));
      const HVX_Vector va8x4567 = Q6_V_vsplat_R(unaligned_load_s32(a8+4));
      a8 += 8;
      const HVX_Vector va9x0123 = Q6_V_vsplat_R(unaligned_load_s32(a9));
      const HVX_Vector va9x4567 = Q6_V_vsplat_R(unaligned_load_s32(a9+4));
      a9 += 8;
      const HVX_Vector va10x0123 = Q6_V_vsplat_R(unaligned_load_s32(a10));
      const HVX_Vector va10x4567 = Q6_V_vsplat_R(unaligned_load_s32(a10+4));
      a10 += 8;
      const HVX_Vector va11x0123 = Q6_V_vsplat_R(unaligned_load_s32(a11));
      const HVX_Vector va11x4567 = Q6_V_vsplat_R(unaligned_load_s32(a11+4));
      a11 += 8;
      const HVX_Vector va12x0123 = Q6_V_vsplat_R(unaligned_load_s32(a12));
      const HVX_Vector va12x4567 = Q6_V_vsplat_R(unaligned_load_s32(a12+4));
      a12 += 8;
      const HVX_Vector va13x0123 = Q6_V_vsplat_R(unaligned_load_s32(a13));
      const HVX_Vector va13x4567 = Q6_V_vsplat_R(unaligned_load_s32(a13+4));
      a13 += 8;
      const HVX_Vector va14x0123 = Q6_V_vsplat_R(unaligned_load_s32(a14));
      const HVX_Vector va14x4567 = Q6_V_vsplat_R(unaligned_load_s32(a14+4));
      a14 += 8;
      const HVX_Vector va15x0123 = Q6_V_vsplat_R(unaligned_load_s32(a15));
      const HVX_Vector va15x4567 = Q6_V_vsplat_R(unaligned_load_s32(a15+4));
      a15 += 8;

      const HVX_Vector vb32x0123 = *((HVX_Vector *)((int8_t *)w));
      const HVX_Vector vb32x4567 = *((HVX_Vector *)((int8_t *)w + 128));
      vacc0x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc0x32, va0x0123, vb32x0123);
      vacc1x0x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x0x32, va0x4567, vb32x4567);
      vacc1x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc1x32, va1x0123, vb32x0123);
      vacc1x1x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x1x32, va1x4567, vb32x4567);
      vacc2x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc2x32, va2x0123, vb32x0123);
      vacc1x2x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x2x32, va2x4567, vb32x4567);
      vacc3x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc3x32, va3x0123, vb32x0123);
      vacc1x3x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x3x32, va3x4567, vb32x4567);
      vacc4x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc4x32, va4x0123, vb32x0123);
      vacc1x4x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x4x32, va4x4567, vb32x4567);
      vacc5x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc5x32, va5x0123, vb32x0123);
      vacc1x5x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x5x32, va5x4567, vb32x4567);
      vacc6x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc6x32, va6x0123, vb32x0123);
      vacc1x6x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x6x32, va6x4567, vb32x4567);
      vacc7x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc7x32, va7x0123, vb32x0123);
      vacc1x7x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x7x32, va7x4567, vb32x4567);
      vacc8x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc8x32, va8x0123, vb32x0123);
      vacc1x8x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x8x32, va8x4567, vb32x4567);
      vacc9x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc9x32, va9x0123, vb32x0123);
      vacc1x9x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x9x32, va9x4567, vb32x4567);
      vacc10x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc10x32, va10x0123, vb32x0123);
      vacc1x10x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x10x32, va10x4567, vb32x4567);
      vacc11x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc11x32, va11x0123, vb32x0123);
      vacc1x11x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x11x32, va11x4567, vb32x4567);
      vacc12x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc12x32, va12x0123, vb32x0123);
      vacc1x12x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x12x32, va12x4567, vb32x4567);
      vacc13x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc13x32, va13x0123, vb32x0123);
      vacc1x13x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x13x32, va13x4567, vb32x4567);
      vacc14x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc14x32, va14x0123, vb32x0123);
      vacc1x14x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x14x32, va14x4567, vb32x4567);
      vacc15x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc15x32, va15x0123, vb32x0123);
      vacc1x15x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x15x32, va15x4567, vb32x4567);

      w = (const int8_t*) w + 256;
      k -= 8 * sizeof(int8_t);
    }
    
    vacc0x32 = Q6_Vw_vadd_VwVw(vacc0x32, vacc1x0x32);
    vacc1x32 = Q6_Vw_vadd_VwVw(vacc1x32, vacc1x1x32);
    vacc2x32 = Q6_Vw_vadd_VwVw(vacc2x32, vacc1x2x32);
    vacc3x32 = Q6_Vw_vadd_VwVw(vacc3x32, vacc1x3x32);
    vacc4x32 = Q6_Vw_vadd_VwVw(vacc4x32, vacc1x4x32);
    vacc5x32 = Q6_Vw_vadd_VwVw(vacc5x32, vacc1x5x32);
    vacc6x32 = Q6_Vw_vadd_VwVw(vacc6x32, vacc1x6x32);
    vacc7x32 = Q6_Vw_vadd_VwVw(vacc7x32, vacc1x7x32);
    vacc8x32 = Q6_Vw_vadd_VwVw(vacc8x32, vacc1x8x32);
    vacc9x32 = Q6_Vw_vadd_VwVw(vacc9x32, vacc1x9x32);
    vacc10x32 = Q6_Vw_vadd_VwVw(vacc10x32, vacc1x10x32);
    vacc11x32 = Q6_Vw_vadd_VwVw(vacc11x32, vacc1x11x32);
    vacc12x32 = Q6_Vw_vadd_VwVw(vacc12x32, vacc1x12x32);
    vacc13x32 = Q6_Vw_vadd_VwVw(vacc13x32, vacc1x13x32);
    vacc14x32 = Q6_Vw_vadd_VwVw(vacc14x32, vacc1x14x32);
    vacc15x32 = Q6_Vw_vadd_VwVw(vacc15x32, vacc1x15x32);

    if (k != 0) {
      const HVX_Vector va0x0123 = Q6_V_vsplat_R(unaligned_load_s32(a0));
      a0 += 4;
      const HVX_Vector va1x0123 = Q6_V_vsplat_R(unaligned_load_s32(a1));
      a1 += 4;
      const HVX_Vector va2x0123 = Q6_V_vsplat_R(unaligned_load_s32(a2));
      a2 += 4;
      const HVX_Vector va3x0123 = Q6_V_vsplat_R(unaligned_load_s32(a3));
      a3 += 4;
      const HVX_Vector va4x0123 = Q6_V_vsplat_R(unaligned_load_s32(a4));
      a4 += 4;
      const HVX_Vector va5x0123 = Q6_V_vsplat_R(unaligned_load_s32(a5));
      a5 += 4;
      const HVX_Vector va6x0123 = Q6_V_vsplat_R(unaligned_load_s32(a6));
      a6 += 4;
      const HVX_Vector va7x0123 = Q6_V_vsplat_R(unaligned_load_s32(a7));
      a7 += 4;
      const HVX_Vector va8x0123 = Q6_V_vsplat_R(unaligned_load_s32(a8));
      a8 += 4;
      const HVX_Vector va9x0123 = Q6_V_vsplat_R(unaligned_load_s32(a9));
      a9 += 4;
      const HVX_Vector va10x0123 = Q6_V_vsplat_R(unaligned_load_s32(a10));
      a10 += 4;
      const HVX_Vector va11x0123 = Q6_V_vsplat_R(unaligned_load_s32(a11));
      a11 += 4;
      const HVX_Vector va12x0123 = Q6_V_vsplat_R(unaligned_load_s32(a12));
      a12 += 4;
      const HVX_Vector va13x0123 = Q6_V_vsplat_R(unaligned_load_s32(a13));
      a13 += 4;
      const HVX_Vector va14x0123 = Q6_V_vsplat_R(unaligned_load_s32(a14));
      a14 += 4;
      const HVX_Vector va15x0123 = Q6_V_vsplat_R(unaligned_load_s32(a15));
      a15 += 4;

      const HVX_Vector vb32x0123 = *((HVX_Vector *)((int8_t *)w));
      vacc0x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc0x32, va0x0123, vb32x0123);
      vacc1x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc1x32, va1x0123, vb32x0123);
      vacc2x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc2x32, va2x0123, vb32x0123);
      vacc3x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc3x32, va3x0123, vb32x0123);
      vacc4x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc4x32, va4x0123, vb32x0123);
      vacc5x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc5x32, va5x0123, vb32x0123);
      vacc6x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc6x32, va6x0123, vb32x0123);
      vacc7x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc7x32, va7x0123, vb32x0123);
      vacc8x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc8x32, va8x0123, vb32x0123);
      vacc9x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc9x32, va9x0123, vb32x0123);
      vacc10x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc10x32, va10x0123, vb32x0123);
      vacc11x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc11x32, va11x0123, vb32x0123);
      vacc12x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc12x32, va12x0123, vb32x0123);
      vacc13x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc13x32, va13x0123, vb32x0123);
      vacc14x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc14x32, va14x0123, vb32x0123);
      vacc15x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc15x32, va15x0123, vb32x0123);

      w = (const int8_t*) w + 128;
      k -= 4 * sizeof(int8_t);
    }
    const HVX_Vector vscale32 = *((HVX_Vector *)w);
    w = (const float*) w + 32;
    HVX_Vector vscaled0x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc0x32));
    HVX_Vector vscaled1x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc1x32));
    HVX_Vector vscaled2x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc2x32));
    HVX_Vector vscaled3x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc3x32));
    HVX_Vector vscaled4x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc4x32));
    HVX_Vector vscaled5x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc5x32));
    HVX_Vector vscaled6x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc6x32));
    HVX_Vector vscaled7x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc7x32));
    HVX_Vector vscaled8x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc8x32));
    HVX_Vector vscaled9x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc9x32));
    HVX_Vector vscaled10x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc10x32));
    HVX_Vector vscaled11x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc11x32));
    HVX_Vector vscaled12x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc12x32));
    HVX_Vector vscaled13x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc13x32));
    HVX_Vector vscaled14x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc14x32));
    HVX_Vector vscaled15x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc15x32));
  
    vscaled0x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled0x32, vscale32));
    vscaled1x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled1x32, vscale32));
    vscaled2x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled2x32, vscale32));
    vscaled3x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled3x32, vscale32));
    vscaled4x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled4x32, vscale32));
    vscaled5x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled5x32, vscale32));
    vscaled6x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled6x32, vscale32));
    vscaled7x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled7x32, vscale32));
    vscaled8x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled8x32, vscale32));
    vscaled9x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled9x32, vscale32));
    vscaled10x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled10x32, vscale32));
    vscaled11x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled11x32, vscale32));
    vscaled12x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled12x32, vscale32));
    vscaled13x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled13x32, vscale32));
    vscaled14x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled14x32, vscale32));
    vscaled15x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled15x32, vscale32));
    
    vscaled0x32 = Q6_Vsf_vmin_VsfVsf(vscaled0x32, voutput_max_less_zero_point);
    vscaled1x32 = Q6_Vsf_vmin_VsfVsf(vscaled1x32, voutput_max_less_zero_point);
    vscaled2x32 = Q6_Vsf_vmin_VsfVsf(vscaled2x32, voutput_max_less_zero_point);
    vscaled3x32 = Q6_Vsf_vmin_VsfVsf(vscaled3x32, voutput_max_less_zero_point);
    vscaled4x32 = Q6_Vsf_vmin_VsfVsf(vscaled4x32, voutput_max_less_zero_point);
    vscaled5x32 = Q6_Vsf_vmin_VsfVsf(vscaled5x32, voutput_max_less_zero_point);
    vscaled6x32 = Q6_Vsf_vmin_VsfVsf(vscaled6x32, voutput_max_less_zero_point);
    vscaled7x32 = Q6_Vsf_vmin_VsfVsf(vscaled7x32, voutput_max_less_zero_point);
    vscaled8x32 = Q6_Vsf_vmin_VsfVsf(vscaled8x32, voutput_max_less_zero_point);
    vscaled9x32 = Q6_Vsf_vmin_VsfVsf(vscaled9x32, voutput_max_less_zero_point);
    vscaled10x32 = Q6_Vsf_vmin_VsfVsf(vscaled10x32, voutput_max_less_zero_point);
    vscaled11x32 = Q6_Vsf_vmin_VsfVsf(vscaled11x32, voutput_max_less_zero_point);
    vscaled12x32 = Q6_Vsf_vmin_VsfVsf(vscaled12x32, voutput_max_less_zero_point);
    vscaled13x32 = Q6_Vsf_vmin_VsfVsf(vscaled13x32, voutput_max_less_zero_point);
    vscaled14x32 = Q6_Vsf_vmin_VsfVsf(vscaled14x32, voutput_max_less_zero_point);
    vscaled15x32 = Q6_Vsf_vmin_VsfVsf(vscaled15x32, voutput_max_less_zero_point);
 
    HVX_Vector vscaled0x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled0x32, Q6_V_vzero()); 
    HVX_Vector vscaled1x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled1x32, Q6_V_vzero()); 
    HVX_Vector vscaled2x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled2x32, Q6_V_vzero()); 
    HVX_Vector vscaled3x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled3x32, Q6_V_vzero()); 
    HVX_Vector vscaled4x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled4x32, Q6_V_vzero()); 
    HVX_Vector vscaled5x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled5x32, Q6_V_vzero()); 
    HVX_Vector vscaled6x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled6x32, Q6_V_vzero()); 
    HVX_Vector vscaled7x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled7x32, Q6_V_vzero()); 
    HVX_Vector vscaled8x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled8x32, Q6_V_vzero()); 
    HVX_Vector vscaled9x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled9x32, Q6_V_vzero()); 
    HVX_Vector vscaled10x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled10x32, Q6_V_vzero()); 
    HVX_Vector vscaled11x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled11x32, Q6_V_vzero()); 
    HVX_Vector vscaled12x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled12x32, Q6_V_vzero()); 
    HVX_Vector vscaled13x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled13x32, Q6_V_vzero()); 
    HVX_Vector vscaled14x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled14x32, Q6_V_vzero()); 
    HVX_Vector vscaled15x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled15x32, Q6_V_vzero()); 
   
    vacc0x32 = Q6_Vw_convert_Vqf32(vscaled0x32_qf);
    vacc1x32 = Q6_Vw_convert_Vqf32(vscaled1x32_qf);
    vacc2x32 = Q6_Vw_convert_Vqf32(vscaled2x32_qf);
    vacc3x32 = Q6_Vw_convert_Vqf32(vscaled3x32_qf);
    vacc4x32 = Q6_Vw_convert_Vqf32(vscaled4x32_qf);
    vacc5x32 = Q6_Vw_convert_Vqf32(vscaled5x32_qf);
    vacc6x32 = Q6_Vw_convert_Vqf32(vscaled6x32_qf);
    vacc7x32 = Q6_Vw_convert_Vqf32(vscaled7x32_qf);
    vacc8x32 = Q6_Vw_convert_Vqf32(vscaled8x32_qf);
    vacc9x32 = Q6_Vw_convert_Vqf32(vscaled9x32_qf);
    vacc10x32 = Q6_Vw_convert_Vqf32(vscaled10x32_qf);
    vacc11x32 = Q6_Vw_convert_Vqf32(vscaled11x32_qf);
    vacc12x32 = Q6_Vw_convert_Vqf32(vscaled12x32_qf);
    vacc13x32 = Q6_Vw_convert_Vqf32(vscaled13x32_qf);
    vacc14x32 = Q6_Vw_convert_Vqf32(vscaled14x32_qf);
    vacc15x32 = Q6_Vw_convert_Vqf32(vscaled15x32_qf);
    
    vacc0x32 = Q6_Vw_vadd_VwVw(vacc0x32, voutput_zero_point);
    vacc1x32 = Q6_Vw_vadd_VwVw(vacc1x32, voutput_zero_point);
    vacc2x32 = Q6_Vw_vadd_VwVw(vacc2x32, voutput_zero_point);
    vacc3x32 = Q6_Vw_vadd_VwVw(vacc3x32, voutput_zero_point);
    vacc4x32 = Q6_Vw_vadd_VwVw(vacc4x32, voutput_zero_point);
    vacc5x32 = Q6_Vw_vadd_VwVw(vacc5x32, voutput_zero_point);
    vacc6x32 = Q6_Vw_vadd_VwVw(vacc6x32, voutput_zero_point);
    vacc7x32 = Q6_Vw_vadd_VwVw(vacc7x32, voutput_zero_point);
    vacc8x32 = Q6_Vw_vadd_VwVw(vacc8x32, voutput_zero_point);
    vacc9x32 = Q6_Vw_vadd_VwVw(vacc9x32, voutput_zero_point);
    vacc10x32 = Q6_Vw_vadd_VwVw(vacc10x32, voutput_zero_point);
    vacc11x32 = Q6_Vw_vadd_VwVw(vacc11x32, voutput_zero_point);
    vacc12x32 = Q6_Vw_vadd_VwVw(vacc12x32, voutput_zero_point);
    vacc13x32 = Q6_Vw_vadd_VwVw(vacc13x32, voutput_zero_point);
    vacc14x32 = Q6_Vw_vadd_VwVw(vacc14x32, voutput_zero_point);
    vacc15x32 = Q6_Vw_vadd_VwVw(vacc15x32, voutput_zero_point);

    HVX_Vector vout0x32 =  Q6_Vh_vpack_VwVw_sat(vacc0x32, vacc0x32);
    HVX_Vector vout1x32 =  Q6_Vh_vpack_VwVw_sat(vacc1x32, vacc1x32);
    HVX_Vector vout2x32 =  Q6_Vh_vpack_VwVw_sat(vacc2x32, vacc2x32);
    HVX_Vector vout3x32 =  Q6_Vh_vpack_VwVw_sat(vacc3x32, vacc3x32);
    HVX_Vector vout4x32 =  Q6_Vh_vpack_VwVw_sat(vacc4x32, vacc4x32);
    HVX_Vector vout5x32 =  Q6_Vh_vpack_VwVw_sat(vacc5x32, vacc5x32);
    HVX_Vector vout6x32 =  Q6_Vh_vpack_VwVw_sat(vacc6x32, vacc6x32);
    HVX_Vector vout7x32 =  Q6_Vh_vpack_VwVw_sat(vacc7x32, vacc7x32);
    HVX_Vector vout8x32 =  Q6_Vh_vpack_VwVw_sat(vacc8x32, vacc8x32);
    HVX_Vector vout9x32 =  Q6_Vh_vpack_VwVw_sat(vacc9x32, vacc9x32);
    HVX_Vector vout10x32 =  Q6_Vh_vpack_VwVw_sat(vacc10x32, vacc10x32);
    HVX_Vector vout11x32 =  Q6_Vh_vpack_VwVw_sat(vacc11x32, vacc11x32);
    HVX_Vector vout12x32 =  Q6_Vh_vpack_VwVw_sat(vacc12x32, vacc12x32);
    HVX_Vector vout13x32 =  Q6_Vh_vpack_VwVw_sat(vacc13x32, vacc13x32);
    HVX_Vector vout14x32 =  Q6_Vh_vpack_VwVw_sat(vacc14x32, vacc14x32);
    HVX_Vector vout15x32 =  Q6_Vh_vpack_VwVw_sat(vacc15x32, vacc15x32);
    
    vout0x32 = Q6_Vb_vpack_VhVh_sat(vout0x32, vout0x32);
    vout1x32 = Q6_Vb_vpack_VhVh_sat(vout1x32, vout1x32);
    vout2x32 = Q6_Vb_vpack_VhVh_sat(vout2x32, vout2x32);
    vout3x32 = Q6_Vb_vpack_VhVh_sat(vout3x32, vout3x32);
    vout4x32 = Q6_Vb_vpack_VhVh_sat(vout4x32, vout4x32);
    vout5x32 = Q6_Vb_vpack_VhVh_sat(vout5x32, vout5x32);
    vout6x32 = Q6_Vb_vpack_VhVh_sat(vout6x32, vout6x32);
    vout7x32 = Q6_Vb_vpack_VhVh_sat(vout7x32, vout7x32);
    vout8x32 = Q6_Vb_vpack_VhVh_sat(vout8x32, vout8x32);
    vout9x32 = Q6_Vb_vpack_VhVh_sat(vout9x32, vout9x32);
    vout10x32 = Q6_Vb_vpack_VhVh_sat(vout10x32, vout10x32);
    vout11x32 = Q6_Vb_vpack_VhVh_sat(vout11x32, vout11x32);
    vout12x32 = Q6_Vb_vpack_VhVh_sat(vout12x32, vout12x32);
    vout13x32 = Q6_Vb_vpack_VhVh_sat(vout13x32, vout13x32);
    vout14x32 = Q6_Vb_vpack_VhVh_sat(vout14x32, vout14x32);
    vout15x32 = Q6_Vb_vpack_VhVh_sat(vout15x32, vout15x32);
    
    vout0x32 = Q6_Vb_vmax_VbVb(vout0x32, voutput_min);
    vout1x32 = Q6_Vb_vmax_VbVb(vout1x32, voutput_min);
    vout2x32 = Q6_Vb_vmax_VbVb(vout2x32, voutput_min);
    vout3x32 = Q6_Vb_vmax_VbVb(vout3x32, voutput_min);
    vout4x32 = Q6_Vb_vmax_VbVb(vout4x32, voutput_min);
    vout5x32 = Q6_Vb_vmax_VbVb(vout5x32, voutput_min);
    vout6x32 = Q6_Vb_vmax_VbVb(vout6x32, voutput_min);
    vout7x32 = Q6_Vb_vmax_VbVb(vout7x32, voutput_min);
    vout8x32 = Q6_Vb_vmax_VbVb(vout8x32, voutput_min);
    vout9x32 = Q6_Vb_vmax_VbVb(vout9x32, voutput_min);
    vout10x32 = Q6_Vb_vmax_VbVb(vout10x32, voutput_min);
    vout11x32 = Q6_Vb_vmax_VbVb(vout11x32, voutput_min);
    vout12x32 = Q6_Vb_vmax_VbVb(vout12x32, voutput_min);
    vout13x32 = Q6_Vb_vmax_VbVb(vout13x32, voutput_min);
    vout14x32 = Q6_Vb_vmax_VbVb(vout14x32, voutput_min);
    vout15x32 = Q6_Vb_vmax_VbVb(vout15x32, voutput_min);
    
   if XNN_LIKELY(nc >= 32) {
      Q6_V_vstu_variable(c0, 32, vout0x32);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      Q6_V_vstu_variable(c1, 32, vout1x32);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      Q6_V_vstu_variable(c2, 32, vout2x32);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      Q6_V_vstu_variable(c3, 32, vout3x32);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);
      Q6_V_vstu_variable(c4, 32, vout4x32);
      c4 = (int8_t*) ((uintptr_t) c4 + cn_stride);
      a4 = (const int8_t*) ((uintptr_t) a4 - kc);
      Q6_V_vstu_variable(c5, 32, vout5x32);
      c5 = (int8_t*) ((uintptr_t) c5 + cn_stride);
      a5 = (const int8_t*) ((uintptr_t) a5 - kc);
      Q6_V_vstu_variable(c6, 32, vout6x32);
      c6 = (int8_t*) ((uintptr_t) c6 + cn_stride);
      a6 = (const int8_t*) ((uintptr_t) a6 - kc);
      Q6_V_vstu_variable(c7, 32, vout7x32);
      c7 = (int8_t*) ((uintptr_t) c7 + cn_stride);
      a7 = (const int8_t*) ((uintptr_t) a7 - kc);
      Q6_V_vstu_variable(c8, 32, vout8x32);
      c8 = (int8_t*) ((uintptr_t) c8 + cn_stride);
      a8 = (const int8_t*) ((uintptr_t) a8 - kc);
      Q6_V_vstu_variable(c9, 32, vout9x32);
      c9 = (int8_t*) ((uintptr_t) c9 + cn_stride);
      a9 = (const int8_t*) ((uintptr_t) a9 - kc);
      Q6_V_vstu_variable(c10, 32, vout10x32);
      c10 = (int8_t*) ((uintptr_t) c10 + cn_stride);
      a10 = (const int8_t*) ((uintptr_t) a10 - kc);
      Q6_V_vstu_variable(c11, 32, vout11x32);
      c11 = (int8_t*) ((uintptr_t) c11 + cn_stride);
      a11 = (const int8_t*) ((uintptr_t) a11 - kc);
      Q6_V_vstu_variable(c12, 32, vout12x32);
      c12 = (int8_t*) ((uintptr_t) c12 + cn_stride);
      a12 = (const int8_t*) ((uintptr_t) a12 - kc);
      Q6_V_vstu_variable(c13, 32, vout13x32);
      c13 = (int8_t*) ((uintptr_t) c13 + cn_stride);
      a13 = (const int8_t*) ((uintptr_t) a13 - kc);
      Q6_V_vstu_variable(c14, 32, vout14x32);
      c14 = (int8_t*) ((uintptr_t) c14 + cn_stride);
      a14 = (const int8_t*) ((uintptr_t) a14 - kc);
      Q6_V_vstu_variable(c15, 32, vout15x32);
      c15 = (int8_t*) ((uintptr_t) c15 + cn_stride);
      a15 = (const int8_t*) ((uintptr_t) a15 - kc);

      nc -= 32;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      Q6_V_vstu_variable(c0, nc, vout0x32);
      Q6_V_vstu_variable(c1, nc, vout1x32);
      Q6_V_vstu_variable(c2, nc, vout2x32);
      Q6_V_vstu_variable(c3, nc, vout3x32);
      Q6_V_vstu_variable(c4, nc, vout4x32);
      Q6_V_vstu_variable(c5, nc, vout5x32);
      Q6_V_vstu_variable(c6, nc, vout6x32);
      Q6_V_vstu_variable(c7, nc, vout7x32);
      Q6_V_vstu_variable(c8, nc, vout8x32);
      Q6_V_vstu_variable(c9, nc, vout9x32);
      Q6_V_vstu_variable(c10, nc, vout10x32);
      Q6_V_vstu_variable(c11, nc, vout11x32);
      Q6_V_vstu_variable(c12, nc, vout12x32);
      Q6_V_vstu_variable(c13, nc, vout13x32);
      Q6_V_vstu_variable(c14, nc, vout14x32);
      Q6_V_vstu_variable(c15, nc, vout15x32);
      nc = 0;
    }
  } while (nc != 0);
}
