// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c4-hvx.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>  // for lrintf

#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"  // for Q6_V_vstu_variable
#include "src/xnnpack/math.h"
#include "src/xnnpack/unaligned.h"



// multiply vacc by vscale and return result as int
// vacc is vector of int32
// vscale is vector of floats
// return is vector of int
static HVX_Vector rescale_fp32(HVX_Vector vacc, HVX_Vector vscale)
{
  XNN_ALIGN(128) int32_t vacc_buffer[32];
  XNN_ALIGN(128) float vscale_buffer[32];

  *((HVX_Vector *)&vacc_buffer) = vacc;
  *((HVX_Vector *)&vscale_buffer) = vscale;

  for (int i = 0; i < 32; ++i) {
    vacc_buffer[i] = (int32_t)lrintf((float)vacc_buffer[i] * vscale_buffer[i]);
  }
  return *(HVX_Vector *)&vacc_buffer;
}

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

  const HVX_Vector voutput_zero_point = Q6_Vh_vsplat_R(params->fp32_scalar.output_zero_point);
  const HVX_Vector voutput_min = Q6_Vb_vsplat_R(params->fp32_scalar.output_min);
  const HVX_Vector voutput_max = Q6_Vb_vsplat_R(params->fp32_scalar.output_max);

  do {
    HVX_Vector vacc0x32 = *((HVX_Vector*)w); w = (const int32_t*) w + 32;
    HVX_Vector vacc1x32 = vacc0x32;
    HVX_Vector vacc2x32 = vacc0x32;
    HVX_Vector vacc3x32 = vacc0x32;
    HVX_Vector vacc4x32 = vacc0x32;
    HVX_Vector vacc5x32 = vacc0x32;
    HVX_Vector vacc6x32 = vacc0x32;
    HVX_Vector vacc7x32 = vacc0x32;
    HVX_Vector vacc8x32 = vacc0x32;
    HVX_Vector vacc9x32 = vacc0x32;
    HVX_Vector vacc10x32 = vacc0x32;
    HVX_Vector vacc11x32 = vacc0x32;
    HVX_Vector vacc12x32 = vacc0x32;
    HVX_Vector vacc13x32 = vacc0x32;
    HVX_Vector vacc14x32 = vacc0x32;
    HVX_Vector vacc15x32 = vacc0x32;

    size_t k = kc;
    for (; k >= 4 * sizeof(int8_t); k -= 4 * sizeof(int8_t)) {
      const HVX_Vector va0x0123 = Q6_V_vsplat_R(unaligned_load_s32(a0)); a0 += 4;
      const HVX_Vector va1x0123 = Q6_V_vsplat_R(unaligned_load_s32(a1)); a1 += 4;
      const HVX_Vector va2x0123 = Q6_V_vsplat_R(unaligned_load_s32(a2)); a2 += 4;
      const HVX_Vector va3x0123 = Q6_V_vsplat_R(unaligned_load_s32(a3)); a3 += 4;
      const HVX_Vector va4x0123 = Q6_V_vsplat_R(unaligned_load_s32(a4)); a4 += 4;
      const HVX_Vector va5x0123 = Q6_V_vsplat_R(unaligned_load_s32(a5)); a5 += 4;
      const HVX_Vector va6x0123 = Q6_V_vsplat_R(unaligned_load_s32(a6)); a6 += 4;
      const HVX_Vector va7x0123 = Q6_V_vsplat_R(unaligned_load_s32(a7)); a7 += 4;
      const HVX_Vector va8x0123 = Q6_V_vsplat_R(unaligned_load_s32(a8)); a8 += 4;
      const HVX_Vector va9x0123 = Q6_V_vsplat_R(unaligned_load_s32(a9)); a9 += 4;
      const HVX_Vector va10x0123 = Q6_V_vsplat_R(unaligned_load_s32(a10)); a10 += 4;
      const HVX_Vector va11x0123 = Q6_V_vsplat_R(unaligned_load_s32(a11)); a11 += 4;
      const HVX_Vector va12x0123 = Q6_V_vsplat_R(unaligned_load_s32(a12)); a12 += 4;
      const HVX_Vector va13x0123 = Q6_V_vsplat_R(unaligned_load_s32(a13)); a13 += 4;
      const HVX_Vector va14x0123 = Q6_V_vsplat_R(unaligned_load_s32(a14)); a14 += 4;
      const HVX_Vector va15x0123 = Q6_V_vsplat_R(unaligned_load_s32(a15)); a15 += 4;

      const HVX_Vector vb32x0123 = *((HVX_Vector *)((int8_t *)w)); w = (const int8_t*) w + 128;

      vacc0x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x32, va0x0123, vb32x0123);
      vacc1x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x32, va1x0123, vb32x0123);
      vacc2x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc2x32, va2x0123, vb32x0123);
      vacc3x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc3x32, va3x0123, vb32x0123);
      vacc4x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc4x32, va4x0123, vb32x0123);
      vacc5x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc5x32, va5x0123, vb32x0123);
      vacc6x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc6x32, va6x0123, vb32x0123);
      vacc7x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc7x32, va7x0123, vb32x0123);
      vacc8x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc8x32, va8x0123, vb32x0123);
      vacc9x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc9x32, va9x0123, vb32x0123);
      vacc10x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc10x32, va10x0123, vb32x0123);
      vacc11x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc11x32, va11x0123, vb32x0123);
      vacc12x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc12x32, va12x0123, vb32x0123);
      vacc13x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc13x32, va13x0123, vb32x0123);
      vacc14x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc14x32, va14x0123, vb32x0123);
      vacc15x32 = Q6_Vw_vrmpyacc_VwVbVb(vacc15x32, va15x0123, vb32x0123);
    }

    const HVX_Vector vscale32 = *((HVX_Vector *)w); w = (const float*) w + 32;
    vacc0x32 = rescale_fp32(vacc0x32, vscale32);
    vacc1x32 = rescale_fp32(vacc1x32, vscale32);
    vacc2x32 = rescale_fp32(vacc2x32, vscale32);
    vacc3x32 = rescale_fp32(vacc3x32, vscale32);
    vacc4x32 = rescale_fp32(vacc4x32, vscale32);
    vacc5x32 = rescale_fp32(vacc5x32, vscale32);
    vacc6x32 = rescale_fp32(vacc6x32, vscale32);
    vacc7x32 = rescale_fp32(vacc7x32, vscale32);
    vacc8x32 = rescale_fp32(vacc8x32, vscale32);
    vacc9x32 = rescale_fp32(vacc9x32, vscale32);
    vacc10x32 = rescale_fp32(vacc10x32, vscale32);
    vacc11x32 = rescale_fp32(vacc11x32, vscale32);
    vacc12x32 = rescale_fp32(vacc12x32, vscale32);
    vacc13x32 = rescale_fp32(vacc13x32, vscale32);
    vacc14x32 = rescale_fp32(vacc14x32, vscale32);
    vacc15x32 = rescale_fp32(vacc15x32, vscale32);

    HVX_Vector vout0x32 = Q6_Vh_vpack_VwVw_sat(vacc0x32, vacc0x32);
    HVX_Vector vout1x32 = Q6_Vh_vpack_VwVw_sat(vacc1x32, vacc1x32);
    HVX_Vector vout2x32 = Q6_Vh_vpack_VwVw_sat(vacc2x32, vacc2x32);
    HVX_Vector vout3x32 = Q6_Vh_vpack_VwVw_sat(vacc3x32, vacc3x32);
    HVX_Vector vout4x32 = Q6_Vh_vpack_VwVw_sat(vacc4x32, vacc4x32);
    HVX_Vector vout5x32 = Q6_Vh_vpack_VwVw_sat(vacc5x32, vacc5x32);
    HVX_Vector vout6x32 = Q6_Vh_vpack_VwVw_sat(vacc6x32, vacc6x32);
    HVX_Vector vout7x32 = Q6_Vh_vpack_VwVw_sat(vacc7x32, vacc7x32);
    HVX_Vector vout8x32 = Q6_Vh_vpack_VwVw_sat(vacc8x32, vacc8x32);
    HVX_Vector vout9x32 = Q6_Vh_vpack_VwVw_sat(vacc9x32, vacc9x32);
    HVX_Vector vout10x32 = Q6_Vh_vpack_VwVw_sat(vacc10x32, vacc10x32);
    HVX_Vector vout11x32 = Q6_Vh_vpack_VwVw_sat(vacc11x32, vacc11x32);
    HVX_Vector vout12x32 = Q6_Vh_vpack_VwVw_sat(vacc12x32, vacc12x32);
    HVX_Vector vout13x32 = Q6_Vh_vpack_VwVw_sat(vacc13x32, vacc13x32);
    HVX_Vector vout14x32 = Q6_Vh_vpack_VwVw_sat(vacc14x32, vacc14x32);
    HVX_Vector vout15x32 = Q6_Vh_vpack_VwVw_sat(vacc15x32, vacc15x32);

    vout0x32 = Q6_Vh_vadd_VhVh_sat(vout0x32, voutput_zero_point);
    vout1x32 = Q6_Vh_vadd_VhVh_sat(vout1x32, voutput_zero_point);
    vout2x32 = Q6_Vh_vadd_VhVh_sat(vout2x32, voutput_zero_point);
    vout3x32 = Q6_Vh_vadd_VhVh_sat(vout3x32, voutput_zero_point);
    vout4x32 = Q6_Vh_vadd_VhVh_sat(vout4x32, voutput_zero_point);
    vout5x32 = Q6_Vh_vadd_VhVh_sat(vout5x32, voutput_zero_point);
    vout6x32 = Q6_Vh_vadd_VhVh_sat(vout6x32, voutput_zero_point);
    vout7x32 = Q6_Vh_vadd_VhVh_sat(vout7x32, voutput_zero_point);
    vout8x32 = Q6_Vh_vadd_VhVh_sat(vout8x32, voutput_zero_point);
    vout9x32 = Q6_Vh_vadd_VhVh_sat(vout9x32, voutput_zero_point);
    vout10x32 = Q6_Vh_vadd_VhVh_sat(vout10x32, voutput_zero_point);
    vout11x32 = Q6_Vh_vadd_VhVh_sat(vout11x32, voutput_zero_point);
    vout12x32 = Q6_Vh_vadd_VhVh_sat(vout12x32, voutput_zero_point);
    vout13x32 = Q6_Vh_vadd_VhVh_sat(vout13x32, voutput_zero_point);
    vout14x32 = Q6_Vh_vadd_VhVh_sat(vout14x32, voutput_zero_point);
    vout15x32 = Q6_Vh_vadd_VhVh_sat(vout15x32, voutput_zero_point);

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

    vout0x32 = Q6_Vb_vmin_VbVb(vout0x32, voutput_max);
    vout1x32 = Q6_Vb_vmin_VbVb(vout1x32, voutput_max);
    vout2x32 = Q6_Vb_vmin_VbVb(vout2x32, voutput_max);
    vout3x32 = Q6_Vb_vmin_VbVb(vout3x32, voutput_max);
    vout4x32 = Q6_Vb_vmin_VbVb(vout4x32, voutput_max);
    vout5x32 = Q6_Vb_vmin_VbVb(vout5x32, voutput_max);
    vout6x32 = Q6_Vb_vmin_VbVb(vout6x32, voutput_max);
    vout7x32 = Q6_Vb_vmin_VbVb(vout7x32, voutput_max);
    vout8x32 = Q6_Vb_vmin_VbVb(vout8x32, voutput_max);
    vout9x32 = Q6_Vb_vmin_VbVb(vout9x32, voutput_max);
    vout10x32 = Q6_Vb_vmin_VbVb(vout10x32, voutput_max);
    vout11x32 = Q6_Vb_vmin_VbVb(vout11x32, voutput_max);
    vout12x32 = Q6_Vb_vmin_VbVb(vout12x32, voutput_max);
    vout13x32 = Q6_Vb_vmin_VbVb(vout13x32, voutput_max);
    vout14x32 = Q6_Vb_vmin_VbVb(vout14x32, voutput_max);
    vout15x32 = Q6_Vb_vmin_VbVb(vout15x32, voutput_max);

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
