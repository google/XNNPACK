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
#if __HVX_ARCH__ >= 73
static XNN_INLINE HVX_Vector rescale_fp32(HVX_Vector vacc, HVX_Vector vscale)
{
  const HVX_Vector vaccf = Q6_Vsf_equals_Vw(vacc);
  const HVX_Vector vscaledqf = Q6_Vqf32_vmpy_VsfVsf(vaccf, vscale);

  // Create a vector of `0.5f` with the same sign as the entries of `a`.
  const HVX_Vector vhalf = Q6_V_vsplat_R(float_as_uint32(0.5f));
  const HVX_Vector vsign_mask = Q6_V_vsplat_R(0x80000000);
  const HVX_Vector vsigned_half = Q6_V_vor_VV(Q6_V_vand_VV(vaccf, vsign_mask), vhalf);
  const HVX_Vector vresult = Q6_Vw_equals_Vsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(vscaledqf, vsigned_half)));
  return vresult;
}
#else
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
#endif  // __HVX_ARCH__ >= 73

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x128c4__hvx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params* restrict params) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 7);
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

  const HVX_Vector voutput_zero_point = Q6_Vh_vsplat_R(params->fp32_scalar.output_zero_point);
  const HVX_Vector voutput_min = Q6_Vb_vsplat_R(params->fp32_scalar.output_min);
  const HVX_Vector voutput_max = Q6_Vb_vsplat_R(params->fp32_scalar.output_max);

  do {
    HVX_Vector vacc0x0 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
    HVX_Vector vacc0x1 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
    HVX_Vector vacc0x2 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
    HVX_Vector vacc0x3 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
    HVX_Vector vacc1x0 = vacc0x0;
    HVX_Vector vacc1x1 = vacc0x1;
    HVX_Vector vacc1x2 = vacc0x2;
    HVX_Vector vacc1x3 = vacc0x3;
    HVX_Vector vacc2x0 = vacc0x0;
    HVX_Vector vacc2x1 = vacc0x1;
    HVX_Vector vacc2x2 = vacc0x2;
    HVX_Vector vacc2x3 = vacc0x3;
    HVX_Vector vacc3x0 = vacc0x0;
    HVX_Vector vacc3x1 = vacc0x1;
    HVX_Vector vacc3x2 = vacc0x2;
    HVX_Vector vacc3x3 = vacc0x3;
    HVX_Vector vacc4x0 = vacc0x0;
    HVX_Vector vacc4x1 = vacc0x1;
    HVX_Vector vacc4x2 = vacc0x2;
    HVX_Vector vacc4x3 = vacc0x3;
    HVX_Vector vacc5x0 = vacc0x0;
    HVX_Vector vacc5x1 = vacc0x1;
    HVX_Vector vacc5x2 = vacc0x2;
    HVX_Vector vacc5x3 = vacc0x3;
    HVX_Vector vacc6x0 = vacc0x0;
    HVX_Vector vacc6x1 = vacc0x1;
    HVX_Vector vacc6x2 = vacc0x2;
    HVX_Vector vacc6x3 = vacc0x3;

    size_t k = kc;
    for (; k >= 4 * sizeof(int8_t); k -= 4 * sizeof(int8_t)) {
      const HVX_Vector va0x0123 = Q6_V_vsplat_R(unaligned_load_s32(a0)); a0 += 4;
      const HVX_Vector va1x0123 = Q6_V_vsplat_R(unaligned_load_s32(a1)); a1 += 4;
      const HVX_Vector va2x0123 = Q6_V_vsplat_R(unaligned_load_s32(a2)); a2 += 4;
      const HVX_Vector va3x0123 = Q6_V_vsplat_R(unaligned_load_s32(a3)); a3 += 4;
      const HVX_Vector va4x0123 = Q6_V_vsplat_R(unaligned_load_s32(a4)); a4 += 4;
      const HVX_Vector va5x0123 = Q6_V_vsplat_R(unaligned_load_s32(a5)); a5 += 4;
      const HVX_Vector va6x0123 = Q6_V_vsplat_R(unaligned_load_s32(a6)); a6 += 4;

      const HVX_Vector vb0x0123 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
      const HVX_Vector vb1x0123 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
      const HVX_Vector vb2x0123 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
      const HVX_Vector vb3x0123 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;

      vacc0x0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x0, va0x0123, vb0x0123);
      vacc0x1 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x1, va0x0123, vb1x0123);
      vacc0x2 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x2, va0x0123, vb2x0123);
      vacc0x3 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x3, va0x0123, vb3x0123);
      vacc1x0 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x0, va1x0123, vb0x0123);
      vacc1x1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x1, va1x0123, vb1x0123);
      vacc1x2 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x2, va1x0123, vb2x0123);
      vacc1x3 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x3, va1x0123, vb3x0123);
      vacc2x0 = Q6_Vw_vrmpyacc_VwVbVb(vacc2x0, va2x0123, vb0x0123);
      vacc2x1 = Q6_Vw_vrmpyacc_VwVbVb(vacc2x1, va2x0123, vb1x0123);
      vacc2x2 = Q6_Vw_vrmpyacc_VwVbVb(vacc2x2, va2x0123, vb2x0123);
      vacc2x3 = Q6_Vw_vrmpyacc_VwVbVb(vacc2x3, va2x0123, vb3x0123);
      vacc3x0 = Q6_Vw_vrmpyacc_VwVbVb(vacc3x0, va3x0123, vb0x0123);
      vacc3x1 = Q6_Vw_vrmpyacc_VwVbVb(vacc3x1, va3x0123, vb1x0123);
      vacc3x2 = Q6_Vw_vrmpyacc_VwVbVb(vacc3x2, va3x0123, vb2x0123);
      vacc3x3 = Q6_Vw_vrmpyacc_VwVbVb(vacc3x3, va3x0123, vb3x0123);
      vacc4x0 = Q6_Vw_vrmpyacc_VwVbVb(vacc4x0, va4x0123, vb0x0123);
      vacc4x1 = Q6_Vw_vrmpyacc_VwVbVb(vacc4x1, va4x0123, vb1x0123);
      vacc4x2 = Q6_Vw_vrmpyacc_VwVbVb(vacc4x2, va4x0123, vb2x0123);
      vacc4x3 = Q6_Vw_vrmpyacc_VwVbVb(vacc4x3, va4x0123, vb3x0123);
      vacc5x0 = Q6_Vw_vrmpyacc_VwVbVb(vacc5x0, va5x0123, vb0x0123);
      vacc5x1 = Q6_Vw_vrmpyacc_VwVbVb(vacc5x1, va5x0123, vb1x0123);
      vacc5x2 = Q6_Vw_vrmpyacc_VwVbVb(vacc5x2, va5x0123, vb2x0123);
      vacc5x3 = Q6_Vw_vrmpyacc_VwVbVb(vacc5x3, va5x0123, vb3x0123);
      vacc6x0 = Q6_Vw_vrmpyacc_VwVbVb(vacc6x0, va6x0123, vb0x0123);
      vacc6x1 = Q6_Vw_vrmpyacc_VwVbVb(vacc6x1, va6x0123, vb1x0123);
      vacc6x2 = Q6_Vw_vrmpyacc_VwVbVb(vacc6x2, va6x0123, vb2x0123);
      vacc6x3 = Q6_Vw_vrmpyacc_VwVbVb(vacc6x3, va6x0123, vb3x0123);
    }

    const HVX_Vector vscale0 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
    vacc0x0 = rescale_fp32(vacc0x0, vscale0);
    vacc1x0 = rescale_fp32(vacc1x0, vscale0);
    vacc2x0 = rescale_fp32(vacc2x0, vscale0);
    vacc3x0 = rescale_fp32(vacc3x0, vscale0);
    vacc4x0 = rescale_fp32(vacc4x0, vscale0);
    vacc5x0 = rescale_fp32(vacc5x0, vscale0);
    vacc6x0 = rescale_fp32(vacc6x0, vscale0);
    const HVX_Vector vscale1 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
    vacc0x1 = rescale_fp32(vacc0x1, vscale1);
    vacc1x1 = rescale_fp32(vacc1x1, vscale1);
    vacc2x1 = rescale_fp32(vacc2x1, vscale1);
    vacc3x1 = rescale_fp32(vacc3x1, vscale1);
    vacc4x1 = rescale_fp32(vacc4x1, vscale1);
    vacc5x1 = rescale_fp32(vacc5x1, vscale1);
    vacc6x1 = rescale_fp32(vacc6x1, vscale1);
    const HVX_Vector vscale2 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
    vacc0x2 = rescale_fp32(vacc0x2, vscale2);
    vacc1x2 = rescale_fp32(vacc1x2, vscale2);
    vacc2x2 = rescale_fp32(vacc2x2, vscale2);
    vacc3x2 = rescale_fp32(vacc3x2, vscale2);
    vacc4x2 = rescale_fp32(vacc4x2, vscale2);
    vacc5x2 = rescale_fp32(vacc5x2, vscale2);
    vacc6x2 = rescale_fp32(vacc6x2, vscale2);
    const HVX_Vector vscale3 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
    vacc0x3 = rescale_fp32(vacc0x3, vscale3);
    vacc1x3 = rescale_fp32(vacc1x3, vscale3);
    vacc2x3 = rescale_fp32(vacc2x3, vscale3);
    vacc3x3 = rescale_fp32(vacc3x3, vscale3);
    vacc4x3 = rescale_fp32(vacc4x3, vscale3);
    vacc5x3 = rescale_fp32(vacc5x3, vscale3);
    vacc6x3 = rescale_fp32(vacc6x3, vscale3);

    HVX_Vector vout0x0 = Q6_Vh_vpack_VwVw_sat(vacc0x1, vacc0x0);
    HVX_Vector vout0x1 = Q6_Vh_vpack_VwVw_sat(vacc0x3, vacc0x2);
    HVX_Vector vout1x0 = Q6_Vh_vpack_VwVw_sat(vacc1x1, vacc1x0);
    HVX_Vector vout1x1 = Q6_Vh_vpack_VwVw_sat(vacc1x3, vacc1x2);
    HVX_Vector vout2x0 = Q6_Vh_vpack_VwVw_sat(vacc2x1, vacc2x0);
    HVX_Vector vout2x1 = Q6_Vh_vpack_VwVw_sat(vacc2x3, vacc2x2);
    HVX_Vector vout3x0 = Q6_Vh_vpack_VwVw_sat(vacc3x1, vacc3x0);
    HVX_Vector vout3x1 = Q6_Vh_vpack_VwVw_sat(vacc3x3, vacc3x2);
    HVX_Vector vout4x0 = Q6_Vh_vpack_VwVw_sat(vacc4x1, vacc4x0);
    HVX_Vector vout4x1 = Q6_Vh_vpack_VwVw_sat(vacc4x3, vacc4x2);
    HVX_Vector vout5x0 = Q6_Vh_vpack_VwVw_sat(vacc5x1, vacc5x0);
    HVX_Vector vout5x1 = Q6_Vh_vpack_VwVw_sat(vacc5x3, vacc5x2);
    HVX_Vector vout6x0 = Q6_Vh_vpack_VwVw_sat(vacc6x1, vacc6x0);
    HVX_Vector vout6x1 = Q6_Vh_vpack_VwVw_sat(vacc6x3, vacc6x2);

    vout0x0 = Q6_Vh_vadd_VhVh_sat(vout0x0, voutput_zero_point);
    vout0x1 = Q6_Vh_vadd_VhVh_sat(vout0x1, voutput_zero_point);
    vout1x0 = Q6_Vh_vadd_VhVh_sat(vout1x0, voutput_zero_point);
    vout1x1 = Q6_Vh_vadd_VhVh_sat(vout1x1, voutput_zero_point);
    vout2x0 = Q6_Vh_vadd_VhVh_sat(vout2x0, voutput_zero_point);
    vout2x1 = Q6_Vh_vadd_VhVh_sat(vout2x1, voutput_zero_point);
    vout3x0 = Q6_Vh_vadd_VhVh_sat(vout3x0, voutput_zero_point);
    vout3x1 = Q6_Vh_vadd_VhVh_sat(vout3x1, voutput_zero_point);
    vout4x0 = Q6_Vh_vadd_VhVh_sat(vout4x0, voutput_zero_point);
    vout4x1 = Q6_Vh_vadd_VhVh_sat(vout4x1, voutput_zero_point);
    vout5x0 = Q6_Vh_vadd_VhVh_sat(vout5x0, voutput_zero_point);
    vout5x1 = Q6_Vh_vadd_VhVh_sat(vout5x1, voutput_zero_point);
    vout6x0 = Q6_Vh_vadd_VhVh_sat(vout6x0, voutput_zero_point);
    vout6x1 = Q6_Vh_vadd_VhVh_sat(vout6x1, voutput_zero_point);

    HVX_Vector vout0 = Q6_Vb_vpack_VhVh_sat(vout0x1, vout0x0);
    HVX_Vector vout1 = Q6_Vb_vpack_VhVh_sat(vout1x1, vout1x0);
    HVX_Vector vout2 = Q6_Vb_vpack_VhVh_sat(vout2x1, vout2x0);
    HVX_Vector vout3 = Q6_Vb_vpack_VhVh_sat(vout3x1, vout3x0);
    HVX_Vector vout4 = Q6_Vb_vpack_VhVh_sat(vout4x1, vout4x0);
    HVX_Vector vout5 = Q6_Vb_vpack_VhVh_sat(vout5x1, vout5x0);
    HVX_Vector vout6 = Q6_Vb_vpack_VhVh_sat(vout6x1, vout6x0);

    vout0 = Q6_Vb_vmax_VbVb(vout0, voutput_min);
    vout1 = Q6_Vb_vmax_VbVb(vout1, voutput_min);
    vout2 = Q6_Vb_vmax_VbVb(vout2, voutput_min);
    vout3 = Q6_Vb_vmax_VbVb(vout3, voutput_min);
    vout4 = Q6_Vb_vmax_VbVb(vout4, voutput_min);
    vout5 = Q6_Vb_vmax_VbVb(vout5, voutput_min);
    vout6 = Q6_Vb_vmax_VbVb(vout6, voutput_min);

    vout0 = Q6_Vb_vmin_VbVb(vout0, voutput_max);
    vout1 = Q6_Vb_vmin_VbVb(vout1, voutput_max);
    vout2 = Q6_Vb_vmin_VbVb(vout2, voutput_max);
    vout3 = Q6_Vb_vmin_VbVb(vout3, voutput_max);
    vout4 = Q6_Vb_vmin_VbVb(vout4, voutput_max);
    vout5 = Q6_Vb_vmin_VbVb(vout5, voutput_max);
    vout6 = Q6_Vb_vmin_VbVb(vout6, voutput_max);

    if XNN_LIKELY(nc >= 128) {
      *((HVX_UVector *)c0) = vout0;
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      *((HVX_UVector *)c1) = vout1;
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      *((HVX_UVector *)c2) = vout2;
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      *((HVX_UVector *)c3) = vout3;
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);
      *((HVX_UVector *)c4) = vout4;
      c4 = (int8_t*) ((uintptr_t) c4 + cn_stride);
      a4 = (const int8_t*) ((uintptr_t) a4 - kc);
      *((HVX_UVector *)c5) = vout5;
      c5 = (int8_t*) ((uintptr_t) c5 + cn_stride);
      a5 = (const int8_t*) ((uintptr_t) a5 - kc);
      *((HVX_UVector *)c6) = vout6;
      c6 = (int8_t*) ((uintptr_t) c6 + cn_stride);
      a6 = (const int8_t*) ((uintptr_t) a6 - kc);

      nc -= 128;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      Q6_V_vstu_variable(c0, nc, vout0);
      Q6_V_vstu_variable(c1, nc, vout1);
      Q6_V_vstu_variable(c2, nc, vout2);
      Q6_V_vstu_variable(c3, nc, vout3);
      Q6_V_vstu_variable(c4, nc, vout4);
      Q6_V_vstu_variable(c5, nc, vout5);
      Q6_V_vstu_variable(c6, nc, vout6);
      nc = 0;
    }
  } while (nc != 0);
}
