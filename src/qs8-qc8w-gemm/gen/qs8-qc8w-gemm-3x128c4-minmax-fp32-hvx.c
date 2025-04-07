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

#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/unaligned.h"


void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x128c4__hvx(
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
  assert(mr <= 3);
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

  const float output_max_less_zero_point = (float)((int32_t) params->fp32_scalar.output_max - (int32_t)params->fp32_scalar.output_zero_point);
  const HVX_Vector voutput_max_less_zero_point = Q6_V_vsplat_R(*(uint32_t *)(&output_max_less_zero_point));
  const HVX_Vector voutput_zero_point = Q6_V_vsplat_R(params->fp32_scalar.output_zero_point);
  const HVX_Vector voutput_min = Q6_Vb_vsplat_R(params->fp32_scalar.output_min);

  do {
    HVX_Vector vacc0x0 = *((HVX_Vector*)w);
    HVX_Vector vacc0x1 = *((HVX_Vector*)w);
    HVX_Vector vacc0x2 = *((HVX_Vector*)w);
    HVX_Vector vacc0x3 = *((HVX_Vector*)w);
    HVX_Vector vacc1x0 = vacc0x0;
    HVX_Vector vacc2x0 = vacc0x0;
    HVX_Vector vacc1x1 = vacc0x1;
    HVX_Vector vacc2x1 = vacc0x1;
    HVX_Vector vacc1x2 = vacc0x2;
    HVX_Vector vacc2x2 = vacc0x2;
    HVX_Vector vacc1x3 = vacc0x3;
    HVX_Vector vacc2x3 = vacc0x3;

    w = (const int32_t*) w + 128;

    size_t k = kc;
    for (; k >= 4 * sizeof(int8_t); k -= 4 * sizeof(int8_t)) {
      const HVX_Vector va0x0123 = Q6_V_vsplat_R(unaligned_load_s32(a0)); a0 += 4;
      const HVX_Vector va1x0123 = Q6_V_vsplat_R(unaligned_load_s32(a1)); a1 += 4;
      const HVX_Vector va2x0123 = Q6_V_vsplat_R(unaligned_load_s32(a2)); a2 += 4;

      const HVX_Vector vb0x0123 = *((HVX_Vector *)((int8_t *)w) + 0);
      const HVX_Vector vb1x0123 = *((HVX_Vector *)((int8_t *)w) + 128);
      const HVX_Vector vb2x0123 = *((HVX_Vector *)((int8_t *)w) + 256);
      const HVX_Vector vb3x0123 = *((HVX_Vector *)((int8_t *)w) + 384);
      vacc0x0 =  Q6_Vw_vrmpyacc_VwVbVb(vacc0x0, va0x0123, vb0x0123);
      vacc0x1 =  Q6_Vw_vrmpyacc_VwVbVb(vacc0x1, va0x0123, vb1x0123);
      vacc0x2 =  Q6_Vw_vrmpyacc_VwVbVb(vacc0x2, va0x0123, vb2x0123);
      vacc0x3 =  Q6_Vw_vrmpyacc_VwVbVb(vacc0x3, va0x0123, vb3x0123);
      vacc1x0 =  Q6_Vw_vrmpyacc_VwVbVb(vacc1x0, va1x0123, vb0x0123);
      vacc1x1 =  Q6_Vw_vrmpyacc_VwVbVb(vacc1x1, va1x0123, vb1x0123);
      vacc1x2 =  Q6_Vw_vrmpyacc_VwVbVb(vacc1x2, va1x0123, vb2x0123);
      vacc1x3 =  Q6_Vw_vrmpyacc_VwVbVb(vacc1x3, va1x0123, vb3x0123);
      vacc2x0 =  Q6_Vw_vrmpyacc_VwVbVb(vacc2x0, va2x0123, vb0x0123);
      vacc2x1 =  Q6_Vw_vrmpyacc_VwVbVb(vacc2x1, va2x0123, vb1x0123);
      vacc2x2 =  Q6_Vw_vrmpyacc_VwVbVb(vacc2x2, va2x0123, vb2x0123);
      vacc2x3 =  Q6_Vw_vrmpyacc_VwVbVb(vacc2x3, va2x0123, vb3x0123);

      w = (const int8_t*) w + 512 * sizeof(int8_t);
    }

    const HVX_Vector vscale0 = *((const HVX_Vector *)((const float *)w) + 0);
    const HVX_Vector vscale1 = *((const HVX_Vector *)((const float *)w) + 32);
    const HVX_Vector vscale2 = *((const HVX_Vector *)((const float *)w) + 64);
    const HVX_Vector vscale3 = *((const HVX_Vector *)((const float *)w) + 96);
    w = (const float*) w + 128;

    HVX_Vector vscaled0x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc0x0));
    HVX_Vector vscaled0x1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc0x1));
    HVX_Vector vscaled0x2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc0x2));
    HVX_Vector vscaled0x3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc0x3));
    HVX_Vector vscaled1x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc1x0));
    HVX_Vector vscaled1x1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc1x1));
    HVX_Vector vscaled1x2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc1x2));
    HVX_Vector vscaled1x3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc1x3));
    HVX_Vector vscaled2x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc2x0));
    HVX_Vector vscaled2x1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc2x1));
    HVX_Vector vscaled2x2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc2x2));
    HVX_Vector vscaled2x3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc2x3));

    vscaled0x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled0x0, vscale0));
    vscaled0x1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled0x1, vscale1));
    vscaled0x2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled0x2, vscale2));
    vscaled0x3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled0x3, vscale3));
    vscaled1x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled1x0, vscale0));
    vscaled1x1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled1x1, vscale1));
    vscaled1x2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled1x2, vscale2));
    vscaled1x3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled1x3, vscale3));
    vscaled2x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled2x0, vscale0));
    vscaled2x1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled2x1, vscale1));
    vscaled2x2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled2x2, vscale2));
    vscaled2x3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled2x3, vscale3));

    vscaled0x0 = Q6_Vsf_vmin_VsfVsf(vscaled0x0, voutput_max_less_zero_point);
    vscaled0x1 = Q6_Vsf_vmin_VsfVsf(vscaled0x1, voutput_max_less_zero_point);
    vscaled0x2 = Q6_Vsf_vmin_VsfVsf(vscaled0x2, voutput_max_less_zero_point);
    vscaled0x3 = Q6_Vsf_vmin_VsfVsf(vscaled0x3, voutput_max_less_zero_point);
    vscaled1x0 = Q6_Vsf_vmin_VsfVsf(vscaled1x0, voutput_max_less_zero_point);
    vscaled1x1 = Q6_Vsf_vmin_VsfVsf(vscaled1x1, voutput_max_less_zero_point);
    vscaled1x2 = Q6_Vsf_vmin_VsfVsf(vscaled1x2, voutput_max_less_zero_point);
    vscaled1x3 = Q6_Vsf_vmin_VsfVsf(vscaled1x3, voutput_max_less_zero_point);
    vscaled2x0 = Q6_Vsf_vmin_VsfVsf(vscaled2x0, voutput_max_less_zero_point);
    vscaled2x1 = Q6_Vsf_vmin_VsfVsf(vscaled2x1, voutput_max_less_zero_point);
    vscaled2x2 = Q6_Vsf_vmin_VsfVsf(vscaled2x2, voutput_max_less_zero_point);
    vscaled2x3 = Q6_Vsf_vmin_VsfVsf(vscaled2x3, voutput_max_less_zero_point);

    HVX_Vector vscaled0x0_qf = Q6_Vqf32_vadd_VsfVsf(vscaled0x0, Q6_V_vzero());
    HVX_Vector vscaled0x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled0x1, Q6_V_vzero());
    HVX_Vector vscaled0x64_qf = Q6_Vqf32_vadd_VsfVsf(vscaled0x2, Q6_V_vzero());
    HVX_Vector vscaled0x96_qf = Q6_Vqf32_vadd_VsfVsf(vscaled0x3, Q6_V_vzero());
    HVX_Vector vscaled1x0_qf = Q6_Vqf32_vadd_VsfVsf(vscaled1x0, Q6_V_vzero());
    HVX_Vector vscaled1x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled1x1, Q6_V_vzero());
    HVX_Vector vscaled1x64_qf = Q6_Vqf32_vadd_VsfVsf(vscaled1x2, Q6_V_vzero());
    HVX_Vector vscaled1x96_qf = Q6_Vqf32_vadd_VsfVsf(vscaled1x3, Q6_V_vzero());
    HVX_Vector vscaled2x0_qf = Q6_Vqf32_vadd_VsfVsf(vscaled2x0, Q6_V_vzero());
    HVX_Vector vscaled2x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled2x1, Q6_V_vzero());
    HVX_Vector vscaled2x64_qf = Q6_Vqf32_vadd_VsfVsf(vscaled2x2, Q6_V_vzero());
    HVX_Vector vscaled2x96_qf = Q6_Vqf32_vadd_VsfVsf(vscaled2x3, Q6_V_vzero());

    vacc0x0 = Q6_Vw_convert_Vqf32(vscaled0x0_qf);
    vacc0x1 = Q6_Vw_convert_Vqf32(vscaled0x32_qf);
    vacc0x2 = Q6_Vw_convert_Vqf32(vscaled0x64_qf);
    vacc0x3 = Q6_Vw_convert_Vqf32(vscaled0x96_qf);
    vacc1x0 = Q6_Vw_convert_Vqf32(vscaled1x0_qf);
    vacc1x1 = Q6_Vw_convert_Vqf32(vscaled1x32_qf);
    vacc1x2 = Q6_Vw_convert_Vqf32(vscaled1x64_qf);
    vacc1x3 = Q6_Vw_convert_Vqf32(vscaled1x96_qf);
    vacc2x0 = Q6_Vw_convert_Vqf32(vscaled2x0_qf);
    vacc2x1 = Q6_Vw_convert_Vqf32(vscaled2x32_qf);
    vacc2x2 = Q6_Vw_convert_Vqf32(vscaled2x64_qf);
    vacc2x3 = Q6_Vw_convert_Vqf32(vscaled2x96_qf);

    vacc0x0 = Q6_Vw_vadd_VwVw(vacc0x0, voutput_zero_point);
    vacc0x1 = Q6_Vw_vadd_VwVw(vacc0x1, voutput_zero_point);
    vacc0x2 = Q6_Vw_vadd_VwVw(vacc0x2, voutput_zero_point);
    vacc0x3 = Q6_Vw_vadd_VwVw(vacc0x3, voutput_zero_point);
    vacc1x0 = Q6_Vw_vadd_VwVw(vacc1x0, voutput_zero_point);
    vacc1x1 = Q6_Vw_vadd_VwVw(vacc1x1, voutput_zero_point);
    vacc1x2 = Q6_Vw_vadd_VwVw(vacc1x2, voutput_zero_point);
    vacc1x3 = Q6_Vw_vadd_VwVw(vacc1x3, voutput_zero_point);
    vacc2x0 = Q6_Vw_vadd_VwVw(vacc2x0, voutput_zero_point);
    vacc2x1 = Q6_Vw_vadd_VwVw(vacc2x1, voutput_zero_point);
    vacc2x2 = Q6_Vw_vadd_VwVw(vacc2x2, voutput_zero_point);
    vacc2x3 = Q6_Vw_vadd_VwVw(vacc2x3, voutput_zero_point);

    HVX_Vector vout0x0 =  Q6_Vh_vpack_VwVw_sat(vacc0x0, vacc0x1);
    HVX_Vector vout0x1 =  Q6_Vh_vpack_VwVw_sat(vacc0x2, vacc0x3);
    HVX_Vector vout1x0 =  Q6_Vh_vpack_VwVw_sat(vacc1x0, vacc1x1);
    HVX_Vector vout1x1 =  Q6_Vh_vpack_VwVw_sat(vacc1x2, vacc1x3);
    HVX_Vector vout2x0 =  Q6_Vh_vpack_VwVw_sat(vacc2x0, vacc2x1);
    HVX_Vector vout2x1 =  Q6_Vh_vpack_VwVw_sat(vacc2x2, vacc2x3);

    HVX_Vector vout0 = Q6_Vb_vpack_VhVh_sat(vout0x0, vout0x1);
    HVX_Vector vout1 = Q6_Vb_vpack_VhVh_sat(vout1x0, vout1x1);
    HVX_Vector vout2 = Q6_Vb_vpack_VhVh_sat(vout2x0, vout2x1);

    vout0 = Q6_Vb_vmax_VbVb(vout0, voutput_min);
    vout1 = Q6_Vb_vmax_VbVb(vout1, voutput_min);
    vout2 = Q6_Vb_vmax_VbVb(vout2, voutput_min);

    if XNN_LIKELY(nc >= 128) {
      Q6_V_vstu_variable(c0, 128, vout0);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      Q6_V_vstu_variable(c1, 128, vout1);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      Q6_V_vstu_variable(c2, 128, vout2);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);

      nc -= 128;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      Q6_V_vstu_variable(c0, nc, vout0);
      Q6_V_vstu_variable(c1, nc, vout1);
      Q6_V_vstu_variable(c2, nc, vout2);
      nc = 0;
    }
  } while (nc != 0);
}
