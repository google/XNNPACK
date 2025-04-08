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


void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x32c4__hvx(
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
    HVX_Vector vacc1x0 = vacc0x0;
    HVX_Vector vacc2x0 = vacc0x0;

    w = (const int32_t*) w + 32;

    size_t k = kc;
    for (; k >= 4 * sizeof(int8_t); k -= 4 * sizeof(int8_t)) {
      const HVX_Vector va0x0123 = Q6_V_vsplat_R(unaligned_load_s32(a0)); a0 += 4;
      const HVX_Vector va1x0123 = Q6_V_vsplat_R(unaligned_load_s32(a1)); a1 += 4;
      const HVX_Vector va2x0123 = Q6_V_vsplat_R(unaligned_load_s32(a2)); a2 += 4;

      const HVX_Vector vb0x0123 = *((HVX_Vector *)((int8_t *)w) + 0);
      vacc0x0 =  Q6_Vw_vrmpyacc_VwVbVb(vacc0x0, va0x0123, vb0x0123);
      vacc1x0 =  Q6_Vw_vrmpyacc_VwVbVb(vacc1x0, va1x0123, vb0x0123);
      vacc2x0 =  Q6_Vw_vrmpyacc_VwVbVb(vacc2x0, va2x0123, vb0x0123);

      w = (const int8_t*) w + 128 * sizeof(int8_t);
    }

    const HVX_Vector vscale0 = *((const HVX_Vector *)((const float *)w) + 0);
    w = (const float*) w + 32;

    HVX_Vector vscaled0x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc0x0));
    HVX_Vector vscaled1x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc1x0));
    HVX_Vector vscaled2x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc2x0));

    vscaled0x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled0x0, vscale0));
    vscaled1x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled1x0, vscale0));
    vscaled2x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled2x0, vscale0));

    vscaled0x0 = Q6_Vsf_vmin_VsfVsf(vscaled0x0, voutput_max_less_zero_point);
    vscaled1x0 = Q6_Vsf_vmin_VsfVsf(vscaled1x0, voutput_max_less_zero_point);
    vscaled2x0 = Q6_Vsf_vmin_VsfVsf(vscaled2x0, voutput_max_less_zero_point);

    HVX_Vector vscaled0x0_qf = Q6_Vqf32_vadd_VsfVsf(vscaled0x0, Q6_V_vzero());
    HVX_Vector vscaled1x0_qf = Q6_Vqf32_vadd_VsfVsf(vscaled1x0, Q6_V_vzero());
    HVX_Vector vscaled2x0_qf = Q6_Vqf32_vadd_VsfVsf(vscaled2x0, Q6_V_vzero());

    vacc0x0 = Q6_Vw_convert_Vqf32(vscaled0x0_qf);
    vacc1x0 = Q6_Vw_convert_Vqf32(vscaled1x0_qf);
    vacc2x0 = Q6_Vw_convert_Vqf32(vscaled2x0_qf);

    vacc0x0 = Q6_Vw_vadd_VwVw(vacc0x0, voutput_zero_point);
    vacc1x0 = Q6_Vw_vadd_VwVw(vacc1x0, voutput_zero_point);
    vacc2x0 = Q6_Vw_vadd_VwVw(vacc2x0, voutput_zero_point);

    HVX_Vector vout0x0 =  Q6_Vh_vpack_VwVw_sat(vacc0x0, vacc0x0);
    HVX_Vector vout1x0 =  Q6_Vh_vpack_VwVw_sat(vacc1x0, vacc1x0);
    HVX_Vector vout2x0 =  Q6_Vh_vpack_VwVw_sat(vacc2x0, vacc2x0);

    HVX_Vector vout0 = Q6_Vb_vpack_VhVh_sat(vout0x0, vout0x0);
    HVX_Vector vout1 = Q6_Vb_vpack_VhVh_sat(vout1x0, vout1x0);
    HVX_Vector vout2 = Q6_Vb_vpack_VhVh_sat(vout2x0, vout2x0);

    vout0 = Q6_Vb_vmax_VbVb(vout0, voutput_min);
    vout1 = Q6_Vb_vmax_VbVb(vout1, voutput_min);
    vout2 = Q6_Vb_vmax_VbVb(vout2, voutput_min);

    if XNN_LIKELY(nc >= 32) {
      Q6_V_vstu_variable(c0, 32, vout0);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      Q6_V_vstu_variable(c1, 32, vout1);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      Q6_V_vstu_variable(c2, 32, vout2);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);

      nc -= 32;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      Q6_V_vstu_variable(c0, nc, vout0);
      Q6_V_vstu_variable(c1, nc, vout1);
      Q6_V_vstu_variable(c2, nc, vout2);
      nc = 0;
    }
  } while (nc != 0);
}
