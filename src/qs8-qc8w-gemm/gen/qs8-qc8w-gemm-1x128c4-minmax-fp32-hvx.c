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


void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x128c4__hvx(
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
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;

  const float output_max_less_zero_point = (float)((int32_t) params->fp32_scalar.output_max - (int32_t)params->fp32_scalar.output_zero_point);
  const HVX_Vector voutput_max_less_zero_point = Q6_V_vsplat_R(*(uint32_t *)(&output_max_less_zero_point));
  const HVX_Vector voutput_zero_point = Q6_V_vsplat_R(params->fp32_scalar.output_zero_point);
  const HVX_Vector voutput_min = Q6_Vb_vsplat_R(params->fp32_scalar.output_min);

  do {
    HVX_Vector vacc0x0 = *((HVX_Vector*)w);
    HVX_Vector vacc0x1 = *((HVX_Vector*)w);
    HVX_Vector vacc0x2 = *((HVX_Vector*)w);
    HVX_Vector vacc0x3 = *((HVX_Vector*)w);

    w = (const int32_t*) w + 128;

    size_t k = kc;
    for (; k >= 4 * sizeof(int8_t); k -= 4 * sizeof(int8_t)) {
      const HVX_Vector va0x0123 = Q6_V_vsplat_R(unaligned_load_s32(a0)); a0 += 4;

      const HVX_Vector vb0x0123 = *((HVX_Vector *)((int8_t *)w) + 0);
      const HVX_Vector vb1x0123 = *((HVX_Vector *)((int8_t *)w) + 128);
      const HVX_Vector vb2x0123 = *((HVX_Vector *)((int8_t *)w) + 256);
      const HVX_Vector vb3x0123 = *((HVX_Vector *)((int8_t *)w) + 384);
      vacc0x0 =  Q6_Vw_vrmpyacc_VwVbVb(vacc0x0, va0x0123, vb0x0123);
      vacc0x1 =  Q6_Vw_vrmpyacc_VwVbVb(vacc0x1, va0x0123, vb1x0123);
      vacc0x2 =  Q6_Vw_vrmpyacc_VwVbVb(vacc0x2, va0x0123, vb2x0123);
      vacc0x3 =  Q6_Vw_vrmpyacc_VwVbVb(vacc0x3, va0x0123, vb3x0123);

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

    vscaled0x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled0x0, vscale0));
    vscaled0x1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled0x1, vscale1));
    vscaled0x2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled0x2, vscale2));
    vscaled0x3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled0x3, vscale3));

    vscaled0x0 = Q6_Vsf_vmin_VsfVsf(vscaled0x0, voutput_max_less_zero_point);
    vscaled0x1 = Q6_Vsf_vmin_VsfVsf(vscaled0x1, voutput_max_less_zero_point);
    vscaled0x2 = Q6_Vsf_vmin_VsfVsf(vscaled0x2, voutput_max_less_zero_point);
    vscaled0x3 = Q6_Vsf_vmin_VsfVsf(vscaled0x3, voutput_max_less_zero_point);

    HVX_Vector vscaled0x0_qf = Q6_Vqf32_vadd_VsfVsf(vscaled0x0, Q6_V_vzero());
    HVX_Vector vscaled0x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled0x1, Q6_V_vzero());
    HVX_Vector vscaled0x64_qf = Q6_Vqf32_vadd_VsfVsf(vscaled0x2, Q6_V_vzero());
    HVX_Vector vscaled0x96_qf = Q6_Vqf32_vadd_VsfVsf(vscaled0x3, Q6_V_vzero());

    vacc0x0 = Q6_Vw_convert_Vqf32(vscaled0x0_qf);
    vacc0x1 = Q6_Vw_convert_Vqf32(vscaled0x32_qf);
    vacc0x2 = Q6_Vw_convert_Vqf32(vscaled0x64_qf);
    vacc0x3 = Q6_Vw_convert_Vqf32(vscaled0x96_qf);

    vacc0x0 = Q6_Vw_vadd_VwVw(vacc0x0, voutput_zero_point);
    vacc0x1 = Q6_Vw_vadd_VwVw(vacc0x1, voutput_zero_point);
    vacc0x2 = Q6_Vw_vadd_VwVw(vacc0x2, voutput_zero_point);
    vacc0x3 = Q6_Vw_vadd_VwVw(vacc0x3, voutput_zero_point);

    HVX_Vector vout0x0 =  Q6_Vh_vpack_VwVw_sat(vacc0x0, vacc0x1);
    HVX_Vector vout0x1 =  Q6_Vh_vpack_VwVw_sat(vacc0x2, vacc0x3);

    HVX_Vector vout0 = Q6_Vb_vpack_VhVh_sat(vout0x0, vout0x1);

    vout0 = Q6_Vb_vmax_VbVb(vout0, voutput_min);

    if XNN_LIKELY(nc >= 128) {
      Q6_V_vstu_variable(c0, 128, vout0);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      nc -= 128;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      Q6_V_vstu_variable(c0, nc, vout0);
      nc = 0;
    }
  } while (nc != 0);
}
