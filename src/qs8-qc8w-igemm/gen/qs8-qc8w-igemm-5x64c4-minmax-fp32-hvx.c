// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/c4-hvx.c.in
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



void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_5x64c4__hvx(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 5);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (5 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  int8_t* c4 = (int8_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }

  const float output_max_less_zero_point = (float)((int32_t) params->fp32_scalar.output_max - (int32_t)params->fp32_scalar.output_zero_point);
  const HVX_Vector voutput_max_less_zero_point = Q6_V_vsplat_R(*(uint32_t *)(&output_max_less_zero_point));
  const HVX_Vector voutput_zero_point = Q6_V_vsplat_R(params->fp32_scalar.output_zero_point);
  const HVX_Vector voutput_min = Q6_Vb_vsplat_R(params->fp32_scalar.output_min);

  do {
    HVX_Vector vacc0x0 = *((HVX_Vector*)w);
    HVX_Vector vacc0x1 = *((HVX_Vector*)w);
    HVX_Vector vacc1x0 = vacc0x0;
    HVX_Vector vacc2x0 = vacc0x0;
    HVX_Vector vacc3x0 = vacc0x0;
    HVX_Vector vacc4x0 = vacc0x0;
    HVX_Vector vacc1x1 = vacc0x1;
    HVX_Vector vacc2x1 = vacc0x1;
    HVX_Vector vacc3x1 = vacc0x1;
    HVX_Vector vacc4x1 = vacc0x1;

    w = (const int32_t*) w + 64;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      }
      const int8_t* restrict a4 = a[4];
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const int8_t*) ((uintptr_t) a4 + a_offset);
      }
      a += 5;

      size_t k = kc;
      for (; k >= 4 * sizeof(int8_t); k -= 4 * sizeof(int8_t)) {
        const HVX_Vector va0x0123 = Q6_V_vsplat_R(unaligned_load_s32(a0)); a0 += 4;
        const HVX_Vector va1x0123 = Q6_V_vsplat_R(unaligned_load_s32(a1)); a1 += 4;
        const HVX_Vector va2x0123 = Q6_V_vsplat_R(unaligned_load_s32(a2)); a2 += 4;
        const HVX_Vector va3x0123 = Q6_V_vsplat_R(unaligned_load_s32(a3)); a3 += 4;
        const HVX_Vector va4x0123 = Q6_V_vsplat_R(unaligned_load_s32(a4)); a4 += 4;

        const HVX_Vector vb0x0123 = *((HVX_Vector *)((int8_t *)w) + 0);
        const HVX_Vector vb1x0123 = *((HVX_Vector *)((int8_t *)w) + 128);
        vacc0x0 =  Q6_Vw_vrmpyacc_VwVbVb(vacc0x0, va0x0123, vb0x0123);
        vacc0x1 =  Q6_Vw_vrmpyacc_VwVbVb(vacc0x1, va0x0123, vb1x0123);
        vacc1x0 =  Q6_Vw_vrmpyacc_VwVbVb(vacc1x0, va1x0123, vb0x0123);
        vacc1x1 =  Q6_Vw_vrmpyacc_VwVbVb(vacc1x1, va1x0123, vb1x0123);
        vacc2x0 =  Q6_Vw_vrmpyacc_VwVbVb(vacc2x0, va2x0123, vb0x0123);
        vacc2x1 =  Q6_Vw_vrmpyacc_VwVbVb(vacc2x1, va2x0123, vb1x0123);
        vacc3x0 =  Q6_Vw_vrmpyacc_VwVbVb(vacc3x0, va3x0123, vb0x0123);
        vacc3x1 =  Q6_Vw_vrmpyacc_VwVbVb(vacc3x1, va3x0123, vb1x0123);
        vacc4x0 =  Q6_Vw_vrmpyacc_VwVbVb(vacc4x0, va4x0123, vb0x0123);
        vacc4x1 =  Q6_Vw_vrmpyacc_VwVbVb(vacc4x1, va4x0123, vb1x0123);

        w = (const int8_t*) w + 256 * sizeof(int8_t);
      }

      p -= 5 * sizeof(void*);
    } while (p != 0);

    const HVX_Vector vscale0 = *((const HVX_Vector *)((const float *)w) + 0);
    const HVX_Vector vscale1 = *((const HVX_Vector *)((const float *)w) + 32);
    w = (const float*) w + 64;

    HVX_Vector vscaled{M}x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc0x0));
    HVX_Vector vscaled{M}x1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc0x1));
    HVX_Vector vscaled{M}x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc1x0));
    HVX_Vector vscaled{M}x1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc1x1));
    HVX_Vector vscaled{M}x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc2x0));
    HVX_Vector vscaled{M}x1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc2x1));
    HVX_Vector vscaled{M}x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc3x0));
    HVX_Vector vscaled{M}x1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc3x1));
    HVX_Vector vscaled{M}x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc4x0));
    HVX_Vector vscaled{M}x1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc4x1));

    vscaled{M}x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled{M}x0, vscale0));
    vscaled{M}x1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled{M}x1, vscale1));
    vscaled{M}x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled{M}x0, vscale0));
    vscaled{M}x1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled{M}x1, vscale1));
    vscaled{M}x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled{M}x0, vscale0));
    vscaled{M}x1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled{M}x1, vscale1));
    vscaled{M}x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled{M}x0, vscale0));
    vscaled{M}x1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled{M}x1, vscale1));
    vscaled{M}x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled{M}x0, vscale0));
    vscaled{M}x1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled{M}x1, vscale1));

    vscaled{M}x0 = Q6_Vsf_vmin_VsfVsf(vscaled{M}x0, voutput_max_less_zero_point);
    vscaled{M}x1 = Q6_Vsf_vmin_VsfVsf(vscaled{M}x1, voutput_max_less_zero_point);
    vscaled{M}x0 = Q6_Vsf_vmin_VsfVsf(vscaled{M}x0, voutput_max_less_zero_point);
    vscaled{M}x1 = Q6_Vsf_vmin_VsfVsf(vscaled{M}x1, voutput_max_less_zero_point);
    vscaled{M}x0 = Q6_Vsf_vmin_VsfVsf(vscaled{M}x0, voutput_max_less_zero_point);
    vscaled{M}x1 = Q6_Vsf_vmin_VsfVsf(vscaled{M}x1, voutput_max_less_zero_point);
    vscaled{M}x0 = Q6_Vsf_vmin_VsfVsf(vscaled{M}x0, voutput_max_less_zero_point);
    vscaled{M}x1 = Q6_Vsf_vmin_VsfVsf(vscaled{M}x1, voutput_max_less_zero_point);
    vscaled{M}x0 = Q6_Vsf_vmin_VsfVsf(vscaled{M}x0, voutput_max_less_zero_point);
    vscaled{M}x1 = Q6_Vsf_vmin_VsfVsf(vscaled{M}x1, voutput_max_less_zero_point);

    HVX_Vector vscaled0x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled{M}x0, Q6_V_vzero());
    HVX_Vector vscaled0x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled{M}x1, Q6_V_vzero());
    HVX_Vector vscaled1x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled{M}x0, Q6_V_vzero());
    HVX_Vector vscaled1x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled{M}x1, Q6_V_vzero());
    HVX_Vector vscaled2x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled{M}x0, Q6_V_vzero());
    HVX_Vector vscaled2x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled{M}x1, Q6_V_vzero());
    HVX_Vector vscaled3x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled{M}x0, Q6_V_vzero());
    HVX_Vector vscaled3x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled{M}x1, Q6_V_vzero());
    HVX_Vector vscaled4x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled{M}x0, Q6_V_vzero());
    HVX_Vector vscaled4x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled{M}x1, Q6_V_vzero());

    vacc0x0 = Q6_Vw_convert_Vqf32(vscaled0x32_qf);
    vacc0x1 = Q6_Vw_convert_Vqf32(vscaled0x32_qf);
    vacc1x0 = Q6_Vw_convert_Vqf32(vscaled1x32_qf);
    vacc1x1 = Q6_Vw_convert_Vqf32(vscaled1x32_qf);
    vacc2x0 = Q6_Vw_convert_Vqf32(vscaled2x32_qf);
    vacc2x1 = Q6_Vw_convert_Vqf32(vscaled2x32_qf);
    vacc3x0 = Q6_Vw_convert_Vqf32(vscaled3x32_qf);
    vacc3x1 = Q6_Vw_convert_Vqf32(vscaled3x32_qf);
    vacc4x0 = Q6_Vw_convert_Vqf32(vscaled4x32_qf);
    vacc4x1 = Q6_Vw_convert_Vqf32(vscaled4x32_qf);

    vacc0x0 = Q6_Vw_vadd_VwVw(vacc0x0, voutput_zero_point);
    vacc0x1 = Q6_Vw_vadd_VwVw(vacc0x1, voutput_zero_point);
    vacc1x0 = Q6_Vw_vadd_VwVw(vacc1x0, voutput_zero_point);
    vacc1x1 = Q6_Vw_vadd_VwVw(vacc1x1, voutput_zero_point);
    vacc2x0 = Q6_Vw_vadd_VwVw(vacc2x0, voutput_zero_point);
    vacc2x1 = Q6_Vw_vadd_VwVw(vacc2x1, voutput_zero_point);
    vacc3x0 = Q6_Vw_vadd_VwVw(vacc3x0, voutput_zero_point);
    vacc3x1 = Q6_Vw_vadd_VwVw(vacc3x1, voutput_zero_point);
    vacc4x0 = Q6_Vw_vadd_VwVw(vacc4x0, voutput_zero_point);
    vacc4x1 = Q6_Vw_vadd_VwVw(vacc4x1, voutput_zero_point);

    HVX_Vector vout0x0 =  Q6_Vh_vpack_VwVw_sat(vacc0x0, vacc0x1);
    HVX_Vector vout1x0 =  Q6_Vh_vpack_VwVw_sat(vacc1x0, vacc1x1);
    HVX_Vector vout2x0 =  Q6_Vh_vpack_VwVw_sat(vacc2x0, vacc2x1);
    HVX_Vector vout3x0 =  Q6_Vh_vpack_VwVw_sat(vacc3x0, vacc3x1);
    HVX_Vector vout4x0 =  Q6_Vh_vpack_VwVw_sat(vacc4x0, vacc4x1);

    HVX_Vector vout0 = Q6_Vb_vpack_VhVh_sat(vout0x0, vout0x1);
    HVX_Vector vout1 = Q6_Vb_vpack_VhVh_sat(vout1x0, vout1x1);
    HVX_Vector vout2 = Q6_Vb_vpack_VhVh_sat(vout2x0, vout2x1);
    HVX_Vector vout3 = Q6_Vb_vpack_VhVh_sat(vout3x0, vout3x1);
    HVX_Vector vout4 = Q6_Vb_vpack_VhVh_sat(vout4x0, vout4x1);

    vout0 = Q6_Vb_vmax_VbVb(vout0, voutput_min);
    vout1 = Q6_Vb_vmax_VbVb(vout1, voutput_min);
    vout2 = Q6_Vb_vmax_VbVb(vout2, voutput_min);
    vout3 = Q6_Vb_vmax_VbVb(vout3, voutput_min);
    vout4 = Q6_Vb_vmax_VbVb(vout4, voutput_min);

    if XNN_LIKELY(nc >= 64) {
      Q6_V_vstu_variable(c4, 64, vout4);
      c4 = (int8_t*) ((uintptr_t) c4 + cn_stride);
      Q6_V_vstu_variable(c3, 64, vout3);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      Q6_V_vstu_variable(c2, 64, vout2);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      Q6_V_vstu_variable(c1, 64, vout1);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      Q6_V_vstu_variable(c0, 64, vout0);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 64;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      Q6_V_vstu_variable(c4, nc, vout4);
      Q6_V_vstu_variable(c3, nc, vout3);
      Q6_V_vstu_variable(c2, nc, vout2);
      Q6_V_vstu_variable(c1, nc, vout1);
      Q6_V_vstu_variable(c0, nc, vout0);
      nc = 0;
    }
  } while (nc != 0);
}
