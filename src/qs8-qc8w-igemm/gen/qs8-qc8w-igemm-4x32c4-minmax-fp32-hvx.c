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



void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x32c4__hvx(
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
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
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
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const float output_max_less_zero_point = (float)((int32_t) params->fp32_scalar.output_max - (int32_t)params->fp32_scalar.output_zero_point);
  const HVX_Vector voutput_max_less_zero_point = Q6_V_vsplat_R(*(uint32_t *)(&output_max_less_zero_point));
  const HVX_Vector voutput_zero_point = Q6_V_vsplat_R(params->fp32_scalar.output_zero_point);
  const HVX_Vector voutput_min = Q6_Vb_vsplat_R(params->fp32_scalar.output_min);

  do {
    HVX_Vector vacc0x32 = *((HVX_Vector*)w);
    HVX_Vector vacc1x0x32 = Q6_V_vsplat_R(0);
    HVX_Vector vacc1x32 = vacc0x32;
    HVX_Vector vacc1x1x32 = vacc1x0x32;
    HVX_Vector vacc2x32 = vacc0x32;
    HVX_Vector vacc1x2x32 = vacc1x0x32;
    HVX_Vector vacc3x32 = vacc0x32;
    HVX_Vector vacc1x3x32 = vacc1x0x32;

    w = (const int32_t*) w + 32;

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
      a += 4;

      size_t k = kc;
      for (; k >= 8 * sizeof(int8_t); k -= 8 * sizeof(int8_t)) {
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

        w = (const int8_t*) w + 256;
      }

      vacc0x32 = Q6_Vw_vadd_VwVw(vacc0x32, vacc1x0x32);
      vacc1x32 = Q6_Vw_vadd_VwVw(vacc1x32, vacc1x1x32);
      vacc2x32 = Q6_Vw_vadd_VwVw(vacc2x32, vacc1x2x32);
      vacc3x32 = Q6_Vw_vadd_VwVw(vacc3x32, vacc1x3x32);

      if (k != 0) {
        assert(k == 4 * sizeof(int8_t));
        const HVX_Vector va0x0123 = Q6_V_vsplat_R(unaligned_load_s32(a0));
        a0 += 4;
        const HVX_Vector va1x0123 = Q6_V_vsplat_R(unaligned_load_s32(a1));
        a1 += 4;
        const HVX_Vector va2x0123 = Q6_V_vsplat_R(unaligned_load_s32(a2));
        a2 += 4;
        const HVX_Vector va3x0123 = Q6_V_vsplat_R(unaligned_load_s32(a3));
        a3 += 4;

        const HVX_Vector vb32x0123 = *((HVX_Vector *)((int8_t *)w));
        vacc0x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc0x32, va0x0123, vb32x0123);
        vacc1x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc1x32, va1x0123, vb32x0123);
        vacc2x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc2x32, va2x0123, vb32x0123);
        vacc3x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc3x32, va3x0123, vb32x0123);

        w = (const int8_t*) w + 128;
        k -= 4 * sizeof(int8_t);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const HVX_Vector vscale32 = *((HVX_Vector *)w);
    w = (const float*) w + 32;
    HVX_Vector vscaled0x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc0x32));
    HVX_Vector vscaled1x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc1x32));
    HVX_Vector vscaled2x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc2x32));
    HVX_Vector vscaled3x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_convert_Vw(vacc3x32));

    vscaled0x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled0x32, vscale32));
    vscaled1x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled1x32, vscale32));
    vscaled2x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled2x32, vscale32));
    vscaled3x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled3x32, vscale32));

    vscaled0x32 = Q6_Vsf_vmin_VsfVsf(vscaled0x32, voutput_max_less_zero_point);
    vscaled1x32 = Q6_Vsf_vmin_VsfVsf(vscaled1x32, voutput_max_less_zero_point);
    vscaled2x32 = Q6_Vsf_vmin_VsfVsf(vscaled2x32, voutput_max_less_zero_point);
    vscaled3x32 = Q6_Vsf_vmin_VsfVsf(vscaled3x32, voutput_max_less_zero_point);
 
    HVX_Vector vscaled0x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled0x32, Q6_V_vzero()); 
    HVX_Vector vscaled1x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled1x32, Q6_V_vzero()); 
    HVX_Vector vscaled2x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled2x32, Q6_V_vzero()); 
    HVX_Vector vscaled3x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled3x32, Q6_V_vzero()); 
   
    vacc0x32 = Q6_Vw_convert_Vqf32(vscaled0x32_qf);
    vacc1x32 = Q6_Vw_convert_Vqf32(vscaled1x32_qf);
    vacc2x32 = Q6_Vw_convert_Vqf32(vscaled2x32_qf);
    vacc3x32 = Q6_Vw_convert_Vqf32(vscaled3x32_qf);

    vacc0x32 = Q6_Vw_vadd_VwVw(vacc0x32, voutput_zero_point);
    vacc1x32 = Q6_Vw_vadd_VwVw(vacc1x32, voutput_zero_point);
    vacc2x32 = Q6_Vw_vadd_VwVw(vacc2x32, voutput_zero_point);
    vacc3x32 = Q6_Vw_vadd_VwVw(vacc3x32, voutput_zero_point);

    HVX_Vector vout0x32 =  Q6_Vh_vpack_VwVw_sat(vacc0x32, vacc0x32);
    HVX_Vector vout1x32 =  Q6_Vh_vpack_VwVw_sat(vacc1x32, vacc1x32);
    HVX_Vector vout2x32 =  Q6_Vh_vpack_VwVw_sat(vacc2x32, vacc2x32);
    HVX_Vector vout3x32 =  Q6_Vh_vpack_VwVw_sat(vacc3x32, vacc3x32);

    vout0x32 = Q6_Vb_vpack_VhVh_sat(vout0x32, vout0x32);
    vout1x32 = Q6_Vb_vpack_VhVh_sat(vout1x32, vout1x32);
    vout2x32 = Q6_Vb_vpack_VhVh_sat(vout2x32, vout2x32);
    vout3x32 = Q6_Vb_vpack_VhVh_sat(vout3x32, vout3x32);

    vout0x32 = Q6_Vb_vmax_VbVb(vout0x32, voutput_min);
    vout1x32 = Q6_Vb_vmax_VbVb(vout1x32, voutput_min);
    vout2x32 = Q6_Vb_vmax_VbVb(vout2x32, voutput_min);
    vout3x32 = Q6_Vb_vmax_VbVb(vout3x32, voutput_min);

    if XNN_LIKELY(nc >= 32) {
      Q6_V_vstu_variable(c3, 32, vout3x32);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      Q6_V_vstu_variable(c2, 32, vout2x32);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      Q6_V_vstu_variable(c1, 32, vout1x32);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      Q6_V_vstu_variable(c0, 32, vout0x32);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 32;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      Q6_V_vstu_variable(c3, nc, vout3x32);
      Q6_V_vstu_variable(c2, nc, vout2x32);
      Q6_V_vstu_variable(c1, nc, vout1x32);
      Q6_V_vstu_variable(c0, nc, vout0x32);
      nc = 0;
    }
  } while (nc != 0);
}
