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



void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x32c4__hvx(
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
  assert(mr <= 8);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (8 * sizeof(void*)) == 0);
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
  int8_t* c5 = (int8_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    c5 = c4;
  }
  int8_t* c6 = (int8_t*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    c6 = c5;
  }
  int8_t* c7 = (int8_t*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 8) {
    c7 = c6;
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
    HVX_Vector vacc4x32 = vacc0x32;
    HVX_Vector vacc1x4x32 = vacc1x0x32;
    HVX_Vector vacc5x32 = vacc0x32;
    HVX_Vector vacc1x5x32 = vacc1x0x32;
    HVX_Vector vacc6x32 = vacc0x32;
    HVX_Vector vacc1x6x32 = vacc1x0x32;
    HVX_Vector vacc7x32 = vacc0x32;
    HVX_Vector vacc1x7x32 = vacc1x0x32;

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
      const int8_t* restrict a4 = a[4];
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const int8_t*) ((uintptr_t) a4 + a_offset);
      }
      const int8_t* restrict a5 = a[5];
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const int8_t*) ((uintptr_t) a5 + a_offset);
      }
      const int8_t* restrict a6 = a[6];
      if XNN_UNPREDICTABLE(a6 != zero) {
        a6 = (const int8_t*) ((uintptr_t) a6 + a_offset);
      }
      const int8_t* restrict a7 = a[7];
      if XNN_UNPREDICTABLE(a7 != zero) {
        a7 = (const int8_t*) ((uintptr_t) a7 + a_offset);
      }
      a += 8;

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

        w = (const int8_t*) w + 256;
      }

      vacc0x32 = Q6_Vw_vadd_VwVw(vacc0x32, vacc1x0x32);
      vacc1x32 = Q6_Vw_vadd_VwVw(vacc1x32, vacc1x1x32);
      vacc2x32 = Q6_Vw_vadd_VwVw(vacc2x32, vacc1x2x32);
      vacc3x32 = Q6_Vw_vadd_VwVw(vacc3x32, vacc1x3x32);
      vacc4x32 = Q6_Vw_vadd_VwVw(vacc4x32, vacc1x4x32);
      vacc5x32 = Q6_Vw_vadd_VwVw(vacc5x32, vacc1x5x32);
      vacc6x32 = Q6_Vw_vadd_VwVw(vacc6x32, vacc1x6x32);
      vacc7x32 = Q6_Vw_vadd_VwVw(vacc7x32, vacc1x7x32);

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
        const HVX_Vector va4x0123 = Q6_V_vsplat_R(unaligned_load_s32(a4));
        a4 += 4;
        const HVX_Vector va5x0123 = Q6_V_vsplat_R(unaligned_load_s32(a5));
        a5 += 4;
        const HVX_Vector va6x0123 = Q6_V_vsplat_R(unaligned_load_s32(a6));
        a6 += 4;
        const HVX_Vector va7x0123 = Q6_V_vsplat_R(unaligned_load_s32(a7));
        a7 += 4;

        const HVX_Vector vb32x0123 = *((HVX_Vector *)((int8_t *)w));
        vacc0x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc0x32, va0x0123, vb32x0123);
        vacc1x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc1x32, va1x0123, vb32x0123);
        vacc2x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc2x32, va2x0123, vb32x0123);
        vacc3x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc3x32, va3x0123, vb32x0123);
        vacc4x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc4x32, va4x0123, vb32x0123);
        vacc5x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc5x32, va5x0123, vb32x0123);
        vacc6x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc6x32, va6x0123, vb32x0123);
        vacc7x32 =  Q6_Vw_vrmpyacc_VwVbVb(vacc7x32, va7x0123, vb32x0123);

        w = (const int8_t*) w + 128;
        k -= 4 * sizeof(int8_t);
      }
      p -= 8 * sizeof(void*);
    } while (p != 0);

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

    vscaled0x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled0x32, vscale32));
    vscaled1x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled1x32, vscale32));
    vscaled2x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled2x32, vscale32));
    vscaled3x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled3x32, vscale32));
    vscaled4x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled4x32, vscale32));
    vscaled5x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled5x32, vscale32));
    vscaled6x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled6x32, vscale32));
    vscaled7x32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vscaled7x32, vscale32));

    vscaled0x32 = Q6_Vsf_vmin_VsfVsf(vscaled0x32, voutput_max_less_zero_point);
    vscaled1x32 = Q6_Vsf_vmin_VsfVsf(vscaled1x32, voutput_max_less_zero_point);
    vscaled2x32 = Q6_Vsf_vmin_VsfVsf(vscaled2x32, voutput_max_less_zero_point);
    vscaled3x32 = Q6_Vsf_vmin_VsfVsf(vscaled3x32, voutput_max_less_zero_point);
    vscaled4x32 = Q6_Vsf_vmin_VsfVsf(vscaled4x32, voutput_max_less_zero_point);
    vscaled5x32 = Q6_Vsf_vmin_VsfVsf(vscaled5x32, voutput_max_less_zero_point);
    vscaled6x32 = Q6_Vsf_vmin_VsfVsf(vscaled6x32, voutput_max_less_zero_point);
    vscaled7x32 = Q6_Vsf_vmin_VsfVsf(vscaled7x32, voutput_max_less_zero_point);
 
    HVX_Vector vscaled0x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled0x32, Q6_V_vzero()); 
    HVX_Vector vscaled1x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled1x32, Q6_V_vzero()); 
    HVX_Vector vscaled2x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled2x32, Q6_V_vzero()); 
    HVX_Vector vscaled3x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled3x32, Q6_V_vzero()); 
    HVX_Vector vscaled4x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled4x32, Q6_V_vzero()); 
    HVX_Vector vscaled5x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled5x32, Q6_V_vzero()); 
    HVX_Vector vscaled6x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled6x32, Q6_V_vzero()); 
    HVX_Vector vscaled7x32_qf = Q6_Vqf32_vadd_VsfVsf(vscaled7x32, Q6_V_vzero()); 
   
    vacc0x32 = Q6_Vw_convert_Vqf32(vscaled0x32_qf);
    vacc1x32 = Q6_Vw_convert_Vqf32(vscaled1x32_qf);
    vacc2x32 = Q6_Vw_convert_Vqf32(vscaled2x32_qf);
    vacc3x32 = Q6_Vw_convert_Vqf32(vscaled3x32_qf);
    vacc4x32 = Q6_Vw_convert_Vqf32(vscaled4x32_qf);
    vacc5x32 = Q6_Vw_convert_Vqf32(vscaled5x32_qf);
    vacc6x32 = Q6_Vw_convert_Vqf32(vscaled6x32_qf);
    vacc7x32 = Q6_Vw_convert_Vqf32(vscaled7x32_qf);

    vacc0x32 = Q6_Vw_vadd_VwVw(vacc0x32, voutput_zero_point);
    vacc1x32 = Q6_Vw_vadd_VwVw(vacc1x32, voutput_zero_point);
    vacc2x32 = Q6_Vw_vadd_VwVw(vacc2x32, voutput_zero_point);
    vacc3x32 = Q6_Vw_vadd_VwVw(vacc3x32, voutput_zero_point);
    vacc4x32 = Q6_Vw_vadd_VwVw(vacc4x32, voutput_zero_point);
    vacc5x32 = Q6_Vw_vadd_VwVw(vacc5x32, voutput_zero_point);
    vacc6x32 = Q6_Vw_vadd_VwVw(vacc6x32, voutput_zero_point);
    vacc7x32 = Q6_Vw_vadd_VwVw(vacc7x32, voutput_zero_point);

    HVX_Vector vout0x32 =  Q6_Vh_vpack_VwVw_sat(vacc0x32, vacc0x32);
    HVX_Vector vout1x32 =  Q6_Vh_vpack_VwVw_sat(vacc1x32, vacc1x32);
    HVX_Vector vout2x32 =  Q6_Vh_vpack_VwVw_sat(vacc2x32, vacc2x32);
    HVX_Vector vout3x32 =  Q6_Vh_vpack_VwVw_sat(vacc3x32, vacc3x32);
    HVX_Vector vout4x32 =  Q6_Vh_vpack_VwVw_sat(vacc4x32, vacc4x32);
    HVX_Vector vout5x32 =  Q6_Vh_vpack_VwVw_sat(vacc5x32, vacc5x32);
    HVX_Vector vout6x32 =  Q6_Vh_vpack_VwVw_sat(vacc6x32, vacc6x32);
    HVX_Vector vout7x32 =  Q6_Vh_vpack_VwVw_sat(vacc7x32, vacc7x32);

    vout0x32 = Q6_Vb_vpack_VhVh_sat(vout0x32, vout0x32);
    vout1x32 = Q6_Vb_vpack_VhVh_sat(vout1x32, vout1x32);
    vout2x32 = Q6_Vb_vpack_VhVh_sat(vout2x32, vout2x32);
    vout3x32 = Q6_Vb_vpack_VhVh_sat(vout3x32, vout3x32);
    vout4x32 = Q6_Vb_vpack_VhVh_sat(vout4x32, vout4x32);
    vout5x32 = Q6_Vb_vpack_VhVh_sat(vout5x32, vout5x32);
    vout6x32 = Q6_Vb_vpack_VhVh_sat(vout6x32, vout6x32);
    vout7x32 = Q6_Vb_vpack_VhVh_sat(vout7x32, vout7x32);

    vout0x32 = Q6_Vb_vmax_VbVb(vout0x32, voutput_min);
    vout1x32 = Q6_Vb_vmax_VbVb(vout1x32, voutput_min);
    vout2x32 = Q6_Vb_vmax_VbVb(vout2x32, voutput_min);
    vout3x32 = Q6_Vb_vmax_VbVb(vout3x32, voutput_min);
    vout4x32 = Q6_Vb_vmax_VbVb(vout4x32, voutput_min);
    vout5x32 = Q6_Vb_vmax_VbVb(vout5x32, voutput_min);
    vout6x32 = Q6_Vb_vmax_VbVb(vout6x32, voutput_min);
    vout7x32 = Q6_Vb_vmax_VbVb(vout7x32, voutput_min);

    if XNN_LIKELY(nc >= 32) {
      Q6_V_vstu_variable(c7, 32, vout7x32);
      c7 = (int8_t*) ((uintptr_t) c7 + cn_stride);
      Q6_V_vstu_variable(c6, 32, vout6x32);
      c6 = (int8_t*) ((uintptr_t) c6 + cn_stride);
      Q6_V_vstu_variable(c5, 32, vout5x32);
      c5 = (int8_t*) ((uintptr_t) c5 + cn_stride);
      Q6_V_vstu_variable(c4, 32, vout4x32);
      c4 = (int8_t*) ((uintptr_t) c4 + cn_stride);
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
      Q6_V_vstu_variable(c7, nc, vout7x32);
      Q6_V_vstu_variable(c6, nc, vout6x32);
      Q6_V_vstu_variable(c5, nc, vout5x32);
      Q6_V_vstu_variable(c4, nc, vout4x32);
      Q6_V_vstu_variable(c3, nc, vout3x32);
      Q6_V_vstu_variable(c2, nc, vout2x32);
      Q6_V_vstu_variable(c1, nc, vout1x32);
      Q6_V_vstu_variable(c0, nc, vout0x32);
      nc = 0;
    }
  } while (nc != 0);
}
