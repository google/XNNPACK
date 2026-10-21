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
#include <stdio.h>  // for printf

#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"  // for Q6_V_vstu_variable
#include "src/xnnpack/math.h"
#include "src/xnnpack/simd/f32-hvx.h"
#include "src/xnnpack/simd/s32-hvx.h"
#include "src/xnnpack/unaligned.h"



void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x128c4__hvx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  // TODO: Use log when fixed
  {
    static int warning_unaligned = 0;
    if ((a_stride & (sizeof(int32_t) - 1)) != 0 && warning_unaligned == 0) {
      printf("HEXAGON GEMM a_stride unaligned.");
      warning_unaligned = 1;
    }
    static int warning_a_unaligned = 0;
    if ((((intptr_t) a) & (sizeof(int32_t) - 1)) != 0 && warning_a_unaligned == 0) {
      printf("HEXAGON GEMM a unaligned.");
      warning_a_unaligned = 1;
    }
    fflush(stdout);
  }

  const HVX_Vector vmin = Q6_V_vsplat_R(float_as_uint32(params->scalar.min));
  const HVX_Vector vmax = Q6_V_vsplat_R(float_as_uint32(params->scalar.max));
  do {
    const HVX_Vector vksum0 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
    const HVX_Vector vksum1 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
    const HVX_Vector vksum2 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
    const HVX_Vector vksum3 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
    const HVX_Vector vinput_zero_point0 = Q6_V_vsplat_R(quantization_params[0].zero_point);
    HVX_Vector vacc0x0 = xnn_mul_s32(vksum0, vinput_zero_point0);
    HVX_Vector vacc0x1 = xnn_mul_s32(vksum1, vinput_zero_point0);
    HVX_Vector vacc0x2 = xnn_mul_s32(vksum2, vinput_zero_point0);
    HVX_Vector vacc0x3 = xnn_mul_s32(vksum3, vinput_zero_point0);
    const HVX_Vector vinput_zero_point1 = Q6_V_vsplat_R(quantization_params[1].zero_point);
    HVX_Vector vacc1x0 = xnn_mul_s32(vksum0, vinput_zero_point1);
    HVX_Vector vacc1x1 = xnn_mul_s32(vksum1, vinput_zero_point1);
    HVX_Vector vacc1x2 = xnn_mul_s32(vksum2, vinput_zero_point1);
    HVX_Vector vacc1x3 = xnn_mul_s32(vksum3, vinput_zero_point1);

    size_t k = kc;
    if (((((intptr_t) a) | a_stride) & (sizeof(int32_t) - 1)) != 0) {
      for (; k >= 8 * sizeof(int8_t); k -= 8 * sizeof(int8_t)) {
        const HVX_Vector va0x0123 = Q6_V_vsplat_R(unaligned_load_s32(a0));
        const HVX_Vector va0x4567 = Q6_V_vsplat_R(unaligned_load_s32(a0+4)); a0 += 8;
        const HVX_Vector va1x0123 = Q6_V_vsplat_R(unaligned_load_s32(a1));
        const HVX_Vector va1x4567 = Q6_V_vsplat_R(unaligned_load_s32(a1+4)); a1 += 8;

        const HVX_Vector vb0x0123 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
        const HVX_Vector vb1x0123 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
        const HVX_Vector vb2x0123 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
        const HVX_Vector vb3x0123 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
        const HVX_Vector vb0x4567 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
        const HVX_Vector vb1x4567 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
        const HVX_Vector vb2x4567 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
        const HVX_Vector vb3x4567 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;

        vacc0x0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x0, va0x0123, vb0x0123);
        vacc0x1 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x1, va0x0123, vb1x0123);
        vacc0x2 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x2, va0x0123, vb2x0123);
        vacc0x3 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x3, va0x0123, vb3x0123);
        vacc1x0 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x0, va1x0123, vb0x0123);
        vacc1x1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x1, va1x0123, vb1x0123);
        vacc1x2 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x2, va1x0123, vb2x0123);
        vacc1x3 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x3, va1x0123, vb3x0123);
        vacc0x0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x0, va0x4567, vb0x4567);
        vacc0x1 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x1, va0x4567, vb1x4567);
        vacc0x2 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x2, va0x4567, vb2x4567);
        vacc0x3 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x3, va0x4567, vb3x4567);
        vacc1x0 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x0, va1x4567, vb0x4567);
        vacc1x1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x1, va1x4567, vb1x4567);
        vacc1x2 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x2, va1x4567, vb2x4567);
        vacc1x3 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x3, va1x4567, vb3x4567);
      }
    } else {
      for (; k >= 8 * sizeof(int8_t); k -= 8 * sizeof(int8_t)) {
        const HVX_Vector va0x0123 = Q6_V_vsplat_R(*((const int32_t*)a0));
        const HVX_Vector va0x4567 = Q6_V_vsplat_R(*((const int32_t*)(a0+4))); a0 += 8;
        const HVX_Vector va1x0123 = Q6_V_vsplat_R(*((const int32_t*)a1));
        const HVX_Vector va1x4567 = Q6_V_vsplat_R(*((const int32_t*)(a1+4))); a1 += 8;

        const HVX_Vector vb0x0123 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
        const HVX_Vector vb1x0123 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
        const HVX_Vector vb2x0123 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
        const HVX_Vector vb3x0123 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
        const HVX_Vector vb0x4567 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
        const HVX_Vector vb1x4567 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
        const HVX_Vector vb2x4567 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
        const HVX_Vector vb3x4567 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;

        vacc0x0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x0, va0x0123, vb0x0123);
        vacc0x1 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x1, va0x0123, vb1x0123);
        vacc0x2 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x2, va0x0123, vb2x0123);
        vacc0x3 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x3, va0x0123, vb3x0123);
        vacc1x0 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x0, va1x0123, vb0x0123);
        vacc1x1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x1, va1x0123, vb1x0123);
        vacc1x2 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x2, va1x0123, vb2x0123);
        vacc1x3 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x3, va1x0123, vb3x0123);
        vacc0x0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x0, va0x4567, vb0x4567);
        vacc0x1 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x1, va0x4567, vb1x4567);
        vacc0x2 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x2, va0x4567, vb2x4567);
        vacc0x3 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x3, va0x4567, vb3x4567);
        vacc1x0 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x0, va1x4567, vb0x4567);
        vacc1x1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x1, va1x4567, vb1x4567);
        vacc1x2 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x2, va1x4567, vb2x4567);
        vacc1x3 = Q6_Vw_vrmpyacc_VwVbVb(vacc1x3, va1x4567, vb3x4567);
      }
    }
    if (k != 0) {
      const HVX_Vector va0x0123 = Q6_V_vsplat_R(unaligned_load_s32(a0)); a0 += 4;
      const HVX_Vector va1x0123 = Q6_V_vsplat_R(unaligned_load_s32(a1)); a1 += 4;

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
    }


    const HVX_Vector vinput_scale0 = Q6_V_vsplat_R(float_as_uint32(quantization_params[0].inv_scale));
    HVX_Vector vout0x0 = xnn_mul_f32(Q6_Vsf_equals_Vw(vacc0x0), vinput_scale0);
    HVX_Vector vout0x1 = xnn_mul_f32(Q6_Vsf_equals_Vw(vacc0x1), vinput_scale0);
    HVX_Vector vout0x2 = xnn_mul_f32(Q6_Vsf_equals_Vw(vacc0x2), vinput_scale0);
    HVX_Vector vout0x3 = xnn_mul_f32(Q6_Vsf_equals_Vw(vacc0x3), vinput_scale0);
    const HVX_Vector vinput_scale1 = Q6_V_vsplat_R(float_as_uint32(quantization_params[1].inv_scale));
    HVX_Vector vout1x0 = xnn_mul_f32(Q6_Vsf_equals_Vw(vacc1x0), vinput_scale1);
    HVX_Vector vout1x1 = xnn_mul_f32(Q6_Vsf_equals_Vw(vacc1x1), vinput_scale1);
    HVX_Vector vout1x2 = xnn_mul_f32(Q6_Vsf_equals_Vw(vacc1x2), vinput_scale1);
    HVX_Vector vout1x3 = xnn_mul_f32(Q6_Vsf_equals_Vw(vacc1x3), vinput_scale1);

    const HVX_Vector vfilter_output_scale0 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
    vout0x0 = xnn_mul_f32(vout0x0, vfilter_output_scale0);
    vout1x0 = xnn_mul_f32(vout1x0, vfilter_output_scale0);
    const HVX_Vector vfilter_output_scale1 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
    vout0x1 = xnn_mul_f32(vout0x1, vfilter_output_scale1);
    vout1x1 = xnn_mul_f32(vout1x1, vfilter_output_scale1);
    const HVX_Vector vfilter_output_scale2 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
    vout0x2 = xnn_mul_f32(vout0x2, vfilter_output_scale2);
    vout1x2 = xnn_mul_f32(vout1x2, vfilter_output_scale2);
    const HVX_Vector vfilter_output_scale3 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
    vout0x3 = xnn_mul_f32(vout0x3, vfilter_output_scale3);
    vout1x3 = xnn_mul_f32(vout1x3, vfilter_output_scale3);

    const HVX_Vector vbias0 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
    vout0x0 = xnn_add_f32(vout0x0, vbias0);
    vout0x0 = xnn_max_f32(vout0x0, vmin);
    vout0x0 = xnn_min_f32(vout0x0, vmax);
    vout1x0 = xnn_add_f32(vout1x0, vbias0);
    vout1x0 = xnn_max_f32(vout1x0, vmin);
    vout1x0 = xnn_min_f32(vout1x0, vmax);
    const HVX_Vector vbias1 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
    vout0x1 = xnn_add_f32(vout0x1, vbias1);
    vout0x1 = xnn_max_f32(vout0x1, vmin);
    vout0x1 = xnn_min_f32(vout0x1, vmax);
    vout1x1 = xnn_add_f32(vout1x1, vbias1);
    vout1x1 = xnn_max_f32(vout1x1, vmin);
    vout1x1 = xnn_min_f32(vout1x1, vmax);
    const HVX_Vector vbias2 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
    vout0x2 = xnn_add_f32(vout0x2, vbias2);
    vout0x2 = xnn_max_f32(vout0x2, vmin);
    vout0x2 = xnn_min_f32(vout0x2, vmax);
    vout1x2 = xnn_add_f32(vout1x2, vbias2);
    vout1x2 = xnn_max_f32(vout1x2, vmin);
    vout1x2 = xnn_min_f32(vout1x2, vmax);
    const HVX_Vector vbias3 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
    vout0x3 = xnn_add_f32(vout0x3, vbias3);
    vout0x3 = xnn_max_f32(vout0x3, vmin);
    vout0x3 = xnn_min_f32(vout0x3, vmax);
    vout1x3 = xnn_add_f32(vout1x3, vbias3);
    vout1x3 = xnn_max_f32(vout1x3, vmin);
    vout1x3 = xnn_min_f32(vout1x3, vmax);

    if XNN_LIKELY(nc >= 128) {
      *((HVX_UVector*)(c0 + 0)) = vout0x0;
      *((HVX_UVector*)(c0 + 32)) = vout0x1;
      *((HVX_UVector*)(c0 + 64)) = vout0x2;
      *((HVX_UVector*)(c0 + 96)) = vout0x3;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      *((HVX_UVector*)(c1 + 0)) = vout1x0;
      *((HVX_UVector*)(c1 + 32)) = vout1x1;
      *((HVX_UVector*)(c1 + 64)) = vout1x2;
      *((HVX_UVector*)(c1 + 96)) = vout1x3;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      nc -= 128;
    } else {
      const size_t nc_tail = min(nc, 32);
      Q6_V_vstu_variable(c0, nc_tail * sizeof(float), vout0x0);
      Q6_V_vstu_variable(c1, nc_tail * sizeof(float), vout1x0);
      if (nc > 32) {
        const size_t nc_tail = min(nc - 32, 32);
        Q6_V_vstu_variable(c0 + 32, nc_tail * sizeof(float), vout0x1);
        Q6_V_vstu_variable(c1 + 32, nc_tail * sizeof(float), vout1x1);
      }
      if (nc > 64) {
        const size_t nc_tail = min(nc - 64, 32);
        Q6_V_vstu_variable(c0 + 64, nc_tail * sizeof(float), vout0x2);
        Q6_V_vstu_variable(c1 + 64, nc_tail * sizeof(float), vout1x2);
      }
      if (nc > 96) {
        const size_t nc_tail = min(nc - 96, 32);
        Q6_V_vstu_variable(c0 + 96, nc_tail * sizeof(float), vout0x3);
        Q6_V_vstu_variable(c1 + 96, nc_tail * sizeof(float), vout1x3);
      }
      nc = 0;
    }
  } while (nc != 0);
}
