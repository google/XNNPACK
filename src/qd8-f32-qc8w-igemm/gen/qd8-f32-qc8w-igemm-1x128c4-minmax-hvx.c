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

#include "src/xnnpack/igemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"  // for Q6_V_vstu_variable
#include "src/xnnpack/math.h"
#include "src/xnnpack/simd/f32-hvx.h"
#include "src/xnnpack/simd/s32-hvx.h"
#include "src/xnnpack/unaligned.h"



void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x128c4__hvx(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const struct xnn_f32_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a != NULL);
  assert(((uintptr_t) (a) % sizeof(int32_t)) == 0);
  assert(w != NULL);
  assert(c != NULL);
  assert(a_offset % sizeof(int32_t) == 0);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  float* c0 = c;

  const HVX_Vector vmin = Q6_V_vsplat_R(float_as_uint32(params->scalar.min));
  const HVX_Vector vmax = Q6_V_vsplat_R(float_as_uint32(params->scalar.max));

  do {
    const HVX_Vector vksum0 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
    const HVX_Vector vksum1 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
    const HVX_Vector vksum2 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
    const HVX_Vector vksum3 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
    const HVX_Vector vinput_zero_point = Q6_V_vsplat_R(quantization_params->zero_point);
    HVX_Vector vacc0x0 = xnn_mul_s32(vksum0, vinput_zero_point);
    HVX_Vector vacc0x1 = xnn_mul_s32(vksum1, vinput_zero_point);
    HVX_Vector vacc0x2 = xnn_mul_s32(vksum2, vinput_zero_point);
    HVX_Vector vacc0x3 = xnn_mul_s32(vksum3, vinput_zero_point);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      a += 1;

      size_t k = kc;
      for (; k >= 4 * sizeof(int8_t); k -= 4 * sizeof(int8_t)) {
        const HVX_Vector va0x0123 = Q6_V_vsplat_R(unaligned_load_s32(a0)); a0 += 4;

        const HVX_Vector vb0x0123 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
        const HVX_Vector vb1x0123 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
        const HVX_Vector vb2x0123 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;
        const HVX_Vector vb3x0123 = *((HVX_Vector *) w); w = (const int8_t*) w + 128;

        vacc0x0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x0, va0x0123, vb0x0123);
        vacc0x1 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x1, va0x0123, vb1x0123);
        vacc0x2 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x2, va0x0123, vb2x0123);
        vacc0x3 = Q6_Vw_vrmpyacc_VwVbVb(vacc0x3, va0x0123, vb3x0123);
      }

      p -= 1 * sizeof(void*);
    } while (p != 0);

    const HVX_Vector vinput_scale = Q6_V_vsplat_R(float_as_uint32(quantization_params->inv_scale));
    HVX_Vector vout0x0 = xnn_mul_f32(Q6_Vsf_equals_Vw(vacc0x0), vinput_scale);
    HVX_Vector vout0x1 = xnn_mul_f32(Q6_Vsf_equals_Vw(vacc0x1), vinput_scale);
    HVX_Vector vout0x2 = xnn_mul_f32(Q6_Vsf_equals_Vw(vacc0x2), vinput_scale);
    HVX_Vector vout0x3 = xnn_mul_f32(Q6_Vsf_equals_Vw(vacc0x3), vinput_scale);

    const HVX_Vector vfilter_output_scale0 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
    vout0x0 = xnn_mul_f32(vout0x0, vfilter_output_scale0);
    const HVX_Vector vfilter_output_scale1 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
    vout0x1 = xnn_mul_f32(vout0x1, vfilter_output_scale1);
    const HVX_Vector vfilter_output_scale2 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
    vout0x2 = xnn_mul_f32(vout0x2, vfilter_output_scale2);
    const HVX_Vector vfilter_output_scale3 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
    vout0x3 = xnn_mul_f32(vout0x3, vfilter_output_scale3);

    const HVX_Vector vbias0 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
    vout0x0 = xnn_add_f32(vout0x0, vbias0);
    vout0x0 = xnn_max_f32(vout0x0, vmin);
    vout0x0 = xnn_min_f32(vout0x0, vmax);
    const HVX_Vector vbias1 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
    vout0x1 = xnn_add_f32(vout0x1, vbias1);
    vout0x1 = xnn_max_f32(vout0x1, vmin);
    vout0x1 = xnn_min_f32(vout0x1, vmax);
    const HVX_Vector vbias2 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
    vout0x2 = xnn_add_f32(vout0x2, vbias2);
    vout0x2 = xnn_max_f32(vout0x2, vmin);
    vout0x2 = xnn_min_f32(vout0x2, vmax);
    const HVX_Vector vbias3 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
    vout0x3 = xnn_add_f32(vout0x3, vbias3);
    vout0x3 = xnn_max_f32(vout0x3, vmin);
    vout0x3 = xnn_min_f32(vout0x3, vmax);

    if XNN_LIKELY(nc >= 128) {
      *((HVX_UVector*)(c0 + 0)) = vout0x0;
      *((HVX_UVector*)(c0 + 32)) = vout0x1;
      *((HVX_UVector*)(c0 + 64)) = vout0x2;
      *((HVX_UVector*)(c0 + 96)) = vout0x3;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 128;
    } else {
      const size_t nc_tail = min(nc, 32);
      Q6_V_vstu_variable(c0, nc_tail * sizeof(float), vout0x0);
      if (nc > 32) {
        const size_t nc_tail = min(nc - 32, 32);
        Q6_V_vstu_variable(c0 + 32, nc_tail * sizeof(float), vout0x1);
      }
      if (nc > 64) {
        const size_t nc_tail = min(nc - 64, 32);
        Q6_V_vstu_variable(c0 + 64, nc_tail * sizeof(float), vout0x2);
      }
      if (nc > 96) {
        const size_t nc_tail = min(nc - 96, 32);
        Q6_V_vstu_variable(c0 + 96, nc_tail * sizeof(float), vout0x3);
      }
      nc = 0;
    }
  } while (nc != 0);
}
