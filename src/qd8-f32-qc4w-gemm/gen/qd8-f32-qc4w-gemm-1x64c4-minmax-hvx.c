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
#include "src/xnnpack/intrinsics-polyfill.h"  // for Q6_V_vstu_variable
#include "src/xnnpack/math.h"
#include "src/xnnpack/simd/f32-hvx.h"
#include "src/xnnpack/simd/s32-hvx.h"
#include "src/xnnpack/unaligned.h"



void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__hvx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_qc4w_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const size_t kc_aligned8 = kc & ~(size_t) 7;
  const int8_t* a0 = a;
  float* c0 = c;


  const HVX_Vector v0x88 = Q6_Vb_vsplat_R(0x88);
  const HVX_Vector vmask0f = Q6_Vb_vsplat_R(0x0F);
  const HVX_Vector vmaskf0 = Q6_Vb_vsplat_R(0xF0);
  const HVX_Vector vones = Q6_Vb_vsplat_R(1);
  int32_t row_sum8_0;
  {
    HVX_Vector vsum = Q6_V_vzero();
    const int8_t* ra_ptr = a0;
    size_t rk = kc;
    for (; rk >= 128; rk -= 128) {
      vsum = Q6_Vw_vrmpyacc_VwVubVb(vsum, vones, *((const HVX_UVector*) ra_ptr));
      ra_ptr += 128;
    }
    vsum = Q6_Vw_vadd_VwVw(vsum, Q6_V_vror_VR(vsum, 64));
    vsum = Q6_Vw_vadd_VwVw(vsum, Q6_V_vror_VR(vsum, 32));
    vsum = Q6_Vw_vadd_VwVw(vsum, Q6_V_vror_VR(vsum, 16));
    vsum = Q6_Vw_vadd_VwVw(vsum, Q6_V_vror_VR(vsum, 8));
    vsum = Q6_Vw_vadd_VwVw(vsum, Q6_V_vror_VR(vsum, 4));
    int32_t sum = Q6_R_vextract_VR(vsum, 0);
    for (; rk != 0; --rk) {
      sum += (int32_t) *ra_ptr++;
    }
    row_sum8_0 = (int32_t) ((uint32_t) sum << 3);
  }
  do {
    HVX_Vector vksum0 = *((const HVX_Vector*) w); w = (const int32_t*) w + 32;
    vksum0 = Q6_Vw_vasr_VwR(vksum0, 4);
    HVX_Vector vksum1 = *((const HVX_Vector*) w); w = (const int32_t*) w + 32;
    vksum1 = Q6_Vw_vasr_VwR(vksum1, 4);
    const HVX_Vector vinput_zero_point0 = Q6_V_vsplat_R(quantization_params[0].zero_point);
    const HVX_Vector vrow_sum8_0 = Q6_V_vsplat_R(row_sum8_0);
    HVX_Vector vacc0x0 = Q6_Vw_vsub_VwVw(xnn_mul_s32(vksum0, vinput_zero_point0), vrow_sum8_0);
    HVX_Vector vacc0x1 = Q6_Vw_vsub_VwVw(xnn_mul_s32(vksum1, vinput_zero_point0), vrow_sum8_0);
    HVX_Vector vacc1x0x0 = Q6_V_vzero();
    HVX_Vector vacc1x0x1 = Q6_V_vzero();

    size_t k = kc;
    // Use unaligned loads if activation pointer is not 4-byte aligned.
    if ((((uintptr_t) a) & (sizeof(int32_t) - 1)) != 0) {
      for (; k >= 8 * sizeof(int8_t); k -= 8 * sizeof(int8_t)) {
        const int32_t ra0x0 = unaligned_load_s32(a0); a0 += 4;
        const int32_t ra0x1 = unaligned_load_s32(a0); a0 += 4;

        const HVX_Vector vw0 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
        const HVX_Vector vu0 = Q6_V_vxor_VV(vw0, v0x88);
        const HVX_Vector vw1 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
        const HVX_Vector vu1 = Q6_V_vxor_VV(vw1, v0x88);

        const HVX_Vector vb0x0 = Q6_V_vand_VV(vu0, vmask0f);
        vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x0, ra0x0);
        const HVX_Vector vb1x0 = Q6_V_vand_VV(vu1, vmask0f);
        vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x0, ra0x0);
        const HVX_Vector vb0x1 = Q6_V_vand_VV(vu0, vmaskf0);
        vacc1x0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x0, vb0x1, ra0x1);
        const HVX_Vector vb1x1 = Q6_V_vand_VV(vu1, vmaskf0);
        vacc1x0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x1, vb1x1, ra0x1);
      }
    } else {
      for (; k >= 8 * sizeof(int8_t); k -= 8 * sizeof(int8_t)) {
        const int32_t ra0x0 = *((const int32_t*) a0); a0 += 4;
        const int32_t ra0x1 = *((const int32_t*) a0); a0 += 4;

        const HVX_Vector vw0 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
        const HVX_Vector vu0 = Q6_V_vxor_VV(vw0, v0x88);
        const HVX_Vector vw1 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
        const HVX_Vector vu1 = Q6_V_vxor_VV(vw1, v0x88);

        const HVX_Vector vb0x0 = Q6_V_vand_VV(vu0, vmask0f);
        vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x0, ra0x0);
        const HVX_Vector vb1x0 = Q6_V_vand_VV(vu1, vmask0f);
        vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x0, ra0x0);
        const HVX_Vector vb0x1 = Q6_V_vand_VV(vu0, vmaskf0);
        vacc1x0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x0, vb0x1, ra0x1);
        const HVX_Vector vb1x1 = Q6_V_vand_VV(vu1, vmaskf0);
        vacc1x0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x1, vb1x1, ra0x1);
      }
    }
    if (k != 0) {
      const HVX_Vector vw0 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
      const HVX_Vector vu0 = Q6_V_vxor_VV(vw0, v0x88);
      const HVX_Vector vw1 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
      const HVX_Vector vu1 = Q6_V_vxor_VV(vw1, v0x88);
      const HVX_Vector vb0x0 = Q6_V_vand_VV(vu0, vmask0f);
      const HVX_Vector vb1x0 = Q6_V_vand_VV(vu1, vmask0f);
      if (k >= 4 * sizeof(int8_t)) {
        const int32_t ra0x0 = unaligned_load_s32(a0);
        vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x0, ra0x0);
        vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x0, ra0x0);
        if (k > 4 * sizeof(int8_t)) {
          const uint32_t tail_mask = UINT32_MAX >> ((8 - k) * 8);
          const HVX_Vector vb0x1 = Q6_V_vand_VV(vu0, vmaskf0);
          const HVX_Vector vb1x1 = Q6_V_vand_VV(vu1, vmaskf0);
          const int32_t ra0x1 = (int32_t) (unaligned_load_u32(a0 + 4) & tail_mask);
          vacc1x0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x0, vb0x1, ra0x1);
          vacc1x0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x1, vb1x1, ra0x1);
        }
      } else {
        const uint32_t tail_mask = UINT32_MAX >> ((4 - k) * 8);
        const int32_t ra0x0 = (int32_t) (unaligned_load_u32(a0) & tail_mask);
        vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x0, ra0x0);
        vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x0, ra0x0);
      }
    }
    vacc0x0 = Q6_Vw_vasracc_VwVwR(vacc0x0, vacc1x0x0, 4);
    vacc0x1 = Q6_Vw_vasracc_VwVwR(vacc0x1, vacc1x0x1, 4);


    const HVX_Vector vinput_scale0 = Q6_V_vsplat_R(float_as_uint32(quantization_params[0].inv_scale));
    HVX_Vector vout0x0 = xnn_mul_f32(xnn_cvt_f32_s32(vacc0x0), vinput_scale0);
    HVX_Vector vout0x1 = xnn_mul_f32(xnn_cvt_f32_s32(vacc0x1), vinput_scale0);

    const HVX_Vector vfilter_output_scale0 = *((const HVX_Vector*) w); w = (const float*) w + 32;
    const HVX_Vector vfilter_output_scale1 = *((const HVX_Vector*) w); w = (const float*) w + 32;
    const HVX_Vector vbias0 = *((const HVX_Vector*) w); w = (const float*) w + 32;
    vout0x0 = xnn_fmadd_f32(vout0x0, vfilter_output_scale0, vbias0);
    const HVX_Vector vbias1 = *((const HVX_Vector*) w); w = (const float*) w + 32;
    vout0x1 = xnn_fmadd_f32(vout0x1, vfilter_output_scale1, vbias1);
    const HVX_Vector vmin = Q6_V_vsplat_R(float_as_uint32(params->scalar.min));
    const HVX_Vector vmax = Q6_V_vsplat_R(float_as_uint32(params->scalar.max));
    vout0x0 = xnn_max_f32(vout0x0, vmin);
    vout0x0 = xnn_min_f32(vout0x0, vmax);
    vout0x1 = xnn_max_f32(vout0x1, vmin);
    vout0x1 = xnn_min_f32(vout0x1, vmax);

    if XNN_LIKELY(nc >= 64) {
      *((HVX_UVector*)(c0 + 0)) = vout0x0;
      *((HVX_UVector*)(c0 + 32)) = vout0x1;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      a0 = (const int8_t*) ((uintptr_t) a0 - kc_aligned8);

      nc -= 64;
    } else {
      const size_t nc_tail = min(nc, 32);
      Q6_V_vstu_variable(c0, nc_tail * sizeof(float), vout0x0);
      if (nc > 32) {
        const size_t nc_tail = min(nc - 32, 32);
        Q6_V_vstu_variable(c0 + 32, nc_tail * sizeof(float), vout0x1);
      }
      nc = 0;
    }
  } while (nc != 0);
}
