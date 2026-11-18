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



void xnn_qd8_f32_qc2w_gemm_minmax_ukernel_4x128c4__hvx(
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
    const float* row_sum,
    const struct xnn_qd8_quantization_params* restrict quantization_params) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);
  assert(row_sum != NULL);

  const size_t kc_aligned16 = kc & ~(size_t) 15;
  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }


  const HVX_Vector vmask03 = Q6_Vb_vsplat_R(0x03);
  do {
    // Load NR int32_t weight sums (ksum), then save pointer to NR float kernel_zero_points (loaded in epilogue).
    const HVX_Vector vksum0 = *((const HVX_Vector*) w); w = (const int32_t*) w + 32;
    const HVX_Vector vksum1 = *((const HVX_Vector*) w); w = (const int32_t*) w + 32;
    const HVX_Vector vksum2 = *((const HVX_Vector*) w); w = (const int32_t*) w + 32;
    const HVX_Vector vksum3 = *((const HVX_Vector*) w); w = (const int32_t*) w + 32;
    const HVX_Vector* kzp = (const HVX_Vector*) w;
    w = (const float*) w + 128;
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
    const HVX_Vector vinput_zero_point2 = Q6_V_vsplat_R(quantization_params[2].zero_point);
    HVX_Vector vacc2x0 = xnn_mul_s32(vksum0, vinput_zero_point2);
    HVX_Vector vacc2x1 = xnn_mul_s32(vksum1, vinput_zero_point2);
    HVX_Vector vacc2x2 = xnn_mul_s32(vksum2, vinput_zero_point2);
    HVX_Vector vacc2x3 = xnn_mul_s32(vksum3, vinput_zero_point2);
    const HVX_Vector vinput_zero_point3 = Q6_V_vsplat_R(quantization_params[3].zero_point);
    HVX_Vector vacc3x0 = xnn_mul_s32(vksum0, vinput_zero_point3);
    HVX_Vector vacc3x1 = xnn_mul_s32(vksum1, vinput_zero_point3);
    HVX_Vector vacc3x2 = xnn_mul_s32(vksum2, vinput_zero_point3);
    HVX_Vector vacc3x3 = xnn_mul_s32(vksum3, vinput_zero_point3);

    size_t k = kc;
    // Use unaligned loads if activation pointer or row stride is not 4-byte aligned.
    if (((((uintptr_t) a) | a_stride) & (sizeof(int32_t) - 1)) != 0) {
      for (; k >= 16 * sizeof(int8_t); k -= 16 * sizeof(int8_t)) {
        const int32_t ra0x0 = unaligned_load_s32(a0); a0 += 4;
        const int32_t ra0x1 = unaligned_load_s32(a0); a0 += 4;
        const int32_t ra0x2 = unaligned_load_s32(a0); a0 += 4;
        const int32_t ra0x3 = unaligned_load_s32(a0); a0 += 4;
        const int32_t ra1x0 = unaligned_load_s32(a1); a1 += 4;
        const int32_t ra1x1 = unaligned_load_s32(a1); a1 += 4;
        const int32_t ra1x2 = unaligned_load_s32(a1); a1 += 4;
        const int32_t ra1x3 = unaligned_load_s32(a1); a1 += 4;
        const int32_t ra2x0 = unaligned_load_s32(a2); a2 += 4;
        const int32_t ra2x1 = unaligned_load_s32(a2); a2 += 4;
        const int32_t ra2x2 = unaligned_load_s32(a2); a2 += 4;
        const int32_t ra2x3 = unaligned_load_s32(a2); a2 += 4;
        const int32_t ra3x0 = unaligned_load_s32(a3); a3 += 4;
        const int32_t ra3x1 = unaligned_load_s32(a3); a3 += 4;
        const int32_t ra3x2 = unaligned_load_s32(a3); a3 += 4;
        const int32_t ra3x3 = unaligned_load_s32(a3); a3 += 4;

        const HVX_Vector vbb0 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
        const HVX_Vector vbb1 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
        const HVX_Vector vbb2 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
        const HVX_Vector vbb3 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;

        const HVX_Vector vb0x0 = Q6_V_vand_VV(vbb0, vmask03);
        const HVX_Vector vb1x0 = Q6_V_vand_VV(vbb1, vmask03);
        const HVX_Vector vb2x0 = Q6_V_vand_VV(vbb2, vmask03);
        const HVX_Vector vb3x0 = Q6_V_vand_VV(vbb3, vmask03);
        vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x0, ra0x0);
        vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x0, ra0x0);
        vacc0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x2, vb2x0, ra0x0);
        vacc0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x3, vb3x0, ra0x0);
        vacc1x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0, vb0x0, ra1x0);
        vacc1x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x1, vb1x0, ra1x0);
        vacc1x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x2, vb2x0, ra1x0);
        vacc1x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x3, vb3x0, ra1x0);
        vacc2x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x0, vb0x0, ra2x0);
        vacc2x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x1, vb1x0, ra2x0);
        vacc2x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x2, vb2x0, ra2x0);
        vacc2x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x3, vb3x0, ra2x0);
        vacc3x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x0, vb0x0, ra3x0);
        vacc3x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x1, vb1x0, ra3x0);
        vacc3x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x2, vb2x0, ra3x0);
        vacc3x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x3, vb3x0, ra3x0);
        const HVX_Vector vb0x1 = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(vbb0, 2), vmask03);
        const HVX_Vector vb1x1 = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(vbb1, 2), vmask03);
        const HVX_Vector vb2x1 = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(vbb2, 2), vmask03);
        const HVX_Vector vb3x1 = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(vbb3, 2), vmask03);
        vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x1, ra0x1);
        vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x1, ra0x1);
        vacc0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x2, vb2x1, ra0x1);
        vacc0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x3, vb3x1, ra0x1);
        vacc1x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0, vb0x1, ra1x1);
        vacc1x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x1, vb1x1, ra1x1);
        vacc1x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x2, vb2x1, ra1x1);
        vacc1x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x3, vb3x1, ra1x1);
        vacc2x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x0, vb0x1, ra2x1);
        vacc2x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x1, vb1x1, ra2x1);
        vacc2x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x2, vb2x1, ra2x1);
        vacc2x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x3, vb3x1, ra2x1);
        vacc3x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x0, vb0x1, ra3x1);
        vacc3x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x1, vb1x1, ra3x1);
        vacc3x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x2, vb2x1, ra3x1);
        vacc3x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x3, vb3x1, ra3x1);
        const HVX_Vector vb0x2 = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(vbb0, 4), vmask03);
        const HVX_Vector vb1x2 = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(vbb1, 4), vmask03);
        const HVX_Vector vb2x2 = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(vbb2, 4), vmask03);
        const HVX_Vector vb3x2 = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(vbb3, 4), vmask03);
        vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x2, ra0x2);
        vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x2, ra0x2);
        vacc0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x2, vb2x2, ra0x2);
        vacc0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x3, vb3x2, ra0x2);
        vacc1x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0, vb0x2, ra1x2);
        vacc1x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x1, vb1x2, ra1x2);
        vacc1x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x2, vb2x2, ra1x2);
        vacc1x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x3, vb3x2, ra1x2);
        vacc2x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x0, vb0x2, ra2x2);
        vacc2x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x1, vb1x2, ra2x2);
        vacc2x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x2, vb2x2, ra2x2);
        vacc2x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x3, vb3x2, ra2x2);
        vacc3x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x0, vb0x2, ra3x2);
        vacc3x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x1, vb1x2, ra3x2);
        vacc3x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x2, vb2x2, ra3x2);
        vacc3x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x3, vb3x2, ra3x2);
        const HVX_Vector vb0x3 = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(vbb0, 6), vmask03);
        const HVX_Vector vb1x3 = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(vbb1, 6), vmask03);
        const HVX_Vector vb2x3 = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(vbb2, 6), vmask03);
        const HVX_Vector vb3x3 = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(vbb3, 6), vmask03);
        vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x3, ra0x3);
        vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x3, ra0x3);
        vacc0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x2, vb2x3, ra0x3);
        vacc0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x3, vb3x3, ra0x3);
        vacc1x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0, vb0x3, ra1x3);
        vacc1x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x1, vb1x3, ra1x3);
        vacc1x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x2, vb2x3, ra1x3);
        vacc1x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x3, vb3x3, ra1x3);
        vacc2x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x0, vb0x3, ra2x3);
        vacc2x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x1, vb1x3, ra2x3);
        vacc2x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x2, vb2x3, ra2x3);
        vacc2x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x3, vb3x3, ra2x3);
        vacc3x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x0, vb0x3, ra3x3);
        vacc3x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x1, vb1x3, ra3x3);
        vacc3x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x2, vb2x3, ra3x3);
        vacc3x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x3, vb3x3, ra3x3);
      }
    } else {
      for (; k >= 16 * sizeof(int8_t); k -= 16 * sizeof(int8_t)) {
        HVX_Vector vbb0 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
        HVX_Vector vbb1 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
        HVX_Vector vbb2 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
        HVX_Vector vbb3 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;

        const int32_t ra0x0 = *((const int32_t*) a0); a0 += 4;
        const int32_t ra1x0 = *((const int32_t*) a1); a1 += 4;
        const int32_t ra2x0 = *((const int32_t*) a2); a2 += 4;
        const int32_t ra3x0 = *((const int32_t*) a3); a3 += 4;
        const HVX_Vector vb0x0 = Q6_V_vand_VV(vbb0, vmask03);
        const HVX_Vector vb1x0 = Q6_V_vand_VV(vbb1, vmask03);
        const HVX_Vector vb2x0 = Q6_V_vand_VV(vbb2, vmask03);
        const HVX_Vector vb3x0 = Q6_V_vand_VV(vbb3, vmask03);
        vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x0, ra0x0);
        vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x0, ra0x0);
        vacc0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x2, vb2x0, ra0x0);
        vacc0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x3, vb3x0, ra0x0);
        vacc1x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0, vb0x0, ra1x0);
        vacc1x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x1, vb1x0, ra1x0);
        vacc1x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x2, vb2x0, ra1x0);
        vacc1x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x3, vb3x0, ra1x0);
        vacc2x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x0, vb0x0, ra2x0);
        vacc2x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x1, vb1x0, ra2x0);
        vacc2x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x2, vb2x0, ra2x0);
        vacc2x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x3, vb3x0, ra2x0);
        vacc3x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x0, vb0x0, ra3x0);
        vacc3x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x1, vb1x0, ra3x0);
        vacc3x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x2, vb2x0, ra3x0);
        vacc3x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x3, vb3x0, ra3x0);

        const int32_t ra0x1 = *((const int32_t*) a0); a0 += 4;
        const int32_t ra1x1 = *((const int32_t*) a1); a1 += 4;
        const int32_t ra2x1 = *((const int32_t*) a2); a2 += 4;
        const int32_t ra3x1 = *((const int32_t*) a3); a3 += 4;
        vbb0 = Q6_Vuw_vlsr_VuwR(vbb0, 2);
        const HVX_Vector vb0x1 = Q6_V_vand_VV(vbb0, vmask03);
        vbb1 = Q6_Vuw_vlsr_VuwR(vbb1, 2);
        const HVX_Vector vb1x1 = Q6_V_vand_VV(vbb1, vmask03);
        vbb2 = Q6_Vuw_vlsr_VuwR(vbb2, 2);
        const HVX_Vector vb2x1 = Q6_V_vand_VV(vbb2, vmask03);
        vbb3 = Q6_Vuw_vlsr_VuwR(vbb3, 2);
        const HVX_Vector vb3x1 = Q6_V_vand_VV(vbb3, vmask03);
        vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x1, ra0x1);
        vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x1, ra0x1);
        vacc0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x2, vb2x1, ra0x1);
        vacc0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x3, vb3x1, ra0x1);
        vacc1x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0, vb0x1, ra1x1);
        vacc1x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x1, vb1x1, ra1x1);
        vacc1x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x2, vb2x1, ra1x1);
        vacc1x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x3, vb3x1, ra1x1);
        vacc2x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x0, vb0x1, ra2x1);
        vacc2x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x1, vb1x1, ra2x1);
        vacc2x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x2, vb2x1, ra2x1);
        vacc2x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x3, vb3x1, ra2x1);
        vacc3x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x0, vb0x1, ra3x1);
        vacc3x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x1, vb1x1, ra3x1);
        vacc3x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x2, vb2x1, ra3x1);
        vacc3x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x3, vb3x1, ra3x1);

        const int32_t ra0x2 = *((const int32_t*) a0); a0 += 4;
        const int32_t ra1x2 = *((const int32_t*) a1); a1 += 4;
        const int32_t ra2x2 = *((const int32_t*) a2); a2 += 4;
        const int32_t ra3x2 = *((const int32_t*) a3); a3 += 4;
        vbb0 = Q6_Vuw_vlsr_VuwR(vbb0, 2);
        const HVX_Vector vb0x2 = Q6_V_vand_VV(vbb0, vmask03);
        vbb1 = Q6_Vuw_vlsr_VuwR(vbb1, 2);
        const HVX_Vector vb1x2 = Q6_V_vand_VV(vbb1, vmask03);
        vbb2 = Q6_Vuw_vlsr_VuwR(vbb2, 2);
        const HVX_Vector vb2x2 = Q6_V_vand_VV(vbb2, vmask03);
        vbb3 = Q6_Vuw_vlsr_VuwR(vbb3, 2);
        const HVX_Vector vb3x2 = Q6_V_vand_VV(vbb3, vmask03);
        vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x2, ra0x2);
        vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x2, ra0x2);
        vacc0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x2, vb2x2, ra0x2);
        vacc0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x3, vb3x2, ra0x2);
        vacc1x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0, vb0x2, ra1x2);
        vacc1x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x1, vb1x2, ra1x2);
        vacc1x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x2, vb2x2, ra1x2);
        vacc1x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x3, vb3x2, ra1x2);
        vacc2x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x0, vb0x2, ra2x2);
        vacc2x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x1, vb1x2, ra2x2);
        vacc2x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x2, vb2x2, ra2x2);
        vacc2x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x3, vb3x2, ra2x2);
        vacc3x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x0, vb0x2, ra3x2);
        vacc3x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x1, vb1x2, ra3x2);
        vacc3x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x2, vb2x2, ra3x2);
        vacc3x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x3, vb3x2, ra3x2);

        const int32_t ra0x3 = *((const int32_t*) a0); a0 += 4;
        const int32_t ra1x3 = *((const int32_t*) a1); a1 += 4;
        const int32_t ra2x3 = *((const int32_t*) a2); a2 += 4;
        const int32_t ra3x3 = *((const int32_t*) a3); a3 += 4;
        vbb0 = Q6_Vuw_vlsr_VuwR(vbb0, 2);
        const HVX_Vector vb0x3 = Q6_V_vand_VV(vbb0, vmask03);
        vbb1 = Q6_Vuw_vlsr_VuwR(vbb1, 2);
        const HVX_Vector vb1x3 = Q6_V_vand_VV(vbb1, vmask03);
        vbb2 = Q6_Vuw_vlsr_VuwR(vbb2, 2);
        const HVX_Vector vb2x3 = Q6_V_vand_VV(vbb2, vmask03);
        vbb3 = Q6_Vuw_vlsr_VuwR(vbb3, 2);
        const HVX_Vector vb3x3 = Q6_V_vand_VV(vbb3, vmask03);
        vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x3, ra0x3);
        vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x3, ra0x3);
        vacc0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x2, vb2x3, ra0x3);
        vacc0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x3, vb3x3, ra0x3);
        vacc1x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0, vb0x3, ra1x3);
        vacc1x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x1, vb1x3, ra1x3);
        vacc1x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x2, vb2x3, ra1x3);
        vacc1x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x3, vb3x3, ra1x3);
        vacc2x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x0, vb0x3, ra2x3);
        vacc2x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x1, vb1x3, ra2x3);
        vacc2x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x2, vb2x3, ra2x3);
        vacc2x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x3, vb3x3, ra2x3);
        vacc3x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x0, vb0x3, ra3x3);
        vacc3x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x1, vb1x3, ra3x3);
        vacc3x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x2, vb2x3, ra3x3);
        vacc3x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x3, vb3x3, ra3x3);
      }
    }
    if (k != 0) {
      const HVX_Vector vbb0 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
      const HVX_Vector vbb1 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
      const HVX_Vector vbb2 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
      const HVX_Vector vbb3 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
      const HVX_Vector vb0x0 = Q6_V_vand_VV(vbb0, vmask03);
      const HVX_Vector vb1x0 = Q6_V_vand_VV(vbb1, vmask03);
      const HVX_Vector vb2x0 = Q6_V_vand_VV(vbb2, vmask03);
      const HVX_Vector vb3x0 = Q6_V_vand_VV(vbb3, vmask03);
      if (k >= 4 * sizeof(int8_t)) {
        const int32_t ra0x0 = unaligned_load_s32(a0);
        vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x0, ra0x0);
        vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x0, ra0x0);
        vacc0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x2, vb2x0, ra0x0);
        vacc0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x3, vb3x0, ra0x0);
        const int32_t ra1x0 = unaligned_load_s32(a1);
        vacc1x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0, vb0x0, ra1x0);
        vacc1x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x1, vb1x0, ra1x0);
        vacc1x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x2, vb2x0, ra1x0);
        vacc1x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x3, vb3x0, ra1x0);
        const int32_t ra2x0 = unaligned_load_s32(a2);
        vacc2x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x0, vb0x0, ra2x0);
        vacc2x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x1, vb1x0, ra2x0);
        vacc2x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x2, vb2x0, ra2x0);
        vacc2x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x3, vb3x0, ra2x0);
        const int32_t ra3x0 = unaligned_load_s32(a3);
        vacc3x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x0, vb0x0, ra3x0);
        vacc3x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x1, vb1x0, ra3x0);
        vacc3x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x2, vb2x0, ra3x0);
        vacc3x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x3, vb3x0, ra3x0);

        if (k > 4 * sizeof(int8_t)) {
          const HVX_Vector vb0x1 = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(vbb0, 2), vmask03);
          const HVX_Vector vb1x1 = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(vbb1, 2), vmask03);
          const HVX_Vector vb2x1 = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(vbb2, 2), vmask03);
          const HVX_Vector vb3x1 = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(vbb3, 2), vmask03);
          if (k >= 8 * sizeof(int8_t)) {
            const int32_t ra0x1 = unaligned_load_s32(a0 + 4);
            vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x1, ra0x1);
            vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x1, ra0x1);
            vacc0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x2, vb2x1, ra0x1);
            vacc0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x3, vb3x1, ra0x1);
            const int32_t ra1x1 = unaligned_load_s32(a1 + 4);
            vacc1x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0, vb0x1, ra1x1);
            vacc1x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x1, vb1x1, ra1x1);
            vacc1x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x2, vb2x1, ra1x1);
            vacc1x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x3, vb3x1, ra1x1);
            const int32_t ra2x1 = unaligned_load_s32(a2 + 4);
            vacc2x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x0, vb0x1, ra2x1);
            vacc2x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x1, vb1x1, ra2x1);
            vacc2x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x2, vb2x1, ra2x1);
            vacc2x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x3, vb3x1, ra2x1);
            const int32_t ra3x1 = unaligned_load_s32(a3 + 4);
            vacc3x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x0, vb0x1, ra3x1);
            vacc3x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x1, vb1x1, ra3x1);
            vacc3x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x2, vb2x1, ra3x1);
            vacc3x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x3, vb3x1, ra3x1);

            if (k > 8 * sizeof(int8_t)) {
              const HVX_Vector vbb_hi0 = Q6_Vuw_vlsr_VuwR(vbb0, 4);
              const HVX_Vector vbb_hi1 = Q6_Vuw_vlsr_VuwR(vbb1, 4);
              const HVX_Vector vbb_hi2 = Q6_Vuw_vlsr_VuwR(vbb2, 4);
              const HVX_Vector vbb_hi3 = Q6_Vuw_vlsr_VuwR(vbb3, 4);
              const HVX_Vector vb0x2 = Q6_V_vand_VV(vbb_hi0, vmask03);
              const HVX_Vector vb1x2 = Q6_V_vand_VV(vbb_hi1, vmask03);
              const HVX_Vector vb2x2 = Q6_V_vand_VV(vbb_hi2, vmask03);
              const HVX_Vector vb3x2 = Q6_V_vand_VV(vbb_hi3, vmask03);
              if (k >= 12 * sizeof(int8_t)) {
                const int32_t ra0x2 = unaligned_load_s32(a0 + 8);
                vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x2, ra0x2);
                vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x2, ra0x2);
                vacc0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x2, vb2x2, ra0x2);
                vacc0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x3, vb3x2, ra0x2);
                const int32_t ra1x2 = unaligned_load_s32(a1 + 8);
                vacc1x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0, vb0x2, ra1x2);
                vacc1x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x1, vb1x2, ra1x2);
                vacc1x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x2, vb2x2, ra1x2);
                vacc1x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x3, vb3x2, ra1x2);
                const int32_t ra2x2 = unaligned_load_s32(a2 + 8);
                vacc2x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x0, vb0x2, ra2x2);
                vacc2x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x1, vb1x2, ra2x2);
                vacc2x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x2, vb2x2, ra2x2);
                vacc2x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x3, vb3x2, ra2x2);
                const int32_t ra3x2 = unaligned_load_s32(a3 + 8);
                vacc3x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x0, vb0x2, ra3x2);
                vacc3x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x1, vb1x2, ra3x2);
                vacc3x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x2, vb2x2, ra3x2);
                vacc3x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x3, vb3x2, ra3x2);

                if (k > 12 * sizeof(int8_t)) {
                  const uint32_t tail_mask = UINT32_MAX >> ((16 - k) * 8);
                  const HVX_Vector vb0x3 = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(vbb_hi0, 2), vmask03);
                  const HVX_Vector vb1x3 = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(vbb_hi1, 2), vmask03);
                  const HVX_Vector vb2x3 = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(vbb_hi2, 2), vmask03);
                  const HVX_Vector vb3x3 = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(vbb_hi3, 2), vmask03);
                  const int32_t ra0x3 = (int32_t) (unaligned_load_u32(a0 + 12) & tail_mask);
                  vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x3, ra0x3);
                  vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x3, ra0x3);
                  vacc0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x2, vb2x3, ra0x3);
                  vacc0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x3, vb3x3, ra0x3);
                  const int32_t ra1x3 = (int32_t) (unaligned_load_u32(a1 + 12) & tail_mask);
                  vacc1x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0, vb0x3, ra1x3);
                  vacc1x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x1, vb1x3, ra1x3);
                  vacc1x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x2, vb2x3, ra1x3);
                  vacc1x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x3, vb3x3, ra1x3);
                  const int32_t ra2x3 = (int32_t) (unaligned_load_u32(a2 + 12) & tail_mask);
                  vacc2x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x0, vb0x3, ra2x3);
                  vacc2x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x1, vb1x3, ra2x3);
                  vacc2x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x2, vb2x3, ra2x3);
                  vacc2x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x3, vb3x3, ra2x3);
                  const int32_t ra3x3 = (int32_t) (unaligned_load_u32(a3 + 12) & tail_mask);
                  vacc3x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x0, vb0x3, ra3x3);
                  vacc3x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x1, vb1x3, ra3x3);
                  vacc3x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x2, vb2x3, ra3x3);
                  vacc3x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x3, vb3x3, ra3x3);
                }
              } else {
                const uint32_t tail_mask = UINT32_MAX >> ((12 - k) * 8);
                const int32_t ra0x2 = (int32_t) (unaligned_load_u32(a0 + 8) & tail_mask);
                vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x2, ra0x2);
                vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x2, ra0x2);
                vacc0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x2, vb2x2, ra0x2);
                vacc0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x3, vb3x2, ra0x2);
                const int32_t ra1x2 = (int32_t) (unaligned_load_u32(a1 + 8) & tail_mask);
                vacc1x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0, vb0x2, ra1x2);
                vacc1x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x1, vb1x2, ra1x2);
                vacc1x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x2, vb2x2, ra1x2);
                vacc1x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x3, vb3x2, ra1x2);
                const int32_t ra2x2 = (int32_t) (unaligned_load_u32(a2 + 8) & tail_mask);
                vacc2x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x0, vb0x2, ra2x2);
                vacc2x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x1, vb1x2, ra2x2);
                vacc2x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x2, vb2x2, ra2x2);
                vacc2x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x3, vb3x2, ra2x2);
                const int32_t ra3x2 = (int32_t) (unaligned_load_u32(a3 + 8) & tail_mask);
                vacc3x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x0, vb0x2, ra3x2);
                vacc3x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x1, vb1x2, ra3x2);
                vacc3x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x2, vb2x2, ra3x2);
                vacc3x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x3, vb3x2, ra3x2);
              }
            }
          } else {
            const uint32_t tail_mask = UINT32_MAX >> ((8 - k) * 8);
            const int32_t ra0x1 = (int32_t) (unaligned_load_u32(a0 + 4) & tail_mask);
            vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x1, ra0x1);
            vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x1, ra0x1);
            vacc0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x2, vb2x1, ra0x1);
            vacc0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x3, vb3x1, ra0x1);
            const int32_t ra1x1 = (int32_t) (unaligned_load_u32(a1 + 4) & tail_mask);
            vacc1x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0, vb0x1, ra1x1);
            vacc1x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x1, vb1x1, ra1x1);
            vacc1x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x2, vb2x1, ra1x1);
            vacc1x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x3, vb3x1, ra1x1);
            const int32_t ra2x1 = (int32_t) (unaligned_load_u32(a2 + 4) & tail_mask);
            vacc2x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x0, vb0x1, ra2x1);
            vacc2x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x1, vb1x1, ra2x1);
            vacc2x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x2, vb2x1, ra2x1);
            vacc2x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x3, vb3x1, ra2x1);
            const int32_t ra3x1 = (int32_t) (unaligned_load_u32(a3 + 4) & tail_mask);
            vacc3x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x0, vb0x1, ra3x1);
            vacc3x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x1, vb1x1, ra3x1);
            vacc3x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x2, vb2x1, ra3x1);
            vacc3x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x3, vb3x1, ra3x1);
          }
        }
      } else {
        const uint32_t tail_mask = UINT32_MAX >> ((4 - k) * 8);
        const int32_t ra0x0 = (int32_t) (unaligned_load_u32(a0) & tail_mask);
        vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x0, ra0x0);
        vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x0, ra0x0);
        vacc0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x2, vb2x0, ra0x0);
        vacc0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x3, vb3x0, ra0x0);
        const int32_t ra1x0 = (int32_t) (unaligned_load_u32(a1) & tail_mask);
        vacc1x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0, vb0x0, ra1x0);
        vacc1x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x1, vb1x0, ra1x0);
        vacc1x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x2, vb2x0, ra1x0);
        vacc1x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x3, vb3x0, ra1x0);
        const int32_t ra2x0 = (int32_t) (unaligned_load_u32(a2) & tail_mask);
        vacc2x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x0, vb0x0, ra2x0);
        vacc2x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x1, vb1x0, ra2x0);
        vacc2x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x2, vb2x0, ra2x0);
        vacc2x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc2x3, vb3x0, ra2x0);
        const int32_t ra3x0 = (int32_t) (unaligned_load_u32(a3) & tail_mask);
        vacc3x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x0, vb0x0, ra3x0);
        vacc3x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x1, vb1x0, ra3x0);
        vacc3x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x2, vb2x0, ra3x0);
        vacc3x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc3x3, vb3x0, ra3x0);
      }
    }


    const HVX_Vector vtwo_row_sum_0 = Q6_V_vsplat_R(float_as_uint32(2.0f * row_sum[0]));
    HVX_Vector vout0x0 = xnn_sub_f32(xnn_cvt_f32_s32(vacc0x0), vtwo_row_sum_0);
    HVX_Vector vout0x1 = xnn_sub_f32(xnn_cvt_f32_s32(vacc0x1), vtwo_row_sum_0);
    HVX_Vector vout0x2 = xnn_sub_f32(xnn_cvt_f32_s32(vacc0x2), vtwo_row_sum_0);
    HVX_Vector vout0x3 = xnn_sub_f32(xnn_cvt_f32_s32(vacc0x3), vtwo_row_sum_0);
    const HVX_Vector vtwo_row_sum_1 = Q6_V_vsplat_R(float_as_uint32(2.0f * row_sum[1]));
    HVX_Vector vout1x0 = xnn_sub_f32(xnn_cvt_f32_s32(vacc1x0), vtwo_row_sum_1);
    HVX_Vector vout1x1 = xnn_sub_f32(xnn_cvt_f32_s32(vacc1x1), vtwo_row_sum_1);
    HVX_Vector vout1x2 = xnn_sub_f32(xnn_cvt_f32_s32(vacc1x2), vtwo_row_sum_1);
    HVX_Vector vout1x3 = xnn_sub_f32(xnn_cvt_f32_s32(vacc1x3), vtwo_row_sum_1);
    const HVX_Vector vtwo_row_sum_2 = Q6_V_vsplat_R(float_as_uint32(2.0f * row_sum[2]));
    HVX_Vector vout2x0 = xnn_sub_f32(xnn_cvt_f32_s32(vacc2x0), vtwo_row_sum_2);
    HVX_Vector vout2x1 = xnn_sub_f32(xnn_cvt_f32_s32(vacc2x1), vtwo_row_sum_2);
    HVX_Vector vout2x2 = xnn_sub_f32(xnn_cvt_f32_s32(vacc2x2), vtwo_row_sum_2);
    HVX_Vector vout2x3 = xnn_sub_f32(xnn_cvt_f32_s32(vacc2x3), vtwo_row_sum_2);
    const HVX_Vector vtwo_row_sum_3 = Q6_V_vsplat_R(float_as_uint32(2.0f * row_sum[3]));
    HVX_Vector vout3x0 = xnn_sub_f32(xnn_cvt_f32_s32(vacc3x0), vtwo_row_sum_3);
    HVX_Vector vout3x1 = xnn_sub_f32(xnn_cvt_f32_s32(vacc3x1), vtwo_row_sum_3);
    HVX_Vector vout3x2 = xnn_sub_f32(xnn_cvt_f32_s32(vacc3x2), vtwo_row_sum_3);
    HVX_Vector vout3x3 = xnn_sub_f32(xnn_cvt_f32_s32(vacc3x3), vtwo_row_sum_3);

    const HVX_Vector vkzp0 = kzp[0];
    const HVX_Vector vkzp1 = kzp[1];
    const HVX_Vector vkzp2 = kzp[2];
    const HVX_Vector vkzp3 = kzp[3];
    const HVX_Vector vkc_izp_less_row_sum_0 = Q6_V_vsplat_R(float_as_uint32((float) kc * (float) quantization_params[0].zero_point - row_sum[0]));
    vout0x0 = xnn_fmadd_f32(vkzp0, vkc_izp_less_row_sum_0, vout0x0);
    vout0x1 = xnn_fmadd_f32(vkzp1, vkc_izp_less_row_sum_0, vout0x1);
    vout0x2 = xnn_fmadd_f32(vkzp2, vkc_izp_less_row_sum_0, vout0x2);
    vout0x3 = xnn_fmadd_f32(vkzp3, vkc_izp_less_row_sum_0, vout0x3);
    const HVX_Vector vkc_izp_less_row_sum_1 = Q6_V_vsplat_R(float_as_uint32((float) kc * (float) quantization_params[1].zero_point - row_sum[1]));
    vout1x0 = xnn_fmadd_f32(vkzp0, vkc_izp_less_row_sum_1, vout1x0);
    vout1x1 = xnn_fmadd_f32(vkzp1, vkc_izp_less_row_sum_1, vout1x1);
    vout1x2 = xnn_fmadd_f32(vkzp2, vkc_izp_less_row_sum_1, vout1x2);
    vout1x3 = xnn_fmadd_f32(vkzp3, vkc_izp_less_row_sum_1, vout1x3);
    const HVX_Vector vkc_izp_less_row_sum_2 = Q6_V_vsplat_R(float_as_uint32((float) kc * (float) quantization_params[2].zero_point - row_sum[2]));
    vout2x0 = xnn_fmadd_f32(vkzp0, vkc_izp_less_row_sum_2, vout2x0);
    vout2x1 = xnn_fmadd_f32(vkzp1, vkc_izp_less_row_sum_2, vout2x1);
    vout2x2 = xnn_fmadd_f32(vkzp2, vkc_izp_less_row_sum_2, vout2x2);
    vout2x3 = xnn_fmadd_f32(vkzp3, vkc_izp_less_row_sum_2, vout2x3);
    const HVX_Vector vkc_izp_less_row_sum_3 = Q6_V_vsplat_R(float_as_uint32((float) kc * (float) quantization_params[3].zero_point - row_sum[3]));
    vout3x0 = xnn_fmadd_f32(vkzp0, vkc_izp_less_row_sum_3, vout3x0);
    vout3x1 = xnn_fmadd_f32(vkzp1, vkc_izp_less_row_sum_3, vout3x1);
    vout3x2 = xnn_fmadd_f32(vkzp2, vkc_izp_less_row_sum_3, vout3x2);
    vout3x3 = xnn_fmadd_f32(vkzp3, vkc_izp_less_row_sum_3, vout3x3);

    const HVX_Vector vinput_scale0 = Q6_V_vsplat_R(float_as_uint32(quantization_params[0].inv_scale));
    vout0x0 = xnn_mul_f32(vout0x0, vinput_scale0);
    vout0x1 = xnn_mul_f32(vout0x1, vinput_scale0);
    vout0x2 = xnn_mul_f32(vout0x2, vinput_scale0);
    vout0x3 = xnn_mul_f32(vout0x3, vinput_scale0);
    const HVX_Vector vinput_scale1 = Q6_V_vsplat_R(float_as_uint32(quantization_params[1].inv_scale));
    vout1x0 = xnn_mul_f32(vout1x0, vinput_scale1);
    vout1x1 = xnn_mul_f32(vout1x1, vinput_scale1);
    vout1x2 = xnn_mul_f32(vout1x2, vinput_scale1);
    vout1x3 = xnn_mul_f32(vout1x3, vinput_scale1);
    const HVX_Vector vinput_scale2 = Q6_V_vsplat_R(float_as_uint32(quantization_params[2].inv_scale));
    vout2x0 = xnn_mul_f32(vout2x0, vinput_scale2);
    vout2x1 = xnn_mul_f32(vout2x1, vinput_scale2);
    vout2x2 = xnn_mul_f32(vout2x2, vinput_scale2);
    vout2x3 = xnn_mul_f32(vout2x3, vinput_scale2);
    const HVX_Vector vinput_scale3 = Q6_V_vsplat_R(float_as_uint32(quantization_params[3].inv_scale));
    vout3x0 = xnn_mul_f32(vout3x0, vinput_scale3);
    vout3x1 = xnn_mul_f32(vout3x1, vinput_scale3);
    vout3x2 = xnn_mul_f32(vout3x2, vinput_scale3);
    vout3x3 = xnn_mul_f32(vout3x3, vinput_scale3);

    const HVX_Vector vfilter_output_scale0 = *((const HVX_Vector*) w); w = (const float*) w + 32;
    const HVX_Vector vfilter_output_scale1 = *((const HVX_Vector*) w); w = (const float*) w + 32;
    const HVX_Vector vfilter_output_scale2 = *((const HVX_Vector*) w); w = (const float*) w + 32;
    const HVX_Vector vfilter_output_scale3 = *((const HVX_Vector*) w); w = (const float*) w + 32;
    const HVX_Vector vbias0 = *((const HVX_Vector*) w); w = (const float*) w + 32;
    vout0x0 = xnn_fmadd_f32(vout0x0, vfilter_output_scale0, vbias0);
    vout1x0 = xnn_fmadd_f32(vout1x0, vfilter_output_scale0, vbias0);
    vout2x0 = xnn_fmadd_f32(vout2x0, vfilter_output_scale0, vbias0);
    vout3x0 = xnn_fmadd_f32(vout3x0, vfilter_output_scale0, vbias0);
    const HVX_Vector vbias1 = *((const HVX_Vector*) w); w = (const float*) w + 32;
    vout0x1 = xnn_fmadd_f32(vout0x1, vfilter_output_scale1, vbias1);
    vout1x1 = xnn_fmadd_f32(vout1x1, vfilter_output_scale1, vbias1);
    vout2x1 = xnn_fmadd_f32(vout2x1, vfilter_output_scale1, vbias1);
    vout3x1 = xnn_fmadd_f32(vout3x1, vfilter_output_scale1, vbias1);
    const HVX_Vector vbias2 = *((const HVX_Vector*) w); w = (const float*) w + 32;
    vout0x2 = xnn_fmadd_f32(vout0x2, vfilter_output_scale2, vbias2);
    vout1x2 = xnn_fmadd_f32(vout1x2, vfilter_output_scale2, vbias2);
    vout2x2 = xnn_fmadd_f32(vout2x2, vfilter_output_scale2, vbias2);
    vout3x2 = xnn_fmadd_f32(vout3x2, vfilter_output_scale2, vbias2);
    const HVX_Vector vbias3 = *((const HVX_Vector*) w); w = (const float*) w + 32;
    vout0x3 = xnn_fmadd_f32(vout0x3, vfilter_output_scale3, vbias3);
    vout1x3 = xnn_fmadd_f32(vout1x3, vfilter_output_scale3, vbias3);
    vout2x3 = xnn_fmadd_f32(vout2x3, vfilter_output_scale3, vbias3);
    vout3x3 = xnn_fmadd_f32(vout3x3, vfilter_output_scale3, vbias3);
    const HVX_Vector vmin = Q6_V_vsplat_R(float_as_uint32(params->scalar.min));
    const HVX_Vector vmax = Q6_V_vsplat_R(float_as_uint32(params->scalar.max));
    vout0x0 = xnn_max_f32(vout0x0, vmin);
    vout0x0 = xnn_min_f32(vout0x0, vmax);
    vout0x1 = xnn_max_f32(vout0x1, vmin);
    vout0x1 = xnn_min_f32(vout0x1, vmax);
    vout0x2 = xnn_max_f32(vout0x2, vmin);
    vout0x2 = xnn_min_f32(vout0x2, vmax);
    vout0x3 = xnn_max_f32(vout0x3, vmin);
    vout0x3 = xnn_min_f32(vout0x3, vmax);
    vout1x0 = xnn_max_f32(vout1x0, vmin);
    vout1x0 = xnn_min_f32(vout1x0, vmax);
    vout1x1 = xnn_max_f32(vout1x1, vmin);
    vout1x1 = xnn_min_f32(vout1x1, vmax);
    vout1x2 = xnn_max_f32(vout1x2, vmin);
    vout1x2 = xnn_min_f32(vout1x2, vmax);
    vout1x3 = xnn_max_f32(vout1x3, vmin);
    vout1x3 = xnn_min_f32(vout1x3, vmax);
    vout2x0 = xnn_max_f32(vout2x0, vmin);
    vout2x0 = xnn_min_f32(vout2x0, vmax);
    vout2x1 = xnn_max_f32(vout2x1, vmin);
    vout2x1 = xnn_min_f32(vout2x1, vmax);
    vout2x2 = xnn_max_f32(vout2x2, vmin);
    vout2x2 = xnn_min_f32(vout2x2, vmax);
    vout2x3 = xnn_max_f32(vout2x3, vmin);
    vout2x3 = xnn_min_f32(vout2x3, vmax);
    vout3x0 = xnn_max_f32(vout3x0, vmin);
    vout3x0 = xnn_min_f32(vout3x0, vmax);
    vout3x1 = xnn_max_f32(vout3x1, vmin);
    vout3x1 = xnn_min_f32(vout3x1, vmax);
    vout3x2 = xnn_max_f32(vout3x2, vmin);
    vout3x2 = xnn_min_f32(vout3x2, vmax);
    vout3x3 = xnn_max_f32(vout3x3, vmin);
    vout3x3 = xnn_min_f32(vout3x3, vmax);

    if XNN_LIKELY(nc >= 128) {
      *((HVX_UVector*)(c0 + 0)) = vout0x0;
      *((HVX_UVector*)(c0 + 32)) = vout0x1;
      *((HVX_UVector*)(c0 + 64)) = vout0x2;
      *((HVX_UVector*)(c0 + 96)) = vout0x3;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      a0 = (const int8_t*) ((uintptr_t) a0 - kc_aligned16);
      *((HVX_UVector*)(c1 + 0)) = vout1x0;
      *((HVX_UVector*)(c1 + 32)) = vout1x1;
      *((HVX_UVector*)(c1 + 64)) = vout1x2;
      *((HVX_UVector*)(c1 + 96)) = vout1x3;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc_aligned16);
      *((HVX_UVector*)(c2 + 0)) = vout2x0;
      *((HVX_UVector*)(c2 + 32)) = vout2x1;
      *((HVX_UVector*)(c2 + 64)) = vout2x2;
      *((HVX_UVector*)(c2 + 96)) = vout2x3;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc_aligned16);
      *((HVX_UVector*)(c3 + 0)) = vout3x0;
      *((HVX_UVector*)(c3 + 32)) = vout3x1;
      *((HVX_UVector*)(c3 + 64)) = vout3x2;
      *((HVX_UVector*)(c3 + 96)) = vout3x3;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc_aligned16);

      nc -= 128;
    } else {
      const size_t nc_tail = min(nc, 32);
      Q6_V_vstu_variable(c0, nc_tail * sizeof(float), vout0x0);
      Q6_V_vstu_variable(c1, nc_tail * sizeof(float), vout1x0);
      Q6_V_vstu_variable(c2, nc_tail * sizeof(float), vout2x0);
      Q6_V_vstu_variable(c3, nc_tail * sizeof(float), vout3x0);
      if (nc > 32) {
        const size_t nc_tail = min(nc - 32, 32);
        Q6_V_vstu_variable(c0 + 32, nc_tail * sizeof(float), vout0x1);
        Q6_V_vstu_variable(c1 + 32, nc_tail * sizeof(float), vout1x1);
        Q6_V_vstu_variable(c2 + 32, nc_tail * sizeof(float), vout2x1);
        Q6_V_vstu_variable(c3 + 32, nc_tail * sizeof(float), vout3x1);
      }
      if (nc > 64) {
        const size_t nc_tail = min(nc - 64, 32);
        Q6_V_vstu_variable(c0 + 64, nc_tail * sizeof(float), vout0x2);
        Q6_V_vstu_variable(c1 + 64, nc_tail * sizeof(float), vout1x2);
        Q6_V_vstu_variable(c2 + 64, nc_tail * sizeof(float), vout2x2);
        Q6_V_vstu_variable(c3 + 64, nc_tail * sizeof(float), vout3x2);
      }
      if (nc > 96) {
        const size_t nc_tail = min(nc - 96, 32);
        Q6_V_vstu_variable(c0 + 96, nc_tail * sizeof(float), vout0x3);
        Q6_V_vstu_variable(c1 + 96, nc_tail * sizeof(float), vout1x3);
        Q6_V_vstu_variable(c2 + 96, nc_tail * sizeof(float), vout2x3);
        Q6_V_vstu_variable(c3 + 96, nc_tail * sizeof(float), vout3x3);
      }
      nc = 0;
    }
  } while (nc != 0);
}
