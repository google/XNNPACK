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



void xnn_qd8_f16_qc2w_gemm_minmax_ukernel_1x128c4__hvx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    xnn_float16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f16_minmax_params* restrict params,
    const float* row_sum,
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
  assert(row_sum != NULL);

  const size_t kc_aligned16 = kc & ~(size_t) 15;
  const int8_t* a0 = a;
  uint16_t* c0 = (uint16_t*) c;


  const uint32_t rmin = float_as_uint32(xnn_float16_to_float(params->scalar.min));
  const uint32_t rmax = float_as_uint32(xnn_float16_to_float(params->scalar.max));
  const HVX_Vector vmask03 = Q6_Vb_vsplat_R(0x03);
  const HVX_Vector vmask0c = Q6_Vb_vsplat_R(0x0C);
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
    HVX_Vector vacc1x0x0 = Q6_V_vzero();
    HVX_Vector vacc1x0x1 = Q6_V_vzero();
    HVX_Vector vacc1x0x2 = Q6_V_vzero();
    HVX_Vector vacc1x0x3 = Q6_V_vzero();

    size_t k = kc;
    // Use unaligned loads if activation pointer is not 4-byte aligned.
    if ((((uintptr_t) a) & (sizeof(int32_t) - 1)) != 0) {
      for (; k >= 16 * sizeof(int8_t); k -= 16 * sizeof(int8_t)) {
        const int32_t ra0x0 = unaligned_load_s32(a0); a0 += 4;
        const int32_t ra0x1 = unaligned_load_s32(a0); a0 += 4;
        const int32_t ra0x2 = unaligned_load_s32(a0); a0 += 4;
        const int32_t ra0x3 = unaligned_load_s32(a0); a0 += 4;

        const HVX_Vector vbb0 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
        const HVX_Vector vbb1 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
        const HVX_Vector vbb2 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
        const HVX_Vector vbb3 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;

        const HVX_Vector vb0x0 = Q6_V_vand_VV(vbb0, vmask03);
        vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x0, ra0x0);
        const HVX_Vector vb1x0 = Q6_V_vand_VV(vbb1, vmask03);
        vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x0, ra0x0);
        const HVX_Vector vb2x0 = Q6_V_vand_VV(vbb2, vmask03);
        vacc0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x2, vb2x0, ra0x0);
        const HVX_Vector vb3x0 = Q6_V_vand_VV(vbb3, vmask03);
        vacc0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x3, vb3x0, ra0x0);
        const HVX_Vector vb0x1 = Q6_V_vand_VV(vbb0, vmask0c);
        vacc1x0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x0, vb0x1, ra0x1);
        const HVX_Vector vb1x1 = Q6_V_vand_VV(vbb1, vmask0c);
        vacc1x0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x1, vb1x1, ra0x1);
        const HVX_Vector vb2x1 = Q6_V_vand_VV(vbb2, vmask0c);
        vacc1x0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x2, vb2x1, ra0x1);
        const HVX_Vector vb3x1 = Q6_V_vand_VV(vbb3, vmask0c);
        vacc1x0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x3, vb3x1, ra0x1);
        const HVX_Vector vbb_hi0 = Q6_Vuw_vlsr_VuwR(vbb0, 4);
        const HVX_Vector vbb_hi1 = Q6_Vuw_vlsr_VuwR(vbb1, 4);
        const HVX_Vector vbb_hi2 = Q6_Vuw_vlsr_VuwR(vbb2, 4);
        const HVX_Vector vbb_hi3 = Q6_Vuw_vlsr_VuwR(vbb3, 4);
        const HVX_Vector vb0x2 = Q6_V_vand_VV(vbb_hi0, vmask03);
        vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x2, ra0x2);
        const HVX_Vector vb1x2 = Q6_V_vand_VV(vbb_hi1, vmask03);
        vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x2, ra0x2);
        const HVX_Vector vb2x2 = Q6_V_vand_VV(vbb_hi2, vmask03);
        vacc0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x2, vb2x2, ra0x2);
        const HVX_Vector vb3x2 = Q6_V_vand_VV(vbb_hi3, vmask03);
        vacc0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x3, vb3x2, ra0x2);
        const HVX_Vector vb0x3 = Q6_V_vand_VV(vbb_hi0, vmask0c);
        vacc1x0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x0, vb0x3, ra0x3);
        const HVX_Vector vb1x3 = Q6_V_vand_VV(vbb_hi1, vmask0c);
        vacc1x0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x1, vb1x3, ra0x3);
        const HVX_Vector vb2x3 = Q6_V_vand_VV(vbb_hi2, vmask0c);
        vacc1x0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x2, vb2x3, ra0x3);
        const HVX_Vector vb3x3 = Q6_V_vand_VV(vbb_hi3, vmask0c);
        vacc1x0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x3, vb3x3, ra0x3);
      }
    } else {
      for (; k >= 16 * sizeof(int8_t); k -= 16 * sizeof(int8_t)) {
        const int32_t ra0x0 = *((const int32_t*) a0); a0 += 4;
        const int32_t ra0x1 = *((const int32_t*) a0); a0 += 4;
        const int32_t ra0x2 = *((const int32_t*) a0); a0 += 4;
        const int32_t ra0x3 = *((const int32_t*) a0); a0 += 4;

        const HVX_Vector vbb0 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
        const HVX_Vector vbb1 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
        const HVX_Vector vbb2 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;
        const HVX_Vector vbb3 = *((const HVX_Vector*) w); w = (const int8_t*) w + 128;

        const HVX_Vector vb0x0 = Q6_V_vand_VV(vbb0, vmask03);
        vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x0, ra0x0);
        const HVX_Vector vb1x0 = Q6_V_vand_VV(vbb1, vmask03);
        vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x0, ra0x0);
        const HVX_Vector vb2x0 = Q6_V_vand_VV(vbb2, vmask03);
        vacc0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x2, vb2x0, ra0x0);
        const HVX_Vector vb3x0 = Q6_V_vand_VV(vbb3, vmask03);
        vacc0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x3, vb3x0, ra0x0);
        const HVX_Vector vb0x1 = Q6_V_vand_VV(vbb0, vmask0c);
        vacc1x0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x0, vb0x1, ra0x1);
        const HVX_Vector vb1x1 = Q6_V_vand_VV(vbb1, vmask0c);
        vacc1x0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x1, vb1x1, ra0x1);
        const HVX_Vector vb2x1 = Q6_V_vand_VV(vbb2, vmask0c);
        vacc1x0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x2, vb2x1, ra0x1);
        const HVX_Vector vb3x1 = Q6_V_vand_VV(vbb3, vmask0c);
        vacc1x0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x3, vb3x1, ra0x1);
        const HVX_Vector vbb_hi0 = Q6_Vuw_vlsr_VuwR(vbb0, 4);
        const HVX_Vector vbb_hi1 = Q6_Vuw_vlsr_VuwR(vbb1, 4);
        const HVX_Vector vbb_hi2 = Q6_Vuw_vlsr_VuwR(vbb2, 4);
        const HVX_Vector vbb_hi3 = Q6_Vuw_vlsr_VuwR(vbb3, 4);
        const HVX_Vector vb0x2 = Q6_V_vand_VV(vbb_hi0, vmask03);
        vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x2, ra0x2);
        const HVX_Vector vb1x2 = Q6_V_vand_VV(vbb_hi1, vmask03);
        vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x2, ra0x2);
        const HVX_Vector vb2x2 = Q6_V_vand_VV(vbb_hi2, vmask03);
        vacc0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x2, vb2x2, ra0x2);
        const HVX_Vector vb3x2 = Q6_V_vand_VV(vbb_hi3, vmask03);
        vacc0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x3, vb3x2, ra0x2);
        const HVX_Vector vb0x3 = Q6_V_vand_VV(vbb_hi0, vmask0c);
        vacc1x0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x0, vb0x3, ra0x3);
        const HVX_Vector vb1x3 = Q6_V_vand_VV(vbb_hi1, vmask0c);
        vacc1x0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x1, vb1x3, ra0x3);
        const HVX_Vector vb2x3 = Q6_V_vand_VV(vbb_hi2, vmask0c);
        vacc1x0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x2, vb2x3, ra0x3);
        const HVX_Vector vb3x3 = Q6_V_vand_VV(vbb_hi3, vmask0c);
        vacc1x0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x3, vb3x3, ra0x3);
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

        if (k > 4 * sizeof(int8_t)) {
          const HVX_Vector vb0x1 = Q6_V_vand_VV(vbb0, vmask0c);
          const HVX_Vector vb1x1 = Q6_V_vand_VV(vbb1, vmask0c);
          const HVX_Vector vb2x1 = Q6_V_vand_VV(vbb2, vmask0c);
          const HVX_Vector vb3x1 = Q6_V_vand_VV(vbb3, vmask0c);
          if (k >= 8 * sizeof(int8_t)) {
            const int32_t ra0x1 = unaligned_load_s32(a0 + 4);
            vacc1x0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x0, vb0x1, ra0x1);
            vacc1x0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x1, vb1x1, ra0x1);
            vacc1x0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x2, vb2x1, ra0x1);
            vacc1x0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x3, vb3x1, ra0x1);

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

                if (k > 12 * sizeof(int8_t)) {
                  const uint32_t tail_mask = UINT32_MAX >> ((16 - k) * 8);
                  const HVX_Vector vb0x3 = Q6_V_vand_VV(vbb_hi0, vmask0c);
                  const HVX_Vector vb1x3 = Q6_V_vand_VV(vbb_hi1, vmask0c);
                  const HVX_Vector vb2x3 = Q6_V_vand_VV(vbb_hi2, vmask0c);
                  const HVX_Vector vb3x3 = Q6_V_vand_VV(vbb_hi3, vmask0c);
                  const int32_t ra0x3 = (int32_t) (unaligned_load_u32(a0 + 12) & tail_mask);
                  vacc1x0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x0, vb0x3, ra0x3);
                  vacc1x0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x1, vb1x3, ra0x3);
                  vacc1x0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x2, vb2x3, ra0x3);
                  vacc1x0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x3, vb3x3, ra0x3);
                }
              } else {
                const uint32_t tail_mask = UINT32_MAX >> ((12 - k) * 8);
                const int32_t ra0x2 = (int32_t) (unaligned_load_u32(a0 + 8) & tail_mask);
                vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x2, ra0x2);
                vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x2, ra0x2);
                vacc0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x2, vb2x2, ra0x2);
                vacc0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x3, vb3x2, ra0x2);
              }
            }
          } else {
            const uint32_t tail_mask = UINT32_MAX >> ((8 - k) * 8);
            const int32_t ra0x1 = (int32_t) (unaligned_load_u32(a0 + 4) & tail_mask);
            vacc1x0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x0, vb0x1, ra0x1);
            vacc1x0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x1, vb1x1, ra0x1);
            vacc1x0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x2, vb2x1, ra0x1);
            vacc1x0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc1x0x3, vb3x1, ra0x1);
          }
        }
      } else {
        const uint32_t tail_mask = UINT32_MAX >> ((4 - k) * 8);
        const int32_t ra0x0 = (int32_t) (unaligned_load_u32(a0) & tail_mask);
        vacc0x0 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x0, vb0x0, ra0x0);
        vacc0x1 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x1, vb1x0, ra0x0);
        vacc0x2 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x2, vb2x0, ra0x0);
        vacc0x3 = Q6_Vw_vrmpyacc_VwVubRb(vacc0x3, vb3x0, ra0x0);
      }
    }
    vacc0x0 = Q6_Vw_vasracc_VwVwR(vacc0x0, vacc1x0x0, 2);
    vacc0x1 = Q6_Vw_vasracc_VwVwR(vacc0x1, vacc1x0x1, 2);
    vacc0x2 = Q6_Vw_vasracc_VwVwR(vacc0x2, vacc1x0x2, 2);
    vacc0x3 = Q6_Vw_vasracc_VwVwR(vacc0x3, vacc1x0x3, 2);


    const HVX_Vector vtwo_row_sum_0 = Q6_V_vsplat_R(float_as_uint32(2.0f * row_sum[0]));
    HVX_Vector vout0x0 = xnn_sub_f32(xnn_cvt_f32_s32(vacc0x0), vtwo_row_sum_0);
    HVX_Vector vout0x1 = xnn_sub_f32(xnn_cvt_f32_s32(vacc0x1), vtwo_row_sum_0);
    HVX_Vector vout0x2 = xnn_sub_f32(xnn_cvt_f32_s32(vacc0x2), vtwo_row_sum_0);
    HVX_Vector vout0x3 = xnn_sub_f32(xnn_cvt_f32_s32(vacc0x3), vtwo_row_sum_0);

    const HVX_Vector vkzp0 = kzp[0];
    const HVX_Vector vkzp1 = kzp[1];
    const HVX_Vector vkzp2 = kzp[2];
    const HVX_Vector vkzp3 = kzp[3];
    const HVX_Vector vkc_izp_less_row_sum_0 = Q6_V_vsplat_R(float_as_uint32((float) kc * (float) quantization_params[0].zero_point - row_sum[0]));
    vout0x0 = xnn_fmadd_f32(vkzp0, vkc_izp_less_row_sum_0, vout0x0);
    vout0x1 = xnn_fmadd_f32(vkzp1, vkc_izp_less_row_sum_0, vout0x1);
    vout0x2 = xnn_fmadd_f32(vkzp2, vkc_izp_less_row_sum_0, vout0x2);
    vout0x3 = xnn_fmadd_f32(vkzp3, vkc_izp_less_row_sum_0, vout0x3);

    const HVX_Vector vinput_scale0 = Q6_V_vsplat_R(float_as_uint32(quantization_params[0].inv_scale));
    vout0x0 = xnn_mul_f32(vout0x0, vinput_scale0);
    vout0x1 = xnn_mul_f32(vout0x1, vinput_scale0);
    vout0x2 = xnn_mul_f32(vout0x2, vinput_scale0);
    vout0x3 = xnn_mul_f32(vout0x3, vinput_scale0);

    const HVX_Vector vfilter_output_scale0 = *((const HVX_Vector*) w); w = (const float*) w + 32;
    const HVX_Vector vfilter_output_scale1 = *((const HVX_Vector*) w); w = (const float*) w + 32;
    const HVX_Vector vfilter_output_scale2 = *((const HVX_Vector*) w); w = (const float*) w + 32;
    const HVX_Vector vfilter_output_scale3 = *((const HVX_Vector*) w); w = (const float*) w + 32;
    const HVX_Vector vbias0 = *((const HVX_Vector*) w); w = (const float*) w + 32;
    vout0x0 = xnn_fmadd_f32(vout0x0, vfilter_output_scale0, vbias0);
    const HVX_Vector vbias1 = *((const HVX_Vector*) w); w = (const float*) w + 32;
    vout0x1 = xnn_fmadd_f32(vout0x1, vfilter_output_scale1, vbias1);
    const HVX_Vector vbias2 = *((const HVX_Vector*) w); w = (const float*) w + 32;
    vout0x2 = xnn_fmadd_f32(vout0x2, vfilter_output_scale2, vbias2);
    const HVX_Vector vbias3 = *((const HVX_Vector*) w); w = (const float*) w + 32;
    vout0x3 = xnn_fmadd_f32(vout0x3, vfilter_output_scale3, vbias3);
    const HVX_Vector vmin = Q6_V_vsplat_R(rmin);
    const HVX_Vector vmax = Q6_V_vsplat_R(rmax);
    vout0x0 = xnn_max_f32(vout0x0, vmin);
    vout0x0 = xnn_min_f32(vout0x0, vmax);
    vout0x1 = xnn_max_f32(vout0x1, vmin);
    vout0x1 = xnn_min_f32(vout0x1, vmax);
    vout0x2 = xnn_max_f32(vout0x2, vmin);
    vout0x2 = xnn_min_f32(vout0x2, vmax);
    vout0x3 = xnn_max_f32(vout0x3, vmin);
    vout0x3 = xnn_min_f32(vout0x3, vmax);

    const HVX_Vector vfp16out0x0 = xnn_cvt_f16_f32_f32(vout0x0, vout0x1);
    const HVX_Vector vfp16out0x1 = xnn_cvt_f16_f32_f32(vout0x2, vout0x3);

    if XNN_LIKELY(nc >= 128) {
      *((HVX_UVector*) (c0 + 0)) = vfp16out0x0;
      *((HVX_UVector*) (c0 + 64)) = vfp16out0x1;
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      a0 = (const int8_t*) ((uintptr_t) a0 - kc_aligned16);

      nc -= 128;
    } else {
      const size_t nc_tail = min(nc, 64);
      Q6_V_vstu_variable(c0, nc_tail * sizeof(uint16_t), vfp16out0x0);
      if (nc > 64) {
        const size_t nc_tail = min(nc - 64, 64);
        Q6_V_vstu_variable(c0 + 64, nc_tail * sizeof(uint16_t), vfp16out0x1);
      }
      nc = 0;
    }
  } while (nc != 0);
}
