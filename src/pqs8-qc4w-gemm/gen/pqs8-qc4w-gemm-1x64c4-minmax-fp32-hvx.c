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
#include <math.h>  // for lrintf

#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"  // for Q6_V_vstu_variable
#include "src/xnnpack/math.h"
#include "src/xnnpack/simd/f32-hvx.h"
#include "src/xnnpack/unaligned.h"


#if __HVX_ARCH__ < 73
static HVX_Vector rescale_fp32(HVX_Vector vacc, HVX_Vector vscale)
{
  XNN_ALIGN(128) int32_t vacc_buffer[32];
  XNN_ALIGN(128) float vscale_buffer[32];

  *((HVX_Vector *)&vacc_buffer) = vacc;
  *((HVX_Vector *)&vscale_buffer) = vscale;

  for (int i = 0; i < 32; ++i) {
    vacc_buffer[i] = (int32_t)lrintf((float)vacc_buffer[i] * vscale_buffer[i]);
  }
  return *(HVX_Vector *)&vacc_buffer;
}
#endif  // __HVX_ARCH__ < 73

size_t xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_1x64c4__hvx_get_mr(void)
{
  return 1;
}

size_t xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_1x64c4__hvx_get_nr(void)
{
  return 64;
}

void xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_1x64c4__hvx(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* lhs_packed,
    const void* w,
    void* dst,
    size_t dst_stride_row,
    size_t dst_stride_col,
    const union xnn_qs8_qc8w_conv_minmax_params* params)
{
  assert(mr != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(lhs_packed != NULL);
  assert(w != NULL);
  assert(dst != NULL);
  (void) dst_stride_col;

  #if __HVX_ARCH__ >= 81
  const HVX_Vector vmagic_bias = xnn_set1_f32(12582912.0f);
  const HVX_Vector vmagic_bias_less_zero_point = Q6_V_vsplat_R(INT32_C(0x4B400000) - (int32_t) params->fp32_scalar.output_zero_point);
  #elif __HVX_ARCH__ >= 73
  const HVX_Vector vmagic_bias = xnn_set1_f32(6291456.5f);
  const HVX_Vector vmagic_bias_less_zero_point = Q6_V_vsplat_R(INT32_C(0x25600000) - (int32_t) params->fp32_scalar.output_zero_point);
  #else
  const HVX_Vector voutput_zero_point = Q6_Vh_vsplat_R(params->fp32_scalar.output_zero_point);
  #endif
  const HVX_Vector voutput_min = Q6_Vb_vsplat_R(params->fp32_scalar.output_min);
  const HVX_Vector voutput_max = Q6_Vb_vsplat_R(params->fp32_scalar.output_max);
  const HVX_Vector vmask = Q6_Vb_vsplat_R(0xF0);

  int8_t* c = (int8_t*) dst;
  do {
    const HVX_Vector vinit0 = *((const HVX_Vector*) w);
    const HVX_Vector vinit1 = *((const HVX_Vector*) ((const int32_t*) w + 32));
    const void* w_panel = (const int32_t*) w + 64;
    const size_t n_store = min(nc, (size_t) 64);

    for (size_t m_idx = 0; m_idx < mr; ++m_idx) {
      const int8_t* a0 = (const int8_t*) lhs_packed + m_idx * kc;
      const int8_t* w_ptr = (const int8_t*) w_panel;
      HVX_Vector vacc0 = vinit0;
      HVX_Vector vacc1 = vinit1;

      size_t k = kc;
      if XNN_LIKELY(((uintptr_t) a0 & (sizeof(uint32_t) - 1)) == 0) {
        for (; k >= 8 * sizeof(int8_t); k -= 8 * sizeof(int8_t)) {
          const HVX_Vector va0x0123 = Q6_V_vsplat_R(*((const uint32_t*) a0) ^ 0x80808080u);
          const HVX_Vector va0x4567 = Q6_V_vsplat_R(*((const uint32_t*) (a0 + 4)) ^ 0x80808080u);
          a0 += 8;

          const HVX_Vector vb0x01234567 = *((const HVX_Vector*) w_ptr);
          const HVX_Vector vb1x01234567 = *((const HVX_Vector*) (w_ptr + 128));
          w_ptr += 256;
          const HVX_Vector vb0x0123 = Q6_V_vand_VV(Q6_Vw_vasl_VwR(vb0x01234567, 4), vmask);
          const HVX_Vector vb1x0123 = Q6_V_vand_VV(Q6_Vw_vasl_VwR(vb1x01234567, 4), vmask);
          const HVX_Vector vb0x4567 = Q6_V_vand_VV(vb0x01234567, vmask);
          const HVX_Vector vb1x4567 = Q6_V_vand_VV(vb1x01234567, vmask);

          vacc0 = Q6_Vw_vrmpyacc_VwVubVb(vacc0, va0x0123, vb0x0123);
          vacc1 = Q6_Vw_vrmpyacc_VwVubVb(vacc1, va0x0123, vb1x0123);
          vacc0 = Q6_Vw_vrmpyacc_VwVubVb(vacc0, va0x4567, vb0x4567);
          vacc1 = Q6_Vw_vrmpyacc_VwVubVb(vacc1, va0x4567, vb1x4567);
        }
      } else {
        for (; k >= 8 * sizeof(int8_t); k -= 8 * sizeof(int8_t)) {
          const HVX_Vector va0x0123 = Q6_V_vsplat_R((uint32_t) unaligned_load_s32(a0) ^ 0x80808080u);
          const HVX_Vector va0x4567 = Q6_V_vsplat_R((uint32_t) unaligned_load_s32(a0 + 4) ^ 0x80808080u);
          a0 += 8;

          const HVX_Vector vb0x01234567 = *((const HVX_Vector*) w_ptr);
          const HVX_Vector vb1x01234567 = *((const HVX_Vector*) (w_ptr + 128));
          w_ptr += 256;
          const HVX_Vector vb0x0123 = Q6_V_vand_VV(Q6_Vw_vasl_VwR(vb0x01234567, 4), vmask);
          const HVX_Vector vb1x0123 = Q6_V_vand_VV(Q6_Vw_vasl_VwR(vb1x01234567, 4), vmask);
          const HVX_Vector vb0x4567 = Q6_V_vand_VV(vb0x01234567, vmask);
          const HVX_Vector vb1x4567 = Q6_V_vand_VV(vb1x01234567, vmask);

          vacc0 = Q6_Vw_vrmpyacc_VwVubVb(vacc0, va0x0123, vb0x0123);
          vacc1 = Q6_Vw_vrmpyacc_VwVubVb(vacc1, va0x0123, vb1x0123);
          vacc0 = Q6_Vw_vrmpyacc_VwVubVb(vacc0, va0x4567, vb0x4567);
          vacc1 = Q6_Vw_vrmpyacc_VwVubVb(vacc1, va0x4567, vb1x4567);
        }
      }
      if (k != 0) {
        const HVX_Vector vb0x01234567 = *((const HVX_Vector*) w_ptr);
        const HVX_Vector vb1x01234567 = *((const HVX_Vector*) (w_ptr + 128));
        w_ptr += 256;
        const HVX_Vector vb0x0123 = Q6_V_vand_VV(Q6_Vw_vasl_VwR(vb0x01234567, 4), vmask);
        const HVX_Vector vb1x0123 = Q6_V_vand_VV(Q6_Vw_vasl_VwR(vb1x01234567, 4), vmask);
        if (k <= 4) {
          uint32_t u0123 = 0;
          if (k == 4) {
            u0123 = (uint32_t) unaligned_load_s32(a0) ^ 0x80808080u;
          } else {
            for (size_t i = 0; i < k; ++i) {
              u0123 |= ((uint32_t) ((uint8_t) a0[i] ^ 0x80u)) << (8 * i);
            }
          }
          const HVX_Vector va0x0123 = Q6_V_vsplat_R(u0123);
          vacc0 = Q6_Vw_vrmpyacc_VwVubVb(vacc0, va0x0123, vb0x0123);
          vacc1 = Q6_Vw_vrmpyacc_VwVubVb(vacc1, va0x0123, vb1x0123);
        } else {
          const HVX_Vector va0x0123 = Q6_V_vsplat_R((uint32_t) unaligned_load_s32(a0) ^ 0x80808080u);
          uint32_t u4567 = 0;
          for (size_t i = 4; i < k; ++i) {
            u4567 |= ((uint32_t) ((uint8_t) a0[i] ^ 0x80u)) << (8 * (i - 4));
          }
          const HVX_Vector va0x4567 = Q6_V_vsplat_R(u4567);
          const HVX_Vector vb0x4567 = Q6_V_vand_VV(vb0x01234567, vmask);
          const HVX_Vector vb1x4567 = Q6_V_vand_VV(vb1x01234567, vmask);
          vacc0 = Q6_Vw_vrmpyacc_VwVubVb(vacc0, va0x0123, vb0x0123);
          vacc1 = Q6_Vw_vrmpyacc_VwVubVb(vacc1, va0x0123, vb1x0123);
          vacc0 = Q6_Vw_vrmpyacc_VwVubVb(vacc0, va0x4567, vb0x4567);
          vacc1 = Q6_Vw_vrmpyacc_VwVubVb(vacc1, va0x4567, vb1x4567);
        }
      }
      vacc0 = Q6_Vw_vasr_VwR(vacc0, 4);
      vacc1 = Q6_Vw_vasr_VwR(vacc1, 4);

      const HVX_Vector vscale0 = *((const HVX_Vector*) w_ptr);
      const HVX_Vector vscale1 = *((const HVX_Vector*) ((const float*) w_ptr + 32));
      #if __HVX_ARCH__ >= 81
      vacc0 = Q6_Vsf_equals_Vw(vacc0);
      vacc1 = Q6_Vsf_equals_Vw(vacc1);
      vacc0 = xnn_fmadd_f32(vacc0, vscale0, vmagic_bias);
      vacc1 = xnn_fmadd_f32(vacc1, vscale1, vmagic_bias);
      vacc0 = Q6_Vw_vsub_VwVw_sat(vacc0, vmagic_bias_less_zero_point);
      vacc1 = Q6_Vw_vsub_VwVw_sat(vacc1, vmagic_bias_less_zero_point);
      HVX_Vector vout_h = Q6_Vh_vpack_VwVw_sat(vacc1, vacc0);
      #elif __HVX_ARCH__ >= 73
      vacc0 = Q6_Vsf_equals_Vw(vacc0);
      vacc1 = Q6_Vsf_equals_Vw(vacc1);
      vacc0 = xnn_fmadd_f32(vacc0, vscale0, vmagic_bias);
      vacc1 = xnn_fmadd_f32(vacc1, vscale1, vmagic_bias);
      vacc0 = Q6_Vw_vsub_VwVw_sat(Q6_Vw_vasr_VwR(vacc0, 1), vmagic_bias_less_zero_point);
      vacc1 = Q6_Vw_vsub_VwVw_sat(Q6_Vw_vasr_VwR(vacc1, 1), vmagic_bias_less_zero_point);
      HVX_Vector vout_h = Q6_Vh_vpack_VwVw_sat(vacc1, vacc0);
      #else
      vacc0 = rescale_fp32(vacc0, vscale0);
      vacc1 = rescale_fp32(vacc1, vscale1);
      HVX_Vector vout_h = Q6_Vh_vpack_VwVw_sat(vacc1, vacc0);
      vout_h = Q6_Vh_vadd_VhVh_sat(vout_h, voutput_zero_point);
      #endif

      HVX_Vector vout = Q6_Vb_vpack_VhVh_sat(vout_h, vout_h);
      vout = Q6_Vb_vmax_VbVb(vout, voutput_min);
      vout = Q6_Vb_vmin_VbVb(vout, voutput_max);
      int8_t* c_row = (int8_t*) ((uintptr_t) c + m_idx * dst_stride_row);
      Q6_V_vstu_variable(c_row, n_store, vout);
    }

    w = (const void*) ((const uintptr_t) w_panel +
                       (round_up_po2(kc, 8) >> 3) * 256 +
                       64 * sizeof(float));
    c += 64;
    nc = (nc > 64) ? (nc - 64) : 0;
  } while (nc != 0);
}
