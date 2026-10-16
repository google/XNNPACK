// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/neon-ld128.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/microparams.h"


void xnn_f32_qc4w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128_fmagic(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_qc4w_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const uint8x16_t vmask = vdupq_n_u8(UINT8_C(0xF));
  const uint8x16_t vmagic_bias = vreinterpretq_u8_u32(vdupq_n_u32(UINT32_C(0x4B000000) + (uint32_t) params->scalar.kernel_zero_point));
  const float32x4_t vmagic_bias_f32 = vreinterpretq_f32_u8(vmagic_bias);
  static const uint8_t ktbl_idx[64] = {
    0, 17, 18, 19,
    1, 17, 18, 19,
    2, 17, 18, 19,
    3, 17, 18, 19,
    4, 17, 18, 19,
    5, 17, 18, 19,
    6, 17, 18, 19,
    7, 17, 18, 19,
    8, 17, 18, 19,
    9, 17, 18, 19,
    10, 17, 18, 19,
    11, 17, 18, 19,
    12, 17, 18, 19,
    13, 17, 18, 19,
    14, 17, 18, 19,
    15, 17, 18, 19,
  };
  const uint8x16_t vtbl_idx0 = vld1q_u8(ktbl_idx + 0);
  const uint8x16_t vtbl_idx1 = vld1q_u8(ktbl_idx + 16);
  const uint8x16_t vtbl_idx2 = vld1q_u8(ktbl_idx + 32);
  const uint8x16_t vtbl_idx3 = vld1q_u8(ktbl_idx + 48);

  do {
    float32x4_t vacc0x0 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vacc0x1 = vld1q_f32(w); w = (const float*) w + 4;

    size_t k = kc;
    if XNN_LIKELY(k >= 4 * sizeof(float)) {
      do {
        const float32x4_t va0 = vld1q_f32(a0); a0 += 4;


        const uint8x16_t vw01234567c0123 = vld1q_u8(w); w = (const uint8_t*) w + 16;
        const uint8x16_t vw01234567c02 = vandq_u8(vw01234567c0123, vmask);
        const uint8x16_t vw01234567c13 = vshrq_n_u8(vw01234567c0123, 4);
        const uint8x16x2_t vtbl01234567c02 = { { vw01234567c02, vmagic_bias } };
        const uint8x16x2_t vtbl01234567c13 = { { vw01234567c13, vmagic_bias } };
        const float32x4_t vb0123c0 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl01234567c02, vtbl_idx0)), vmagic_bias_f32);
        const float32x4_t vb4567c0 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl01234567c02, vtbl_idx1)), vmagic_bias_f32);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c0, vget_low_f32(va0), 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c0, vget_low_f32(va0), 0);

        const float32x4_t vb0123c1 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl01234567c13, vtbl_idx0)), vmagic_bias_f32);
        const float32x4_t vb4567c1 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl01234567c13, vtbl_idx1)), vmagic_bias_f32);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c1, vget_low_f32(va0), 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c1, vget_low_f32(va0), 1);

        const float32x4_t vb0123c2 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl01234567c02, vtbl_idx2)), vmagic_bias_f32);
        const float32x4_t vb4567c2 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl01234567c02, vtbl_idx3)), vmagic_bias_f32);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c2, vget_high_f32(va0), 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c2, vget_high_f32(va0), 0);

        const float32x4_t vb0123c3 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl01234567c13, vtbl_idx2)), vmagic_bias_f32);
        const float32x4_t vb4567c3 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl01234567c13, vtbl_idx3)), vmagic_bias_f32);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c3, vget_high_f32(va0), 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c3, vget_high_f32(va0), 1);
        k -= 4 * sizeof(float);
      } while (k >= 4 * sizeof(float));
    }

    if XNN_UNLIKELY(k != 0) {
      if XNN_UNLIKELY(k & (2 * sizeof(float))) {
        const float32x2_t va0 = vld1_f32(a0); a0 += 2;


        const uint8x8_t vraw01234567c01 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const uint8x16_t vw01234567c01 = vcombine_u8(vand_u8(vraw01234567c01, vget_low_u8(vmask)), vshr_n_u8(vraw01234567c01, 4));
        const uint8x16x2_t vtbl01234567c01 = { { vw01234567c01, vmagic_bias } };
        const float32x4_t vb0123c0 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl01234567c01, vtbl_idx0)), vmagic_bias_f32);
        const float32x4_t vb4567c0 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl01234567c01, vtbl_idx1)), vmagic_bias_f32);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c0, va0, 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c0, va0, 0);

        const float32x4_t vb0123c1 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl01234567c01, vtbl_idx2)), vmagic_bias_f32);
        const float32x4_t vb4567c1 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl01234567c01, vtbl_idx3)), vmagic_bias_f32);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c1, va0, 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c1, va0, 1);
      }
      if XNN_UNLIKELY(k & (1 * sizeof(float))) {
        const float32x4_t va0 = vld1q_dup_f32(a0); a0 += 1;

        const uint8x16_t vw01234567 = vcombine_u8(vld1_u8(w), vdup_n_u8(0)); w = (const uint8_t*) w + 8;
        const uint8x16x2_t vtbl01234567 = { { vw01234567, vmagic_bias } };
        const float32x4_t vb0123 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl01234567, vtbl_idx0)), vmagic_bias_f32);
        const float32x4_t vb4567 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl01234567, vtbl_idx1)), vmagic_bias_f32);

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567);
      }
    }
    const float32x4_t vscale0123 = vld1q_f32(w); w = ((const float*) w + 4);
    const float32x4_t vscale4567 = vld1q_f32(w); w = ((const float*) w + 4);
    vacc0x0 = vmulq_f32(vacc0x0, vscale0123);
    vacc0x1 = vmulq_f32(vacc0x1, vscale4567);
    const float32x4_t vmax = vdupq_n_f32(params->scalar.max);
    vacc0x0 = vminq_f32(vacc0x0, vmax);
    vacc0x1 = vminq_f32(vacc0x1, vmax);

    const float32x4_t vmin = vdupq_n_f32(params->scalar.min);
    vacc0x0 = vmaxq_f32(vacc0x0, vmin);
    vacc0x1 = vmaxq_f32(vacc0x1, vmin);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c0, vacc0x0);
      vst1q_f32(c0 + 4, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;

    } else {
      if (nc & 4) {
        vst1q_f32(c0, vacc0x0); c0 += 4;

        vacc0x0 = vacc0x1;
      }
      float32x2_t vacc0 = vget_low_f32(vacc0x0);
      if (nc & 2) {
        vst1_f32(c0, vacc0); c0 += 2;

        vacc0 = vget_high_f32(vacc0x0);
      }
      if (nc & 1) {
        vst1_lane_f32(c0, vacc0, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
