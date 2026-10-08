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


void xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128_fmagic(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* restrict params)
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
  // int8 weights are converted to f32 with a table lookup: each weight byte
  // is XORed with 0x80 (biased to unsigned), then TBL assembles the 32-bit
  // pattern 0x4B0000xx = (float) (2^23 + 128 + weight), using bytes from the
  // magic bias register (0x80, 0x00, 0x00, 0x4B) as the second table.
  // Subtracting the magic bias (2^23 + 128) yields the exact weight value.
  const uint8x16_t vsign_mask = vdupq_n_u8(UINT8_C(0x80));
  const uint8x16_t vmagic_bias = vreinterpretq_u8_u32(vdupq_n_u32(UINT32_C(0x4B000080)));
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
    float32x4_t vacc0x2 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vacc0x3 = vld1q_f32(w); w = (const float*) w + 4;

    size_t k = kc;
    if XNN_LIKELY(k >= 4 * sizeof(float)) {
      do {
        const float32x4_t va0 = vld1q_f32(a0); a0 += 4;


        const uint8x16_t vw0123456789ABCDEFc0 = veorq_u8(vld1q_u8(w), vsign_mask); w = (const uint8_t*) w + 16;
        const uint8x16x2_t vtbl0123456789ABCDEFc0 = { { vw0123456789ABCDEFc0, vmagic_bias } };
        const float32x4_t vb0123c0 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEFc0, vtbl_idx0)), vmagic_bias_f32);
        const float32x4_t vb4567c0 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEFc0, vtbl_idx1)), vmagic_bias_f32);
        const float32x4_t vb89ABc0 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEFc0, vtbl_idx2)), vmagic_bias_f32);
        const float32x4_t vbCDEFc0 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEFc0, vtbl_idx3)), vmagic_bias_f32);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c0, vget_low_f32(va0), 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c0, vget_low_f32(va0), 0);
        vacc0x2 = vfmaq_lane_f32(vacc0x2, vb89ABc0, vget_low_f32(va0), 0);
        vacc0x3 = vfmaq_lane_f32(vacc0x3, vbCDEFc0, vget_low_f32(va0), 0);

        const uint8x16_t vw0123456789ABCDEFc1 = veorq_u8(vld1q_u8(w), vsign_mask); w = (const uint8_t*) w + 16;
        const uint8x16x2_t vtbl0123456789ABCDEFc1 = { { vw0123456789ABCDEFc1, vmagic_bias } };
        const float32x4_t vb0123c1 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEFc1, vtbl_idx0)), vmagic_bias_f32);
        const float32x4_t vb4567c1 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEFc1, vtbl_idx1)), vmagic_bias_f32);
        const float32x4_t vb89ABc1 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEFc1, vtbl_idx2)), vmagic_bias_f32);
        const float32x4_t vbCDEFc1 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEFc1, vtbl_idx3)), vmagic_bias_f32);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c1, vget_low_f32(va0), 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c1, vget_low_f32(va0), 1);
        vacc0x2 = vfmaq_lane_f32(vacc0x2, vb89ABc1, vget_low_f32(va0), 1);
        vacc0x3 = vfmaq_lane_f32(vacc0x3, vbCDEFc1, vget_low_f32(va0), 1);

        const uint8x16_t vw0123456789ABCDEFc2 = veorq_u8(vld1q_u8(w), vsign_mask); w = (const uint8_t*) w + 16;
        const uint8x16x2_t vtbl0123456789ABCDEFc2 = { { vw0123456789ABCDEFc2, vmagic_bias } };
        const float32x4_t vb0123c2 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEFc2, vtbl_idx0)), vmagic_bias_f32);
        const float32x4_t vb4567c2 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEFc2, vtbl_idx1)), vmagic_bias_f32);
        const float32x4_t vb89ABc2 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEFc2, vtbl_idx2)), vmagic_bias_f32);
        const float32x4_t vbCDEFc2 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEFc2, vtbl_idx3)), vmagic_bias_f32);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c2, vget_high_f32(va0), 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c2, vget_high_f32(va0), 0);
        vacc0x2 = vfmaq_lane_f32(vacc0x2, vb89ABc2, vget_high_f32(va0), 0);
        vacc0x3 = vfmaq_lane_f32(vacc0x3, vbCDEFc2, vget_high_f32(va0), 0);

        const uint8x16_t vw0123456789ABCDEFc3 = veorq_u8(vld1q_u8(w), vsign_mask); w = (const uint8_t*) w + 16;
        const uint8x16x2_t vtbl0123456789ABCDEFc3 = { { vw0123456789ABCDEFc3, vmagic_bias } };
        const float32x4_t vb0123c3 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEFc3, vtbl_idx0)), vmagic_bias_f32);
        const float32x4_t vb4567c3 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEFc3, vtbl_idx1)), vmagic_bias_f32);
        const float32x4_t vb89ABc3 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEFc3, vtbl_idx2)), vmagic_bias_f32);
        const float32x4_t vbCDEFc3 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEFc3, vtbl_idx3)), vmagic_bias_f32);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c3, vget_high_f32(va0), 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c3, vget_high_f32(va0), 1);
        vacc0x2 = vfmaq_lane_f32(vacc0x2, vb89ABc3, vget_high_f32(va0), 1);
        vacc0x3 = vfmaq_lane_f32(vacc0x3, vbCDEFc3, vget_high_f32(va0), 1);
        k -= 4 * sizeof(float);
      } while (k >= 4 * sizeof(float));
    }

    if XNN_UNLIKELY(k != 0) {
      if XNN_UNLIKELY(k & (2 * sizeof(float))) {
        const float32x2_t va0 = vld1_f32(a0); a0 += 2;


        const uint8x16_t vw0123456789ABCDEFc0 = veorq_u8(vld1q_u8(w), vsign_mask); w = (const uint8_t*) w + 16;
        const uint8x16x2_t vtbl0123456789ABCDEFc0 = { { vw0123456789ABCDEFc0, vmagic_bias } };
        const float32x4_t vb0123c0 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEFc0, vtbl_idx0)), vmagic_bias_f32);
        const float32x4_t vb4567c0 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEFc0, vtbl_idx1)), vmagic_bias_f32);
        const float32x4_t vb89ABc0 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEFc0, vtbl_idx2)), vmagic_bias_f32);
        const float32x4_t vbCDEFc0 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEFc0, vtbl_idx3)), vmagic_bias_f32);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c0, va0, 0);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c0, va0, 0);
        vacc0x2 = vfmaq_lane_f32(vacc0x2, vb89ABc0, va0, 0);
        vacc0x3 = vfmaq_lane_f32(vacc0x3, vbCDEFc0, va0, 0);

        const uint8x16_t vw0123456789ABCDEFc1 = veorq_u8(vld1q_u8(w), vsign_mask); w = (const uint8_t*) w + 16;
        const uint8x16x2_t vtbl0123456789ABCDEFc1 = { { vw0123456789ABCDEFc1, vmagic_bias } };
        const float32x4_t vb0123c1 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEFc1, vtbl_idx0)), vmagic_bias_f32);
        const float32x4_t vb4567c1 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEFc1, vtbl_idx1)), vmagic_bias_f32);
        const float32x4_t vb89ABc1 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEFc1, vtbl_idx2)), vmagic_bias_f32);
        const float32x4_t vbCDEFc1 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEFc1, vtbl_idx3)), vmagic_bias_f32);

        vacc0x0 = vfmaq_lane_f32(vacc0x0, vb0123c1, va0, 1);
        vacc0x1 = vfmaq_lane_f32(vacc0x1, vb4567c1, va0, 1);
        vacc0x2 = vfmaq_lane_f32(vacc0x2, vb89ABc1, va0, 1);
        vacc0x3 = vfmaq_lane_f32(vacc0x3, vbCDEFc1, va0, 1);
      }
      if XNN_UNLIKELY(k & (1 * sizeof(float))) {
        const float32x4_t va0 = vld1q_dup_f32(a0); a0 += 1;

        const uint8x16_t vw0123456789ABCDEF = veorq_u8(vld1q_u8(w), vsign_mask); w = (const uint8_t*) w + 16;
        const uint8x16x2_t vtbl0123456789ABCDEF = { { vw0123456789ABCDEF, vmagic_bias } };
        const float32x4_t vb0123 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEF, vtbl_idx0)), vmagic_bias_f32);
        const float32x4_t vb4567 = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEF, vtbl_idx1)), vmagic_bias_f32);
        const float32x4_t vb89AB = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEF, vtbl_idx2)), vmagic_bias_f32);
        const float32x4_t vbCDEF = vsubq_f32(vreinterpretq_f32_u8(vqtbl2q_u8(vtbl0123456789ABCDEF, vtbl_idx3)), vmagic_bias_f32);

        vacc0x0 = vfmaq_f32(vacc0x0, va0, vb0123);
        vacc0x1 = vfmaq_f32(vacc0x1, va0, vb4567);
        vacc0x2 = vfmaq_f32(vacc0x2, va0, vb89AB);
        vacc0x3 = vfmaq_f32(vacc0x3, va0, vbCDEF);
      }
    }
    const float32x4_t vscale0123 = vld1q_f32(w); w = ((const float*) w + 4);
    const float32x4_t vscale4567 = vld1q_f32(w); w = ((const float*) w + 4);
    const float32x4_t vscale89AB = vld1q_f32(w); w = ((const float*) w + 4);
    const float32x4_t vscaleCDEF = vld1q_f32(w); w = ((const float*) w + 4);
    vacc0x0 = vmulq_f32(vacc0x0, vscale0123);
    vacc0x1 = vmulq_f32(vacc0x1, vscale4567);
    vacc0x2 = vmulq_f32(vacc0x2, vscale89AB);
    vacc0x3 = vmulq_f32(vacc0x3, vscaleCDEF);
    const float32x4_t vmax = vdupq_n_f32(params->scalar.max);
    vacc0x0 = vminq_f32(vacc0x0, vmax);
    vacc0x1 = vminq_f32(vacc0x1, vmax);
    vacc0x2 = vminq_f32(vacc0x2, vmax);
    vacc0x3 = vminq_f32(vacc0x3, vmax);

    const float32x4_t vmin = vdupq_n_f32(params->scalar.min);
    vacc0x0 = vmaxq_f32(vacc0x0, vmin);
    vacc0x1 = vmaxq_f32(vacc0x1, vmin);
    vacc0x2 = vmaxq_f32(vacc0x2, vmin);
    vacc0x3 = vmaxq_f32(vacc0x3, vmin);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f32(c0, vacc0x0);
      vst1q_f32(c0 + 4, vacc0x1);
      vst1q_f32(c0 + 8, vacc0x2);
      vst1q_f32(c0 + 12, vacc0x3);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 16;

    } else {
      if (nc & 8) {
        vst1q_f32(c0, vacc0x0); c0 += 4;
        vst1q_f32(c0, vacc0x1); c0 += 4;

        vacc0x0 = vacc0x2;
        vacc0x1 = vacc0x3;
      }
      if (nc & 4) {
        vst1q_f32(c0, vacc0x0); c0 += 4;

        vacc0x0 = vacc0x1;
        vacc0x1 = vacc0x2;
        vacc0x2 = vacc0x3;
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
