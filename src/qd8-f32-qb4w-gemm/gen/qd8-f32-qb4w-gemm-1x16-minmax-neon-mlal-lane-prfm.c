// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/neon-mlal-lane.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/gemm.h"
#include "xnnpack/math.h"
#include "xnnpack/prefetch.h"


void xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qb4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  size_t bl = params->scalar.blocksize;
  assert(bl <= round_up_po2(kc, 2));
  assert(bl != 0);
  assert(bl % 32 == 0);
  const int8_t* a0 = a;
  float* c0 = c;

  const int8x8_t vmask = vmov_n_s8(INT8_C(0xF0));
  kc = round_up_po2(kc, 2);
  do {
    float32x4_t vksum0123 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vksum4567 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vksum89AB = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vksumCDEF = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vzp0 = vcvtq_f32_s32(vld1q_dup_s32(&quantization_params[0].zero_point));
    float32x4_t vout0x0123 = vmulq_f32(vksum0123, vzp0);
    float32x4_t vout0x4567 = vmulq_f32(vksum4567, vzp0);
    float32x4_t vout0x89AB = vmulq_f32(vksum89AB, vzp0);
    float32x4_t vout0xCDEF = vmulq_f32(vksumCDEF, vzp0);

    for (size_t kb=0; kb < kc; kb += bl) {
      int32x4_t vacc0x0123 = vdupq_n_s32(0);
      int32x4_t vacc0x4567 = vdupq_n_s32(0);
      int32x4_t vacc0x89AB = vdupq_n_s32(0);
      int32x4_t vacc0xCDEF = vdupq_n_s32(0);

      size_t k = bl;
      while (k >= 8 * sizeof(int8_t)) {
        const int8x8_t va0 = vld1_s8(a0); a0 += 8;
        const int16x8_t vxa0 = vmovl_s8(va0);

        const int8x8_t vb01234567c01 = vld1_s8(w); w = (const uint8_t*) w + 8;
        const int8x8_t vb89ABCDEFc01 = vld1_s8(w); w = (const uint8_t*) w + 8;
        const int8x8_t vb01234567c23 = vld1_s8(w); w = (const uint8_t*) w + 8;
        const int8x8_t vb89ABCDEFc23 = vld1_s8(w); w = (const uint8_t*) w + 8;
        const int8x8_t vb01234567c45 = vld1_s8(w); w = (const uint8_t*) w + 8;
        const int8x8_t vb89ABCDEFc45 = vld1_s8(w); w = (const uint8_t*) w + 8;
        const int8x8_t vb01234567c67 = vld1_s8(w); w = (const uint8_t*) w + 8;
        const int8x8_t vb89ABCDEFc67 = vld1_s8(w); w = (const uint8_t*) w + 8;
        const int8x8_t vb01234567c0 = vshl_n_s8(vb01234567c01, 4);
        const int8x8_t vb01234567c1 = vand_s8(vb01234567c01, vmask);
        const int8x8_t vb89ABCDEFc0 = vshl_n_s8(vb89ABCDEFc01, 4);
        const int8x8_t vb89ABCDEFc1 = vand_s8(vb89ABCDEFc01, vmask);
        const int8x8_t vb01234567c2 = vshl_n_s8(vb01234567c23, 4);
        const int8x8_t vb01234567c3 = vand_s8(vb01234567c23, vmask);
        const int8x8_t vb89ABCDEFc2 = vshl_n_s8(vb89ABCDEFc23, 4);
        const int8x8_t vb89ABCDEFc3 = vand_s8(vb89ABCDEFc23, vmask);
        const int8x8_t vb01234567c4 = vshl_n_s8(vb01234567c45, 4);
        const int8x8_t vb01234567c5 = vand_s8(vb01234567c45, vmask);
        const int8x8_t vb89ABCDEFc4 = vshl_n_s8(vb89ABCDEFc45, 4);
        const int8x8_t vb89ABCDEFc5 = vand_s8(vb89ABCDEFc45, vmask);
        const int8x8_t vb01234567c6 = vshl_n_s8(vb01234567c67, 4);
        const int8x8_t vb01234567c7 = vand_s8(vb01234567c67, vmask);
        const int8x8_t vb89ABCDEFc6 = vshl_n_s8(vb89ABCDEFc67, 4);
        const int8x8_t vb89ABCDEFc7 = vand_s8(vb89ABCDEFc67, vmask);

        const int16x8_t vxb01234567c0 = vmovl_s8(vb01234567c0);
        const int16x8_t vxb89ABCDEFc0 = vmovl_s8(vb89ABCDEFc0);
        const int16x8_t vxb01234567c1 = vmovl_s8(vb01234567c1);
        const int16x8_t vxb89ABCDEFc1 = vmovl_s8(vb89ABCDEFc1);
        const int16x8_t vxb01234567c2 = vmovl_s8(vb01234567c2);
        const int16x8_t vxb89ABCDEFc2 = vmovl_s8(vb89ABCDEFc2);
        const int16x8_t vxb01234567c3 = vmovl_s8(vb01234567c3);
        const int16x8_t vxb89ABCDEFc3 = vmovl_s8(vb89ABCDEFc3);
        const int16x8_t vxb01234567c4 = vmovl_s8(vb01234567c4);
        const int16x8_t vxb89ABCDEFc4 = vmovl_s8(vb89ABCDEFc4);
        const int16x8_t vxb01234567c5 = vmovl_s8(vb01234567c5);
        const int16x8_t vxb89ABCDEFc5 = vmovl_s8(vb89ABCDEFc5);
        const int16x8_t vxb01234567c6 = vmovl_s8(vb01234567c6);
        const int16x8_t vxb89ABCDEFc6 = vmovl_s8(vb89ABCDEFc6);
        const int16x8_t vxb01234567c7 = vmovl_s8(vb01234567c7);
        const int16x8_t vxb89ABCDEFc7 = vmovl_s8(vb89ABCDEFc7);

        xnn_prefetch_to_l1((const int8_t*) w + 448);
        xnn_prefetch_to_l1((const int8_t*) w + 512);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa0), 3);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa0), 3);

        k -= 8 * sizeof(int8_t);
      }
      if XNN_UNLIKELY(k != 0) {
        const int8x8_t va0 = vld1_s8(a0); a0 = (const int8_t*) ((uintptr_t) a0 + k);
        const int16x8_t vxa0 = vmovl_s8(va0);

        const int8x8_t vb01234567c01 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb89ABCDEFc01 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb01234567c0 = vshl_n_s8(vb01234567c01, 4);
        const int8x8_t vb01234567c1 = vand_s8(vb01234567c01, vmask);
        const int8x8_t vb89ABCDEFc0 = vshl_n_s8(vb89ABCDEFc01, 4);
        const int8x8_t vb89ABCDEFc1 = vand_s8(vb89ABCDEFc01, vmask);

        const int16x8_t vxb01234567c0 = vmovl_s8(vb01234567c0);
        const int16x8_t vxb89ABCDEFc0 = vmovl_s8(vb89ABCDEFc0);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);

      if (k >= 2 * sizeof(int8_t)) {
          const int16x8_t vxb01234567c1 = vmovl_s8(vb01234567c1);
          const int16x8_t vxb89ABCDEFc1 = vmovl_s8(vb89ABCDEFc1);

          vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
          vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
          vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
          vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);

          if (k > 2 * sizeof(int8_t)) {
            const int8x8_t vb01234567c23 = vld1_s8(w); w = (const int8_t*) w + 8;
            const int8x8_t vb89ABCDEFc23 = vld1_s8(w); w = (const int8_t*) w + 8;
            const int8x8_t vb01234567c2 = vshl_n_s8(vb01234567c23, 4);
            const int8x8_t vb01234567c3 = vand_s8(vb01234567c23, vmask);
            const int8x8_t vb89ABCDEFc2 = vshl_n_s8(vb89ABCDEFc23, 4);
            const int8x8_t vb89ABCDEFc3 = vand_s8(vb89ABCDEFc23, vmask);
            const int16x8_t vxb01234567c2 = vmovl_s8(vb01234567c2);
            const int16x8_t vxb89ABCDEFc2 = vmovl_s8(vb89ABCDEFc2);

            vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
            vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
            vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
            vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);

            if (k >= 4 * sizeof(int8_t)) {
              const int16x8_t vxb01234567c3 = vmovl_s8(vb01234567c3);
              const int16x8_t vxb89ABCDEFc3 = vmovl_s8(vb89ABCDEFc3);

              vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
              vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
              vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
              vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);

              if (k > 4 * sizeof(int8_t)) {
                const int8x8_t vb01234567c45 = vld1_s8(w); w = (const int8_t*) w + 8;
                const int8x8_t vb89ABCDEFc45 = vld1_s8(w); w = (const int8_t*) w + 8;
                const int8x8_t vb01234567c4 = vshl_n_s8(vb01234567c45, 4);
                const int8x8_t vb01234567c5 = vand_s8(vb01234567c45, vmask);
                const int8x8_t vb89ABCDEFc4 = vshl_n_s8(vb89ABCDEFc45, 4);
                const int8x8_t vb89ABCDEFc5 = vand_s8(vb89ABCDEFc45, vmask);
                const int16x8_t vxb01234567c4 = vmovl_s8(vb01234567c4);
                const int16x8_t vxb89ABCDEFc4 = vmovl_s8(vb89ABCDEFc4);

                vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
                vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
                vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
                vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);

                if (k >= 6 * sizeof(int8_t)) {
                  const int16x8_t vxb01234567c5 = vmovl_s8(vb01234567c5);
                  const int16x8_t vxb89ABCDEFc5 = vmovl_s8(vb89ABCDEFc5);

                  vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                  vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                  vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
                  vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);

                }
              }
            }
          }
        }
      }

    const float32x4_t vfilter_output_scale0123 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)); w = (const uint16_t*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)); w = (const uint16_t*) w + 4;
    const float32x4_t vfilter_output_scale89AB = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)); w = (const uint16_t*) w + 4;
    const float32x4_t vfilter_output_scaleCDEF = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)); w = (const uint16_t*) w + 4;

    const float32x4_t vf0x0123 = vcvtq_f32_s32(vacc0x0123);
    const float32x4_t vf0x4567 = vcvtq_f32_s32(vacc0x4567);
    const float32x4_t vf0x89AB = vcvtq_f32_s32(vacc0x89AB);
    const float32x4_t vf0xCDEF = vcvtq_f32_s32(vacc0xCDEF);

    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_f32(vout0x0123, vf0x0123, vfilter_output_scale0123);
      vout0x4567 = vfmaq_f32(vout0x4567, vf0x4567, vfilter_output_scale4567);
      vout0x89AB = vfmaq_f32(vout0x89AB, vf0x89AB, vfilter_output_scale89AB);
      vout0xCDEF = vfmaq_f32(vout0xCDEF, vf0xCDEF, vfilter_output_scaleCDEF);
    #else
      vout0x0123 = vmlaq_f32(vout0x0123, vf0x0123, vfilter_output_scale0123);
      vout0x4567 = vmlaq_f32(vout0x4567, vf0x4567, vfilter_output_scale4567);
      vout0x89AB = vmlaq_f32(vout0x89AB, vf0x89AB, vfilter_output_scale89AB);
      vout0xCDEF = vmlaq_f32(vout0xCDEF, vf0xCDEF, vfilter_output_scaleCDEF);
    #endif
    }
    const float32x4_t vinput_scale0 = vld1q_dup_f32(&quantization_params[0].inv_scale);

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;

    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vinput_scale0);
    #else
      vout0x0123 = vmlaq_f32(vbias0123, vout0x0123, vinput_scale0);
    #endif
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;

    #if XNN_ARCH_ARM64
      vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vinput_scale0);
    #else
      vout0x4567 = vmlaq_f32(vbias4567, vout0x4567, vinput_scale0);
    #endif
    const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;

    #if XNN_ARCH_ARM64
      vout0x89AB = vfmaq_f32(vbias89AB, vout0x89AB, vinput_scale0);
    #else
      vout0x89AB = vmlaq_f32(vbias89AB, vout0x89AB, vinput_scale0);
    #endif
    const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;

    #if XNN_ARCH_ARM64
      vout0xCDEF = vfmaq_f32(vbiasCDEF, vout0xCDEF, vinput_scale0);
    #else
      vout0xCDEF = vmlaq_f32(vbiasCDEF, vout0xCDEF, vinput_scale0);
    #endif

    const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
    vout0x0123 = vmaxq_f32(vout0x0123, voutput_min);
    vout0x4567 = vmaxq_f32(vout0x4567, voutput_min);
    vout0x89AB = vmaxq_f32(vout0x89AB, voutput_min);
    vout0xCDEF = vmaxq_f32(vout0xCDEF, voutput_min);

    const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
    vout0x0123 = vminq_f32(vout0x0123, voutput_max);
    vout0x4567 = vminq_f32(vout0x4567, voutput_max);
    vout0x89AB = vminq_f32(vout0x89AB, voutput_max);
    vout0xCDEF = vminq_f32(vout0xCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f32(&c0[0], vout0x0123);
      vst1q_f32(&c0[4], vout0x4567);
      vst1q_f32(&c0[8], vout0x89AB);
      vst1q_f32(&c0[12], vout0xCDEF);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 16;
    } else {
      if (nc & 8) {
        vst1q_f32(c0, vout0x0123); c0 += 4;
        vout0x0123 = vout0x89AB;
        vst1q_f32(c0, vout0x4567); c0 += 4;
        vout0x4567 = vout0xCDEF;
      }
      if (nc & 4) {
        vst1q_f32(c0, vout0x0123); c0 += 4;
        vout0x0123 = vout0x4567;
      }
      float32x2_t vout0x01 = vget_low_f32(vout0x0123);
      if (nc & 2) {
        vst1_f32(c0, vout0x01); c0 += 2;
        vout0x01 = vget_high_f32(vout0x0123);
      }
      if (nc & 1) {
        vst1_lane_f32(c0, vout0x01, 0);
      }
      nc = 0;
    }
  } while (nc != 0);
}
