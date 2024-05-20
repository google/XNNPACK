// Auto-generated file. Do not edit!
//   Template: src/qs8-rsum/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/reduce.h>

void xnn_qs8_rsum_minmax_fp32_ukernel__neon_u64(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);

  // 256 int8s may be summed into an int16 before overflowing
  // Each vpada does 2 adds so we can do 128 vpada
  // There 1 registers so batch size is 128
  int num_batches = batch  >> 7;
  int32x4_t vacc0 = vmovq_n_s32(0);

  for (; num_batches > 0; --num_batches) {
    int16x8_t vacc16_0 = vmovq_n_s16(0);
    for (size_t current_batch = 128; current_batch > 0; current_batch -= 64) {
      const int8x16_t vt0 = vld1q_s8(input); input += 16;
      const int8x16_t vt1 = vld1q_s8(input); input += 16;
      const int8x16_t vt2 = vld1q_s8(input); input += 16;
      const int8x16_t vt3 = vld1q_s8(input); input += 16;

      vacc16_0 = vpadalq_s8(vacc16_0, vt0);
      vacc16_0 = vpadalq_s8(vacc16_0, vt1);
      vacc16_0 = vpadalq_s8(vacc16_0, vt2);
      vacc16_0 = vpadalq_s8(vacc16_0, vt3);
    }
    vacc0 = vpadalq_s16(vacc0, vacc16_0);
    batch -= 128;
  }

  if (XNN_UNLIKELY(batch != 0)) {
    int16x8_t vacc16_0 = vmovq_n_s16(0);
    for (; batch >= 64; batch -= 64) {
      const int8x16_t vt0 = vld1q_s8(input); input += 16;
      const int8x16_t vt1 = vld1q_s8(input); input += 16;
      const int8x16_t vt2 = vld1q_s8(input); input += 16;
      const int8x16_t vt3 = vld1q_s8(input); input += 16;

      vacc16_0 = vpadalq_s8(vacc16_0, vt0);
      vacc16_0 = vpadalq_s8(vacc16_0, vt1);
      vacc16_0 = vpadalq_s8(vacc16_0, vt2);
      vacc16_0 = vpadalq_s8(vacc16_0, vt3);
    }
    for (; batch >= 16; batch -= 16) {
      const int8x16_t vt = vld1q_s8(input); input += 16;
      vacc16_0 = vpadalq_s8(vacc16_0, vt);
    }
    if (XNN_UNLIKELY(batch != 0)) {
      const int8x16_t vt = vld1q_s8(input);
      const int8x16_t vmask = vld1q_s8(&params->fp32_neon.mask_table[15 - batch]);
      const int8x16_t vtm = vmulq_s8(vt, vmask);
      vacc16_0 = vpadalq_s8(vacc16_0, vtm);
    }
    vacc0 = vpadalq_s16(vacc0, vacc16_0);
  }
  #if XNN_ARCH_ARM64
    const int32_t vacc = vaddvq_s32(vacc0);
  #else
    int32x2_t vacc_lo = vadd_s32(vget_low_s32(vacc0), vget_high_s32(vacc0));
    vacc_lo = vpadd_s32(vacc_lo, vacc_lo);
    const int32_t vacc = vget_lane_s32(vacc_lo, 0);
  #endif

  const int32_t vinit_bias = params->fp32_neon.init_bias;
  const float vscale = params->fp32_neon.scale;
  const int32_t output_min = params->fp32_neon.output_min;
  const int32_t output_max = params->fp32_neon.output_max;
  const float vmagic_bias = params->fp32_neon.magic_bias;
  const int32_t vmagic_bias_less_output_zero_point = params->fp32_neon.magic_bias_less_output_zero_point;

  float vfpacc = (float) (vacc + vinit_bias) * vscale;
  vfpacc += vmagic_bias;
  int32_t vout = (int32_t) float_as_uint32(vfpacc);
  vout -= vmagic_bias_less_output_zero_point;
  vout = math_max_s32(vout, output_min);
  vout = math_min_s32(vout, output_max);
  *output += (int8_t) vout;
}
