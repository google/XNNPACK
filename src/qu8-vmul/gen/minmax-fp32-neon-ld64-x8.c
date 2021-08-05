// Auto-generated file. Do not edit!
//   Template: src/qs8-vmul/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/vadd.h>


void xnn_qu8_vmul_minmax_fp32_ukernel__neon_ld64_x8(
    size_t n,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN XNN_DISABLE_MSAN
{
  const uint8x8_t va_zero_point = vld1_dup_u8(params->fp32_neon.a_zero_point);
  const uint8x8_t vb_zero_point = vld1_dup_u8(params->fp32_neon.b_zero_point);
  const float32x4_t vscale = vld1q_dup_f32(&params->fp32_neon.scale);
  const float32x4_t voutput_min_less_zero_point = vld1q_dup_f32(&params->fp32_neon.output_min_less_zero_point);
  const float32x4_t voutput_max_less_zero_point = vld1q_dup_f32(&params->fp32_neon.output_max_less_zero_point);
  const float32x4_t vmagic_bias = vld1q_dup_f32(&params->fp32_neon.magic_bias);
  const int32x4_t vmagic_bias_less_zero_point = vld1q_dup_s32(&params->fp32_neon.magic_bias_less_zero_point);

  for (; n >= 8 * sizeof(uint8_t); n -= 8 * sizeof(uint8_t)) {
    const uint8x8_t va01234567 = vld1_u8(input_a); input_a += 8;
    const uint8x8_t vb01234567 = vld1_u8(input_b); input_b += 8;

    const int16x8_t vxa01234567 = vreinterpretq_s16_u16(vsubl_u8(va01234567, va_zero_point));
    const int16x8_t vxb01234567 = vreinterpretq_s16_u16(vsubl_u8(vb01234567, vb_zero_point));

    int32x4_t vacc0123 = vmull_s16(vget_low_s16(vxa01234567), vget_low_s16(vxb01234567));
    int32x4_t vacc4567 = vmull_s16(vget_high_s16(vxa01234567), vget_high_s16(vxb01234567));

    float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
    float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);

    vfpacc0123 = vmulq_f32(vfpacc0123, vscale);
    vfpacc4567 = vmulq_f32(vfpacc4567, vscale);

    vfpacc0123 = vmaxq_f32(vfpacc0123, voutput_min_less_zero_point);
    vfpacc4567 = vmaxq_f32(vfpacc4567, voutput_min_less_zero_point);

    vfpacc0123 = vminq_f32(vfpacc0123, voutput_max_less_zero_point);
    vfpacc4567 = vminq_f32(vfpacc4567, voutput_max_less_zero_point);

    vacc0123 = vreinterpretq_s32_f32(vaddq_f32(vfpacc0123, vmagic_bias));
    vacc4567 = vreinterpretq_s32_f32(vaddq_f32(vfpacc4567, vmagic_bias));

    vacc0123 = vsubq_s32(vacc0123, vmagic_bias_less_zero_point);
    vacc4567 = vsubq_s32(vacc4567, vmagic_bias_less_zero_point);

#if XNN_ARCH_ARM64
      const int16x8_t vacc01234567 = vuzp1q_s16(vreinterpretq_s16_s32(vacc0123), vreinterpretq_s16_s32(vacc4567));

      uint8x8_t vout01234567 = vreinterpret_u8_s8(vmovn_s16(vacc01234567));
#else
      const int16x8_t vacc01234567 = vcombine_s16(vmovn_s32(vacc0123), vmovn_s32(vacc4567));

      uint8x8_t vout01234567 = vreinterpret_u8_s8(vmovn_s16(vacc01234567));
#endif

    vst1_u8(output, vout01234567); output += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    {
      const uint8x8_t va01234567 = vld1_u8(input_a);
      const uint8x8_t vb01234567 = vld1_u8(input_b);

      const int16x8_t vxa01234567 = vreinterpretq_s16_u16(vsubl_u8(va01234567, va_zero_point));
      const int16x8_t vxb01234567 = vreinterpretq_s16_u16(vsubl_u8(vb01234567, vb_zero_point));

      int32x4_t vacc0123 = vmull_s16(vget_low_s16(vxa01234567), vget_low_s16(vxb01234567));
      int32x4_t vacc4567 = vmull_s16(vget_high_s16(vxa01234567), vget_high_s16(vxb01234567));

      float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
      float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);

      vfpacc0123 = vmulq_f32(vfpacc0123, vscale);
      vfpacc4567 = vmulq_f32(vfpacc4567, vscale);

      vfpacc0123 = vmaxq_f32(vfpacc0123, voutput_min_less_zero_point);
      vfpacc4567 = vmaxq_f32(vfpacc4567, voutput_min_less_zero_point);

      vfpacc0123 = vminq_f32(vfpacc0123, voutput_max_less_zero_point);
      vfpacc4567 = vminq_f32(vfpacc4567, voutput_max_less_zero_point);

      vacc0123 = vreinterpretq_s32_f32(vaddq_f32(vfpacc0123, vmagic_bias));
      vacc4567 = vreinterpretq_s32_f32(vaddq_f32(vfpacc4567, vmagic_bias));

      vacc0123 = vsubq_s32(vacc0123, vmagic_bias_less_zero_point);
      vacc4567 = vsubq_s32(vacc4567, vmagic_bias_less_zero_point);

#if XNN_ARCH_ARM64
      const int16x8_t vacc01234567 = vuzp1q_s16(vreinterpretq_s16_s32(vacc0123), vreinterpretq_s16_s32(vacc4567));
      uint8x8_t vout01234567 = vreinterpret_u8_s8(vmovn_s16(vacc01234567));
#else
      const int16x8_t vacc01234567 = vcombine_s16(vmovn_s32(vacc0123), vmovn_s32(vacc4567));
      uint8x8_t vout01234567 = vreinterpret_u8_s8(vmovn_s16(vacc01234567));
#endif

      if (n & (4 * sizeof(uint8_t))) {
        vst1_lane_u32(__builtin_assume_aligned(output, 1), vreinterpret_u32_u8(vout01234567), 0); output += 4;
        vout01234567 = vext_u8(vout01234567, vout01234567, 4);
      }
      if (n & (2 * sizeof(uint8_t))) {
        vst1_lane_u16(__builtin_assume_aligned(output, 1), vreinterpret_u16_u8(vout01234567), 0); output += 2;
        vout01234567 = vext_u8(vout01234567, vout01234567, 2);
      }
      if (n & (1 * sizeof(uint8_t))) {
        vst1_lane_u8(output, vout01234567, 0);
      }
    }
  }
}
