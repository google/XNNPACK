// Auto-generated file. Do not edit!
//   Template: src/qs8-vmulc/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vbinary.h"


void xnn_qu8_vmulc_minmax_fp32_ukernel__neonv8_ld128_u16(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  #if XNN_ARCH_ARM64
    const uint8x16_t va_zero_point = vld1q_dup_u8(&params->scalar.a_zero_point);
  #else
    const uint8x8_t va_zero_point = vld1_dup_u8(&params->scalar.a_zero_point);
  #endif
  const float32x4_t vscale = vld1q_dup_f32(&params->scalar.scale);
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->scalar.output_zero_point);
  const uint8x16_t voutput_min = vld1q_dup_u8(&params->scalar.output_min);
  const uint8x16_t voutput_max = vld1q_dup_u8(&params->scalar.output_max);

  const uint8x8_t vb = vld1_dup_u8(input_b);
  const uint8x8_t vb_zero_point = vld1_dup_u8(&params->scalar.b_zero_point);
  const int16x8_t vxb = vreinterpretq_s16_u16(vsubl_u8(vb, vb_zero_point));
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    const uint8x16_t va0123456789ABCDEF = vld1q_u8(input_a); input_a += 16;

    #if XNN_ARCH_ARM64
      const int16x8_t vxa01234567 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(va0123456789ABCDEF), vget_low_u8(va_zero_point)));
      const int16x8_t vxa89ABCDEF = vreinterpretq_s16_u16(vsubl_high_u8(va0123456789ABCDEF, va_zero_point));
    #else  // !XNN_ARCH_ARM64
      const int16x8_t vxa01234567 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(va0123456789ABCDEF), va_zero_point));
      const int16x8_t vxa89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(va0123456789ABCDEF), va_zero_point));
    #endif  // XNN_ARCH_ARM64

    int32x4_t vacc0123 = vmull_s16(vget_low_s16(vxa01234567), vget_low_s16(vxb));
    int32x4_t vacc4567 = vmull_s16(vget_high_s16(vxa01234567), vget_high_s16(vxb));
    int32x4_t vacc89AB = vmull_s16(vget_low_s16(vxa89ABCDEF), vget_low_s16(vxb));
    int32x4_t vaccCDEF = vmull_s16(vget_high_s16(vxa89ABCDEF), vget_high_s16(vxb));

    float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
    float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);
    float32x4_t vfpacc89AB = vcvtq_f32_s32(vacc89AB);
    float32x4_t vfpaccCDEF = vcvtq_f32_s32(vaccCDEF);

    vfpacc0123 = vmulq_f32(vfpacc0123, vscale);
    vfpacc4567 = vmulq_f32(vfpacc4567, vscale);
    vfpacc89AB = vmulq_f32(vfpacc89AB, vscale);
    vfpaccCDEF = vmulq_f32(vfpaccCDEF, vscale);

    vacc0123 = vcvtnq_s32_f32(vfpacc0123);
    vacc4567 = vcvtnq_s32_f32(vfpacc4567);
    vacc89AB = vcvtnq_s32_f32(vfpacc89AB);
    vaccCDEF = vcvtnq_s32_f32(vfpaccCDEF);

    #if XNN_ARCH_ARM64
      int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);
      int16x8_t vacc89ABCDEF = vqmovn_high_s32(vqmovn_s32(vacc89AB), vaccCDEF);
    #else
      int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
      int16x8_t vacc89ABCDEF = vcombine_s16(vqmovn_s32(vacc89AB), vqmovn_s32(vaccCDEF));
    #endif

    vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);
    vacc89ABCDEF = vqaddq_s16(vacc89ABCDEF, voutput_zero_point);

    #if XNN_ARCH_ARM64
      uint8x16_t vout0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc01234567), vacc89ABCDEF);
    #else
      uint8x16_t vout0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc01234567), vqmovun_s16(vacc89ABCDEF));
    #endif

    vout0123456789ABCDEF = vmaxq_u8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = vminq_u8(vout0123456789ABCDEF, voutput_max);

    vst1q_u8(output, vout0123456789ABCDEF); output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const uint8x8_t va01234567 = vld1_u8(input_a); input_a += 8;

      #if XNN_ARCH_ARM64
        const int16x8_t vxa01234567 = vreinterpretq_s16_u16(vsubl_u8(va01234567, vget_low_u8(va_zero_point)));
      #else  // !XNN_ARCH_ARM64
        const int16x8_t vxa01234567 = vreinterpretq_s16_u16(vsubl_u8(va01234567, va_zero_point));
      #endif  // XNN_ARCH_ARM64

      int32x4_t vacc0123 = vmull_s16(vget_low_s16(vxa01234567), vget_low_s16(vxb));
      int32x4_t vacc4567 = vmull_s16(vget_high_s16(vxa01234567), vget_high_s16(vxb));

      float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
      float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);

      vfpacc0123 = vmulq_f32(vfpacc0123, vscale);
      vfpacc4567 = vmulq_f32(vfpacc4567, vscale);

      vacc0123 = vcvtnq_s32_f32(vfpacc0123);
      vacc4567 = vcvtnq_s32_f32(vfpacc4567);

      #if XNN_ARCH_ARM64
        int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);
      #else
        int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
      #endif

      vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);

      uint8x8_t vout01234567 = vqmovun_s16(vacc01234567);

      vout01234567 = vmax_u8(vout01234567, vget_low_u8(voutput_min));
      vout01234567 = vmin_u8(vout01234567, vget_low_u8(voutput_max));
      if XNN_LIKELY(batch >= (8 * sizeof(uint8_t))) {
        vst1_u8(output, vout01234567); output += 8;
        batch -= 8 * sizeof(uint8_t);
      } else {
        if (batch & (4 * sizeof(uint8_t))) {
          vst1_lane_u32((void*) output, vreinterpret_u32_u8(vout01234567), 0); output += 4;
          vout01234567 = vext_u8(vout01234567, vout01234567, 4);
        }
        if (batch & (2 * sizeof(uint8_t))) {
          vst1_lane_u16((void*) output, vreinterpret_u16_u8(vout01234567), 0); output += 2;
          vout01234567 = vext_u8(vout01234567, vout01234567, 2);
        }
        if (batch & (1 * sizeof(uint8_t))) {
          vst1_lane_u8(output, vout01234567, 0);
        }
        batch = 0;
      }
    } while (batch != 0);
  }
}
