// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-avgpool/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/avgpool.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_qu8_avgpool_minmax_fp32_ukernel_9p__neon_u16(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const uint8_t** input,
    size_t input_offset,
    size_t input_pixel_stride,
    const uint8_t* zero,
    const float* multiplier,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const struct xnn_qu8_avgpool_minmax_params* restrict params) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const int32x4_t vinit_bias = vdupq_n_s32(params->scalar.init_bias);
  float32x4_t vscale = vdupq_n_f32(params->scalar.scale);
#if XNN_ARCH_ARM64
  const int16x8_t voutput_zero_point = vdupq_n_s16(params->scalar.output_zero_point);
#else
  const float32x4_t vmagic_bias = vdupq_n_f32(12582912.0f);
  const int32x4_t vmagic_bias_less_output_zero_point =
      vdupq_n_s32(INT32_C(0x4B400000) - (int32_t) params->scalar.output_zero_point);
#endif
  const uint8x16_t voutput_min = vdupq_n_u8(params->scalar.output_min);
  const uint8x16_t voutput_max = vdupq_n_u8(params->scalar.output_max);

  do {
    if (multiplier != NULL) {
      vscale = vdupq_n_f32(*multiplier++);
    }

    uint8_t* o = output;
    size_t c_offset = 0;
    size_t c = channels;
    for (; c >= 16; c -= 16) {
      const uint8_t** i = input;
      size_t k = kernel_elements;
      int32x4_t vacc0123 = vdupq_n_s32(0);
      int32x4_t vacc4567 = vdupq_n_s32(0);
      int32x4_t vacc89AB = vdupq_n_s32(0);
      int32x4_t vaccCDEF = vdupq_n_s32(0);
      do {
        vacc0123 = vaddq_s32(vacc0123, vinit_bias);
        vacc4567 = vaddq_s32(vacc4567, vinit_bias);
        vacc89AB = vaddq_s32(vacc89AB, vinit_bias);
        vaccCDEF = vaddq_s32(vaccCDEF, vinit_bias);

        const uint8_t* i0 = 0 < k ? *i++ : zero;
        assert(i0 != NULL);
        if XNN_UNPREDICTABLE(i0 != zero) {
          i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
        }
        i0 += c_offset;
        const uint8x8_t vi0_lo = vld1_u8(i0);
        const uint8x8_t vi0_hi = vld1_u8(i0 + 8);
        const uint8_t* i1 = 1 < k ? *i++ : zero;
        assert(i1 != NULL);
        if XNN_UNPREDICTABLE(i1 != zero) {
          i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
        }
        i1 += c_offset;
        const uint8x8_t vi1_lo = vld1_u8(i1);
        const uint8x8_t vi1_hi = vld1_u8(i1 + 8);
        const uint8_t* i2 = 2 < k ? *i++ : zero;
        assert(i2 != NULL);
        if XNN_UNPREDICTABLE(i2 != zero) {
          i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
        }
        i2 += c_offset;
        const uint8x8_t vi2_lo = vld1_u8(i2);
        const uint8x8_t vi2_hi = vld1_u8(i2 + 8);
        const uint8_t* i3 = 3 < k ? *i++ : zero;
        assert(i3 != NULL);
        if XNN_UNPREDICTABLE(i3 != zero) {
          i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
        }
        i3 += c_offset;
        const uint8x8_t vi3_lo = vld1_u8(i3);
        const uint8x8_t vi3_hi = vld1_u8(i3 + 8);
        const uint8_t* i4 = 4 < k ? *i++ : zero;
        assert(i4 != NULL);
        if XNN_UNPREDICTABLE(i4 != zero) {
          i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
        }
        i4 += c_offset;
        const uint8x8_t vi4_lo = vld1_u8(i4);
        const uint8x8_t vi4_hi = vld1_u8(i4 + 8);
        const uint8_t* i5 = 5 < k ? *i++ : zero;
        assert(i5 != NULL);
        if XNN_UNPREDICTABLE(i5 != zero) {
          i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
        }
        i5 += c_offset;
        const uint8x8_t vi5_lo = vld1_u8(i5);
        const uint8x8_t vi5_hi = vld1_u8(i5 + 8);
        const uint8_t* i6 = 6 < k ? *i++ : zero;
        assert(i6 != NULL);
        if XNN_UNPREDICTABLE(i6 != zero) {
          i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
        }
        i6 += c_offset;
        const uint8x8_t vi6_lo = vld1_u8(i6);
        const uint8x8_t vi6_hi = vld1_u8(i6 + 8);
        const uint8_t* i7 = 7 < k ? *i++ : zero;
        assert(i7 != NULL);
        if XNN_UNPREDICTABLE(i7 != zero) {
          i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
        }
        i7 += c_offset;
        const uint8x8_t vi7_lo = vld1_u8(i7);
        const uint8x8_t vi7_hi = vld1_u8(i7 + 8);
        const uint8_t* i8 = 8 < k ? *i++ : zero;
        assert(i8 != NULL);
        if XNN_UNPREDICTABLE(i8 != zero) {
          i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
        }
        i8 += c_offset;
        const uint8x8_t vi8_lo = vld1_u8(i8);
        const uint8x8_t vi8_hi = vld1_u8(i8 + 8);

        const int16x8_t vsum018_lo = vreinterpretq_s16_u16(vaddw_u8(vaddl_u8(vi0_lo, vi1_lo), vi8_lo));
        const int16x8_t vsum23_lo = vreinterpretq_s16_u16(vaddl_u8(vi2_lo, vi3_lo));
        const int16x8_t vsum45_lo = vreinterpretq_s16_u16(vaddl_u8(vi4_lo, vi5_lo));
        const int16x8_t vsum67_lo = vreinterpretq_s16_u16(vaddl_u8(vi6_lo, vi7_lo));
        const int16x8_t vsum018_hi = vreinterpretq_s16_u16(vaddw_u8(vaddl_u8(vi0_hi, vi1_hi), vi8_hi));
        const int16x8_t vsum23_hi = vreinterpretq_s16_u16(vaddl_u8(vi2_hi, vi3_hi));
        const int16x8_t vsum45_hi = vreinterpretq_s16_u16(vaddl_u8(vi4_hi, vi5_hi));
        const int16x8_t vsum67_hi = vreinterpretq_s16_u16(vaddl_u8(vi6_hi, vi7_hi));

        const int16x8_t vsum_lo = vaddq_s16(vaddq_s16(vsum23_lo, vsum45_lo), vaddq_s16(vsum018_lo, vsum67_lo));
        const int16x8_t vsum_hi = vaddq_s16(vaddq_s16(vsum23_hi, vsum45_hi), vaddq_s16(vsum018_hi, vsum67_hi));

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vsum_lo));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vsum_lo));
        vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vsum_hi));
        vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vsum_hi));

        k = doz(k, 9);
      } while (k != 0);

      float32x4_t vfpacc0123 = vmulq_f32(vcvtq_f32_s32(vacc0123), vscale);
      float32x4_t vfpacc4567 = vmulq_f32(vcvtq_f32_s32(vacc4567), vscale);
      float32x4_t vfpacc89AB = vmulq_f32(vcvtq_f32_s32(vacc89AB), vscale);
      float32x4_t vfpaccCDEF = vmulq_f32(vcvtq_f32_s32(vaccCDEF), vscale);

#if XNN_ARCH_ARM64
      vacc0123 = vcvtnq_s32_f32(vfpacc0123);
      vacc4567 = vcvtnq_s32_f32(vfpacc4567);
      vacc89AB = vcvtnq_s32_f32(vfpacc89AB);
      vaccCDEF = vcvtnq_s32_f32(vfpaccCDEF);

      const int16x8_t vacc01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567), voutput_zero_point);
      const int16x8_t vacc89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc89AB), vaccCDEF), voutput_zero_point);
      uint8x16_t vout = vqmovun_high_s16(vqmovun_s16(vacc01234567), vacc89ABCDEF);
#else
      vacc0123 = vqsubq_s32(vreinterpretq_s32_f32(vaddq_f32(vfpacc0123, vmagic_bias)), vmagic_bias_less_output_zero_point);
      vacc4567 = vqsubq_s32(vreinterpretq_s32_f32(vaddq_f32(vfpacc4567, vmagic_bias)), vmagic_bias_less_output_zero_point);
      vacc89AB = vqsubq_s32(vreinterpretq_s32_f32(vaddq_f32(vfpacc89AB, vmagic_bias)), vmagic_bias_less_output_zero_point);
      vaccCDEF = vqsubq_s32(vreinterpretq_s32_f32(vaddq_f32(vfpaccCDEF, vmagic_bias)), vmagic_bias_less_output_zero_point);

      const int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
      const int16x8_t vacc89ABCDEF = vcombine_s16(vqmovn_s32(vacc89AB), vqmovn_s32(vaccCDEF));
      uint8x16_t vout = vcombine_u8(vqmovun_s16(vacc01234567), vqmovun_s16(vacc89ABCDEF));
#endif
      vout = vmaxq_u8(vout, voutput_min);
      vout = vminq_u8(vout, voutput_max);

      vst1q_u8(o, vout);
      o += 16;
      c_offset += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      do {
        const uint8_t** i = input;
        size_t k = kernel_elements;
        int32x4_t vacc0123 = vdupq_n_s32(0);
        int32x4_t vacc4567 = vdupq_n_s32(0);
        do {
          vacc0123 = vaddq_s32(vacc0123, vinit_bias);
          vacc4567 = vaddq_s32(vacc4567, vinit_bias);

          const uint8_t* i0 = 0 < k ? *i++ : zero;
          assert(i0 != NULL);
          if XNN_UNPREDICTABLE(i0 != zero) {
            i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
          }
          i0 += c_offset;
          const uint8x8_t vi0 = vld1_u8(i0);
          const uint8_t* i1 = 1 < k ? *i++ : zero;
          assert(i1 != NULL);
          if XNN_UNPREDICTABLE(i1 != zero) {
            i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
          }
          i1 += c_offset;
          const uint8x8_t vi1 = vld1_u8(i1);
          const uint8_t* i2 = 2 < k ? *i++ : zero;
          assert(i2 != NULL);
          if XNN_UNPREDICTABLE(i2 != zero) {
            i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
          }
          i2 += c_offset;
          const uint8x8_t vi2 = vld1_u8(i2);
          const uint8_t* i3 = 3 < k ? *i++ : zero;
          assert(i3 != NULL);
          if XNN_UNPREDICTABLE(i3 != zero) {
            i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
          }
          i3 += c_offset;
          const uint8x8_t vi3 = vld1_u8(i3);
          const uint8_t* i4 = 4 < k ? *i++ : zero;
          assert(i4 != NULL);
          if XNN_UNPREDICTABLE(i4 != zero) {
            i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
          }
          i4 += c_offset;
          const uint8x8_t vi4 = vld1_u8(i4);
          const uint8_t* i5 = 5 < k ? *i++ : zero;
          assert(i5 != NULL);
          if XNN_UNPREDICTABLE(i5 != zero) {
            i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
          }
          i5 += c_offset;
          const uint8x8_t vi5 = vld1_u8(i5);
          const uint8_t* i6 = 6 < k ? *i++ : zero;
          assert(i6 != NULL);
          if XNN_UNPREDICTABLE(i6 != zero) {
            i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
          }
          i6 += c_offset;
          const uint8x8_t vi6 = vld1_u8(i6);
          const uint8_t* i7 = 7 < k ? *i++ : zero;
          assert(i7 != NULL);
          if XNN_UNPREDICTABLE(i7 != zero) {
            i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
          }
          i7 += c_offset;
          const uint8x8_t vi7 = vld1_u8(i7);
          const uint8_t* i8 = 8 < k ? *i++ : zero;
          assert(i8 != NULL);
          if XNN_UNPREDICTABLE(i8 != zero) {
            i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
          }
          i8 += c_offset;
          const uint8x8_t vi8 = vld1_u8(i8);

          const int16x8_t vsum018 = vreinterpretq_s16_u16(vaddw_u8(vaddl_u8(vi0, vi1), vi8));
          const int16x8_t vsum23 = vreinterpretq_s16_u16(vaddl_u8(vi2, vi3));
          const int16x8_t vsum45 = vreinterpretq_s16_u16(vaddl_u8(vi4, vi5));
          const int16x8_t vsum67 = vreinterpretq_s16_u16(vaddl_u8(vi6, vi7));

          const int16x8_t vsum = vaddq_s16(vaddq_s16(vsum23, vsum45), vaddq_s16(vsum018, vsum67));

          vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vsum));
          vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vsum));

          k = doz(k, 9);
        } while (k != 0);

        float32x4_t vfpacc0123 = vmulq_f32(vcvtq_f32_s32(vacc0123), vscale);
        float32x4_t vfpacc4567 = vmulq_f32(vcvtq_f32_s32(vacc4567), vscale);

#if XNN_ARCH_ARM64
        vacc0123 = vcvtnq_s32_f32(vfpacc0123);
        vacc4567 = vcvtnq_s32_f32(vfpacc4567);
        const int16x8_t vacc01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567), voutput_zero_point);
#else
        vacc0123 = vqsubq_s32(vreinterpretq_s32_f32(vaddq_f32(vfpacc0123, vmagic_bias)), vmagic_bias_less_output_zero_point);
        vacc4567 = vqsubq_s32(vreinterpretq_s32_f32(vaddq_f32(vfpacc4567, vmagic_bias)), vmagic_bias_less_output_zero_point);
        const int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
#endif
        uint8x8_t vout = vqmovun_s16(vacc01234567);
        vout = vmax_u8(vout, vget_low_u8(voutput_min));
        vout = vmin_u8(vout, vget_low_u8(voutput_max));

        if XNN_LIKELY(c >= 8) {
          vst1_u8(o, vout);
          o += 8;
          c_offset += 8;
          c -= 8;
        } else {
          if (c & 4) {
            vst1_lane_u32((void*) o, vreinterpret_u32_u8(vout), 0); o += 4;
            vout = vext_u8(vout, vout, 4);
          }
          if (c & 2) {
            vst1_lane_u16((void*) o, vreinterpret_u16_u8(vout), 0); o += 2;
            vout = vext_u8(vout, vout, 2);
          }
          if (c & 1) {
            vst1_lane_u8(o, vout, 0);
          }
          c = 0;
        }
      } while (c != 0);
    }

    input = (const uint8_t**) ((uintptr_t) input + input_increment);
    input_offset += input_pixel_stride;
    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
