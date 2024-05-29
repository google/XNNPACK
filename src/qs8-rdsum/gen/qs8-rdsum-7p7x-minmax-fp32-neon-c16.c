// Auto-generated file. Do not edit!
//   Template: src/qs8-rdsum/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/reduce.h>
#include <xnnpack/math.h>


void xnn_qs8_rdsum_minmax_fp32_ukernel_7p7x__neon_c16(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t input_increment = 7 * input_stride;
  const int32x4_t vinit_bias = vld1q_dup_s32(&params->fp32_neon.init_bias);
  for (; channels >= 16; channels -= 16) {
    const int8_t* i0 = input;
    const int8_t* i1 = (const int8_t*) ((uintptr_t) input + 1 * input_stride);
    const int8_t* i2 = (const int8_t*) ((uintptr_t) input + 2 * input_stride);
    const int8_t* i3 = (const int8_t*) ((uintptr_t) input + 3 * input_stride);
    const int8_t* i4 = (const int8_t*) ((uintptr_t) input + 4 * input_stride);
    const int8_t* i5 = (const int8_t*) ((uintptr_t) input + 5 * input_stride);
    const int8_t* i6 = (const int8_t*) ((uintptr_t) input + 6 * input_stride);

    int32x4_t vacc0123 = vinit_bias;
    int32x4_t vacc4567 = vinit_bias;
    int32x4_t vacc89AB = vinit_bias;
    int32x4_t vaccCDEF = vinit_bias;

    // 256 int8s may be summed into an int16 before overflowing
    // To prevent handling the tails of the inner 256 loop, we round 256 down to
    // the nearest integer multiple of ACCUMULATORS.
    int num_batches = floor((rows + 251) / 252);
    int r = rows;
    for (; num_batches > 0; --num_batches) {
      int16x8_t vacc16_01234567 = vmovq_n_s16(0);
      int16x8_t vacc16_89ABCDEF = vmovq_n_s16(0);
      for (int current_batch = min(r, 252); current_batch > 0; current_batch -= 7) {
        if XNN_UNPREDICTABLE(current_batch < 2) {
          i1 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch <= 2) {
          i2 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch < 4) {
          i3 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch <= 4) {
          i4 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch < 6) {
          i5 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch <= 6) {
          i6 = zero;
        }
        int8x8_t vin01234567;
        int8x8_t vin89ABCDEF;
        vin01234567 = vld1_s8(&i0[0]);
        vin89ABCDEF = vld1_s8(&i0[8]);
        vacc16_01234567 = vaddw_s8(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = vaddw_s8(vacc16_89ABCDEF, vin89ABCDEF);
        vin01234567 = vld1_s8(&i1[0]);
        vin89ABCDEF = vld1_s8(&i1[8]);
        vacc16_01234567 = vaddw_s8(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = vaddw_s8(vacc16_89ABCDEF, vin89ABCDEF);
        vin01234567 = vld1_s8(&i2[0]);
        vin89ABCDEF = vld1_s8(&i2[8]);
        vacc16_01234567 = vaddw_s8(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = vaddw_s8(vacc16_89ABCDEF, vin89ABCDEF);
        vin01234567 = vld1_s8(&i3[0]);
        vin89ABCDEF = vld1_s8(&i3[8]);
        vacc16_01234567 = vaddw_s8(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = vaddw_s8(vacc16_89ABCDEF, vin89ABCDEF);
        vin01234567 = vld1_s8(&i4[0]);
        vin89ABCDEF = vld1_s8(&i4[8]);
        vacc16_01234567 = vaddw_s8(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = vaddw_s8(vacc16_89ABCDEF, vin89ABCDEF);
        vin01234567 = vld1_s8(&i5[0]);
        vin89ABCDEF = vld1_s8(&i5[8]);
        vacc16_01234567 = vaddw_s8(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = vaddw_s8(vacc16_89ABCDEF, vin89ABCDEF);
        vin01234567 = vld1_s8(&i6[0]);
        vin89ABCDEF = vld1_s8(&i6[8]);
        vacc16_01234567 = vaddw_s8(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = vaddw_s8(vacc16_89ABCDEF, vin89ABCDEF);
        i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
        i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
        i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
        i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
        i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
        i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
        i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
      }
      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vacc16_01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vacc16_01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vacc16_89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vacc16_89ABCDEF));
      r = doz(r, 252);
    }

    const float32x4_t vscale = vld1q_dup_f32(&params->fp32_neon.scale);
    const float32x4_t vmagic_bias = vld1q_dup_f32(&params->fp32_neon.magic_bias);
    const int32x4_t vmagic_bias_less_output_zero_point = vld1q_dup_s32(&params->fp32_neon.magic_bias_less_output_zero_point);
    const int8x8_t voutput_min = vld1_dup_s8(&params->fp32_neon.output_min);
    const int8x8_t voutput_max = vld1_dup_s8(&params->fp32_neon.output_max);

    float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
    float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);
    float32x4_t vfpacc89AB = vcvtq_f32_s32(vacc89AB);
    float32x4_t vfpaccCDEF = vcvtq_f32_s32(vaccCDEF);
    vfpacc0123 = vmulq_f32(vfpacc0123, vscale);
    vfpacc4567 = vmulq_f32(vfpacc4567, vscale);
    vfpacc89AB = vmulq_f32(vfpacc89AB, vscale);
    vfpaccCDEF = vmulq_f32(vfpaccCDEF, vscale);

    vacc0123 = vreinterpretq_s32_f32(vaddq_f32(vfpacc0123, vmagic_bias));
    vacc4567 = vreinterpretq_s32_f32(vaddq_f32(vfpacc4567, vmagic_bias));
    vacc89AB = vreinterpretq_s32_f32(vaddq_f32(vfpacc89AB, vmagic_bias));
    vaccCDEF = vreinterpretq_s32_f32(vaddq_f32(vfpaccCDEF, vmagic_bias));
    vacc0123 = vqsubq_s32(vacc0123, vmagic_bias_less_output_zero_point);
    vacc4567 = vqsubq_s32(vacc4567, vmagic_bias_less_output_zero_point);
    vacc89AB = vqsubq_s32(vacc89AB, vmagic_bias_less_output_zero_point);
    vaccCDEF = vqsubq_s32(vaccCDEF, vmagic_bias_less_output_zero_point);

    #if XNN_ARCH_ARM64
      int16x8_t vacc16_01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);
      int16x8_t vacc16_89ABCDEF = vqmovn_high_s32(vqmovn_s32(vacc89AB), vaccCDEF);
    #else  // !XNN_ARCH_ARM64
      int16x8_t vacc16_01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
      int16x8_t vacc16_89ABCDEF = vcombine_s16(vqmovn_s32(vacc89AB), vqmovn_s32(vaccCDEF));
    #endif  // !XNN_ARCH_ARM64
    int8x8_t vacc8_01234567 = vqmovn_s16(vacc16_01234567);
    int8x8_t vacc8_89ABCDEF = vqmovn_s16(vacc16_89ABCDEF);
    vacc8_01234567 = vmin_s8(vacc8_01234567, voutput_max);
    vacc8_89ABCDEF = vmin_s8(vacc8_89ABCDEF, voutput_max);
    vacc8_01234567 = vmax_s8(vacc8_01234567, voutput_min);
    vacc8_89ABCDEF = vmax_s8(vacc8_89ABCDEF, voutput_min);

    const int8_t* o = output;
    int8x8_t vo01234567 = vld1_s8(o); o += 8;
    int8x8_t vo89ABCDEF = vld1_s8(o); o += 8;
    vacc8_01234567 = vadd_s8(vo01234567, vacc8_01234567);
    vacc8_89ABCDEF = vadd_s8(vo89ABCDEF, vacc8_89ABCDEF);
    vst1_s8(output, vacc8_01234567); output += 8;
    vst1_s8(output, vacc8_89ABCDEF); output += 8;

    input = (const int8_t*) ((uintptr_t) input + 16 * sizeof(int8_t));
  }
  if (channels != 0) {
    input_increment = 7 * input_stride;
    // 256 int8s may be summed into an int16 before overflowing.
    do {
      int num_batches = floor((rows + 251) / 252);
      int r = rows;
      const int8_t* i0 = input;
      const int8_t* i1 = (const int8_t*) ((uintptr_t) input + 1 * input_stride);
      const int8_t* i2 = (const int8_t*) ((uintptr_t) input + 2 * input_stride);
      const int8_t* i3 = (const int8_t*) ((uintptr_t) input + 3 * input_stride);
      const int8_t* i4 = (const int8_t*) ((uintptr_t) input + 4 * input_stride);
      const int8_t* i5 = (const int8_t*) ((uintptr_t) input + 5 * input_stride);
      const int8_t* i6 = (const int8_t*) ((uintptr_t) input + 6 * input_stride);

      int32x4_t vacc0 = vinit_bias;
      int32x4_t vacc1 = vinit_bias;

      for (; num_batches > 0; --num_batches) {
        int16x8_t vacc16 = vmovq_n_s16(0);
        for (int current_batch = min(r, 252); current_batch > 0; current_batch -= 7) {
          if XNN_UNPREDICTABLE(current_batch < 2) {
            i1 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch <= 2) {
            i2 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch < 4) {
            i3 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch <= 4) {
            i4 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch < 6) {
            i5 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch <= 6) {
            i6 = zero;
          }

          int8x8_t vin0 = vld1_s8(&i0[0]);
          int8x8_t vin1 = vld1_s8(&i1[0]);
          int8x8_t vin2 = vld1_s8(&i2[0]);
          int8x8_t vin3 = vld1_s8(&i3[0]);
          int8x8_t vin4 = vld1_s8(&i4[0]);
          int8x8_t vin5 = vld1_s8(&i5[0]);
          int8x8_t vin6 = vld1_s8(&i6[0]);
          vacc16 = vaddw_s8(vacc16, vin0);
          vacc16 = vaddw_s8(vacc16, vin1);
          vacc16 = vaddw_s8(vacc16, vin2);
          vacc16 = vaddw_s8(vacc16, vin3);
          vacc16 = vaddw_s8(vacc16, vin4);
          vacc16 = vaddw_s8(vacc16, vin5);
          vacc16 = vaddw_s8(vacc16, vin6);
          i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
          i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
          i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
          i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
          i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
          i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
          i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
        }
        vacc0 = vaddw_s16(vacc0, vget_low_s16(vacc16));
        vacc1 = vaddw_s16(vacc1, vget_high_s16(vacc16));
        r = doz(r, 252);
      }

      const float32x4_t vscale = vld1q_dup_f32(&params->fp32_neon.scale);
      const float32x4_t vmagic_bias = vld1q_dup_f32(&params->fp32_neon.magic_bias);
      const int32x4_t vmagic_bias_less_output_zero_point = vld1q_dup_s32(&params->fp32_neon.magic_bias_less_output_zero_point);
      const int8x8_t voutput_min = vld1_dup_s8(&params->fp32_neon.output_min);
      const int8x8_t voutput_max = vld1_dup_s8(&params->fp32_neon.output_max);

      float32x4_t vfpacc0 = vcvtq_f32_s32(vacc0);
      float32x4_t vfpacc1 = vcvtq_f32_s32(vacc1);
      vfpacc0 = vmulq_f32(vfpacc0, vscale);
      vfpacc1 = vmulq_f32(vfpacc1, vscale);
      vacc0 = vreinterpretq_s32_f32(vaddq_f32(vfpacc0, vmagic_bias));
      vacc1 = vreinterpretq_s32_f32(vaddq_f32(vfpacc1, vmagic_bias));
      vacc0 = vqsubq_s32(vacc0, vmagic_bias_less_output_zero_point);
      vacc1 = vqsubq_s32(vacc1, vmagic_bias_less_output_zero_point);
      #if XNN_ARCH_ARM64
        int16x8_t vacc16 = vqmovn_high_s32(vqmovn_s32(vacc0), vacc1);
      #else  // !XNN_ARCH_ARM64
        int16x8_t vacc16 = vcombine_s16(vqmovn_s32(vacc0), vqmovn_s32(vacc1));
      #endif  // !XNN_ARCH_ARM64

      int8x8_t vacc8 = vqmovn_s16(vacc16);
      vacc8 = vmin_s8(vacc8, voutput_max);
      vacc8 = vmax_s8(vacc8, voutput_min);

      if XNN_LIKELY(channels >= 8) {
        int8x8_t vo = vld1_s8(output);
        vacc8 = vadd_s8(vo, vacc8);
        vst1_s8(output, vacc8); output += 8;
        channels -= 8;
        input = (const int8_t*) ((uintptr_t) input + 8 * sizeof(int8_t));
      } else {
        if (channels & 4) {
          int8x8_t vo = vreinterpret_s8_s32(vld1_s32((const int32_t*) output));
          vacc8 = vadd_s8(vo, vacc8);
          vst1_lane_s32((int32_t*) output, vreinterpret_s32_s8(vacc8), 0); output += 4;
          vacc8 = vext_s8(vacc8, vacc8, 4);
        }
        if (channels & 2) {
          int8x8_t vo = vreinterpret_s8_s16(vld1_s16((const int16_t*) output));
          vacc8 = vadd_s8(vo, vacc8);
          vst1_lane_s16((int32_t*) output, vreinterpret_s16_s8(vacc8), 0); output += 2;
          vacc8 = vext_s8(vacc8, vacc8, 2);
        }
        if (channels & 1) {
          *output += vget_lane_s8(vacc8, 0);
        }
        channels = 0;
      }
    } while (channels != 0);
  }
}
