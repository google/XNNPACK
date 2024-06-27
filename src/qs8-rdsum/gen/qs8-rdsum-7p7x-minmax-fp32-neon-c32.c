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

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"
#include "xnnpack/math.h"


void xnn_qs8_rdsum_ukernel_7p7x__neon_c32(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int32_t* output,
    const union xnn_qs8_rsum_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t input_increment = 7 * input_stride;
  for (; channels >= 32; channels -= 32) {
    const int8_t* i0 = input;
    const int8_t* i1 = (const int8_t*) ((uintptr_t) input + 1 * input_stride);
    const int8_t* i2 = (const int8_t*) ((uintptr_t) input + 2 * input_stride);
    const int8_t* i3 = (const int8_t*) ((uintptr_t) input + 3 * input_stride);
    const int8_t* i4 = (const int8_t*) ((uintptr_t) input + 4 * input_stride);
    const int8_t* i5 = (const int8_t*) ((uintptr_t) input + 5 * input_stride);
    const int8_t* i6 = (const int8_t*) ((uintptr_t) input + 6 * input_stride);

    int32x4_t vacc0123 = vdupq_n_s32(0);
    int32x4_t vacc4567 = vdupq_n_s32(0);
    int32x4_t vacc89AB = vdupq_n_s32(0);
    int32x4_t vaccCDEF = vdupq_n_s32(0);
    int32x4_t vaccGHIJ = vdupq_n_s32(0);
    int32x4_t vaccKLMN = vdupq_n_s32(0);
    int32x4_t vaccOPQR = vdupq_n_s32(0);
    int32x4_t vaccSTUV = vdupq_n_s32(0);

    // 256 int8s may be summed into an int16 before overflowing
    // To prevent handling the tails of the inner 256 loop, we round 256 down to
    // the nearest integer multiple of ACCUMULATORS.
    int r = rows;
    while (r > 0) {
      int16x8_t vacc16_01234567 = vmovq_n_s16(0);
      int16x8_t vacc16_89ABCDEF = vmovq_n_s16(0);
      int16x8_t vacc16_GHIJKLMN = vmovq_n_s16(0);
      int16x8_t vacc16_OPQRSTUV = vmovq_n_s16(0);
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
        int8x8_t vinGHIJKLMN;
        int8x8_t vinOPQRSTUV;
        vin01234567 = vld1_s8(&i0[0]);
        vin89ABCDEF = vld1_s8(&i0[8]);
        vinGHIJKLMN = vld1_s8(&i0[16]);
        vinOPQRSTUV = vld1_s8(&i0[24]);
        vacc16_01234567 = vaddw_s8(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = vaddw_s8(vacc16_89ABCDEF, vin89ABCDEF);
        vacc16_GHIJKLMN = vaddw_s8(vacc16_GHIJKLMN, vinGHIJKLMN);
        vacc16_OPQRSTUV = vaddw_s8(vacc16_OPQRSTUV, vinOPQRSTUV);
        vin01234567 = vld1_s8(&i1[0]);
        vin89ABCDEF = vld1_s8(&i1[8]);
        vinGHIJKLMN = vld1_s8(&i1[16]);
        vinOPQRSTUV = vld1_s8(&i1[24]);
        vacc16_01234567 = vaddw_s8(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = vaddw_s8(vacc16_89ABCDEF, vin89ABCDEF);
        vacc16_GHIJKLMN = vaddw_s8(vacc16_GHIJKLMN, vinGHIJKLMN);
        vacc16_OPQRSTUV = vaddw_s8(vacc16_OPQRSTUV, vinOPQRSTUV);
        vin01234567 = vld1_s8(&i2[0]);
        vin89ABCDEF = vld1_s8(&i2[8]);
        vinGHIJKLMN = vld1_s8(&i2[16]);
        vinOPQRSTUV = vld1_s8(&i2[24]);
        vacc16_01234567 = vaddw_s8(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = vaddw_s8(vacc16_89ABCDEF, vin89ABCDEF);
        vacc16_GHIJKLMN = vaddw_s8(vacc16_GHIJKLMN, vinGHIJKLMN);
        vacc16_OPQRSTUV = vaddw_s8(vacc16_OPQRSTUV, vinOPQRSTUV);
        vin01234567 = vld1_s8(&i3[0]);
        vin89ABCDEF = vld1_s8(&i3[8]);
        vinGHIJKLMN = vld1_s8(&i3[16]);
        vinOPQRSTUV = vld1_s8(&i3[24]);
        vacc16_01234567 = vaddw_s8(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = vaddw_s8(vacc16_89ABCDEF, vin89ABCDEF);
        vacc16_GHIJKLMN = vaddw_s8(vacc16_GHIJKLMN, vinGHIJKLMN);
        vacc16_OPQRSTUV = vaddw_s8(vacc16_OPQRSTUV, vinOPQRSTUV);
        vin01234567 = vld1_s8(&i4[0]);
        vin89ABCDEF = vld1_s8(&i4[8]);
        vinGHIJKLMN = vld1_s8(&i4[16]);
        vinOPQRSTUV = vld1_s8(&i4[24]);
        vacc16_01234567 = vaddw_s8(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = vaddw_s8(vacc16_89ABCDEF, vin89ABCDEF);
        vacc16_GHIJKLMN = vaddw_s8(vacc16_GHIJKLMN, vinGHIJKLMN);
        vacc16_OPQRSTUV = vaddw_s8(vacc16_OPQRSTUV, vinOPQRSTUV);
        vin01234567 = vld1_s8(&i5[0]);
        vin89ABCDEF = vld1_s8(&i5[8]);
        vinGHIJKLMN = vld1_s8(&i5[16]);
        vinOPQRSTUV = vld1_s8(&i5[24]);
        vacc16_01234567 = vaddw_s8(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = vaddw_s8(vacc16_89ABCDEF, vin89ABCDEF);
        vacc16_GHIJKLMN = vaddw_s8(vacc16_GHIJKLMN, vinGHIJKLMN);
        vacc16_OPQRSTUV = vaddw_s8(vacc16_OPQRSTUV, vinOPQRSTUV);
        vin01234567 = vld1_s8(&i6[0]);
        vin89ABCDEF = vld1_s8(&i6[8]);
        vinGHIJKLMN = vld1_s8(&i6[16]);
        vinOPQRSTUV = vld1_s8(&i6[24]);
        vacc16_01234567 = vaddw_s8(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = vaddw_s8(vacc16_89ABCDEF, vin89ABCDEF);
        vacc16_GHIJKLMN = vaddw_s8(vacc16_GHIJKLMN, vinGHIJKLMN);
        vacc16_OPQRSTUV = vaddw_s8(vacc16_OPQRSTUV, vinOPQRSTUV);
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
      vaccGHIJ = vaddw_s16(vaccGHIJ, vget_low_s16(vacc16_GHIJKLMN));
      vaccKLMN = vaddw_s16(vaccKLMN, vget_high_s16(vacc16_GHIJKLMN));
      vaccOPQR = vaddw_s16(vaccOPQR, vget_low_s16(vacc16_OPQRSTUV));
      vaccSTUV = vaddw_s16(vaccSTUV, vget_high_s16(vacc16_OPQRSTUV));
      r = doz(r, 252);
    }

    const int32_t* o = output;
    int32x4_t vo0123 = vld1q_s32(o); o += 4;
    int32x4_t vo4567 = vld1q_s32(o); o += 4;
    int32x4_t vo89AB = vld1q_s32(o); o += 4;
    int32x4_t voCDEF = vld1q_s32(o); o += 4;
    int32x4_t voGHIJ = vld1q_s32(o); o += 4;
    int32x4_t voKLMN = vld1q_s32(o); o += 4;
    int32x4_t voOPQR = vld1q_s32(o); o += 4;
    int32x4_t voSTUV = vld1q_s32(o); o += 4;
    vacc0123 = vaddq_s32(vo0123, vacc0123);
    vacc4567 = vaddq_s32(vo4567, vacc4567);
    vacc89AB = vaddq_s32(vo89AB, vacc89AB);
    vaccCDEF = vaddq_s32(voCDEF, vaccCDEF);
    vaccGHIJ = vaddq_s32(voGHIJ, vaccGHIJ);
    vaccKLMN = vaddq_s32(voKLMN, vaccKLMN);
    vaccOPQR = vaddq_s32(voOPQR, vaccOPQR);
    vaccSTUV = vaddq_s32(voSTUV, vaccSTUV);
    vst1q_s32(output, vacc0123); output += 4;
    vst1q_s32(output, vacc4567); output += 4;
    vst1q_s32(output, vacc89AB); output += 4;
    vst1q_s32(output, vaccCDEF); output += 4;
    vst1q_s32(output, vaccGHIJ); output += 4;
    vst1q_s32(output, vaccKLMN); output += 4;
    vst1q_s32(output, vaccOPQR); output += 4;
    vst1q_s32(output, vaccSTUV); output += 4;

    input = (const int8_t*) ((uintptr_t) input + 32 * sizeof(int8_t));
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

      int32x4_t vacc0 = vdupq_n_s32(0);
      int32x4_t vacc1 = vdupq_n_s32(0);

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

      if XNN_LIKELY(channels >= 8) {
        int32x4_t vo0 = vld1q_s32(output);
        int32x4_t vo1 = vld1q_s32(output + 4);
        vo0 = vaddq_s32(vo0, vacc0);
        vo1 = vaddq_s32(vo1, vacc1);
        vst1q_s32(output, vo0); output += 4;
        vst1q_s32(output, vo1); output += 4;
        channels -= 8;
        input = (const int8_t*) ((uintptr_t) input + 8 * sizeof(int8_t));
      } else {
        if (channels & 4) {
          int32x4_t vo = vld1q_s32(output);
          vo = vaddq_s32(vo, vacc0);
          vst1q_s32(output, vo); output += 4;
          vacc0 = vacc1;
        }
        if (channels & 2) {
          int32x2_t vo = vld1_s32(output);
          vo = vadd_s32(vo, vget_low_s32(vacc0));
          vst1_s32(output, vo); output += 2;
          vacc0 = vextq_s32(vacc0, vacc0, 2);
        }
        if (channels & 1) {
          *output += vgetq_lane_s32(vacc0, 0);
        }
        channels = 0;
      }
    } while (channels != 0);
  }
}
