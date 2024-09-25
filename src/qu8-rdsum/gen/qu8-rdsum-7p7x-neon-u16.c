// Auto-generated file. Do not edit!
//   Template: src/qu8-rdsum/neon.c.in
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


void xnn_qu8_rdsum_ukernel_7p7x__neon_u16(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint32_t* output,
    const struct xnn_qs8_rsum_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t input_increment = 7 * input_stride;
  for (; channels >= 16; channels -= 16) {
    const uint8_t* i0 = input;
    const uint8_t* i1 = (const uint8_t*) ((uintptr_t) input + 1 * input_stride);
    const uint8_t* i2 = (const uint8_t*) ((uintptr_t) input + 2 * input_stride);
    const uint8_t* i3 = (const uint8_t*) ((uintptr_t) input + 3 * input_stride);
    const uint8_t* i4 = (const uint8_t*) ((uintptr_t) input + 4 * input_stride);
    const uint8_t* i5 = (const uint8_t*) ((uintptr_t) input + 5 * input_stride);
    const uint8_t* i6 = (const uint8_t*) ((uintptr_t) input + 6 * input_stride);

    uint32x4_t vacc0 = vdupq_n_u32(0);
    uint32x4_t vacc4 = vdupq_n_u32(0);
    uint32x4_t vacc8 = vdupq_n_u32(0);
    uint32x4_t vacc12 = vdupq_n_u32(0);

    // 256 uint8s may be summed into an uint16 before overflowing
    // To prevent handling the tails of the inner 256 loop, we round 256 down to
    // the nearest integer multiple of ACCUMULATORS.
    int r = rows;
    while (r > 0) {
      uint16x8_t vacc16_4 = vmovq_n_u16(0);
      uint16x8_t vacc16_12 = vmovq_n_u16(0);
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
        uint8x16_t vin4;
        uint8x16_t vin12;
        vin4 = vld1q_u8(&i0[0]);
        vin12 = vld1q_u8(&i0[8]);
        vacc16_4 = vaddw_u8(vacc16_4, vget_low_u8(vin4));
        vacc16_12 = vaddw_u8(vacc16_12, vget_low_u8(vin12));
        vin4 = vld1q_u8(&i1[0]);
        vin12 = vld1q_u8(&i1[8]);
        vacc16_4 = vaddw_u8(vacc16_4, vget_low_u8(vin4));
        vacc16_12 = vaddw_u8(vacc16_12, vget_low_u8(vin12));
        vin4 = vld1q_u8(&i2[0]);
        vin12 = vld1q_u8(&i2[8]);
        vacc16_4 = vaddw_u8(vacc16_4, vget_low_u8(vin4));
        vacc16_12 = vaddw_u8(vacc16_12, vget_low_u8(vin12));
        vin4 = vld1q_u8(&i3[0]);
        vin12 = vld1q_u8(&i3[8]);
        vacc16_4 = vaddw_u8(vacc16_4, vget_low_u8(vin4));
        vacc16_12 = vaddw_u8(vacc16_12, vget_low_u8(vin12));
        vin4 = vld1q_u8(&i4[0]);
        vin12 = vld1q_u8(&i4[8]);
        vacc16_4 = vaddw_u8(vacc16_4, vget_low_u8(vin4));
        vacc16_12 = vaddw_u8(vacc16_12, vget_low_u8(vin12));
        vin4 = vld1q_u8(&i5[0]);
        vin12 = vld1q_u8(&i5[8]);
        vacc16_4 = vaddw_u8(vacc16_4, vget_low_u8(vin4));
        vacc16_12 = vaddw_u8(vacc16_12, vget_low_u8(vin12));
        vin4 = vld1q_u8(&i6[0]);
        vin12 = vld1q_u8(&i6[8]);
        vacc16_4 = vaddw_u8(vacc16_4, vget_low_u8(vin4));
        vacc16_12 = vaddw_u8(vacc16_12, vget_low_u8(vin12));
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_increment);
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_increment);
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_increment);
        i5 = (const uint8_t*) ((uintptr_t) i5 + input_increment);
        i6 = (const uint8_t*) ((uintptr_t) i6 + input_increment);
      }
      vacc0 = vaddw_u16(vacc0, vget_low_u16(vacc16_4));
      vacc4 = vaddw_u16(vacc4, vget_high_u16(vacc16_4));
      vacc8 = vaddw_u16(vacc8, vget_low_u16(vacc16_12));
      vacc12 = vaddw_u16(vacc12, vget_high_u16(vacc16_12));
      r = doz(r, 252);
    }

    const uint32_t* o = output;
    uint32x4_t vo0 = vld1q_u32(o); o += 4;
    uint32x4_t vo4 = vld1q_u32(o); o += 4;
    uint32x4_t vo8 = vld1q_u32(o); o += 4;
    uint32x4_t vo12 = vld1q_u32(o); o += 4;
    vacc0 = vaddq_u32(vo0, vacc0);
    vacc4 = vaddq_u32(vo4, vacc4);
    vacc8 = vaddq_u32(vo8, vacc8);
    vacc12 = vaddq_u32(vo12, vacc12);
    vst1q_u32(output, vacc0); output += 4;
    vst1q_u32(output, vacc4); output += 4;
    vst1q_u32(output, vacc8); output += 4;
    vst1q_u32(output, vacc12); output += 4;

    input = (const uint8_t*) ((uintptr_t) input + 16 * sizeof(uint8_t));
  }
  if (channels != 0) {
    assert(channels >= 1 && channels <= 15);

    input_increment = 7 * input_stride;
    // 256 uint8s may be summed into an uint16 before overflowing.
    do {
      int num_batches = floor((rows + 251) / 252);
      int r = rows;
      const uint8_t* i0 = input;
      const uint8_t* i1 = (const uint8_t*) ((uintptr_t) input + 1 * input_stride);
      const uint8_t* i2 = (const uint8_t*) ((uintptr_t) input + 2 * input_stride);
      const uint8_t* i3 = (const uint8_t*) ((uintptr_t) input + 3 * input_stride);
      const uint8_t* i4 = (const uint8_t*) ((uintptr_t) input + 4 * input_stride);
      const uint8_t* i5 = (const uint8_t*) ((uintptr_t) input + 5 * input_stride);
      const uint8_t* i6 = (const uint8_t*) ((uintptr_t) input + 6 * input_stride);

      uint32x4_t vacc0 = vdupq_n_u32(0);
      uint32x4_t vacc1 = vdupq_n_u32(0);

      for (; num_batches > 0; --num_batches) {
        uint16x8_t vacc16 = vmovq_n_u16(0);
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

          uint8x8_t vin0 = vld1_u8(&i0[0]);
          uint8x8_t vin1 = vld1_u8(&i1[0]);
          uint8x8_t vin2 = vld1_u8(&i2[0]);
          uint8x8_t vin3 = vld1_u8(&i3[0]);
          uint8x8_t vin4 = vld1_u8(&i4[0]);
          uint8x8_t vin5 = vld1_u8(&i5[0]);
          uint8x8_t vin6 = vld1_u8(&i6[0]);
          i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
          i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
          i2 = (const uint8_t*) ((uintptr_t) i2 + input_increment);
          i3 = (const uint8_t*) ((uintptr_t) i3 + input_increment);
          i4 = (const uint8_t*) ((uintptr_t) i4 + input_increment);
          i5 = (const uint8_t*) ((uintptr_t) i5 + input_increment);
          i6 = (const uint8_t*) ((uintptr_t) i6 + input_increment);
          vacc16 = vaddw_u8(vacc16, vin0);
          vacc16 = vaddw_u8(vacc16, vin1);
          vacc16 = vaddw_u8(vacc16, vin2);
          vacc16 = vaddw_u8(vacc16, vin3);
          vacc16 = vaddw_u8(vacc16, vin4);
          vacc16 = vaddw_u8(vacc16, vin5);
          vacc16 = vaddw_u8(vacc16, vin6);
        }
        vacc0 = vaddw_u16(vacc0, vget_low_u16(vacc16));
        vacc1 = vaddw_u16(vacc1, vget_high_u16(vacc16));
        r = doz(r, 252);
      }

      if XNN_LIKELY(channels >= 8) {
        uint32x4_t vo0 = vld1q_u32(output);
        uint32x4_t vo1 = vld1q_u32(output + 4);
        vo0 = vaddq_u32(vo0, vacc0);
        vo1 = vaddq_u32(vo1, vacc1);
        vst1q_u32(output, vo0); output += 4;
        vst1q_u32(output, vo1); output += 4;
        channels -= 8;
        input = (const uint8_t*) ((uintptr_t) input + 8 * sizeof(uint8_t));
      } else {
        if (channels & 4) {
          uint32x4_t vo = vld1q_u32(output);
          vo = vaddq_u32(vo, vacc0);
          vst1q_u32(output, vo); output += 4;
          vacc0 = vacc1;
        }
        if (channels & 2) {
          uint32x2_t vo = vld1_u32(output);
          vo = vadd_u32(vo, vget_low_u32(vacc0));
          vst1_u32(output, vo); output += 2;
          vacc0 = vextq_u32(vacc0, vacc0, 2);
        }
        if (channels & 1) {
          *output += vgetq_lane_u32(vacc0, 0);
        }
        channels = 0;
      }
    } while (channels != 0);
  }
}
