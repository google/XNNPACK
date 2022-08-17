// Auto-generated file. Do not edit!
//   Template: src/qs8-vcvt/armsimd32.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_acle.h>

#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>
#include <xnnpack/unaligned.h>
#include <xnnpack/vcvt.h>


void xnn_qs8_vcvt_ukernel__armsimd32_x8(
    size_t n,
    const int8_t* x,
    int8_t* y,
    const union xnn_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  const int16x2_t vminus_input_zero_point = (int16x2_t) params->armsimd32.minus_input_zero_point;
  const int32_t vbias = params->armsimd32.bias;
  const int32_t vmultiplier = params->armsimd32.multiplier;
  for (; n >= 8 * sizeof(int8_t); n -= 8 * sizeof(int8_t)) {
    const int8x4_t vx0123 = (int8x4_t) unaligned_indexed_load_u32(x, 0);
    const int8x4_t vx4567 = (int8x4_t) unaligned_indexed_load_u32(x, 1);
    x += 8;

    const int16x2_t vx02 = __sxtab16(vminus_input_zero_point, vx0123);
    const int16x2_t vx13 = __sxtab16(vminus_input_zero_point, __ror(vx0123, 8));
    const int16x2_t vx46 = __sxtab16(vminus_input_zero_point, vx4567);
    const int16x2_t vx57 = __sxtab16(vminus_input_zero_point, __ror(vx4567, 8));

    int32_t vacc0 = __smlawb(vmultiplier, vx02, vbias);
    int32_t vacc1 = __smlawb(vmultiplier, vx13, vbias);
    int32_t vacc2 = __smlawt(vmultiplier, vx02, vbias);
    int32_t vacc3 = __smlawt(vmultiplier, vx13, vbias);
    int32_t vacc4 = __smlawb(vmultiplier, vx46, vbias);
    int32_t vacc5 = __smlawb(vmultiplier, vx57, vbias);
    int32_t vacc6 = __smlawt(vmultiplier, vx46, vbias);
    int32_t vacc7 = __smlawt(vmultiplier, vx57, vbias);

    vacc0 = __ssat(math_asr_s32(vacc0, 1), 8);
    vacc1 = __ssat(math_asr_s32(vacc1, 1), 8);
    vacc2 = __ssat(math_asr_s32(vacc2, 1), 8);
    vacc3 = __ssat(math_asr_s32(vacc3, 1), 8);
    vacc4 = __ssat(math_asr_s32(vacc4, 1), 8);
    vacc5 = __ssat(math_asr_s32(vacc5, 1), 8);
    vacc6 = __ssat(math_asr_s32(vacc6, 1), 8);
    vacc7 = __ssat(math_asr_s32(vacc7, 1), 8);

    y[0] = (int8_t) vacc0;
    y[1] = (int8_t) vacc1;
    y[2] = (int8_t) vacc2;
    y[3] = (int8_t) vacc3;
    y[4] = (int8_t) vacc4;
    y[5] = (int8_t) vacc5;
    y[6] = (int8_t) vacc6;
    y[7] = (int8_t) vacc7;
    y += 8;
  }
  for (; n >= 4 * sizeof(int8_t); n -= 4 * sizeof(int8_t)) {
    const int8x4_t vx0123 = (int8x4_t) unaligned_load_u32(x);
    x += 4;

    const int16x2_t vx02 = __sxtab16(vminus_input_zero_point, vx0123);
    const int16x2_t vx13 = __sxtab16(vminus_input_zero_point, __ror(vx0123, 8));

    int32_t vacc0 = __smlawb(vmultiplier, vx02, vbias);
    int32_t vacc1 = __smlawb(vmultiplier, vx13, vbias);
    int32_t vacc2 = __smlawt(vmultiplier, vx02, vbias);
    int32_t vacc3 = __smlawt(vmultiplier, vx13, vbias);

    vacc0 = __ssat(math_asr_s32(vacc0, 1), 8);
    vacc1 = __ssat(math_asr_s32(vacc1, 1), 8);
    vacc2 = __ssat(math_asr_s32(vacc2, 1), 8);
    vacc3 = __ssat(math_asr_s32(vacc3, 1), 8);

    y[0] = (int8_t) vacc0;
    y[1] = (int8_t) vacc1;
    y[2] = (int8_t) vacc2;
    y[3] = (int8_t) vacc3;
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const int8x4_t vx0123 = (int8x4_t) unaligned_load_u32(x);

    const int16x2_t vx02 = __sxtab16(vminus_input_zero_point, vx0123);
    const int16x2_t vx13 = __sxtab16(vminus_input_zero_point, __ror(vx0123, 8));

    int32_t vacc0 = __smlawb(vmultiplier, vx02, vbias);
    int32_t vacc1 = __smlawb(vmultiplier, vx13, vbias);
    const int32_t vacc2 = __smlawt(vmultiplier, vx02, vbias);

    vacc0 = __ssat(math_asr_s32(vacc0, 1), 8);
    vacc1 = __ssat(math_asr_s32(vacc1, 1), 8);

    if (n & (2 * sizeof(int8_t))) {
      y[0] = (int8_t) vacc0;
      y[1] = (int8_t) vacc1;
      vacc0 = __ssat(math_asr_s32(vacc2, 1), 8);
      y += 2;
    }
    if (n & (1 * sizeof(int8_t))) {
      y[0] = (int8_t) vacc0;
    }
  }
}
