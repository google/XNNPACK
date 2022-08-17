// Auto-generated file. Do not edit!
//   Template: src/qs8-vlrelu/armsimd32.c.in
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
#include <xnnpack/vlrelu.h>


void xnn_qu8_vlrelu_ukernel__armsimd32_x4(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const union xnn_qu8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  const uint16x2_t vinput_zero_point = (uint16x2_t) params->armsimd32.input_zero_point;
  const int16x2_t vpositive_multiplier = (int16x2_t) params->armsimd32.positive_multiplier;
  const int16x2_t vnegative_multiplier = (int16x2_t) params->armsimd32.negative_multiplier;
  const int32_t vbias = params->armsimd32.bias;
  for (; n >= 4 * sizeof(uint8_t); n -= 4 * sizeof(uint8_t)) {
    const uint8x4_t vx0123 = (uint8x4_t) unaligned_load_u32(x);
    x += 4;

    uint16x2_t vx02 = __uxtb16(vx0123);
    uint16x2_t vx13 = __uxtb16(__ror(vx0123, 8));

    vx02 = __usub16(vinput_zero_point, vx02);
    const int16x2_t vmultiplier02 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);
    vx13 = __usub16(vinput_zero_point, vx13);
    const int16x2_t vmultiplier13 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);

    int32_t vacc0 = __smlabb(vmultiplier02, vx02, vbias);
    int32_t vacc1 = __smlabb(vmultiplier13, vx13, vbias);
    int32_t vacc2 = __smlatt(vmultiplier02, vx02, vbias);
    int32_t vacc3 = __smlatt(vmultiplier13, vx13, vbias);

    vacc0 = __usat(math_asr_s32(vacc0, 8), 8);
    vacc1 = __usat(math_asr_s32(vacc1, 8), 8);
    vacc2 = __usat(math_asr_s32(vacc2, 8), 8);
    vacc3 = __usat(math_asr_s32(vacc3, 8), 8);

    y[0] = (uint8_t) vacc0;
    y[1] = (uint8_t) vacc1;
    y[2] = (uint8_t) vacc2;
    y[3] = (uint8_t) vacc3;
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const uint8x4_t vx0123 = (uint8x4_t) unaligned_load_u32(x);

    uint16x2_t vx02 = __uxtb16(vx0123);
    uint16x2_t vx13 = __uxtb16(__ror(vx0123, 8));

    vx02 = __usub16(vinput_zero_point, vx02);
    const int16x2_t vmultiplier02 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);
    vx13 = __usub16(vinput_zero_point, vx13);
    const int16x2_t vmultiplier13 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);

    int32_t vacc0 = __smlabb(vmultiplier02, vx02, vbias);
    int32_t vacc1 = __smlabb(vmultiplier13, vx13, vbias);
    const int32_t vacc2 = __smlatt(vmultiplier02, vx02, vbias);

    vacc0 = __usat(math_asr_s32(vacc0, 8), 8);
    vacc1 = __usat(math_asr_s32(vacc1, 8), 8);

    if (n & (2 * sizeof(uint8_t))) {
      y[0] = (uint8_t) vacc0;
      y[1] = (uint8_t) vacc1;
      vacc0 = __usat(math_asr_s32(vacc2, 8), 8);
      y += 2;
    }
    if (n & (1 * sizeof(uint8_t))) {
      y[0] = (uint8_t) vacc0;
    }
  }
}
