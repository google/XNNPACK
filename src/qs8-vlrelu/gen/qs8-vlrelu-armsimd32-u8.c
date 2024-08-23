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

#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/unaligned.h"
#include "xnnpack/vlrelu.h"


void xnn_qs8_vlrelu_ukernel__armsimd32_u8(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int16x2_t vinput_zero_point = (int16x2_t) broadcast2x_uint16(params->scalar.input_zero_point);
  const int16x2_t vpositive_multiplier = (int16x2_t) broadcast2x_uint16(-params->scalar.positive_multiplier);
  const int16x2_t vnegative_multiplier = (int16x2_t) broadcast2x_uint16(-params->scalar.negative_multiplier);
  const int32_t vbias = (params->scalar.output_zero_point << 8) + 0x80;
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    const int8x4_t vx0123 = (int8x4_t) unaligned_indexed_load_u32(input, 0);
    const int8x4_t vx4567 = (int8x4_t) unaligned_indexed_load_u32(input, 1);
    input += 8;

    int16x2_t vx02 = __sxtb16(vx0123);
    int16x2_t vx13 = __sxtb16(__ror(vx0123, 8));
    int16x2_t vx46 = __sxtb16(vx4567);
    int16x2_t vx57 = __sxtb16(__ror(vx4567, 8));

    vx02 = __ssub16(vinput_zero_point, vx02);
    const int16x2_t vmultiplier02 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);
    vx13 = __ssub16(vinput_zero_point, vx13);
    const int16x2_t vmultiplier13 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);
    vx46 = __ssub16(vinput_zero_point, vx46);
    const int16x2_t vmultiplier46 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);
    vx57 = __ssub16(vinput_zero_point, vx57);
    const int16x2_t vmultiplier57 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);

    int32_t vacc0 = __smlabb(vmultiplier02, vx02, vbias);
    int32_t vacc1 = __smlabb(vmultiplier13, vx13, vbias);
    int32_t vacc2 = __smlatt(vmultiplier02, vx02, vbias);
    int32_t vacc3 = __smlatt(vmultiplier13, vx13, vbias);
    int32_t vacc4 = __smlabb(vmultiplier46, vx46, vbias);
    int32_t vacc5 = __smlabb(vmultiplier57, vx57, vbias);
    int32_t vacc6 = __smlatt(vmultiplier46, vx46, vbias);
    int32_t vacc7 = __smlatt(vmultiplier57, vx57, vbias);

    vacc0 = __ssat(math_asr_s32(vacc0, 8), 8);
    vacc1 = __ssat(math_asr_s32(vacc1, 8), 8);
    vacc2 = __ssat(math_asr_s32(vacc2, 8), 8);
    vacc3 = __ssat(math_asr_s32(vacc3, 8), 8);
    vacc4 = __ssat(math_asr_s32(vacc4, 8), 8);
    vacc5 = __ssat(math_asr_s32(vacc5, 8), 8);
    vacc6 = __ssat(math_asr_s32(vacc6, 8), 8);
    vacc7 = __ssat(math_asr_s32(vacc7, 8), 8);

    output[0] = (int8_t) vacc0;
    output[1] = (int8_t) vacc1;
    output[2] = (int8_t) vacc2;
    output[3] = (int8_t) vacc3;
    output[4] = (int8_t) vacc4;
    output[5] = (int8_t) vacc5;
    output[6] = (int8_t) vacc6;
    output[7] = (int8_t) vacc7;
    output += 8;
  }
  for (; batch >= 4 * sizeof(int8_t); batch -= 4 * sizeof(int8_t)) {
    const int8x4_t vx0123 = (int8x4_t) unaligned_load_u32(input);
    input += 4;

    int16x2_t vx02 = __sxtb16(vx0123);
    int16x2_t vx13 = __sxtb16(__ror(vx0123, 8));

    vx02 = __ssub16(vinput_zero_point, vx02);
    const int16x2_t vmultiplier02 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);
    vx13 = __ssub16(vinput_zero_point, vx13);
    const int16x2_t vmultiplier13 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);

    int32_t vacc0 = __smlabb(vmultiplier02, vx02, vbias);
    int32_t vacc1 = __smlabb(vmultiplier13, vx13, vbias);
    int32_t vacc2 = __smlatt(vmultiplier02, vx02, vbias);
    int32_t vacc3 = __smlatt(vmultiplier13, vx13, vbias);

    vacc0 = __ssat(math_asr_s32(vacc0, 8), 8);
    vacc1 = __ssat(math_asr_s32(vacc1, 8), 8);
    vacc2 = __ssat(math_asr_s32(vacc2, 8), 8);
    vacc3 = __ssat(math_asr_s32(vacc3, 8), 8);

    output[0] = (int8_t) vacc0;
    output[1] = (int8_t) vacc1;
    output[2] = (int8_t) vacc2;
    output[3] = (int8_t) vacc3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const int8x4_t vx0123 = (int8x4_t) unaligned_load_u32(input);

    int16x2_t vx02 = __sxtb16(vx0123);
    int16x2_t vx13 = __sxtb16(__ror(vx0123, 8));

    vx02 = __ssub16(vinput_zero_point, vx02);
    const int16x2_t vmultiplier02 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);
    vx13 = __ssub16(vinput_zero_point, vx13);
    const int16x2_t vmultiplier13 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);

    int32_t vacc0 = __smlabb(vmultiplier02, vx02, vbias);
    int32_t vacc1 = __smlabb(vmultiplier13, vx13, vbias);
    const int32_t vacc2 = __smlatt(vmultiplier02, vx02, vbias);

    vacc0 = __ssat(math_asr_s32(vacc0, 8), 8);
    vacc1 = __ssat(math_asr_s32(vacc1, 8), 8);

    if (batch & (2 * sizeof(int8_t))) {
      output[0] = (int8_t) vacc0;
      output[1] = (int8_t) vacc1;
      vacc0 = __ssat(math_asr_s32(vacc2, 8), 8);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      output[0] = (int8_t) vacc0;
    }
  }
}
