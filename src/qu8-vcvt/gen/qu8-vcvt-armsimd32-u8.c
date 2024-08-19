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

#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/unaligned.h"
#include "xnnpack/vcvt.h"


void xnn_qu8_vcvt_ukernel__armsimd32_u8(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16x2_t vminus_input_zero_point = (uint16x2_t) broadcast2x_uint16(-params->scalar.input_zero_point);
  const int32_t vbias = ((int32_t) params->scalar.output_zero_point << 1) + INT32_C(1);
  const int32_t vmultiplier = (int32_t) params->scalar.multiplier << 9;
  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    const uint8x4_t vx0123 = (uint8x4_t) unaligned_indexed_load_u32(input, 0);
    const uint8x4_t vx4567 = (uint8x4_t) unaligned_indexed_load_u32(input, 1);
    input += 8;

    const uint16x2_t vx02 = __uxtab16(vminus_input_zero_point, vx0123);
    const uint16x2_t vx13 = __uxtab16(vminus_input_zero_point, __ror(vx0123, 8));
    const uint16x2_t vx46 = __uxtab16(vminus_input_zero_point, vx4567);
    const uint16x2_t vx57 = __uxtab16(vminus_input_zero_point, __ror(vx4567, 8));

    int32_t vacc0 = __smlawb(vmultiplier, vx02, vbias);
    int32_t vacc1 = __smlawb(vmultiplier, vx13, vbias);
    int32_t vacc2 = __smlawt(vmultiplier, vx02, vbias);
    int32_t vacc3 = __smlawt(vmultiplier, vx13, vbias);
    int32_t vacc4 = __smlawb(vmultiplier, vx46, vbias);
    int32_t vacc5 = __smlawb(vmultiplier, vx57, vbias);
    int32_t vacc6 = __smlawt(vmultiplier, vx46, vbias);
    int32_t vacc7 = __smlawt(vmultiplier, vx57, vbias);

    vacc0 = __usat(math_asr_s32(vacc0, 1), 8);
    vacc1 = __usat(math_asr_s32(vacc1, 1), 8);
    vacc2 = __usat(math_asr_s32(vacc2, 1), 8);
    vacc3 = __usat(math_asr_s32(vacc3, 1), 8);
    vacc4 = __usat(math_asr_s32(vacc4, 1), 8);
    vacc5 = __usat(math_asr_s32(vacc5, 1), 8);
    vacc6 = __usat(math_asr_s32(vacc6, 1), 8);
    vacc7 = __usat(math_asr_s32(vacc7, 1), 8);

    output[0] = (uint8_t) vacc0;
    output[1] = (uint8_t) vacc1;
    output[2] = (uint8_t) vacc2;
    output[3] = (uint8_t) vacc3;
    output[4] = (uint8_t) vacc4;
    output[5] = (uint8_t) vacc5;
    output[6] = (uint8_t) vacc6;
    output[7] = (uint8_t) vacc7;
    output += 8;
  }
  for (; batch >= 4 * sizeof(uint8_t); batch -= 4 * sizeof(uint8_t)) {
    const uint8x4_t vx0123 = (uint8x4_t) unaligned_load_u32(input);
    input += 4;

    const uint16x2_t vx02 = __uxtab16(vminus_input_zero_point, vx0123);
    const uint16x2_t vx13 = __uxtab16(vminus_input_zero_point, __ror(vx0123, 8));

    int32_t vacc0 = __smlawb(vmultiplier, vx02, vbias);
    int32_t vacc1 = __smlawb(vmultiplier, vx13, vbias);
    int32_t vacc2 = __smlawt(vmultiplier, vx02, vbias);
    int32_t vacc3 = __smlawt(vmultiplier, vx13, vbias);

    vacc0 = __usat(math_asr_s32(vacc0, 1), 8);
    vacc1 = __usat(math_asr_s32(vacc1, 1), 8);
    vacc2 = __usat(math_asr_s32(vacc2, 1), 8);
    vacc3 = __usat(math_asr_s32(vacc3, 1), 8);

    output[0] = (uint8_t) vacc0;
    output[1] = (uint8_t) vacc1;
    output[2] = (uint8_t) vacc2;
    output[3] = (uint8_t) vacc3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const uint8x4_t vx0123 = (uint8x4_t) unaligned_load_u32(input);

    const uint16x2_t vx02 = __uxtab16(vminus_input_zero_point, vx0123);
    const uint16x2_t vx13 = __uxtab16(vminus_input_zero_point, __ror(vx0123, 8));

    int32_t vacc0 = __smlawb(vmultiplier, vx02, vbias);
    int32_t vacc1 = __smlawb(vmultiplier, vx13, vbias);
    const int32_t vacc2 = __smlawt(vmultiplier, vx02, vbias);

    vacc0 = __usat(math_asr_s32(vacc0, 1), 8);
    vacc1 = __usat(math_asr_s32(vacc1, 1), 8);

    if (batch & (2 * sizeof(uint8_t))) {
      output[0] = (uint8_t) vacc0;
      output[1] = (uint8_t) vacc1;
      vacc0 = __usat(math_asr_s32(vacc2, 1), 8);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      output[0] = (uint8_t) vacc0;
    }
  }
}
