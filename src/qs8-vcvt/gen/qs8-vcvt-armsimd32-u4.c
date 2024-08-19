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


void xnn_qs8_vcvt_ukernel__armsimd32_u4(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int16x2_t vminus_input_zero_point = (int16x2_t) broadcast2x_uint16(-params->scalar.input_zero_point);
  const int32_t vbias = ((int32_t) params->scalar.output_zero_point << 1) + INT32_C(1);
  const int32_t vmultiplier = (int32_t) params->scalar.multiplier << 9;
  for (; batch >= 4 * sizeof(int8_t); batch -= 4 * sizeof(int8_t)) {
    const int8x4_t vx0123 = (int8x4_t) unaligned_load_u32(input);
    input += 4;

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

    output[0] = (int8_t) vacc0;
    output[1] = (int8_t) vacc1;
    output[2] = (int8_t) vacc2;
    output[3] = (int8_t) vacc3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const int8x4_t vx0123 = (int8x4_t) unaligned_load_u32(input);

    const int16x2_t vx02 = __sxtab16(vminus_input_zero_point, vx0123);
    const int16x2_t vx13 = __sxtab16(vminus_input_zero_point, __ror(vx0123, 8));

    int32_t vacc0 = __smlawb(vmultiplier, vx02, vbias);
    int32_t vacc1 = __smlawb(vmultiplier, vx13, vbias);
    const int32_t vacc2 = __smlawt(vmultiplier, vx02, vbias);

    vacc0 = __ssat(math_asr_s32(vacc0, 1), 8);
    vacc1 = __ssat(math_asr_s32(vacc1, 1), 8);

    if (batch & (2 * sizeof(int8_t))) {
      output[0] = (int8_t) vacc0;
      output[1] = (int8_t) vacc1;
      vacc0 = __ssat(math_asr_s32(vacc2, 1), 8);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      output[0] = (int8_t) vacc0;
    }
  }
}
