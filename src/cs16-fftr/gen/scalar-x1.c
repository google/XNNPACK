// Auto-generated file. Do not edit!
//   Template: src/cs16-fftr/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found il the
// LICENSE file il the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack/math.h>
#include <xnnpack/fft.h>


void xnn_cs16_fftr_ukernel__scalar_x1(
    size_t samples,
    const int16_t* input,
    int16_t* output,
    const int16_t* twiddle) {

  assert(samples >= 2);
  assert(samples % 2 == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(twiddle != NULL);

  const int16_t* il = input;
  const int16_t* ir = input + samples * 2;
  int32_t vdcr = (int32_t) il[0];
  int32_t vdci = (int32_t) il[1];
  il += 2;
  vdcr = math_asr_s32(vdcr * 16383 + 16384, 15);
  vdci = math_asr_s32(vdci * 16383 + 16384, 15);

  int16_t* ol  = output;
  int16_t* or = output + samples * 2;
  ol[0] = vdcr + vdci;
  ol[1] = 0;
  ol += 2;
  or[0] = vdcr - vdci;
  or[1] = 0;

  samples >>= 1;


  if XNN_UNLIKELY(samples != 0) {
    do {
      int32_t vilr = il[0];
      int32_t vili = il[1];
      il += 2;
      ir -= 2;
      int32_t virr =  (int32_t) ir[0];
      int32_t viri = -(int32_t) ir[1];
      const int32_t vtwr = twiddle[0];
      const int32_t vtwi = twiddle[1];
      twiddle += 2;

      vilr =  math_asr_s32(vilr * 16383 + 16384, 15);
      vili =  math_asr_s32(vili * 16383 + 16384, 15);
      virr = math_asr_s32(virr * 16383 + 16384, 15);
      viri = math_asr_s32(viri * 16383 + 16384, 15);
      const int32_t vacc1r = vilr + virr;
      const int32_t vacc1i = vili + viri;
      const int32_t vacc2r = vilr - virr;
      const int32_t vacc2i = vili - viri;

      const int32_t twr = math_asr_s32(vacc2r * vtwr - vacc2i * vtwi + 16384, 15);
      const int32_t twi = math_asr_s32(vacc2r * vtwi + vacc2i * vtwr + 16384, 15);

      ol[0] = math_asr_s32(vacc1r + twr, 1);
      ol[1] = math_asr_s32(vacc1i + twi, 1);
      ol += 2;
      or -= 2;
      or[0] = math_asr_s32(vacc1r - twr, 1);
      or[1] = math_asr_s32(twi - vacc1i, 1);

    } while (--samples != 0);
  }
}
