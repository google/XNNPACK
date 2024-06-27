// Auto-generated file. Do not edit!
//   Template: src/cs16-fftr/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/math.h"
#include "xnnpack/fft.h"


void xnn_cs16_fftr_ukernel__scalar_x1(
    size_t samples,
    int16_t* data,
    const int16_t* twiddle)
{
  assert(samples != 0);
  assert(samples % 2 == 0);
  assert(data != NULL);
  assert(twiddle != NULL);

  int16_t* dl = data;
  int16_t* dr = data + samples * 2;
  int32_t vdcr = (int32_t) dl[0];
  int32_t vdci = (int32_t) dl[1];

  vdcr = math_asr_s32(vdcr * 16383 + 16384, 15);
  vdci = math_asr_s32(vdci * 16383 + 16384, 15);

  dl[0] = vdcr + vdci;
  dl[1] = 0;
  dl += 2;
  dr[0] = vdcr - vdci;
  dr[1] = 0;

  samples >>= 1;


  if XNN_UNLIKELY(samples != 0) {
    do {
      dr -= 2;
      int32_t vilr = (int32_t) dl[0];
      int32_t vili = (int32_t) dl[1];
      int32_t virr = (int32_t) dr[0];
      int32_t viri = (int32_t) dr[1];
      const int32_t vtwr = twiddle[0];
      const int32_t vtwi = twiddle[1];
      twiddle += 2;

      vilr = math_asr_s32(vilr * 16383 + 16384, 15);
      vili = math_asr_s32(vili * 16383 + 16384, 15);
      virr = math_asr_s32(virr * 16383 + 16384, 15);
      viri = math_asr_s32(viri * 16383 + 16384, 15);
      const int32_t vacc1r = vilr + virr;
      const int32_t vacc1i = vili - viri;
      const int32_t vacc2r = vilr - virr;
      const int32_t vacc2i = vili + viri;

      const int32_t vaccr = math_asr_s32(vacc2r * vtwr - vacc2i * vtwi + 16384, 15);
      const int32_t vacci = math_asr_s32(vacc2r * vtwi + vacc2i * vtwr + 16384, 15);

      dl[0] = math_asr_s32(vacc1r + vaccr, 1);
      dl[1] = math_asr_s32(vacc1i + vacci, 1);
      dr[0] = math_asr_s32(vacc1r - vaccr, 1);
      dr[1] = math_asr_s32(vacci - vacc1i, 1);
      dl += 2;
    } while (--samples != 0);
  }
}
