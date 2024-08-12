// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"


void xnn_u8_rmax_ukernel__sse2_u16(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  if XNN_LIKELY(batch >= 16) {
    __m128i vmax = _mm_setzero_si128();
    do {
      const __m128i vx = _mm_loadu_si128((const __m128i*) input);
      input += 16;
      vmax = _mm_max_epu8(vmax, vx);
      batch -= 16;
    } while (batch >= 16);
    if (batch != 0) {
      const size_t x_increment = batch - 16;
      input = (const uint8_t*) ((uintptr_t) input + x_increment);
      const __m128i vx = _mm_loadu_si128((const __m128i*) input);
      vmax = _mm_max_epu8(vmax, vx);
    }
    vmax = _mm_max_epu8(vmax, _mm_unpackhi_epi64(vmax, vmax));
    vmax = _mm_max_epu8(vmax, _mm_srli_epi64(vmax, 32));
    vmax = _mm_max_epu8(vmax, _mm_srli_epi32(vmax, 16));
    vmax = _mm_max_epu8(vmax, _mm_srli_epi16(vmax, 8));
    *output = (uint8_t) _mm_cvtsi128_si32(vmax);
  } else {
    uint8_t vmax = 0;
    do {
      const uint8_t vx = *input++;
      vmax = vx > vmax ? vx : vmax;
    } while (--batch != 0);
    *output = vmax;
  }
}
