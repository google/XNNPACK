// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vunary/sse2.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <emmintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/unaligned.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vabs_ukernel__sse2_u8(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  const __m128i vnonsign_mask = _mm_set1_epi16(0x7FFF);
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    __m128i vacc = _mm_loadu_si128((const __m128i*) i);
    i += 8;
    vacc = _mm_and_si128(vacc, vnonsign_mask);
    _mm_storeu_si128((__m128i*) o, vacc);
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m128i vacc = _mm_loadu_si128((const __m128i*) i);
    vacc = _mm_and_si128(vacc, vnonsign_mask);
    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) o, vacc);
      o += 4;
      vacc = _mm_unpackhi_epi64(vacc, vacc);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(vacc));
      o += 2;
      vacc = _mm_srli_epi64(vacc, 32);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      *o = (uint16_t) _mm_extract_epi16(vacc, 0);
    }
  }
}
