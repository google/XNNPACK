// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"
#include "xnnpack/unaligned.h"
#include "xnnpack/vunary.h"


void xnn_u8_vclamp_ukernel__avx512skx_u256(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const struct xnn_u8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512i voutput_min = _mm512_set1_epi8(params->scalar.min);
  const __m512i voutput_max = _mm512_set1_epi8(params->scalar.max);

  for (; batch >= 256; batch -= 256) {
    __m512i vacc0 = _mm512_loadu_si512((const __m512i*) input);
    __m512i vacc1 = _mm512_loadu_si512((const __m512i*) input + 1);
    __m512i vacc2 = _mm512_loadu_si512((const __m512i*) input + 2);
    __m512i vacc3 = _mm512_loadu_si512((const __m512i*) input + 3);
    input += 256;

    vacc0 = _mm512_max_epu8(vacc0, voutput_min);
    vacc1 = _mm512_max_epu8(vacc1, voutput_min);
    vacc2 = _mm512_max_epu8(vacc2, voutput_min);
    vacc3 = _mm512_max_epu8(vacc3, voutput_min);

    vacc0 = _mm512_min_epu8(vacc0, voutput_max);
    vacc1 = _mm512_min_epu8(vacc1, voutput_max);
    vacc2 = _mm512_min_epu8(vacc2, voutput_max);
    vacc3 = _mm512_min_epu8(vacc3, voutput_max);

    _mm512_storeu_si512((__m512i*) output, vacc0);
    _mm512_storeu_si512((__m512i*) output + 1, vacc1);
    _mm512_storeu_si512((__m512i*) output + 2, vacc2);
    _mm512_storeu_si512((__m512i*) output + 3, vacc3);
    output += 256;
  }
  for (; batch >= 64; batch -= 64) {
    __m512i vacc = _mm512_loadu_si512((const __m512i*) input);
    input += 64;

    vacc = _mm512_min_epu8(vacc, voutput_max);
    vacc = _mm512_max_epu8(vacc, voutput_min);

    _mm512_storeu_si512((__m512i*) output, vacc);
    output += 64;
  }

  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 && batch <= 63);
    const __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT64_C(1) << batch) - UINT64_C(1)));
    __m512i vacc = _mm512_maskz_loadu_epi8(vmask, input);

    vacc = _mm512_min_epu8(vacc, voutput_max);
    vacc = _mm512_max_epu8(vacc, voutput_min);

    _mm512_mask_storeu_epi8(output, vmask, vacc);
  }
}
