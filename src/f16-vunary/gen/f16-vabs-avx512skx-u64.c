// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vunary/avx512skx.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vabs_ukernel__avx512skx_u64(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  const __m512i vnonsign_mask = _mm512_set1_epi16(0x7FFF);
  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512i vacc0 = _mm512_loadu_si512(i);
    __m512i vacc1 = _mm512_loadu_si512(i + 32);
    i += 64;

    vacc0 = _mm512_and_si512(vacc0, vnonsign_mask);
    vacc1 = _mm512_and_si512(vacc1, vnonsign_mask);

    _mm512_storeu_si512(o, vacc0);
    _mm512_storeu_si512(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512i vacc = _mm512_loadu_si512(i);
    i += 32;
    vacc = _mm512_and_si512(vacc, vnonsign_mask);
    _mm512_storeu_si512(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));

    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT16;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512i vacc = _mm512_maskz_loadu_epi16(vmask, i);
    vacc = _mm512_and_si512(vacc, vnonsign_mask);
    _mm512_mask_storeu_epi16(o, vmask, vacc);
  }
}
