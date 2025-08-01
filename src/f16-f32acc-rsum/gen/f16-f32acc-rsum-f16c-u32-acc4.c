// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-f32acc-rsum/f16c.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
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
#include "src/xnnpack/reduce.h"
#include "src/xnnpack/unaligned.h"


void xnn_f16_f32acc_rsum_ukernel__f16c_u32_acc4(
    size_t batch,
    const xnn_float16* input,
    float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params)
{
  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  __m256 vacc0 = _mm256_setzero_ps();
  __m256 vacc1 = _mm256_setzero_ps();
  __m256 vacc2 = _mm256_setzero_ps();
  __m256 vacc3 = _mm256_setzero_ps();
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const __m256 vt0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    const __m256 vt1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    const __m256 vt2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 16)));
    const __m256 vt3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 24)));
    i += 32;

    vacc0 = _mm256_add_ps(vacc0, vt0);
    vacc1 = _mm256_add_ps(vacc1, vt1);
    vacc2 = _mm256_add_ps(vacc2, vt2);
    vacc3 = _mm256_add_ps(vacc3, vt3);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const __m256 vt = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;
    batch -= 8 * sizeof(uint16_t);
    vacc0 = _mm256_add_ps(vacc0, vt);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const __m256 vt = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;
    batch -= 8 * sizeof(uint16_t);
    vacc1 = _mm256_add_ps(vacc1, vt);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const __m256 vt = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;
    batch -= 8 * sizeof(uint16_t);
    vacc2 = _mm256_add_ps(vacc2, vt);
  }
  vacc0 = _mm256_add_ps(vacc0, vacc2);
  vacc1 = _mm256_add_ps(vacc1, vacc3);
  vacc0 = _mm256_add_ps(vacc0, vacc1);
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));
    const __m128i vmask = _mm_loadu_si128((const __m128i*) ((uintptr_t) &mask_table[7] - batch));
    const __m128i vh = _mm_castps_si128(_mm_maskload_ps((const float*) i, vmask));
    const __m256 vt = _mm256_cvtph_ps(vh);
    vacc0 = _mm256_add_ps(vacc0, vt);
    i = (const void*) ((uintptr_t) i + batch);
    if (batch & (1 * sizeof(uint16_t))) {
      const __m128i vh = _mm_insert_epi16(_mm_setzero_si128(), (int) unaligned_load_u16(i - 1), 0);
      const __m256 vt = _mm256_zextps128_ps256(_mm_cvtph_ps(vh));
      vacc0 = _mm256_add_ps(vacc0, vt);
    }
  }
  __m128 vacc = _mm_add_ps(_mm256_castps256_ps128(vacc0), _mm256_extractf128_ps(vacc0, 1));
  vacc = _mm_add_ps(vacc, _mm_movehl_ps(vacc, vacc));
  vacc = _mm_add_ss(vacc, _mm_movehdup_ps(vacc));
  vacc = _mm_mul_ss(vacc, _mm_load_ss(&params->scalar.scale));

  float vout = _mm_cvtss_f32(vacc);
  *output += vout;
}
