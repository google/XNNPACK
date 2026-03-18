// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-f32acc-rsum2/avx512skx.c.in
//   Generator: tools/xngen
//
// Copyright 2024-2025 Google LLC
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


void xnn_f16_f32acc_rsum2_ukernel__avx512skx_u16(
    size_t batch, const xnn_float16* input, float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  __m512 vacc0 = _mm512_setzero_ps();
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    __m512 vt0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;

    vt0 = _mm512_mul_ps(vt0, vt0);

    vacc0 = _mm512_add_ps(vacc0, vt0);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 15 * sizeof(uint16_t));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT16;
    const __mmask16 vmask =
        _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vt = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, i));

    vt = _mm512_mul_ps(vt, vt);

    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  const __m256 vacc256 = _mm256_add_ps(
    _mm512_castps512_ps256(vacc0),
    _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vacc0), 1)));
  __m128 vacc = _mm_add_ps(_mm256_castps256_ps128(vacc256),
                           _mm256_extractf128_ps(vacc256, 1));
  vacc = _mm_add_ps(vacc, _mm_movehl_ps(vacc, vacc));
  vacc = _mm_add_ss(vacc, _mm_movehdup_ps(vacc));
  vacc = _mm_mul_ss(vacc, _mm_load_ss(&params->scalar.scale));

  float vout = _mm_cvtss_f32(vacc);
  *output += vout;
}

void xnn_f16_f32acc_rsum2_ukernel__avx512skx_u32_acc2(
    size_t batch, const xnn_float16* input, float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  __m512 vacc0 = _mm512_setzero_ps();
  __m512 vacc1 = _mm512_setzero_ps();
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512 vt0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    __m512 vt1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 16)));
    i += 32;

    vt0 = _mm512_mul_ps(vt0, vt0);
    vt1 = _mm512_mul_ps(vt1, vt1);

    vacc0 = _mm512_add_ps(vacc0, vt0);
    vacc1 = _mm512_add_ps(vacc1, vt1);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  vacc0 = _mm512_add_ps(vacc0, vacc1);
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 15 * sizeof(uint16_t));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT16;
    const __mmask16 vmask =
        _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vt = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, i));

    vt = _mm512_mul_ps(vt, vt);

    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  const __m256 vacc256 = _mm256_add_ps(
    _mm512_castps512_ps256(vacc0),
    _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vacc0), 1)));
  __m128 vacc = _mm_add_ps(_mm256_castps256_ps128(vacc256),
                           _mm256_extractf128_ps(vacc256, 1));
  vacc = _mm_add_ps(vacc, _mm_movehl_ps(vacc, vacc));
  vacc = _mm_add_ss(vacc, _mm_movehdup_ps(vacc));
  vacc = _mm_mul_ss(vacc, _mm_load_ss(&params->scalar.scale));

  float vout = _mm_cvtss_f32(vacc);
  *output += vout;
}

void xnn_f16_f32acc_rsum2_ukernel__avx512skx_u32(
    size_t batch, const xnn_float16* input, float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  __m512 vacc0 = _mm512_setzero_ps();
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512 vt0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    __m512 vt1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 16)));
    i += 32;

    vt0 = _mm512_mul_ps(vt0, vt0);
    vt1 = _mm512_mul_ps(vt1, vt1);

    vacc0 = _mm512_add_ps(vacc0, vt0);
    vacc0 = _mm512_add_ps(vacc0, vt1);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 15 * sizeof(uint16_t));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT16;
    const __mmask16 vmask =
        _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vt = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, i));

    vt = _mm512_mul_ps(vt, vt);

    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  const __m256 vacc256 = _mm256_add_ps(
    _mm512_castps512_ps256(vacc0),
    _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vacc0), 1)));
  __m128 vacc = _mm_add_ps(_mm256_castps256_ps128(vacc256),
                           _mm256_extractf128_ps(vacc256, 1));
  vacc = _mm_add_ps(vacc, _mm_movehl_ps(vacc, vacc));
  vacc = _mm_add_ss(vacc, _mm_movehdup_ps(vacc));
  vacc = _mm_mul_ss(vacc, _mm_load_ss(&params->scalar.scale));

  float vout = _mm_cvtss_f32(vacc);
  *output += vout;
}

void xnn_f16_f32acc_rsum2_ukernel__avx512skx_u48_acc3(
    size_t batch, const xnn_float16* input, float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  __m512 vacc0 = _mm512_setzero_ps();
  __m512 vacc1 = _mm512_setzero_ps();
  __m512 vacc2 = _mm512_setzero_ps();
  for (; batch >= 48 * sizeof(uint16_t); batch -= 48 * sizeof(uint16_t)) {
    __m512 vt0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    __m512 vt1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 16)));
    __m512 vt2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 32)));
    i += 48;

    vt0 = _mm512_mul_ps(vt0, vt0);
    vt1 = _mm512_mul_ps(vt1, vt1);
    vt2 = _mm512_mul_ps(vt2, vt2);

    vacc0 = _mm512_add_ps(vacc0, vt0);
    vacc1 = _mm512_add_ps(vacc1, vt1);
    vacc2 = _mm512_add_ps(vacc2, vt2);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc1 = _mm512_add_ps(vacc1, vt);
  }
  vacc0 = _mm512_add_ps(vacc0, vacc2);
  vacc0 = _mm512_add_ps(vacc0, vacc1);
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 15 * sizeof(uint16_t));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT16;
    const __mmask16 vmask =
        _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vt = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, i));

    vt = _mm512_mul_ps(vt, vt);

    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  const __m256 vacc256 = _mm256_add_ps(
    _mm512_castps512_ps256(vacc0),
    _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vacc0), 1)));
  __m128 vacc = _mm_add_ps(_mm256_castps256_ps128(vacc256),
                           _mm256_extractf128_ps(vacc256, 1));
  vacc = _mm_add_ps(vacc, _mm_movehl_ps(vacc, vacc));
  vacc = _mm_add_ss(vacc, _mm_movehdup_ps(vacc));
  vacc = _mm_mul_ss(vacc, _mm_load_ss(&params->scalar.scale));

  float vout = _mm_cvtss_f32(vacc);
  *output += vout;
}

void xnn_f16_f32acc_rsum2_ukernel__avx512skx_u48(
    size_t batch, const xnn_float16* input, float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  __m512 vacc0 = _mm512_setzero_ps();
  for (; batch >= 48 * sizeof(uint16_t); batch -= 48 * sizeof(uint16_t)) {
    __m512 vt0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    __m512 vt1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 16)));
    __m512 vt2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 32)));
    i += 48;

    vt0 = _mm512_mul_ps(vt0, vt0);
    vt1 = _mm512_mul_ps(vt1, vt1);
    vt2 = _mm512_mul_ps(vt2, vt2);

    vacc0 = _mm512_add_ps(vacc0, vt0);
    vacc0 = _mm512_add_ps(vacc0, vt1);
    vacc0 = _mm512_add_ps(vacc0, vt2);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 15 * sizeof(uint16_t));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT16;
    const __mmask16 vmask =
        _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vt = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, i));

    vt = _mm512_mul_ps(vt, vt);

    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  const __m256 vacc256 = _mm256_add_ps(
    _mm512_castps512_ps256(vacc0),
    _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vacc0), 1)));
  __m128 vacc = _mm_add_ps(_mm256_castps256_ps128(vacc256),
                           _mm256_extractf128_ps(vacc256, 1));
  vacc = _mm_add_ps(vacc, _mm_movehl_ps(vacc, vacc));
  vacc = _mm_add_ss(vacc, _mm_movehdup_ps(vacc));
  vacc = _mm_mul_ss(vacc, _mm_load_ss(&params->scalar.scale));

  float vout = _mm_cvtss_f32(vacc);
  *output += vout;
}

void xnn_f16_f32acc_rsum2_ukernel__avx512skx_u64_acc4(
    size_t batch, const xnn_float16* input, float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  __m512 vacc0 = _mm512_setzero_ps();
  __m512 vacc1 = _mm512_setzero_ps();
  __m512 vacc2 = _mm512_setzero_ps();
  __m512 vacc3 = _mm512_setzero_ps();
  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512 vt0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    __m512 vt1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 16)));
    __m512 vt2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 32)));
    __m512 vt3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 48)));
    i += 64;

    vt0 = _mm512_mul_ps(vt0, vt0);
    vt1 = _mm512_mul_ps(vt1, vt1);
    vt2 = _mm512_mul_ps(vt2, vt2);
    vt3 = _mm512_mul_ps(vt3, vt3);

    vacc0 = _mm512_add_ps(vacc0, vt0);
    vacc1 = _mm512_add_ps(vacc1, vt1);
    vacc2 = _mm512_add_ps(vacc2, vt2);
    vacc3 = _mm512_add_ps(vacc3, vt3);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc1 = _mm512_add_ps(vacc1, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc2 = _mm512_add_ps(vacc2, vt);
  }
  vacc0 = _mm512_add_ps(vacc0, vacc2);
  vacc1 = _mm512_add_ps(vacc1, vacc3);
  vacc0 = _mm512_add_ps(vacc0, vacc1);
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 15 * sizeof(uint16_t));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT16;
    const __mmask16 vmask =
        _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vt = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, i));

    vt = _mm512_mul_ps(vt, vt);

    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  const __m256 vacc256 = _mm256_add_ps(
    _mm512_castps512_ps256(vacc0),
    _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vacc0), 1)));
  __m128 vacc = _mm_add_ps(_mm256_castps256_ps128(vacc256),
                           _mm256_extractf128_ps(vacc256, 1));
  vacc = _mm_add_ps(vacc, _mm_movehl_ps(vacc, vacc));
  vacc = _mm_add_ss(vacc, _mm_movehdup_ps(vacc));
  vacc = _mm_mul_ss(vacc, _mm_load_ss(&params->scalar.scale));

  float vout = _mm_cvtss_f32(vacc);
  *output += vout;
}

void xnn_f16_f32acc_rsum2_ukernel__avx512skx_u64_acc2(
    size_t batch, const xnn_float16* input, float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  __m512 vacc0 = _mm512_setzero_ps();
  __m512 vacc1 = _mm512_setzero_ps();
  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512 vt0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    __m512 vt1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 16)));
    __m512 vt2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 32)));
    __m512 vt3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 48)));
    i += 64;

    vt0 = _mm512_mul_ps(vt0, vt0);
    vt1 = _mm512_mul_ps(vt1, vt1);
    vt2 = _mm512_mul_ps(vt2, vt2);
    vt3 = _mm512_mul_ps(vt3, vt3);

    vacc0 = _mm512_add_ps(vacc0, vt0);
    vacc1 = _mm512_add_ps(vacc1, vt1);
    vacc0 = _mm512_add_ps(vacc0, vt2);
    vacc1 = _mm512_add_ps(vacc1, vt3);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc1 = _mm512_add_ps(vacc1, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  vacc0 = _mm512_add_ps(vacc0, vacc1);
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 15 * sizeof(uint16_t));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT16;
    const __mmask16 vmask =
        _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vt = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, i));

    vt = _mm512_mul_ps(vt, vt);

    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  const __m256 vacc256 = _mm256_add_ps(
    _mm512_castps512_ps256(vacc0),
    _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vacc0), 1)));
  __m128 vacc = _mm_add_ps(_mm256_castps256_ps128(vacc256),
                           _mm256_extractf128_ps(vacc256, 1));
  vacc = _mm_add_ps(vacc, _mm_movehl_ps(vacc, vacc));
  vacc = _mm_add_ss(vacc, _mm_movehdup_ps(vacc));
  vacc = _mm_mul_ss(vacc, _mm_load_ss(&params->scalar.scale));

  float vout = _mm_cvtss_f32(vacc);
  *output += vout;
}

void xnn_f16_f32acc_rsum2_ukernel__avx512skx_u64(
    size_t batch, const xnn_float16* input, float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  __m512 vacc0 = _mm512_setzero_ps();
  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512 vt0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    __m512 vt1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 16)));
    __m512 vt2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 32)));
    __m512 vt3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 48)));
    i += 64;

    vt0 = _mm512_mul_ps(vt0, vt0);
    vt1 = _mm512_mul_ps(vt1, vt1);
    vt2 = _mm512_mul_ps(vt2, vt2);
    vt3 = _mm512_mul_ps(vt3, vt3);

    vacc0 = _mm512_add_ps(vacc0, vt0);
    vacc0 = _mm512_add_ps(vacc0, vt1);
    vacc0 = _mm512_add_ps(vacc0, vt2);
    vacc0 = _mm512_add_ps(vacc0, vt3);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 15 * sizeof(uint16_t));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT16;
    const __mmask16 vmask =
        _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vt = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, i));

    vt = _mm512_mul_ps(vt, vt);

    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  const __m256 vacc256 = _mm256_add_ps(
    _mm512_castps512_ps256(vacc0),
    _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vacc0), 1)));
  __m128 vacc = _mm_add_ps(_mm256_castps256_ps128(vacc256),
                           _mm256_extractf128_ps(vacc256, 1));
  vacc = _mm_add_ps(vacc, _mm_movehl_ps(vacc, vacc));
  vacc = _mm_add_ss(vacc, _mm_movehdup_ps(vacc));
  vacc = _mm_mul_ss(vacc, _mm_load_ss(&params->scalar.scale));

  float vout = _mm_cvtss_f32(vacc);
  *output += vout;
}

void xnn_f16_f32acc_rsum2_ukernel__avx512skx_u128_acc8(
    size_t batch, const xnn_float16* input, float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  __m512 vacc0 = _mm512_setzero_ps();
  __m512 vacc1 = _mm512_setzero_ps();
  __m512 vacc2 = _mm512_setzero_ps();
  __m512 vacc3 = _mm512_setzero_ps();
  __m512 vacc4 = _mm512_setzero_ps();
  __m512 vacc5 = _mm512_setzero_ps();
  __m512 vacc6 = _mm512_setzero_ps();
  __m512 vacc7 = _mm512_setzero_ps();
  for (; batch >= 128 * sizeof(uint16_t); batch -= 128 * sizeof(uint16_t)) {
    __m512 vt0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    __m512 vt1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 16)));
    __m512 vt2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 32)));
    __m512 vt3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 48)));
    __m512 vt4 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 64)));
    __m512 vt5 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 80)));
    __m512 vt6 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 96)));
    __m512 vt7 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 112)));
    i += 128;

    vt0 = _mm512_mul_ps(vt0, vt0);
    vt1 = _mm512_mul_ps(vt1, vt1);
    vt2 = _mm512_mul_ps(vt2, vt2);
    vt3 = _mm512_mul_ps(vt3, vt3);
    vt4 = _mm512_mul_ps(vt4, vt4);
    vt5 = _mm512_mul_ps(vt5, vt5);
    vt6 = _mm512_mul_ps(vt6, vt6);
    vt7 = _mm512_mul_ps(vt7, vt7);

    vacc0 = _mm512_add_ps(vacc0, vt0);
    vacc1 = _mm512_add_ps(vacc1, vt1);
    vacc2 = _mm512_add_ps(vacc2, vt2);
    vacc3 = _mm512_add_ps(vacc3, vt3);
    vacc4 = _mm512_add_ps(vacc4, vt4);
    vacc5 = _mm512_add_ps(vacc5, vt5);
    vacc6 = _mm512_add_ps(vacc6, vt6);
    vacc7 = _mm512_add_ps(vacc7, vt7);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc1 = _mm512_add_ps(vacc1, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc2 = _mm512_add_ps(vacc2, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc3 = _mm512_add_ps(vacc3, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc4 = _mm512_add_ps(vacc4, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc5 = _mm512_add_ps(vacc5, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc6 = _mm512_add_ps(vacc6, vt);
  }
  vacc0 = _mm512_add_ps(vacc0, vacc4);
  vacc1 = _mm512_add_ps(vacc1, vacc5);
  vacc2 = _mm512_add_ps(vacc2, vacc6);
  vacc3 = _mm512_add_ps(vacc3, vacc7);
  vacc0 = _mm512_add_ps(vacc0, vacc2);
  vacc1 = _mm512_add_ps(vacc1, vacc3);
  vacc0 = _mm512_add_ps(vacc0, vacc1);
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 15 * sizeof(uint16_t));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT16;
    const __mmask16 vmask =
        _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vt = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, i));

    vt = _mm512_mul_ps(vt, vt);

    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  const __m256 vacc256 = _mm256_add_ps(
    _mm512_castps512_ps256(vacc0),
    _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vacc0), 1)));
  __m128 vacc = _mm_add_ps(_mm256_castps256_ps128(vacc256),
                           _mm256_extractf128_ps(vacc256, 1));
  vacc = _mm_add_ps(vacc, _mm_movehl_ps(vacc, vacc));
  vacc = _mm_add_ss(vacc, _mm_movehdup_ps(vacc));
  vacc = _mm_mul_ss(vacc, _mm_load_ss(&params->scalar.scale));

  float vout = _mm_cvtss_f32(vacc);
  *output += vout;
}

void xnn_f16_f32acc_rsum2_ukernel__avx512skx_u128_acc4(
    size_t batch, const xnn_float16* input, float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  __m512 vacc0 = _mm512_setzero_ps();
  __m512 vacc1 = _mm512_setzero_ps();
  __m512 vacc2 = _mm512_setzero_ps();
  __m512 vacc3 = _mm512_setzero_ps();
  for (; batch >= 128 * sizeof(uint16_t); batch -= 128 * sizeof(uint16_t)) {
    __m512 vt0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    __m512 vt1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 16)));
    __m512 vt2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 32)));
    __m512 vt3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 48)));
    __m512 vt4 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 64)));
    __m512 vt5 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 80)));
    __m512 vt6 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 96)));
    __m512 vt7 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 112)));
    i += 128;

    vt0 = _mm512_mul_ps(vt0, vt0);
    vt1 = _mm512_mul_ps(vt1, vt1);
    vt2 = _mm512_mul_ps(vt2, vt2);
    vt3 = _mm512_mul_ps(vt3, vt3);
    vt4 = _mm512_mul_ps(vt4, vt4);
    vt5 = _mm512_mul_ps(vt5, vt5);
    vt6 = _mm512_mul_ps(vt6, vt6);
    vt7 = _mm512_mul_ps(vt7, vt7);

    vacc0 = _mm512_add_ps(vacc0, vt0);
    vacc1 = _mm512_add_ps(vacc1, vt1);
    vacc2 = _mm512_add_ps(vacc2, vt2);
    vacc3 = _mm512_add_ps(vacc3, vt3);
    vacc0 = _mm512_add_ps(vacc0, vt4);
    vacc1 = _mm512_add_ps(vacc1, vt5);
    vacc2 = _mm512_add_ps(vacc2, vt6);
    vacc3 = _mm512_add_ps(vacc3, vt7);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc1 = _mm512_add_ps(vacc1, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc2 = _mm512_add_ps(vacc2, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc3 = _mm512_add_ps(vacc3, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc1 = _mm512_add_ps(vacc1, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc2 = _mm512_add_ps(vacc2, vt);
  }
  vacc0 = _mm512_add_ps(vacc0, vacc2);
  vacc1 = _mm512_add_ps(vacc1, vacc3);
  vacc0 = _mm512_add_ps(vacc0, vacc1);
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 15 * sizeof(uint16_t));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT16;
    const __mmask16 vmask =
        _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vt = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, i));

    vt = _mm512_mul_ps(vt, vt);

    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  const __m256 vacc256 = _mm256_add_ps(
    _mm512_castps512_ps256(vacc0),
    _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vacc0), 1)));
  __m128 vacc = _mm_add_ps(_mm256_castps256_ps128(vacc256),
                           _mm256_extractf128_ps(vacc256, 1));
  vacc = _mm_add_ps(vacc, _mm_movehl_ps(vacc, vacc));
  vacc = _mm_add_ss(vacc, _mm_movehdup_ps(vacc));
  vacc = _mm_mul_ss(vacc, _mm_load_ss(&params->scalar.scale));

  float vout = _mm_cvtss_f32(vacc);
  *output += vout;
}

void xnn_f16_f32acc_rsum2_ukernel__avx512skx_u128_acc2(
    size_t batch, const xnn_float16* input, float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  __m512 vacc0 = _mm512_setzero_ps();
  __m512 vacc1 = _mm512_setzero_ps();
  for (; batch >= 128 * sizeof(uint16_t); batch -= 128 * sizeof(uint16_t)) {
    __m512 vt0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    __m512 vt1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 16)));
    __m512 vt2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 32)));
    __m512 vt3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 48)));
    __m512 vt4 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 64)));
    __m512 vt5 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 80)));
    __m512 vt6 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 96)));
    __m512 vt7 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 112)));
    i += 128;

    vt0 = _mm512_mul_ps(vt0, vt0);
    vt1 = _mm512_mul_ps(vt1, vt1);
    vt2 = _mm512_mul_ps(vt2, vt2);
    vt3 = _mm512_mul_ps(vt3, vt3);
    vt4 = _mm512_mul_ps(vt4, vt4);
    vt5 = _mm512_mul_ps(vt5, vt5);
    vt6 = _mm512_mul_ps(vt6, vt6);
    vt7 = _mm512_mul_ps(vt7, vt7);

    vacc0 = _mm512_add_ps(vacc0, vt0);
    vacc1 = _mm512_add_ps(vacc1, vt1);
    vacc0 = _mm512_add_ps(vacc0, vt2);
    vacc1 = _mm512_add_ps(vacc1, vt3);
    vacc0 = _mm512_add_ps(vacc0, vt4);
    vacc1 = _mm512_add_ps(vacc1, vt5);
    vacc0 = _mm512_add_ps(vacc0, vt6);
    vacc1 = _mm512_add_ps(vacc1, vt7);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc1 = _mm512_add_ps(vacc1, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc1 = _mm512_add_ps(vacc1, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc1 = _mm512_add_ps(vacc1, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  vacc0 = _mm512_add_ps(vacc0, vacc1);
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 15 * sizeof(uint16_t));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT16;
    const __mmask16 vmask =
        _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vt = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, i));

    vt = _mm512_mul_ps(vt, vt);

    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  const __m256 vacc256 = _mm256_add_ps(
    _mm512_castps512_ps256(vacc0),
    _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vacc0), 1)));
  __m128 vacc = _mm_add_ps(_mm256_castps256_ps128(vacc256),
                           _mm256_extractf128_ps(vacc256, 1));
  vacc = _mm_add_ps(vacc, _mm_movehl_ps(vacc, vacc));
  vacc = _mm_add_ss(vacc, _mm_movehdup_ps(vacc));
  vacc = _mm_mul_ss(vacc, _mm_load_ss(&params->scalar.scale));

  float vout = _mm_cvtss_f32(vacc);
  *output += vout;
}

void xnn_f16_f32acc_rsum2_ukernel__avx512skx_u128(
    size_t batch, const xnn_float16* input, float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  __m512 vacc0 = _mm512_setzero_ps();
  for (; batch >= 128 * sizeof(uint16_t); batch -= 128 * sizeof(uint16_t)) {
    __m512 vt0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    __m512 vt1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 16)));
    __m512 vt2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 32)));
    __m512 vt3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 48)));
    __m512 vt4 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 64)));
    __m512 vt5 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 80)));
    __m512 vt6 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 96)));
    __m512 vt7 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (i + 112)));
    i += 128;

    vt0 = _mm512_mul_ps(vt0, vt0);
    vt1 = _mm512_mul_ps(vt1, vt1);
    vt2 = _mm512_mul_ps(vt2, vt2);
    vt3 = _mm512_mul_ps(vt3, vt3);
    vt4 = _mm512_mul_ps(vt4, vt4);
    vt5 = _mm512_mul_ps(vt5, vt5);
    vt6 = _mm512_mul_ps(vt6, vt6);
    vt7 = _mm512_mul_ps(vt7, vt7);

    vacc0 = _mm512_add_ps(vacc0, vt0);
    vacc0 = _mm512_add_ps(vacc0, vt1);
    vacc0 = _mm512_add_ps(vacc0, vt2);
    vacc0 = _mm512_add_ps(vacc0, vt3);
    vacc0 = _mm512_add_ps(vacc0, vt4);
    vacc0 = _mm512_add_ps(vacc0, vt5);
    vacc0 = _mm512_add_ps(vacc0, vt6);
    vacc0 = _mm512_add_ps(vacc0, vt7);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  if (batch >= 16 * sizeof(uint16_t)) {
    __m512 vt = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;
    batch -= 16 * sizeof(uint16_t);
    vt = _mm512_mul_ps(vt, vt);
    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 15 * sizeof(uint16_t));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT16;
    const __mmask16 vmask =
        _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vt = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, i));

    vt = _mm512_mul_ps(vt, vt);

    vacc0 = _mm512_add_ps(vacc0, vt);
  }
  const __m256 vacc256 = _mm256_add_ps(
    _mm512_castps512_ps256(vacc0),
    _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(vacc0), 1)));
  __m128 vacc = _mm_add_ps(_mm256_castps256_ps128(vacc256),
                           _mm256_extractf128_ps(vacc256, 1));
  vacc = _mm_add_ps(vacc, _mm_movehl_ps(vacc, vacc));
  vacc = _mm_add_ss(vacc, _mm_movehdup_ps(vacc));
  vacc = _mm_mul_ss(vacc, _mm_load_ss(&params->scalar.scale));

  float vout = _mm_cvtss_f32(vacc);
  *output += vout;
}
