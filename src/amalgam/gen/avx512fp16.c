// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/reduce.h>
#include <xnnpack/vbinary.h>


void xnn_f16_rmax_ukernel__avx512fp16_u128_acc4(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* i = (const uint16_t*) input;
  __m512h vmax0 = _mm512_castsi512_ph(_mm512_set1_epi16(*i));
  __m512h vmax1 = vmax0;
  __m512h vmax2 = vmax0;
  __m512h vmax3 = vmax0;
  for (; batch >= 128 * sizeof(uint16_t); batch -= 128 * sizeof(uint16_t)) {
    const __m512h vt0 = _mm512_loadu_ph(i);
    const __m512h vt1 = _mm512_loadu_ph((i + 32));
    const __m512h vt2 = _mm512_loadu_ph((i + 64));
    const __m512h vt3 = _mm512_loadu_ph((i + 96));
    i += 128;

    vmax0 = _mm512_max_ph(vmax0, vt0);
    vmax1 = _mm512_max_ph(vmax1, vt1);
    vmax2 = _mm512_max_ph(vmax2, vt2);
    vmax3 = _mm512_max_ph(vmax3, vt3);
  }
  vmax0 = _mm512_max_ph(vmax0, vmax1);
  vmax2 = _mm512_max_ph(vmax2, vmax3);
  vmax0 = _mm512_max_ph(vmax0, vmax2);
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const __m512h vt = _mm512_loadu_ph(i);
    i += 32;

    vmax0 = _mm512_max_ph(vmax0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512h vt = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i));

    vmax0 = _mm512_mask_max_ph(vmax0, vmask, vmax0, vt);
  }
  __m256h vmax256 = _mm256_max_ph(_mm512_castph512_ph256(vmax0), _mm256_castpd_ph(_mm512_extractf64x4_pd(_mm512_castph_pd(vmax0), 1)));
  __m128h vmax = _mm_max_ph(_mm256_castph256_ph128(vmax256), _mm_castps_ph(_mm256_extractf128_ps(_mm256_castph_ps(vmax256), 1)));
  vmax = _mm_max_ph(vmax, _mm_castps_ph(_mm_movehl_ps(_mm_castph_ps(vmax), _mm_castph_ps(vmax))));
  vmax = _mm_max_ph(vmax, _mm_castps_ph(_mm_movehdup_ps(_mm_castph_ps(vmax))));
  vmax = _mm_max_sh(vmax, _mm_castsi128_ph(_mm_srli_epi32(_mm_castph_si128(vmax), 16)));

  *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(_mm_castph_si128(vmax), 0);
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_rminmax_ukernel__avx512fp16_u128_acc4(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* i = (const uint16_t*) input;
  __m512h vmin0 = _mm512_castsi512_ph(_mm512_set1_epi16(*i));
  __m512h vmax0 = vmin0;
  __m512h vmin1 = vmin0;
  __m512h vmax1 = vmax0;
  __m512h vmin2 = vmin0;
  __m512h vmax2 = vmax0;
  __m512h vmin3 = vmin0;
  __m512h vmax3 = vmax0;
  for (; batch >= 128 * sizeof(uint16_t); batch -= 128 * sizeof(uint16_t)) {
    const __m512h vt0 = _mm512_loadu_ph(i);
    const __m512h vt1 = _mm512_loadu_ph((i + 32));
    const __m512h vt2 = _mm512_loadu_ph((i + 64));
    const __m512h vt3 = _mm512_loadu_ph((i + 96));
    i += 128;

    vmin0 = _mm512_min_ph(vmin0, vt0);
    vmax0 = _mm512_max_ph(vmax0, vt0);
    vmin1 = _mm512_min_ph(vmin1, vt1);
    vmax1 = _mm512_max_ph(vmax1, vt1);
    vmin2 = _mm512_min_ph(vmin2, vt2);
    vmax2 = _mm512_max_ph(vmax2, vt2);
    vmin3 = _mm512_min_ph(vmin3, vt3);
    vmax3 = _mm512_max_ph(vmax3, vt3);
  }
  vmin0 = _mm512_min_ph(vmin0, vmin1);
  vmax0 = _mm512_max_ph(vmax0, vmax1);
  vmin2 = _mm512_min_ph(vmin2, vmin3);
  vmax2 = _mm512_max_ph(vmax2, vmax3);
  vmin0 = _mm512_min_ph(vmin0, vmin2);
  vmax0 = _mm512_max_ph(vmax0, vmax2);
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const __m512h vt = _mm512_loadu_ph(i);
    i += 32;

    vmin0 = _mm512_min_ph(vmin0, vt);
    vmax0 = _mm512_max_ph(vmax0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512h vt = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i));

    vmin0 = _mm512_mask_min_ph(vmin0, vmask, vmin0, vt);
    vmax0 = _mm512_mask_max_ph(vmax0, vmask, vmax0, vt);
  }
  __m256h vmin256 = _mm256_min_ph(_mm512_castph512_ph256(vmin0), _mm256_castpd_ph(_mm512_extractf64x4_pd(_mm512_castph_pd(vmin0), 1)));
  __m256h vmax256 = _mm256_max_ph(_mm512_castph512_ph256(vmax0), _mm256_castpd_ph(_mm512_extractf64x4_pd(_mm512_castph_pd(vmax0), 1)));
  __m128h vmin = _mm_min_ph(_mm256_castph256_ph128(vmin256), _mm_castps_ph(_mm256_extractf128_ps(_mm256_castph_ps(vmin256), 1)));
  __m128h vmax = _mm_max_ph(_mm256_castph256_ph128(vmax256), _mm_castps_ph(_mm256_extractf128_ps(_mm256_castph_ps(vmax256), 1)));
  vmin = _mm_min_ph(vmin, _mm_castps_ph(_mm_movehl_ps(_mm_castph_ps(vmin), _mm_castph_ps(vmin))));
  vmax = _mm_max_ph(vmax, _mm_castps_ph(_mm_movehl_ps(_mm_castph_ps(vmax), _mm_castph_ps(vmax))));
  vmin = _mm_min_ph(vmin, _mm_castps_ph(_mm_movehdup_ps(_mm_castph_ps(vmin))));
  vmax = _mm_max_ph(vmax, _mm_castps_ph(_mm_movehdup_ps(_mm_castph_ps(vmax))));
  vmin = _mm_min_sh(vmin, _mm_castsi128_ph(_mm_srli_epi32(_mm_castph_si128(vmin), 16)));
  vmax = _mm_max_sh(vmax, _mm_castsi128_ph(_mm_srli_epi32(_mm_castph_si128(vmax), 16)));

  *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(_mm_castph_si128(vmin), 0);
  *((uint16_t*) output + 1) = (uint16_t) _mm_extract_epi16(_mm_castph_si128(vmax), 0);
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vadd_minmax_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h voutput_min = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
  const __m512h voutput_max = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_add_ph(vacc0, _mm512_loadu_ph(b));
    vacc1 = _mm512_add_ph(vacc1, _mm512_loadu_ph(b + 32));
    b += 64;


    vacc0 = _mm512_max_ph(voutput_min, vacc0);
    vacc1 = _mm512_max_ph(voutput_min, vacc1);

    vacc0 = _mm512_min_ph(voutput_max, vacc0);
    vacc1 = _mm512_min_ph(voutput_max, vacc1);

    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_add_ph(vacc, _mm512_loadu_ph(b));
    b += 32;

    vacc = _mm512_max_ph(voutput_min, vacc);
    vacc = _mm512_min_ph(voutput_max, vacc);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_add_ph(vmask, vacc, _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, b)));

    vacc = _mm512_maskz_max_ph(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ph(vmask, voutput_max, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vaddc_minmax_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h voutput_min = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
  const __m512h voutput_max = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));
  const __m512h vb = _mm512_castsi512_ph(_mm512_set1_epi16(*b));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_add_ph(vacc0, vb);
    vacc1 = _mm512_add_ph(vacc1, vb);


    vacc0 = _mm512_max_ph(voutput_min, vacc0);
    vacc1 = _mm512_max_ph(voutput_min, vacc1);

    vacc0 = _mm512_min_ph(voutput_max, vacc0);
    vacc1 = _mm512_min_ph(voutput_max, vacc1);

    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_add_ph(vacc, vb);
    vacc = _mm512_max_ph(voutput_min, vacc);
    vacc = _mm512_min_ph(voutput_max, vacc);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_add_ph(vmask, vacc, vb);
    vacc = _mm512_maskz_max_ph(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ph(vmask, voutput_max, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vdiv_minmax_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h voutput_min = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
  const __m512h voutput_max = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_div_ph(vacc0, _mm512_loadu_ph(b));
    vacc1 = _mm512_div_ph(vacc1, _mm512_loadu_ph(b + 32));
    b += 64;


    vacc0 = _mm512_max_ph(voutput_min, vacc0);
    vacc1 = _mm512_max_ph(voutput_min, vacc1);

    vacc0 = _mm512_min_ph(voutput_max, vacc0);
    vacc1 = _mm512_min_ph(voutput_max, vacc1);

    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_div_ph(vacc, _mm512_loadu_ph(b));
    b += 32;

    vacc = _mm512_max_ph(voutput_min, vacc);
    vacc = _mm512_min_ph(voutput_max, vacc);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_div_ph(vmask, vacc, _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, b)));

    vacc = _mm512_maskz_max_ph(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ph(vmask, voutput_max, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vdivc_minmax_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h voutput_min = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
  const __m512h voutput_max = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));
  const __m512h vb = _mm512_castsi512_ph(_mm512_set1_epi16(*b));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_div_ph(vacc0, vb);
    vacc1 = _mm512_div_ph(vacc1, vb);


    vacc0 = _mm512_max_ph(voutput_min, vacc0);
    vacc1 = _mm512_max_ph(voutput_min, vacc1);

    vacc0 = _mm512_min_ph(voutput_max, vacc0);
    vacc1 = _mm512_min_ph(voutput_max, vacc1);

    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_div_ph(vacc, vb);
    vacc = _mm512_max_ph(voutput_min, vacc);
    vacc = _mm512_min_ph(voutput_max, vacc);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_div_ph(vmask, vacc, vb);
    vacc = _mm512_maskz_max_ph(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ph(vmask, voutput_max, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vmax_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;


  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_max_ph(vacc0, _mm512_loadu_ph(b));
    vacc1 = _mm512_max_ph(vacc1, _mm512_loadu_ph(b + 32));
    b += 64;



    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_max_ph(vacc, _mm512_loadu_ph(b));
    b += 32;


    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_max_ph(vmask, vacc, _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, b)));

    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vmaxc_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h vb = _mm512_castsi512_ph(_mm512_set1_epi16(*b));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_max_ph(vacc0, vb);
    vacc1 = _mm512_max_ph(vacc1, vb);



    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_max_ph(vacc, vb);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_max_ph(vmask, vacc, vb);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vmin_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;


  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_min_ph(vacc0, _mm512_loadu_ph(b));
    vacc1 = _mm512_min_ph(vacc1, _mm512_loadu_ph(b + 32));
    b += 64;



    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_min_ph(vacc, _mm512_loadu_ph(b));
    b += 32;


    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_min_ph(vmask, vacc, _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, b)));

    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vminc_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h vb = _mm512_castsi512_ph(_mm512_set1_epi16(*b));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_min_ph(vacc0, vb);
    vacc1 = _mm512_min_ph(vacc1, vb);



    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_min_ph(vacc, vb);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_min_ph(vmask, vacc, vb);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vmul_minmax_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h voutput_min = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
  const __m512h voutput_max = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_mul_ph(vacc0, _mm512_loadu_ph(b));
    vacc1 = _mm512_mul_ph(vacc1, _mm512_loadu_ph(b + 32));
    b += 64;


    vacc0 = _mm512_max_ph(voutput_min, vacc0);
    vacc1 = _mm512_max_ph(voutput_min, vacc1);

    vacc0 = _mm512_min_ph(voutput_max, vacc0);
    vacc1 = _mm512_min_ph(voutput_max, vacc1);

    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_mul_ph(vacc, _mm512_loadu_ph(b));
    b += 32;

    vacc = _mm512_max_ph(voutput_min, vacc);
    vacc = _mm512_min_ph(voutput_max, vacc);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_mul_ph(vmask, vacc, _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, b)));

    vacc = _mm512_maskz_max_ph(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ph(vmask, voutput_max, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vmulc_minmax_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h voutput_min = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
  const __m512h voutput_max = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));
  const __m512h vb = _mm512_castsi512_ph(_mm512_set1_epi16(*b));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_mul_ph(vacc0, vb);
    vacc1 = _mm512_mul_ph(vacc1, vb);


    vacc0 = _mm512_max_ph(voutput_min, vacc0);
    vacc1 = _mm512_max_ph(voutput_min, vacc1);

    vacc0 = _mm512_min_ph(voutput_max, vacc0);
    vacc1 = _mm512_min_ph(voutput_max, vacc1);

    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_mul_ph(vacc, vb);
    vacc = _mm512_max_ph(voutput_min, vacc);
    vacc = _mm512_min_ph(voutput_max, vacc);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_mul_ph(vmask, vacc, vb);
    vacc = _mm512_maskz_max_ph(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ph(vmask, voutput_max, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vrdivc_minmax_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h voutput_min = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
  const __m512h voutput_max = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));
  const __m512h vb = _mm512_castsi512_ph(_mm512_set1_epi16(*b));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_div_ph(vb, vacc0);
    vacc1 = _mm512_div_ph(vb, vacc1);


    vacc0 = _mm512_max_ph(voutput_min, vacc0);
    vacc1 = _mm512_max_ph(voutput_min, vacc1);

    vacc0 = _mm512_min_ph(voutput_max, vacc0);
    vacc1 = _mm512_min_ph(voutput_max, vacc1);

    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_div_ph(vb, vacc);
    vacc = _mm512_max_ph(voutput_min, vacc);
    vacc = _mm512_min_ph(voutput_max, vacc);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_div_ph(vmask, vb, vacc);
    vacc = _mm512_maskz_max_ph(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ph(vmask, voutput_max, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vrsubc_minmax_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h voutput_min = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
  const __m512h voutput_max = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));
  const __m512h vb = _mm512_castsi512_ph(_mm512_set1_epi16(*b));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_sub_ph(vb, vacc0);
    vacc1 = _mm512_sub_ph(vb, vacc1);


    vacc0 = _mm512_max_ph(voutput_min, vacc0);
    vacc1 = _mm512_max_ph(voutput_min, vacc1);

    vacc0 = _mm512_min_ph(voutput_max, vacc0);
    vacc1 = _mm512_min_ph(voutput_max, vacc1);

    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_sub_ph(vb, vacc);
    vacc = _mm512_max_ph(voutput_min, vacc);
    vacc = _mm512_min_ph(voutput_max, vacc);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_sub_ph(vmask, vb, vacc);
    vacc = _mm512_maskz_max_ph(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ph(vmask, voutput_max, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vsqrdiff_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;


  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_sub_ph(vacc0, _mm512_loadu_ph(b));
    vacc1 = _mm512_sub_ph(vacc1, _mm512_loadu_ph(b + 32));
    b += 64;

    vacc0 = _mm512_mul_ph(vacc0, vacc0);
    vacc1 = _mm512_mul_ph(vacc1, vacc1);


    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_sub_ph(vacc, _mm512_loadu_ph(b));
    b += 32;

    vacc = _mm512_mul_ph(vacc, vacc);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_sub_ph(vmask, vacc, _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, b)));

    vacc = _mm512_maskz_mul_ph(vmask, vacc, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vsqrdiffc_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h vb = _mm512_castsi512_ph(_mm512_set1_epi16(*b));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_sub_ph(vacc0, vb);
    vacc1 = _mm512_sub_ph(vacc1, vb);

    vacc0 = _mm512_mul_ph(vacc0, vacc0);
    vacc1 = _mm512_mul_ph(vacc1, vacc1);


    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_sub_ph(vacc, vb);
    vacc = _mm512_mul_ph(vacc, vacc);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_sub_ph(vmask, vacc, vb);
    vacc = _mm512_maskz_mul_ph(vmask, vacc, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vsub_minmax_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h voutput_min = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
  const __m512h voutput_max = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_sub_ph(vacc0, _mm512_loadu_ph(b));
    vacc1 = _mm512_sub_ph(vacc1, _mm512_loadu_ph(b + 32));
    b += 64;


    vacc0 = _mm512_max_ph(voutput_min, vacc0);
    vacc1 = _mm512_max_ph(voutput_min, vacc1);

    vacc0 = _mm512_min_ph(voutput_max, vacc0);
    vacc1 = _mm512_min_ph(voutput_max, vacc1);

    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_sub_ph(vacc, _mm512_loadu_ph(b));
    b += 32;

    vacc = _mm512_max_ph(voutput_min, vacc);
    vacc = _mm512_min_ph(voutput_max, vacc);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_sub_ph(vmask, vacc, _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, b)));

    vacc = _mm512_maskz_max_ph(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ph(vmask, voutput_max, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vsubc_minmax_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h voutput_min = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
  const __m512h voutput_max = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));
  const __m512h vb = _mm512_castsi512_ph(_mm512_set1_epi16(*b));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_sub_ph(vacc0, vb);
    vacc1 = _mm512_sub_ph(vacc1, vb);


    vacc0 = _mm512_max_ph(voutput_min, vacc0);
    vacc1 = _mm512_max_ph(voutput_min, vacc1);

    vacc0 = _mm512_min_ph(voutput_max, vacc0);
    vacc1 = _mm512_min_ph(voutput_max, vacc1);

    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_sub_ph(vacc, vb);
    vacc = _mm512_max_ph(voutput_min, vacc);
    vacc = _mm512_min_ph(voutput_max, vacc);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_sub_ph(vmask, vacc, vb);
    vacc = _mm512_maskz_max_ph(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ph(vmask, voutput_max, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}
