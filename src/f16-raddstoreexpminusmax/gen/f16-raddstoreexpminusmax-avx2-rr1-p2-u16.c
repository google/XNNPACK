// Auto-generated file. Do not edit!
//   Template: src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/raddstoreexpminusmax.h"


void xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u16(
    size_t batch,
    const void* input,
    const void* max,
    void* output,
    void* sum,
    const union xnn_f16_expminus_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p0f);
  const __m256 vmagic_bias = _mm256_set1_ps(0x1.8000FEp23f);
  const __m256 vminus_ln2 = _mm256_set1_ps(-0x1.62E43p-1f);
  const __m256 vc2 = _mm256_set1_ps(0x1.FF3A32p-2f);
  const __m256 vc1 = _mm256_set1_ps(0x1.039E10p+0f);
  const __m256 vdenorm_cutoff = _mm256_set1_ps(-0x1.368000p+3f);

  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vminus_ln2);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vc1);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const __m256 vi_max = _mm256_cvtph_ps(_mm_set1_epi16((short) *((const uint16_t*) max)));

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  __m256 vacc0 = _mm256_setzero_ps();
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const __m256 vi0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    const __m256 vi1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    i += 16;

    const __m256 vx0 = _mm256_sub_ps(vi0, vi_max);
    const __m256 vx1 = _mm256_sub_ps(vi1, vi_max);

    __m256 vn0 = _mm256_fmadd_ps(vx0, vlog2e, vmagic_bias);
    __m256 vn1 = _mm256_fmadd_ps(vx1, vlog2e, vmagic_bias);

    const __m256 vs0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn0), 23));
    const __m256 vs1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn1), 23));

    vn0 = _mm256_sub_ps(vn0, vmagic_bias);
    vn1 = _mm256_sub_ps(vn1, vmagic_bias);

    __m256 vt0 = _mm256_fmadd_ps(vn0, vminus_ln2, vx0);
    __m256 vt1 = _mm256_fmadd_ps(vn1, vminus_ln2, vx1);

    const __m256 vp0 = _mm256_fmadd_ps(vc2, vt0, vc1);
    const __m256 vp1 = _mm256_fmadd_ps(vc2, vt1, vc1);

    vt0 = _mm256_mul_ps(vt0, vs0);
    vt1 = _mm256_mul_ps(vt1, vs1);

    __m256 vf0 = _mm256_fmadd_ps(vt0, vp0, vs0);
    __m256 vf1 = _mm256_fmadd_ps(vt1, vp1, vs1);

    vf0 = _mm256_andnot_ps(_mm256_cmp_ps(vx0, vdenorm_cutoff, _CMP_LT_OS), vf0);
    vf1 = _mm256_andnot_ps(_mm256_cmp_ps(vx1, vdenorm_cutoff, _CMP_LT_OS), vf1);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vf0, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vf1, _MM_FROUND_TO_NEAREST_INT));
    o += 16;

    vacc0 = _mm256_add_ps(vacc0, vf0);
    vacc0 = _mm256_add_ps(vacc0, vf1);
  }

  __m256 vacc = vacc0;
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 vi = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    const __m256 vx = _mm256_sub_ps(vi, vi_max);

    __m256 vn = _mm256_fmadd_ps(vx, vlog2e, vmagic_bias);

    const __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));

    vn = _mm256_sub_ps(vn, vmagic_bias);

    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vx);

    const __m256 vp = _mm256_fmadd_ps(vc2, vt, vc1);
    vt = _mm256_mul_ps(vt, vs);
    __m256 vf = _mm256_fmadd_ps(vt, vp, vs);
    vf = _mm256_andnot_ps(_mm256_cmp_ps(vx, vdenorm_cutoff, _CMP_LT_OS), vf);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT));
    o += 8;

    vacc = _mm256_add_ps(vacc, vf);
  }
  __m128 vacc_lo = _mm_add_ps(_mm256_castps256_ps128(vacc), _mm256_extractf128_ps(vacc, 1));
  if (batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));

    const __m256 vi = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));

    const __m256 vx = _mm256_sub_ps(vi, vi_max);

    __m256 vn = _mm256_fmadd_ps(vx, vlog2e, vmagic_bias);

    const __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));

    vn = _mm256_sub_ps(vn, vmagic_bias);

    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vx);

    const __m256 vp = _mm256_fmadd_ps(vc2, vt, vc1);
    vt = _mm256_mul_ps(vt, vs);
    __m256 vf = _mm256_fmadd_ps(vt, vp, vs);
    vf = _mm256_andnot_ps(_mm256_cmp_ps(vx, vdenorm_cutoff, _CMP_LT_OS), vf);

    __m128i vh = _mm256_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT);
    __m128 vf_lo = _mm256_castps256_ps128(vf);
    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) o, vh);
      vh = _mm_unpackhi_epi64(vh, vh);
      vacc_lo = _mm_add_ps(vacc_lo, vf_lo);
      vf_lo = _mm256_extractf128_ps(vf, 1);
      o += 4;
    }
    if (batch & (2 * sizeof(uint16_t))) {
      _mm_storeu_si32(o, vh);
      vh = _mm_srli_epi64(vh, 32);
      vacc_lo = _mm_blend_ps(_mm_add_ps(vacc_lo, vf_lo), vacc_lo, 0xC);
      vf_lo = _mm_movehl_ps(vf_lo, vf_lo);
      o += 2;
    }
    if (batch & (1 * sizeof(uint16_t))) {
      *o = (uint16_t) _mm_extract_epi16(vh, 0);
      vacc_lo = _mm_add_ss(vacc_lo, vf_lo);
    }
  }
  vacc_lo = _mm_add_ps(vacc_lo, _mm_movehl_ps(vacc_lo, vacc_lo));
  vacc_lo = _mm_add_ss(vacc_lo, _mm_movehdup_ps(vacc_lo));
  *((uint16_t*) sum) = (uint16_t) _mm_extract_epi16(_mm_cvtps_ph(vacc_lo, _MM_FROUND_TO_NEAREST_INT), 0);
  _mm256_zeroupper();
}
