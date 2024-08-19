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


void xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u80_acc2(
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
  __m256 vacc1 = _mm256_setzero_ps();
  for (; batch >= 80 * sizeof(uint16_t); batch -= 80 * sizeof(uint16_t)) {
    const __m256 vi0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    const __m256 vi1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    const __m256 vi2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 16)));
    const __m256 vi3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 24)));
    const __m256 vi4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 32)));
    const __m256 vi5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 40)));
    const __m256 vi6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 48)));
    const __m256 vi7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 56)));
    const __m256 vi8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 64)));
    const __m256 vi9 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 72)));
    i += 80;

    const __m256 vx0 = _mm256_sub_ps(vi0, vi_max);
    const __m256 vx1 = _mm256_sub_ps(vi1, vi_max);
    const __m256 vx2 = _mm256_sub_ps(vi2, vi_max);
    const __m256 vx3 = _mm256_sub_ps(vi3, vi_max);
    const __m256 vx4 = _mm256_sub_ps(vi4, vi_max);
    const __m256 vx5 = _mm256_sub_ps(vi5, vi_max);
    const __m256 vx6 = _mm256_sub_ps(vi6, vi_max);
    const __m256 vx7 = _mm256_sub_ps(vi7, vi_max);
    const __m256 vx8 = _mm256_sub_ps(vi8, vi_max);
    const __m256 vx9 = _mm256_sub_ps(vi9, vi_max);

    __m256 vn0 = _mm256_fmadd_ps(vx0, vlog2e, vmagic_bias);
    __m256 vn1 = _mm256_fmadd_ps(vx1, vlog2e, vmagic_bias);
    __m256 vn2 = _mm256_fmadd_ps(vx2, vlog2e, vmagic_bias);
    __m256 vn3 = _mm256_fmadd_ps(vx3, vlog2e, vmagic_bias);
    __m256 vn4 = _mm256_fmadd_ps(vx4, vlog2e, vmagic_bias);
    __m256 vn5 = _mm256_fmadd_ps(vx5, vlog2e, vmagic_bias);
    __m256 vn6 = _mm256_fmadd_ps(vx6, vlog2e, vmagic_bias);
    __m256 vn7 = _mm256_fmadd_ps(vx7, vlog2e, vmagic_bias);
    __m256 vn8 = _mm256_fmadd_ps(vx8, vlog2e, vmagic_bias);
    __m256 vn9 = _mm256_fmadd_ps(vx9, vlog2e, vmagic_bias);

    const __m256 vs0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn0), 23));
    const __m256 vs1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn1), 23));
    const __m256 vs2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn2), 23));
    const __m256 vs3 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn3), 23));
    const __m256 vs4 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn4), 23));
    const __m256 vs5 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn5), 23));
    const __m256 vs6 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn6), 23));
    const __m256 vs7 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn7), 23));
    const __m256 vs8 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn8), 23));
    const __m256 vs9 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn9), 23));

    vn0 = _mm256_sub_ps(vn0, vmagic_bias);
    vn1 = _mm256_sub_ps(vn1, vmagic_bias);
    vn2 = _mm256_sub_ps(vn2, vmagic_bias);
    vn3 = _mm256_sub_ps(vn3, vmagic_bias);
    vn4 = _mm256_sub_ps(vn4, vmagic_bias);
    vn5 = _mm256_sub_ps(vn5, vmagic_bias);
    vn6 = _mm256_sub_ps(vn6, vmagic_bias);
    vn7 = _mm256_sub_ps(vn7, vmagic_bias);
    vn8 = _mm256_sub_ps(vn8, vmagic_bias);
    vn9 = _mm256_sub_ps(vn9, vmagic_bias);

    __m256 vt0 = _mm256_fmadd_ps(vn0, vminus_ln2, vx0);
    __m256 vt1 = _mm256_fmadd_ps(vn1, vminus_ln2, vx1);
    __m256 vt2 = _mm256_fmadd_ps(vn2, vminus_ln2, vx2);
    __m256 vt3 = _mm256_fmadd_ps(vn3, vminus_ln2, vx3);
    __m256 vt4 = _mm256_fmadd_ps(vn4, vminus_ln2, vx4);
    __m256 vt5 = _mm256_fmadd_ps(vn5, vminus_ln2, vx5);
    __m256 vt6 = _mm256_fmadd_ps(vn6, vminus_ln2, vx6);
    __m256 vt7 = _mm256_fmadd_ps(vn7, vminus_ln2, vx7);
    __m256 vt8 = _mm256_fmadd_ps(vn8, vminus_ln2, vx8);
    __m256 vt9 = _mm256_fmadd_ps(vn9, vminus_ln2, vx9);

    const __m256 vp0 = _mm256_fmadd_ps(vc2, vt0, vc1);
    const __m256 vp1 = _mm256_fmadd_ps(vc2, vt1, vc1);
    const __m256 vp2 = _mm256_fmadd_ps(vc2, vt2, vc1);
    const __m256 vp3 = _mm256_fmadd_ps(vc2, vt3, vc1);
    const __m256 vp4 = _mm256_fmadd_ps(vc2, vt4, vc1);
    const __m256 vp5 = _mm256_fmadd_ps(vc2, vt5, vc1);
    const __m256 vp6 = _mm256_fmadd_ps(vc2, vt6, vc1);
    const __m256 vp7 = _mm256_fmadd_ps(vc2, vt7, vc1);
    const __m256 vp8 = _mm256_fmadd_ps(vc2, vt8, vc1);
    const __m256 vp9 = _mm256_fmadd_ps(vc2, vt9, vc1);

    vt0 = _mm256_mul_ps(vt0, vs0);
    vt1 = _mm256_mul_ps(vt1, vs1);
    vt2 = _mm256_mul_ps(vt2, vs2);
    vt3 = _mm256_mul_ps(vt3, vs3);
    vt4 = _mm256_mul_ps(vt4, vs4);
    vt5 = _mm256_mul_ps(vt5, vs5);
    vt6 = _mm256_mul_ps(vt6, vs6);
    vt7 = _mm256_mul_ps(vt7, vs7);
    vt8 = _mm256_mul_ps(vt8, vs8);
    vt9 = _mm256_mul_ps(vt9, vs9);

    __m256 vf0 = _mm256_fmadd_ps(vt0, vp0, vs0);
    __m256 vf1 = _mm256_fmadd_ps(vt1, vp1, vs1);
    __m256 vf2 = _mm256_fmadd_ps(vt2, vp2, vs2);
    __m256 vf3 = _mm256_fmadd_ps(vt3, vp3, vs3);
    __m256 vf4 = _mm256_fmadd_ps(vt4, vp4, vs4);
    __m256 vf5 = _mm256_fmadd_ps(vt5, vp5, vs5);
    __m256 vf6 = _mm256_fmadd_ps(vt6, vp6, vs6);
    __m256 vf7 = _mm256_fmadd_ps(vt7, vp7, vs7);
    __m256 vf8 = _mm256_fmadd_ps(vt8, vp8, vs8);
    __m256 vf9 = _mm256_fmadd_ps(vt9, vp9, vs9);

    vf0 = _mm256_andnot_ps(_mm256_cmp_ps(vx0, vdenorm_cutoff, _CMP_LT_OS), vf0);
    vf1 = _mm256_andnot_ps(_mm256_cmp_ps(vx1, vdenorm_cutoff, _CMP_LT_OS), vf1);
    vf2 = _mm256_andnot_ps(_mm256_cmp_ps(vx2, vdenorm_cutoff, _CMP_LT_OS), vf2);
    vf3 = _mm256_andnot_ps(_mm256_cmp_ps(vx3, vdenorm_cutoff, _CMP_LT_OS), vf3);
    vf4 = _mm256_andnot_ps(_mm256_cmp_ps(vx4, vdenorm_cutoff, _CMP_LT_OS), vf4);
    vf5 = _mm256_andnot_ps(_mm256_cmp_ps(vx5, vdenorm_cutoff, _CMP_LT_OS), vf5);
    vf6 = _mm256_andnot_ps(_mm256_cmp_ps(vx6, vdenorm_cutoff, _CMP_LT_OS), vf6);
    vf7 = _mm256_andnot_ps(_mm256_cmp_ps(vx7, vdenorm_cutoff, _CMP_LT_OS), vf7);
    vf8 = _mm256_andnot_ps(_mm256_cmp_ps(vx8, vdenorm_cutoff, _CMP_LT_OS), vf8);
    vf9 = _mm256_andnot_ps(_mm256_cmp_ps(vx9, vdenorm_cutoff, _CMP_LT_OS), vf9);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vf0, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vf1, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 16), _mm256_cvtps_ph(vf2, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 24), _mm256_cvtps_ph(vf3, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 32), _mm256_cvtps_ph(vf4, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 40), _mm256_cvtps_ph(vf5, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 48), _mm256_cvtps_ph(vf6, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 56), _mm256_cvtps_ph(vf7, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 64), _mm256_cvtps_ph(vf8, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 72), _mm256_cvtps_ph(vf9, _MM_FROUND_TO_NEAREST_INT));
    o += 80;

    vacc0 = _mm256_add_ps(vacc0, vf0);
    vacc1 = _mm256_add_ps(vacc1, vf1);
    vacc0 = _mm256_add_ps(vacc0, vf2);
    vacc1 = _mm256_add_ps(vacc1, vf3);
    vacc0 = _mm256_add_ps(vacc0, vf4);
    vacc1 = _mm256_add_ps(vacc1, vf5);
    vacc0 = _mm256_add_ps(vacc0, vf6);
    vacc1 = _mm256_add_ps(vacc1, vf7);
    vacc0 = _mm256_add_ps(vacc0, vf8);
    vacc1 = _mm256_add_ps(vacc1, vf9);
  }
  vacc0 = _mm256_add_ps(vacc0, vacc1);

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
