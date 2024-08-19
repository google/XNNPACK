// Auto-generated file. Do not edit!
//   Template: src/f32-velu/avx2-rr1-lut16-p3-gather.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/vunary.h"


extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_16[16];

void xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  const __m256 vsat_cutoff = _mm256_set1_ps(-0x1.154246p+4f);
  const __m256 vmagic_bias = _mm256_set1_ps(0x1.800000p19f);
  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p+0f);
  const __m256i vindex_mask = _mm256_set1_epi32(UINT32_C(0xF));
  const __m256 vminus_ln2 = _mm256_set1_ps(-0x1.62E430p-1f);
  const __m256 vc3 = _mm256_set1_ps(0x1.55561Cp-3f);
  const __m256 vc2 = _mm256_set1_ps(0x1.0001ECp-1f);

  XNN_FORCE_REALIZATION(vsat_cutoff);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vindex_mask);
  XNN_FORCE_REALIZATION(vminus_ln2);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);

  const __m256 vprescale = _mm256_set1_ps(params->scalar.prescale);
  const __m256 valpha = _mm256_set1_ps(params->scalar.alpha);
  const __m256 vbeta = _mm256_set1_ps(params->scalar.beta);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    __m256 vx = _mm256_loadu_ps(input);
    input += 8;

    const __m256 vz = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx, vprescale));

    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m256i vidx = _mm256_and_si256(_mm256_castps_si256(vn), vindex_mask);
    const __m256i vl = _mm256_i32gather_epi32(xnn_table_exp2minus_k_over_16, vidx, sizeof(float));

    const __m256i ven = _mm256_slli_epi32(_mm256_castps_si256(vn), 19);
    vn = _mm256_sub_ps(vn, vmagic_bias);

    __m256 vs = _mm256_castsi256_ps(_mm256_add_epi32(vl, ven));
    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    __m256 vp = _mm256_fmadd_ps(vc3, vt, vc2);
    vp = _mm256_mul_ps(vp, vt);

    vt = _mm256_mul_ps(vt, vs);
    vs = _mm256_fmsub_ps(vs, valpha, valpha);
    vp = _mm256_fmadd_ps(vp, vt, vt);
    const __m256 ve = _mm256_fmadd_ps(vp, valpha, vs);

    vx = _mm256_mul_ps(vx, vbeta);
    const __m256 vy = _mm256_blendv_ps(vx, ve, vx);

    _mm256_storeu_ps(output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7] - batch));

    __m256 vx = _mm256_maskload_ps(input, vmask);

    const __m256 vz = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx, vprescale));

    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m256i vidx = _mm256_and_si256(_mm256_castps_si256(vn), vindex_mask);
    const __m256i vl = _mm256_i32gather_epi32(xnn_table_exp2minus_k_over_16, vidx, sizeof(float));

    const __m256i ven = _mm256_slli_epi32(_mm256_castps_si256(vn), 19);
    vn = _mm256_sub_ps(vn, vmagic_bias);

    __m256 vs = _mm256_castsi256_ps(_mm256_add_epi32(vl, ven));
    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    __m256 vp = _mm256_fmadd_ps(vc3, vt, vc2);
    vp = _mm256_mul_ps(vp, vt);

    vt = _mm256_mul_ps(vt, vs);
    vs = _mm256_fmsub_ps(vs, valpha, valpha);
    vp = _mm256_fmadd_ps(vp, vt, vt);
    const __m256 ve = _mm256_fmadd_ps(vp, valpha, vs);

    vx = _mm256_mul_ps(vx, vbeta);
    const __m256 vy = _mm256_blendv_ps(vx, ve, vx);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (batch & (4 * sizeof(float))) {
      _mm_storeu_ps(output, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      output += 4;
    }
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy_lo);
    }
  }
}
