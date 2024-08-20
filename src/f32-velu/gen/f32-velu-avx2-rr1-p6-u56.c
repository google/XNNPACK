// Auto-generated file. Do not edit!
//   Template: src/f32-velu/avx2-rr1-p6.c.in
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


void xnn_f32_velu_ukernel__avx2_rr1_p6_u56(
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
  const __m256 vmagic_bias = _mm256_set1_ps(0x1.8000FEp23f);
  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p+0f);
  const __m256 vminus_ln2 = _mm256_set1_ps(-0x1.62E430p-1f);
  const __m256 vc6 = _mm256_set1_ps(0x1.6B7338p-10f);
  const __m256 vc5 = _mm256_set1_ps(0x1.12278Ep-7f);
  const __m256 vc4 = _mm256_set1_ps(0x1.555716p-5f);
  const __m256 vc3 = _mm256_set1_ps(0x1.5554B0p-3f);
  const __m256 vc2 = _mm256_set1_ps(0x1.FFFFFEp-2f);

  XNN_FORCE_REALIZATION(vsat_cutoff);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vminus_ln2);
  XNN_FORCE_REALIZATION(vc6);
  XNN_FORCE_REALIZATION(vc5);
  XNN_FORCE_REALIZATION(vc4);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);

  const __m256 vprescale = _mm256_set1_ps(params->scalar.prescale);
  const __m256 valpha = _mm256_set1_ps(params->scalar.alpha);
  const __m256 vbeta = _mm256_set1_ps(params->scalar.beta);

  for (; batch >= 56 * sizeof(float); batch -= 56 * sizeof(float)) {
    __m256 vx0 = _mm256_loadu_ps(input);
    __m256 vx1 = _mm256_loadu_ps(input + 8);
    __m256 vx2 = _mm256_loadu_ps(input + 16);
    __m256 vx3 = _mm256_loadu_ps(input + 24);
    __m256 vx4 = _mm256_loadu_ps(input + 32);
    __m256 vx5 = _mm256_loadu_ps(input + 40);
    __m256 vx6 = _mm256_loadu_ps(input + 48);
    input += 56;

    const __m256 vz0 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx0, vprescale));
    const __m256 vz1 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx1, vprescale));
    const __m256 vz2 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx2, vprescale));
    const __m256 vz3 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx3, vprescale));
    const __m256 vz4 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx4, vprescale));
    const __m256 vz5 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx5, vprescale));
    const __m256 vz6 = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx6, vprescale));

    __m256 vn0 = _mm256_fmadd_ps(vz0, vlog2e, vmagic_bias);
    __m256 vn1 = _mm256_fmadd_ps(vz1, vlog2e, vmagic_bias);
    __m256 vn2 = _mm256_fmadd_ps(vz2, vlog2e, vmagic_bias);
    __m256 vn3 = _mm256_fmadd_ps(vz3, vlog2e, vmagic_bias);
    __m256 vn4 = _mm256_fmadd_ps(vz4, vlog2e, vmagic_bias);
    __m256 vn5 = _mm256_fmadd_ps(vz5, vlog2e, vmagic_bias);
    __m256 vn6 = _mm256_fmadd_ps(vz6, vlog2e, vmagic_bias);

    __m256 vs0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn0), 23));
    vn0 = _mm256_sub_ps(vn0, vmagic_bias);
    __m256 vs1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn1), 23));
    vn1 = _mm256_sub_ps(vn1, vmagic_bias);
    __m256 vs2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn2), 23));
    vn2 = _mm256_sub_ps(vn2, vmagic_bias);
    __m256 vs3 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn3), 23));
    vn3 = _mm256_sub_ps(vn3, vmagic_bias);
    __m256 vs4 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn4), 23));
    vn4 = _mm256_sub_ps(vn4, vmagic_bias);
    __m256 vs5 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn5), 23));
    vn5 = _mm256_sub_ps(vn5, vmagic_bias);
    __m256 vs6 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn6), 23));
    vn6 = _mm256_sub_ps(vn6, vmagic_bias);

    __m256 vt0 = _mm256_fmadd_ps(vn0, vminus_ln2, vz0);
    __m256 vt1 = _mm256_fmadd_ps(vn1, vminus_ln2, vz1);
    __m256 vt2 = _mm256_fmadd_ps(vn2, vminus_ln2, vz2);
    __m256 vt3 = _mm256_fmadd_ps(vn3, vminus_ln2, vz3);
    __m256 vt4 = _mm256_fmadd_ps(vn4, vminus_ln2, vz4);
    __m256 vt5 = _mm256_fmadd_ps(vn5, vminus_ln2, vz5);
    __m256 vt6 = _mm256_fmadd_ps(vn6, vminus_ln2, vz6);

    __m256 vp0 = _mm256_fmadd_ps(vc6, vt0, vc5);
    __m256 vp1 = _mm256_fmadd_ps(vc6, vt1, vc5);
    __m256 vp2 = _mm256_fmadd_ps(vc6, vt2, vc5);
    __m256 vp3 = _mm256_fmadd_ps(vc6, vt3, vc5);
    __m256 vp4 = _mm256_fmadd_ps(vc6, vt4, vc5);
    __m256 vp5 = _mm256_fmadd_ps(vc6, vt5, vc5);
    __m256 vp6 = _mm256_fmadd_ps(vc6, vt6, vc5);

    vp0 = _mm256_fmadd_ps(vp0, vt0, vc4);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc4);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc4);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc4);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vc4);
    vp5 = _mm256_fmadd_ps(vp5, vt5, vc4);
    vp6 = _mm256_fmadd_ps(vp6, vt6, vc4);

    vp0 = _mm256_fmadd_ps(vp0, vt0, vc3);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc3);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc3);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc3);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vc3);
    vp5 = _mm256_fmadd_ps(vp5, vt5, vc3);
    vp6 = _mm256_fmadd_ps(vp6, vt6, vc3);

    vp0 = _mm256_fmadd_ps(vp0, vt0, vc2);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc2);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc2);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc2);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vc2);
    vp5 = _mm256_fmadd_ps(vp5, vt5, vc2);
    vp6 = _mm256_fmadd_ps(vp6, vt6, vc2);

    vp0 = _mm256_mul_ps(vp0, vt0);
    vt0 = _mm256_mul_ps(vt0, vs0);
    vp1 = _mm256_mul_ps(vp1, vt1);
    vt1 = _mm256_mul_ps(vt1, vs1);
    vp2 = _mm256_mul_ps(vp2, vt2);
    vt2 = _mm256_mul_ps(vt2, vs2);
    vp3 = _mm256_mul_ps(vp3, vt3);
    vt3 = _mm256_mul_ps(vt3, vs3);
    vp4 = _mm256_mul_ps(vp4, vt4);
    vt4 = _mm256_mul_ps(vt4, vs4);
    vp5 = _mm256_mul_ps(vp5, vt5);
    vt5 = _mm256_mul_ps(vt5, vs5);
    vp6 = _mm256_mul_ps(vp6, vt6);
    vt6 = _mm256_mul_ps(vt6, vs6);

    vs0 = _mm256_fmsub_ps(vs0, valpha, valpha);
    vp0 = _mm256_fmadd_ps(vp0, vt0, vt0);
    vs1 = _mm256_fmsub_ps(vs1, valpha, valpha);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vt1);
    vs2 = _mm256_fmsub_ps(vs2, valpha, valpha);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vt2);
    vs3 = _mm256_fmsub_ps(vs3, valpha, valpha);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vt3);
    vs4 = _mm256_fmsub_ps(vs4, valpha, valpha);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vt4);
    vs5 = _mm256_fmsub_ps(vs5, valpha, valpha);
    vp5 = _mm256_fmadd_ps(vp5, vt5, vt5);
    vs6 = _mm256_fmsub_ps(vs6, valpha, valpha);
    vp6 = _mm256_fmadd_ps(vp6, vt6, vt6);

    const __m256 ve0 = _mm256_fmadd_ps(vp0, valpha, vs0);
    vx0 = _mm256_mul_ps(vx0, vbeta);
    const __m256 ve1 = _mm256_fmadd_ps(vp1, valpha, vs1);
    vx1 = _mm256_mul_ps(vx1, vbeta);
    const __m256 ve2 = _mm256_fmadd_ps(vp2, valpha, vs2);
    vx2 = _mm256_mul_ps(vx2, vbeta);
    const __m256 ve3 = _mm256_fmadd_ps(vp3, valpha, vs3);
    vx3 = _mm256_mul_ps(vx3, vbeta);
    const __m256 ve4 = _mm256_fmadd_ps(vp4, valpha, vs4);
    vx4 = _mm256_mul_ps(vx4, vbeta);
    const __m256 ve5 = _mm256_fmadd_ps(vp5, valpha, vs5);
    vx5 = _mm256_mul_ps(vx5, vbeta);
    const __m256 ve6 = _mm256_fmadd_ps(vp6, valpha, vs6);
    vx6 = _mm256_mul_ps(vx6, vbeta);

    const __m256 vy0 = _mm256_blendv_ps(vx0, ve0, vx0);
    const __m256 vy1 = _mm256_blendv_ps(vx1, ve1, vx1);
    const __m256 vy2 = _mm256_blendv_ps(vx2, ve2, vx2);
    const __m256 vy3 = _mm256_blendv_ps(vx3, ve3, vx3);
    const __m256 vy4 = _mm256_blendv_ps(vx4, ve4, vx4);
    const __m256 vy5 = _mm256_blendv_ps(vx5, ve5, vx5);
    const __m256 vy6 = _mm256_blendv_ps(vx6, ve6, vx6);

    _mm256_storeu_ps(output, vy0);
    _mm256_storeu_ps(output + 8, vy1);
    _mm256_storeu_ps(output + 16, vy2);
    _mm256_storeu_ps(output + 24, vy3);
    _mm256_storeu_ps(output + 32, vy4);
    _mm256_storeu_ps(output + 40, vy5);
    _mm256_storeu_ps(output + 48, vy6);
    output += 56;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    __m256 vx = _mm256_loadu_ps(input);
    input += 8;

    const __m256 vz = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx, vprescale));

    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);
    __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));
    vn = _mm256_sub_ps(vn, vmagic_bias);

    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    __m256 vp = _mm256_fmadd_ps(vc6, vt, vc5);
    vp = _mm256_fmadd_ps(vp, vt, vc4);
    vp = _mm256_fmadd_ps(vp, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
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
    __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));
    vn = _mm256_sub_ps(vn, vmagic_bias);

    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    __m256 vp = _mm256_fmadd_ps(vc6, vt, vc5);
    vp = _mm256_fmadd_ps(vp, vt, vc4);
    vp = _mm256_fmadd_ps(vp, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
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
