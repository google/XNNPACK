// Auto-generated file. Do not edit!
//   Template: src/f32-velu/avx2-rr1-lut8-p4-perm.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

void xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x8(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n % sizeof(float) == 0);

  const __m256 vprescale = _mm256_broadcast_ps((const __m128*) params->sse.prescale);
  const __m256 valpha = _mm256_broadcast_ps((const __m128*) params->sse.alpha);
  const __m256 vbeta = _mm256_broadcast_ps((const __m128*) params->sse.beta);

  const __m256 vsat_cutoff = _mm256_set1_ps(-0x1.154246p+4f);
  const __m256 vmagic_bias = _mm256_set1_ps(0x1.800000p20f);
  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p+0f);
  const __m256i vtable = _mm256_set_epi32(
    0x3F7AC0C7, 0x3F7744FD, 0x3F75672A, 0x3F7504F3, 0x3F75FED7, 0x3F7837F0, 0x3F7B95C2, 0x3F800000);
  const __m256 vminus_ln2 = _mm256_set1_ps(-0x1.62E43p-1f);
  const __m256 vc4 = _mm256_set1_ps(0x1.5558ECp-5f);
  const __m256 vc3 = _mm256_set1_ps(0x1.555C20p-3f);
  const __m256 vc2 = _mm256_set1_ps(0x1.000000p-1f);

  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    __m256 vx = _mm256_loadu_ps(x);
    x += 8;

    const __m256 vz = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx, vprescale));

    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m256i ven = _mm256_slli_epi32(_mm256_castps_si256(vn), 20);
    const __m256i vl = _mm256_permutevar8x32_epi32(vtable, _mm256_castps_si256(vn));
    __m256 vs = _mm256_castsi256_ps(_mm256_add_epi32(vl, ven));
    vn = _mm256_sub_ps(vn, vmagic_bias);

    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    __m256 vp = _mm256_fmadd_ps(vc4, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_mul_ps(vp, vt);

    vt = _mm256_mul_ps(vt, vs);
    vs = _mm256_fmsub_ps(vs, valpha, valpha);
    vp = _mm256_fmadd_ps(vp, vt, vt);
    const __m256 ve = _mm256_fmadd_ps(vp, valpha, vs);

    vx = _mm256_mul_ps(vx, vbeta);
    const __m256 vy = _mm256_blendv_ps(vx, ve, vx);

    _mm256_storeu_ps(y, vy);
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(float));
    assert(n <= 7 * sizeof(float));
    __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7] - n));

    __m256 vx = _mm256_maskload_ps(x, vmask);

    const __m256 vz = _mm256_max_ps(vsat_cutoff, _mm256_mul_ps(vx, vprescale));

    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m256i ven = _mm256_slli_epi32(_mm256_castps_si256(vn), 20);
    const __m256i vl = _mm256_permutevar8x32_epi32(vtable, _mm256_castps_si256(vn));
    __m256 vs = _mm256_castsi256_ps(_mm256_add_epi32(vl, ven));
    vn = _mm256_sub_ps(vn, vmagic_bias);

    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    __m256 vp = _mm256_fmadd_ps(vc4, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_mul_ps(vp, vt);

    vt = _mm256_mul_ps(vt, vs);
    vs = _mm256_fmsub_ps(vs, valpha, valpha);
    vp = _mm256_fmadd_ps(vp, vt, vt);
    const __m256 ve = _mm256_fmadd_ps(vp, valpha, vs);

    vx = _mm256_mul_ps(vx, vbeta);
    const __m256 vy = _mm256_blendv_ps(vx, ve, vx);

    #if XNN_COMPILER_HAS_FEATURE(memory_sanitizer)
      __m128 vy_lo = _mm256_castps256_ps128(vy);
      if (n & (4 * sizeof(float))) {
        _mm_storeu_ps(y, vy_lo);
        vy_lo = _mm256_extractf128_ps(vy, 1);
        y += 4;
      }
      if (n & (2 * sizeof(float))) {
        _mm_storel_pi((__m64*) y, vy_lo);
        vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
        y += 2;
      }
      if (n & (1 * sizeof(float))) {
        _mm_store_ss(y, vy_lo);
      }
    #else
      // Triggers spurious MSan failures in the calling code.
      _mm256_maskstore_ps(y, vmask, vy);
    #endif
  }
}
