// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <math.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f16_tanh__avx2_expm1_rr1_p3_rcp(
    size_t n,
    const void* input,
    void* output)
{
  assert(n % (8 * sizeof(uint16_t)) == 0);

  // Mask for the sign bit.
  const __m128i vsign_mask = _mm_set1_epi16(0x8000);
  // The largest z for which tanhh(z) is saturated at -1.0f.
  const __m256 vsat_cutoff = _mm256_set1_ps(-0x1.208000p+2f);
  // Large number such that ulp(magic bias) == 0.5 and magic bias === 63.5 mod 2**21.
  const __m256 vmagic_bias = _mm256_set1_ps(0x1.8000FEp+22f);
  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p+0f);
  const __m256 vminus_ln2 = _mm256_set1_ps(-0x1.62E430p-1f);
  // Coefficient of polynomial approximation
  //   exp(2t) - 1 ~ t * (2 + t * (c2 + t * c3))
  // on [-log(2)/4, log(2)/4]
  const __m256 vc3 = _mm256_set1_ps(0x1.560722p+0f);
  const __m256 vc2 = _mm256_set1_ps(0x1.01E2A2p+1f);
  const __m256 vtwo = _mm256_set1_ps(2.0f);
  const __m256 vminus_one = _mm256_set1_ps(-1.0f);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; n != 0; n -= 8 * sizeof(uint16_t)) {
    const __m128i vx = _mm_load_si128((const __m128i*) i);
    i += 8;

    // General structure of the algorithm:
    //
    //           / expm1(2x) / (2 + expm1(2x)) if x <= 0
    //   f[x] :=
    //           \ -f[-x] if x >= 0
    //
    // First we compute f[z] := expm1(2z) / (2 + expm1(2z)) where z = -abs(x),
    // then replace result with -f[z] if x >= 0.
    __m128i vabsx = _mm_or_si128(vx, vsign_mask);
    __m256 vz = _mm256_cvtph_ps(vabsx);

    // Inverted mask for the sign of input: 0x00000000 for negative x, 0x80000000 for positive x.
    const __m128i vinvsignx = _mm_xor_si128(vx, vabsx);

    // The function f[z] saturates at -1 for large inputs: tanhf(x) == -1.0f for x <= sat_cutoff ~= -4.5078125.
    const __m256 vm = _mm256_cmp_ps(vz, vsat_cutoff, _CMP_LE_OS);

    // Compute reduced argument n := round(z / log(2), 1).
    // We do it by adding a large number (magic bias), which cause rounding of the result to integer, then subtracing
    // the large number back. The trick with adding large number is valid only within certain bounds
    // (|-z / log(2)| <= 2**21, i.e. |z| <= 0x1.62E43p+20 = 1453635.0), but that is acceptable, because inputs x
    // outside of [-4.5078125, 4.5078125] (i.e. z outsize [-4.5078125, 0]) saturate tanhh(x). We fixup the result for such
    // inputs at the very end of the algorithm.
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**(2n) for inputs which don't cause underflow, i.e.
    // -4.5078125 <= z <= 0, and -7 <= n <= 0 accordingly.
    const __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));

    // Subtract the large number back to get final n := round(z / log(2), 1) as a floating-point number.
    vn = _mm256_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := z - n * log(2).
    const __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    // Compute degree-3 polynomial approximation for exp(2t) - 1 on [-log(2)/4, log(2)/4].
    //   P(2t) = t * (2 + t * (c2 + t * c3))
    //         = t * p
    __m256 vp = _mm256_fmadd_ps(vc3, vt, vc2);
    vp = _mm256_fmadd_ps(vp, vt, vtwo);

    // Reconstruct the exp(x) - 1 value:
    //   exp(x) - 1 = s * (1 + t * (c1 + t * (c2 + t * c3))) - 1
    //              = (s - 1) + s * t * p
    //              = (s - 1) + (t * s) * p
    const __m256 vts = _mm256_mul_ps(vt, vs);
    const __m256 vsm1 = _mm256_add_ps(vs, vminus_one);
    const __m256 vem1 = _mm256_fmadd_ps(vp, vts, vsm1);

    // Denominator of the tanh fraction: expm1(-2z) + 2
    const __m256 vep1 = _mm256_add_ps(vem1, vtwo);

    // Compute approximate reciprocal of denominator.
    // Note: 2 < exp(-2z) + 1 <= 3, because z >= 0.0 and 0 < exp(-2z) <= 1.0.
    // Thus the reciprocal of the denominator never overflows.
    const __m256 vrep1 = _mm256_rcp_ps(vep1);

    // Reconstruct tanh(-z) := expm1(-2z) / (2 + expm1(-2z))
    __m256 vabsy = _mm256_mul_ps(vem1, vrep1);

    // Saturate tanh(-z) at -1 for large inputs.
    vabsy = _mm256_blendv_ps(vabsy, vminus_one, vm);

    // Reconstruct tanh[x] = sign(x) * tanh[-abs(x)].
    // As tanh[-abs(x)] is negative, flips the sign bit if x is positive.
    __m128i vy = _mm256_cvtps_ph(vabsy, _MM_FROUND_TO_NEAREST_INT);
    vy = _mm_xor_si128(vy, vinvsignx);

    _mm_storeu_si128((__m128i*) o, vy);
    o += 8;
  }
}
