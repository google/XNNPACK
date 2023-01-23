// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/math-stubs.h>


// Table of exp2(k / 64) values decremented (as integer) by (k << 17), k = 0..63
extern XNN_INTERNAL const float xnn_table_exp2minus_k_over_64[64];

void xnn_math_f32_sigmoid__avx2_rr1_lut64_p2_gather_nr2fma1adj(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (8 * sizeof(float)) == 0);

  // Floating-point mask with only the sign bit set
  const __m256 vsign_mask = _mm256_set1_ps(-0.0f);
  // Large number such that ulp(magic bias) == exp2(-6)
  const __m256 vmagic_bias = _mm256_set1_ps(0x1.800000p17f);
  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p0f);
  // Mask for the lowest 6 bits
  const __m256 vindex_mask = _mm256_castsi256_ps(_mm256_set1_epi32(INT32_C(0x3F)));
  const __m256 vminus_ln2 = _mm256_set1_ps(-0x1.62E43p-1f);
  // Coefficient of polynomial approximation of exp(t) ~ 1 + t * (1 + t * c2) on [-log(2)/128, log(2)/128]
  const __m256 vc2 = _mm256_set1_ps(0x1.FFFF0Ap-2f);
  const __m256 vone = _mm256_set1_ps(1.0f);
  // The smallest x for which sigmoidf(x) is normalized.
  // This number is also the smallest x for which expf(x) is normalized.
  const __m256 vdenorm_cutoff = _mm256_set1_ps(-0x1.5D589Ep+6f);

  for (; n != 0; n -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(input);

    // General structure of the algorithm:
    //
    //           / exp(x) / (1 + exp(x)) if x <= 0
    //   f[x] :=
    //           \ 1 - f[-x] if x >= 0
    //
    // First we compute f[z] := exp(z) / (1 + exp(z)) where z = -abs(x), then replace result with 1 - f[z] if x >= 0.
    const __m256 vz = _mm256_or_ps(vx, vsign_mask);

    // Compute reduced argument n := round(z / log(2), 6).
    // We do it by adding a large number (magic bias), which cause rounding of the result to 6 fractional bits, then
    // subtracing the large number back. The addition is combined with multiplication by log2e into a single FMA
    // instruction. The trick with adding large number is valid only within certain bounds (|z / log(2)| <= 2**16, i.e.
    // |z| <= 0x1.62E43p+15 = 45426.09375), but that is acceptable, because inputs x outside of [-87.336544, 17.328678]
    // (i.e. z outsize [87.336544, 0]) underflow or saturate sigmoidf(x). We fixup the result  for such inputs at the
    // very end of the algorithm.
    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);

    // Create a floating-point number s (scale) such that s := 2**n for such inputs that sigmoidf(z) is normalized,
    // i.e. -87.33642 <= z <= 0. As n has 6 fractional bits, we split s == 2**n = 2**int(n) * 2**frac(n). We create s
    // in two steps:
    // 1. Fetch 2**frac(n) from the table using the 6 low bits of n, as integer. Note that the fetched values are in
    //    the [1.0, 2.0) range, i.e. their floating-point exponent is 0.
    // 2. Adjust fecthed value by addition of int(n) to its floating-point exponent. The result is always a normalized
    //    number, because for -87.33642 <= z <= 0 (inputs for which sigmoidf(z) is normalized) we have
    //    -126 <= int(n) <= 0, and thus the adjusted exponent is not lower than -126.
    //
    // Shift bits 6:14 into 23:31 (position of floating-point exponent).
    __m256i ve = _mm256_slli_epi32(_mm256_castps_si256(vn), 17);

    // Use bits 0:6 of n, as integer, as an index for table lookup of l := 2**frac(n).
    const __m256i vidx = _mm256_castps_si256(_mm256_and_ps(vn, vindex_mask));
    const __m256i vl = _mm256_i32gather_epi32((const int*) xnn_table_exp2minus_k_over_64, vidx, sizeof(float));
    // Adjust exponent of the value l fetched from the table to get the final s value.
    const __m256 vs = _mm256_castsi256_ps(_mm256_add_epi32(vl, ve));

    // Subtract the large number back to get the final n := round(z / log(2), 6) as a floating-point number.
    vn = _mm256_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := z - n * log(2).
    const __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    // Compute degree-2 polynomial approximation for exp(t) on [-log(2)/128, log(2)/128].
    //   P(t) = 1 + t * (1 + t * c2) = 1 + (t + t * (t * c2)) = 1 + p
    __m256 vp = _mm256_mul_ps(vt, vc2);
    vp = _mm256_fmadd_ps(vt, vp, vt);

    // Reconstruct the exp(z) value:
    //   e = s * (1 + t * (1 + t * c2))
    //     = s * (1 + p)
    //     = s + s * p
    const __m256 vy = _mm256_fmadd_ps(vs, vp, vs);

    // Denominator of the sigmoid fraction: 1.0 + exp(z)
    const __m256 vd = _mm256_add_ps(vy, vone);

    // Use Newton-Raphson method (2 iterations) to compute reciprocal of denominator.
    // Note: 1 < d <= 2, because z >= 0.0 and 0 < exp(-z) <= 1.0.
    // Thus the reciprocal of the denominator never overflows.
    __m256 vr = _mm256_rcp_ps(vd);
    vr = _mm256_fmadd_ps(_mm256_fnmadd_ps(vr, vd, vone), vr, vr);
    vr = _mm256_fmadd_ps(_mm256_fnmadd_ps(vr, vd, vone), vr, vr);

    // Reconstruct sigmoid(z) = exp(z) / (1.0 + exp(z)) with adjustment to match IEEE division result
    __m256 vf = _mm256_mul_ps(vy, vr);
    vf = _mm256_fmadd_ps(_mm256_fnmadd_ps(vf, vd, vy), vr, vf);

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = _mm256_andnot_ps(_mm256_cmp_ps(vz, vdenorm_cutoff, _CMP_LT_OS), vf);

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(z) : 1.0 - sigmoid(z)
    vf = _mm256_blendv_ps(_mm256_sub_ps(vone, vf), vf, vx);

    _mm256_storeu_ps(output, vf);

    input += 8;
    output += 8;
  }
}
