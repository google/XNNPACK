// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f32_expminus__scalar_rr2_p5(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % sizeof(float) == 0);

  // Large number such that ulp(magic bias) == 1 and magic bias === 127 mod 2**22.
  const float vmagic_bias = 0x1.8000FEp23f;
  const float vlog2e = 0x1.715476p+0f;
  // Last 7 bits are zeroes
  const float vminus_ln2_hi = -0x1.62E400p-1f;
  const float vminus_ln2_lo = -0x1.7F7D1Cp-20f;
  // Coefficient of polynomial approximation
  //   exp(t) ~ 1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
  // on [-log(2)/2, log(2)/2]
  const float vc5 = 0x1.0F9F9Cp-7f;
  const float vc4 = 0x1.573A1Ap-5f;
  const float vc3 = 0x1.555A80p-3f;
  const float vc2 = 0x1.FFFDC6p-2f;
  const float vc1 = 0x1.FFFFF6p-1f;
  // The smallest x for which expf(x) is normalized.
  const float vdenorm_cutoff = -0x1.5D589Ep6f;

  for (; n != 0; n -= sizeof(float)) {
    const float vx = *input++;

    // Compute reduced argument n := round(x / log(2)).
    // We do it by adding a large number (magic bias) to the product x * (1/log(2)), which cause rounding of the result
    // to an integer, then subtracing the large number back. The trick with adding large number is valid only within
    // certain bounds (|x / log(2)| <= 2**22, i.e. |x| <= 0x1.62E43p+21 = 2907270.0), but that is acceptable, because
    // inputs outside of [-87.336540, 0.0] underflow expf(x) anyway. We fixup the result for such inputs at the very
    // end of the algorithm.
    float vn = vx * vlog2e + vmagic_bias;

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= n <= 0 accordingly.
    const float vs = uint32_as_float(float_as_uint32(vn) << 23);

    // Subtract the large number back to get final n := round(x / log(2)) as a floating-point number.
    vn -= vmagic_bias;

    // Compute reduced argument t := x - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    float vt = vn * vminus_ln2_hi + vx;
    vt = vn * vminus_ln2_lo + vt;

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2]:
    //   P(t) = 1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))) = 1 + t * p
    float vp = vc5 * vt + vc4;
    vp = vp * vt + vc3;
    vp = vp * vt + vc2;
    vp = vp * vt + vc1;

    // Reconstruct the exp(x) value:
    //   exp(x) = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //          = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //          = s + (t * s) * p
    vt *= vs;
    float vf = vt * vp + vs;

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    if XNN_UNPREDICTABLE(vx < vdenorm_cutoff) {
      vf = 0.0f;
    }

    *output++ = vf;
  }
}
