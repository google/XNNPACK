// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f32_expm1minus__scalar_rr2_p6(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  // Large number such that ulp(magic bias) == 1 and magic bias === 127 mod 2**22.
  const float vmagic_bias = 0x1.8000FEp23f;
  const float vlog2e = 0x1.715476p+0f;
  // The largest x for which expm1f(x) is saturated at -1.0f.
  const float vsat_cutoff = -0x1.154246p+4f;
  // Last 5 bits are zeroes
  const float vminus_ln2_hi = -0x1.62E440p-1f;
  const float vminus_ln2_lo = 0x1.0105C6p-21f;
  // Coefficient of polynomial approximation
  //   exp(t) - 1 ~ t * (1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
  // on [-log(2)/2, log(2)/2]
  const float vc6 = 0x1.6b7338p-10f;
  const float vc5 = 0x1.12278Ep-7f;
  const float vc4 = 0x1.555716p-5f;
  const float vc3 = 0x1.5554B0p-3f;
  const float vc2 = 0x1.FFFFFEp-2f;
  const float vone = 1.0f;

  for (; n != 0; n -= sizeof(float)) {
    float vx = *input++;

    // Compute reduced argument n := round(x / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of the result to integer, then subtracing
    // the large number back. The trick with adding large number is valid only within certain bounds
    // (|x / log(2)| <= 2**22, i.e. |x| <= 0x1.62E43p+21 = 2907270.0), but that is acceptable, because inputs x are
    // restricted to [-17.328680, 0].
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    float vn = vx * vlog2e + vmagic_bias;

    // Create a floating-point number s (scale) such that s == 2**n for valid inputs, i.e.
    // -17.328680 <= x <= 0.0, and -25 <= n <= 0 accordingly.
    float vs = uint32_as_float(float_as_uint32(vn) << 23);

    // Subtract the large number back to get final n := round(x / log(2)).
    vn -= vmagic_bias;

    // Compute reduced argument t := x - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    float vt = vn * vminus_ln2_hi + vx;
    vt = vn * vminus_ln2_lo + vt;

    // The function saturates at -1 for large negative inputs: expm1f(x) == -1.0f for x <= sat_cutoff ~= -17.328680.
    // To guarantee this behaviour, we zero out s (scale) and t (reduced argument) for x <= sat_cutoff.
    if XNN_UNPREDICTABLE(vx <= vsat_cutoff) {
      vs = 0.0f;
      vt = 0.0f;
    }

    // Compute degree-6 polynomial approximation for exp(t) - 1 on [-log(2)/2, log(2)/2].
    //   P(t) = t * (1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
    //        = t + t * (t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))) = t + t * p
    float vp = vc6 * vt + vc5;
    vp = vp * vt + vc4;
    vp = vp * vt + vc3;
    vp = vp * vt + vc2;
    vp *= vt;

    // Reconstruct the exp(x) - 1 value:
    //   exp(x) - 1 = s * (1 + t * (1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))) - 1
    //              = (s - 1) + s * (t + t * p)
    //              = ((t * s) + (t * s) * p) + (s - 1)
    vt *= vs;
    const float vsm1 = vs - vone;
    vp = vp * vt + vt;
    const float vf = vp + vsm1;

    *output++ = vf;
  }
}
