// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <math.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f32_sigmoid__scalar_rr2_p5_div(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % sizeof(float) == 0);

  // Large number such that ulp(magic bias) == 1 and magic bias === 127 mod 2**22.
  const float vmagic_bias = 0x1.8000FEp23f;
  const float vminus_log2e = -0x1.715476p+0f;
  // Last 7 bits are zeroes
  const float vln2_hi = 0x1.62E400p-1f;
  const float vln2_lo = 0x1.7F7D1Cp-20f;
  // Coefficient of polynomial approximation of
  // exp(-t) ~ 1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))) on [-log(2)/2, log(2)/2]
  const float vc5 = -0x1.0F9F9Cp-7f;
  const float vc4 =  0x1.573A1Ap-5f;
  const float vc3 = -0x1.555A80p-3f;
  const float vc2 =  0x1.FFFDC6p-2f;
  const float vc1 = -0x1.FFFFF6p-1f;
  const float vone = 1.0f;
  // The largest z for which sigmoidf(-z) is normalized.
  // This number is also the largest z for which expf(-z) is normalized.
  const float vdenorm_cutoff = 0x1.5D589Ep+6f;

  for (; n != 0; n -= sizeof(float)) {
    const float vx = *input++;

    // General structure of the algorithm:
    //
    //           / exp(x) / (1 + exp(x)) if x <= 0
    //   f[x] :=
    //           \ 1 - f[-x] if x >= 0
    //
    // First we compute f[-z] := exp(-z) / (1 + exp(-z)) where z = abs(x),
    // then replace result with 1 - f[-z] if x >= 0.
    const float vz = fabsf(vx);

    // Compute reduced argument n := round(-z / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of the result to integer, then subtracing
    // the large number back. The trick with adding large number is valid only within certain bounds
    // (|-z / log(2)| <= 2**22, i.e. |z| <= 0x1.62E43p+21 = 2907270.0), but that is acceptable, because inputs x
    // outside of [-87.336544, 17.328678] (i.e. z outsize [0, 87.336544]) underflow or saturate sigmoidf(x). We fixup
    // the result for such inputs at the very end of the algorithm.
    float vn = vz * vminus_log2e + vmagic_bias;

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -87.336544 <= -z <= 0.0, and -126 <= n <= 0 accordingly.
    const float vs = uint32_as_float(float_as_uint32(vn) << 23);

    // Subtract the large number back to get the final n := round(-z / log(2)) as a floating-point number.
    vn -= vmagic_bias;

    // Compute reduced argument t := z + n * log(2). Note that -t = -z - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    float vt = vn * vln2_hi + vz;
    vt = vn * vln2_lo + vt;

    // Compute degree-5 polynomial approximation for exp(-t) on [-log(2)/2, log(2)/2]:
    //   P(t) = 1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))) = 1 + t * p
    float vp = vt * vc5 + vc4;
    vp = vt * vp + vc3;
    vp = vt * vp + vc2;
    vp = vt * vp + vc1;

    // Reconstruct the exp(-z) value:
    //   e = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s * (1 + t * p)
    //     = s + (t * s) * p
    vt *= vs;
    const float ve = vt * vp + vs;

    // Reconstruct sigmoid(-z) = exp(-z) / (1.0 + exp(-z))
    float vf = ve / (ve + vone);

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    if XNN_UNPREDICTABLE(vz > vdenorm_cutoff) {
      vf = 0.0f;
    }

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(-z) : 1.0 - sigmoid(-z)
    if XNN_UNPREDICTABLE(vx > 0.0f) {
      vf = vone - vf;
    }

    *output++ = vf;
  }
}
