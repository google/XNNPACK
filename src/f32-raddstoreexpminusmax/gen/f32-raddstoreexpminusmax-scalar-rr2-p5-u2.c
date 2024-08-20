// Auto-generated file. Do not edit!
//   Template: src/f32-raddstoreexpminusmax/scalar-rr2-p5.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/raddstoreexpminusmax.h"


void xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u2(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const union xnn_f32_expminus_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  const float vlog2e = 0x1.715476p+0f;
  const float vmagic_bias = 0x1.8000FEp23f;
  const float vminus_ln2_hi = -0x1.62E400p-1f;
  const float vminus_ln2_lo = -0x1.7F7D1Cp-20f;
  const float vc5 = 0x1.0F9F9Cp-7f;
  const float vc4 = 0x1.573A1Ap-5f;
  const float vc3 = 0x1.555A80p-3f;
  const float vc2 = 0x1.FFFDC6p-2f;
  const float vc1 = 0x1.FFFFF6p-1f;
  const float vdenorm_cutoff = -0x1.5D589Ep6f;

  const float vi_max = *max;

  float vacc0 = 0.0f;
  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    // Load 2 inputs at a time.
    const float vi0 = input[0];
    const float vi1 = input[1];
    input += 2;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const float vx0 = vi0 - vi_max;
    const float vx1 = vi1 - vi_max;

    // Compute reduced argument n := round(x / log(2)).
    // We do it by adding a large number (magic bias) to the product x * (1/log(2)), which cause rounding of the result
    // to an integer, then subtracing the large number back. The trick with adding large number is valid only within
    // certain bounds (|x| <= 2**22), but that's ok, because inputs outside of [-87.336540, 0.0] underflow expf(x)
    // anyway. We fixup the result for such inputs at the very end of the algorithm.
    float vn0 = vx0 * vlog2e + vmagic_bias;
    float vn1 = vx1 * vlog2e + vmagic_bias;

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= n <= 0 accordingly.
    const float vs0 = uint32_as_float(float_as_uint32(vn0) << 23);
    const float vs1 = uint32_as_float(float_as_uint32(vn1) << 23);

    // Subtract the large number back to get final n := round(x / log(2)).
    vn0 -= vmagic_bias;
    vn1 -= vmagic_bias;

    // Compute reduced argument t := x - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    float vt0 = vn0 * vminus_ln2_hi + vx0;
    float vt1 = vn1 * vminus_ln2_hi + vx1;

    vt0 = vn0 * vminus_ln2_lo + vt0;
    vt1 = vn1 * vminus_ln2_lo + vt1;

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    float vp0 = vc5 * vt0 + vc4;
    float vp1 = vc5 * vt1 + vc4;

    vp0 = vp0 * vt0 + vc3;
    vp1 = vp1 * vt1 + vc3;

    vp0 = vp0 * vt0 + vc2;
    vp1 = vp1 * vt1 + vc2;

    vp0 = vp0 * vt0 + vc1;
    vp1 = vp1 * vt1 + vc1;

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt0 *= vs0;
    vt1 *= vs1;

    float vf0 = vt0 * vp0 + vs0;
    float vf1 = vt1 * vp1 + vs1;

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    if XNN_UNPREDICTABLE(vx0 < vdenorm_cutoff) {
      vf0 = 0.0f;
    }
    if XNN_UNPREDICTABLE(vx1 < vdenorm_cutoff) {
      vf1 = 0.0f;
    }

    // Store 2 outputs at a time.
    output[0] = vf0;
    output[1] = vf1;
    output += 2;

    // Accumulate computed exponents.
    vacc0 += vf0;
    vacc0 += vf1;
  }

  float vacc = vacc0;
  for (; batch >= sizeof(float); batch -= sizeof(float)) {
    // Load 1 input at a time.
    const float vi = *input++;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const float vx = vi - vi_max;

    // Compute reduced argument n := round(x / log(2)).
    // We do it by adding a large number (magic bias) to the product x * (1/log(2)), which cause rounding of the result
    // to an integer, then subtracing the large number back. The trick with adding large number is valid only within
    // certain bounds (|x| <= 2**22), but that's ok, because inputs outside of [-87.336540, 0.0] underflow expf(x)
    // anyway. We fixup the result for such inputs at the very end of the algorithm.
    float vn = vx * vlog2e + vmagic_bias;

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= n <= 0 accordingly.
    const float vs = uint32_as_float(float_as_uint32(vn) << 23);

    // Subtract the large number back to get final n := round(x / log(2)).
    vn -= vmagic_bias;

    // Compute reduced argument t := x - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    float vt = vn * vminus_ln2_hi + vx;
    vt = vn * vminus_ln2_lo + vt;

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    float vp = vc5 * vt + vc4;
    vp = vp * vt + vc3;
    vp = vp * vt + vc2;
    vp = vp * vt + vc1;

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt *= vs;
    float vf = vt * vp + vs;

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    if XNN_UNPREDICTABLE(vx < vdenorm_cutoff) {
      vf = 0.0f;
    }

    // Store 1 output at a time.
    *output++ = vf;

    // Accumulate computed exponents.
    vacc += vf;
  }
  *sum = vacc;
}
