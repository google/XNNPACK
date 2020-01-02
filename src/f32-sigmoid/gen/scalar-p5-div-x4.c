// Auto-generated file. Do not edit!
//   Template: src/f32-sigmoid/scalar-p5-div.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>

#include <fp16/bitcasts.h>


void xnn_f32_sigmoid_ukernel__scalar_p5_div_x4(
    size_t n,
    const float* x,
    float* y,
    const void* params)
{
  assert(n % sizeof(float) == 0);

  const float vmagic_bias = 0x1.8000FEp23f;
  // The largest z for which sigmoidf(-z) is normalized.
  // This number is also the largest z for which expf(-z) is normalized.
  const float vdenorm_cutoff = 0x1.5D589Ep+6f;
  const float vminus_log2e = -0x1.715476p+0f;
  // Last 7 bits are zeroes
  const float vln2_hi = 0x1.62E400p-1f;
  const float vln2_lo = 0x1.7F7D1Cp-20f;
  const float vone = 1.0f;

  const float vc1 = -0x1.FFFFF6p-1f;
  const float vc2 =  0x1.FFFDC6p-2f;
  const float vc3 = -0x1.555A80p-3f;
  const float vc4 =  0x1.573A1Ap-5f;
  const float vc5 = -0x1.0F9F9Cp-7f;

  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const float vx0 = x[0];
    const float vx1 = x[1];
    const float vx2 = x[2];
    const float vx3 = x[3];
    x += 4;

    // General structure of the algorithm:
    //           / exp(x) / (1 + exp(x)) if x <= 0
    //   f[x] := 
    //           \ 1 - f[-x] if x >= 0
    //
    // First we compute f[-z] := exp(-z) / (1 + exp(-z)) where z = abs(x),
    // then replace result with 1 - f[-z] if x >= 0.
    const float vz0 = fabsf(vx0);
    const float vz1 = fabsf(vx1);
    const float vz2 = fabsf(vx2);
    const float vz3 = fabsf(vx3);

    // Compute reduced argument n := round(-z / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of result to an integer, then subtracing the
    // large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|x| <= 2**22), but thats ok, because
    // inputs x outside of [-87.336544, 17.328678] (i.e. z outsize [0, 87.336544]) underflow or saturate sigmoidf(x)
    // anyway. We fixup the result for such inputs at the very end of the algorithm.
    float vn0 = vz0 * vminus_log2e + vmagic_bias;
    float vn1 = vz1 * vminus_log2e + vmagic_bias;
    float vn2 = vz2 * vminus_log2e + vmagic_bias;
    float vn3 = vz3 * vminus_log2e + vmagic_bias;

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -87.336544 <= -z <= 0.0, and -126 <= n <= 0 accordingly.
    const float vs0 = fp32_from_bits(fp32_to_bits(vn0) << 23);
    const float vs1 = fp32_from_bits(fp32_to_bits(vn1) << 23);
    const float vs2 = fp32_from_bits(fp32_to_bits(vn2) << 23);
    const float vs3 = fp32_from_bits(fp32_to_bits(vn3) << 23);

    // Subtract the large number back to get the final n := round(-z / log(2)) as a floating-point number.
    vn0 -= vmagic_bias;
    vn1 -= vmagic_bias;
    vn2 -= vmagic_bias;
    vn3 -= vmagic_bias;

    // Compute reduced argument t := z + n * log(2). Note that -t = -z - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    float vt0 = vn0 * vln2_hi + vz0;
    float vt1 = vn1 * vln2_hi + vz1;
    float vt2 = vn2 * vln2_hi + vz2;
    float vt3 = vn3 * vln2_hi + vz3;

    vt0 = vn0 * vln2_lo + vt0;
    vt1 = vn1 * vln2_lo + vt1;
    vt2 = vn2 * vln2_lo + vt2;
    vt3 = vn3 * vln2_lo + vt3;

    // Compute degree-5 polynomial approximation for exp(-t) on [-log(2)/2, log(2)/2]:
    //   P5(t) = 1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    float vp0 = vt0 * vc5 + vc4;
    float vp1 = vt1 * vc5 + vc4;
    float vp2 = vt2 * vc5 + vc4;
    float vp3 = vt3 * vc5 + vc4;

    vp0 = vt0 * vp0 + vc3;
    vp1 = vt1 * vp1 + vc3;
    vp2 = vt2 * vp2 + vc3;
    vp3 = vt3 * vp3 + vc3;

    vp0 = vt0 * vp0 + vc2;
    vp1 = vt1 * vp1 + vc2;
    vp2 = vt2 * vp2 + vc2;
    vp3 = vt3 * vp3 + vc2;

    vp0 = vt0 * vp0 + vc1;
    vp1 = vt1 * vp1 + vc1;
    vp2 = vt2 * vp2 + vc1;
    vp3 = vt3 * vp3 + vc1;

    // Reconstruct the exp(-z) value:
    //   e = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt0 *= vs0;
    vt1 *= vs1;
    vt2 *= vs2;
    vt3 *= vs3;

    const float ve0 = vt0 * vp0 + vs0;
    const float ve1 = vt1 * vp1 + vs1;
    const float ve2 = vt2 * vp2 + vs2;
    const float ve3 = vt3 * vp3 + vs3;

    // Reconstruct sigmoid(-z) = exp(-z) / (1.0 + exp(-z))
    float vf0 = ve0 / (ve0 + vone);
    float vf1 = ve1 / (ve1 + vone);
    float vf2 = ve2 / (ve2 + vone);
    float vf3 = ve3 / (ve3 + vone);

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    if XNN_UNPREDICTABLE(vz0 > vdenorm_cutoff) {
      vf0 = 0.0f;
    }
    if XNN_UNPREDICTABLE(vz1 > vdenorm_cutoff) {
      vf1 = 0.0f;
    }
    if XNN_UNPREDICTABLE(vz2 > vdenorm_cutoff) {
      vf2 = 0.0f;
    }
    if XNN_UNPREDICTABLE(vz3 > vdenorm_cutoff) {
      vf3 = 0.0f;
    }

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(-z) : 1.0 - sigmoid(-z)
    if XNN_UNPREDICTABLE(vx0 > 0.0f) {
      vf0 = vone - vf0;
    }
    if XNN_UNPREDICTABLE(vx1 > 0.0f) {
      vf1 = vone - vf1;
    }
    if XNN_UNPREDICTABLE(vx2 > 0.0f) {
      vf2 = vone - vf2;
    }
    if XNN_UNPREDICTABLE(vx3 > 0.0f) {
      vf3 = vone - vf3;
    }

    y[0] = vf0;
    y[1] = vf1;
    y[2] = vf2;
    y[3] = vf3;
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const float vx = *x++;

      // General structure of the algorithm:
      //           / exp(x) / (1 + exp(x)) if x <= 0
      //   f[x] := 
      //           \ 1 - f[-x] if x >= 0
      //
      // First we compute f[-z] := exp(-z) / (1 + exp(-z)) where z = abs(x),
      // then replace result with 1 - f[-z] if x >= 0.
      const float vz = fabsf(vx);

      // Compute reduced argument n := round(-z / log(2)).
      // We do it by adding a large number (magic bias), which cause rounding of result to an integer, then subtracing the
      // large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
      // The trick with adding large number is valid only within certain bounds (|x| <= 2**22), but thats ok, because
      // inputs x outside of [-87.336544, 17.328678] (i.e. z outsize [0, 87.336544]) underflow or saturate sigmoidf(x)
      // anyway. We fixup the result for such inputs at the very end of the algorithm.
      float vn = vz * vminus_log2e + vmagic_bias;

      // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
      // -87.336544 <= -z <= 0.0, and -126 <= n <= 0 accordingly.
      const float vs = fp32_from_bits(fp32_to_bits(vn) << 23);

      // Subtract the large number back to get the final n := round(-z / log(2)) as a floating-point number.
      vn -= vmagic_bias;

      // Compute reduced argument t := z + n * log(2). Note that -t = -z - n * log(2).
      // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
      float vt = vn * vln2_hi + vz;
      vt = vn * vln2_lo + vt;

      // Compute degree-5 polynomial approximation for exp(-t) on [-log(2)/2, log(2)/2]:
      //   P5(t) = 1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
      float vp = vt * vc5 + vc4;
      vp = vt * vp + vc3;
      vp = vt * vp + vc2;
      vp = vt * vp + vc1;

      // Reconstruct the exp(-z) value:
      //   e = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
      //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
      //     = s + (t * s) * p
      vt *= vs;
      const float ve = vt * vp + vs;

      // Reconstruct sigmoid(-z) = exp(-z) / (1.0 + exp(-z))
      float vf = ve / (ve + vone);

      // For inputs above denormal cutoff, replace output with +0.0f.
      // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
      if XNN_UNPREDICTABLE(vz > vdenorm_cutoff) {
        vf = 0.0f;
      }

      // Reconstruct sigmoid(x) = x < 0 ? sigmoid(-z) : 1.0 - sigmoid(-z)
      if XNN_UNPREDICTABLE(vx > 0.0f) {
        vf = vone - vf;
      }

      *y++ = vf;

      n -= sizeof(float);
    } while (n != 0);
  }
}
