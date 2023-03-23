// Auto-generated file. Do not edit!
//   Template: src/math/f32-tanh-neon-expm1minus.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f32_tanh__neonfma_expm1minus_rr1_p6h5ts_nr2recpsadj(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % sizeof(float32x4_t) == 0);

  // The smallest z for which tanhf(-z) is saturated at -1.0f.
  const float32x4_t vsat_cutoff = vmovq_n_f32(0x1.205968p+3f);
  const float32x4_t vminus_log2e = vmovq_n_f32(-0x1.715476p+0f);
  // Large number such that ulp(magic bias) == 0.5 and magic bias === 63.5 mod 2**21.
  const float32x4_t vmagic_bias = vmovq_n_f32(0x1.8000FEp+22f);
  const float32x4_t vln2 = vmovq_n_f32(0x1.62E430p-1f);
  // Coefficients of polynomial approximation
  //   exp(-2t) - 1 ~ t * (-2 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
  // on [-log(2)/4, log(2)/4]
  const float32x4_t vc6 = vmovq_n_f32(0x1.6B7338p-4f);
  const float32x4_t vc5 = vmovq_n_f32(-0x1.12278Ep-2f);
  const float32x4_t vc4 = vmovq_n_f32(0x1.555716p-1f);
  const float32x4_t vc3 = vmovq_n_f32(-0x1.5554B0p+0f);
  const float32x4_t vc2 = vmovq_n_f32(0x1.FFFFFEp+0f);
  const float32x4_t vtwo = vmovq_n_f32(2.0f);
  const float32x4_t vone = vmovq_n_f32(1.0f);
  // Mask for the sign bit.
  const uint32x4_t vsign_mask = vmovq_n_u32(UINT32_C(0x80000000));

  for (; n != 0; n -= sizeof(float32x4_t)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;

    // General structure of the algorithm:
    //
    //           / -expm1(-2x) / (2 + expm1(-2x)) if x >= 0
    //   f(x) :=
    //           \ -f(-x) if x <= 0
    //
    // First we compute y := expm1(-2z) / (2 + expm1(-2z)) where z = abs(x),
    // then set its sign according to the sign of x: f(x) := sign(x) * abs(y).
    float32x4_t vz = vabsq_f32(vx);

    // The function saturates at -1 for large positive inputs: tanhf(-z) == -1.0f for z >= sat_cutoff ~= 9.010913.
    // To guarantee this behaviour, we clip input z at sat_cutoff, and leverage the fact that for our implementation
    // tanhf(sat_cutoff) == -1.0f. NaN inputs are passed unchanged.
    vz = vminq_f32(vz, vsat_cutoff);

    // Compute reduced argument n := round(-z / log(2), 1).
    // We do it by adding a large number (magic bias), which cause rounding of the result to 1 fractional bit,
    // then subtracing the large number back. The trick with adding large number is valid only within certain bounds
    // (|-z / log(2)| <= 2**21, i.e. |z| <= 0x1.62E43p+20 = 1453635.0), but that is acceptable, because inputs x
    // outside of [-9.010913, 9.010913] (i.e. z outsize [0, 9.010913]) saturate tanhf(x).
    // Additionally, we fuse addition of the floating-point exponent bias (127) into the magic bias.
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    float32x4_t vn = vfmaq_f32(vmagic_bias, vz, vminus_log2e);

    // Create a floating-point number s (scale) such that s == 2**(2n) for inputs which don't cause underflow, i.e.
    // 0 <= z <= 9.010913, and -13 <= n <= 0 accordingly.
    const float32x4_t vs = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn), 23));

    // Subtract the large number back to get final n := round(-z / log(2), 1) as a floating-point number.
    vn = vsubq_f32(vn, vmagic_bias);

    // Compute reduced argument t := z + n * log(2). Note that -t = -z - n * log(2).
    const float32x4_t vt = vfmaq_f32(vz, vn, vln2);

    // Compute degree-6 polynomial approximation for exp(-2t) - 1 on [-log(2)/4, log(2)/4].
    //   P(t) = t * (-2 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
    //        = t * (-p)
    float32x4_t vp = vfmaq_f32(vc5, vc6, vt);
    vp = vfmaq_f32(vc4, vp, vt);
    vp = vfmaq_f32(vc3, vp, vt);
    vp = vfmaq_f32(vc2, vp, vt);
    vp = vfmsq_f32(vtwo, vp, vt);

    // Reconstruct the exp(-2z) - 1 value:
    //   exp(-2z) - 1 = s * (t * (-2 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))) + 1) - 1
    //                = s * t * (-p) + (s - 1)
    //                = (s - 1) - (t * s) * p
    const float32x4_t vts = vmulq_f32(vt, vs);
    const float32x4_t vsmo = vsubq_f32(vs, vone);
    const float32x4_t vemo = vfmsq_f32(vsmo, vp, vts);

    // Denominator of the tanh fraction: exp(-2z) + 1 = expm1(-2z) + 2
    const float32x4_t vepo = vaddq_f32(vemo, vtwo);

    // Use Newton-Raphson method (2 iterations) to compute reciprocal of the denominator.
    // Note: 2 < exp(-2z) + 1 <= 3, because z <= 0 and 0 < exp(-2z) <= 1.
    // Thus the reciprocal of the denominator never overflows.
    float32x4_t vrepo = vrecpeq_f32(vepo);
    float32x4_t verepo = vrecpsq_f32(vrepo, vepo);
    vrepo = vmulq_f32(vrepo, verepo);
    verepo = vrecpsq_f32(vrepo, vepo);
    vrepo = vmulq_f32(vrepo, verepo);

    // Reconstruct y = expm1(-2z) / (expm1(-2z) + 2)
    float32x4_t vy = vmulq_f32(vemo, vrepo);

    // Adjust reconstructred expm1(-2z) / (2 + expm1(-2z)) to match the correctly rounded division result
    const float32x4_t vey = vfmsq_f32(vemo, vy, vepo);
    vy = vfmaq_f32(vy, vey, vrepo);

    // Reconstruct tanh(x) = copysign(y, x)
    vy = vbslq_f32(vsign_mask, vx, vy);

    vst1q_f32(output, vy); output += 4;
  }
}
