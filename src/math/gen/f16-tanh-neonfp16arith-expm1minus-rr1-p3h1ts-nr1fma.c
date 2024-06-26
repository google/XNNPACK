// Auto-generated file. Do not edit!
//   Template: src/math/f16-tanh-neonfp16arith-expm1minus.c.in
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

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/math-stubs.h"

void xnn_math_f16_tanh__neonfp16arith_expm1minus_rr1_p3h1ts_nr1fma(
    size_t n,
    const void* input,
    void* output)
{
  assert(n % sizeof(float16x8_t) == 0);

  // The smallest z for which tanhh(-z) is saturated at -1.0h.
  const float16x8_t vsat_cutoff = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x4482)));  // 0x1.208p+2h
  // Large number such that ulp(magic bias) == 0.5 and magic bias === 7.5 mod 2**8.
  const float16x8_t vmagic_bias = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x620F)));  // 0x1.83Cp+9h
  const float16x8_t vminus_log2e = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xBDC5)));  // -0x1.714p+0h
  const float16x8_t vln2 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x398C)));  // 0x1.630p-1h
  // Coefficients of polynomial approximation
  //   exp(-2t) - 1 ~ -2 * (t + t * (t * (c2 + t * c3)))
  // on [-log(2)/4, log(2)/4]
  const float16x8_t vc3 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x395B)));  // 0x1.56Cp-1h
  const float16x8_t vc2 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xBC08)));  // -0x1.020p+0h
  const float16x8_t vtwo = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x4000)));  // 2.0h
  const float16x8_t vminus_one = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xBC00)));  // -1.0h
  // Mask for the sign bit.
  const uint16x8_t vsign_mask = vmovq_n_u16(UINT16_C(0x8000));

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; n != 0; n -= sizeof(float16x8_t)) {
    const float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    // General structure of the algorithm:
    //
    //           / -expm1(-2x) / (2 + expm1(-2x)) if x >= 0
    //   f(x) :=
    //           \ -f(-x) if x <= 0
    //
    // First we compute y := expm1(-2z) / (2 + expm1(-2z)) where z = abs(x),
    // then set its sign according to the sign of x: f(x) := sign(x) * abs(y).
    float16x8_t vz = vabsq_f16(vx);

    // The function saturates at -1 for large positive inputs: tanhh(-z) == -1.0h for z >= sat_cutoff ~= 9.010913.
    // To guarantee this behaviour, we clip input z at sat_cutoff, and leverage the fact that for our implementation
    // tanhf(sat_cutoff) == -1.0h. NaN inputs are passed unchanged.
    vz = vminq_f16(vz, vsat_cutoff);

    // Compute reduced argument n := round(-z / log(2), 1).
    // We do it by adding a large number (magic bias), which cause rounding of the result to 1 fractional bit,
    // then subtracing the large number back. The trick with adding large number is valid only within certain bounds
    // (|-z / log(2)| <= 2**10, i.e. |z| <= 0x1.630p+7 = 177.5), but that is acceptable, because inputs x
    // outside of [-4.5078125, 4.5078125] (i.e. z outsize [0, 4.5078125]) saturate tanhh(x).
    // Additionally, we fuse addition of the floating-point exponent bias (15) into the magic bias.
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    float16x8_t vn = vfmaq_f16(vmagic_bias, vz, vminus_log2e);

    // Create a floating-point number s (scale) such that s == 2**(2n) for inputs which don't cause underflow, i.e.
    // 0 <= z <= 4.5078125, and -7 <= n <= 0 accordingly.
    const float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));

    // Subtract the large number back to get final n := round(-z / log(2), 1) as a floating-point number.
    vn = vsubq_f16(vn, vmagic_bias);

    // Compute reduced argument t := z + n * log(2). Note that -t = -z - n * log(2).
    const float16x8_t vt = vfmaq_f16(vz, vn, vln2);

    // Compute degree-3 polynomial approximation for exp(-2t) - 1 on [-log(2)/4, log(2)/4].
    //   P(t) = -2 * (t + t * (t * (c2 + t * c3)))
    //        = -2 * (t + t * p)
    float16x8_t vp = vfmaq_f16(vc2, vc3, vt);
    vp = vmulq_f16(vp, vt);

    // Reconstruct the exp(-2z) - 1 value:
    //   exp(-2z) - 1 = s * (-2 * (t + t * (t * (c2 + t * c3))) + 1) - 1
    //                = s * (-2 * (t + t * p) + 1) - 1
    //                = (s - 1) - 2 * ((t * s) + (t * s) * p)
    const float16x8_t vts = vmulq_f16(vt, vs);
    const float16x8_t vsmo = vaddq_f16(vs, vminus_one);
    vp = vfmaq_f16(vts, vp, vts);
    const float16x8_t vemo = vfmsq_f16(vsmo, vp, vtwo);

    // Denominator of the tanh fraction: exp(-2z) + 1 = expm1(-2z) + 2
    const float16x8_t vepo = vaddq_f16(vemo, vtwo);

    // Use Newton-Raphson method (1 iteration) to compute reciprocal of the denominator.
    // Note: 2 < exp(-2z) + 1 <= 3, because z <= 0 and 0 < exp(-2z) <= 1.
    // Thus the reciprocal of the denominator never overflows.
    float16x8_t vrepo = vrecpeq_f16(vepo);
    const float16x8_t verepo = vfmaq_f16(vminus_one, vrepo, vepo);
    vrepo = vfmsq_f16(vrepo, vrepo, verepo);

    // Reconstruct y = expm1(-2z) / (expm1(-2z) + 2)
    float16x8_t vy = vmulq_f16(vemo, vrepo);



    // Reconstruct tanh(x) = copysign(y, x)
    vy = vbslq_f16(vsign_mask, vx, vy);

    vst1q_u16(o, vreinterpretq_u16_f16(vy)); o += 8;
  }
}
