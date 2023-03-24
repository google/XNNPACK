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


// Table of exp2(k / 8) values decremented (as integer) by (k << 20), k = 0..7
extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_8[8];

void xnn_math_f32_tanh__neonfma_expm1minus_rr1_lut8_p4h3ps_nr1recps1fma(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % sizeof(float32x4_t) == 0);

  // The smallest z for which tanhf(-z) is saturated at -1.0f.
  const float32x4_t vsat_cutoff = vmovq_n_f32(0x1.205968p+3f);
  const float32x4_t vminus_log2e = vmovq_n_f32(-0x1.715476p+0f);
  // Large number such that ulp(magic bias) == exp2(-4)
  const float32x4_t vmagic_bias = vmovq_n_f32(0x1.800000p+19f);
  // Mask for the lowest 3 bits
  const uint64x2_t vindex_mask = vreinterpretq_u64_u32(vmovq_n_u32(UINT32_C(0x7)));
  const float32x4_t vln2 = vmovq_n_f32(0x1.62E430p-1f);
  // Coefficients of polynomial approximation
  //   exp(-2t) - 1 ~ t * (-2 + t * (c2 + t * (c3 + t * c4)))
  // on [-log(2)/32, log(2)/32]
  const float32x4_t vc4 = vmovq_n_f32(0x1.5558ECp-1f);
  const float32x4_t vc3 = vmovq_n_f32(-0x1.555C20p+0f);
  const float32x4_t vc2 = vmovq_n_f32(0x1.000000p+1f);
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

    // Compute reduced argument n := round(-z / log(2), 4).
    // We do it by adding a large number (magic bias), which cause rounding of the result to 4 fractional bits,
    // then subtracing the large number back. The trick with adding large number is valid only within certain bounds
    // (|-z / log(2)| <= 2**18, i.e. |z| <= 0x1.62E43p+17 = 181704.375), but that is acceptable, because inputs x
    // outside of [-9.010913, 9.010913] (i.e. z outsize [0, 9.010913]) saturate tanhf(x).
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    float32x4_t vn = vfmaq_f32(vmagic_bias, vz, vminus_log2e);

    // Create a floating-point number s (scale) such that s := 2**(2n) for valid inputs, i.e. 0 <= z <= 9.010913. As
    // n has 4 fractional bits, we split s == 2**(2n) = 2**int(2n) * 2**frac(2n). We create s in two steps:
    // 1. Fetch 2**frac(2n) from the table using the 3 low bits of n, as integer. Note that the fetched values are in
    //    the [1.0, 2.0) range, i.e. their unbiased floating-point exponent is 0.
    // 2. Adjust fetched value by addition of int(2n) to its floating-point exponent. The result is always a normalized
    //    number, because for 0 <= z <= 9.010913 we have -13 <= int(n) <= 0, and thus the adjusted exponent is not
    //    lower than -13.
    //
    // Shift bits 3:11 into 23:31 (position of floating-point exponent).
    const uint32x4_t ve = vshlq_n_u32(vreinterpretq_u32_f32(vn), 20);

    // Use bits 0:3 bits of n, as integer, as an index for table lookup of l := 2**frac(n).
    const uint64x2_t vidx = vandq_u64(vreinterpretq_u64_f32(vn), vindex_mask);
    const uint64_t vidx_lo = vgetq_lane_u64(vidx, 0);
    const uint64_t vidx_hi = vgetq_lane_u64(vidx, 1);
    uint32x2_t vl_lo = vld1_dup_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) vidx_lo]);
    uint32x2_t vl_hi = vld1_dup_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) vidx_hi]);
    vl_lo = vld1_lane_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) (vidx_lo >> 32)], vl_lo, 1);
    vl_hi = vld1_lane_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) (vidx_hi >> 32)], vl_hi, 1);
    const uint32x4_t vl = vcombine_u32(vl_lo, vl_hi);

    // Adjust exponent of the value l fetched from the table to get the final s value.
    const float32x4_t vs = vreinterpretq_f32_u32(vaddq_u32(vl, ve));

    // Subtract the large number back to get final n := round(-z / log(2), 4) as a floating-point number.
    vn = vsubq_f32(vn, vmagic_bias);

    // Compute reduced argument t := z + n * log(2). Note that -t = -z - n * log(2).
    const float32x4_t vt = vfmaq_f32(vz, vn, vln2);

    // Compute degree-4 polynomial approximation for exp(-2t) - 1 on [-log(2)/32, log(2)/32].
    //   P(t) = t * (-2 + t * (c2 + t * (c3 + t * c4)))
    //        = t * (-p)
    float32x4_t vp = vfmaq_f32(vc3, vc4, vt);
    vp = vfmaq_f32(vc2, vp, vt);
    vp = vfmsq_f32(vtwo, vp, vt);

    // Reconstruct the exp(-2z) - 1 value:
    //   exp(-2z) - 1 = s * (t * (-2 + t * (c2 + t * (c3 + t * c4))) + 1) - 1
    //                = s * t * (-p) + (s - 1)
    //                = (s - 1) - (p * s) * t
    const float32x4_t vps = vmulq_f32(vp, vs);
    const float32x4_t vsmo = vsubq_f32(vs, vone);
    const float32x4_t vemo = vfmsq_f32(vsmo, vt, vps);

    // Denominator of the tanh fraction: exp(-2z) + 1 = expm1(-2z) + 2
    const float32x4_t vepo = vaddq_f32(vemo, vtwo);

    // Use Newton-Raphson method (2 iterations) to compute reciprocal of the denominator.
    // Note: 2 < exp(-2z) + 1 <= 3, because z <= 0 and 0 < exp(-2z) <= 1.
    // Thus the reciprocal of the denominator never overflows.
    float32x4_t vrepo = vrecpeq_f32(vepo);
    float32x4_t verepo = vrecpsq_f32(vrepo, vepo);
    vrepo = vmulq_f32(vrepo, verepo);
    verepo = vfmsq_f32(vone, vrepo, vepo);
    vrepo = vfmaq_f32(vrepo, vrepo, verepo);

    // Reconstruct y = expm1(-2z) / (expm1(-2z) + 2)
    float32x4_t vy = vmulq_f32(vemo, vrepo);


    // Reconstruct tanh(x) = copysign(y, x)
    vy = vbslq_f32(vsign_mask, vx, vy);

    vst1q_f32(output, vy); output += 4;
  }
}
