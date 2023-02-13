// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <arm_neon.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f16_expm1minus__neonfp16arith_rr2_p3(
    size_t n,
    const void* input,
    void* output)
{
  assert(n % (8 * sizeof(uint16_t)) == 0);

  // The largest x for which expm1f(x) is saturated at -1.0f.
  const float16x8_t vsat_cutoff = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xC829)));  // -0x1.0A4p+3h
  // Large number such that ulp(magic bias) == 1 and magic bias === 15 mod 2**9.
  const float16x8_t vmagic_bias = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x660F)));  // 0x1.83Cp+10h
  const float16x8_t vlog2e = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3DC5)));  // 0x1.714p+0h
  const float16x8_t vminus_ln2_hi = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xB98C)));  // -0x1.630p-1h
  const float16x8_t vminus_ln2_lo = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x0AF4)));  // 0x1.BD0p-13h
  // Coefficient of polynomial approximation
  //   exp(t) - 1 ~ t * (1 + t * (c2 + t * c3))
  // on [-log(2)/2, log(2)/2]
  const float16x8_t vc3 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x315B)));  // 0x1.56Cp-3h
  const float16x8_t vc2 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3808)));  // 0x1.020p-1h
  const float16x8_t vone = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3C00)));  // 1.0h

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; n != 0; n -= 8 * sizeof(uint16_t)) {
    float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    // The function saturates at -1 for large negative inputs: expm1h(x) == -1.0h for x <= sat_cutoff ~= -8.3203125.
    // To guarantee this behaviour, we clip input at sat_cutoff, and leverage the fact that for our implementation
    // expm1m(sat_cutoff) == -1.0f. NaN inputs are passed unchanged.
    vx = vmaxq_f16(vx, vsat_cutoff);

    // Compute reduced argument n := round(x / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of the result to integer, then subtracing
    // the large number back. The addition is combined with multiplication by log2e into a single FMA instruction. The
    // trick with adding large number is valid only within certain bounds (|x / log(2)| <= 2**9, i.e.
    // |x| <= 0x1.630p+8 = 355.0), but that is acceptable, because inputs x are restricted to [-8.3203125, 0].
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    float16x8_t vn = vfmaq_f16(vmagic_bias, vx, vlog2e);

    // Create a floating-point number s (scale) such that s == 2**n for valid inputs, i.e.
    // -8.3203125 <= x <= 0.0, and -12 <= n <= 0 accordingly.
    // For NaN inputs, s would have zero mantissa and can have arbitrary sign and exponent, depending on the input
    // NaN payload. In these cases, n and t are NaNs with the same payload as input while s is non-NaN, and thus
    // input payload would be propagated in all computations.
    const float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));

    // Subtract the large number back to get final n := round(x / log(2)).
    vn = vsubq_f16(vn, vmagic_bias);

    // Compute reduced argument t := x - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    float16x8_t vt = vfmaq_f16(vx, vn, vminus_ln2_hi);
    vt = vfmaq_f16(vt, vn, vminus_ln2_lo);

    // Compute degree-3 polynomial approximation for exp(t) - 1 on [-log(2)/2, log(2)/2].
    //   P(t) = t * (1 + t * (c2 + t * c3))
    //        = t + t * (t * (c2 + t * c3)) = t + t * p
    float16x8_t vp = vfmaq_f16(vc2, vc3, vt);
    vp = vmulq_f16(vp, vt);

    // Reconstruct the exp(x) - 1 value:
    //   exp(x) - 1 = s * (1 + t * (1 + t * (c2 + t * c3))) - 1
    //              = (s - 1) + s * (t + t * p)
    //              = ((t * s) + (t * s) * p) + (s - 1)
    vt = vmulq_f16(vt, vs);
    const float16x8_t vsm1 = vsubq_f16(vs, vone);
    vp = vfmaq_f16(vt, vp, vt);
    const float16x8_t vf = vaddq_f16(vp, vsm1);

    vst1q_u16(o, vreinterpretq_u16_f16(vf)); o += 8;
  }
}
