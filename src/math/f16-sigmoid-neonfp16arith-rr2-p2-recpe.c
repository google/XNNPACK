// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <arm_neon.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f16_sigmoid__neonfp16arith_rr2_p2_recpe(
    size_t n,
    const void* input,
    void* output)
{
  assert(n % (8 * sizeof(uint16_t)) == 0);

  // Large number such that ulp(magic bias) == 1 and magic bias === 15 mod 2**9.
  const float16x8_t vmagic_bias = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x660F)));  // 0x1.83Cp+10h
  const float16x8_t vminus_log2e = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xBDC5)));  // -0x1.714p+0h
  const float16x8_t vln2_hi = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x398C)));  // 0x1.630p-1h
  const float16x8_t vln2_lo = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x8AF4)));  // -0x1.BD0p-13h
  // Coefficient of polynomial approximation
  //   exp(-t) ~ 1 + t * (c1 + t * c2)
  // on [-log(2)/2, log(2)/2]
  const float16x8_t vc2 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x37F9)));  // 0x1.FE4p-2h
  const float16x8_t vc1 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xBC0E)));  // -0x1.038p+0h
  const float16x8_t vone = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3C00)));  // 1.0h
  // The largest z for which sigmoidh(-z) is normalized.
  // This number is also the largest z for which exph(-z) is normalized.
  const float16x8_t vdenorm_cutoff = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xC8DA)));  // -0x1.368p+3h

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; n != 0; n -= 8 * sizeof(uint16_t)) {
    const float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    // General structure of the algorithm:
    //
    //           / exp(x) / (1 + exp(x)) if x <= 0
    //   f[x] :=
    //           \ 1 - f[-x] if x >= 0
    //
    // First we compute f[-z] := exp(-z) / (1 + exp(-z)) where z = abs(x),
    // then replace result with 1 - f[-z] if x >= 0.
    const float16x8_t vz = vabsq_f16(vx);

    // Compute reduced argument n := round(-z / log(2)).
    // We do it by adding a large number (magic bias) to the product z * (-1/log(2)), which cause rounding of the
    // result to an integer, then subtracing the large number back. The first addition is combined with multiplication
    // by -log2e into a single FMA instruction. The trick with adding large number is valid only within certain bounds
    // (|-x / log(2)| <= 2**9, i.e. |z| <= 0x1.630p+8 = 355.0), but that is acceptable, because inputs outside
    // of [-9.703125, 8.3125] (i.e. z outside [0, 9.703125]) underflow or saturate sigmoidh(x). We fixup the result for
    // such inputs at the very end of the algorithm.
    float16x8_t vn = vfmaq_f16(vmagic_bias, vz, vminus_log2e);

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -9.703125 <= -z <= 0.0, and -14 <= n <= 0 accordingly.
    const float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));

    // Subtract the large number back to get the final n := round(-z / log(2)) as a floating-point number.
    vn = vsubq_f16(vn, vmagic_bias);

    // Compute reduced argument t := z - n * log(2). Note that -t = -z - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent -log(2)) to improve accuracy.
    float16x8_t vt = vfmaq_f16(vz, vn, vln2_hi);
    vt = vfmaq_f16(vt, vn, vln2_lo);

    // Compute degree-2 polynomial approximation for exp(-t) on [-log(2)/2, log(2)/2]:
    //   P(t) = 1 + t * (c1 + t * c2) = 1 + t * p
    float16x8_t vp = vfmaq_f16(vc1, vc2, vt);

    // Reconstruct the exp(-z) value:
    //   e = s * (1 + t * (c1 + t * c2)
    //     = s * (1 + t * p)
    //     = s + (t * s) * p
    vt = vmulq_f16(vt, vs);
    float16x8_t ve = vfmaq_f16(vs, vp, vt);

    // Denominator of the sigmoid fraction: 1.0 + exp(-z)
    float16x8_t vd = vaddq_f16(ve, vone);

    // Compute approximate reciprocal of denominator.
    // Note: 1 < d <= 2, because z >= 0.0 and 0 < exp(-z) <= 1.0.
    // Thus the reciprocal of the denominator never overflows.
    const float16x8_t vr = vrecpeq_f16(vd);

    // Reconstruct sigmoid(-z) = exp(-z) / (1.0 + exp(-z))
    float16x8_t vf = vmulq_f16(ve, vr);

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf), vcagtq_f16(vx, vdenorm_cutoff)));

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(-z) : 1.0 - sigmoid(-z)
    const uint16x8_t vm = vcltq_f16(vx, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    vf = vbslq_f16(vm, vf, vsubq_f16(vone, vf));

    vst1q_u16(o, vreinterpretq_u16_f16(vf)); o += 8;
  }
}
