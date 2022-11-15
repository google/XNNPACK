// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <arm_neon.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f16_expminus__neonfp16arith_rr1_p3(
    size_t n,
    const void* input,
    void* output)
{
  assert(n % (8 * sizeof(uint16_t)) == 0);

  // Large number such that ulp(magic bias) == 1 and magic bias === 15 mod 2**9.
  const float16x8_t vmagic_bias = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x660F)));  // 0x1.83Cp+10h
  const float16x8_t vlog2e = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3DC5)));  // 0x1.714p+0h
  const float16x8_t vminus_ln2 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xB98C)));  // -0x1.630p-1h
  // Coefficient of polynomial approximation
  //   exp(t) ~ 1 + t * (1 + t * (c2 + t * c3))
  // on [-log(2)/2, log(2)/2]
  const float16x8_t vc3 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3156)));  // 0x1.558p-3h
  const float16x8_t vc2 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3808)));  // 0x1.020p-1h
  const float16x8_t vone = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3C00)));  // 1.0h
  // The smallest x for which exph(x) is normalized.
  const float16x8_t vdenorm_cutoff = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xC8DA)));  // -0x1.368p+3h

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; n != 0; n -= 8 * sizeof(uint16_t)) {
    const float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    // Compute reduced argument n := round(x / log(2)).
    // We do it by adding a large number (magic bias) to the product x * (1/log(2)), which cause rounding of the result
    // to an integer, then subtracing the large number back. The first addition is combined with multiplication by
    // log2e into a single FMA instruction. The trick with adding large number is valid only within certain bounds
    // (|x / log(2)| <= 2**9, i.e. |x| <= 0x1.630p+8 = 355.0), but that is acceptable, because inputs outside
    // of [-9.703125, 0.0] underflow exph(x) anyway. We fixup the result for such inputs at the very end of the
    // algorithm.
    float16x8_t vn = vfmaq_f16(vmagic_bias, vx, vlog2e);

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -9.703125 <= x <= 0.0, and -14 <= n <= 0 accordingly.
    const float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));

    // Subtract the large number back to get final n := round(x / log(2)) as a floating-point number.
    vn = vsubq_f16(vn, vmagic_bias);

    // Compute reduced argument t := x - n * log(2).
    float16x8_t vt = vfmaq_f16(vx, vn, vminus_ln2);

    // Compute degree-3 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2]:
    //   P(t) = 1 + t * (1 + t * (c2 + t * c3)) = 1 + t * p
    float16x8_t vp = vfmaq_f16(vc2, vc3, vt);
    vp = vfmaq_f16(vone, vp, vt);

    // Reconstruct the exp(x) value:
    //   exp(x) = s * (1 + t * (1 + t * (c2 + t * c3)))
    //          = s + (t * s) * (1 + t * (c2 + t * c3))
    //          = s + (t * s) * p
    vt = vmulq_f16(vt, vs);
    float16x8_t vf = vfmaq_f16(vs, vp, vt);

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf), vcltq_f16(vx, vdenorm_cutoff)));
    vst1q_u16(o, vreinterpretq_u16_f16(vf)); o += 8;
  }
}
