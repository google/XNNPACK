// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>

#include <arm_neon.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f16_exp__neonfp16arith_rr2_p3(
    size_t n,
    const void* input,
    void* output)
{
  assert(n % (8 * sizeof(uint16_t)) == 0);

  const float16x8_t vmagic_bias = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x6600)));  // 0x1.800p+10h
  // The smallest x for which exph(x) is non-zero.
  const float16x8_t vzero_cutoff = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xCC55)));  // -0x1.154p+4h
  // The largest x for which exph(x) is finite.
  const float16x8_t vinf_cutoff = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x498F)));  // 0x1.63Cp+3h
  const float16x8_t vlog2e = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3DC5)));  // 0x1.714p+0h
  const float16x8_t vminus_ln2_hi = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xB98C)));  // -0x1.630p-1h
  const float16x8_t vminus_ln2_lo = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x0AF4)));  // 0x1.BD0p-13h
  const float16x8_t vplus_inf = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x7C00)));

  // Coefficient of polynomial approximation
  //   exp(t) - 1 ~ t * (1 + t * (c2 + t * c3))
  // on [-log(2)/2, log(2)/2]
  const float16x8_t vc3 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x315B)));  // 0x1.56Cp-3h
  const float16x8_t vc2 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3808)));  // 0x1.020p-1h
  const float16x8_t vone = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3C00)));  // 1.0h

  const int16x8_t vmin_exponent = vmovq_n_s16(INT16_C(0xC800));
  const int16x8_t vmax_exponent = vreinterpretq_s16_f16(vone);
  const int16x8_t vdefault_exponent = vmax_exponent;

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; n != 0; n -= 8 * sizeof(uint16_t)) {
    const float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    // Compute reduced argument n := round(x / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of result to an integer, then subtracing the
    // large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|x / log(2)| <= 2**9, i.e.
    // |x| <= 0x1.630p+8 = 355), but that's ok, because inputs outside of [-17.328125, 11.1171875] underflow or overflow
    // exph(x) anyway. We fixup the result for such inputs at the very end of the algorithm.
    float16x8_t vn = vfmaq_f16(vmagic_bias, vx, vlog2e);

    // Create two floating-point numbers, sn (scale, normal) and so (scale, overflow) such that sn * so == 2**n
    // for inputs which don't cause overflow, i.e. -17.328125 <= x <= 11.1171875, and -25 <= n <= 16 accordingly.
    // We need to use two numbers rather than one because a normalized half-precision exponent must be in [-14, 15]
    // range, which is insufficient to cover [-25, 16] range of n.
    // - When n is within [-14, 15], sn == 2**n and so == 1.0.
    // - When n < -14, sn == 2**(-14) and so == 2**(n + 14).
    // - When n > 15, sn == 2**15 and so == 2**(n - 15).
    int16x8_t veo = vshlq_n_s16(vreinterpretq_s16_f16(vn), 10);
    int16x8_t ven = vmaxq_s16(veo, vmin_exponent);
    ven = vminq_s16(ven, vmax_exponent);
    veo = vsubq_s16(veo, ven);
    const float16x8_t vsn = vreinterpretq_f16_s16(vaddq_s16(ven, vdefault_exponent));
    const float16x8_t vso = vreinterpretq_f16_s16(vaddq_s16(veo, vdefault_exponent));

    // Subtract the large number back to get final n := round(x / log(2)).
    vn = vsubq_f16(vn, vmagic_bias);

    // Compute reduced argument t := x - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    float16x8_t vt = vfmaq_f16(vx, vn, vminus_ln2_hi);
    vt = vfmaq_f16(vt, vn, vminus_ln2_lo);

    // Compute degree-3 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    float16x8_t vp = vfmaq_f16(vc2, vc3, vt);
    vp = vfmaq_f16(vone, vp, vt);

    // Reconstruct the final f value:
    //   f = so * sn * (1 + t * (1 + t * (c2 + t * c3)))
    //     = sn * (so + (t * so) * (1 + t * (c2 + t * c3)))
    //     = sn * (so + (t * so) * p)
    vt = vmulq_f16(vt, vso);
    float16x8_t vf = vmulq_f16(vsn, vfmaq_f16(vso, vt, vp));

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf), vcltq_f16(vx, vzero_cutoff)));
    // For inputs above inf cutoff, replace output with +inf.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = vbslq_f16(vcgtq_f16(vx, vinf_cutoff), vplus_inf, vf);
    vst1q_u16(o, vreinterpretq_u16_f16(vf)); o += 8;
  }
}
