// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>

#include <arm_neon.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_exp__neonfma_p5(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  const float32x4_t vmagic_bias = vmovq_n_f32(0x1.800000p+23f);
  // The smallest x for which expf(x) is non-zero.
  const float32x4_t vzero_cutoff = vmovq_n_f32(-0x1.9FE368p+6f);
  // The largest x for which expf(x) is finite.
  const float32x4_t vinf_cutoff = vmovq_n_f32(0x1.62E42Ep+6f);
  const float32x4_t vlog2e = vmovq_n_f32(0x1.715476p+0f);
  const float32x4_t vminus_ln2_hi = vmovq_n_f32(-0x1.62E43p-1f);
  const float32x4_t vminus_ln2_lo = vmovq_n_f32(0x1.05C61p-29f);
  const float32x4_t vplus_inf = vmovq_n_f32(INFINITY);

  const float32x4_t vc1 = vmovq_n_f32(0x1.FFFFF6p-1f);
  const float32x4_t vc2 = vmovq_n_f32(0x1.FFFDC6p-2f);
  const float32x4_t vc3 = vmovq_n_f32(0x1.555A80p-3f);
  const float32x4_t vc4 = vmovq_n_f32(0x1.573A1Ap-5f);
  const float32x4_t vc5 = vmovq_n_f32(0x1.0F9F9Cp-7f);

  const int32x4_t vmin_exponent = vmovq_n_s32(INT32_C(0xC1000000));
  const int32x4_t vmax_exponent = vmovq_n_s32(INT32_C(0x3F800000));
  const int32x4_t vdefault_exponent = vmax_exponent;

  for (; n != 0; n -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;

    // Compute reduced argument n := round(x / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of result to an integer, then subtracing the
    // large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|x| <= 2**22), but thats ok, because
    // inputs outside of [-103.97207, 88.72283] underflow or overflow expf(x) anyway. We fixup the result for such
    // inputs at the very end of the algorithm.
    float32x4_t vn = vfmaq_f32(vmagic_bias, vx, vlog2e);

    // Create two floating-point numbers, sn (scale, normal) and so (scale, overflow) such that sn * so == 2**n
    // for inputs which don't cause overflow, i.e. -103.97207 <= x <= 88.72283, and -150 <= n <= 128 accordingly.
    // We need to use two numbers rather than one because a normalized single-precision exponent must be in [-127, 126]
    // range, which is insufficient to cover [-150, 128] range of n.
    // - When n is within [-127, 126], sn == 2**n and so == 1.0.
    // - When n < -127, sn == 2**(-127) and so == 2**(n + 127).
    // - When n > 126, sn == 2**126 and so == 2**(n - 126).
    int32x4_t veo = vshlq_n_s32(vreinterpretq_s32_f32(vn), 23);
    int32x4_t ven = vmaxq_s32(veo, vmin_exponent);
    ven = vminq_s32(ven, vmax_exponent);
    veo = vsubq_s32(veo, ven);
    const float32x4_t vsn = vreinterpretq_f32_s32(vaddq_s32(ven, vdefault_exponent));
    const float32x4_t vso = vreinterpretq_f32_s32(vaddq_s32(veo, vdefault_exponent));

    // Subtract the large number back to get final n := round(x / log(2)).
    vn = vsubq_f32(vn, vmagic_bias);

    // Compute reduced argument t := x - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    float32x4_t vt = vfmaq_f32(vx, vn, vminus_ln2_hi);
    vt = vfmaq_f32(vt, vn, vminus_ln2_lo);

    // Compute degree-5 polynomial approxiatmion for exp(t) on [-log(2)/2, log(2)/2].
    float32x4_t vp = vfmaq_f32(vc4, vc5, vt);
    vp = vfmaq_f32(vc3, vp, vt);
    vp = vfmaq_f32(vc2, vp, vt);
    vp = vfmaq_f32(vc1, vp, vt);

    // Reconstruct the final f value:
    //   f = so * sn * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = sn * (so + (t * so) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))))
    //     = sn * (so + (t * so) * p)
    vt = vmulq_f32(vt, vso);
    float32x4_t vf = vmulq_f32(vsn, vfmaq_f32(vso, vt, vp));

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcltq_f32(vx, vzero_cutoff)));
    // For inputs above inf cutoff, replace output with +inf.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = vbslq_f32(vcgtq_f32(vx, vinf_cutoff), vplus_inf, vf);
    vst1q_f32(output, vf); output += 4;
  }
}
