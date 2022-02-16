// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <arm_neon.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_expm1minus__neonfma_rr1_p6(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  // The largest x for which expm1f(x) is saturated at -1.0f.
  const float32x4_t vsat_cutoff = vmovq_n_f32(-0x1.154246p+4f);
  // Large number such that ulp(magic bias) == 1 and magic bias === 127 mod 2**22.
  const float32x4_t vmagic_bias = vmovq_n_f32(0x1.8000FEp23f);
  const float32x4_t vlog2e = vmovq_n_f32(0x1.715476p+0f);
  const float32x4_t vminus_ln2 = vmovq_n_f32(-0x1.62E430p-1f);
  // Coefficient of polynomial approximation
  //   exp(t) - 1 ~ t * (1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
  // on [-log(2)/2, log(2)/2]
  const float32x4_t vc6 = vmovq_n_f32(0x1.6b7338p-10f);
  const float32x4_t vc5 = vmovq_n_f32(0x1.12278Ep-7f);
  const float32x4_t vc4 = vmovq_n_f32(0x1.555716p-5f);
  const float32x4_t vc3 = vmovq_n_f32(0x1.5554B0p-3f);
  const float32x4_t vc2 = vmovq_n_f32(0x1.FFFFFEp-2f);
  const float32x4_t vone = vmovq_n_f32(1.0f);

  for (; n != 0; n -= 4 * sizeof(float)) {
    float32x4_t vx = vld1q_f32(input); input += 4;

    // The function saturates at -1 for large negative inputs: expm1f(x) == -1.0f for x <= sat_cutoff ~= -17.328680.
    // To guarantee this behaviour, we clip input at sat_cutoff, and leverage the fact that for our implementation
    // expm1f(sat_cutoff) == -1.0f. NaN inputs are passed unchanged.
    vx = vmaxq_f32(vx, vsat_cutoff);

    // Compute reduced argument n := round(x / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of the result to integer, then subtracing
    // the large number back. The addition is combined with multiplication by log2e into a single FMA instruction. The
    // trick with adding large number is valid only within certain bounds (|x / log(2)| <= 2**22, i.e.
    // |x| <= 0x1.62E43p+21 = 2907270.0), but that is acceptable, because inputs x are restricted to [-17.328680, 0].
    // Note that addition-subtraction of the large number doesn't cause overflow for inputs in this range.
    float32x4_t vn = vfmaq_f32(vmagic_bias, vx, vlog2e);

    // Create a floating-point number s (scale) such that s == 2**n for valid inputs, i.e.
    // -17.328680 <= x <= 0.0, and -25 <= n <= 0 accordingly.
    // For NaN inputs, s would have zero mantissa and can have arbitrary sign and exponent, depending on the input
    // NaN payload. In these cases, n and t are NaNs with the same payload as input while s is non-NaN, and thus
    // input payload would be propagated in all computations.
    const float32x4_t vs = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn), 23));

    // Subtract the large number back to get final n := round(x / log(2)).
    vn = vsubq_f32(vn, vmagic_bias);

    // Compute reduced argument t := x - n * log(2).
    float32x4_t vt = vfmaq_f32(vx, vn, vminus_ln2);

    // Compute degree-6 polynomial approximation for exp(t) - 1 on [-log(2)/2, log(2)/2].
    //   P(t) = t * (1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
    //        = t + t * (t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))) = t + t * p
    float32x4_t vp = vfmaq_f32(vc5, vc6, vt);
    vp = vfmaq_f32(vc4, vp, vt);
    vp = vfmaq_f32(vc3, vp, vt);
    vp = vfmaq_f32(vc2, vp, vt);
    vp = vmulq_f32(vp, vt);

    // Reconstruct the exp(x) - 1 value:
    //   exp(x) - 1 = s * (1 + t * (1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))) - 1
    //              = (s - 1) + s * (t + t * p)
    //              = ((t * s) + (t * s) * p) + (s - 1)
    vt = vmulq_f32(vt, vs);
    const float32x4_t vsm1 = vsubq_f32(vs, vone);
    vp = vfmaq_f32(vt, vp, vt);
    const float32x4_t vf = vaddq_f32(vp, vsm1);

    vst1q_f32(output, vf); output += 4;
  }
}
