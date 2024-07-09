// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>

#include "xnnpack/math-stubs.h"


void xnn_math_f32_sigmoid__neonfma_rr2_p5_nr1recps1fma(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  // Large number such that ulp(magic bias) == 1 and magic bias === 127 mod 2**22.
  const float32x4_t vmagic_bias = vmovq_n_f32(0x1.8000FEp23f);
  const float32x4_t vminus_log2e = vmovq_n_f32(-0x1.715476p+0f);
  const float32x4_t vln2_hi = vmovq_n_f32(0x1.62E43p-1f);
  const float32x4_t vln2_lo = vmovq_n_f32(-0x1.05C61p-29f);
  // Coefficient of polynomial approximation of
  // exp(-t) ~ 1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))) on [-log(2)/2, log(2)/2]
  const float32x4_t vc5 = vmovq_n_f32(-0x1.0F9F9Cp-7f);
  const float32x4_t vc4 = vmovq_n_f32(0x1.573A1Ap-5f);
  const float32x4_t vc3 = vmovq_n_f32(-0x1.555A80p-3f);
  const float32x4_t vc2 = vmovq_n_f32(0x1.FFFDC6p-2f);
  const float32x4_t vc1 = vmovq_n_f32(-0x1.FFFFF6p-1f);
  const float32x4_t vone = vmovq_n_f32(1.0f);
  // The largest z for which sigmoidf(-z) is normalized.
  // This number is also the largest z for which expf(-z) is normalized.
  const float32x4_t vdenorm_cutoff = vmovq_n_f32(-0x1.5D589Ep+6f);

  for (; n != 0; n -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;

    // General structure of the algorithm:
    //
    //           / exp(x) / (1 + exp(x)) if x <= 0
    //   f[x] :=
    //           \ 1 - f[-x] if x >= 0
    //
    // First we compute f[-z] := exp(-z) / (1 + exp(-z)) where z = abs(x),
    // then replace result with 1 - f[-z] if x >= 0.
    const float32x4_t vz = vabsq_f32(vx);

    // Compute reduced argument n := round(-z / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of the result to integer, then subtracing
    // the large number back. The trick with adding large number is valid only within certain bounds
    // (|-z / log(2)| <= 2**22, i.e. |z| <= 0x1.62E43p+22 = 5814540.0), but that is acceptable, because inputs x
    // outside of [-87.336544, 17.328678] (i.e. z outsize [0, 87.336544]) underflow or saturate sigmoidf(x). We fixup
    // the result for such inputs at the very end of the algorithm.
    float32x4_t vn = vfmaq_f32(vmagic_bias, vz, vminus_log2e);

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -87.336544 <= -z <= 0.0, and -126 <= n <= 0 accordingly.
    const float32x4_t vs = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn), 23));

    // Subtract the large number back to get the final n := round(-z / log(2)) as a floating-point number.
    vn = vsubq_f32(vn, vmagic_bias);

    // Compute reduced argument t := z + n * log(2). Note that -t = -z - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    float32x4_t vt = vfmaq_f32(vz, vn, vln2_hi);
    vt = vfmaq_f32(vt, vn, vln2_lo);

    // Compute degree-5 polynomial approximation for exp(-t) on [-log(2)/2, log(2)/2]:
    //   P(t) = 1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))) = 1 + t * p
    float32x4_t vp = vfmaq_f32(vc4, vc5, vt);
    vp = vfmaq_f32(vc3, vp, vt);
    vp = vfmaq_f32(vc2, vp, vt);
    vp = vfmaq_f32(vc1, vp, vt);

    // Reconstruct the exp(-z) value:
    //   e = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s * (1 + t * p)
    //     = s + (t * s) * p
    vt = vmulq_f32(vt, vs);
    float32x4_t ve = vfmaq_f32(vs, vp, vt);

    // Denominator of the sigmoid fraction: 1.0 + exp(-z)
    float32x4_t vd = vaddq_f32(ve, vone);

    // Use Newton-Raphson method (2 iterations) to compute reciprocal of denominator.
    // Note: 1 < d <= 2, because z >= 0.0 and 0 < exp(-z) <= 1.0.
    // Thus the reciprocal of the denominator never overflows.
    float32x4_t vr = vrecpeq_f32(vd);
    vr = vmulq_f32(vr, vrecpsq_f32(vr, vd));
    vr = vfmaq_f32(vr, vr, vfmsq_f32(vone, vr, vd));

    // Reconstruct sigmoid(-z) = exp(-z) / (1.0 + exp(-z))
    float32x4_t vf = vmulq_f32(ve, vr);

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcagtq_f32(vx, vdenorm_cutoff)));

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(-z) : 1.0 - sigmoid(-z)
    const uint32x4_t vm = vcltq_f32(vx, vmovq_n_f32(0.0f));
    vf = vbslq_f32(vm, vf, vsubq_f32(vone, vf));

    vst1q_f32(output, vf); output += 4;
  }
}
