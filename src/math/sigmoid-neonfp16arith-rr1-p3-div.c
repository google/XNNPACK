// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <arm_neon.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f16_sigmoid__neonfp16arith_rr1_p3_div(
    size_t n,
    const void* input,
    void* output)
{
  assert(n % (8 * sizeof(__fp16)) == 0);

  // Large number such that ulp(magic bias) == 1 and magic bias === 15 mod 2**9.
  const float16x8_t vmagic_bias = vmovq_n_f16(0x1.83Cp+10f);
  const float16x8_t vminus_log2e = vmovq_n_f16(-0x1.714p+0f);
  const float16x8_t vln2 = vmovq_n_f16(0x1.630p-1f);
  // Coefficient of polynomial approximation
  //   exp(-t) ~ 1 + t * (-1 + t * (c2 + t * c3))
  // on [-log(2)/2, log(2)/2]
  const float16x8_t vone = vmovq_n_f16(1.0f);
  const float16x8_t vc2 = vmovq_n_f16(0x1.020p-1f);
  const float16x8_t vc3 = vmovq_n_f16(-0x1.558p-3f);
  // The largest z for which sigmoidh(-z) is normalized.
  // This number is also the largest z for which exph(-z) is normalized.
  const float16x8_t vdenorm_cutoff = vmovq_n_f16(-0x1.368p+3f);

  const __fp16* i = (const __fp16*) input;
  __fp16* o = (__fp16*) output;
  for (; n != 0; n -= 8 * sizeof(__fp16)) {
    const float16x8_t vx = vld1q_f16(i); i += 8;

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
    float16x8_t vt = vfmaq_f16(vz, vn, vln2);

    // Compute degree-3 polynomial approximation for exp(-t) on [-log(2)/2, log(2)/2]:
    //   P(t) = 1 + t * (-1 + t * (c2 + t * c3)) = -(1 - t * p)
    float16x8_t vp = vfmaq_f16(vc2, vc3, vt);
    vp = vfmsq_f16(vone, vp, vt);

    // Reconstruct the exp(-z) value:
    //   e = s * (1 + t * (-1 + t * (c2 + t * c3))
    //     = s * (1 - t * (-p))
    //     = s - (t * s) * (-p)
    vt = vmulq_f16(vt, vs);
    float16x8_t ve = vfmsq_f16(vs, vp, vt);

    // Denominator of the sigmoid fraction: 1.0 + exp(-z)
    float16x8_t vd = vaddq_f16(ve, vone);

    // Reconstruct sigmoid(-z) = exp(-z) / (1.0 + exp(-z))
    float16x8_t vf = vdivq_f16(ve, vd);

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf), vcagtq_f16(vx, vdenorm_cutoff)));

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(-z) : 1.0 - sigmoid(-z)
    const uint16x8_t vm = vcltq_f16(vx, vmovq_n_f16(0.0f));
    vf = vbslq_f16(vm, vf, vsubq_f16(vone, vf));

    vst1q_f16(o, vf); o += 8;
  }
}
