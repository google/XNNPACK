// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <psimd.h>

#include <xnnpack/common.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f32_sigmoid__psimd_p5_div(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  const psimd_f32 vmagic_bias = psimd_splat_f32(0x1.8000FEp23f);
  // The largest z for which sigmoidf(-z) is normalized.
  // This number is also the largest z for which expf(-z) is normalized.
  const psimd_f32 vdenorm_cutoff = psimd_splat_f32(0x1.5D589Ep+6f);
  const psimd_f32 vminus_log2e = psimd_splat_f32(-0x1.715476p+0f);
  // Last 7 bits are zeroes
  const psimd_f32 vln2_hi = psimd_splat_f32(0x1.62E400p-1f);
  const psimd_f32 vln2_lo = psimd_splat_f32(0x1.7F7D1Cp-20f);
  const psimd_f32 vone = psimd_splat_f32(1.0f);

  const psimd_f32 vc1 = psimd_splat_f32(-0x1.FFFFF6p-1f);
  const psimd_f32 vc2 = psimd_splat_f32( 0x1.FFFDC6p-2f);
  const psimd_f32 vc3 = psimd_splat_f32(-0x1.555A80p-3f);
  const psimd_f32 vc4 = psimd_splat_f32( 0x1.573A1Ap-5f);
  const psimd_f32 vc5 = psimd_splat_f32(-0x1.0F9F9Cp-7f);

  for (; n != 0; n -= 4 * sizeof(float)) {
    const psimd_f32 vx = psimd_load_f32(input);
    input += 4;

    // General structure of the algorithm:
    //           / exp(x) / (1 + exp(x)) if x <= 0
    //   f[x] := 
    //           \ 1 - f[-x] if x >= 0
    //
    // First we compute f[-z] := exp(-z) / (1 + exp(-z)) where z = abs(x),
    // then replace result with 1 - f[-z] if x >= 0.
    const psimd_f32 vz = psimd_abs_f32(vx);

    // Compute reduced argument n := round(-z / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of result to an integer, then subtracing the
    // large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|x| <= 2**22), but thats ok, because
    // inputs x outside of [-87.336544, 17.328678] (i.e. z outsize [0, 87.336544]) underflow or saturate sigmoidf(x)
    // anyway. We fixup the result for such inputs at the very end of the algorithm.
    psimd_f32 vn = psimd_qfma_f32(vmagic_bias, vz, vminus_log2e);

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -87.336544 <= -z <= 0.0, and -126 <= n <= 0 accordingly.
    const psimd_f32 vs = (psimd_f32) ((psimd_u32) vn << 23);

    // Subtract the large number back to get the final n := round(-z / log(2)) as a floating-point number.
    vn = psimd_sub_f32(vn, vmagic_bias);

    // Compute reduced argument t := z + n * log(2). Note that -t = -z - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    psimd_f32 vt = psimd_qfma_f32(vz, vn, vln2_hi);
    vt = psimd_qfma_f32(vt, vn, vln2_lo);

    // Compute degree-5 polynomial approximation for exp(-t) on [-log(2)/2, log(2)/2]:
    //   P5(t) = 1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    psimd_f32 vp = psimd_qfma_f32(vc4, vt, vc5);
    vp = psimd_qfma_f32(vc3, vt, vp);
    vp = psimd_qfma_f32(vc2, vt, vp);
    vp = psimd_qfma_f32(vc1, vt, vp);

    // Reconstruct the exp(-z) value:
    //   e = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt = psimd_mul_f32(vt, vs);
    const psimd_f32 ve = psimd_qfma_f32(vs, vt, vp);

    // Reconstruct sigmoid(-z) = exp(-z) / (1.0 + exp(-z))
    psimd_f32 vf = psimd_div_f32(ve, psimd_add_f32(ve, vone));

    // For inputs below denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = psimd_andnotmask_f32(vz > vdenorm_cutoff, vf);

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(-z) : 1.0 - sigmoid(-z)
    vf = psimd_signblend_f32(vx, vf, psimd_sub_f32(vone, vf));

    psimd_store_f32(output, vf);
    output += 4;
  }
}
