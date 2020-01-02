// Auto-generated file. Do not edit!
//   Template: src/f32-sigmoid/psimd-p5-div.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


void xnn_f32_sigmoid_ukernel__psimd_p5_div_x24(
    size_t n,
    const float* x,
    float* y,
    const void* params)
{
  assert(n % sizeof(float) == 0);

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

  for (; n >= 24 * sizeof(float); n -= 24 * sizeof(float)) {
    const psimd_f32 vx0123 = psimd_load_f32(x);
    const psimd_f32 vx4567 = psimd_load_f32(x + 4);
    const psimd_f32 vx89AB = psimd_load_f32(x + 8);
    const psimd_f32 vxCDEF = psimd_load_f32(x + 12);
    const psimd_f32 vxGHIJ = psimd_load_f32(x + 16);
    const psimd_f32 vxKLMN = psimd_load_f32(x + 20);
    x += 24;

    // General structure of the algorithm:
    //           / exp(x) / (1 + exp(x)) if x <= 0
    //   f[x] := 
    //           \ 1 - f[-x] if x >= 0
    //
    // First we compute f[-z] := exp(-z) / (1 + exp(-z)) where z = abs(x),
    // then replace result with 1 - f[-z] if x >= 0.
    const psimd_f32 vz0123 = psimd_abs_f32(vx0123);
    const psimd_f32 vz4567 = psimd_abs_f32(vx4567);
    const psimd_f32 vz89AB = psimd_abs_f32(vx89AB);
    const psimd_f32 vzCDEF = psimd_abs_f32(vxCDEF);
    const psimd_f32 vzGHIJ = psimd_abs_f32(vxGHIJ);
    const psimd_f32 vzKLMN = psimd_abs_f32(vxKLMN);

    // Compute reduced argument n := round(-z / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of result to an integer, then subtracing the
    // large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|x| <= 2**22), but thats ok, because
    // inputs x outside of [-87.336544, 17.328678] (i.e. z outsize [0, 87.336544]) underflow or saturate sigmoidf(x)
    // anyway. We fixup the result for such inputs at the very end of the algorithm.
    psimd_f32 vn0123 = psimd_qfma_f32(vmagic_bias, vz0123, vminus_log2e);
    psimd_f32 vn4567 = psimd_qfma_f32(vmagic_bias, vz4567, vminus_log2e);
    psimd_f32 vn89AB = psimd_qfma_f32(vmagic_bias, vz89AB, vminus_log2e);
    psimd_f32 vnCDEF = psimd_qfma_f32(vmagic_bias, vzCDEF, vminus_log2e);
    psimd_f32 vnGHIJ = psimd_qfma_f32(vmagic_bias, vzGHIJ, vminus_log2e);
    psimd_f32 vnKLMN = psimd_qfma_f32(vmagic_bias, vzKLMN, vminus_log2e);

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -87.336544 <= -z <= 0.0, and -126 <= n <= 0 accordingly.
    const psimd_f32 vs0123 = (psimd_f32) ((psimd_u32) vn0123 << 23);
    const psimd_f32 vs4567 = (psimd_f32) ((psimd_u32) vn4567 << 23);
    const psimd_f32 vs89AB = (psimd_f32) ((psimd_u32) vn89AB << 23);
    const psimd_f32 vsCDEF = (psimd_f32) ((psimd_u32) vnCDEF << 23);
    const psimd_f32 vsGHIJ = (psimd_f32) ((psimd_u32) vnGHIJ << 23);
    const psimd_f32 vsKLMN = (psimd_f32) ((psimd_u32) vnKLMN << 23);

    // Subtract the large number back to get the final n := round(-z / log(2)) as a floating-point number.
    vn0123 = psimd_sub_f32(vn0123, vmagic_bias);
    vn4567 = psimd_sub_f32(vn4567, vmagic_bias);
    vn89AB = psimd_sub_f32(vn89AB, vmagic_bias);
    vnCDEF = psimd_sub_f32(vnCDEF, vmagic_bias);
    vnGHIJ = psimd_sub_f32(vnGHIJ, vmagic_bias);
    vnKLMN = psimd_sub_f32(vnKLMN, vmagic_bias);

    // Compute reduced argument t := z + n * log(2). Note that -t = -z - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    psimd_f32 vt0123 = psimd_qfma_f32(vz0123, vn0123, vln2_hi);
    psimd_f32 vt4567 = psimd_qfma_f32(vz4567, vn4567, vln2_hi);
    psimd_f32 vt89AB = psimd_qfma_f32(vz89AB, vn89AB, vln2_hi);
    psimd_f32 vtCDEF = psimd_qfma_f32(vzCDEF, vnCDEF, vln2_hi);
    psimd_f32 vtGHIJ = psimd_qfma_f32(vzGHIJ, vnGHIJ, vln2_hi);
    psimd_f32 vtKLMN = psimd_qfma_f32(vzKLMN, vnKLMN, vln2_hi);

    vt0123 = psimd_qfma_f32(vt0123, vn0123, vln2_lo);
    vt4567 = psimd_qfma_f32(vt4567, vn4567, vln2_lo);
    vt89AB = psimd_qfma_f32(vt89AB, vn89AB, vln2_lo);
    vtCDEF = psimd_qfma_f32(vtCDEF, vnCDEF, vln2_lo);
    vtGHIJ = psimd_qfma_f32(vtGHIJ, vnGHIJ, vln2_lo);
    vtKLMN = psimd_qfma_f32(vtKLMN, vnKLMN, vln2_lo);

    // Compute degree-5 polynomial approximation for exp(-t) on [-log(2)/2, log(2)/2]:
    //   P5(t) = 1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    psimd_f32 vp0123 = psimd_qfma_f32(vc4, vt0123, vc5);
    psimd_f32 vp4567 = psimd_qfma_f32(vc4, vt4567, vc5);
    psimd_f32 vp89AB = psimd_qfma_f32(vc4, vt89AB, vc5);
    psimd_f32 vpCDEF = psimd_qfma_f32(vc4, vtCDEF, vc5);
    psimd_f32 vpGHIJ = psimd_qfma_f32(vc4, vtGHIJ, vc5);
    psimd_f32 vpKLMN = psimd_qfma_f32(vc4, vtKLMN, vc5);

    vp0123 = psimd_qfma_f32(vc3, vt0123, vp0123);
    vp4567 = psimd_qfma_f32(vc3, vt4567, vp4567);
    vp89AB = psimd_qfma_f32(vc3, vt89AB, vp89AB);
    vpCDEF = psimd_qfma_f32(vc3, vtCDEF, vpCDEF);
    vpGHIJ = psimd_qfma_f32(vc3, vtGHIJ, vpGHIJ);
    vpKLMN = psimd_qfma_f32(vc3, vtKLMN, vpKLMN);

    vp0123 = psimd_qfma_f32(vc2, vt0123, vp0123);
    vp4567 = psimd_qfma_f32(vc2, vt4567, vp4567);
    vp89AB = psimd_qfma_f32(vc2, vt89AB, vp89AB);
    vpCDEF = psimd_qfma_f32(vc2, vtCDEF, vpCDEF);
    vpGHIJ = psimd_qfma_f32(vc2, vtGHIJ, vpGHIJ);
    vpKLMN = psimd_qfma_f32(vc2, vtKLMN, vpKLMN);

    vp0123 = psimd_qfma_f32(vc1, vt0123, vp0123);
    vp4567 = psimd_qfma_f32(vc1, vt4567, vp4567);
    vp89AB = psimd_qfma_f32(vc1, vt89AB, vp89AB);
    vpCDEF = psimd_qfma_f32(vc1, vtCDEF, vpCDEF);
    vpGHIJ = psimd_qfma_f32(vc1, vtGHIJ, vpGHIJ);
    vpKLMN = psimd_qfma_f32(vc1, vtKLMN, vpKLMN);

    // Reconstruct the exp(-z) value:
    //   e = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt0123 = psimd_mul_f32(vt0123, vs0123);
    vt4567 = psimd_mul_f32(vt4567, vs4567);
    vt89AB = psimd_mul_f32(vt89AB, vs89AB);
    vtCDEF = psimd_mul_f32(vtCDEF, vsCDEF);
    vtGHIJ = psimd_mul_f32(vtGHIJ, vsGHIJ);
    vtKLMN = psimd_mul_f32(vtKLMN, vsKLMN);

    const psimd_f32 ve0123 = psimd_qfma_f32(vs0123, vt0123, vp0123);
    const psimd_f32 ve4567 = psimd_qfma_f32(vs4567, vt4567, vp4567);
    const psimd_f32 ve89AB = psimd_qfma_f32(vs89AB, vt89AB, vp89AB);
    const psimd_f32 veCDEF = psimd_qfma_f32(vsCDEF, vtCDEF, vpCDEF);
    const psimd_f32 veGHIJ = psimd_qfma_f32(vsGHIJ, vtGHIJ, vpGHIJ);
    const psimd_f32 veKLMN = psimd_qfma_f32(vsKLMN, vtKLMN, vpKLMN);

    // Reconstruct sigmoid(-z) = exp(-z) / (1.0 + exp(-z))
    psimd_f32 vf0123 = psimd_div_f32(ve0123, psimd_add_f32(ve0123, vone));
    psimd_f32 vf4567 = psimd_div_f32(ve4567, psimd_add_f32(ve4567, vone));
    psimd_f32 vf89AB = psimd_div_f32(ve89AB, psimd_add_f32(ve89AB, vone));
    psimd_f32 vfCDEF = psimd_div_f32(veCDEF, psimd_add_f32(veCDEF, vone));
    psimd_f32 vfGHIJ = psimd_div_f32(veGHIJ, psimd_add_f32(veGHIJ, vone));
    psimd_f32 vfKLMN = psimd_div_f32(veKLMN, psimd_add_f32(veKLMN, vone));

    // For inputs above denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf0123 = psimd_andnotmask_f32(vz0123 > vdenorm_cutoff, vf0123);
    vf4567 = psimd_andnotmask_f32(vz4567 > vdenorm_cutoff, vf4567);
    vf89AB = psimd_andnotmask_f32(vz89AB > vdenorm_cutoff, vf89AB);
    vfCDEF = psimd_andnotmask_f32(vzCDEF > vdenorm_cutoff, vfCDEF);
    vfGHIJ = psimd_andnotmask_f32(vzGHIJ > vdenorm_cutoff, vfGHIJ);
    vfKLMN = psimd_andnotmask_f32(vzKLMN > vdenorm_cutoff, vfKLMN);

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(-z) : 1.0 - sigmoid(-z)
    vf0123 = psimd_signblend_f32(vx0123, vf0123, psimd_sub_f32(vone, vf0123));
    vf4567 = psimd_signblend_f32(vx4567, vf4567, psimd_sub_f32(vone, vf4567));
    vf89AB = psimd_signblend_f32(vx89AB, vf89AB, psimd_sub_f32(vone, vf89AB));
    vfCDEF = psimd_signblend_f32(vxCDEF, vfCDEF, psimd_sub_f32(vone, vfCDEF));
    vfGHIJ = psimd_signblend_f32(vxGHIJ, vfGHIJ, psimd_sub_f32(vone, vfGHIJ));
    vfKLMN = psimd_signblend_f32(vxKLMN, vfKLMN, psimd_sub_f32(vone, vfKLMN));

    psimd_store_f32(y, vf0123);
    psimd_store_f32(y + 4, vf4567);
    psimd_store_f32(y + 8, vf89AB);
    psimd_store_f32(y + 12, vfCDEF);
    psimd_store_f32(y + 16, vfGHIJ);
    psimd_store_f32(y + 20, vfKLMN);
    y += 24;
  }
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const psimd_f32 vx = psimd_load_f32(x);
    x += 4;

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

    // For inputs above denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = psimd_andnotmask_f32(vz > vdenorm_cutoff, vf);

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(-z) : 1.0 - sigmoid(-z)
    vf = psimd_signblend_f32(vx, vf, psimd_sub_f32(vone, vf));

    psimd_store_f32(y, vf);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const psimd_f32 vx = psimd_load_f32(x);

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

    // For inputs above denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = psimd_andnotmask_f32(vz > vdenorm_cutoff, vf);

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(-z) : 1.0 - sigmoid(-z)
    vf = psimd_signblend_f32(vx, vf, psimd_sub_f32(vone, vf));

    if (n & (2 * sizeof(float))) {
      psimd_store2_f32(y, vf);
      vf = psimd_concat_hi_f32(vf, vf);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      psimd_store1_f32(y, vf);
    }
  }
}
