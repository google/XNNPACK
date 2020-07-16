// Auto-generated file. Do not edit!
//   Template: src/f32-sigmoid/wasmsimd-p5-div.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


void xnn_f32_sigmoid_ukernel__wasmsimd_p5_div_x24(
    size_t n,
    const float* x,
    float* y,
    const void* params) XNN_DISABLE_TSAN
{
  assert(n % sizeof(float) == 0);

  const v128_t vmagic_bias = wasm_f32x4_splat(0x1.8000FEp23f);
  // The largest z for which sigmoidf(-z) is normalized.
  // This number is also the largest z for which expf(-z) is normalized.
  const v128_t vdenorm_cutoff = wasm_f32x4_splat(0x1.5D589Ep+6f);
  const v128_t vminus_log2e = wasm_f32x4_splat(-0x1.715476p+0f);
  // Last 7 bits are zeroes
  const v128_t vln2_hi = wasm_f32x4_splat(0x1.62E400p-1f);
  const v128_t vln2_lo = wasm_f32x4_splat(0x1.7F7D1Cp-20f);
  const v128_t vone = wasm_f32x4_splat(1.0f);

  const v128_t vc1 = wasm_f32x4_splat(-0x1.FFFFF6p-1f);
  const v128_t vc2 = wasm_f32x4_splat( 0x1.FFFDC6p-2f);
  const v128_t vc3 = wasm_f32x4_splat(-0x1.555A80p-3f);
  const v128_t vc4 = wasm_f32x4_splat( 0x1.573A1Ap-5f);
  const v128_t vc5 = wasm_f32x4_splat(-0x1.0F9F9Cp-7f);

  for (; n >= 24 * sizeof(float); n -= 24 * sizeof(float)) {
    const v128_t vx0123 = wasm_v128_load(x);
    const v128_t vx4567 = wasm_v128_load(x + 4);
    const v128_t vx89AB = wasm_v128_load(x + 8);
    const v128_t vxCDEF = wasm_v128_load(x + 12);
    const v128_t vxGHIJ = wasm_v128_load(x + 16);
    const v128_t vxKLMN = wasm_v128_load(x + 20);
    x += 24;

    // General structure of the algorithm:
    //           / exp(x) / (1 + exp(x)) if x <= 0
    //   f[x] := 
    //           \ 1 - f[-x] if x >= 0
    //
    // First we compute f[-z] := exp(-z) / (1 + exp(-z)) where z = abs(x),
    // then replace result with 1 - f[-z] if x >= 0.
    const v128_t vz0123 = wasm_f32x4_abs(vx0123);
    const v128_t vz4567 = wasm_f32x4_abs(vx4567);
    const v128_t vz89AB = wasm_f32x4_abs(vx89AB);
    const v128_t vzCDEF = wasm_f32x4_abs(vxCDEF);
    const v128_t vzGHIJ = wasm_f32x4_abs(vxGHIJ);
    const v128_t vzKLMN = wasm_f32x4_abs(vxKLMN);

    // Compute reduced argument n := round(-z / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of result to an integer, then subtracing the
    // large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|x| <= 2**22), but thats ok, because
    // inputs x outside of [-87.336544, 17.328678] (i.e. z outsize [0, 87.336544]) underflow or saturate sigmoidf(x)
    // anyway. We fixup the result for such inputs at the very end of the algorithm.
    v128_t vn0123 = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vz0123, vminus_log2e));
    v128_t vn4567 = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vz4567, vminus_log2e));
    v128_t vn89AB = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vz89AB, vminus_log2e));
    v128_t vnCDEF = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vzCDEF, vminus_log2e));
    v128_t vnGHIJ = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vzGHIJ, vminus_log2e));
    v128_t vnKLMN = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vzKLMN, vminus_log2e));

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -87.336544 <= -z <= 0.0, and -126 <= n <= 0 accordingly.
    const v128_t vs0123 = wasm_i32x4_shl(vn0123, 23);
    const v128_t vs4567 = wasm_i32x4_shl(vn4567, 23);
    const v128_t vs89AB = wasm_i32x4_shl(vn89AB, 23);
    const v128_t vsCDEF = wasm_i32x4_shl(vnCDEF, 23);
    const v128_t vsGHIJ = wasm_i32x4_shl(vnGHIJ, 23);
    const v128_t vsKLMN = wasm_i32x4_shl(vnKLMN, 23);

    // Subtract the large number back to get the final n := round(-z / log(2)) as a floating-point number.
    vn0123 = wasm_f32x4_sub(vn0123, vmagic_bias);
    vn4567 = wasm_f32x4_sub(vn4567, vmagic_bias);
    vn89AB = wasm_f32x4_sub(vn89AB, vmagic_bias);
    vnCDEF = wasm_f32x4_sub(vnCDEF, vmagic_bias);
    vnGHIJ = wasm_f32x4_sub(vnGHIJ, vmagic_bias);
    vnKLMN = wasm_f32x4_sub(vnKLMN, vmagic_bias);

    // Compute reduced argument t := z + n * log(2). Note that -t = -z - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    v128_t vt0123 = wasm_f32x4_add(vz0123, wasm_f32x4_mul(vn0123, vln2_hi));
    v128_t vt4567 = wasm_f32x4_add(vz4567, wasm_f32x4_mul(vn4567, vln2_hi));
    v128_t vt89AB = wasm_f32x4_add(vz89AB, wasm_f32x4_mul(vn89AB, vln2_hi));
    v128_t vtCDEF = wasm_f32x4_add(vzCDEF, wasm_f32x4_mul(vnCDEF, vln2_hi));
    v128_t vtGHIJ = wasm_f32x4_add(vzGHIJ, wasm_f32x4_mul(vnGHIJ, vln2_hi));
    v128_t vtKLMN = wasm_f32x4_add(vzKLMN, wasm_f32x4_mul(vnKLMN, vln2_hi));

    vt0123 = wasm_f32x4_add(vt0123, wasm_f32x4_mul(vn0123, vln2_lo));
    vt4567 = wasm_f32x4_add(vt4567, wasm_f32x4_mul(vn4567, vln2_lo));
    vt89AB = wasm_f32x4_add(vt89AB, wasm_f32x4_mul(vn89AB, vln2_lo));
    vtCDEF = wasm_f32x4_add(vtCDEF, wasm_f32x4_mul(vnCDEF, vln2_lo));
    vtGHIJ = wasm_f32x4_add(vtGHIJ, wasm_f32x4_mul(vnGHIJ, vln2_lo));
    vtKLMN = wasm_f32x4_add(vtKLMN, wasm_f32x4_mul(vnKLMN, vln2_lo));

    // Compute degree-5 polynomial approximation for exp(-t) on [-log(2)/2, log(2)/2]:
    //   P5(t) = 1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    v128_t vp0123 = wasm_f32x4_add(vc4, wasm_f32x4_mul(vt0123, vc5));
    v128_t vp4567 = wasm_f32x4_add(vc4, wasm_f32x4_mul(vt4567, vc5));
    v128_t vp89AB = wasm_f32x4_add(vc4, wasm_f32x4_mul(vt89AB, vc5));
    v128_t vpCDEF = wasm_f32x4_add(vc4, wasm_f32x4_mul(vtCDEF, vc5));
    v128_t vpGHIJ = wasm_f32x4_add(vc4, wasm_f32x4_mul(vtGHIJ, vc5));
    v128_t vpKLMN = wasm_f32x4_add(vc4, wasm_f32x4_mul(vtKLMN, vc5));

    vp0123 = wasm_f32x4_add(vc3, wasm_f32x4_mul(vt0123, vp0123));
    vp4567 = wasm_f32x4_add(vc3, wasm_f32x4_mul(vt4567, vp4567));
    vp89AB = wasm_f32x4_add(vc3, wasm_f32x4_mul(vt89AB, vp89AB));
    vpCDEF = wasm_f32x4_add(vc3, wasm_f32x4_mul(vtCDEF, vpCDEF));
    vpGHIJ = wasm_f32x4_add(vc3, wasm_f32x4_mul(vtGHIJ, vpGHIJ));
    vpKLMN = wasm_f32x4_add(vc3, wasm_f32x4_mul(vtKLMN, vpKLMN));

    vp0123 = wasm_f32x4_add(vc2, wasm_f32x4_mul(vt0123, vp0123));
    vp4567 = wasm_f32x4_add(vc2, wasm_f32x4_mul(vt4567, vp4567));
    vp89AB = wasm_f32x4_add(vc2, wasm_f32x4_mul(vt89AB, vp89AB));
    vpCDEF = wasm_f32x4_add(vc2, wasm_f32x4_mul(vtCDEF, vpCDEF));
    vpGHIJ = wasm_f32x4_add(vc2, wasm_f32x4_mul(vtGHIJ, vpGHIJ));
    vpKLMN = wasm_f32x4_add(vc2, wasm_f32x4_mul(vtKLMN, vpKLMN));

    vp0123 = wasm_f32x4_add(vc1, wasm_f32x4_mul(vt0123, vp0123));
    vp4567 = wasm_f32x4_add(vc1, wasm_f32x4_mul(vt4567, vp4567));
    vp89AB = wasm_f32x4_add(vc1, wasm_f32x4_mul(vt89AB, vp89AB));
    vpCDEF = wasm_f32x4_add(vc1, wasm_f32x4_mul(vtCDEF, vpCDEF));
    vpGHIJ = wasm_f32x4_add(vc1, wasm_f32x4_mul(vtGHIJ, vpGHIJ));
    vpKLMN = wasm_f32x4_add(vc1, wasm_f32x4_mul(vtKLMN, vpKLMN));

    // Reconstruct the exp(-z) value:
    //   e = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt0123 = wasm_f32x4_mul(vt0123, vs0123);
    vt4567 = wasm_f32x4_mul(vt4567, vs4567);
    vt89AB = wasm_f32x4_mul(vt89AB, vs89AB);
    vtCDEF = wasm_f32x4_mul(vtCDEF, vsCDEF);
    vtGHIJ = wasm_f32x4_mul(vtGHIJ, vsGHIJ);
    vtKLMN = wasm_f32x4_mul(vtKLMN, vsKLMN);

    const v128_t ve0123 = wasm_f32x4_add(vs0123, wasm_f32x4_mul(vt0123, vp0123));
    const v128_t ve4567 = wasm_f32x4_add(vs4567, wasm_f32x4_mul(vt4567, vp4567));
    const v128_t ve89AB = wasm_f32x4_add(vs89AB, wasm_f32x4_mul(vt89AB, vp89AB));
    const v128_t veCDEF = wasm_f32x4_add(vsCDEF, wasm_f32x4_mul(vtCDEF, vpCDEF));
    const v128_t veGHIJ = wasm_f32x4_add(vsGHIJ, wasm_f32x4_mul(vtGHIJ, vpGHIJ));
    const v128_t veKLMN = wasm_f32x4_add(vsKLMN, wasm_f32x4_mul(vtKLMN, vpKLMN));

    // Reconstruct sigmoid(-z) = exp(-z) / (1.0 + exp(-z))
    v128_t vf0123 = wasm_f32x4_div(ve0123, wasm_f32x4_add(ve0123, vone));
    v128_t vf4567 = wasm_f32x4_div(ve4567, wasm_f32x4_add(ve4567, vone));
    v128_t vf89AB = wasm_f32x4_div(ve89AB, wasm_f32x4_add(ve89AB, vone));
    v128_t vfCDEF = wasm_f32x4_div(veCDEF, wasm_f32x4_add(veCDEF, vone));
    v128_t vfGHIJ = wasm_f32x4_div(veGHIJ, wasm_f32x4_add(veGHIJ, vone));
    v128_t vfKLMN = wasm_f32x4_div(veKLMN, wasm_f32x4_add(veKLMN, vone));

    // For inputs above denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf0123 = wasm_v128_andnot(vf0123, wasm_f32x4_gt(vz0123, vdenorm_cutoff));
    vf4567 = wasm_v128_andnot(vf4567, wasm_f32x4_gt(vz4567, vdenorm_cutoff));
    vf89AB = wasm_v128_andnot(vf89AB, wasm_f32x4_gt(vz89AB, vdenorm_cutoff));
    vfCDEF = wasm_v128_andnot(vfCDEF, wasm_f32x4_gt(vzCDEF, vdenorm_cutoff));
    vfGHIJ = wasm_v128_andnot(vfGHIJ, wasm_f32x4_gt(vzGHIJ, vdenorm_cutoff));
    vfKLMN = wasm_v128_andnot(vfKLMN, wasm_f32x4_gt(vzKLMN, vdenorm_cutoff));

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(-z) : 1.0 - sigmoid(-z)
    vf0123 = wasm_v128_bitselect(vf0123, wasm_f32x4_sub(vone, vf0123), wasm_i32x4_shr(vx0123, 31));
    vf4567 = wasm_v128_bitselect(vf4567, wasm_f32x4_sub(vone, vf4567), wasm_i32x4_shr(vx4567, 31));
    vf89AB = wasm_v128_bitselect(vf89AB, wasm_f32x4_sub(vone, vf89AB), wasm_i32x4_shr(vx89AB, 31));
    vfCDEF = wasm_v128_bitselect(vfCDEF, wasm_f32x4_sub(vone, vfCDEF), wasm_i32x4_shr(vxCDEF, 31));
    vfGHIJ = wasm_v128_bitselect(vfGHIJ, wasm_f32x4_sub(vone, vfGHIJ), wasm_i32x4_shr(vxGHIJ, 31));
    vfKLMN = wasm_v128_bitselect(vfKLMN, wasm_f32x4_sub(vone, vfKLMN), wasm_i32x4_shr(vxKLMN, 31));

    wasm_v128_store(y, vf0123);
    wasm_v128_store(y + 4, vf4567);
    wasm_v128_store(y + 8, vf89AB);
    wasm_v128_store(y + 12, vfCDEF);
    wasm_v128_store(y + 16, vfGHIJ);
    wasm_v128_store(y + 20, vfKLMN);
    y += 24;
  }
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(x);
    x += 4;

    // General structure of the algorithm:
    //           / exp(x) / (1 + exp(x)) if x <= 0
    //   f[x] := 
    //           \ 1 - f[-x] if x >= 0
    //
    // First we compute f[-z] := exp(-z) / (1 + exp(-z)) where z = abs(x),
    // then replace result with 1 - f[-z] if x >= 0.
    const v128_t vz = wasm_f32x4_abs(vx);

    // Compute reduced argument n := round(-z / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of result to an integer, then subtracing the
    // large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|x| <= 2**22), but thats ok, because
    // inputs x outside of [-87.336544, 17.328678] (i.e. z outsize [0, 87.336544]) underflow or saturate sigmoidf(x)
    // anyway. We fixup the result for such inputs at the very end of the algorithm.
    v128_t vn = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vz, vminus_log2e));

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -87.336544 <= -z <= 0.0, and -126 <= n <= 0 accordingly.
    const v128_t vs = wasm_i32x4_shl(vn, 23);

    // Subtract the large number back to get the final n := round(-z / log(2)) as a floating-point number.
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    // Compute reduced argument t := z + n * log(2). Note that -t = -z - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    v128_t vt = wasm_f32x4_add(vz, wasm_f32x4_mul(vn, vln2_hi));
    vt = wasm_f32x4_add(vt, wasm_f32x4_mul(vn, vln2_lo));

    // Compute degree-5 polynomial approximation for exp(-t) on [-log(2)/2, log(2)/2]:
    //   P5(t) = 1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    v128_t vp = wasm_f32x4_add(vc4, wasm_f32x4_mul(vt, vc5));
    vp = wasm_f32x4_add(vc3, wasm_f32x4_mul(vt, vp));
    vp = wasm_f32x4_add(vc2, wasm_f32x4_mul(vt, vp));
    vp = wasm_f32x4_add(vc1, wasm_f32x4_mul(vt, vp));

    // Reconstruct the exp(-z) value:
    //   e = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt = wasm_f32x4_mul(vt, vs);
    const v128_t ve = wasm_f32x4_add(vs, wasm_f32x4_mul(vt, vp));

    // Reconstruct sigmoid(-z) = exp(-z) / (1.0 + exp(-z))
    v128_t vf = wasm_f32x4_div(ve, wasm_f32x4_add(ve, vone));

    // For inputs above denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = wasm_v128_andnot(vf, wasm_f32x4_gt(vz, vdenorm_cutoff));

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(-z) : 1.0 - sigmoid(-z)
    vf = wasm_v128_bitselect(vf, wasm_f32x4_sub(vone, vf), wasm_i32x4_shr(vx, 31));

    wasm_v128_store(y, vf);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const v128_t vx = wasm_v128_load(x);

    // General structure of the algorithm:
    //           / exp(x) / (1 + exp(x)) if x <= 0
    //   f[x] := 
    //           \ 1 - f[-x] if x >= 0
    //
    // First we compute f[-z] := exp(-z) / (1 + exp(-z)) where z = abs(x),
    // then replace result with 1 - f[-z] if x >= 0.
    const v128_t vz = wasm_f32x4_abs(vx);

    // Compute reduced argument n := round(-z / log(2)).
    // We do it by adding a large number (magic bias), which cause rounding of result to an integer, then subtracing the
    // large number back. The first addition is combined with multiplication by log2e into a single FMA instruction.
    // The trick with adding large number is valid only within certain bounds (|x| <= 2**22), but thats ok, because
    // inputs x outside of [-87.336544, 17.328678] (i.e. z outsize [0, 87.336544]) underflow or saturate sigmoidf(x)
    // anyway. We fixup the result for such inputs at the very end of the algorithm.
    v128_t vn = wasm_f32x4_add(vmagic_bias, wasm_f32x4_mul(vz, vminus_log2e));

    // Create a floating-point number s (scale) such that s == 2**n for inputs which don't cause underflow, i.e.
    // -87.336544 <= -z <= 0.0, and -126 <= n <= 0 accordingly.
    const v128_t vs = wasm_i32x4_shl(vn, 23);

    // Subtract the large number back to get the final n := round(-z / log(2)) as a floating-point number.
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    // Compute reduced argument t := z + n * log(2). Note that -t = -z - n * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    v128_t vt = wasm_f32x4_add(vz, wasm_f32x4_mul(vn, vln2_hi));
    vt = wasm_f32x4_add(vt, wasm_f32x4_mul(vn, vln2_lo));

    // Compute degree-5 polynomial approximation for exp(-t) on [-log(2)/2, log(2)/2]:
    //   P5(t) = 1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    v128_t vp = wasm_f32x4_add(vc4, wasm_f32x4_mul(vt, vc5));
    vp = wasm_f32x4_add(vc3, wasm_f32x4_mul(vt, vp));
    vp = wasm_f32x4_add(vc2, wasm_f32x4_mul(vt, vp));
    vp = wasm_f32x4_add(vc1, wasm_f32x4_mul(vt, vp));

    // Reconstruct the exp(-z) value:
    //   e = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt = wasm_f32x4_mul(vt, vs);
    const v128_t ve = wasm_f32x4_add(vs, wasm_f32x4_mul(vt, vp));

    // Reconstruct sigmoid(-z) = exp(-z) / (1.0 + exp(-z))
    v128_t vf = wasm_f32x4_div(ve, wasm_f32x4_add(ve, vone));

    // For inputs above denormal cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = wasm_v128_andnot(vf, wasm_f32x4_gt(vz, vdenorm_cutoff));

    // Reconstruct sigmoid(x) = x < 0 ? sigmoid(-z) : 1.0 - sigmoid(-z)
    vf = wasm_v128_bitselect(vf, wasm_f32x4_sub(vone, vf), wasm_i32x4_shr(vx, 31));

    if (n & (2 * sizeof(float))) {
      *((double*) y) = wasm_f64x2_extract_lane(vf, 0);
      vf = wasm_v32x4_shuffle(vf, vf, 2, 3, 2, 3);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      *y = wasm_f32x4_extract_lane(vf, 0);
    }
  }
}
