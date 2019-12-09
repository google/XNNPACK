// Auto-generated file. Do not edit!
//   Template: src/f32-vscaleextexp/avx2-p5.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/vscaleextexp.h>


static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

void xnn_f32_vscaleextexp_ukernel__avx2_p5_x24(
    size_t elements,
    const float* x,
    float* y,
    float scale_value,
    float scale_exp)
{
  assert(elements % sizeof(float) == 0);

  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p+0f);
  const __m256 vminus_ln2_hi = _mm256_set1_ps(-0x1.62E43p-1f);
  const __m256 vminus_ln2_lo = _mm256_set1_ps(0x1.05C61p-29f);

  // The smallest elements such that 2**elements is considered non-negligible.
  // For smaller elements, 2**elements is replaced with zero.
  const __m256 vmin_exponent = _mm256_set1_ps(-127.0f);
  const __m256 vmagic_bias = _mm256_set1_ps(0x1.8000FEp23f);

  const __m256 vc0 = _mm256_set1_ps(1.0f);
  const __m256 vc1 = _mm256_set1_ps(0x1.FFFFF6p-1f);
  const __m256 vc2 = _mm256_set1_ps(0x1.FFFDC6p-2f);
  const __m256 vc3 = _mm256_set1_ps(0x1.555A80p-3f);
  const __m256 vc4 = _mm256_set1_ps(0x1.573A1Ap-5f);
  const __m256 vc5 = _mm256_set1_ps(0x1.0F9F9Cp-7f);

  const __m256 vscalev = _mm256_set1_ps(scale_value);
  const __m256 vscalee = _mm256_set1_ps(scale_exp);

  for (; elements >= 24 * sizeof(float); elements -= 24 * sizeof(float)) {
    // Load 24 (3x8) inputs at a time.
    const __m256 vx0 = _mm256_loadu_ps(x);
    const __m256 vx1 = _mm256_loadu_ps(x + 8);
    const __m256 vx2 = _mm256_loadu_ps(x + 16);
    x += 24;

    // Compute reduced argument elements := round(x / log(2)).
    const __m256 vn0 = _mm256_round_ps(_mm256_mul_ps(vx0, vlog2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 vn1 = _mm256_round_ps(_mm256_mul_ps(vx1, vlog2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 vn2 = _mm256_round_ps(_mm256_mul_ps(vx2, vlog2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // Compute reduced argument t := x - elements * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m256 vt0 = _mm256_fmadd_ps(vn0, vminus_ln2_hi, vx0);
    __m256 vt1 = _mm256_fmadd_ps(vn1, vminus_ln2_hi, vx1);
    __m256 vt2 = _mm256_fmadd_ps(vn2, vminus_ln2_hi, vx2);

    vt0 = _mm256_fmadd_ps(vn0, vminus_ln2_lo, vt0);
    vt1 = _mm256_fmadd_ps(vn1, vminus_ln2_lo, vt1);
    vt2 = _mm256_fmadd_ps(vn2, vminus_ln2_lo, vt2);

    // Compute degree-5 polynomial approxiatmion for exp(t) on [-log(2)/2, log(2)/2].
    __m256 vp0 = _mm256_fmadd_ps(vc5, vt0, vc4);
    __m256 vp1 = _mm256_fmadd_ps(vc5, vt1, vc4);
    __m256 vp2 = _mm256_fmadd_ps(vc5, vt2, vc4);

    vp0 = _mm256_fmadd_ps(vp0, vt0, vc3);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc3);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc3);

    vp0 = _mm256_fmadd_ps(vp0, vt0, vc2);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc2);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc2);

    vp0 = _mm256_fmadd_ps(vp0, vt0, vc1);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc1);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc1);

    vp0 = _mm256_fmadd_ps(vp0, vt0, vc0);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc0);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc0);

    // Multiply "extended" floating-point numbers in ("mantissa", "exponent") representation where
    //  - vnX is "exponent"
    //  - vpX is "mantissa"
    //   
    // exp2(ae) * av * exp2(be) * bv = 
    //   = exp2(ae + be) * (av * bv)
    __m256 vf0 = _mm256_mul_ps(vp0, vscalev);
    __m256 vf1 = _mm256_mul_ps(vp1, vscalev);
    __m256 vf2 = _mm256_mul_ps(vp2, vscalev);

    __m256 ve0 = _mm256_add_ps(vn0, vscalee);
    __m256 ve1 = _mm256_add_ps(vn1, vscalee);
    __m256 ve2 = _mm256_add_ps(vn2, vscalee);

    // For computational efficiency, replace exp2(e) with 0.0f when e <= -127.0.
    // This replacement is done in two steps:
    // 1. Clamp minimum e at -127.0.
    // 2. Map e to scale factor 0.0 when e == -127.0
    ve0 = _mm256_max_ps(ve0, vmin_exponent);
    ve1 = _mm256_max_ps(ve1, vmin_exponent);
    ve2 = _mm256_max_ps(ve2, vmin_exponent);

    // Convert exponents into scale factors:
    // - s = exp2(e) when e > -127.0
    // - s = 0.0 when e <= -127.0
    const __m256 vs0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(_mm256_add_ps(ve0, vmagic_bias)), 23));
    const __m256 vs1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(_mm256_add_ps(ve1, vmagic_bias)), 23));
    const __m256 vs2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(_mm256_add_ps(ve2, vmagic_bias)), 23));

    // Multiply "mantissa" by the scale factor.
    vf0 = _mm256_mul_ps(vf0, vs0);
    vf1 = _mm256_mul_ps(vf1, vs1);
    vf2 = _mm256_mul_ps(vf2, vs2);

    // Store 24 (3x8) outputs at a time.
    _mm256_storeu_ps(y, vf0);
    _mm256_storeu_ps(y + 8, vf1);
    _mm256_storeu_ps(y + 16, vf2);
    y += 24;
  }

  for (; elements >= 8 * sizeof(float); elements -= 8 * sizeof(float)) {
    // Load 8 inputs at a time.
    const __m256 vx = _mm256_loadu_ps(x);
    x += 8;

    // Compute reduced argument elements := round(x / log(2)).
    const __m256 vn = _mm256_round_ps(_mm256_mul_ps(vx, vlog2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // Compute reduced argument t := x - elements * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2_hi, vx);
    vt = _mm256_fmadd_ps(vn, vminus_ln2_lo, vt);

    // Compute degree-5 polynomial approxiatmion for exp(t) on [-log(2)/2, log(2)/2].
    __m256 vp = _mm256_fmadd_ps(vc5, vt, vc4);
    vp = _mm256_fmadd_ps(vp, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_fmadd_ps(vp, vt, vc1);
    vp = _mm256_fmadd_ps(vp, vt, vc0);

    // Multiply "extended" floating-point numbers in ("mantissa", "exponent") representation.
    __m256 vf = _mm256_mul_ps(vp, vscalev);
    __m256 ve = _mm256_add_ps(vn, vscalee);

    // For computational efficiency, replace exp2(e) with 0.0f when e <= -127.0.
    ve = _mm256_max_ps(ve, vmin_exponent);

    // Convert exponents into scale factors.
    const __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(_mm256_add_ps(ve, vmagic_bias)), 23));

    // Multiply "mantissa" by the scale factor.
    vf = _mm256_mul_ps(vf, vs);

    // Store 8 results at a time.
    _mm256_storeu_ps(y, vf);
    y += 8;
  }
  if XNN_UNLIKELY(elements != 0) {
    assert(elements >= 1 * sizeof(float));
    assert(elements <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7] - elements));

    // Load up to 7 inputs at a time.
    const __m256 vx = _mm256_maskload_ps(x, vmask);

    // Compute reduced argument elements := round(x / log(2)).
    const __m256 vn = _mm256_round_ps(_mm256_mul_ps(vx, vlog2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // Compute reduced argument t := x - elements * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2_hi, vx);
    vt = _mm256_fmadd_ps(vn, vminus_ln2_lo, vt);

    // Compute degree-5 polynomial approxiatmion for exp(t) on [-log(2)/2, log(2)/2].
    __m256 vp = _mm256_fmadd_ps(vc5, vt, vc4);
    vp = _mm256_fmadd_ps(vp, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_fmadd_ps(vp, vt, vc1);
    vp = _mm256_fmadd_ps(vp, vt, vc0);

    // Multiply "extended" floating-point numbers in ("mantissa", "exponent") representation.
    __m256 vf = _mm256_mul_ps(vp, vscalev);
    __m256 ve = _mm256_add_ps(vn, vscalee);

    // For computational efficiency, replace exp2(e) with 0.0f when e <= -127.0.
    ve = _mm256_max_ps(ve, vmin_exponent);

    // Convert exponents into scale factors.
    const __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(_mm256_add_ps(ve, vmagic_bias)), 23));

    // Multiply "mantissa" by the scale factor.
    vf = _mm256_mul_ps(vf, vs);

    // Store up to 7 inputs at a time.
    _mm256_maskstore_ps(y, vmask, vf);
  }
  _mm256_zeroupper();
}
