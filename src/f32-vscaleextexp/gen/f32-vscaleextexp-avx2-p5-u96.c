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

#include "xnnpack/common.h"
#include "xnnpack/vscaleextexp.h"


static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

void xnn_f32_vscaleextexp_ukernel__avx2_p5_u96(
    size_t batch,
    const float* input,
    float* output,
    float scale_value,
    float scale_exp)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256 vlog2e = _mm256_set1_ps(0x1.715476p+0f);
  const __m256 vminus_ln2_hi = _mm256_set1_ps(-0x1.62E43p-1f);
  const __m256 vminus_ln2_lo = _mm256_set1_ps(0x1.05C61p-29f);

  // The smallest batch such that 2**batch is considered non-negligible.
  // For smaller batch, 2**batch is replaced with zero.
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

  for (; batch >= 96 * sizeof(float); batch -= 96 * sizeof(float)) {
    // Load 96 (12x8) inputs at a time.
    const __m256 vx0 = _mm256_loadu_ps(input);
    const __m256 vx1 = _mm256_loadu_ps(input + 8);
    const __m256 vx2 = _mm256_loadu_ps(input + 16);
    const __m256 vx3 = _mm256_loadu_ps(input + 24);
    const __m256 vx4 = _mm256_loadu_ps(input + 32);
    const __m256 vx5 = _mm256_loadu_ps(input + 40);
    const __m256 vx6 = _mm256_loadu_ps(input + 48);
    const __m256 vx7 = _mm256_loadu_ps(input + 56);
    const __m256 vx8 = _mm256_loadu_ps(input + 64);
    const __m256 vx9 = _mm256_loadu_ps(input + 72);
    const __m256 vx10 = _mm256_loadu_ps(input + 80);
    const __m256 vx11 = _mm256_loadu_ps(input + 88);
    input += 96;

    // Compute reduced argument batch := round(input / log(2)).
    const __m256 vn0 = _mm256_round_ps(_mm256_mul_ps(vx0, vlog2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 vn1 = _mm256_round_ps(_mm256_mul_ps(vx1, vlog2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 vn2 = _mm256_round_ps(_mm256_mul_ps(vx2, vlog2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 vn3 = _mm256_round_ps(_mm256_mul_ps(vx3, vlog2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 vn4 = _mm256_round_ps(_mm256_mul_ps(vx4, vlog2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 vn5 = _mm256_round_ps(_mm256_mul_ps(vx5, vlog2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 vn6 = _mm256_round_ps(_mm256_mul_ps(vx6, vlog2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 vn7 = _mm256_round_ps(_mm256_mul_ps(vx7, vlog2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 vn8 = _mm256_round_ps(_mm256_mul_ps(vx8, vlog2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 vn9 = _mm256_round_ps(_mm256_mul_ps(vx9, vlog2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 vn10 = _mm256_round_ps(_mm256_mul_ps(vx10, vlog2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 vn11 = _mm256_round_ps(_mm256_mul_ps(vx11, vlog2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // Compute reduced argument t := input - batch * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m256 vt0 = _mm256_fmadd_ps(vn0, vminus_ln2_hi, vx0);
    __m256 vt1 = _mm256_fmadd_ps(vn1, vminus_ln2_hi, vx1);
    __m256 vt2 = _mm256_fmadd_ps(vn2, vminus_ln2_hi, vx2);
    __m256 vt3 = _mm256_fmadd_ps(vn3, vminus_ln2_hi, vx3);
    __m256 vt4 = _mm256_fmadd_ps(vn4, vminus_ln2_hi, vx4);
    __m256 vt5 = _mm256_fmadd_ps(vn5, vminus_ln2_hi, vx5);
    __m256 vt6 = _mm256_fmadd_ps(vn6, vminus_ln2_hi, vx6);
    __m256 vt7 = _mm256_fmadd_ps(vn7, vminus_ln2_hi, vx7);
    __m256 vt8 = _mm256_fmadd_ps(vn8, vminus_ln2_hi, vx8);
    __m256 vt9 = _mm256_fmadd_ps(vn9, vminus_ln2_hi, vx9);
    __m256 vt10 = _mm256_fmadd_ps(vn10, vminus_ln2_hi, vx10);
    __m256 vt11 = _mm256_fmadd_ps(vn11, vminus_ln2_hi, vx11);

    vt0 = _mm256_fmadd_ps(vn0, vminus_ln2_lo, vt0);
    vt1 = _mm256_fmadd_ps(vn1, vminus_ln2_lo, vt1);
    vt2 = _mm256_fmadd_ps(vn2, vminus_ln2_lo, vt2);
    vt3 = _mm256_fmadd_ps(vn3, vminus_ln2_lo, vt3);
    vt4 = _mm256_fmadd_ps(vn4, vminus_ln2_lo, vt4);
    vt5 = _mm256_fmadd_ps(vn5, vminus_ln2_lo, vt5);
    vt6 = _mm256_fmadd_ps(vn6, vminus_ln2_lo, vt6);
    vt7 = _mm256_fmadd_ps(vn7, vminus_ln2_lo, vt7);
    vt8 = _mm256_fmadd_ps(vn8, vminus_ln2_lo, vt8);
    vt9 = _mm256_fmadd_ps(vn9, vminus_ln2_lo, vt9);
    vt10 = _mm256_fmadd_ps(vn10, vminus_ln2_lo, vt10);
    vt11 = _mm256_fmadd_ps(vn11, vminus_ln2_lo, vt11);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    __m256 vp0 = _mm256_fmadd_ps(vc5, vt0, vc4);
    __m256 vp1 = _mm256_fmadd_ps(vc5, vt1, vc4);
    __m256 vp2 = _mm256_fmadd_ps(vc5, vt2, vc4);
    __m256 vp3 = _mm256_fmadd_ps(vc5, vt3, vc4);
    __m256 vp4 = _mm256_fmadd_ps(vc5, vt4, vc4);
    __m256 vp5 = _mm256_fmadd_ps(vc5, vt5, vc4);
    __m256 vp6 = _mm256_fmadd_ps(vc5, vt6, vc4);
    __m256 vp7 = _mm256_fmadd_ps(vc5, vt7, vc4);
    __m256 vp8 = _mm256_fmadd_ps(vc5, vt8, vc4);
    __m256 vp9 = _mm256_fmadd_ps(vc5, vt9, vc4);
    __m256 vp10 = _mm256_fmadd_ps(vc5, vt10, vc4);
    __m256 vp11 = _mm256_fmadd_ps(vc5, vt11, vc4);

    vp0 = _mm256_fmadd_ps(vp0, vt0, vc3);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc3);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc3);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc3);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vc3);
    vp5 = _mm256_fmadd_ps(vp5, vt5, vc3);
    vp6 = _mm256_fmadd_ps(vp6, vt6, vc3);
    vp7 = _mm256_fmadd_ps(vp7, vt7, vc3);
    vp8 = _mm256_fmadd_ps(vp8, vt8, vc3);
    vp9 = _mm256_fmadd_ps(vp9, vt9, vc3);
    vp10 = _mm256_fmadd_ps(vp10, vt10, vc3);
    vp11 = _mm256_fmadd_ps(vp11, vt11, vc3);

    vp0 = _mm256_fmadd_ps(vp0, vt0, vc2);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc2);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc2);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc2);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vc2);
    vp5 = _mm256_fmadd_ps(vp5, vt5, vc2);
    vp6 = _mm256_fmadd_ps(vp6, vt6, vc2);
    vp7 = _mm256_fmadd_ps(vp7, vt7, vc2);
    vp8 = _mm256_fmadd_ps(vp8, vt8, vc2);
    vp9 = _mm256_fmadd_ps(vp9, vt9, vc2);
    vp10 = _mm256_fmadd_ps(vp10, vt10, vc2);
    vp11 = _mm256_fmadd_ps(vp11, vt11, vc2);

    vp0 = _mm256_fmadd_ps(vp0, vt0, vc1);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc1);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc1);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc1);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vc1);
    vp5 = _mm256_fmadd_ps(vp5, vt5, vc1);
    vp6 = _mm256_fmadd_ps(vp6, vt6, vc1);
    vp7 = _mm256_fmadd_ps(vp7, vt7, vc1);
    vp8 = _mm256_fmadd_ps(vp8, vt8, vc1);
    vp9 = _mm256_fmadd_ps(vp9, vt9, vc1);
    vp10 = _mm256_fmadd_ps(vp10, vt10, vc1);
    vp11 = _mm256_fmadd_ps(vp11, vt11, vc1);

    vp0 = _mm256_fmadd_ps(vp0, vt0, vc0);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc0);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc0);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc0);
    vp4 = _mm256_fmadd_ps(vp4, vt4, vc0);
    vp5 = _mm256_fmadd_ps(vp5, vt5, vc0);
    vp6 = _mm256_fmadd_ps(vp6, vt6, vc0);
    vp7 = _mm256_fmadd_ps(vp7, vt7, vc0);
    vp8 = _mm256_fmadd_ps(vp8, vt8, vc0);
    vp9 = _mm256_fmadd_ps(vp9, vt9, vc0);
    vp10 = _mm256_fmadd_ps(vp10, vt10, vc0);
    vp11 = _mm256_fmadd_ps(vp11, vt11, vc0);

    // Multiply "extended" floating-point numbers in ("mantissa", "exponent") representation where
    //  - vnX is "exponent"
    //  - vpX is "mantissa"
    //
    // exp2(ae) * av * exp2(be) * bv =
    //   = exp2(ae + be) * (av * bv)
    __m256 vf0 = _mm256_mul_ps(vp0, vscalev);
    __m256 vf1 = _mm256_mul_ps(vp1, vscalev);
    __m256 vf2 = _mm256_mul_ps(vp2, vscalev);
    __m256 vf3 = _mm256_mul_ps(vp3, vscalev);
    __m256 vf4 = _mm256_mul_ps(vp4, vscalev);
    __m256 vf5 = _mm256_mul_ps(vp5, vscalev);
    __m256 vf6 = _mm256_mul_ps(vp6, vscalev);
    __m256 vf7 = _mm256_mul_ps(vp7, vscalev);
    __m256 vf8 = _mm256_mul_ps(vp8, vscalev);
    __m256 vf9 = _mm256_mul_ps(vp9, vscalev);
    __m256 vf10 = _mm256_mul_ps(vp10, vscalev);
    __m256 vf11 = _mm256_mul_ps(vp11, vscalev);

    __m256 ve0 = _mm256_add_ps(vn0, vscalee);
    __m256 ve1 = _mm256_add_ps(vn1, vscalee);
    __m256 ve2 = _mm256_add_ps(vn2, vscalee);
    __m256 ve3 = _mm256_add_ps(vn3, vscalee);
    __m256 ve4 = _mm256_add_ps(vn4, vscalee);
    __m256 ve5 = _mm256_add_ps(vn5, vscalee);
    __m256 ve6 = _mm256_add_ps(vn6, vscalee);
    __m256 ve7 = _mm256_add_ps(vn7, vscalee);
    __m256 ve8 = _mm256_add_ps(vn8, vscalee);
    __m256 ve9 = _mm256_add_ps(vn9, vscalee);
    __m256 ve10 = _mm256_add_ps(vn10, vscalee);
    __m256 ve11 = _mm256_add_ps(vn11, vscalee);

    // For computational efficiency, replace exp2(e) with 0.0f when e <= -127.0.
    // This replacement is done in two steps:
    // 1. Clamp minimum e at -127.0.
    // 2. Map e to scale factor 0.0 when e == -127.0
    ve0 = _mm256_max_ps(ve0, vmin_exponent);
    ve1 = _mm256_max_ps(ve1, vmin_exponent);
    ve2 = _mm256_max_ps(ve2, vmin_exponent);
    ve3 = _mm256_max_ps(ve3, vmin_exponent);
    ve4 = _mm256_max_ps(ve4, vmin_exponent);
    ve5 = _mm256_max_ps(ve5, vmin_exponent);
    ve6 = _mm256_max_ps(ve6, vmin_exponent);
    ve7 = _mm256_max_ps(ve7, vmin_exponent);
    ve8 = _mm256_max_ps(ve8, vmin_exponent);
    ve9 = _mm256_max_ps(ve9, vmin_exponent);
    ve10 = _mm256_max_ps(ve10, vmin_exponent);
    ve11 = _mm256_max_ps(ve11, vmin_exponent);

    // Convert exponents into scale factors:
    // - s = exp2(e) when e > -127.0
    // - s = 0.0 when e <= -127.0
    const __m256 vs0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(_mm256_add_ps(ve0, vmagic_bias)), 23));
    const __m256 vs1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(_mm256_add_ps(ve1, vmagic_bias)), 23));
    const __m256 vs2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(_mm256_add_ps(ve2, vmagic_bias)), 23));
    const __m256 vs3 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(_mm256_add_ps(ve3, vmagic_bias)), 23));
    const __m256 vs4 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(_mm256_add_ps(ve4, vmagic_bias)), 23));
    const __m256 vs5 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(_mm256_add_ps(ve5, vmagic_bias)), 23));
    const __m256 vs6 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(_mm256_add_ps(ve6, vmagic_bias)), 23));
    const __m256 vs7 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(_mm256_add_ps(ve7, vmagic_bias)), 23));
    const __m256 vs8 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(_mm256_add_ps(ve8, vmagic_bias)), 23));
    const __m256 vs9 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(_mm256_add_ps(ve9, vmagic_bias)), 23));
    const __m256 vs10 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(_mm256_add_ps(ve10, vmagic_bias)), 23));
    const __m256 vs11 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(_mm256_add_ps(ve11, vmagic_bias)), 23));

    // Multiply "mantissa" by the scale factor.
    vf0 = _mm256_mul_ps(vf0, vs0);
    vf1 = _mm256_mul_ps(vf1, vs1);
    vf2 = _mm256_mul_ps(vf2, vs2);
    vf3 = _mm256_mul_ps(vf3, vs3);
    vf4 = _mm256_mul_ps(vf4, vs4);
    vf5 = _mm256_mul_ps(vf5, vs5);
    vf6 = _mm256_mul_ps(vf6, vs6);
    vf7 = _mm256_mul_ps(vf7, vs7);
    vf8 = _mm256_mul_ps(vf8, vs8);
    vf9 = _mm256_mul_ps(vf9, vs9);
    vf10 = _mm256_mul_ps(vf10, vs10);
    vf11 = _mm256_mul_ps(vf11, vs11);

    // Store 96 (12x8) outputs at a time.
    _mm256_storeu_ps(output, vf0);
    _mm256_storeu_ps(output + 8, vf1);
    _mm256_storeu_ps(output + 16, vf2);
    _mm256_storeu_ps(output + 24, vf3);
    _mm256_storeu_ps(output + 32, vf4);
    _mm256_storeu_ps(output + 40, vf5);
    _mm256_storeu_ps(output + 48, vf6);
    _mm256_storeu_ps(output + 56, vf7);
    _mm256_storeu_ps(output + 64, vf8);
    _mm256_storeu_ps(output + 72, vf9);
    _mm256_storeu_ps(output + 80, vf10);
    _mm256_storeu_ps(output + 88, vf11);
    output += 96;
  }

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    // Load 8 inputs at a time.
    const __m256 vx = _mm256_loadu_ps(input);
    input += 8;

    // Compute reduced argument batch := round(input / log(2)).
    const __m256 vn = _mm256_round_ps(_mm256_mul_ps(vx, vlog2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // Compute reduced argument t := input - batch * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2_hi, vx);
    vt = _mm256_fmadd_ps(vn, vminus_ln2_lo, vt);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
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
    _mm256_storeu_ps(output, vf);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7] - batch));

    // Load up to 7 inputs at a time.
    const __m256 vx = _mm256_maskload_ps(input, vmask);

    // Compute reduced argument batch := round(input / log(2)).
    const __m256 vn = _mm256_round_ps(_mm256_mul_ps(vx, vlog2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // Compute reduced argument t := input - batch * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2_hi, vx);
    vt = _mm256_fmadd_ps(vn, vminus_ln2_lo, vt);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
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
    _mm256_maskstore_ps(output, vmask, vf);
  }
  _mm256_zeroupper();
}
