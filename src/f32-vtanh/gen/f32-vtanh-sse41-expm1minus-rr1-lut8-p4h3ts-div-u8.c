// Auto-generated file. Do not edit!
//   Template: src/f32-vtanh/sse-expm1minus.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <smmintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/microparams.h>
#include <xnnpack/vunary.h>


// Table of exp2(k / 8) values decremented (as integer) by (k << 20), k = 0..7
extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_8[8];

void xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vsign_mask = _mm_load_ps(params->sse_expm1minus_rr1_lut8_p4h3.sign_mask);
  const __m128 vsat_cutoff = _mm_load_ps(params->sse_expm1minus_rr1_lut8_p4h3.sat_cutoff);
  const __m128 vlog2e = _mm_load_ps(params->sse_expm1minus_rr1_lut8_p4h3.log2e);
  const __m128 vmagic_bias = _mm_load_ps(params->sse_expm1minus_rr1_lut8_p4h3.magic_bias);
  const __m128i vindex_mask = _mm_load_si128((const __m128i*) params->sse_expm1minus_rr1_lut8_p4h3.index_mask);
  const __m128 vminus_ln2 = _mm_load_ps(params->sse_expm1minus_rr1_lut8_p4h3.minus_ln2);
  const __m128 vc4 = _mm_load_ps(params->sse_expm1minus_rr1_lut8_p4h3.c4);
  const __m128 vc3 = _mm_load_ps(params->sse_expm1minus_rr1_lut8_p4h3.c3);
  const __m128 vc2 = _mm_load_ps(params->sse_expm1minus_rr1_lut8_p4h3.c2);
  const __m128 vminus_two = _mm_load_ps(params->sse_expm1minus_rr1_lut8_p4h3.minus_two);
  const __m128 vminus_one = _mm_load_ps(params->sse_expm1minus_rr1_lut8_p4h3.minus_one);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(input);
    const __m128 vx4567 = _mm_loadu_ps(input + 4);
    input += 8;

    __m128 vz0123 = _mm_or_ps(vx0123, vsign_mask);
    __m128 vz4567 = _mm_or_ps(vx4567, vsign_mask);

    const __m128 vinvsignx0123 = _mm_xor_ps(vx0123, vz0123);
    const __m128 vinvsignx4567 = _mm_xor_ps(vx4567, vz4567);

    vz0123 = _mm_max_ps(vsat_cutoff, vz0123);
    vz4567 = _mm_max_ps(vsat_cutoff, vz4567);

    __m128 vn0123 = _mm_add_ps(_mm_mul_ps(vz0123, vlog2e), vmagic_bias);
    __m128 vn4567 = _mm_add_ps(_mm_mul_ps(vz4567, vlog2e), vmagic_bias);

    const __m128i ve0123 = _mm_slli_epi32(_mm_castps_si128(vn0123), 20);
    const __m128i ve4567 = _mm_slli_epi32(_mm_castps_si128(vn4567), 20);

    #if XNN_ARCH_X86_64
      __m128i vidx0123 = _mm_and_si128(_mm_castps_si128(vn0123), vindex_mask);
      __m128i vidx4567 = _mm_and_si128(_mm_castps_si128(vn4567), vindex_mask);

      const uint64_t vidx01 = (uint64_t) _mm_cvtsi128_si64(vidx0123);
      vidx0123 = _mm_unpackhi_epi64(vidx0123, vidx0123);
      const uint64_t vidx45 = (uint64_t) _mm_cvtsi128_si64(vidx4567);
      vidx4567 = _mm_unpackhi_epi64(vidx4567, vidx4567);

      const uint64_t vidx23 = (uint64_t) _mm_cvtsi128_si64(vidx0123);
      const uint64_t vidx67 = (uint64_t) _mm_cvtsi128_si64(vidx4567);

      __m128i vl0123 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx01]);
      vl0123 = _mm_insert_epi32(vl0123, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx01 >> 32)], 1);
      vl0123 = _mm_insert_epi32(vl0123, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx23], 2);
      vl0123 = _mm_insert_epi32(vl0123, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx23 >> 32)], 3);
      __m128i vl4567 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx45]);
      vl4567 = _mm_insert_epi32(vl4567, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx45 >> 32)], 1);
      vl4567 = _mm_insert_epi32(vl4567, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx67], 2);
      vl4567 = _mm_insert_epi32(vl4567, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx67 >> 32)], 3);
    #else
      const __m128i vidx0123 = _mm_and_si128(_mm_castps_si128(vn0123), vindex_mask);
      const __m128i vidx4567 = _mm_and_si128(_mm_castps_si128(vn4567), vindex_mask);

      const uint32_t vidx0 = (uint32_t) _mm_cvtsi128_si32(vidx0123);
      const uint32_t vidx4 = (uint32_t) _mm_cvtsi128_si32(vidx4567);

      __m128i vl0123 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[vidx0]);
      __m128i vl4567 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[vidx4]);

      const uint32_t vidx1 = (uint32_t) _mm_extract_epi16(vidx0123, 2);
      const uint32_t vidx5 = (uint32_t) _mm_extract_epi16(vidx4567, 2);

      vl0123 = _mm_insert_epi32(vl0123, (int) xnn_table_exp2minus_k_over_8[vidx1], 1);
      vl4567 = _mm_insert_epi32(vl4567, (int) xnn_table_exp2minus_k_over_8[vidx5], 1);

      const uint32_t vidx2 = (uint32_t) _mm_extract_epi16(vidx0123, 4);
      const uint32_t vidx6 = (uint32_t) _mm_extract_epi16(vidx4567, 4);

      vl0123 = _mm_insert_epi32(vl0123, (int) xnn_table_exp2minus_k_over_8[vidx2], 2);
      vl4567 = _mm_insert_epi32(vl4567, (int) xnn_table_exp2minus_k_over_8[vidx6], 2);

      const uint32_t vidx3 = (uint32_t) _mm_extract_epi16(vidx0123, 6);
      const uint32_t vidx7 = (uint32_t) _mm_extract_epi16(vidx4567, 6);

      vl0123 = _mm_insert_epi32(vl0123, (int) xnn_table_exp2minus_k_over_8[vidx3], 3);
      vl4567 = _mm_insert_epi32(vl4567, (int) xnn_table_exp2minus_k_over_8[vidx7], 3);
    #endif

    const __m128 vs0123 = _mm_castsi128_ps(_mm_add_epi32(vl0123, ve0123));
    const __m128 vs4567 = _mm_castsi128_ps(_mm_add_epi32(vl4567, ve4567));

    vn0123 = _mm_sub_ps(vn0123, vmagic_bias);
    vn4567 = _mm_sub_ps(vn4567, vmagic_bias);

    const __m128 vt0123 = _mm_add_ps(_mm_mul_ps(vn0123, vminus_ln2), vz0123);
    const __m128 vt4567 = _mm_add_ps(_mm_mul_ps(vn4567, vminus_ln2), vz4567);

    __m128 vp0123 = _mm_add_ps(_mm_mul_ps(vc4, vt0123), vc3);
    __m128 vp4567 = _mm_add_ps(_mm_mul_ps(vc4, vt4567), vc3);
    vp0123 = _mm_add_ps(_mm_mul_ps(vp0123, vt0123), vc2);
    vp4567 = _mm_add_ps(_mm_mul_ps(vp4567, vt4567), vc2);
    vp0123 = _mm_sub_ps(_mm_mul_ps(vp0123, vt0123), vminus_two);
    vp4567 = _mm_sub_ps(_mm_mul_ps(vp4567, vt4567), vminus_two);

    const __m128 vts0123 = _mm_mul_ps(vt0123, vs0123);
    const __m128 vsmo0123 = _mm_add_ps(vs0123, vminus_one);
    const __m128 vts4567 = _mm_mul_ps(vt4567, vs4567);
    const __m128 vsmo4567 = _mm_add_ps(vs4567, vminus_one);
    const __m128 vemo0123 = _mm_add_ps(_mm_mul_ps(vp0123, vts0123), vsmo0123);
    const __m128 vemo4567 = _mm_add_ps(_mm_mul_ps(vp4567, vts4567), vsmo4567);

    const __m128 vepo0123 = _mm_sub_ps(vemo0123, vminus_two);
    const __m128 vepo4567 = _mm_sub_ps(vemo4567, vminus_two);

    __m128 vy0123 = _mm_div_ps(vemo0123, vepo0123);
    __m128 vy4567 = _mm_div_ps(vemo4567, vepo4567);


    vy0123 = _mm_xor_ps(vy0123, vinvsignx0123);
    vy4567 = _mm_xor_ps(vy4567, vinvsignx4567);

    _mm_storeu_ps(output, vy0123);
    _mm_storeu_ps(output + 4, vy4567);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vx = _mm_loadu_ps(input);
    input += 4;

    __m128 vz = _mm_or_ps(vx, vsign_mask);

    const __m128 vinvsignx = _mm_xor_ps(vx, vz);

    vz = _mm_max_ps(vsat_cutoff, vz);

    __m128 vn = _mm_add_ps(_mm_mul_ps(vz, vlog2e), vmagic_bias);

    const __m128i ve = _mm_slli_epi32(_mm_castps_si128(vn), 20);

    #if XNN_ARCH_X86_64
      __m128i vidx = _mm_and_si128(_mm_castps_si128(vn), vindex_mask);
      const uint64_t vidx_lo = (uint64_t) _mm_cvtsi128_si64(vidx);
      vidx = _mm_unpackhi_epi64(vidx, vidx);
      const uint64_t vidx_hi = (uint64_t) _mm_cvtsi128_si64(vidx);
      __m128i vl = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx_lo]);
      vl = _mm_insert_epi32(vl, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx_lo >> 32)], 1);
      vl = _mm_insert_epi32(vl, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx_hi], 2);
      vl = _mm_insert_epi32(vl, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx_hi >> 32)], 3);
    #else
      const __m128i vidx = _mm_and_si128(_mm_castps_si128(vn), vindex_mask);
      const uint32_t vidx0 = (uint32_t) _mm_cvtsi128_si32(vidx);
      __m128i vl = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[vidx0]);
      const uint32_t vidx1 = (uint32_t) _mm_extract_epi16(vidx, 2);
      vl = _mm_insert_epi32(vl, (int) xnn_table_exp2minus_k_over_8[vidx1], 1);
      const uint32_t vidx2 = (uint32_t) _mm_extract_epi16(vidx, 4);
      vl = _mm_insert_epi32(vl, (int) xnn_table_exp2minus_k_over_8[vidx2], 2);
      const uint32_t vidx3 = (uint32_t) _mm_extract_epi16(vidx, 6);
      vl = _mm_insert_epi32(vl, (int) xnn_table_exp2minus_k_over_8[vidx3], 3);
    #endif

    const __m128 vs = _mm_castsi128_ps(_mm_add_epi32(vl, ve));

    vn = _mm_sub_ps(vn, vmagic_bias);

    const __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2), vz);

    __m128 vp = _mm_add_ps(_mm_mul_ps(vc4, vt), vc3);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc2);
    vp = _mm_sub_ps(_mm_mul_ps(vp, vt), vminus_two);

    const __m128 vts = _mm_mul_ps(vt, vs);
    const __m128 vsmo = _mm_add_ps(vs, vminus_one);
    const __m128 vemo = _mm_add_ps(_mm_mul_ps(vp, vts), vsmo);

    const __m128 vepo = _mm_sub_ps(vemo, vminus_two);

    __m128 vy = _mm_div_ps(vemo, vepo);


    vy = _mm_xor_ps(vy, vinvsignx);

    _mm_storeu_ps(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 vx = _mm_loadu_ps(input);

    __m128 vz = _mm_or_ps(vx, vsign_mask);

    const __m128 vinvsignx = _mm_xor_ps(vx, vz);

    vz = _mm_max_ps(vsat_cutoff, vz);

    __m128 vn = _mm_add_ps(_mm_mul_ps(vz, vlog2e), vmagic_bias);

    const __m128i ve = _mm_slli_epi32(_mm_castps_si128(vn), 20);

    #if XNN_ARCH_X86_64
      __m128i vidx = _mm_and_si128(_mm_castps_si128(vn), vindex_mask);
      const uint64_t vidx_lo = (uint64_t) _mm_cvtsi128_si64(vidx);
      vidx = _mm_unpackhi_epi64(vidx, vidx);
      const uint64_t vidx_hi = (uint64_t) _mm_cvtsi128_si64(vidx);
      __m128i vl = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx_lo]);
      vl = _mm_insert_epi32(vl, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx_lo >> 32)], 1);
      vl = _mm_insert_epi32(vl, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx_hi], 2);
      vl = _mm_insert_epi32(vl, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx_hi >> 32)], 3);
    #else
      const __m128i vidx = _mm_and_si128(_mm_castps_si128(vn), vindex_mask);
      const uint32_t vidx0 = (uint32_t) _mm_cvtsi128_si32(vidx);
      __m128i vl = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[vidx0]);
      const uint32_t vidx1 = (uint32_t) _mm_extract_epi16(vidx, 2);
      vl = _mm_insert_epi32(vl, (int) xnn_table_exp2minus_k_over_8[vidx1], 1);
      const uint32_t vidx2 = (uint32_t) _mm_extract_epi16(vidx, 4);
      vl = _mm_insert_epi32(vl, (int) xnn_table_exp2minus_k_over_8[vidx2], 2);
      const uint32_t vidx3 = (uint32_t) _mm_extract_epi16(vidx, 6);
      vl = _mm_insert_epi32(vl, (int) xnn_table_exp2minus_k_over_8[vidx3], 3);
    #endif

    const __m128 vs = _mm_castsi128_ps(_mm_add_epi32(vl, ve));

    vn = _mm_sub_ps(vn, vmagic_bias);

    const __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2), vz);

    __m128 vp = _mm_add_ps(_mm_mul_ps(vc4, vt), vc3);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc2);
    vp = _mm_sub_ps(_mm_mul_ps(vp, vt), vminus_two);

    const __m128 vts = _mm_mul_ps(vt, vs);
    const __m128 vsmo = _mm_add_ps(vs, vminus_one);
    const __m128 vemo = _mm_add_ps(_mm_mul_ps(vp, vts), vsmo);

    const __m128 vepo = _mm_sub_ps(vemo, vminus_two);

    __m128 vy = _mm_div_ps(vemo, vepo);


    vy = _mm_xor_ps(vy, vinvsignx);

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy);
      vy = _mm_movehl_ps(vy, vy);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy);
    }
  }
}
