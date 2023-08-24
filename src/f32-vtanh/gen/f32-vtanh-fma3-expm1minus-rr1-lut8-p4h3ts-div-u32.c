// Auto-generated file. Do not edit!
//   Template: src/f32-vtanh/avx-expm1minus-lut.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/microparams.h>
#include <xnnpack/vunary.h>

// Table of exp2(k / 8) values decremented (as integer) by (k << 20), k = 0..7
extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_8[8];

void xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u32(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256 vsign_mask = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3.sign_mask);
  const __m256 vsat_cutoff = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3.sat_cutoff);
  const __m256 vlog2e = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3.log2e);
  const __m256 vmagic_bias = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3.magic_bias);
  const __m128i vindex_mask = _mm_load_si128((const __m128i*) params->avx_expm1minus_rr1_lut8_p4h3.index_mask);
  const __m256 vminus_ln2 = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3.minus_ln2);
  const __m256 vc4 = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3.c4);
  const __m256 vc3 = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3.c3);
  const __m256 vc2 = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3.c2);
  const __m256 vtwo = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3.two);
  const __m256 vminus_one = _mm256_load_ps(params->avx_expm1minus_rr1_lut8_p4h3.minus_one);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const __m256 vx01234567 = _mm256_loadu_ps(input);
    const __m256 vx89ABCDEF = _mm256_loadu_ps(input + 8);
    const __m256 vxGHIJKLMN = _mm256_loadu_ps(input + 16);
    const __m256 vxOPQRSTUV = _mm256_loadu_ps(input + 24);
    input += 32;

    __m256 vz01234567 = _mm256_or_ps(vx01234567, vsign_mask);
    __m256 vz89ABCDEF = _mm256_or_ps(vx89ABCDEF, vsign_mask);
    __m256 vzGHIJKLMN = _mm256_or_ps(vxGHIJKLMN, vsign_mask);
    __m256 vzOPQRSTUV = _mm256_or_ps(vxOPQRSTUV, vsign_mask);

    const __m256 vinvsignx01234567 = _mm256_xor_ps(vx01234567, vz01234567);
    vz01234567 = _mm256_max_ps(vsat_cutoff, vz01234567);
    const __m256 vinvsignx89ABCDEF = _mm256_xor_ps(vx89ABCDEF, vz89ABCDEF);
    vz89ABCDEF = _mm256_max_ps(vsat_cutoff, vz89ABCDEF);
    const __m256 vinvsignxGHIJKLMN = _mm256_xor_ps(vxGHIJKLMN, vzGHIJKLMN);
    vzGHIJKLMN = _mm256_max_ps(vsat_cutoff, vzGHIJKLMN);
    const __m256 vinvsignxOPQRSTUV = _mm256_xor_ps(vxOPQRSTUV, vzOPQRSTUV);
    vzOPQRSTUV = _mm256_max_ps(vsat_cutoff, vzOPQRSTUV);

    __m256 vn01234567 = _mm256_fmadd_ps(vz01234567, vlog2e, vmagic_bias);
    __m256 vn89ABCDEF = _mm256_fmadd_ps(vz89ABCDEF, vlog2e, vmagic_bias);
    __m256 vnGHIJKLMN = _mm256_fmadd_ps(vzGHIJKLMN, vlog2e, vmagic_bias);
    __m256 vnOPQRSTUV = _mm256_fmadd_ps(vzOPQRSTUV, vlog2e, vmagic_bias);

    const __m128 vn4567 = _mm256_extractf128_ps(vn01234567, 1);
    const __m128i ve0123 = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn01234567)), 20);
    const __m128 vnCDEF = _mm256_extractf128_ps(vn89ABCDEF, 1);
    const __m128i ve89AB = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn89ABCDEF)), 20);
    const __m128 vnKLMN = _mm256_extractf128_ps(vnGHIJKLMN, 1);
    const __m128i veGHIJ = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vnGHIJKLMN)), 20);
    const __m128 vnSTUV = _mm256_extractf128_ps(vnOPQRSTUV, 1);
    const __m128i veOPQR = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vnOPQRSTUV)), 20);

    const __m128i ve4567 = _mm_slli_epi32(_mm_castps_si128(vn4567), 20);
    const __m128i vidx0123 = _mm_and_si128(_mm_castps_si128(_mm256_castps256_ps128(vn01234567)), vindex_mask);
    const __m128i vidx4567 = _mm_and_si128(_mm_castps_si128(vn4567), vindex_mask);
    const __m128i veCDEF = _mm_slli_epi32(_mm_castps_si128(vnCDEF), 20);
    const __m128i vidx89AB = _mm_and_si128(_mm_castps_si128(_mm256_castps256_ps128(vn89ABCDEF)), vindex_mask);
    const __m128i vidxCDEF = _mm_and_si128(_mm_castps_si128(vnCDEF), vindex_mask);
    const __m128i veKLMN = _mm_slli_epi32(_mm_castps_si128(vnKLMN), 20);
    const __m128i vidxGHIJ = _mm_and_si128(_mm_castps_si128(_mm256_castps256_ps128(vnGHIJKLMN)), vindex_mask);
    const __m128i vidxKLMN = _mm_and_si128(_mm_castps_si128(vnKLMN), vindex_mask);
    const __m128i veSTUV = _mm_slli_epi32(_mm_castps_si128(vnSTUV), 20);
    const __m128i vidxOPQR = _mm_and_si128(_mm_castps_si128(_mm256_castps256_ps128(vnOPQRSTUV)), vindex_mask);
    const __m128i vidxSTUV = _mm_and_si128(_mm_castps_si128(vnSTUV), vindex_mask);

    #if XNN_ARCH_X86_64
      const uint64_t vidx01 = (uint64_t) _mm_cvtsi128_si64(vidx0123);
      const uint64_t vidx45 = (uint64_t) _mm_cvtsi128_si64(vidx4567);
      const uint64_t vidx89 = (uint64_t) _mm_cvtsi128_si64(vidx89AB);
      const uint64_t vidxCD = (uint64_t) _mm_cvtsi128_si64(vidxCDEF);
      const uint64_t vidxGH = (uint64_t) _mm_cvtsi128_si64(vidxGHIJ);
      const uint64_t vidxKL = (uint64_t) _mm_cvtsi128_si64(vidxKLMN);
      const uint64_t vidxOP = (uint64_t) _mm_cvtsi128_si64(vidxOPQR);
      const uint64_t vidxST = (uint64_t) _mm_cvtsi128_si64(vidxSTUV);

      __m128i vl0123 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx01]);
      __m128i vl4567 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx45]);
      __m128i vl89AB = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx89]);
      __m128i vlCDEF = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxCD]);
      __m128i vlGHIJ = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxGH]);
      __m128i vlKLMN = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxKL]);
      __m128i vlOPQR = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxOP]);
      __m128i vlSTUV = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxST]);

      vl0123 = _mm_insert_epi32(vl0123, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx01 >> 32)], 1);
      vl4567 = _mm_insert_epi32(vl4567, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx45 >> 32)], 1);
      const uint64_t vidx23 = (uint64_t) _mm_extract_epi64(vidx0123, 1);
      const uint64_t vidx67 = (uint64_t) _mm_extract_epi64(vidx4567, 1);
      vl89AB = _mm_insert_epi32(vl89AB, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx89 >> 32)], 1);
      vlCDEF = _mm_insert_epi32(vlCDEF, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidxCD >> 32)], 1);
      const uint64_t vidxAB = (uint64_t) _mm_extract_epi64(vidx89AB, 1);
      const uint64_t vidxEF = (uint64_t) _mm_extract_epi64(vidxCDEF, 1);
      vlGHIJ = _mm_insert_epi32(vlGHIJ, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidxGH >> 32)], 1);
      vlKLMN = _mm_insert_epi32(vlKLMN, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidxKL >> 32)], 1);
      const uint64_t vidxIJ = (uint64_t) _mm_extract_epi64(vidxGHIJ, 1);
      const uint64_t vidxMN = (uint64_t) _mm_extract_epi64(vidxKLMN, 1);
      vlOPQR = _mm_insert_epi32(vlOPQR, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidxOP >> 32)], 1);
      vlSTUV = _mm_insert_epi32(vlSTUV, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidxST >> 32)], 1);
      const uint64_t vidxQR = (uint64_t) _mm_extract_epi64(vidxOPQR, 1);
      const uint64_t vidxUV = (uint64_t) _mm_extract_epi64(vidxSTUV, 1);

      vl0123 = _mm_insert_epi32(vl0123, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx23], 2);
      vl4567 = _mm_insert_epi32(vl4567, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx67], 2);
      vl89AB = _mm_insert_epi32(vl89AB, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxAB], 2);
      vlCDEF = _mm_insert_epi32(vlCDEF, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxEF], 2);
      vlGHIJ = _mm_insert_epi32(vlGHIJ, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxIJ], 2);
      vlKLMN = _mm_insert_epi32(vlKLMN, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxMN], 2);
      vlOPQR = _mm_insert_epi32(vlOPQR, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxQR], 2);
      vlSTUV = _mm_insert_epi32(vlSTUV, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxUV], 2);

      vl0123 = _mm_insert_epi32(vl0123, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx23 >> 32)], 3);
      vl4567 = _mm_insert_epi32(vl4567, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx67 >> 32)], 3);
      vl89AB = _mm_insert_epi32(vl89AB, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidxAB >> 32)], 3);
      vlCDEF = _mm_insert_epi32(vlCDEF, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidxEF >> 32)], 3);
      vlGHIJ = _mm_insert_epi32(vlGHIJ, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidxIJ >> 32)], 3);
      vlKLMN = _mm_insert_epi32(vlKLMN, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidxMN >> 32)], 3);
      vlOPQR = _mm_insert_epi32(vlOPQR, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidxQR >> 32)], 3);
      vlSTUV = _mm_insert_epi32(vlSTUV, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidxUV >> 32)], 3);
    #else
      const uint32_t vidx0 = (uint32_t) _mm_cvtsi128_si32(vidx0123);
      const uint32_t vidx4 = (uint32_t) _mm_cvtsi128_si32(vidx4567);
      const uint32_t vidx8 = (uint32_t) _mm_cvtsi128_si32(vidx89AB);
      const uint32_t vidxC = (uint32_t) _mm_cvtsi128_si32(vidxCDEF);
      const uint32_t vidxG = (uint32_t) _mm_cvtsi128_si32(vidxGHIJ);
      const uint32_t vidxK = (uint32_t) _mm_cvtsi128_si32(vidxKLMN);
      const uint32_t vidxO = (uint32_t) _mm_cvtsi128_si32(vidxOPQR);
      const uint32_t vidxS = (uint32_t) _mm_cvtsi128_si32(vidxSTUV);

      __m128i vl0123 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx0]);
      __m128i vl4567 = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx4]);
      const uint32_t vidx1 = (uint32_t) _mm_extract_epi32(vidx0123, 1);
      const uint32_t vidx5 = (uint32_t) _mm_extract_epi32(vidx4567, 1);
      __m128i vl89AB = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx8]);
      __m128i vlCDEF = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxC]);
      const uint32_t vidx9 = (uint32_t) _mm_extract_epi32(vidx89AB, 1);
      const uint32_t vidxD = (uint32_t) _mm_extract_epi32(vidxCDEF, 1);
      __m128i vlGHIJ = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxG]);
      __m128i vlKLMN = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxK]);
      const uint32_t vidxH = (uint32_t) _mm_extract_epi32(vidxGHIJ, 1);
      const uint32_t vidxL = (uint32_t) _mm_extract_epi32(vidxKLMN, 1);
      __m128i vlOPQR = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxO]);
      __m128i vlSTUV = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxS]);
      const uint32_t vidxP = (uint32_t) _mm_extract_epi32(vidxOPQR, 1);
      const uint32_t vidxT = (uint32_t) _mm_extract_epi32(vidxSTUV, 1);

      vl0123 = _mm_insert_epi32(vl0123, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx1], 1);
      vl4567 = _mm_insert_epi32(vl4567, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx5], 1);
      const uint32_t vidx2 = (uint32_t) _mm_extract_epi32(vidx0123, 2);
      const uint32_t vidx6 = (uint32_t) _mm_extract_epi32(vidx4567, 2);
      vl89AB = _mm_insert_epi32(vl89AB, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx9], 1);
      vlCDEF = _mm_insert_epi32(vlCDEF, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxD], 1);
      const uint32_t vidxA = (uint32_t) _mm_extract_epi32(vidx89AB, 2);
      const uint32_t vidxE = (uint32_t) _mm_extract_epi32(vidxCDEF, 2);
      vlGHIJ = _mm_insert_epi32(vlGHIJ, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxH], 1);
      vlKLMN = _mm_insert_epi32(vlKLMN, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxL], 1);
      const uint32_t vidxI = (uint32_t) _mm_extract_epi32(vidxGHIJ, 2);
      const uint32_t vidxM = (uint32_t) _mm_extract_epi32(vidxKLMN, 2);
      vlOPQR = _mm_insert_epi32(vlOPQR, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxP], 1);
      vlSTUV = _mm_insert_epi32(vlSTUV, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxT], 1);
      const uint32_t vidxQ = (uint32_t) _mm_extract_epi32(vidxOPQR, 2);
      const uint32_t vidxU = (uint32_t) _mm_extract_epi32(vidxSTUV, 2);

      vl0123 = _mm_insert_epi32(vl0123, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx2], 2);
      vl4567 = _mm_insert_epi32(vl4567, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx6], 2);
      const uint32_t vidx3 = (uint32_t) _mm_extract_epi32(vidx0123, 3);
      const uint32_t vidx7 = (uint32_t) _mm_extract_epi32(vidx4567, 3);
      vl89AB = _mm_insert_epi32(vl89AB, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxA], 2);
      vlCDEF = _mm_insert_epi32(vlCDEF, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxE], 2);
      const uint32_t vidxB = (uint32_t) _mm_extract_epi32(vidx89AB, 3);
      const uint32_t vidxF = (uint32_t) _mm_extract_epi32(vidxCDEF, 3);
      vlGHIJ = _mm_insert_epi32(vlGHIJ, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxI], 2);
      vlKLMN = _mm_insert_epi32(vlKLMN, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxM], 2);
      const uint32_t vidxJ = (uint32_t) _mm_extract_epi32(vidxGHIJ, 3);
      const uint32_t vidxN = (uint32_t) _mm_extract_epi32(vidxKLMN, 3);
      vlOPQR = _mm_insert_epi32(vlOPQR, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxQ], 2);
      vlSTUV = _mm_insert_epi32(vlSTUV, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxU], 2);
      const uint32_t vidxR = (uint32_t) _mm_extract_epi32(vidxOPQR, 3);
      const uint32_t vidxV = (uint32_t) _mm_extract_epi32(vidxSTUV, 3);

      vl0123 = _mm_insert_epi32(vl0123, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx3], 3);
      vl4567 = _mm_insert_epi32(vl4567, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx7], 3);
      vl89AB = _mm_insert_epi32(vl89AB, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxB], 3);
      vlCDEF = _mm_insert_epi32(vlCDEF, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxF], 3);
      vlGHIJ = _mm_insert_epi32(vlGHIJ, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxJ], 3);
      vlKLMN = _mm_insert_epi32(vlKLMN, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxN], 3);
      vlOPQR = _mm_insert_epi32(vlOPQR, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxR], 3);
      vlSTUV = _mm_insert_epi32(vlSTUV, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidxV], 3);
    #endif

    const __m128 vs0123 = _mm_castsi128_ps(_mm_add_epi32(vl0123, ve0123));
    const __m128 vs4567 = _mm_castsi128_ps(_mm_add_epi32(vl4567, ve4567));
    const __m128 vs89AB = _mm_castsi128_ps(_mm_add_epi32(vl89AB, ve89AB));
    const __m128 vsCDEF = _mm_castsi128_ps(_mm_add_epi32(vlCDEF, veCDEF));
    const __m128 vsGHIJ = _mm_castsi128_ps(_mm_add_epi32(vlGHIJ, veGHIJ));
    const __m128 vsKLMN = _mm_castsi128_ps(_mm_add_epi32(vlKLMN, veKLMN));
    const __m128 vsOPQR = _mm_castsi128_ps(_mm_add_epi32(vlOPQR, veOPQR));
    const __m128 vsSTUV = _mm_castsi128_ps(_mm_add_epi32(vlSTUV, veSTUV));

    const __m256 vs01234567 = _mm256_insertf128_ps(_mm256_castps128_ps256(vs0123), vs4567, 1);
    vn01234567 = _mm256_sub_ps(vn01234567, vmagic_bias);
    const __m256 vs89ABCDEF = _mm256_insertf128_ps(_mm256_castps128_ps256(vs89AB), vsCDEF, 1);
    vn89ABCDEF = _mm256_sub_ps(vn89ABCDEF, vmagic_bias);
    const __m256 vsGHIJKLMN = _mm256_insertf128_ps(_mm256_castps128_ps256(vsGHIJ), vsKLMN, 1);
    vnGHIJKLMN = _mm256_sub_ps(vnGHIJKLMN, vmagic_bias);
    const __m256 vsOPQRSTUV = _mm256_insertf128_ps(_mm256_castps128_ps256(vsOPQR), vsSTUV, 1);
    vnOPQRSTUV = _mm256_sub_ps(vnOPQRSTUV, vmagic_bias);

    const __m256 vt01234567 = _mm256_fmadd_ps(vn01234567, vminus_ln2, vz01234567);
    const __m256 vt89ABCDEF = _mm256_fmadd_ps(vn89ABCDEF, vminus_ln2, vz89ABCDEF);
    const __m256 vtGHIJKLMN = _mm256_fmadd_ps(vnGHIJKLMN, vminus_ln2, vzGHIJKLMN);
    const __m256 vtOPQRSTUV = _mm256_fmadd_ps(vnOPQRSTUV, vminus_ln2, vzOPQRSTUV);

    __m256 vp01234567 = vc4;
    __m256 vp89ABCDEF = vc4;
    __m256 vpGHIJKLMN = vc4;
    __m256 vpOPQRSTUV = vc4;
    vp01234567 = _mm256_fmadd_ps(vp01234567, vt01234567, vc3);
    vp89ABCDEF = _mm256_fmadd_ps(vp89ABCDEF, vt89ABCDEF, vc3);
    vpGHIJKLMN = _mm256_fmadd_ps(vpGHIJKLMN, vtGHIJKLMN, vc3);
    vpOPQRSTUV = _mm256_fmadd_ps(vpOPQRSTUV, vtOPQRSTUV, vc3);
    vp01234567 = _mm256_fmadd_ps(vp01234567, vt01234567, vc2);
    vp89ABCDEF = _mm256_fmadd_ps(vp89ABCDEF, vt89ABCDEF, vc2);
    vpGHIJKLMN = _mm256_fmadd_ps(vpGHIJKLMN, vtGHIJKLMN, vc2);
    vpOPQRSTUV = _mm256_fmadd_ps(vpOPQRSTUV, vtOPQRSTUV, vc2);
    vp01234567 = _mm256_fmadd_ps(vp01234567, vt01234567, vtwo);
    vp89ABCDEF = _mm256_fmadd_ps(vp89ABCDEF, vt89ABCDEF, vtwo);
    vpGHIJKLMN = _mm256_fmadd_ps(vpGHIJKLMN, vtGHIJKLMN, vtwo);
    vpOPQRSTUV = _mm256_fmadd_ps(vpOPQRSTUV, vtOPQRSTUV, vtwo);

    const __m256 vts01234567 = _mm256_mul_ps(vt01234567, vs01234567);
    const __m256 vsmo01234567 = _mm256_add_ps(vs01234567, vminus_one);
    const __m256 vts89ABCDEF = _mm256_mul_ps(vt89ABCDEF, vs89ABCDEF);
    const __m256 vsmo89ABCDEF = _mm256_add_ps(vs89ABCDEF, vminus_one);
    const __m256 vtsGHIJKLMN = _mm256_mul_ps(vtGHIJKLMN, vsGHIJKLMN);
    const __m256 vsmoGHIJKLMN = _mm256_add_ps(vsGHIJKLMN, vminus_one);
    const __m256 vtsOPQRSTUV = _mm256_mul_ps(vtOPQRSTUV, vsOPQRSTUV);
    const __m256 vsmoOPQRSTUV = _mm256_add_ps(vsOPQRSTUV, vminus_one);
    const __m256 vemo01234567 = _mm256_fmadd_ps(vp01234567, vts01234567, vsmo01234567);
    const __m256 vemo89ABCDEF = _mm256_fmadd_ps(vp89ABCDEF, vts89ABCDEF, vsmo89ABCDEF);
    const __m256 vemoGHIJKLMN = _mm256_fmadd_ps(vpGHIJKLMN, vtsGHIJKLMN, vsmoGHIJKLMN);
    const __m256 vemoOPQRSTUV = _mm256_fmadd_ps(vpOPQRSTUV, vtsOPQRSTUV, vsmoOPQRSTUV);
    const __m256 vepo01234567 = _mm256_add_ps(vemo01234567, vtwo);
    const __m256 vepo89ABCDEF = _mm256_add_ps(vemo89ABCDEF, vtwo);
    const __m256 vepoGHIJKLMN = _mm256_add_ps(vemoGHIJKLMN, vtwo);
    const __m256 vepoOPQRSTUV = _mm256_add_ps(vemoOPQRSTUV, vtwo);

    __m256 vy01234567 = _mm256_div_ps(vemo01234567, vepo01234567);
    __m256 vy89ABCDEF = _mm256_div_ps(vemo89ABCDEF, vepo89ABCDEF);
    __m256 vyGHIJKLMN = _mm256_div_ps(vemoGHIJKLMN, vepoGHIJKLMN);
    __m256 vyOPQRSTUV = _mm256_div_ps(vemoOPQRSTUV, vepoOPQRSTUV);

    vy01234567 = _mm256_xor_ps(vy01234567, vinvsignx01234567);
    vy89ABCDEF = _mm256_xor_ps(vy89ABCDEF, vinvsignx89ABCDEF);
    vyGHIJKLMN = _mm256_xor_ps(vyGHIJKLMN, vinvsignxGHIJKLMN);
    vyOPQRSTUV = _mm256_xor_ps(vyOPQRSTUV, vinvsignxOPQRSTUV);

    _mm256_storeu_ps(output, vy01234567);
    _mm256_storeu_ps(output + 8, vy89ABCDEF);
    _mm256_storeu_ps(output + 16, vyGHIJKLMN);
    _mm256_storeu_ps(output + 24, vyOPQRSTUV);
    output += 32;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(input);
    input += 8;

    __m256 vz = _mm256_or_ps(vx, vsign_mask);
    const __m256 vinvsignx = _mm256_xor_ps(vx, vz);
    vz = _mm256_max_ps(vsat_cutoff, vz);

    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);

    const __m128 vn_hi = _mm256_extractf128_ps(vn, 1);
    const __m128i ve_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn)), 20);
    const __m128i ve_hi = _mm_slli_epi32(_mm_castps_si128(vn_hi), 20);

    const __m128i vidx_lo = _mm_and_si128(_mm_castps_si128(_mm256_castps256_ps128(vn)), vindex_mask);
    const __m128i vidx_hi = _mm_and_si128(_mm_castps_si128(vn_hi), vindex_mask);
    #if XNN_ARCH_X86_64
      const uint64_t vidx01 = (uint64_t) _mm_cvtsi128_si64(vidx_lo);
      const uint64_t vidx45 = (uint64_t) _mm_cvtsi128_si64(vidx_hi);
      __m128i vl_lo = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx01]);
      __m128i vl_hi = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx45]);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx01 >> 32)], 1);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx45 >> 32)], 1);
      const uint64_t vidx23 = (uint64_t) _mm_extract_epi64(vidx_lo, 1);
      const uint64_t vidx67 = (uint64_t) _mm_extract_epi64(vidx_hi, 1);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx23], 2);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx67], 2);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx23 >> 32)], 3);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx67 >> 32)], 3);
    #else
      const uint32_t vidx0 = (uint32_t) _mm_cvtsi128_si32(vidx_lo);
      const uint32_t vidx4 = (uint32_t) _mm_cvtsi128_si32(vidx_hi);
      __m128i vl_lo = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx0]);
      __m128i vl_hi = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx4]);
      const uint32_t vidx1 = (uint32_t) _mm_extract_epi32(vidx_lo, 1);
      const uint32_t vidx5 = (uint32_t) _mm_extract_epi32(vidx_hi, 1);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx1], 1);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx5], 1);
      const uint32_t vidx2 = (uint32_t) _mm_extract_epi32(vidx_lo, 2);
      const uint32_t vidx6 = (uint32_t) _mm_extract_epi32(vidx_hi, 2);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx2], 2);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx6], 2);
      const uint32_t vidx3 = (uint32_t) _mm_extract_epi32(vidx_lo, 3);
      const uint32_t vidx7 = (uint32_t) _mm_extract_epi32(vidx_hi, 3);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx3], 3);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx7], 3);
    #endif

    const __m128 vs_lo = _mm_castsi128_ps(_mm_add_epi32(vl_lo, ve_lo));
    const __m128 vs_hi = _mm_castsi128_ps(_mm_add_epi32(vl_hi, ve_hi));
    const __m256 vs = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo), vs_hi, 1);

    vn = _mm256_sub_ps(vn, vmagic_bias);

    const __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    __m256 vp = vc4;
    vp = _mm256_fmadd_ps(vp, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_fmadd_ps(vp, vt, vtwo);

    const __m256 vts = _mm256_mul_ps(vt, vs);
    const __m256 vsmo = _mm256_add_ps(vs, vminus_one);
    const __m256 vemo = _mm256_fmadd_ps(vp, vts, vsmo);
    const __m256 vepo = _mm256_add_ps(vemo, vtwo);

    __m256 vy = _mm256_div_ps(vemo, vepo);

    vy = _mm256_xor_ps(vy, vinvsignx);

    _mm256_storeu_ps(output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx_expm1minus_rr1_lut8_p4h3.mask_table[7] - batch));

    const __m256 vx = _mm256_maskload_ps(input, vmask);

    __m256 vz = _mm256_or_ps(vx, vsign_mask);
    const __m256 vinvsignx = _mm256_xor_ps(vx, vz);
    vz = _mm256_max_ps(vsat_cutoff, vz);

    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);

    const __m128 vn_hi = _mm256_extractf128_ps(vn, 1);
    const __m128i ve_lo = _mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn)), 20);
    const __m128i ve_hi = _mm_slli_epi32(_mm_castps_si128(vn_hi), 20);

    const __m128i vidx_lo = _mm_and_si128(_mm_castps_si128(_mm256_castps256_ps128(vn)), vindex_mask);
    const __m128i vidx_hi = _mm_and_si128(_mm_castps_si128(vn_hi), vindex_mask);
    #if XNN_ARCH_X86_64
      const uint64_t vidx01 = (uint64_t) _mm_cvtsi128_si64(vidx_lo);
      const uint64_t vidx45 = (uint64_t) _mm_cvtsi128_si64(vidx_hi);
      __m128i vl_lo = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx01]);
      __m128i vl_hi = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx45]);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx01 >> 32)], 1);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx45 >> 32)], 1);
      const uint64_t vidx23 = (uint64_t) _mm_extract_epi64(vidx_lo, 1);
      const uint64_t vidx67 = (uint64_t) _mm_extract_epi64(vidx_hi, 1);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx23], 2);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx67], 2);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx23 >> 32)], 3);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) (vidx67 >> 32)], 3);
    #else
      const uint32_t vidx0 = (uint32_t) _mm_cvtsi128_si32(vidx_lo);
      const uint32_t vidx4 = (uint32_t) _mm_cvtsi128_si32(vidx_hi);
      __m128i vl_lo = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx0]);
      __m128i vl_hi = _mm_cvtsi32_si128((int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx4]);
      const uint32_t vidx1 = (uint32_t) _mm_extract_epi32(vidx_lo, 1);
      const uint32_t vidx5 = (uint32_t) _mm_extract_epi32(vidx_hi, 1);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx1], 1);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx5], 1);
      const uint32_t vidx2 = (uint32_t) _mm_extract_epi32(vidx_lo, 2);
      const uint32_t vidx6 = (uint32_t) _mm_extract_epi32(vidx_hi, 2);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx2], 2);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx6], 2);
      const uint32_t vidx3 = (uint32_t) _mm_extract_epi32(vidx_lo, 3);
      const uint32_t vidx7 = (uint32_t) _mm_extract_epi32(vidx_hi, 3);
      vl_lo = _mm_insert_epi32(vl_lo, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx3], 3);
      vl_hi = _mm_insert_epi32(vl_hi, (int) xnn_table_exp2minus_k_over_8[(uint32_t) vidx7], 3);
    #endif

    const __m128 vs_lo = _mm_castsi128_ps(_mm_add_epi32(vl_lo, ve_lo));
    const __m128 vs_hi = _mm_castsi128_ps(_mm_add_epi32(vl_hi, ve_hi));
    const __m256 vs = _mm256_insertf128_ps(_mm256_castps128_ps256(vs_lo), vs_hi, 1);

    vn = _mm256_sub_ps(vn, vmagic_bias);

    const __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    __m256 vp = vc4;
    vp = _mm256_fmadd_ps(vp, vt, vc3);
    vp = _mm256_fmadd_ps(vp, vt, vc2);
    vp = _mm256_fmadd_ps(vp, vt, vtwo);

    const __m256 vts = _mm256_mul_ps(vt, vs);
    const __m256 vsmo = _mm256_add_ps(vs, vminus_one);
    const __m256 vemo = _mm256_fmadd_ps(vp, vts, vsmo);
    const __m256 vepo = _mm256_add_ps(vemo, vtwo);

    __m256 vy = _mm256_div_ps(vemo, vepo);

    vy = _mm256_xor_ps(vy, vinvsignx);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (batch & (4 * sizeof(float))) {
      _mm_storeu_ps(output, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      output += 4;
    }
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy_lo);
    }
  }
}
