// Auto-generated file. Do not edit!
//   Template: src/f16-f32-vcvt/sse-int32.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/vcvt.h"


void xnn_f16_f32_vcvt_ukernel__sse2_int32_u32(
    size_t batch,
    const void* input,
    float* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vsign_mask = _mm_set1_epi32(UINT32_C(0x80000000));
  const __m128i vexp_offset = _mm_set1_epi32(UINT32_C(0x70000000));
  const __m128 vexp_scale = _mm_set1_ps(0x1.0p-112f);
  const __m128i vmagic_bias = _mm_set1_epi32(UINT32_C(0x3F000000));
  const __m128i vdenorm_cutoff = _mm_set1_epi32(UINT32_C(0x08000000));

  XNN_FORCE_REALIZATION(vsign_mask);
  XNN_FORCE_REALIZATION(vexp_offset);
  XNN_FORCE_REALIZATION(vexp_scale);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const uint16_t* i = (const uint16_t*) input;
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const __m128i vh0 = _mm_loadu_si128((const __m128i*) i);
    const __m128i vh1 = _mm_loadu_si128((const __m128i*) (i + 8));
    const __m128i vh2 = _mm_loadu_si128((const __m128i*) (i + 16));
    const __m128i vh3 = _mm_loadu_si128((const __m128i*) (i + 24));
    i += 32;

    const __m128i vw0 = _mm_unpacklo_epi16(_mm_setzero_si128(), vh0);
    const __m128i vw1 = _mm_unpackhi_epi16(_mm_setzero_si128(), vh0);
    const __m128i vw2 = _mm_unpacklo_epi16(_mm_setzero_si128(), vh1);
    const __m128i vw3 = _mm_unpackhi_epi16(_mm_setzero_si128(), vh1);
    const __m128i vw4 = _mm_unpacklo_epi16(_mm_setzero_si128(), vh2);
    const __m128i vw5 = _mm_unpackhi_epi16(_mm_setzero_si128(), vh2);
    const __m128i vw6 = _mm_unpacklo_epi16(_mm_setzero_si128(), vh3);
    const __m128i vw7 = _mm_unpackhi_epi16(_mm_setzero_si128(), vh3);

    const __m128i vsign0 = _mm_and_si128(vw0, vsign_mask);
    const __m128i vsign1 = _mm_and_si128(vw1, vsign_mask);
    const __m128i vsign2 = _mm_and_si128(vw2, vsign_mask);
    const __m128i vsign3 = _mm_and_si128(vw3, vsign_mask);
    const __m128i vsign4 = _mm_and_si128(vw4, vsign_mask);
    const __m128i vsign5 = _mm_and_si128(vw5, vsign_mask);
    const __m128i vsign6 = _mm_and_si128(vw6, vsign_mask);
    const __m128i vsign7 = _mm_and_si128(vw7, vsign_mask);

    const __m128i vnonsign0 = _mm_xor_si128(vw0, vsign0);
    const __m128i vnonsign1 = _mm_xor_si128(vw1, vsign1);
    const __m128i vnonsign2 = _mm_xor_si128(vw2, vsign2);
    const __m128i vnonsign3 = _mm_xor_si128(vw3, vsign3);
    const __m128i vnonsign4 = _mm_xor_si128(vw4, vsign4);
    const __m128i vnonsign5 = _mm_xor_si128(vw5, vsign5);
    const __m128i vnonsign6 = _mm_xor_si128(vw6, vsign6);
    const __m128i vnonsign7 = _mm_xor_si128(vw7, vsign7);

    const __m128i vnorm0 = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_add_epi32(_mm_srli_epi32(vnonsign0, 3), vexp_offset)), vexp_scale));
    const __m128i vnorm1 = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_add_epi32(_mm_srli_epi32(vnonsign1, 3), vexp_offset)), vexp_scale));
    const __m128i vnorm2 = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_add_epi32(_mm_srli_epi32(vnonsign2, 3), vexp_offset)), vexp_scale));
    const __m128i vnorm3 = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_add_epi32(_mm_srli_epi32(vnonsign3, 3), vexp_offset)), vexp_scale));
    const __m128i vnorm4 = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_add_epi32(_mm_srli_epi32(vnonsign4, 3), vexp_offset)), vexp_scale));
    const __m128i vnorm5 = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_add_epi32(_mm_srli_epi32(vnonsign5, 3), vexp_offset)), vexp_scale));
    const __m128i vnorm6 = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_add_epi32(_mm_srli_epi32(vnonsign6, 3), vexp_offset)), vexp_scale));
    const __m128i vnorm7 = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_add_epi32(_mm_srli_epi32(vnonsign7, 3), vexp_offset)), vexp_scale));

    const __m128i vdenorm0 = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_or_si128(_mm_srli_epi32(vnonsign0, 16), vmagic_bias)), _mm_castsi128_ps(vmagic_bias)));
    const __m128i vdenorm1 = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_or_si128(_mm_srli_epi32(vnonsign1, 16), vmagic_bias)), _mm_castsi128_ps(vmagic_bias)));
    const __m128i vdenorm2 = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_or_si128(_mm_srli_epi32(vnonsign2, 16), vmagic_bias)), _mm_castsi128_ps(vmagic_bias)));
    const __m128i vdenorm3 = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_or_si128(_mm_srli_epi32(vnonsign3, 16), vmagic_bias)), _mm_castsi128_ps(vmagic_bias)));
    const __m128i vdenorm4 = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_or_si128(_mm_srli_epi32(vnonsign4, 16), vmagic_bias)), _mm_castsi128_ps(vmagic_bias)));
    const __m128i vdenorm5 = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_or_si128(_mm_srli_epi32(vnonsign5, 16), vmagic_bias)), _mm_castsi128_ps(vmagic_bias)));
    const __m128i vdenorm6 = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_or_si128(_mm_srli_epi32(vnonsign6, 16), vmagic_bias)), _mm_castsi128_ps(vmagic_bias)));
    const __m128i vdenorm7 = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_or_si128(_mm_srli_epi32(vnonsign7, 16), vmagic_bias)), _mm_castsi128_ps(vmagic_bias)));

    const __m128i vmask0 = _mm_cmpgt_epi32(vnonsign0, vdenorm_cutoff);
    const __m128i vmask1 = _mm_cmpgt_epi32(vnonsign1, vdenorm_cutoff);
    const __m128i vmask2 = _mm_cmpgt_epi32(vnonsign2, vdenorm_cutoff);
    const __m128i vmask3 = _mm_cmpgt_epi32(vnonsign3, vdenorm_cutoff);
    const __m128i vmask4 = _mm_cmpgt_epi32(vnonsign4, vdenorm_cutoff);
    const __m128i vmask5 = _mm_cmpgt_epi32(vnonsign5, vdenorm_cutoff);
    const __m128i vmask6 = _mm_cmpgt_epi32(vnonsign6, vdenorm_cutoff);
    const __m128i vmask7 = _mm_cmpgt_epi32(vnonsign7, vdenorm_cutoff);

    const __m128i vf0 = _mm_or_si128(vsign0,
      _mm_or_si128(_mm_and_si128(vmask0, vnorm0), _mm_andnot_si128(vmask0, vdenorm0)));
    const __m128i vf1 = _mm_or_si128(vsign1,
      _mm_or_si128(_mm_and_si128(vmask1, vnorm1), _mm_andnot_si128(vmask1, vdenorm1)));
    const __m128i vf2 = _mm_or_si128(vsign2,
      _mm_or_si128(_mm_and_si128(vmask2, vnorm2), _mm_andnot_si128(vmask2, vdenorm2)));
    const __m128i vf3 = _mm_or_si128(vsign3,
      _mm_or_si128(_mm_and_si128(vmask3, vnorm3), _mm_andnot_si128(vmask3, vdenorm3)));
    const __m128i vf4 = _mm_or_si128(vsign4,
      _mm_or_si128(_mm_and_si128(vmask4, vnorm4), _mm_andnot_si128(vmask4, vdenorm4)));
    const __m128i vf5 = _mm_or_si128(vsign5,
      _mm_or_si128(_mm_and_si128(vmask5, vnorm5), _mm_andnot_si128(vmask5, vdenorm5)));
    const __m128i vf6 = _mm_or_si128(vsign6,
      _mm_or_si128(_mm_and_si128(vmask6, vnorm6), _mm_andnot_si128(vmask6, vdenorm6)));
    const __m128i vf7 = _mm_or_si128(vsign7,
      _mm_or_si128(_mm_and_si128(vmask7, vnorm7), _mm_andnot_si128(vmask7, vdenorm7)));

    _mm_storeu_ps(output, _mm_castsi128_ps(vf0));
    _mm_storeu_ps(output + 4, _mm_castsi128_ps(vf1));
    _mm_storeu_ps(output + 8, _mm_castsi128_ps(vf2));
    _mm_storeu_ps(output + 12, _mm_castsi128_ps(vf3));
    _mm_storeu_ps(output + 16, _mm_castsi128_ps(vf4));
    _mm_storeu_ps(output + 20, _mm_castsi128_ps(vf5));
    _mm_storeu_ps(output + 24, _mm_castsi128_ps(vf6));
    _mm_storeu_ps(output + 28, _mm_castsi128_ps(vf7));
    output += 32;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m128i vh = _mm_loadu_si128((const __m128i*) i);
    i += 8;

    const __m128i vw_lo = _mm_unpacklo_epi16(_mm_setzero_si128(), vh);
    const __m128i vw_hi = _mm_unpackhi_epi16(_mm_setzero_si128(), vh);

    const __m128i vsign_lo = _mm_and_si128(vw_lo, vsign_mask);
    const __m128i vsign_hi = _mm_and_si128(vw_hi, vsign_mask);

    const __m128i vnonsign_lo = _mm_xor_si128(vw_lo, vsign_lo);
    const __m128i vnonsign_hi = _mm_xor_si128(vw_hi, vsign_hi);

    const __m128i vnorm_lo = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_add_epi32(_mm_srli_epi32(vnonsign_lo, 3), vexp_offset)), vexp_scale));
    const __m128i vnorm_hi = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_add_epi32(_mm_srli_epi32(vnonsign_hi, 3), vexp_offset)), vexp_scale));

    const __m128i vdenorm_lo = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_or_si128(_mm_srli_epi32(vnonsign_lo, 16), vmagic_bias)), _mm_castsi128_ps(vmagic_bias)));
    const __m128i vdenorm_hi = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_or_si128(_mm_srli_epi32(vnonsign_hi, 16), vmagic_bias)), _mm_castsi128_ps(vmagic_bias)));

    const __m128i vmask_lo = _mm_cmpgt_epi32(vnonsign_lo, vdenorm_cutoff);
    const __m128i vf_lo = _mm_or_si128(vsign_lo,
      _mm_or_si128(_mm_and_si128(vmask_lo, vnorm_lo), _mm_andnot_si128(vmask_lo, vdenorm_lo)));

    const __m128i vmask_hi = _mm_cmpgt_epi32(vnonsign_hi, vdenorm_cutoff);
    const __m128i vf_hi = _mm_or_si128(vsign_hi,
      _mm_or_si128(_mm_and_si128(vmask_hi, vnorm_hi), _mm_andnot_si128(vmask_hi, vdenorm_hi)));

    _mm_storeu_ps(output, _mm_castsi128_ps(vf_lo));
    _mm_storeu_ps(output + 4, _mm_castsi128_ps(vf_hi));
    output += 8;
  }
  if XNN_UNPREDICTABLE(batch != 0) {
    const __m128i vh = _mm_loadu_si128((const __m128i*) i);

    const __m128i vw_lo = _mm_unpacklo_epi16(_mm_setzero_si128(), vh);
    const __m128i vw_hi = _mm_unpackhi_epi16(_mm_setzero_si128(), vh);

    const __m128i vsign_lo = _mm_and_si128(vw_lo, vsign_mask);
    const __m128i vsign_hi = _mm_and_si128(vw_hi, vsign_mask);

    const __m128i vnonsign_lo = _mm_xor_si128(vw_lo, vsign_lo);
    const __m128i vnonsign_hi = _mm_xor_si128(vw_hi, vsign_hi);

    const __m128i vnorm_lo = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_add_epi32(_mm_srli_epi32(vnonsign_lo, 3), vexp_offset)), vexp_scale));
    const __m128i vnorm_hi = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_add_epi32(_mm_srli_epi32(vnonsign_hi, 3), vexp_offset)), vexp_scale));

    const __m128i vdenorm_lo = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_or_si128(_mm_srli_epi32(vnonsign_lo, 16), vmagic_bias)), _mm_castsi128_ps(vmagic_bias)));
    const __m128i vdenorm_hi = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_or_si128(_mm_srli_epi32(vnonsign_hi, 16), vmagic_bias)), _mm_castsi128_ps(vmagic_bias)));

    const __m128i vmask_lo = _mm_cmpgt_epi32(vnonsign_lo, vdenorm_cutoff);
    __m128i vf = _mm_or_si128(vsign_lo,
      _mm_or_si128(_mm_and_si128(vmask_lo, vnorm_lo), _mm_andnot_si128(vmask_lo, vdenorm_lo)));

    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storeu_ps(output, _mm_castsi128_ps(vf));
      output += 4;

      const __m128i vmask_hi = _mm_cmpgt_epi32(vnonsign_hi, vdenorm_cutoff);
      vf = _mm_or_si128(vsign_hi,
        _mm_or_si128(_mm_and_si128(vmask_hi, vnorm_hi), _mm_andnot_si128(vmask_hi, vdenorm_hi)));
    }
    if (batch & (2 * sizeof(uint16_t))) {
      _mm_storel_pi((__m64*) output, _mm_castsi128_ps(vf));
      output += 2;

      vf = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(vf), _mm_castsi128_ps(vf)));
    }
    if (batch & (1 * sizeof(uint16_t))) {
      _mm_store_ss(output, _mm_castsi128_ps(vf));
    }
  }
}
