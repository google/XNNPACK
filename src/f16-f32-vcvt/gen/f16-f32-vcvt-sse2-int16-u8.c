// Auto-generated file. Do not edit!
//   Template: src/f16-f32-vcvt/sse-int16.c.in
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


void xnn_f16_f32_vcvt_ukernel__sse2_int16_u8(
    size_t batch,
    const void* input,
    float* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vsign_mask = _mm_set1_epi16(UINT16_C(0x8000));
  const __m128i vexp_offset = _mm_set1_epi16(UINT16_C(0x7000));
  const __m128 vexp_scale = _mm_set1_ps(0x1.0p-112f);
  const __m128i vmagic_mask = _mm_set1_epi16(UINT16_C(0x3F00));
  const __m128 vmagic_bias = _mm_set1_ps(0.5f);
  const __m128i vdenorm_cutoff = _mm_set1_epi16(UINT16_C(0x0400));

  XNN_FORCE_REALIZATION(vsign_mask);
  XNN_FORCE_REALIZATION(vexp_offset);
  XNN_FORCE_REALIZATION(vexp_scale);
  XNN_FORCE_REALIZATION(vmagic_mask);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const uint16_t* i = (const uint16_t*) input;
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m128i vh = _mm_loadu_si128((const __m128i*) i);
    i += 8;

    const __m128i vsign = _mm_and_si128(vh, vsign_mask);

    const __m128i vnonsign = _mm_xor_si128(vh, vsign);

    const __m128i vprenorm_lo = _mm_slli_epi16(vnonsign, 13);
    const __m128i vprenorm_hi = _mm_add_epi16(_mm_srli_epi16(vnonsign, 3), vexp_offset);

    const __m128i vnorm_lo = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vprenorm_lo, vprenorm_hi)), vexp_scale));
    const __m128i vnorm_hi = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vprenorm_lo, vprenorm_hi)), vexp_scale));

    const __m128i vdenorm_lo = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vnonsign, vmagic_mask)), vmagic_bias));
    const __m128i vdenorm_hi = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vnonsign, vmagic_mask)), vmagic_bias));

    const __m128i vmask = _mm_cmpgt_epi16(vnonsign, vdenorm_cutoff);

    const __m128i vxmask_lo = _mm_unpacklo_epi16(vmask, vmask);
    const __m128i vf_lo = _mm_or_si128(_mm_unpacklo_epi16(_mm_setzero_si128(), vsign),
      _mm_or_si128(_mm_and_si128(vxmask_lo, vnorm_lo), _mm_andnot_si128(vxmask_lo, vdenorm_lo)));

    const __m128i vxmask_hi = _mm_unpackhi_epi16(vmask, vmask);
    const __m128i vf_hi = _mm_or_si128(_mm_unpackhi_epi16(_mm_setzero_si128(), vsign),
      _mm_or_si128(_mm_and_si128(vxmask_hi, vnorm_hi), _mm_andnot_si128(vxmask_hi, vdenorm_hi)));

    _mm_storeu_ps(output, _mm_castsi128_ps(vf_lo));
    _mm_storeu_ps(output + 4, _mm_castsi128_ps(vf_hi));
    output += 8;
  }
  if XNN_UNPREDICTABLE(batch != 0) {
    const __m128i vh = _mm_loadu_si128((const __m128i*) i);

    const __m128i vsign = _mm_and_si128(vh, vsign_mask);

    const __m128i vnonsign = _mm_xor_si128(vh, vsign);

    const __m128i vprenorm_lo = _mm_slli_epi16(vnonsign, 13);
    const __m128i vprenorm_hi = _mm_add_epi16(_mm_srli_epi16(vnonsign, 3), vexp_offset);

    const __m128i vnorm_lo = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vprenorm_lo, vprenorm_hi)), vexp_scale));
    const __m128i vnorm_hi = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vprenorm_lo, vprenorm_hi)), vexp_scale));

    const __m128i vdenorm_lo = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vnonsign, vmagic_mask)), vmagic_bias));
    const __m128i vdenorm_hi = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vnonsign, vmagic_mask)), vmagic_bias));

    const __m128i vmask = _mm_cmpgt_epi16(vnonsign, vdenorm_cutoff);

    const __m128i vxmask_lo = _mm_unpacklo_epi16(vmask, vmask);
    __m128i vf = _mm_or_si128(_mm_unpacklo_epi16(_mm_setzero_si128(), vsign),
      _mm_or_si128(_mm_and_si128(vxmask_lo, vnorm_lo), _mm_andnot_si128(vxmask_lo, vdenorm_lo)));

    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storeu_ps(output, _mm_castsi128_ps(vf));
      output += 4;

      const __m128i vxmask_hi = _mm_unpackhi_epi16(vmask, vmask);
      vf = _mm_or_si128(_mm_unpackhi_epi16(_mm_setzero_si128(), vsign),
        _mm_or_si128(_mm_and_si128(vxmask_hi, vnorm_hi), _mm_andnot_si128(vxmask_hi, vdenorm_hi)));
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
