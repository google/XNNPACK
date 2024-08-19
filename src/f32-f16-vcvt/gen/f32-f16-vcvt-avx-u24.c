// Auto-generated file. Do not edit!
//   Template: src/f32-f16-vcvt/sse.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <smmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/unaligned.h"
#include "xnnpack/vcvt.h"


void xnn_f32_f16_vcvt_ukernel__avx_u24(
    size_t batch,
    const float* input,
    void* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vnonsign_mask = _mm_castsi128_ps(_mm_set1_epi32(UINT32_C(0x7FFFFFFF)));
  const __m128i vexp_bias = _mm_set1_epi32(UINT32_C(0x07800000));
  const __m128 vscale_to_inf = _mm_set1_ps(0x1.0p+112f);
  const __m128i vexpw_max = _mm_set1_epi32(UINT32_C(0x7F800000));
  const __m128 vscale_to_zero = _mm_set1_ps(0x1.0p-110f);
  const __m128i vbias_min = _mm_set1_epi32(UINT32_C(0x40008000));  // 0x8000, 0x4000, 0x8000, 0x4000, ...
  const __m128i vmanth_mask = _mm_set1_epi32(UINT32_C(0x00000FFF));
  const __m128i vexph_mask = _mm_set1_epi32(UINT32_C(0x00007C00));
  const __m128i vnanh = _mm_set1_epi16(UINT16_C(0x7E00));

  XNN_FORCE_REALIZATION(vnonsign_mask);
  XNN_FORCE_REALIZATION(vexp_bias);
  XNN_FORCE_REALIZATION(vscale_to_inf);
  XNN_FORCE_REALIZATION(vexpw_max);
  XNN_FORCE_REALIZATION(vscale_to_zero);
  XNN_FORCE_REALIZATION(vbias_min);
  XNN_FORCE_REALIZATION(vmanth_mask);
  XNN_FORCE_REALIZATION(vexph_mask);
  XNN_FORCE_REALIZATION(vnanh);

  uint16_t* o = (uint16_t*) output;
  for (; batch >= 24 * sizeof(float); batch -= 24 * sizeof(float)) {
    const __m128 vx0 = _mm_loadu_ps(input);
    const __m128 vx1 = _mm_loadu_ps(input + 4);
    const __m128 vx2 = _mm_loadu_ps(input + 8);
    const __m128 vx3 = _mm_loadu_ps(input + 12);
    const __m128 vx4 = _mm_loadu_ps(input + 16);
    const __m128 vx5 = _mm_loadu_ps(input + 20);
    input += 24;

    const __m128 vabsx0 = _mm_and_ps(vx0, vnonsign_mask);
    const __m128 vabsx1 = _mm_and_ps(vx1, vnonsign_mask);
    const __m128 vabsx2 = _mm_and_ps(vx2, vnonsign_mask);
    const __m128 vabsx3 = _mm_and_ps(vx3, vnonsign_mask);
    const __m128 vabsx4 = _mm_and_ps(vx4, vnonsign_mask);
    const __m128 vabsx5 = _mm_and_ps(vx5, vnonsign_mask);

    const __m128 vsignx0 = _mm_xor_ps(vx0, vabsx0);
    const __m128 vsignx1 = _mm_xor_ps(vx1, vabsx1);
    const __m128 vsignx2 = _mm_xor_ps(vx2, vabsx2);
    const __m128 vsignx3 = _mm_xor_ps(vx3, vabsx3);
    const __m128 vsignx4 = _mm_xor_ps(vx4, vabsx4);
    const __m128 vsignx5 = _mm_xor_ps(vx5, vabsx5);

    __m128i vbias0 = _mm_add_epi32(_mm_castps_si128(vabsx0), vexp_bias);
    __m128i vbias1 = _mm_add_epi32(_mm_castps_si128(vabsx1), vexp_bias);
    __m128i vbias2 = _mm_add_epi32(_mm_castps_si128(vabsx2), vexp_bias);
    __m128i vbias3 = _mm_add_epi32(_mm_castps_si128(vabsx3), vexp_bias);
    __m128i vbias4 = _mm_add_epi32(_mm_castps_si128(vabsx4), vexp_bias);
    __m128i vbias5 = _mm_add_epi32(_mm_castps_si128(vabsx5), vexp_bias);

    __m128 vf0 = _mm_mul_ps(vabsx0, vscale_to_inf);
    __m128 vf1 = _mm_mul_ps(vabsx1, vscale_to_inf);
    __m128 vf2 = _mm_mul_ps(vabsx2, vscale_to_inf);
    __m128 vf3 = _mm_mul_ps(vabsx3, vscale_to_inf);
    __m128 vf4 = _mm_mul_ps(vabsx4, vscale_to_inf);
    __m128 vf5 = _mm_mul_ps(vabsx5, vscale_to_inf);

    const __m128i vnanmaskw0 = _mm_cmpgt_epi32(_mm_castps_si128(vabsx0), vexpw_max);
    const __m128i vnanmaskw1 = _mm_cmpgt_epi32(_mm_castps_si128(vabsx1), vexpw_max);
    const __m128i vnanmaskw2 = _mm_cmpgt_epi32(_mm_castps_si128(vabsx2), vexpw_max);
    const __m128i vnanmaskw3 = _mm_cmpgt_epi32(_mm_castps_si128(vabsx3), vexpw_max);
    const __m128i vnanmaskw4 = _mm_cmpgt_epi32(_mm_castps_si128(vabsx4), vexpw_max);
    const __m128i vnanmaskw5 = _mm_cmpgt_epi32(_mm_castps_si128(vabsx5), vexpw_max);

    vbias0 = _mm_and_si128(vbias0, vexpw_max);
    vbias1 = _mm_and_si128(vbias1, vexpw_max);
    vbias2 = _mm_and_si128(vbias2, vexpw_max);
    vbias3 = _mm_and_si128(vbias3, vexpw_max);
    vbias4 = _mm_and_si128(vbias4, vexpw_max);
    vbias5 = _mm_and_si128(vbias5, vexpw_max);

    vf0 = _mm_mul_ps(vf0, vscale_to_zero);
    vf1 = _mm_mul_ps(vf1, vscale_to_zero);
    vf2 = _mm_mul_ps(vf2, vscale_to_zero);
    vf3 = _mm_mul_ps(vf3, vscale_to_zero);
    vf4 = _mm_mul_ps(vf4, vscale_to_zero);
    vf5 = _mm_mul_ps(vf5, vscale_to_zero);

    const __m128i vnanmaskh0 = _mm_packs_epi32(vnanmaskw0, vnanmaskw1);
    const __m128i vnanmaskh1 = _mm_packs_epi32(vnanmaskw2, vnanmaskw3);
    const __m128i vnanmaskh2 = _mm_packs_epi32(vnanmaskw4, vnanmaskw5);

    const __m128i vsignh0 = _mm_packs_epi32(_mm_castps_si128(vsignx0), _mm_castps_si128(vsignx1));
    const __m128i vsignh1 = _mm_packs_epi32(_mm_castps_si128(vsignx2), _mm_castps_si128(vsignx3));
    const __m128i vsignh2 = _mm_packs_epi32(_mm_castps_si128(vsignx4), _mm_castps_si128(vsignx5));

    vbias0 = _mm_max_epi16(vbias0, vbias_min);
    vbias1 = _mm_max_epi16(vbias1, vbias_min);
    vbias2 = _mm_max_epi16(vbias2, vbias_min);
    vbias3 = _mm_max_epi16(vbias3, vbias_min);
    vbias4 = _mm_max_epi16(vbias4, vbias_min);
    vbias5 = _mm_max_epi16(vbias5, vbias_min);


    vf0 = _mm_add_ps(vf0, _mm_castsi128_ps(vbias0));
    vf1 = _mm_add_ps(vf1, _mm_castsi128_ps(vbias1));
    vf2 = _mm_add_ps(vf2, _mm_castsi128_ps(vbias2));
    vf3 = _mm_add_ps(vf3, _mm_castsi128_ps(vbias3));
    vf4 = _mm_add_ps(vf4, _mm_castsi128_ps(vbias4));
    vf5 = _mm_add_ps(vf5, _mm_castsi128_ps(vbias5));


    __m128i vexpw0 = _mm_srli_epi32(_mm_castps_si128(vf0), 13);
    __m128i vexpw1 = _mm_srli_epi32(_mm_castps_si128(vf1), 13);
    __m128i vexpw2 = _mm_srli_epi32(_mm_castps_si128(vf2), 13);
    __m128i vexpw3 = _mm_srli_epi32(_mm_castps_si128(vf3), 13);
    __m128i vexpw4 = _mm_srli_epi32(_mm_castps_si128(vf4), 13);
    __m128i vexpw5 = _mm_srli_epi32(_mm_castps_si128(vf5), 13);

    const __m128i vmantw0 = _mm_and_si128(_mm_castps_si128(vf0), vmanth_mask);
    const __m128i vmantw1 = _mm_and_si128(_mm_castps_si128(vf1), vmanth_mask);
    const __m128i vmantw2 = _mm_and_si128(_mm_castps_si128(vf2), vmanth_mask);
    const __m128i vmantw3 = _mm_and_si128(_mm_castps_si128(vf3), vmanth_mask);
    const __m128i vmantw4 = _mm_and_si128(_mm_castps_si128(vf4), vmanth_mask);
    const __m128i vmantw5 = _mm_and_si128(_mm_castps_si128(vf5), vmanth_mask);

    vexpw0 = _mm_and_si128(vexpw0, vexph_mask);
    vexpw1 = _mm_and_si128(vexpw1, vexph_mask);
    vexpw2 = _mm_and_si128(vexpw2, vexph_mask);
    vexpw3 = _mm_and_si128(vexpw3, vexph_mask);
    vexpw4 = _mm_and_si128(vexpw4, vexph_mask);
    vexpw5 = _mm_and_si128(vexpw5, vexph_mask);

    const __m128i vnonsignw0 = _mm_add_epi32(vmantw0, vexpw0);
    const __m128i vnonsignw1 = _mm_add_epi32(vmantw1, vexpw1);
    const __m128i vnonsignw2 = _mm_add_epi32(vmantw2, vexpw2);
    const __m128i vnonsignw3 = _mm_add_epi32(vmantw3, vexpw3);
    const __m128i vnonsignw4 = _mm_add_epi32(vmantw4, vexpw4);
    const __m128i vnonsignw5 = _mm_add_epi32(vmantw5, vexpw5);

    const __m128i vnonsignh0 = _mm_packs_epi32(vnonsignw0, vnonsignw1);
    const __m128i vnonsignh1 = _mm_packs_epi32(vnonsignw2, vnonsignw3);
    const __m128i vnonsignh2 = _mm_packs_epi32(vnonsignw4, vnonsignw5);

    const __m128i vabsh0 = _mm_blendv_epi8(vnonsignh0, vnanh, vnanmaskh0);
    const __m128i vabsh1 = _mm_blendv_epi8(vnonsignh1, vnanh, vnanmaskh1);
    const __m128i vabsh2 = _mm_blendv_epi8(vnonsignh2, vnanh, vnanmaskh2);

    const __m128i vh0 = _mm_or_si128(vabsh0, vsignh0);
    const __m128i vh1 = _mm_or_si128(vabsh1, vsignh1);
    const __m128i vh2 = _mm_or_si128(vabsh2, vsignh2);

    _mm_storeu_si128((__m128i*) o, vh0);
    _mm_storeu_si128((__m128i*) (o + 8), vh1);
    _mm_storeu_si128((__m128i*) (o + 16), vh2);
    o += 24;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 vx_lo = _mm_loadu_ps(input);
    const __m128 vx_hi = _mm_loadu_ps(input + 4);
    input += 8;

    const __m128 vabsx_lo = _mm_and_ps(vx_lo, vnonsign_mask);
    const __m128 vabsx_hi = _mm_and_ps(vx_hi, vnonsign_mask);

    const __m128 vsignx_lo = _mm_xor_ps(vx_lo, vabsx_lo);
    const __m128 vsignx_hi = _mm_xor_ps(vx_hi, vabsx_hi);
    __m128i vbias_lo = _mm_add_epi32(_mm_castps_si128(vabsx_lo), vexp_bias);
    __m128i vbias_hi = _mm_add_epi32(_mm_castps_si128(vabsx_hi), vexp_bias);
    __m128 vf_lo = _mm_mul_ps(vabsx_lo, vscale_to_inf);
    __m128 vf_hi = _mm_mul_ps(vabsx_hi, vscale_to_inf);
    const __m128i vnanmaskw_lo = _mm_cmpgt_epi32(_mm_castps_si128(vabsx_lo), vexpw_max);
    const __m128i vnanmaskw_hi = _mm_cmpgt_epi32(_mm_castps_si128(vabsx_hi), vexpw_max);

    vbias_lo = _mm_and_si128(vbias_lo, vexpw_max);
    vbias_hi = _mm_and_si128(vbias_hi, vexpw_max);
    vf_lo = _mm_mul_ps(vf_lo, vscale_to_zero);
    vf_hi = _mm_mul_ps(vf_hi, vscale_to_zero);
    const __m128i vnanmaskh = _mm_packs_epi32(vnanmaskw_lo, vnanmaskw_hi);
    const __m128i vsignh = _mm_packs_epi32(_mm_castps_si128(vsignx_lo), _mm_castps_si128(vsignx_hi));

    vbias_lo = _mm_max_epi16(vbias_lo, vbias_min);
    vbias_hi = _mm_max_epi16(vbias_hi, vbias_min);

    vf_lo = _mm_add_ps(vf_lo, _mm_castsi128_ps(vbias_lo));
    vf_hi = _mm_add_ps(vf_hi, _mm_castsi128_ps(vbias_hi));

    __m128i vexpw_lo = _mm_srli_epi32(_mm_castps_si128(vf_lo), 13);
    __m128i vexpw_hi = _mm_srli_epi32(_mm_castps_si128(vf_hi), 13);
    const __m128i vmantw_lo = _mm_and_si128(_mm_castps_si128(vf_lo), vmanth_mask);
    const __m128i vmantw_hi = _mm_and_si128(_mm_castps_si128(vf_hi), vmanth_mask);

    vexpw_lo = _mm_and_si128(vexpw_lo, vexph_mask);
    vexpw_hi = _mm_and_si128(vexpw_hi, vexph_mask);

    const __m128i vnonsignw_lo = _mm_add_epi32(vmantw_lo, vexpw_lo);
    const __m128i vnonsignw_hi = _mm_add_epi32(vmantw_hi, vexpw_hi);

    const __m128i vnonsignh = _mm_packs_epi32(vnonsignw_lo, vnonsignw_hi);

    const __m128i vabsh = _mm_blendv_epi8(vnonsignh, vnanh, vnanmaskh);

    const __m128i vh = _mm_or_si128(vabsh, vsignh);

    _mm_storeu_si128((__m128i*) o, vh);
    o += 8;
  }
  if XNN_UNPREDICTABLE(batch != 0) {
    const __m128 vx_lo = _mm_loadu_ps(input);
    const float* input_hi = (const float*) ((uintptr_t) input + (batch & (4 * sizeof(float))));
    const __m128 vx_hi = _mm_loadu_ps(input_hi);

    const __m128 vabsx_lo = _mm_and_ps(vx_lo, vnonsign_mask);
    const __m128 vabsx_hi = _mm_and_ps(vx_hi, vnonsign_mask);

    const __m128 vsignx_lo = _mm_xor_ps(vx_lo, vabsx_lo);
    const __m128 vsignx_hi = _mm_xor_ps(vx_hi, vabsx_hi);
    __m128i vbias_lo = _mm_add_epi32(_mm_castps_si128(vabsx_lo), vexp_bias);
    __m128i vbias_hi = _mm_add_epi32(_mm_castps_si128(vabsx_hi), vexp_bias);
    __m128 vf_lo = _mm_mul_ps(vabsx_lo, vscale_to_inf);
    __m128 vf_hi = _mm_mul_ps(vabsx_hi, vscale_to_inf);
    const __m128i vnanmaskw_lo = _mm_cmpgt_epi32(_mm_castps_si128(vabsx_lo), vexpw_max);
    const __m128i vnanmaskw_hi = _mm_cmpgt_epi32(_mm_castps_si128(vabsx_hi), vexpw_max);

    vbias_lo = _mm_and_si128(vbias_lo, vexpw_max);
    vbias_hi = _mm_and_si128(vbias_hi, vexpw_max);
    vf_lo = _mm_mul_ps(vf_lo, vscale_to_zero);
    vf_hi = _mm_mul_ps(vf_hi, vscale_to_zero);
    const __m128i vnanmaskh = _mm_packs_epi32(vnanmaskw_lo, vnanmaskw_hi);
    const __m128i vsignh = _mm_packs_epi32(_mm_castps_si128(vsignx_lo), _mm_castps_si128(vsignx_hi));

    vbias_lo = _mm_max_epi16(vbias_lo, vbias_min);
    vbias_hi = _mm_max_epi16(vbias_hi, vbias_min);

    vf_lo = _mm_add_ps(vf_lo, _mm_castsi128_ps(vbias_lo));
    vf_hi = _mm_add_ps(vf_hi, _mm_castsi128_ps(vbias_hi));

    __m128i vexpw_lo = _mm_srli_epi32(_mm_castps_si128(vf_lo), 13);
    __m128i vexpw_hi = _mm_srli_epi32(_mm_castps_si128(vf_hi), 13);
    const __m128i vmantw_lo = _mm_and_si128(_mm_castps_si128(vf_lo), vmanth_mask);
    const __m128i vmantw_hi = _mm_and_si128(_mm_castps_si128(vf_hi), vmanth_mask);

    vexpw_lo = _mm_and_si128(vexpw_lo, vexph_mask);
    vexpw_hi = _mm_and_si128(vexpw_hi, vexph_mask);

    const __m128i vnonsignw_lo = _mm_add_epi32(vmantw_lo, vexpw_lo);
    const __m128i vnonsignw_hi = _mm_add_epi32(vmantw_hi, vexpw_hi);

    const __m128i vnonsignh = _mm_packs_epi32(vnonsignw_lo, vnonsignw_hi);

    const __m128i vabsh = _mm_blendv_epi8(vnonsignh, vnanh, vnanmaskh);

    __m128i vh = _mm_or_si128(vabsh, vsignh);

    if (batch & (4 * sizeof(float))) {
      _mm_storel_epi64((__m128i*) o, vh);
      vh = _mm_unpackhi_epi64(vh, vh);
      o += 4;
    }
    if (batch & (2 * sizeof(float))) {
      unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(vh));
      vh = _mm_srli_epi64(vh, 32);
      o += 2;
    }
    if (batch & (1 * sizeof(float))) {
      *o = (uint16_t) _mm_extract_epi16(vh, 0);
    }
  }
}
