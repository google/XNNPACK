// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Generator: tools/update-microkernels.py -a

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/dwconv.h"
#include "xnnpack/gavgpool.h"
#include "xnnpack/gemm.h"
#include "xnnpack/ibilinear.h"
#include "xnnpack/igemm.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/maxpool.h"
#include "xnnpack/microparams.h"
#include "xnnpack/prelu.h"
#include "xnnpack/simd/s32-sse41.h"
#include "xnnpack/unaligned.h"
#include "xnnpack/vbinary.h"
#include "xnnpack/vcvt.h"
#include "xnnpack/vlrelu.h"
#include "xnnpack/vunary.h"


void xnn_f16_f32_vcvt_ukernel__sse41_int16_u16(
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
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const __m128i vh0 = _mm_loadu_si128((const __m128i*) i);
    const __m128i vh1 = _mm_loadu_si128((const __m128i*) (i + 8));
    i += 16;

    const __m128i vsign0 = _mm_and_si128(vh0, vsign_mask);
    const __m128i vsign1 = _mm_and_si128(vh1, vsign_mask);

    const __m128i vnonsign0 = _mm_xor_si128(vh0, vsign0);
    const __m128i vnonsign1 = _mm_xor_si128(vh1, vsign1);

    const __m128i vprenorm0 = _mm_slli_epi16(vnonsign0, 13);
    const __m128i vprenorm1 = _mm_add_epi16(_mm_srli_epi16(vnonsign0, 3), vexp_offset);
    const __m128i vprenorm2 = _mm_slli_epi16(vnonsign1, 13);
    const __m128i vprenorm3 = _mm_add_epi16(_mm_srli_epi16(vnonsign1, 3), vexp_offset);

    const __m128i vnorm0 = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vprenorm0, vprenorm1)), vexp_scale));
    const __m128i vnorm1 = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vprenorm0, vprenorm1)), vexp_scale));
    const __m128i vnorm2 = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vprenorm2, vprenorm3)), vexp_scale));
    const __m128i vnorm3 = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vprenorm2, vprenorm3)), vexp_scale));

    const __m128i vdenorm0 = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vnonsign0, vmagic_mask)), vmagic_bias));
    const __m128i vdenorm1 = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vnonsign0, vmagic_mask)), vmagic_bias));
    const __m128i vdenorm2 = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpacklo_epi16(vnonsign1, vmagic_mask)), vmagic_bias));
    const __m128i vdenorm3 = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_unpackhi_epi16(vnonsign1, vmagic_mask)), vmagic_bias));

    const __m128i vmask0 = _mm_cmpgt_epi16(vnonsign0, vdenorm_cutoff);
    const __m128i vmask1 = _mm_cmpgt_epi16(vnonsign1, vdenorm_cutoff);

    const __m128i vf0 = _mm_or_si128(_mm_unpacklo_epi16(_mm_setzero_si128(), vsign0),
      _mm_blendv_epi8(vdenorm0, vnorm0, _mm_cvtepi16_epi32(vmask0)));
    const __m128i vf1 = _mm_or_si128(_mm_unpackhi_epi16(_mm_setzero_si128(), vsign0),
      _mm_blendv_epi8(vdenorm1, vnorm1, _mm_unpackhi_epi16(vmask0, vmask0)));
    const __m128i vf2 = _mm_or_si128(_mm_unpacklo_epi16(_mm_setzero_si128(), vsign1),
      _mm_blendv_epi8(vdenorm2, vnorm2, _mm_cvtepi16_epi32(vmask1)));
    const __m128i vf3 = _mm_or_si128(_mm_unpackhi_epi16(_mm_setzero_si128(), vsign1),
      _mm_blendv_epi8(vdenorm3, vnorm3, _mm_unpackhi_epi16(vmask1, vmask1)));

    _mm_storeu_ps(output, _mm_castsi128_ps(vf0));
    _mm_storeu_ps(output + 4, _mm_castsi128_ps(vf1));
    _mm_storeu_ps(output + 8, _mm_castsi128_ps(vf2));
    _mm_storeu_ps(output + 12, _mm_castsi128_ps(vf3));
    output += 16;
  }
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

    const __m128i vf_lo = _mm_or_si128(_mm_unpacklo_epi16(_mm_setzero_si128(), vsign),
      _mm_blendv_epi8(vdenorm_lo, vnorm_lo, _mm_cvtepi16_epi32(vmask)));

    const __m128i vf_hi = _mm_or_si128(_mm_unpackhi_epi16(_mm_setzero_si128(), vsign),
      _mm_blendv_epi8(vdenorm_hi, vnorm_hi, _mm_unpackhi_epi16(vmask, vmask)));

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

    __m128i vf = _mm_or_si128(_mm_unpacklo_epi16(_mm_setzero_si128(), vsign),
      _mm_blendv_epi8(vdenorm_lo, vnorm_lo, _mm_cvtepi16_epi32(vmask)));

    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storeu_ps(output, _mm_castsi128_ps(vf));
      output += 4;

      vf = _mm_or_si128(_mm_unpackhi_epi16(_mm_setzero_si128(), vsign),
        _mm_blendv_epi8(vdenorm_hi, vnorm_hi, _mm_unpackhi_epi16(vmask, vmask)));
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

void xnn_f32_f16_vcvt_ukernel__sse41_u8(
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

void xnn_f32_prelu_ukernel__sse41_2x8(
    size_t rows,
    size_t channels,
    const float* restrict input,
    size_t input_stride,
    const float* restrict weights,
    float* restrict output,
    size_t output_stride) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const float* i0 = input;
  float* o0 = output;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const float* w = weights;
    size_t c = channels;
    for (; c >= 8 * sizeof(float); c -= 8 * sizeof(float)) {
      const __m128 vw0123 = _mm_load_ps(w);
      const __m128 vw4567 = _mm_load_ps(w + 4);
      w += 8;

      const __m128 vi0x0123 = _mm_loadu_ps(i0);
      const __m128 vi0x4567 = _mm_loadu_ps(i0 + 4);
      i0 += 8;
      const __m128 vi1x0123 = _mm_loadu_ps(i1);
      const __m128 vi1x4567 = _mm_loadu_ps(i1 + 4);
      i1 += 8;

      const __m128 vprod0x0123 = _mm_mul_ps(vi0x0123, vw0123);
      const __m128 vprod0x4567 = _mm_mul_ps(vi0x4567, vw4567);
      const __m128 vprod1x0123 = _mm_mul_ps(vi1x0123, vw0123);
      const __m128 vprod1x4567 = _mm_mul_ps(vi1x4567, vw4567);

      const __m128 vacc0x0123 = _mm_blendv_ps(vi0x0123, vprod0x0123, vi0x0123);
      const __m128 vacc0x4567 = _mm_blendv_ps(vi0x4567, vprod0x4567, vi0x4567);
      const __m128 vacc1x0123 = _mm_blendv_ps(vi1x0123, vprod1x0123, vi1x0123);
      const __m128 vacc1x4567 = _mm_blendv_ps(vi1x4567, vprod1x4567, vi1x4567);

      _mm_storeu_ps(o0, vacc0x0123);
      _mm_storeu_ps(o0 + 4, vacc0x4567);
      o0 += 8;
      _mm_storeu_ps(o1, vacc1x0123);
      _mm_storeu_ps(o1 + 4, vacc1x4567);
      o1 += 8;
    }
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const __m128 vw0123 = _mm_load_ps(w);
      w += 4;

      const __m128 vi0x0123 = _mm_loadu_ps(i0);
      i0 += 4;
      const __m128 vi1x0123 = _mm_loadu_ps(i1);
      i1 += 4;

      const __m128 vprod0x0123 = _mm_mul_ps(vi0x0123, vw0123);
      const __m128 vprod1x0123 = _mm_mul_ps(vi1x0123, vw0123);

      __m128 vacc0x0123 = _mm_blendv_ps(vi0x0123, vprod0x0123, vi0x0123);
      __m128 vacc1x0123 = _mm_blendv_ps(vi1x0123, vprod1x0123, vi1x0123);

      _mm_storeu_ps(o0, vacc0x0123);
      o0 += 4;
      _mm_storeu_ps(o1, vacc1x0123);
      o1 += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      const __m128 vw0123 = _mm_load_ps(w);
      w = (const float*) ((uintptr_t) w + c);

      const __m128 vi0x0123 = _mm_loadu_ps(i0);
      i0 = (const float*) ((uintptr_t) i0 + c);
      const __m128 vi1x0123 = _mm_loadu_ps(i1);
      i1 = (const float*) ((uintptr_t) i1 + c);

      const __m128 vprod0x0123 = _mm_mul_ps(vi0x0123, vw0123);
      const __m128 vprod1x0123 = _mm_mul_ps(vi1x0123, vw0123);

      __m128 vacc0x0123 = _mm_blendv_ps(vi0x0123, vprod0x0123, vi0x0123);
      __m128 vacc1x0123 = _mm_blendv_ps(vi1x0123, vprod1x0123, vi1x0123);

      if (c & (2 * sizeof(float))) {
        _mm_storel_pi((__m64*) o0, vacc0x0123);
        _mm_storel_pi((__m64*) o1, vacc1x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);

        o0 += 2;
        o1 += 2;
      }
      if (c & (1 * sizeof(float))) {
        _mm_store_ss(o0, vacc0x0123);
        _mm_store_ss(o1, vacc1x0123);

        o0 += 1;
        o1 += 1;
      }
    }
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}

void xnn_f32_qc4w_gemm_minmax_ukernel_1x8__sse41_dup(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const __m128i vmagic_bias_c0 = _mm_load_si128((const __m128i*) params->sse.magic_bias_c0);
  const __m128i vmagic_bias_c1 = _mm_load_si128((const __m128i*) params->sse.magic_bias_c1);
  const __m128 vmagic_bias_plus_kernel_zero_point_c0 = _mm_load_ps(params->sse.magic_bias_plus_kernel_zero_point_c0);
  const __m128 vmagic_bias_plus_kernel_zero_point_c1 = _mm_load_ps(params->sse.magic_bias_plus_kernel_zero_point_c1);

  const __m128 vmax = _mm_set1_ps(params->sse.max);
  const __m128 vmin = _mm_set1_ps(params->sse.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m128 vacc0x0123 = _mm_loadu_ps((const float*) w + 0);
    __m128 vacc0x4567 = _mm_loadu_ps((const float*) w + 4);
    w = (const float*) w + 8;

    size_t k = kc;
    for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
      const __m128 va0 = _mm_loadu_ps(a0);
      a0 += 4;


      const __m128 va0c0000 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(0, 0, 0, 0)));

      const __m128i vbi0123c01 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((uint32_t) unaligned_load_u32((const uint8_t*) w + 0)));
      const __m128i vbi4567c01 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((uint32_t) unaligned_load_u32((const uint8_t*) w + 4)));
      const __m128i vbi0123c23 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((uint32_t) unaligned_load_u32((const uint8_t*) w + 8)));
      const __m128i vbi4567c23 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((uint32_t) unaligned_load_u32((const uint8_t*) w + 12)));
      const __m128 vbm0123c0 = _mm_castsi128_ps(_mm_or_si128(vbi0123c01, vmagic_bias_c0));
      const __m128 vbm0123c1 = _mm_castsi128_ps(_mm_or_si128(vbi0123c01, vmagic_bias_c1));
      const __m128 vbm0123c2 = _mm_castsi128_ps(_mm_or_si128(vbi0123c23, vmagic_bias_c0));
      const __m128 vbm0123c3 = _mm_castsi128_ps(_mm_or_si128(vbi0123c23, vmagic_bias_c1));
      const __m128 vbm4567c0 = _mm_castsi128_ps(_mm_or_si128(vbi4567c01, vmagic_bias_c0));
      const __m128 vbm4567c1 = _mm_castsi128_ps(_mm_or_si128(vbi4567c01, vmagic_bias_c1));
      const __m128 vbm4567c2 = _mm_castsi128_ps(_mm_or_si128(vbi4567c23, vmagic_bias_c0));
      const __m128 vbm4567c3 = _mm_castsi128_ps(_mm_or_si128(vbi4567c23, vmagic_bias_c1));
      const __m128 vb0123c0 = _mm_sub_ps(vbm0123c0, vmagic_bias_plus_kernel_zero_point_c0);
      const __m128 vb0123c1 = _mm_sub_ps(vbm0123c1, vmagic_bias_plus_kernel_zero_point_c1);
      const __m128 vb0123c2 = _mm_sub_ps(vbm0123c2, vmagic_bias_plus_kernel_zero_point_c0);
      const __m128 vb0123c3 = _mm_sub_ps(vbm0123c3, vmagic_bias_plus_kernel_zero_point_c1);
      const __m128 vb4567c0 = _mm_sub_ps(vbm4567c0, vmagic_bias_plus_kernel_zero_point_c0);
      const __m128 vb4567c1 = _mm_sub_ps(vbm4567c1, vmagic_bias_plus_kernel_zero_point_c1);
      const __m128 vb4567c2 = _mm_sub_ps(vbm4567c2, vmagic_bias_plus_kernel_zero_point_c0);
      const __m128 vb4567c3 = _mm_sub_ps(vbm4567c3, vmagic_bias_plus_kernel_zero_point_c1);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c0000, vb0123c0));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c0000, vb4567c0));

      const __m128 va0c1111 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(1, 1, 1, 1)));


      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c1111, vb0123c1));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c1111, vb4567c1));

      const __m128 va0c2222 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(2, 2, 2, 2)));


      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c2222, vb0123c2));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c2222, vb4567c2));

      const __m128 va0c3333 = _mm_shuffle_ps(va0, va0, _MM_SHUFFLE(3, 3, 3, 3));


      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c3333, vb0123c3));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c3333, vb4567c3));

      w = (const int8_t*) w + 16;
    }
    if XNN_UNLIKELY(k >= 2 * sizeof(float)) {
      const __m128 va0 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i *) a0));
      a0 += 2;

      const __m128 va0c0000 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(0, 0, 0, 0)));

      const __m128i vbi0123c01 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((uint32_t) unaligned_load_u32((const uint8_t*) w + 0)));
      const __m128i vbi4567c01 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((uint32_t) unaligned_load_u32((const uint8_t*) w + 4)));
      const __m128 vbm0123c0 = _mm_castsi128_ps(_mm_or_si128(vbi0123c01, vmagic_bias_c0));
      const __m128 vbm0123c1 = _mm_castsi128_ps(_mm_or_si128(vbi0123c01, vmagic_bias_c1));
      const __m128 vbm4567c0 = _mm_castsi128_ps(_mm_or_si128(vbi4567c01, vmagic_bias_c0));
      const __m128 vbm4567c1 = _mm_castsi128_ps(_mm_or_si128(vbi4567c01, vmagic_bias_c1));
      const __m128 vb0123c0 = _mm_sub_ps(vbm0123c0, vmagic_bias_plus_kernel_zero_point_c0);
      const __m128 vb0123c1 = _mm_sub_ps(vbm0123c1, vmagic_bias_plus_kernel_zero_point_c1);
      const __m128 vb4567c0 = _mm_sub_ps(vbm4567c0, vmagic_bias_plus_kernel_zero_point_c0);
      const __m128 vb4567c1 = _mm_sub_ps(vbm4567c1, vmagic_bias_plus_kernel_zero_point_c1);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c0000, vb0123c0));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c0000, vb4567c0));

      const __m128 va0c1111 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(1, 1, 1, 1)));


      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c1111, vb0123c1));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c1111, vb4567c1));

      w = (const int8_t*) w + 8;
      k -= 2 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      const __m128 va0 = _mm_load1_ps(a0);
      a0 += 1;

      const __m128i vbi0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((uint32_t) unaligned_load_u32((const uint8_t*) w + 0)));
      const __m128i vbi4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((uint32_t) unaligned_load_u32((const uint8_t*) w + 4)));
      const __m128 vbm0123 = _mm_castsi128_ps(_mm_or_si128(vbi0123, vmagic_bias_c0));
      const __m128 vbm4567 = _mm_castsi128_ps(_mm_or_si128(vbi4567, vmagic_bias_c0));
      const __m128 vb0123 = _mm_sub_ps(vbm0123, vmagic_bias_plus_kernel_zero_point_c0);
      const __m128 vb4567 = _mm_sub_ps(vbm4567, vmagic_bias_plus_kernel_zero_point_c0);
      w = (const int8_t*) w + 8;

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0, vb0123));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0, vb4567));

      k -= sizeof(float);
    }

    const __m128 vscale0123 = _mm_loadu_ps((const float*) w + 0);
    vacc0x0123 = _mm_mul_ps(vacc0x0123, vscale0123);
    const __m128 vscale4567 = _mm_loadu_ps((const float*) w + 4);
    vacc0x4567 = _mm_mul_ps(vacc0x4567, vscale4567);
    w = (const float*) w + 8;
    vacc0x0123 = _mm_min_ps(vacc0x0123, vmax);
    vacc0x4567 = _mm_min_ps(vacc0x4567, vmax);

    vacc0x0123 = _mm_max_ps(vacc0x0123, vmin);
    vacc0x4567 = _mm_max_ps(vacc0x4567, vmin);

    if XNN_LIKELY(nc >= 8) {
      _mm_storeu_ps(c0, vacc0x0123);
      _mm_storeu_ps(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_qc4w_gemm_minmax_ukernel_4x8__sse41_dup(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }
  const __m128i vmagic_bias_c0 = _mm_load_si128((const __m128i*) params->sse.magic_bias_c0);
  const __m128i vmagic_bias_c1 = _mm_load_si128((const __m128i*) params->sse.magic_bias_c1);
  const __m128 vmagic_bias_plus_kernel_zero_point_c0 = _mm_load_ps(params->sse.magic_bias_plus_kernel_zero_point_c0);
  const __m128 vmagic_bias_plus_kernel_zero_point_c1 = _mm_load_ps(params->sse.magic_bias_plus_kernel_zero_point_c1);

  const __m128 vmax = _mm_set1_ps(params->sse.max);
  const __m128 vmin = _mm_set1_ps(params->sse.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m128 vacc0x0123 = _mm_loadu_ps((const float*) w + 0);
    __m128 vacc0x4567 = _mm_loadu_ps((const float*) w + 4);
    __m128 vacc1x0123 = vacc0x0123;
    __m128 vacc1x4567 = vacc0x4567;
    __m128 vacc2x0123 = vacc0x0123;
    __m128 vacc2x4567 = vacc0x4567;
    __m128 vacc3x0123 = vacc0x0123;
    __m128 vacc3x4567 = vacc0x4567;
    w = (const float*) w + 8;

    size_t k = kc;
    for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
      const __m128 va0 = _mm_loadu_ps(a0);
      a0 += 4;
      const __m128 va1 = _mm_loadu_ps(a1);
      a1 += 4;
      const __m128 va2 = _mm_loadu_ps(a2);
      a2 += 4;
      const __m128 va3 = _mm_loadu_ps(a3);
      a3 += 4;


      const __m128 va0c0000 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(0, 0, 0, 0)));
      const __m128 va1c0000 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va1), _MM_SHUFFLE(0, 0, 0, 0)));
      const __m128 va2c0000 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va2), _MM_SHUFFLE(0, 0, 0, 0)));
      const __m128 va3c0000 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va3), _MM_SHUFFLE(0, 0, 0, 0)));

      const __m128i vbi0123c01 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((uint32_t) unaligned_load_u32((const uint8_t*) w + 0)));
      const __m128i vbi4567c01 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((uint32_t) unaligned_load_u32((const uint8_t*) w + 4)));
      const __m128i vbi0123c23 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((uint32_t) unaligned_load_u32((const uint8_t*) w + 8)));
      const __m128i vbi4567c23 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((uint32_t) unaligned_load_u32((const uint8_t*) w + 12)));
      const __m128 vbm0123c0 = _mm_castsi128_ps(_mm_or_si128(vbi0123c01, vmagic_bias_c0));
      const __m128 vbm0123c1 = _mm_castsi128_ps(_mm_or_si128(vbi0123c01, vmagic_bias_c1));
      const __m128 vbm0123c2 = _mm_castsi128_ps(_mm_or_si128(vbi0123c23, vmagic_bias_c0));
      const __m128 vbm0123c3 = _mm_castsi128_ps(_mm_or_si128(vbi0123c23, vmagic_bias_c1));
      const __m128 vbm4567c0 = _mm_castsi128_ps(_mm_or_si128(vbi4567c01, vmagic_bias_c0));
      const __m128 vbm4567c1 = _mm_castsi128_ps(_mm_or_si128(vbi4567c01, vmagic_bias_c1));
      const __m128 vbm4567c2 = _mm_castsi128_ps(_mm_or_si128(vbi4567c23, vmagic_bias_c0));
      const __m128 vbm4567c3 = _mm_castsi128_ps(_mm_or_si128(vbi4567c23, vmagic_bias_c1));
      const __m128 vb0123c0 = _mm_sub_ps(vbm0123c0, vmagic_bias_plus_kernel_zero_point_c0);
      const __m128 vb0123c1 = _mm_sub_ps(vbm0123c1, vmagic_bias_plus_kernel_zero_point_c1);
      const __m128 vb0123c2 = _mm_sub_ps(vbm0123c2, vmagic_bias_plus_kernel_zero_point_c0);
      const __m128 vb0123c3 = _mm_sub_ps(vbm0123c3, vmagic_bias_plus_kernel_zero_point_c1);
      const __m128 vb4567c0 = _mm_sub_ps(vbm4567c0, vmagic_bias_plus_kernel_zero_point_c0);
      const __m128 vb4567c1 = _mm_sub_ps(vbm4567c1, vmagic_bias_plus_kernel_zero_point_c1);
      const __m128 vb4567c2 = _mm_sub_ps(vbm4567c2, vmagic_bias_plus_kernel_zero_point_c0);
      const __m128 vb4567c3 = _mm_sub_ps(vbm4567c3, vmagic_bias_plus_kernel_zero_point_c1);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c0000, vb0123c0));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1c0000, vb0123c0));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2c0000, vb0123c0));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3c0000, vb0123c0));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c0000, vb4567c0));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1c0000, vb4567c0));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2c0000, vb4567c0));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3c0000, vb4567c0));

      const __m128 va0c1111 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(1, 1, 1, 1)));
      const __m128 va1c1111 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va1), _MM_SHUFFLE(1, 1, 1, 1)));
      const __m128 va2c1111 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va2), _MM_SHUFFLE(1, 1, 1, 1)));
      const __m128 va3c1111 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va3), _MM_SHUFFLE(1, 1, 1, 1)));


      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c1111, vb0123c1));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1c1111, vb0123c1));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2c1111, vb0123c1));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3c1111, vb0123c1));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c1111, vb4567c1));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1c1111, vb4567c1));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2c1111, vb4567c1));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3c1111, vb4567c1));

      const __m128 va0c2222 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(2, 2, 2, 2)));
      const __m128 va1c2222 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va1), _MM_SHUFFLE(2, 2, 2, 2)));
      const __m128 va2c2222 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va2), _MM_SHUFFLE(2, 2, 2, 2)));
      const __m128 va3c2222 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va3), _MM_SHUFFLE(2, 2, 2, 2)));


      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c2222, vb0123c2));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1c2222, vb0123c2));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2c2222, vb0123c2));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3c2222, vb0123c2));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c2222, vb4567c2));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1c2222, vb4567c2));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2c2222, vb4567c2));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3c2222, vb4567c2));

      const __m128 va0c3333 = _mm_shuffle_ps(va0, va0, _MM_SHUFFLE(3, 3, 3, 3));
      const __m128 va1c3333 = _mm_shuffle_ps(va1, va1, _MM_SHUFFLE(3, 3, 3, 3));
      const __m128 va2c3333 = _mm_shuffle_ps(va2, va2, _MM_SHUFFLE(3, 3, 3, 3));
      const __m128 va3c3333 = _mm_shuffle_ps(va3, va3, _MM_SHUFFLE(3, 3, 3, 3));


      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c3333, vb0123c3));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1c3333, vb0123c3));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2c3333, vb0123c3));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3c3333, vb0123c3));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c3333, vb4567c3));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1c3333, vb4567c3));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2c3333, vb4567c3));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3c3333, vb4567c3));

      w = (const int8_t*) w + 16;
    }
    if XNN_UNLIKELY(k >= 2 * sizeof(float)) {
      const __m128 va0 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i *) a0));
      a0 += 2;
      const __m128 va1 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i *) a1));
      a1 += 2;
      const __m128 va2 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i *) a2));
      a2 += 2;
      const __m128 va3 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i *) a3));
      a3 += 2;

      const __m128 va0c0000 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(0, 0, 0, 0)));
      const __m128 va1c0000 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va1), _MM_SHUFFLE(0, 0, 0, 0)));
      const __m128 va2c0000 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va2), _MM_SHUFFLE(0, 0, 0, 0)));
      const __m128 va3c0000 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va3), _MM_SHUFFLE(0, 0, 0, 0)));

      const __m128i vbi0123c01 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((uint32_t) unaligned_load_u32((const uint8_t*) w + 0)));
      const __m128i vbi4567c01 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((uint32_t) unaligned_load_u32((const uint8_t*) w + 4)));
      const __m128 vbm0123c0 = _mm_castsi128_ps(_mm_or_si128(vbi0123c01, vmagic_bias_c0));
      const __m128 vbm0123c1 = _mm_castsi128_ps(_mm_or_si128(vbi0123c01, vmagic_bias_c1));
      const __m128 vbm4567c0 = _mm_castsi128_ps(_mm_or_si128(vbi4567c01, vmagic_bias_c0));
      const __m128 vbm4567c1 = _mm_castsi128_ps(_mm_or_si128(vbi4567c01, vmagic_bias_c1));
      const __m128 vb0123c0 = _mm_sub_ps(vbm0123c0, vmagic_bias_plus_kernel_zero_point_c0);
      const __m128 vb0123c1 = _mm_sub_ps(vbm0123c1, vmagic_bias_plus_kernel_zero_point_c1);
      const __m128 vb4567c0 = _mm_sub_ps(vbm4567c0, vmagic_bias_plus_kernel_zero_point_c0);
      const __m128 vb4567c1 = _mm_sub_ps(vbm4567c1, vmagic_bias_plus_kernel_zero_point_c1);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c0000, vb0123c0));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1c0000, vb0123c0));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2c0000, vb0123c0));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3c0000, vb0123c0));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c0000, vb4567c0));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1c0000, vb4567c0));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2c0000, vb4567c0));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3c0000, vb4567c0));

      const __m128 va0c1111 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(1, 1, 1, 1)));
      const __m128 va1c1111 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va1), _MM_SHUFFLE(1, 1, 1, 1)));
      const __m128 va2c1111 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va2), _MM_SHUFFLE(1, 1, 1, 1)));
      const __m128 va3c1111 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va3), _MM_SHUFFLE(1, 1, 1, 1)));


      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c1111, vb0123c1));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1c1111, vb0123c1));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2c1111, vb0123c1));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3c1111, vb0123c1));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c1111, vb4567c1));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1c1111, vb4567c1));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2c1111, vb4567c1));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3c1111, vb4567c1));

      w = (const int8_t*) w + 8;
      k -= 2 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      const __m128 va0 = _mm_load1_ps(a0);
      a0 += 1;
      const __m128 va1 = _mm_load1_ps(a1);
      a1 += 1;
      const __m128 va2 = _mm_load1_ps(a2);
      a2 += 1;
      const __m128 va3 = _mm_load1_ps(a3);
      a3 += 1;

      const __m128i vbi0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((uint32_t) unaligned_load_u32((const uint8_t*) w + 0)));
      const __m128i vbi4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((uint32_t) unaligned_load_u32((const uint8_t*) w + 4)));
      const __m128 vbm0123 = _mm_castsi128_ps(_mm_or_si128(vbi0123, vmagic_bias_c0));
      const __m128 vbm4567 = _mm_castsi128_ps(_mm_or_si128(vbi4567, vmagic_bias_c0));
      const __m128 vb0123 = _mm_sub_ps(vbm0123, vmagic_bias_plus_kernel_zero_point_c0);
      const __m128 vb4567 = _mm_sub_ps(vbm4567, vmagic_bias_plus_kernel_zero_point_c0);
      w = (const int8_t*) w + 8;

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0, vb0123));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1, vb0123));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2, vb0123));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3, vb0123));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0, vb4567));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1, vb4567));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2, vb4567));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3, vb4567));

      k -= sizeof(float);
    }

    const __m128 vscale0123 = _mm_loadu_ps((const float*) w + 0);
    vacc0x0123 = _mm_mul_ps(vacc0x0123, vscale0123);
    vacc1x0123 = _mm_mul_ps(vacc1x0123, vscale0123);
    vacc2x0123 = _mm_mul_ps(vacc2x0123, vscale0123);
    vacc3x0123 = _mm_mul_ps(vacc3x0123, vscale0123);
    const __m128 vscale4567 = _mm_loadu_ps((const float*) w + 4);
    vacc0x4567 = _mm_mul_ps(vacc0x4567, vscale4567);
    vacc1x4567 = _mm_mul_ps(vacc1x4567, vscale4567);
    vacc2x4567 = _mm_mul_ps(vacc2x4567, vscale4567);
    vacc3x4567 = _mm_mul_ps(vacc3x4567, vscale4567);
    w = (const float*) w + 8;
    vacc0x0123 = _mm_min_ps(vacc0x0123, vmax);
    vacc1x0123 = _mm_min_ps(vacc1x0123, vmax);
    vacc2x0123 = _mm_min_ps(vacc2x0123, vmax);
    vacc3x0123 = _mm_min_ps(vacc3x0123, vmax);
    vacc0x4567 = _mm_min_ps(vacc0x4567, vmax);
    vacc1x4567 = _mm_min_ps(vacc1x4567, vmax);
    vacc2x4567 = _mm_min_ps(vacc2x4567, vmax);
    vacc3x4567 = _mm_min_ps(vacc3x4567, vmax);

    vacc0x0123 = _mm_max_ps(vacc0x0123, vmin);
    vacc1x0123 = _mm_max_ps(vacc1x0123, vmin);
    vacc2x0123 = _mm_max_ps(vacc2x0123, vmin);
    vacc3x0123 = _mm_max_ps(vacc3x0123, vmin);
    vacc0x4567 = _mm_max_ps(vacc0x4567, vmin);
    vacc1x4567 = _mm_max_ps(vacc1x4567, vmin);
    vacc2x4567 = _mm_max_ps(vacc2x4567, vmin);
    vacc3x4567 = _mm_max_ps(vacc3x4567, vmin);

    if XNN_LIKELY(nc >= 8) {
      _mm_storeu_ps(c0, vacc0x0123);
      _mm_storeu_ps(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      _mm_storeu_ps(c1, vacc1x0123);
      _mm_storeu_ps(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm_storeu_ps(c2, vacc2x0123);
      _mm_storeu_ps(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm_storeu_ps(c3, vacc3x0123);
      _mm_storeu_ps(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);
        _mm_storeu_ps(c1, vacc1x0123);
        _mm_storeu_ps(c2, vacc2x0123);
        _mm_storeu_ps(c3, vacc3x0123);

        vacc0x0123 = vacc0x4567;
        vacc1x0123 = vacc1x4567;
        vacc2x0123 = vacc2x4567;
        vacc3x0123 = vacc3x4567;

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vacc0x0123);
        _mm_storel_pi((__m64*) c1, vacc1x0123);
        _mm_storel_pi((__m64*) c2, vacc2x0123);
        _mm_storel_pi((__m64*) c3, vacc3x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);
        vacc2x0123 = _mm_movehl_ps(vacc2x0123, vacc2x0123);
        vacc3x0123 = _mm_movehl_ps(vacc3x0123, vacc3x0123);

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vacc0x0123);
        _mm_store_ss(c1, vacc1x0123);
        _mm_store_ss(c2, vacc2x0123);
        _mm_store_ss(c3, vacc3x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_qc8w_gemm_minmax_ukernel_1x8__sse41_dup(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  const __m128 vmax = _mm_set1_ps(params->sse.max);
  const __m128 vmin = _mm_set1_ps(params->sse.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m128 vacc0x0123 = _mm_loadu_ps((const float*) w + 0);
    __m128 vacc0x4567 = _mm_loadu_ps((const float*) w + 4);
    w = (const float*) w + 8;

    size_t k = kc;
    for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
      const __m128 va0 = _mm_loadu_ps(a0);
      a0 += 4;


      const __m128 va0c0000 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(0, 0, 0, 0)));

      const __m128i vbi0123c0 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 0)));
      const __m128i vbi4567c0 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 4)));
      const __m128 vb0123c0 = _mm_cvtepi32_ps(vbi0123c0);
      const __m128 vb4567c0 = _mm_cvtepi32_ps(vbi4567c0);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c0000, vb0123c0));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c0000, vb4567c0));

      const __m128 va0c1111 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(1, 1, 1, 1)));

      const __m128i vbi0123c1 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 8)));
      const __m128i vbi4567c1 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 12)));
      const __m128 vb0123c1 = _mm_cvtepi32_ps(vbi0123c1);
      const __m128 vb4567c1 = _mm_cvtepi32_ps(vbi4567c1);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c1111, vb0123c1));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c1111, vb4567c1));

      const __m128 va0c2222 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(2, 2, 2, 2)));

      const __m128i vbi0123c2 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 16)));
      const __m128i vbi4567c2 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 20)));
      const __m128 vb0123c2 = _mm_cvtepi32_ps(vbi0123c2);
      const __m128 vb4567c2 = _mm_cvtepi32_ps(vbi4567c2);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c2222, vb0123c2));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c2222, vb4567c2));

      const __m128 va0c3333 = _mm_shuffle_ps(va0, va0, _MM_SHUFFLE(3, 3, 3, 3));

      const __m128i vbi0123c3 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 24)));
      const __m128i vbi4567c3 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 28)));
      const __m128 vb0123c3 = _mm_cvtepi32_ps(vbi0123c3);
      const __m128 vb4567c3 = _mm_cvtepi32_ps(vbi4567c3);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c3333, vb0123c3));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c3333, vb4567c3));

      w = (const int8_t*) w + 32;
    }
    if XNN_UNLIKELY(k >= 2 * sizeof(float)) {
      const __m128 va0 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i *) a0));
      a0 += 2;

      const __m128 va0c0000 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(0, 0, 0, 0)));

      const __m128i vbi0123c0 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 0)));
      const __m128i vbi4567c0 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 4)));
      const __m128 vb0123c0 = _mm_cvtepi32_ps(vbi0123c0);
      const __m128 vb4567c0 = _mm_cvtepi32_ps(vbi4567c0);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c0000, vb0123c0));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c0000, vb4567c0));

      const __m128 va0c1111 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(1, 1, 1, 1)));

      const __m128i vbi0123c1 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 8)));
      const __m128i vbi4567c1 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 12)));
      const __m128 vb0123c1 = _mm_cvtepi32_ps(vbi0123c1);
      const __m128 vb4567c1 = _mm_cvtepi32_ps(vbi4567c1);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c1111, vb0123c1));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c1111, vb4567c1));

      w = (const int8_t*) w + 16;
      k -= 2 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      const __m128 va0 = _mm_load1_ps(a0);
      a0 += 1;

      const __m128i vbi0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const void*) w)));
      const __m128i vbi4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 4)));
      const __m128 vb0123 = _mm_cvtepi32_ps(vbi0123);
      const __m128 vb4567 = _mm_cvtepi32_ps(vbi4567);
      w = (const int8_t*) w + 8;

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0, vb0123));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0, vb4567));

      k -= sizeof(float);
    }

    const __m128 vscale0123 = _mm_loadu_ps((const float*) w + 0);
    vacc0x0123 = _mm_mul_ps(vacc0x0123, vscale0123);
    const __m128 vscale4567 = _mm_loadu_ps((const float*) w + 4);
    vacc0x4567 = _mm_mul_ps(vacc0x4567, vscale4567);
    w = (const float*) w + 8;
    vacc0x0123 = _mm_min_ps(vacc0x0123, vmax);
    vacc0x4567 = _mm_min_ps(vacc0x4567, vmax);

    vacc0x0123 = _mm_max_ps(vacc0x0123, vmin);
    vacc0x4567 = _mm_max_ps(vacc0x4567, vmin);

    if XNN_LIKELY(nc >= 8) {
      _mm_storeu_ps(c0, vacc0x0123);
      _mm_storeu_ps(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_qc8w_gemm_minmax_ukernel_4x8__sse41_dup(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const __m128 vmax = _mm_set1_ps(params->sse.max);
  const __m128 vmin = _mm_set1_ps(params->sse.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m128 vacc0x0123 = _mm_loadu_ps((const float*) w + 0);
    __m128 vacc0x4567 = _mm_loadu_ps((const float*) w + 4);
    __m128 vacc1x0123 = vacc0x0123;
    __m128 vacc1x4567 = vacc0x4567;
    __m128 vacc2x0123 = vacc0x0123;
    __m128 vacc2x4567 = vacc0x4567;
    __m128 vacc3x0123 = vacc0x0123;
    __m128 vacc3x4567 = vacc0x4567;
    w = (const float*) w + 8;

    size_t k = kc;
    for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
      const __m128 va0 = _mm_loadu_ps(a0);
      a0 += 4;
      const __m128 va1 = _mm_loadu_ps(a1);
      a1 += 4;
      const __m128 va2 = _mm_loadu_ps(a2);
      a2 += 4;
      const __m128 va3 = _mm_loadu_ps(a3);
      a3 += 4;


      const __m128 va0c0000 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(0, 0, 0, 0)));
      const __m128 va1c0000 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va1), _MM_SHUFFLE(0, 0, 0, 0)));
      const __m128 va2c0000 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va2), _MM_SHUFFLE(0, 0, 0, 0)));
      const __m128 va3c0000 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va3), _MM_SHUFFLE(0, 0, 0, 0)));

      const __m128i vbi0123c0 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 0)));
      const __m128i vbi4567c0 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 4)));
      const __m128 vb0123c0 = _mm_cvtepi32_ps(vbi0123c0);
      const __m128 vb4567c0 = _mm_cvtepi32_ps(vbi4567c0);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c0000, vb0123c0));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1c0000, vb0123c0));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2c0000, vb0123c0));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3c0000, vb0123c0));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c0000, vb4567c0));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1c0000, vb4567c0));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2c0000, vb4567c0));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3c0000, vb4567c0));

      const __m128 va0c1111 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(1, 1, 1, 1)));
      const __m128 va1c1111 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va1), _MM_SHUFFLE(1, 1, 1, 1)));
      const __m128 va2c1111 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va2), _MM_SHUFFLE(1, 1, 1, 1)));
      const __m128 va3c1111 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va3), _MM_SHUFFLE(1, 1, 1, 1)));

      const __m128i vbi0123c1 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 8)));
      const __m128i vbi4567c1 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 12)));
      const __m128 vb0123c1 = _mm_cvtepi32_ps(vbi0123c1);
      const __m128 vb4567c1 = _mm_cvtepi32_ps(vbi4567c1);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c1111, vb0123c1));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1c1111, vb0123c1));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2c1111, vb0123c1));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3c1111, vb0123c1));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c1111, vb4567c1));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1c1111, vb4567c1));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2c1111, vb4567c1));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3c1111, vb4567c1));

      const __m128 va0c2222 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(2, 2, 2, 2)));
      const __m128 va1c2222 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va1), _MM_SHUFFLE(2, 2, 2, 2)));
      const __m128 va2c2222 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va2), _MM_SHUFFLE(2, 2, 2, 2)));
      const __m128 va3c2222 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va3), _MM_SHUFFLE(2, 2, 2, 2)));

      const __m128i vbi0123c2 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 16)));
      const __m128i vbi4567c2 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 20)));
      const __m128 vb0123c2 = _mm_cvtepi32_ps(vbi0123c2);
      const __m128 vb4567c2 = _mm_cvtepi32_ps(vbi4567c2);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c2222, vb0123c2));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1c2222, vb0123c2));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2c2222, vb0123c2));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3c2222, vb0123c2));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c2222, vb4567c2));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1c2222, vb4567c2));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2c2222, vb4567c2));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3c2222, vb4567c2));

      const __m128 va0c3333 = _mm_shuffle_ps(va0, va0, _MM_SHUFFLE(3, 3, 3, 3));
      const __m128 va1c3333 = _mm_shuffle_ps(va1, va1, _MM_SHUFFLE(3, 3, 3, 3));
      const __m128 va2c3333 = _mm_shuffle_ps(va2, va2, _MM_SHUFFLE(3, 3, 3, 3));
      const __m128 va3c3333 = _mm_shuffle_ps(va3, va3, _MM_SHUFFLE(3, 3, 3, 3));

      const __m128i vbi0123c3 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 24)));
      const __m128i vbi4567c3 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 28)));
      const __m128 vb0123c3 = _mm_cvtepi32_ps(vbi0123c3);
      const __m128 vb4567c3 = _mm_cvtepi32_ps(vbi4567c3);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c3333, vb0123c3));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1c3333, vb0123c3));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2c3333, vb0123c3));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3c3333, vb0123c3));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c3333, vb4567c3));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1c3333, vb4567c3));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2c3333, vb4567c3));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3c3333, vb4567c3));

      w = (const int8_t*) w + 32;
    }
    if XNN_UNLIKELY(k >= 2 * sizeof(float)) {
      const __m128 va0 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i *) a0));
      a0 += 2;
      const __m128 va1 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i *) a1));
      a1 += 2;
      const __m128 va2 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i *) a2));
      a2 += 2;
      const __m128 va3 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i *) a3));
      a3 += 2;

      const __m128 va0c0000 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(0, 0, 0, 0)));
      const __m128 va1c0000 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va1), _MM_SHUFFLE(0, 0, 0, 0)));
      const __m128 va2c0000 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va2), _MM_SHUFFLE(0, 0, 0, 0)));
      const __m128 va3c0000 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va3), _MM_SHUFFLE(0, 0, 0, 0)));

      const __m128i vbi0123c0 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 0)));
      const __m128i vbi4567c0 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 4)));
      const __m128 vb0123c0 = _mm_cvtepi32_ps(vbi0123c0);
      const __m128 vb4567c0 = _mm_cvtepi32_ps(vbi4567c0);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c0000, vb0123c0));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1c0000, vb0123c0));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2c0000, vb0123c0));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3c0000, vb0123c0));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c0000, vb4567c0));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1c0000, vb4567c0));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2c0000, vb4567c0));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3c0000, vb4567c0));

      const __m128 va0c1111 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(1, 1, 1, 1)));
      const __m128 va1c1111 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va1), _MM_SHUFFLE(1, 1, 1, 1)));
      const __m128 va2c1111 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va2), _MM_SHUFFLE(1, 1, 1, 1)));
      const __m128 va3c1111 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va3), _MM_SHUFFLE(1, 1, 1, 1)));

      const __m128i vbi0123c1 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 8)));
      const __m128i vbi4567c1 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 12)));
      const __m128 vb0123c1 = _mm_cvtepi32_ps(vbi0123c1);
      const __m128 vb4567c1 = _mm_cvtepi32_ps(vbi4567c1);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c1111, vb0123c1));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1c1111, vb0123c1));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2c1111, vb0123c1));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3c1111, vb0123c1));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c1111, vb4567c1));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1c1111, vb4567c1));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2c1111, vb4567c1));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3c1111, vb4567c1));

      w = (const int8_t*) w + 16;
      k -= 2 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      const __m128 va0 = _mm_load1_ps(a0);
      a0 += 1;
      const __m128 va1 = _mm_load1_ps(a1);
      a1 += 1;
      const __m128 va2 = _mm_load1_ps(a2);
      a2 += 1;
      const __m128 va3 = _mm_load1_ps(a3);
      a3 += 1;

      const __m128i vbi0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const void*) w)));
      const __m128i vbi4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32((const int8_t*) w + 4)));
      const __m128 vb0123 = _mm_cvtepi32_ps(vbi0123);
      const __m128 vb4567 = _mm_cvtepi32_ps(vbi4567);
      w = (const int8_t*) w + 8;

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0, vb0123));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1, vb0123));
      vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2, vb0123));
      vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3, vb0123));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0, vb4567));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1, vb4567));
      vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2, vb4567));
      vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3, vb4567));

      k -= sizeof(float);
    }

    const __m128 vscale0123 = _mm_loadu_ps((const float*) w + 0);
    vacc0x0123 = _mm_mul_ps(vacc0x0123, vscale0123);
    vacc1x0123 = _mm_mul_ps(vacc1x0123, vscale0123);
    vacc2x0123 = _mm_mul_ps(vacc2x0123, vscale0123);
    vacc3x0123 = _mm_mul_ps(vacc3x0123, vscale0123);
    const __m128 vscale4567 = _mm_loadu_ps((const float*) w + 4);
    vacc0x4567 = _mm_mul_ps(vacc0x4567, vscale4567);
    vacc1x4567 = _mm_mul_ps(vacc1x4567, vscale4567);
    vacc2x4567 = _mm_mul_ps(vacc2x4567, vscale4567);
    vacc3x4567 = _mm_mul_ps(vacc3x4567, vscale4567);
    w = (const float*) w + 8;
    vacc0x0123 = _mm_min_ps(vacc0x0123, vmax);
    vacc1x0123 = _mm_min_ps(vacc1x0123, vmax);
    vacc2x0123 = _mm_min_ps(vacc2x0123, vmax);
    vacc3x0123 = _mm_min_ps(vacc3x0123, vmax);
    vacc0x4567 = _mm_min_ps(vacc0x4567, vmax);
    vacc1x4567 = _mm_min_ps(vacc1x4567, vmax);
    vacc2x4567 = _mm_min_ps(vacc2x4567, vmax);
    vacc3x4567 = _mm_min_ps(vacc3x4567, vmax);

    vacc0x0123 = _mm_max_ps(vacc0x0123, vmin);
    vacc1x0123 = _mm_max_ps(vacc1x0123, vmin);
    vacc2x0123 = _mm_max_ps(vacc2x0123, vmin);
    vacc3x0123 = _mm_max_ps(vacc3x0123, vmin);
    vacc0x4567 = _mm_max_ps(vacc0x4567, vmin);
    vacc1x4567 = _mm_max_ps(vacc1x4567, vmin);
    vacc2x4567 = _mm_max_ps(vacc2x4567, vmin);
    vacc3x4567 = _mm_max_ps(vacc3x4567, vmin);

    if XNN_LIKELY(nc >= 8) {
      _mm_storeu_ps(c0, vacc0x0123);
      _mm_storeu_ps(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      _mm_storeu_ps(c1, vacc1x0123);
      _mm_storeu_ps(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm_storeu_ps(c2, vacc2x0123);
      _mm_storeu_ps(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm_storeu_ps(c3, vacc3x0123);
      _mm_storeu_ps(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);
        _mm_storeu_ps(c1, vacc1x0123);
        _mm_storeu_ps(c2, vacc2x0123);
        _mm_storeu_ps(c3, vacc3x0123);

        vacc0x0123 = vacc0x4567;
        vacc1x0123 = vacc1x4567;
        vacc2x0123 = vacc2x4567;
        vacc3x0123 = vacc3x4567;

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vacc0x0123);
        _mm_storel_pi((__m64*) c1, vacc1x0123);
        _mm_storel_pi((__m64*) c2, vacc2x0123);
        _mm_storel_pi((__m64*) c3, vacc3x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);
        vacc2x0123 = _mm_movehl_ps(vacc2x0123, vacc2x0123);
        vacc3x0123 = _mm_movehl_ps(vacc3x0123, vacc3x0123);

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vacc0x0123);
        _mm_store_ss(c1, vacc1x0123);
        _mm_store_ss(c2, vacc2x0123);
        _mm_store_ss(c3, vacc3x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_qs8_vcvt_ukernel__sse41_u32(
    size_t batch,
    const float* input,
    int8_t* output,
    const union xnn_f32_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vscale = _mm_set1_ps(params->scalar.scale);
  const __m128 voutput_max_less_zero_point = _mm_set1_ps((float) ((int32_t) params->scalar.output_max - (int32_t) params->scalar.output_zero_point));
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8(params->scalar.output_min);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(voutput_max_less_zero_point);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m128 vx0123 = _mm_loadu_ps(input);
    __m128 vx4567 = _mm_loadu_ps(input + 4);
    __m128 vx89AB = _mm_loadu_ps(input + 8);
    __m128 vxCDEF = _mm_loadu_ps(input + 12);
    __m128 vxGHIJ = _mm_loadu_ps(input + 16);
    __m128 vxKLMN = _mm_loadu_ps(input + 20);
    __m128 vxOPQR = _mm_loadu_ps(input + 24);
    __m128 vxSTUV = _mm_loadu_ps(input + 28);
    input += 32;

    vx0123 = _mm_mul_ps(vx0123, vscale);
    vx4567 = _mm_mul_ps(vx4567, vscale);
    vx89AB = _mm_mul_ps(vx89AB, vscale);
    vxCDEF = _mm_mul_ps(vxCDEF, vscale);
    vxGHIJ = _mm_mul_ps(vxGHIJ, vscale);
    vxKLMN = _mm_mul_ps(vxKLMN, vscale);
    vxOPQR = _mm_mul_ps(vxOPQR, vscale);
    vxSTUV = _mm_mul_ps(vxSTUV, vscale);

    vx0123 = _mm_min_ps(vx0123, voutput_max_less_zero_point);
    vx4567 = _mm_min_ps(vx4567, voutput_max_less_zero_point);
    vx89AB = _mm_min_ps(vx89AB, voutput_max_less_zero_point);
    vxCDEF = _mm_min_ps(vxCDEF, voutput_max_less_zero_point);
    vxGHIJ = _mm_min_ps(vxGHIJ, voutput_max_less_zero_point);
    vxKLMN = _mm_min_ps(vxKLMN, voutput_max_less_zero_point);
    vxOPQR = _mm_min_ps(vxOPQR, voutput_max_less_zero_point);
    vxSTUV = _mm_min_ps(vxSTUV, voutput_max_less_zero_point);

    const __m128i vy0123 = _mm_cvtps_epi32(vx0123);
    const __m128i vy4567 = _mm_cvtps_epi32(vx4567);
    const __m128i vy89AB = _mm_cvtps_epi32(vx89AB);
    const __m128i vyCDEF = _mm_cvtps_epi32(vxCDEF);
    const __m128i vyGHIJ = _mm_cvtps_epi32(vxGHIJ);
    const __m128i vyKLMN = _mm_cvtps_epi32(vxKLMN);
    const __m128i vyOPQR = _mm_cvtps_epi32(vxOPQR);
    const __m128i vySTUV = _mm_cvtps_epi32(vxSTUV);

    __m128i vy01234567 = _mm_packs_epi32(vy0123, vy4567);
    __m128i vy89ABCDEF = _mm_packs_epi32(vy89AB, vyCDEF);
    __m128i vyGHIJKLMN = _mm_packs_epi32(vyGHIJ, vyKLMN);
    __m128i vyOPQRSTUV = _mm_packs_epi32(vyOPQR, vySTUV);

    vy01234567 = _mm_adds_epi16(vy01234567, voutput_zero_point);
    vy89ABCDEF = _mm_adds_epi16(vy89ABCDEF, voutput_zero_point);
    vyGHIJKLMN = _mm_adds_epi16(vyGHIJKLMN, voutput_zero_point);
    vyOPQRSTUV = _mm_adds_epi16(vyOPQRSTUV, voutput_zero_point);


    __m128i vy0123456789ABCDEF = _mm_packs_epi16(vy01234567, vy89ABCDEF);
    __m128i vyGHIJKLMNOPQRSTUV = _mm_packs_epi16(vyGHIJKLMN, vyOPQRSTUV);

    vy0123456789ABCDEF = _mm_max_epi8(vy0123456789ABCDEF, voutput_min);
    vyGHIJKLMNOPQRSTUV = _mm_max_epi8(vyGHIJKLMNOPQRSTUV, voutput_min);

    _mm_storeu_si128((__m128i*) output, vy0123456789ABCDEF);
    _mm_storeu_si128((__m128i*) (output + 16), vyGHIJKLMNOPQRSTUV);
    output += 32;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    __m128 vx_lo = _mm_loadu_ps(input);
    __m128 vx_hi = _mm_loadu_ps(input + 4);
    input += 8;

    vx_lo = _mm_mul_ps(vx_lo, vscale);
    vx_hi = _mm_mul_ps(vx_hi, vscale);

    vx_lo = _mm_min_ps(vx_lo, voutput_max_less_zero_point);
    vx_hi = _mm_min_ps(vx_hi, voutput_max_less_zero_point);

    const __m128i vy_lo = _mm_cvtps_epi32(vx_lo);
    const __m128i vy_hi = _mm_cvtps_epi32(vx_hi);

    __m128i vy = _mm_packs_epi32(vy_lo, vy_hi);
    vy = _mm_adds_epi16(vy, voutput_zero_point);
    vy = _mm_packs_epi16(vy, vy);
    vy = _mm_max_epi8(vy, voutput_min);

    _mm_storel_epi64((__m128i*) output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m128 vx_lo = _mm_loadu_ps(input);
    const float* x_hi = (const float*) ((uintptr_t) input + (batch & (4 * sizeof(float))));
    __m128 vx_hi = _mm_loadu_ps(x_hi);

    vx_lo = _mm_mul_ps(vx_lo, vscale);
    vx_hi = _mm_mul_ps(vx_hi, vscale);

    vx_lo = _mm_min_ps(vx_lo, voutput_max_less_zero_point);
    vx_hi = _mm_min_ps(vx_hi, voutput_max_less_zero_point);

    const __m128i vy_lo = _mm_cvtps_epi32(vx_lo);
    const __m128i vy_hi = _mm_cvtps_epi32(vx_hi);

    __m128i vy = _mm_packs_epi32(vy_lo, vy_hi);
    vy = _mm_adds_epi16(vy, voutput_zero_point);
    vy = _mm_packs_epi16(vy, vy);
    vy = _mm_max_epi8(vy, voutput_min);

    if (batch & (4 * sizeof(float))) {
      unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vy));
      output += 4;
      vy = _mm_srli_epi64(vy, 32);
    }
    if (batch & (2 * sizeof(float))) {
      unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vy, 0));
      output += 2;
      vy = _mm_srli_epi32(vy, 16);
    }
    if (batch & (1 * sizeof(float))) {
      *output = (int8_t) _mm_extract_epi8(vy, 0);
    }
  }
}

void xnn_f32_vlrelu_ukernel__sse41_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vslope = _mm_set1_ps(params->scalar.slope);
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(input);
    const __m128 vx4567 = _mm_loadu_ps(input + 4);
    input += 8;

    __m128 vacc0123 = _mm_mul_ps(vx0123, vslope);
    __m128 vacc4567 = _mm_mul_ps(vx4567, vslope);

    vacc0123 = _mm_blendv_ps(vx0123, vacc0123, vx0123);
    vacc4567 = _mm_blendv_ps(vx4567, vacc4567, vx4567);

    _mm_storeu_ps(output, vacc0123);
    _mm_storeu_ps(output + 4, vacc4567);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vx = _mm_loadu_ps(input);
    input += 4;

    __m128 vacc = _mm_mul_ps(vx, vslope);
    vacc = _mm_blendv_ps(vx, vacc, vx);

    _mm_storeu_ps(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 vx = _mm_loadu_ps(input);

    __m128 vacc = _mm_mul_ps(vx, vslope);
    vacc = _mm_blendv_ps(vx, vacc, vx);

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc);
      vacc = _mm_movehl_ps(vacc, vacc);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc);
    }
  }
}

void xnn_f32_vrndd_ukernel__sse41_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(input);
    const __m128 vx4567 = _mm_loadu_ps(input + 4);
    input += 8;

    const __m128 vy0123 = _mm_round_ps(vx0123, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    const __m128 vy4567 = _mm_round_ps(vx4567, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

    _mm_storeu_ps(output, vy0123);
    _mm_storeu_ps(output + 4, vy4567);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vx = _mm_loadu_ps(input);
    input += 4;

    const __m128 vy = _mm_round_ps(vx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

    _mm_storeu_ps(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 vx = _mm_loadu_ps(input);
    __m128 vy = _mm_round_ps(vx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
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

void xnn_f32_vrndne_ukernel__sse41_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(input);
    const __m128 vx4567 = _mm_loadu_ps(input + 4);
    input += 8;

    const __m128 vy0123 = _mm_round_ps(vx0123, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m128 vy4567 = _mm_round_ps(vx4567, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    _mm_storeu_ps(output, vy0123);
    _mm_storeu_ps(output + 4, vy4567);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vx = _mm_loadu_ps(input);
    input += 4;

    const __m128 vy = _mm_round_ps(vx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    _mm_storeu_ps(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 vx = _mm_loadu_ps(input);
    __m128 vy = _mm_round_ps(vx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
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

void xnn_f32_vrndu_ukernel__sse41_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(input);
    const __m128 vx4567 = _mm_loadu_ps(input + 4);
    input += 8;

    const __m128 vy0123 = _mm_round_ps(vx0123, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
    const __m128 vy4567 = _mm_round_ps(vx4567, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);

    _mm_storeu_ps(output, vy0123);
    _mm_storeu_ps(output + 4, vy4567);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vx = _mm_loadu_ps(input);
    input += 4;

    const __m128 vy = _mm_round_ps(vx, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);

    _mm_storeu_ps(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 vx = _mm_loadu_ps(input);
    __m128 vy = _mm_round_ps(vx, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
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

void xnn_f32_vrndz_ukernel__sse41_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(input);
    const __m128 vx4567 = _mm_loadu_ps(input + 4);
    input += 8;

    const __m128 vy0123 = _mm_round_ps(vx0123, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    const __m128 vy4567 = _mm_round_ps(vx4567, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);

    _mm_storeu_ps(output, vy0123);
    _mm_storeu_ps(output + 4, vy4567);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vx = _mm_loadu_ps(input);
    input += 4;

    const __m128 vy = _mm_round_ps(vx, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);

    _mm_storeu_ps(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 vx = _mm_loadu_ps(input);
    __m128 vy = _mm_round_ps(vx, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
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

extern XNN_INTERNAL const float xnn_table_exp2minus_k_over_64[64];

void xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vsign_mask = _mm_set1_ps(-0.0f);
  const __m128 vmagic_bias = _mm_set1_ps(0x1.800000p17f);
  const __m128 vlog2e = _mm_set1_ps(0x1.715476p0f);
  const __m128i vindex_mask = _mm_set1_epi32(UINT32_C(0x3F));
  const __m128 vminus_ln2_hi = _mm_set1_ps(-0x1.630000p-1f);
  const __m128 vminus_ln2_lo = _mm_set1_ps(0x1.BD0106p-13f);
  const __m128 vc2 = _mm_set1_ps(0x1.FFFF0Ap-2f);
  const __m128 vone = _mm_set1_ps(1.0f);
  const __m128 vdenorm_cutoff = _mm_set1_ps(-0x1.5D589Ep+6f);

  XNN_FORCE_REALIZATION(vsign_mask);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vindex_mask);
  XNN_FORCE_REALIZATION(vminus_ln2_hi);
  XNN_FORCE_REALIZATION(vminus_ln2_lo);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vone);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 vx0123 = _mm_loadu_ps(input);
    const __m128 vx4567 = _mm_loadu_ps(input + 4);
    input += 8;

    const __m128 vz0123 = _mm_or_ps(vx0123, vsign_mask);
    const __m128 vz4567 = _mm_or_ps(vx4567, vsign_mask);

    __m128 vn0123 = _mm_add_ps(_mm_mul_ps(vz0123, vlog2e), vmagic_bias);
    __m128 vn4567 = _mm_add_ps(_mm_mul_ps(vz4567, vlog2e), vmagic_bias);

    const __m128i ve0123 = _mm_slli_epi32(_mm_castps_si128(vn0123), 17);
    const __m128i ve4567 = _mm_slli_epi32(_mm_castps_si128(vn4567), 17);

    const __m128i vidx0123 = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn0123), vindex_mask), 2);
    const __m128i vidx4567 = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn4567), vindex_mask), 2);

    #if XNN_ARCH_X86_64
      const uint64_t vidx01 = (uint64_t) _mm_cvtsi128_si64(vidx0123);
      const uint64_t vidx23 = (uint64_t) _mm_extract_epi64(vidx0123, 1);
      const __m128i vl0   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx01)));
      const __m128i vl2 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx23)));
      const __m128i vl01 = _mm_insert_epi32(vl0, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) (vidx01 >> 32))), 1);
      const __m128i vl23 = _mm_insert_epi32(vl2, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) (vidx23 >> 32))), 1);
      const __m128i vl0123 = _mm_unpacklo_epi64(vl01, vl23);
      const uint64_t vidx45 = (uint64_t) _mm_cvtsi128_si64(vidx4567);
      const uint64_t vidx67 = (uint64_t) _mm_extract_epi64(vidx4567, 1);
      const __m128i vl4   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx45)));
      const __m128i vl6 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx67)));
      const __m128i vl45 = _mm_insert_epi32(vl4, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) (vidx45 >> 32))), 1);
      const __m128i vl67 = _mm_insert_epi32(vl6, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) (vidx67 >> 32))), 1);
      const __m128i vl4567 = _mm_unpacklo_epi64(vl45, vl67);
    #else  // !XNN_ARCH_X86_64
      const uint32_t vidx0 = (uint32_t) _mm_cvtsi128_si32(vidx0123);
      const uint32_t vidx1 = (uint32_t) _mm_extract_epi16(vidx0123, 2);
      const uint32_t vidx2 = (uint32_t) _mm_extract_epi16(vidx0123, 4);
      const uint32_t vidx3 = (uint32_t) _mm_extract_epi16(vidx0123, 6);
      const __m128i vl0   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + vidx0)));
      const __m128i vl2 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + vidx2)));
      const __m128i vl01 = _mm_insert_epi32(vl0, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + vidx1)), 1);
      const __m128i vl23 = _mm_insert_epi32(vl2, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + vidx3)), 1);
      const __m128i vl0123 = _mm_unpacklo_epi64(vl01, vl23);
      const uint32_t vidx4 = (uint32_t) _mm_cvtsi128_si32(vidx4567);
      const uint32_t vidx5 = (uint32_t) _mm_extract_epi16(vidx4567, 2);
      const uint32_t vidx6 = (uint32_t) _mm_extract_epi16(vidx4567, 4);
      const uint32_t vidx7 = (uint32_t) _mm_extract_epi16(vidx4567, 6);
      const __m128i vl4   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + vidx4)));
      const __m128i vl6 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + vidx6)));
      const __m128i vl45 = _mm_insert_epi32(vl4, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + vidx5)), 1);
      const __m128i vl67 = _mm_insert_epi32(vl6, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + vidx7)), 1);
      const __m128i vl4567 = _mm_unpacklo_epi64(vl45, vl67);
    #endif  // XNN_ARCH_X86_64

    const __m128 vs0123 = _mm_castsi128_ps(_mm_add_epi32(vl0123, ve0123));
    const __m128 vs4567 = _mm_castsi128_ps(_mm_add_epi32(vl4567, ve4567));

    vn0123 = _mm_sub_ps(vn0123, vmagic_bias);
    vn4567 = _mm_sub_ps(vn4567, vmagic_bias);

    __m128 vt0123 = _mm_add_ps(vz0123, _mm_mul_ps(vn0123, vminus_ln2_hi));
    __m128 vt4567 = _mm_add_ps(vz4567, _mm_mul_ps(vn4567, vminus_ln2_hi));

    vt0123 = _mm_add_ps(vt0123, _mm_mul_ps(vn0123, vminus_ln2_lo));
    vt4567 = _mm_add_ps(vt4567, _mm_mul_ps(vn4567, vminus_ln2_lo));

    __m128 vp0123 = _mm_mul_ps(vt0123, vc2);
    __m128 vp4567 = _mm_mul_ps(vt4567, vc2);

    vp0123 = _mm_add_ps(vt0123, _mm_mul_ps(vp0123, vt0123));
    vp4567 = _mm_add_ps(vt4567, _mm_mul_ps(vp4567, vt4567));

    const __m128 vy0123 = _mm_add_ps(vs0123, _mm_mul_ps(vs0123, vp0123));
    const __m128 vy4567 = _mm_add_ps(vs4567, _mm_mul_ps(vs4567, vp4567));

    __m128 vf0123 = _mm_div_ps(vy0123, _mm_add_ps(vy0123, vone));
    __m128 vf4567 = _mm_div_ps(vy4567, _mm_add_ps(vy4567, vone));

    vf0123 = _mm_andnot_ps(_mm_cmplt_ps(vz0123, vdenorm_cutoff), vf0123);
    vf4567 = _mm_andnot_ps(_mm_cmplt_ps(vz4567, vdenorm_cutoff), vf4567);

    vf0123 = _mm_blendv_ps(_mm_sub_ps(vone, vf0123), vf0123, vx0123);
    vf4567 = _mm_blendv_ps(_mm_sub_ps(vone, vf4567), vf4567, vx4567);

    _mm_storeu_ps(output, vf0123);
    _mm_storeu_ps(output + 4, vf4567);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 vx = _mm_loadu_ps(input);
    input += 4;

    const __m128 vz = _mm_or_ps(vx, vsign_mask);

    __m128 vn = _mm_add_ps(_mm_mul_ps(vz, vlog2e), vmagic_bias);
    const __m128i ve = _mm_slli_epi32(_mm_castps_si128(vn), 17);

    const __m128i vidx = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn), vindex_mask), 2);
    #if XNN_ARCH_X86_64
      const uint64_t vidx_lo = (uint64_t) _mm_cvtsi128_si64(vidx);
      const uint64_t vidx_hi = (uint64_t) _mm_extract_epi64(vidx, 1);
      const __m128i vl_ll   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx_lo)));
      const __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx_hi)));
      const __m128i vl_lo = _mm_insert_epi32(vl_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) (vidx_lo >> 32))), 1);
      const __m128i vl_hi = _mm_insert_epi32(vl_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) (vidx_hi >> 32))), 1);
    #else  // !XNN_ARCH_X86_64
      const __m128i vl_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) _mm_cvtsi128_si32(vidx))));
      const __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) _mm_extract_epi16(vidx, 4))));
      const __m128i vl_lo = _mm_insert_epi32(vl_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) _mm_extract_epi16(vidx, 2))), 1);
      const __m128i vl_hi = _mm_insert_epi32(vl_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) _mm_extract_epi16(vidx, 6))), 1);
    #endif  // XNN_ARCH_X86_64
    const __m128i vl = _mm_unpacklo_epi64(vl_lo, vl_hi);

    const __m128 vs = _mm_castsi128_ps(_mm_add_epi32(vl, ve));
    vn = _mm_sub_ps(vn, vmagic_bias);

    __m128 vt = _mm_add_ps(vz, _mm_mul_ps(vn, vminus_ln2_hi));
    vt = _mm_add_ps(vt, _mm_mul_ps(vn, vminus_ln2_lo));

    __m128 vp = _mm_mul_ps(vt, vc2);
    vp = _mm_add_ps(vt, _mm_mul_ps(vp, vt));

    const __m128 vy = _mm_add_ps(vs, _mm_mul_ps(vs, vp));

    __m128 vf = _mm_div_ps(vy, _mm_add_ps(vy, vone));
    vf = _mm_andnot_ps(_mm_cmplt_ps(vz, vdenorm_cutoff), vf);
    vf = _mm_blendv_ps(_mm_sub_ps(vone, vf), vf, vx);

    _mm_storeu_ps(output, vf);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 vx = _mm_loadu_ps(input);

    const __m128 vz = _mm_or_ps(vx, vsign_mask);

    __m128 vn = _mm_add_ps(_mm_mul_ps(vz, vlog2e), vmagic_bias);
    const __m128i ve = _mm_slli_epi32(_mm_castps_si128(vn), 17);

    const __m128i vidx = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn), vindex_mask), 2);
    #if XNN_ARCH_X86_64
      const uint64_t vidx_lo = (uint64_t) _mm_cvtsi128_si64(vidx);
      const uint64_t vidx_hi = (uint64_t) _mm_extract_epi64(vidx, 1);
      const __m128i vl_ll   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx_lo)));
      const __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) vidx_hi)));
      const __m128i vl_lo = _mm_insert_epi32(vl_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) (vidx_lo >> 32))), 1);
      const __m128i vl_hi = _mm_insert_epi32(vl_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) (vidx_hi >> 32))), 1);
    #else  // !XNN_ARCH_X86_64
      const __m128i vl_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) _mm_cvtsi128_si32(vidx))));
      const __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) _mm_extract_epi16(vidx, 4))));
      const __m128i vl_lo = _mm_insert_epi32(vl_ll, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) _mm_extract_epi16(vidx, 2))), 1);
      const __m128i vl_hi = _mm_insert_epi32(vl_hl, *((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_64 + (uint32_t) _mm_extract_epi16(vidx, 6))), 1);
    #endif  // XNN_ARCH_X86_64
    const __m128i vl = _mm_unpacklo_epi64(vl_lo, vl_hi);

    const __m128 vs = _mm_castsi128_ps(_mm_add_epi32(vl, ve));
    vn = _mm_sub_ps(vn, vmagic_bias);

    __m128 vt = _mm_add_ps(vz, _mm_mul_ps(vn, vminus_ln2_hi));
    vt = _mm_add_ps(vt, _mm_mul_ps(vn, vminus_ln2_lo));

    __m128 vp = _mm_mul_ps(vt, vc2);
    vp = _mm_add_ps(vt, _mm_mul_ps(vp, vt));

    const __m128 vy = _mm_add_ps(vs, _mm_mul_ps(vs, vp));

    __m128 vf = _mm_div_ps(vy, _mm_add_ps(vy, vone));
    vf = _mm_andnot_ps(_mm_cmplt_ps(vz, vdenorm_cutoff), vf);
    vf = _mm_blendv_ps(_mm_sub_ps(vone, vf), vf, vx);

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vf);
      vf = _mm_movehl_ps(vf, vf);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vf);
    }
  }
}

void xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x4c8__sse41_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qb4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  size_t bl = params->sse.blocksize;
  assert(bl <= round_up_po2(kc, 2));
  assert(bl != 0);
  assert(bl % 32 == 0);
  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  const __m128i vmask = _mm_set1_epi8(0xF0);
  XNN_FORCE_REALIZATION(vmask);
  do {
    const __m128 vksum = _mm_load_ps((const float*) w);
    __m128i vinput_zero_point0 = _mm_cvtsi32_si128(*((const int*) &quantization_params[0].zero_point));
    vinput_zero_point0 = _mm_shuffle_epi32(vinput_zero_point0, _MM_SHUFFLE(0, 0, 0, 0));

    __m128 vinput_zero_point0_float = _mm_cvtepi32_ps(vinput_zero_point0);
    __m128 vout0x0123 = _mm_mul_ps(vksum, vinput_zero_point0_float);
    w = (const int32_t*) w + 4;

    for (size_t kb=0; kb < kc; kb += bl) {
      __m128i vacc0x0 = _mm_setzero_si128();
      __m128i vacc0x1 = _mm_setzero_si128();
      __m128i vacc0x2 = _mm_setzero_si128();
      __m128i vacc0x3 = _mm_setzero_si128();
      size_t k = bl;

      while (k >= 16 * sizeof(int8_t)) {
        const __m128i va0c0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0c0 = _mm_cvtepi8_epi16(va0c0);
        a0 += 8;

        const __m128i vb01c01 = _mm_loadu_si128((const __m128i*) w);
        const __m128i vbs01c0 = _mm_slli_epi32(vb01c01, 4);
        const __m128i vb01c0 = _mm_and_si128(vbs01c0, vmask);
        const __m128i vsb01c0 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb01c0);
        const __m128i vxb0c0 = _mm_unpacklo_epi8(vb01c0, vsb01c0);
        const __m128i vxb1c0 = _mm_unpackhi_epi8(vb01c0, vsb01c0);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0c0, vxb0c0));
        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0c0, vxb1c0));
        const __m128i vb23c01 = _mm_loadu_si128((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vbs23c0 = _mm_slli_epi32(vb23c01, 4);
        const __m128i vb23c0 = _mm_and_si128(vbs23c0, vmask);
        const __m128i vsb23c0 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb23c0);
        const __m128i vxb2c0 = _mm_unpacklo_epi8(vb23c0, vsb23c0);
        const __m128i vxb3c0 = _mm_unpackhi_epi8(vb23c0, vsb23c0);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0c0, vxb2c0));
        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0c0, vxb3c0));

        const __m128i va0c1 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0c1 = _mm_cvtepi8_epi16(va0c1);
        a0 += 8;

        const __m128i vb01c1 = _mm_and_si128(vb01c01, vmask);
        const __m128i vsb01c1 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb01c1);
        const __m128i vxb0c1 = _mm_unpacklo_epi8(vb01c1, vsb01c1);
        const __m128i vxb1c1 = _mm_unpackhi_epi8(vb01c1, vsb01c1);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0c1, vxb0c1));
        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0c1, vxb1c1));
        const __m128i vb23c1 = _mm_and_si128(vb23c01, vmask);
        const __m128i vsb23c1 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb23c1);
        const __m128i vxb2c1 = _mm_unpacklo_epi8(vb23c1, vsb23c1);
        const __m128i vxb3c1 = _mm_unpackhi_epi8(vb23c1, vsb23c1);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0c1, vxb2c1));
        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0c1, vxb3c1));


        w = (const int8_t*) w + 32;
        k -= 16 * sizeof(int8_t);
      }

      while (k >= 8 * sizeof(int8_t)) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
        a0 += 8;

        __m128i vb01 = _mm_loadu_si128((const __m128i*) w);
        vb01 = _mm_slli_epi32(vb01, 4);
        vb01 = _mm_and_si128(vb01, vmask);

        const __m128i vxbm1 = _mm_unpackhi_epi8(vb01, vb01);
        const __m128i vxb0 = _mm_cvtepi8_epi16(vb01);
        const __m128i vxb1 = _mm_srai_epi16(vxbm1, 8);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        __m128i vb23 = _mm_loadu_si128((const __m128i*) ((const int8_t*) w + 16));
        vb23 = _mm_slli_epi32(vb23, 4);
        vb23 = _mm_and_si128(vb23, vmask);

        const __m128i vxbm3 = _mm_unpackhi_epi8(vb23, vb23);
        const __m128i vxb2 = _mm_cvtepi8_epi16(vb23);
        const __m128i vxb3 = _mm_srai_epi16(vxbm3, 8);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));

        w = (const int8_t*) w + 32;
        k -= 8 * sizeof(int8_t);
      }
      // accumulate float
      const __m128 vfilter_output_scale0123 = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i*) w)), 16));
      w = (const uint16_t*) w + 4;

      const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
      const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);

        __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);

      vout0x0123 = _mm_add_ps(vout0x0123, _mm_mul_ps(_mm_cvtepi32_ps(vacc0x0123), vfilter_output_scale0123));
    }

    const __m128 vinput_scale0 = _mm_load1_ps(&quantization_params[0].inv_scale);

    vout0x0123 = _mm_mul_ps(vout0x0123, vinput_scale0);


    const __m128 vbias0123 = _mm_loadu_ps((const float*) w);
    w = (const float*) w + 4;
    vout0x0123 = _mm_add_ps(vout0x0123, vbias0123);

    vout0x0123 = _mm_max_ps(vout0x0123, vmin);

    vout0x0123 = _mm_min_ps(vout0x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c0, vout0x0123);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vout0x0123);
        vout0x0123 = _mm_unpackhi_ps(vout0x0123, vout0x0123);
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vout0x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qb4w_gemm_minmax_ukernel_3x4c8__sse41_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qb4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  size_t bl = params->sse.blocksize;
  assert(bl <= round_up_po2(kc, 2));
  assert(bl != 0);
  assert(bl % 32 == 0);
  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  const __m128i vmask = _mm_set1_epi8(0xF0);
  XNN_FORCE_REALIZATION(vmask);
  do {
    const __m128 vksum = _mm_load_ps((const float*) w);
    const __m128i vinput_zero_point01 = _mm_loadu_si128((const __m128i*) &quantization_params[0]);
    const __m128i vinput_zero_point0 = _mm_shuffle_epi32(vinput_zero_point01, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128i vinput_zero_point1 = _mm_shuffle_epi32(vinput_zero_point01, _MM_SHUFFLE(2, 2, 2, 2));
    __m128i vinput_zero_point2 = _mm_cvtsi32_si128(*((const int*) &quantization_params[2].zero_point));
    vinput_zero_point2 = _mm_shuffle_epi32(vinput_zero_point2, _MM_SHUFFLE(0, 0, 0, 0));

    __m128 vinput_zero_point0_float = _mm_cvtepi32_ps(vinput_zero_point0);
    __m128 vout0x0123 = _mm_mul_ps(vksum, vinput_zero_point0_float);
    __m128 vinput_zero_point1_float = _mm_cvtepi32_ps(vinput_zero_point1);
    __m128 vout1x0123 = _mm_mul_ps(vksum, vinput_zero_point1_float);
    __m128 vinput_zero_point2_float = _mm_cvtepi32_ps(vinput_zero_point2);
    __m128 vout2x0123 = _mm_mul_ps(vksum, vinput_zero_point2_float);
    w = (const int32_t*) w + 4;

    for (size_t kb=0; kb < kc; kb += bl) {
      __m128i vacc0x0 = _mm_setzero_si128();
      __m128i vacc0x1 = _mm_setzero_si128();
      __m128i vacc0x2 = _mm_setzero_si128();
      __m128i vacc0x3 = _mm_setzero_si128();
      __m128i vacc1x0 = _mm_setzero_si128();
      __m128i vacc1x1 = _mm_setzero_si128();
      __m128i vacc1x2 = _mm_setzero_si128();
      __m128i vacc1x3 = _mm_setzero_si128();
      __m128i vacc2x0 = _mm_setzero_si128();
      __m128i vacc2x1 = _mm_setzero_si128();
      __m128i vacc2x2 = _mm_setzero_si128();
      __m128i vacc2x3 = _mm_setzero_si128();
      size_t k = bl;

      while (k >= 16 * sizeof(int8_t)) {
        const __m128i va0c0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0c0 = _mm_cvtepi8_epi16(va0c0);
        a0 += 8;
        const __m128i va1c0 = _mm_loadl_epi64((const __m128i*) a1);
        const __m128i vxa1c0 = _mm_cvtepi8_epi16(va1c0);
        a1 += 8;
        const __m128i va2c0 = _mm_loadl_epi64((const __m128i*) a2);
        const __m128i vxa2c0 = _mm_cvtepi8_epi16(va2c0);
        a2 += 8;

        const __m128i vb01c01 = _mm_loadu_si128((const __m128i*) w);
        const __m128i vbs01c0 = _mm_slli_epi32(vb01c01, 4);
        const __m128i vb01c0 = _mm_and_si128(vbs01c0, vmask);
        const __m128i vsb01c0 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb01c0);
        const __m128i vxb0c0 = _mm_unpacklo_epi8(vb01c0, vsb01c0);
        const __m128i vxb1c0 = _mm_unpackhi_epi8(vb01c0, vsb01c0);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0c0, vxb0c0));
        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0c0, vxb1c0));
        vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1c0, vxb0c0));
        vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1c0, vxb1c0));
        vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2c0, vxb0c0));
        vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2c0, vxb1c0));
        const __m128i vb23c01 = _mm_loadu_si128((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vbs23c0 = _mm_slli_epi32(vb23c01, 4);
        const __m128i vb23c0 = _mm_and_si128(vbs23c0, vmask);
        const __m128i vsb23c0 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb23c0);
        const __m128i vxb2c0 = _mm_unpacklo_epi8(vb23c0, vsb23c0);
        const __m128i vxb3c0 = _mm_unpackhi_epi8(vb23c0, vsb23c0);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0c0, vxb2c0));
        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0c0, vxb3c0));
        vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1c0, vxb2c0));
        vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1c0, vxb3c0));
        vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2c0, vxb2c0));
        vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2c0, vxb3c0));

        const __m128i va0c1 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0c1 = _mm_cvtepi8_epi16(va0c1);
        a0 += 8;
        const __m128i va1c1 = _mm_loadl_epi64((const __m128i*) a1);
        const __m128i vxa1c1 = _mm_cvtepi8_epi16(va1c1);
        a1 += 8;
        const __m128i va2c1 = _mm_loadl_epi64((const __m128i*) a2);
        const __m128i vxa2c1 = _mm_cvtepi8_epi16(va2c1);
        a2 += 8;

        const __m128i vb01c1 = _mm_and_si128(vb01c01, vmask);
        const __m128i vsb01c1 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb01c1);
        const __m128i vxb0c1 = _mm_unpacklo_epi8(vb01c1, vsb01c1);
        const __m128i vxb1c1 = _mm_unpackhi_epi8(vb01c1, vsb01c1);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0c1, vxb0c1));
        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0c1, vxb1c1));
        vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1c1, vxb0c1));
        vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1c1, vxb1c1));
        vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2c1, vxb0c1));
        vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2c1, vxb1c1));
        const __m128i vb23c1 = _mm_and_si128(vb23c01, vmask);
        const __m128i vsb23c1 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb23c1);
        const __m128i vxb2c1 = _mm_unpacklo_epi8(vb23c1, vsb23c1);
        const __m128i vxb3c1 = _mm_unpackhi_epi8(vb23c1, vsb23c1);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0c1, vxb2c1));
        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0c1, vxb3c1));
        vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1c1, vxb2c1));
        vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1c1, vxb3c1));
        vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2c1, vxb2c1));
        vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2c1, vxb3c1));


        w = (const int8_t*) w + 32;
        k -= 16 * sizeof(int8_t);
      }

      while (k >= 8 * sizeof(int8_t)) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
        a0 += 8;
        const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
        const __m128i vxa1 = _mm_cvtepi8_epi16(va1);
        a1 += 8;
        const __m128i va2 = _mm_loadl_epi64((const __m128i*) a2);
        const __m128i vxa2 = _mm_cvtepi8_epi16(va2);
        a2 += 8;

        __m128i vb01 = _mm_loadu_si128((const __m128i*) w);
        vb01 = _mm_slli_epi32(vb01, 4);
        vb01 = _mm_and_si128(vb01, vmask);

        const __m128i vxbm1 = _mm_unpackhi_epi8(vb01, vb01);
        const __m128i vxb0 = _mm_cvtepi8_epi16(vb01);
        const __m128i vxb1 = _mm_srai_epi16(vxbm1, 8);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
        vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
        vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2, vxb0));
        vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2, vxb1));
        __m128i vb23 = _mm_loadu_si128((const __m128i*) ((const int8_t*) w + 16));
        vb23 = _mm_slli_epi32(vb23, 4);
        vb23 = _mm_and_si128(vb23, vmask);

        const __m128i vxbm3 = _mm_unpackhi_epi8(vb23, vb23);
        const __m128i vxb2 = _mm_cvtepi8_epi16(vb23);
        const __m128i vxb3 = _mm_srai_epi16(vxbm3, 8);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
        vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
        vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));
        vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2, vxb2));
        vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2, vxb3));

        w = (const int8_t*) w + 32;
        k -= 8 * sizeof(int8_t);
      }
      // accumulate float
      const __m128 vfilter_output_scale0123 = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i*) w)), 16));
      w = (const uint16_t*) w + 4;

      const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
      const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);
      const __m128i vacc1x01 = _mm_hadd_epi32(vacc1x0, vacc1x1);
      const __m128i vacc1x23 = _mm_hadd_epi32(vacc1x2, vacc1x3);
      const __m128i vacc2x01 = _mm_hadd_epi32(vacc2x0, vacc2x1);
      const __m128i vacc2x23 = _mm_hadd_epi32(vacc2x2, vacc2x3);

        __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
        __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);
        __m128i vacc2x0123 = _mm_hadd_epi32(vacc2x01, vacc2x23);

      vout0x0123 = _mm_add_ps(vout0x0123, _mm_mul_ps(_mm_cvtepi32_ps(vacc0x0123), vfilter_output_scale0123));
      vout1x0123 = _mm_add_ps(vout1x0123, _mm_mul_ps(_mm_cvtepi32_ps(vacc1x0123), vfilter_output_scale0123));
      vout2x0123 = _mm_add_ps(vout2x0123, _mm_mul_ps(_mm_cvtepi32_ps(vacc2x0123), vfilter_output_scale0123));
    }

    const __m128i vinput_scale01 = _mm_loadu_si128((const __m128i*) &quantization_params[0]);
    const __m128 vinput_scale0 = _mm_castsi128_ps(_mm_shuffle_epi32(vinput_scale01, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128 vinput_scale1 = _mm_castsi128_ps(_mm_shuffle_epi32(vinput_scale01, _MM_SHUFFLE(3, 3, 3, 3)));
    const __m128 vinput_scale2 = _mm_load1_ps(&quantization_params[2].inv_scale);

    vout0x0123 = _mm_mul_ps(vout0x0123, vinput_scale0);
    vout1x0123 = _mm_mul_ps(vout1x0123, vinput_scale1);
    vout2x0123 = _mm_mul_ps(vout2x0123, vinput_scale2);


    const __m128 vbias0123 = _mm_loadu_ps((const float*) w);
    w = (const float*) w + 4;
    vout0x0123 = _mm_add_ps(vout0x0123, vbias0123);
    vout1x0123 = _mm_add_ps(vout1x0123, vbias0123);
    vout2x0123 = _mm_add_ps(vout2x0123, vbias0123);

    vout0x0123 = _mm_max_ps(vout0x0123, vmin);
    vout1x0123 = _mm_max_ps(vout1x0123, vmin);
    vout2x0123 = _mm_max_ps(vout2x0123, vmin);

    vout0x0123 = _mm_min_ps(vout0x0123, vmax);
    vout1x0123 = _mm_min_ps(vout1x0123, vmax);
    vout2x0123 = _mm_min_ps(vout2x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c0, vout0x0123);
      _mm_storeu_ps(c1, vout1x0123);
      _mm_storeu_ps(c2, vout2x0123);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vout0x0123);
        vout0x0123 = _mm_unpackhi_ps(vout0x0123, vout0x0123);
        c0 += 2;
        _mm_storel_pi((__m64*) c1, vout1x0123);
        vout1x0123 = _mm_unpackhi_ps(vout1x0123, vout1x0123);
        c1 += 2;
        _mm_storel_pi((__m64*) c2, vout2x0123);
        vout2x0123 = _mm_unpackhi_ps(vout2x0123, vout2x0123);
        c2 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vout0x0123);
        _mm_store_ss(c1, vout1x0123);
        _mm_store_ss(c2, vout2x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse41_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  const __m128i vmask = _mm_set1_epi8(0xF0);
  XNN_FORCE_REALIZATION(vmask);
  do {
    const __m128i vksum = _mm_load_si128((const __m128i*) w);
    __m128i vinput_zero_point0 = _mm_cvtsi32_si128(*((const int*) &quantization_params[0].zero_point));
    vinput_zero_point0 = _mm_shuffle_epi32(vinput_zero_point0, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128i vzero = _mm_setzero_si128();
    const __m128i vinit0 = _mm_mullo_epi32(vksum, vinput_zero_point0);
    __m128i vacc0x0 = _mm_blend_epi16(vinit0, vzero, 0xFC);
    __m128i vacc0x1 = _mm_blend_epi16(vinit0, vzero, 0xF3);
    __m128i vacc0x2 = _mm_blend_epi16(vinit0, vzero, 0xCF);
    __m128i vacc0x3 = _mm_blend_epi16(vinit0, vzero, 0x3F);
    w = (const int32_t*) w + 4;

    size_t k = kc;

    while (k >= 16 * sizeof(int8_t)) {
      const __m128i va0c0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0c0 = _mm_cvtepi8_epi16(va0c0);
      a0 += 8;

      const __m128i vb01c01 = _mm_load_si128((const __m128i*) w);
      const __m128i vbs01c0 = _mm_slli_epi32(vb01c01, 4);
      const __m128i vb01c0 = _mm_and_si128(vbs01c0, vmask);
      const __m128i vsb01c0 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb01c0);
      const __m128i vxb0c0 = _mm_unpacklo_epi8(vb01c0, vsb01c0);
      const __m128i vxb1c0 = _mm_unpackhi_epi8(vb01c0, vsb01c0);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0c0, vxb0c0));
      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0c0, vxb1c0));
      const __m128i vb23c01 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      const __m128i vbs23c0 = _mm_slli_epi32(vb23c01, 4);
      const __m128i vb23c0 = _mm_and_si128(vbs23c0, vmask);
      const __m128i vsb23c0 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb23c0);
      const __m128i vxb2c0 = _mm_unpacklo_epi8(vb23c0, vsb23c0);
      const __m128i vxb3c0 = _mm_unpackhi_epi8(vb23c0, vsb23c0);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0c0, vxb2c0));
      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0c0, vxb3c0));

      const __m128i va0c1 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0c1 = _mm_cvtepi8_epi16(va0c1);
      a0 += 8;

      const __m128i vb01c1 = _mm_and_si128(vb01c01, vmask);
      const __m128i vsb01c1 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb01c1);
      const __m128i vxb0c1 = _mm_unpacklo_epi8(vb01c1, vsb01c1);
      const __m128i vxb1c1 = _mm_unpackhi_epi8(vb01c1, vsb01c1);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0c1, vxb0c1));
      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0c1, vxb1c1));
      const __m128i vb23c1 = _mm_and_si128(vb23c01, vmask);
      const __m128i vsb23c1 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb23c1);
      const __m128i vxb2c1 = _mm_unpacklo_epi8(vb23c1, vsb23c1);
      const __m128i vxb3c1 = _mm_unpackhi_epi8(vb23c1, vsb23c1);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0c1, vxb2c1));
      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0c1, vxb3c1));


      w = (const int8_t*) w + 32;
      k -= 16 * sizeof(int8_t);
    }

    while (k >= 8 * sizeof(int8_t)) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
      a0 += 8;

      __m128i vb01 = _mm_load_si128((const __m128i*) w);
      vb01 = _mm_slli_epi32(vb01, 4);
      vb01 = _mm_and_si128(vb01, vmask);

      const __m128i vxbm1 = _mm_unpackhi_epi8(vb01, vb01);
      const __m128i vxb0 = _mm_cvtepi8_epi16(vb01);
      const __m128i vxb1 = _mm_srai_epi16(vxbm1, 8);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      vb23 = _mm_slli_epi32(vb23, 4);
      vb23 = _mm_and_si128(vb23, vmask);

      const __m128i vxbm3 = _mm_unpackhi_epi8(vb23, vb23);
      const __m128i vxb2 = _mm_cvtepi8_epi16(vb23);
      const __m128i vxb3 = _mm_srai_epi16(vxbm3, 8);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));

      w = (const int8_t*) w + 32;
      k -= 8 * sizeof(int8_t);
    }

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);

    vacc0x0123 = _mm_srai_epi32(vacc0x0123, 4);
    __m128 vout0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 vinput_scale0 = _mm_load1_ps(&quantization_params[0].inv_scale);

    vout0x0123 = _mm_mul_ps(vout0x0123, vinput_scale0);

    const __m128 vfilter_output_scale0123 = _mm_load_ps((const float*) w);
    vout0x0123 = _mm_mul_ps(vout0x0123, vfilter_output_scale0123);

    const __m128 vbias0123 = _mm_load_ps((const float*) w + 4);
    w = (const float*) w + 8;
    vout0x0123 = _mm_add_ps(vout0x0123, vbias0123);

    vout0x0123 = _mm_max_ps(vout0x0123, vmin);

    vout0x0123 = _mm_min_ps(vout0x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c0, vout0x0123);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vout0x0123);
        vout0x0123 = _mm_unpackhi_ps(vout0x0123, vout0x0123);
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vout0x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__sse41_ld128(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  const __m128i vmask = _mm_set1_epi8(0xF0);
  XNN_FORCE_REALIZATION(vmask);
  do {
    const __m128i vksum = _mm_load_si128((const __m128i*) w);
    const __m128i vinput_zero_point01 = _mm_loadu_si128((const __m128i*) &quantization_params[0]);
    const __m128i vinput_zero_point0 = _mm_shuffle_epi32(vinput_zero_point01, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128i vinput_zero_point1 = _mm_shuffle_epi32(vinput_zero_point01, _MM_SHUFFLE(2, 2, 2, 2));
    __m128i vinput_zero_point2 = _mm_cvtsi32_si128(*((const int*) &quantization_params[2].zero_point));
    vinput_zero_point2 = _mm_shuffle_epi32(vinput_zero_point2, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128i vzero = _mm_setzero_si128();
    const __m128i vinit0 = _mm_mullo_epi32(vksum, vinput_zero_point0);
    const __m128i vinit1 = _mm_mullo_epi32(vksum, vinput_zero_point1);
    const __m128i vinit2 = _mm_mullo_epi32(vksum, vinput_zero_point2);
    __m128i vacc0x0 = _mm_blend_epi16(vinit0, vzero, 0xFC);
    __m128i vacc0x1 = _mm_blend_epi16(vinit0, vzero, 0xF3);
    __m128i vacc0x2 = _mm_blend_epi16(vinit0, vzero, 0xCF);
    __m128i vacc0x3 = _mm_blend_epi16(vinit0, vzero, 0x3F);
    __m128i vacc1x0 = _mm_blend_epi16(vinit1, vzero, 0xFC);
    __m128i vacc1x1 = _mm_blend_epi16(vinit1, vzero, 0xF3);
    __m128i vacc1x2 = _mm_blend_epi16(vinit1, vzero, 0xCF);
    __m128i vacc1x3 = _mm_blend_epi16(vinit1, vzero, 0x3F);
    __m128i vacc2x0 = _mm_blend_epi16(vinit2, vzero, 0xFC);
    __m128i vacc2x1 = _mm_blend_epi16(vinit2, vzero, 0xF3);
    __m128i vacc2x2 = _mm_blend_epi16(vinit2, vzero, 0xCF);
    __m128i vacc2x3 = _mm_blend_epi16(vinit2, vzero, 0x3F);
    w = (const int32_t*) w + 4;

    size_t k = kc;

    while (k >= 16 * sizeof(int8_t)) {
      const __m128i va0c0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0c0 = _mm_cvtepi8_epi16(va0c0);
      a0 += 8;
      const __m128i va1c0 = _mm_loadl_epi64((const __m128i*) a1);
      const __m128i vxa1c0 = _mm_cvtepi8_epi16(va1c0);
      a1 += 8;
      const __m128i va2c0 = _mm_loadl_epi64((const __m128i*) a2);
      const __m128i vxa2c0 = _mm_cvtepi8_epi16(va2c0);
      a2 += 8;

      const __m128i vb01c01 = _mm_load_si128((const __m128i*) w);
      const __m128i vbs01c0 = _mm_slli_epi32(vb01c01, 4);
      const __m128i vb01c0 = _mm_and_si128(vbs01c0, vmask);
      const __m128i vsb01c0 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb01c0);
      const __m128i vxb0c0 = _mm_unpacklo_epi8(vb01c0, vsb01c0);
      const __m128i vxb1c0 = _mm_unpackhi_epi8(vb01c0, vsb01c0);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0c0, vxb0c0));
      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0c0, vxb1c0));
      vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1c0, vxb0c0));
      vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1c0, vxb1c0));
      vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2c0, vxb0c0));
      vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2c0, vxb1c0));
      const __m128i vb23c01 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      const __m128i vbs23c0 = _mm_slli_epi32(vb23c01, 4);
      const __m128i vb23c0 = _mm_and_si128(vbs23c0, vmask);
      const __m128i vsb23c0 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb23c0);
      const __m128i vxb2c0 = _mm_unpacklo_epi8(vb23c0, vsb23c0);
      const __m128i vxb3c0 = _mm_unpackhi_epi8(vb23c0, vsb23c0);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0c0, vxb2c0));
      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0c0, vxb3c0));
      vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1c0, vxb2c0));
      vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1c0, vxb3c0));
      vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2c0, vxb2c0));
      vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2c0, vxb3c0));

      const __m128i va0c1 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0c1 = _mm_cvtepi8_epi16(va0c1);
      a0 += 8;
      const __m128i va1c1 = _mm_loadl_epi64((const __m128i*) a1);
      const __m128i vxa1c1 = _mm_cvtepi8_epi16(va1c1);
      a1 += 8;
      const __m128i va2c1 = _mm_loadl_epi64((const __m128i*) a2);
      const __m128i vxa2c1 = _mm_cvtepi8_epi16(va2c1);
      a2 += 8;

      const __m128i vb01c1 = _mm_and_si128(vb01c01, vmask);
      const __m128i vsb01c1 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb01c1);
      const __m128i vxb0c1 = _mm_unpacklo_epi8(vb01c1, vsb01c1);
      const __m128i vxb1c1 = _mm_unpackhi_epi8(vb01c1, vsb01c1);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0c1, vxb0c1));
      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0c1, vxb1c1));
      vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1c1, vxb0c1));
      vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1c1, vxb1c1));
      vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2c1, vxb0c1));
      vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2c1, vxb1c1));
      const __m128i vb23c1 = _mm_and_si128(vb23c01, vmask);
      const __m128i vsb23c1 = _mm_cmpgt_epi8(_mm_setzero_si128(), vb23c1);
      const __m128i vxb2c1 = _mm_unpacklo_epi8(vb23c1, vsb23c1);
      const __m128i vxb3c1 = _mm_unpackhi_epi8(vb23c1, vsb23c1);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0c1, vxb2c1));
      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0c1, vxb3c1));
      vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1c1, vxb2c1));
      vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1c1, vxb3c1));
      vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2c1, vxb2c1));
      vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2c1, vxb3c1));


      w = (const int8_t*) w + 32;
      k -= 16 * sizeof(int8_t);
    }

    while (k >= 8 * sizeof(int8_t)) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
      a0 += 8;
      const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
      const __m128i vxa1 = _mm_cvtepi8_epi16(va1);
      a1 += 8;
      const __m128i va2 = _mm_loadl_epi64((const __m128i*) a2);
      const __m128i vxa2 = _mm_cvtepi8_epi16(va2);
      a2 += 8;

      __m128i vb01 = _mm_load_si128((const __m128i*) w);
      vb01 = _mm_slli_epi32(vb01, 4);
      vb01 = _mm_and_si128(vb01, vmask);

      const __m128i vxbm1 = _mm_unpackhi_epi8(vb01, vb01);
      const __m128i vxb0 = _mm_cvtepi8_epi16(vb01);
      const __m128i vxb1 = _mm_srai_epi16(vxbm1, 8);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
      vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
      vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2, vxb0));
      vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2, vxb1));
      __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      vb23 = _mm_slli_epi32(vb23, 4);
      vb23 = _mm_and_si128(vb23, vmask);

      const __m128i vxbm3 = _mm_unpackhi_epi8(vb23, vb23);
      const __m128i vxb2 = _mm_cvtepi8_epi16(vb23);
      const __m128i vxb3 = _mm_srai_epi16(vxbm3, 8);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
      vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
      vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));
      vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2, vxb2));
      vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2, vxb3));

      w = (const int8_t*) w + 32;
      k -= 8 * sizeof(int8_t);
    }

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);
    const __m128i vacc1x01 = _mm_hadd_epi32(vacc1x0, vacc1x1);
    const __m128i vacc1x23 = _mm_hadd_epi32(vacc1x2, vacc1x3);
    const __m128i vacc2x01 = _mm_hadd_epi32(vacc2x0, vacc2x1);
    const __m128i vacc2x23 = _mm_hadd_epi32(vacc2x2, vacc2x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
    __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);
    __m128i vacc2x0123 = _mm_hadd_epi32(vacc2x01, vacc2x23);

    vacc0x0123 = _mm_srai_epi32(vacc0x0123, 4);
    vacc1x0123 = _mm_srai_epi32(vacc1x0123, 4);
    vacc2x0123 = _mm_srai_epi32(vacc2x0123, 4);
    __m128 vout0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vout1x0123 = _mm_cvtepi32_ps(vacc1x0123);
    __m128 vout2x0123 = _mm_cvtepi32_ps(vacc2x0123);

    const __m128i vinput_scale01 = _mm_loadu_si128((const __m128i*) &quantization_params[0]);
    const __m128 vinput_scale0 = _mm_castsi128_ps(_mm_shuffle_epi32(vinput_scale01, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128 vinput_scale1 = _mm_castsi128_ps(_mm_shuffle_epi32(vinput_scale01, _MM_SHUFFLE(3, 3, 3, 3)));
    const __m128 vinput_scale2 = _mm_load1_ps(&quantization_params[2].inv_scale);

    vout0x0123 = _mm_mul_ps(vout0x0123, vinput_scale0);
    vout1x0123 = _mm_mul_ps(vout1x0123, vinput_scale1);
    vout2x0123 = _mm_mul_ps(vout2x0123, vinput_scale2);

    const __m128 vfilter_output_scale0123 = _mm_load_ps((const float*) w);
    vout0x0123 = _mm_mul_ps(vout0x0123, vfilter_output_scale0123);
    vout1x0123 = _mm_mul_ps(vout1x0123, vfilter_output_scale0123);
    vout2x0123 = _mm_mul_ps(vout2x0123, vfilter_output_scale0123);

    const __m128 vbias0123 = _mm_load_ps((const float*) w + 4);
    w = (const float*) w + 8;
    vout0x0123 = _mm_add_ps(vout0x0123, vbias0123);
    vout1x0123 = _mm_add_ps(vout1x0123, vbias0123);
    vout2x0123 = _mm_add_ps(vout2x0123, vbias0123);

    vout0x0123 = _mm_max_ps(vout0x0123, vmin);
    vout1x0123 = _mm_max_ps(vout1x0123, vmin);
    vout2x0123 = _mm_max_ps(vout2x0123, vmin);

    vout0x0123 = _mm_min_ps(vout0x0123, vmax);
    vout1x0123 = _mm_min_ps(vout1x0123, vmax);
    vout2x0123 = _mm_min_ps(vout2x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c0, vout0x0123);
      _mm_storeu_ps(c1, vout1x0123);
      _mm_storeu_ps(c2, vout2x0123);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vout0x0123);
        vout0x0123 = _mm_unpackhi_ps(vout0x0123, vout0x0123);
        c0 += 2;
        _mm_storel_pi((__m64*) c1, vout1x0123);
        vout1x0123 = _mm_unpackhi_ps(vout1x0123, vout1x0123);
        c1 += 2;
        _mm_storel_pi((__m64*) c2, vout2x0123);
        vout2x0123 = _mm_unpackhi_ps(vout2x0123, vout2x0123);
        c2 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vout0x0123);
        _mm_store_ss(c1, vout1x0123);
        _mm_store_ss(c2, vout2x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse41_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    const __m128i vksum = _mm_load_si128((const __m128i*) w);
    __m128i vinput_zero_point0 = _mm_cvtsi32_si128(*((const int*) &quantization_params[0].zero_point));
    vinput_zero_point0 = _mm_shuffle_epi32(vinput_zero_point0, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128i vzero = _mm_setzero_si128();
    const __m128i vinit0 = _mm_mullo_epi32(vksum, vinput_zero_point0);
    __m128i vacc0x0 = _mm_blend_epi16(vinit0, vzero, 0xFC);
    __m128i vacc0x1 = _mm_blend_epi16(vinit0, vzero, 0xF3);
    __m128i vacc0x2 = _mm_blend_epi16(vinit0, vzero, 0xCF);
    __m128i vacc0x3 = _mm_blend_epi16(vinit0, vzero, 0x3F);
    w = (const int32_t*) w + 4;

    size_t k = kc;


    while (k >= 8 * sizeof(int8_t)) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
      a0 += 8;

      const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);

      const __m128i vxb0 = _mm_cvtepi8_epi16(vb0);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));

      const __m128i vxb1 = _mm_cvtepi8_epi16(vb1);

      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));

      const __m128i vxb2 = _mm_cvtepi8_epi16(vb2);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));

      const __m128i vxb3 = _mm_cvtepi8_epi16(vb3);

      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));

      w = (const int8_t*) w + 32;
      k -= 8 * sizeof(int8_t);
    }

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);

    __m128 vout0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 vinput_scale0 = _mm_load1_ps(&quantization_params[0].inv_scale);

    vout0x0123 = _mm_mul_ps(vout0x0123, vinput_scale0);

    const __m128 vfilter_output_scale0123 = _mm_load_ps((const float*) w);
    vout0x0123 = _mm_mul_ps(vout0x0123, vfilter_output_scale0123);

    const __m128 vbias0123 = _mm_load_ps((const float*) w + 4);
    w = (const float*) w + 8;
    vout0x0123 = _mm_add_ps(vout0x0123, vbias0123);

    vout0x0123 = _mm_max_ps(vout0x0123, vmin);

    vout0x0123 = _mm_min_ps(vout0x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c0, vout0x0123);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vout0x0123);
        vout0x0123 = _mm_unpackhi_ps(vout0x0123, vout0x0123);
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vout0x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse41_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    const __m128i vksum = _mm_load_si128((const __m128i*) w);
    const __m128i vinput_zero_point01 = _mm_loadu_si128((const __m128i*) &quantization_params[0]);
    const __m128i vinput_zero_point0 = _mm_shuffle_epi32(vinput_zero_point01, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128i vinput_zero_point1 = _mm_shuffle_epi32(vinput_zero_point01, _MM_SHUFFLE(2, 2, 2, 2));
    __m128i vinput_zero_point2 = _mm_cvtsi32_si128(*((const int*) &quantization_params[2].zero_point));
    vinput_zero_point2 = _mm_shuffle_epi32(vinput_zero_point2, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128i vzero = _mm_setzero_si128();
    const __m128i vinit0 = _mm_mullo_epi32(vksum, vinput_zero_point0);
    const __m128i vinit1 = _mm_mullo_epi32(vksum, vinput_zero_point1);
    const __m128i vinit2 = _mm_mullo_epi32(vksum, vinput_zero_point2);
    __m128i vacc0x0 = _mm_blend_epi16(vinit0, vzero, 0xFC);
    __m128i vacc0x1 = _mm_blend_epi16(vinit0, vzero, 0xF3);
    __m128i vacc0x2 = _mm_blend_epi16(vinit0, vzero, 0xCF);
    __m128i vacc0x3 = _mm_blend_epi16(vinit0, vzero, 0x3F);
    __m128i vacc1x0 = _mm_blend_epi16(vinit1, vzero, 0xFC);
    __m128i vacc1x1 = _mm_blend_epi16(vinit1, vzero, 0xF3);
    __m128i vacc1x2 = _mm_blend_epi16(vinit1, vzero, 0xCF);
    __m128i vacc1x3 = _mm_blend_epi16(vinit1, vzero, 0x3F);
    __m128i vacc2x0 = _mm_blend_epi16(vinit2, vzero, 0xFC);
    __m128i vacc2x1 = _mm_blend_epi16(vinit2, vzero, 0xF3);
    __m128i vacc2x2 = _mm_blend_epi16(vinit2, vzero, 0xCF);
    __m128i vacc2x3 = _mm_blend_epi16(vinit2, vzero, 0x3F);
    w = (const int32_t*) w + 4;

    size_t k = kc;


    while (k >= 8 * sizeof(int8_t)) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
      a0 += 8;
      const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
      const __m128i vxa1 = _mm_cvtepi8_epi16(va1);
      a1 += 8;
      const __m128i va2 = _mm_loadl_epi64((const __m128i*) a2);
      const __m128i vxa2 = _mm_cvtepi8_epi16(va2);
      a2 += 8;

      const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);

      const __m128i vxb0 = _mm_cvtepi8_epi16(vb0);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
      vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2, vxb0));
      const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));

      const __m128i vxb1 = _mm_cvtepi8_epi16(vb1);

      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
      vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2, vxb1));
      const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));

      const __m128i vxb2 = _mm_cvtepi8_epi16(vb2);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
      vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2, vxb2));
      const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));

      const __m128i vxb3 = _mm_cvtepi8_epi16(vb3);

      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
      vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));
      vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2, vxb3));

      w = (const int8_t*) w + 32;
      k -= 8 * sizeof(int8_t);
    }

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);
    const __m128i vacc1x01 = _mm_hadd_epi32(vacc1x0, vacc1x1);
    const __m128i vacc1x23 = _mm_hadd_epi32(vacc1x2, vacc1x3);
    const __m128i vacc2x01 = _mm_hadd_epi32(vacc2x0, vacc2x1);
    const __m128i vacc2x23 = _mm_hadd_epi32(vacc2x2, vacc2x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
    __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);
    __m128i vacc2x0123 = _mm_hadd_epi32(vacc2x01, vacc2x23);

    __m128 vout0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vout1x0123 = _mm_cvtepi32_ps(vacc1x0123);
    __m128 vout2x0123 = _mm_cvtepi32_ps(vacc2x0123);

    const __m128i vinput_scale01 = _mm_loadu_si128((const __m128i*) &quantization_params[0]);
    const __m128 vinput_scale0 = _mm_castsi128_ps(_mm_shuffle_epi32(vinput_scale01, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128 vinput_scale1 = _mm_castsi128_ps(_mm_shuffle_epi32(vinput_scale01, _MM_SHUFFLE(3, 3, 3, 3)));
    const __m128 vinput_scale2 = _mm_load1_ps(&quantization_params[2].inv_scale);

    vout0x0123 = _mm_mul_ps(vout0x0123, vinput_scale0);
    vout1x0123 = _mm_mul_ps(vout1x0123, vinput_scale1);
    vout2x0123 = _mm_mul_ps(vout2x0123, vinput_scale2);

    const __m128 vfilter_output_scale0123 = _mm_load_ps((const float*) w);
    vout0x0123 = _mm_mul_ps(vout0x0123, vfilter_output_scale0123);
    vout1x0123 = _mm_mul_ps(vout1x0123, vfilter_output_scale0123);
    vout2x0123 = _mm_mul_ps(vout2x0123, vfilter_output_scale0123);

    const __m128 vbias0123 = _mm_load_ps((const float*) w + 4);
    w = (const float*) w + 8;
    vout0x0123 = _mm_add_ps(vout0x0123, vbias0123);
    vout1x0123 = _mm_add_ps(vout1x0123, vbias0123);
    vout2x0123 = _mm_add_ps(vout2x0123, vbias0123);

    vout0x0123 = _mm_max_ps(vout0x0123, vmin);
    vout1x0123 = _mm_max_ps(vout1x0123, vmin);
    vout2x0123 = _mm_max_ps(vout2x0123, vmin);

    vout0x0123 = _mm_min_ps(vout0x0123, vmax);
    vout1x0123 = _mm_min_ps(vout1x0123, vmax);
    vout2x0123 = _mm_min_ps(vout2x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c0, vout0x0123);
      _mm_storeu_ps(c1, vout1x0123);
      _mm_storeu_ps(c2, vout2x0123);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vout0x0123);
        vout0x0123 = _mm_unpackhi_ps(vout0x0123, vout0x0123);
        c0 += 2;
        _mm_storel_pi((__m64*) c1, vout1x0123);
        vout1x0123 = _mm_unpackhi_ps(vout1x0123, vout1x0123);
        c1 += 2;
        _mm_storel_pi((__m64*) c2, vout2x0123);
        vout2x0123 = _mm_unpackhi_ps(vout2x0123, vout2x0123);
        c2 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vout0x0123);
        _mm_store_ss(c1, vout1x0123);
        _mm_store_ss(c2, vout2x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c8__sse41_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  float* c0 = c;

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  __m128i vinput_zero_point = _mm_cvtsi32_si128(*((const int*) &quantization_params->zero_point));
  vinput_zero_point = _mm_shuffle_epi32(vinput_zero_point, _MM_SHUFFLE(0, 0, 0, 0));
  const __m128 vinput_scale = _mm_load1_ps(&quantization_params->inv_scale);
  do {
    const __m128i vksum = _mm_load_si128((const __m128i*) w);
    const __m128i vzero = _mm_setzero_si128();
    const __m128i vinit0 = _mm_mullo_epi32(vksum, vinput_zero_point);
    __m128i vacc0x0 = _mm_blend_epi16(vinit0, vzero, 0xFC);
    __m128i vacc0x1 = _mm_blend_epi16(vinit0, vzero, 0xF3);
    __m128i vacc0x2 = _mm_blend_epi16(vinit0, vzero, 0xCF);
    __m128i vacc0x3 = _mm_blend_epi16(vinit0, vzero, 0x3F);
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      a += 1;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
        a0 += 8;

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
        const __m128i vxb0 = _mm_cvtepi8_epi16(vb0);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));
        const __m128i vxb1 = _mm_cvtepi8_epi16(vb1);

        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vxb2 = _mm_cvtepi8_epi16(vb2);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));
        const __m128i vxb3 = _mm_cvtepi8_epi16(vb3);

        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));

        w = (const void*) ((const int8_t*) w + 32);
        k += 8 * sizeof(int8_t);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);

    __m128 vout0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    vout0x0123 = _mm_mul_ps(vout0x0123, vinput_scale);

    const __m128 vfilter_output_scale0123 = _mm_load_ps((const float*) w);
    vout0x0123 = _mm_mul_ps(vout0x0123, vfilter_output_scale0123);

    const __m128 vbias0123 = _mm_load_ps((const float*) w + 4);
    w = (const float*) w + 8;
    vout0x0123 = _mm_add_ps(vout0x0123, vbias0123);

    vout0x0123 = _mm_max_ps(vout0x0123, vmin);

    vout0x0123 = _mm_min_ps(vout0x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c0, vout0x0123);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vout0x0123);
        vout0x0123 = _mm_unpackhi_ps(vout0x0123, vout0x0123);
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vout0x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x4c8__sse41_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (3 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  __m128i vinput_zero_point = _mm_cvtsi32_si128(*((const int*) &quantization_params->zero_point));
  vinput_zero_point = _mm_shuffle_epi32(vinput_zero_point, _MM_SHUFFLE(0, 0, 0, 0));
  const __m128 vinput_scale = _mm_load1_ps(&quantization_params->inv_scale);
  do {
    const __m128i vksum = _mm_load_si128((const __m128i*) w);
    const __m128i vzero = _mm_setzero_si128();
    const __m128i vinit0 = _mm_mullo_epi32(vksum, vinput_zero_point);
    const __m128i vinit1 = _mm_mullo_epi32(vksum, vinput_zero_point);
    const __m128i vinit2 = _mm_mullo_epi32(vksum, vinput_zero_point);
    __m128i vacc0x0 = _mm_blend_epi16(vinit0, vzero, 0xFC);
    __m128i vacc0x1 = _mm_blend_epi16(vinit0, vzero, 0xF3);
    __m128i vacc0x2 = _mm_blend_epi16(vinit0, vzero, 0xCF);
    __m128i vacc0x3 = _mm_blend_epi16(vinit0, vzero, 0x3F);
    __m128i vacc1x0 = _mm_blend_epi16(vinit1, vzero, 0xFC);
    __m128i vacc1x1 = _mm_blend_epi16(vinit1, vzero, 0xF3);
    __m128i vacc1x2 = _mm_blend_epi16(vinit1, vzero, 0xCF);
    __m128i vacc1x3 = _mm_blend_epi16(vinit1, vzero, 0x3F);
    __m128i vacc2x0 = _mm_blend_epi16(vinit2, vzero, 0xFC);
    __m128i vacc2x1 = _mm_blend_epi16(vinit2, vzero, 0xF3);
    __m128i vacc2x2 = _mm_blend_epi16(vinit2, vzero, 0xCF);
    __m128i vacc2x3 = _mm_blend_epi16(vinit2, vzero, 0x3F);
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      } else {
        a1 = zero_data;
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      } else {
        a2 = zero_data;
      }
      a += 3;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
        a0 += 8;
        const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
        const __m128i vxa1 = _mm_cvtepi8_epi16(va1);
        a1 += 8;
        const __m128i va2 = _mm_loadl_epi64((const __m128i*) a2);
        const __m128i vxa2 = _mm_cvtepi8_epi16(va2);
        a2 += 8;

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
        const __m128i vxb0 = _mm_cvtepi8_epi16(vb0);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
        vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2, vxb0));
        const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));
        const __m128i vxb1 = _mm_cvtepi8_epi16(vb1);

        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
        vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2, vxb1));
        const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vxb2 = _mm_cvtepi8_epi16(vb2);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
        vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2, vxb2));
        const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));
        const __m128i vxb3 = _mm_cvtepi8_epi16(vb3);

        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
        vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));
        vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2, vxb3));

        w = (const void*) ((const int8_t*) w + 32);
        k += 8 * sizeof(int8_t);
      }
      p -= 3 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);
    const __m128i vacc1x01 = _mm_hadd_epi32(vacc1x0, vacc1x1);
    const __m128i vacc1x23 = _mm_hadd_epi32(vacc1x2, vacc1x3);
    const __m128i vacc2x01 = _mm_hadd_epi32(vacc2x0, vacc2x1);
    const __m128i vacc2x23 = _mm_hadd_epi32(vacc2x2, vacc2x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
    __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);
    __m128i vacc2x0123 = _mm_hadd_epi32(vacc2x01, vacc2x23);

    __m128 vout0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vout1x0123 = _mm_cvtepi32_ps(vacc1x0123);
    __m128 vout2x0123 = _mm_cvtepi32_ps(vacc2x0123);

    vout0x0123 = _mm_mul_ps(vout0x0123, vinput_scale);
    vout1x0123 = _mm_mul_ps(vout1x0123, vinput_scale);
    vout2x0123 = _mm_mul_ps(vout2x0123, vinput_scale);

    const __m128 vfilter_output_scale0123 = _mm_load_ps((const float*) w);
    vout0x0123 = _mm_mul_ps(vout0x0123, vfilter_output_scale0123);
    vout1x0123 = _mm_mul_ps(vout1x0123, vfilter_output_scale0123);
    vout2x0123 = _mm_mul_ps(vout2x0123, vfilter_output_scale0123);

    const __m128 vbias0123 = _mm_load_ps((const float*) w + 4);
    w = (const float*) w + 8;
    vout0x0123 = _mm_add_ps(vout0x0123, vbias0123);
    vout1x0123 = _mm_add_ps(vout1x0123, vbias0123);
    vout2x0123 = _mm_add_ps(vout2x0123, vbias0123);

    vout0x0123 = _mm_max_ps(vout0x0123, vmin);
    vout1x0123 = _mm_max_ps(vout1x0123, vmin);
    vout2x0123 = _mm_max_ps(vout2x0123, vmin);

    vout0x0123 = _mm_min_ps(vout0x0123, vmax);
    vout1x0123 = _mm_min_ps(vout1x0123, vmax);
    vout2x0123 = _mm_min_ps(vout2x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c2, vout2x0123);
      _mm_storeu_ps(c1, vout1x0123);
      _mm_storeu_ps(c0, vout0x0123);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c2, vout2x0123);
        vout2x0123 = _mm_unpackhi_ps(vout2x0123, vout2x0123);
        c2 += 2;
        _mm_storel_pi((__m64*) c1, vout1x0123);
        vout1x0123 = _mm_unpackhi_ps(vout1x0123, vout1x0123);
        c1 += 2;
        _mm_storel_pi((__m64*) c0, vout0x0123);
        vout0x0123 = _mm_unpackhi_ps(vout0x0123, vout0x0123);
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c2, vout2x0123);
        _mm_store_ss(c1, vout1x0123);
        _mm_store_ss(c0, vout0x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs16_qs8_vcvt_ukernel__sse41_u16(
    size_t batch,
    const int16_t* input,
    int8_t* output,
    const union xnn_qs16_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);
  XNN_ALIGN(16) static const uint8_t shuffle01[16] = {
    0x80, 0x80, 0, 1, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 2, 3, 0x80, 0x80, 0x80, 0x80,
  };
  XNN_ALIGN(16) static const uint8_t shuffle23[16] = {
    0x80, 0x80, 4, 5, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 6, 7, 0x80, 0x80, 0x80, 0x80,
  };
  XNN_ALIGN(16) static const uint8_t shuffle45[16] = {
    0x80, 0x80, 8, 9, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 10, 11, 0x80, 0x80, 0x80, 0x80,
  };
  XNN_ALIGN(16) static const uint8_t shuffle67[16] = {
    0x80, 0x80, 12, 13, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 14, 15, 0x80, 0x80, 0x80, 0x80,
  };

  const __m128i vmultiplier = _mm_set1_epi32(params->scalar.multiplier);
  const __m128i vbias = _mm_set1_epi64x((int64_t) ((uint64_t) params->scalar.output_zero_point << 32) + INT64_C(0x80000000));
  XNN_FORCE_REALIZATION(vmultiplier);
  XNN_FORCE_REALIZATION(vbias);
  const __m128i vshuffle01 = _mm_load_si128((const __m128i*) shuffle01);
  const __m128i vshuffle23 = _mm_load_si128((const __m128i*) shuffle23);

  const __m128i vshuffle45 = _mm_load_si128((const __m128i*) shuffle45);
  const __m128i vshuffle67 = _mm_load_si128((const __m128i*) shuffle67);
  for (; batch >= 16 * sizeof(int16_t); batch -= 16 * sizeof(int16_t)) {
    const __m128i vx0 = _mm_loadu_si128((const __m128i*) input); input += 8;
    const __m128i vx2 = _mm_loadu_si128((const __m128i*) input); input += 8;

    // Move int16 to upper part of int32
    __m128i vacc0lo   = _mm_shuffle_epi8(vx0, vshuffle01);
    __m128i vacc0hi   = _mm_shuffle_epi8(vx0, vshuffle23);
    __m128i vacc1lo = _mm_shuffle_epi8(vx0, vshuffle45);
    __m128i vacc1hi = _mm_shuffle_epi8(vx0, vshuffle67);
    __m128i vacc2lo   = _mm_shuffle_epi8(vx2, vshuffle01);
    __m128i vacc2hi   = _mm_shuffle_epi8(vx2, vshuffle23);
    __m128i vacc3lo = _mm_shuffle_epi8(vx2, vshuffle45);
    __m128i vacc3hi = _mm_shuffle_epi8(vx2, vshuffle67);

    vacc0lo = _mm_mul_epi32(vacc0lo, vmultiplier);
    vacc0hi = _mm_mul_epi32(vacc0hi, vmultiplier);
    vacc1lo = _mm_mul_epi32(vacc1lo, vmultiplier);
    vacc1hi = _mm_mul_epi32(vacc1hi, vmultiplier);
    vacc2lo = _mm_mul_epi32(vacc2lo, vmultiplier);
    vacc2hi = _mm_mul_epi32(vacc2hi, vmultiplier);
    vacc3lo = _mm_mul_epi32(vacc3lo, vmultiplier);
    vacc3hi = _mm_mul_epi32(vacc3hi, vmultiplier);

    vacc0lo = _mm_add_epi64(vacc0lo, vbias);
    vacc0hi = _mm_add_epi64(vacc0hi, vbias);
    vacc1lo = _mm_add_epi64(vacc1lo, vbias);
    vacc1hi = _mm_add_epi64(vacc1hi, vbias);
    vacc2lo = _mm_add_epi64(vacc2lo, vbias);
    vacc2hi = _mm_add_epi64(vacc2hi, vbias);
    vacc3lo = _mm_add_epi64(vacc3lo, vbias);
    vacc3hi = _mm_add_epi64(vacc3hi, vbias);

    __m128i vacc0 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacc0lo), _mm_castsi128_ps(vacc0hi), _MM_SHUFFLE(3, 1, 3, 1)));
    __m128i vacc1 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacc1lo), _mm_castsi128_ps(vacc1hi), _MM_SHUFFLE(3, 1, 3, 1)));
    __m128i vacc2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacc2lo), _mm_castsi128_ps(vacc2hi), _MM_SHUFFLE(3, 1, 3, 1)));
    __m128i vacc3 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacc3lo), _mm_castsi128_ps(vacc3hi), _MM_SHUFFLE(3, 1, 3, 1)));

    // Pack 8 ints into 8 shorts
    vacc0 = _mm_packs_epi32(vacc0, vacc1);
    vacc2 = _mm_packs_epi32(vacc2, vacc3);

    // Pack 16 shorts into 16 bytes
    const __m128i vy0 = _mm_packs_epi16(vacc0, vacc2);

    _mm_storeu_si128((__m128i*) output, vy0); output += 16;
  }

  for (; batch >= 4 * sizeof(int16_t); batch -= 4 * sizeof(int16_t)) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) input); input += 4;
    __m128i vacclo = _mm_shuffle_epi8(vx, vshuffle01);
    __m128i vacchi = _mm_shuffle_epi8(vx, vshuffle23);
    vacclo = _mm_mul_epi32(vacclo, vmultiplier);
    vacchi = _mm_mul_epi32(vacchi, vmultiplier);
    vacclo = _mm_add_epi64(vacclo, vbias);
    vacchi = _mm_add_epi64(vacchi, vbias);
    __m128i vacc = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacclo), _mm_castsi128_ps(vacchi), _MM_SHUFFLE(3, 1, 3, 1)));
    vacc = _mm_packs_epi32(vacc, vacc);
    const __m128i vy = _mm_packs_epi16(vacc, vacc);

    _mm_storeu_si32(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int16_t));
    assert(batch <= 3 * sizeof(int16_t));

    const __m128i vx = _mm_loadu_si128((const __m128i*) input);
    __m128i vacclo = _mm_shuffle_epi8(vx, vshuffle01);
    __m128i vacchi = _mm_shuffle_epi8(vx, vshuffle23);
    vacclo = _mm_mul_epi32(vacclo, vmultiplier);
    vacchi = _mm_mul_epi32(vacchi, vmultiplier);
    vacclo = _mm_add_epi64(vacclo, vbias);
    vacchi = _mm_add_epi64(vacchi, vbias);
    __m128i vacc = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacclo), _mm_castsi128_ps(vacchi), _MM_SHUFFLE(3, 1, 3, 1)));
    vacc = _mm_packs_epi32(vacc, vacc);
    __m128i vy = _mm_packs_epi16(vacc, vacc);

    if (batch & (2 * sizeof(int16_t))) {
      _mm_storeu_si16(output, vy);
      vy = _mm_srli_epi32(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(int16_t))) {
      *output = (int8_t) _mm_extract_epi8(vy, 0);
    }
  }
}

void xnn_qs8_dwconv_minmax_fp32_ukernel_25p8c__sse41_mul16_add16(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    const int8_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const int8_t*) ((uintptr_t) i9 + input_offset);
    }
    const int8_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const int8_t*) ((uintptr_t) i10 + input_offset);
    }
    const int8_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const int8_t*) ((uintptr_t) i11 + input_offset);
    }
    const int8_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const int8_t*) ((uintptr_t) i12 + input_offset);
    }
    const int8_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const int8_t*) ((uintptr_t) i13 + input_offset);
    }
    const int8_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const int8_t*) ((uintptr_t) i14 + input_offset);
    }
    const int8_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const int8_t*) ((uintptr_t) i15 + input_offset);
    }
    const int8_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const int8_t*) ((uintptr_t) i16 + input_offset);
    }
    const int8_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const int8_t*) ((uintptr_t) i17 + input_offset);
    }
    const int8_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const int8_t*) ((uintptr_t) i18 + input_offset);
    }
    const int8_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const int8_t*) ((uintptr_t) i19 + input_offset);
    }
    const int8_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const int8_t*) ((uintptr_t) i20 + input_offset);
    }
    const int8_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const int8_t*) ((uintptr_t) i21 + input_offset);
    }
    const int8_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const int8_t*) ((uintptr_t) i22 + input_offset);
    }
    const int8_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const int8_t*) ((uintptr_t) i23 + input_offset);
    }
    const int8_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const int8_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 8; c -= 8) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
      i0 += 8;


      __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);


      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
      i1 += 8;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi1x01234567, vxk1x01234567));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
      i2 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);


      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t)));
      const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);
      i3 += 8;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi3x01234567, vxk3x01234567));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(int8_t)));
      const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);
      i4 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);


      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(int8_t)));
      const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);
      i5 += 8;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi5x01234567, vxk5x01234567));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(int8_t)));
      const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);
      i6 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);


      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(int8_t)));
      const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);
      i7 += 8;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi7x01234567, vxk7x01234567));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(int8_t)));
      const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);
      i8 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);


      const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
      const __m128i vxi9x01234567 = _mm_cvtepi8_epi16(vi9x01234567);
      const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(int8_t)));
      const __m128i vxk9x01234567 = _mm_cvtepi8_epi16(vk9x01234567);
      i9 += 8;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi9x01234567, vxk9x01234567));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
      const __m128i vxi10x01234567 = _mm_cvtepi8_epi16(vi10x01234567);
      const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 80 * sizeof(int8_t)));
      const __m128i vxk10x01234567 = _mm_cvtepi8_epi16(vk10x01234567);
      i10 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);


      const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
      const __m128i vxi11x01234567 = _mm_cvtepi8_epi16(vi11x01234567);
      const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 88 * sizeof(int8_t)));
      const __m128i vxk11x01234567 = _mm_cvtepi8_epi16(vk11x01234567);
      i11 += 8;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi11x01234567, vxk11x01234567));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
      const __m128i vxi12x01234567 = _mm_cvtepi8_epi16(vi12x01234567);
      const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 96 * sizeof(int8_t)));
      const __m128i vxk12x01234567 = _mm_cvtepi8_epi16(vk12x01234567);
      i12 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);


      const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
      const __m128i vxi13x01234567 = _mm_cvtepi8_epi16(vi13x01234567);
      const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 104 * sizeof(int8_t)));
      const __m128i vxk13x01234567 = _mm_cvtepi8_epi16(vk13x01234567);
      i13 += 8;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi13x01234567, vxk13x01234567));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
      const __m128i vxi14x01234567 = _mm_cvtepi8_epi16(vi14x01234567);
      const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 112 * sizeof(int8_t)));
      const __m128i vxk14x01234567 = _mm_cvtepi8_epi16(vk14x01234567);
      i14 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);


      const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
      const __m128i vxi15x01234567 = _mm_cvtepi8_epi16(vi15x01234567);
      const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 120 * sizeof(int8_t)));
      const __m128i vxk15x01234567 = _mm_cvtepi8_epi16(vk15x01234567);
      i15 += 8;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi15x01234567, vxk15x01234567));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
      const __m128i vxi16x01234567 = _mm_cvtepi8_epi16(vi16x01234567);
      const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 128 * sizeof(int8_t)));
      const __m128i vxk16x01234567 = _mm_cvtepi8_epi16(vk16x01234567);
      i16 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);


      const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
      const __m128i vxi17x01234567 = _mm_cvtepi8_epi16(vi17x01234567);
      const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 136 * sizeof(int8_t)));
      const __m128i vxk17x01234567 = _mm_cvtepi8_epi16(vk17x01234567);
      i17 += 8;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi17x01234567, vxk17x01234567));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
      const __m128i vxi18x01234567 = _mm_cvtepi8_epi16(vi18x01234567);
      const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 144 * sizeof(int8_t)));
      const __m128i vxk18x01234567 = _mm_cvtepi8_epi16(vk18x01234567);
      i18 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);


      const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
      const __m128i vxi19x01234567 = _mm_cvtepi8_epi16(vi19x01234567);
      const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 152 * sizeof(int8_t)));
      const __m128i vxk19x01234567 = _mm_cvtepi8_epi16(vk19x01234567);
      i19 += 8;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi19x01234567, vxk19x01234567));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
      const __m128i vxi20x01234567 = _mm_cvtepi8_epi16(vi20x01234567);
      const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 160 * sizeof(int8_t)));
      const __m128i vxk20x01234567 = _mm_cvtepi8_epi16(vk20x01234567);
      i20 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);


      const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
      const __m128i vxi21x01234567 = _mm_cvtepi8_epi16(vi21x01234567);
      const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 168 * sizeof(int8_t)));
      const __m128i vxk21x01234567 = _mm_cvtepi8_epi16(vk21x01234567);
      i21 += 8;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi21x01234567, vxk21x01234567));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
      const __m128i vxi22x01234567 = _mm_cvtepi8_epi16(vi22x01234567);
      const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 176 * sizeof(int8_t)));
      const __m128i vxk22x01234567 = _mm_cvtepi8_epi16(vk22x01234567);
      i22 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);


      const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
      const __m128i vxi23x01234567 = _mm_cvtepi8_epi16(vi23x01234567);
      const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 184 * sizeof(int8_t)));
      const __m128i vxk23x01234567 = _mm_cvtepi8_epi16(vk23x01234567);
      i23 += 8;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi23x01234567, vxk23x01234567));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
      const __m128i vxi24x01234567 = _mm_cvtepi8_epi16(vi24x01234567);
      const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 192 * sizeof(int8_t)));
      const __m128i vxk24x01234567 = _mm_cvtepi8_epi16(vk24x01234567);
      i24 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      w = (const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 200 * sizeof(int8_t));

      __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

      const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
      vscaled0123 = _mm_mul_ps(vscaled0123, vscale);
      vscaled4567 = _mm_mul_ps(vscaled4567, vscale);

      const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
      vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
      vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vscaled0123);
      vacc4567 = _mm_cvtps_epi32(vscaled4567);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse4.output_min);
      vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);

      _mm_storel_epi64((__m128i*) output, vout0123456701234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t)));
        const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);


        __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);


        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t)));
        const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi1x01234567, vxk1x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t)));
        const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);


        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t)));
        const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi3x01234567, vxk3x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(int8_t)));
        const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);


        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(int8_t)));
        const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi5x01234567, vxk5x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(int8_t)));
        const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);


        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(int8_t)));
        const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi7x01234567, vxk7x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(int8_t)));
        const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);


        const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
        const __m128i vxi9x01234567 = _mm_cvtepi8_epi16(vi9x01234567);
        const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(int8_t)));
        const __m128i vxk9x01234567 = _mm_cvtepi8_epi16(vk9x01234567);


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi9x01234567, vxk9x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
        const __m128i vxi10x01234567 = _mm_cvtepi8_epi16(vi10x01234567);
        const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 80 * sizeof(int8_t)));
        const __m128i vxk10x01234567 = _mm_cvtepi8_epi16(vk10x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);


        const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
        const __m128i vxi11x01234567 = _mm_cvtepi8_epi16(vi11x01234567);
        const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 88 * sizeof(int8_t)));
        const __m128i vxk11x01234567 = _mm_cvtepi8_epi16(vk11x01234567);


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi11x01234567, vxk11x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
        const __m128i vxi12x01234567 = _mm_cvtepi8_epi16(vi12x01234567);
        const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 96 * sizeof(int8_t)));
        const __m128i vxk12x01234567 = _mm_cvtepi8_epi16(vk12x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);


        const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
        const __m128i vxi13x01234567 = _mm_cvtepi8_epi16(vi13x01234567);
        const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 104 * sizeof(int8_t)));
        const __m128i vxk13x01234567 = _mm_cvtepi8_epi16(vk13x01234567);


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi13x01234567, vxk13x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
        const __m128i vxi14x01234567 = _mm_cvtepi8_epi16(vi14x01234567);
        const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 112 * sizeof(int8_t)));
        const __m128i vxk14x01234567 = _mm_cvtepi8_epi16(vk14x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);


        const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
        const __m128i vxi15x01234567 = _mm_cvtepi8_epi16(vi15x01234567);
        const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 120 * sizeof(int8_t)));
        const __m128i vxk15x01234567 = _mm_cvtepi8_epi16(vk15x01234567);


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi15x01234567, vxk15x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
        const __m128i vxi16x01234567 = _mm_cvtepi8_epi16(vi16x01234567);
        const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 128 * sizeof(int8_t)));
        const __m128i vxk16x01234567 = _mm_cvtepi8_epi16(vk16x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);


        const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
        const __m128i vxi17x01234567 = _mm_cvtepi8_epi16(vi17x01234567);
        const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 136 * sizeof(int8_t)));
        const __m128i vxk17x01234567 = _mm_cvtepi8_epi16(vk17x01234567);


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi17x01234567, vxk17x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
        const __m128i vxi18x01234567 = _mm_cvtepi8_epi16(vi18x01234567);
        const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 144 * sizeof(int8_t)));
        const __m128i vxk18x01234567 = _mm_cvtepi8_epi16(vk18x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);


        const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
        const __m128i vxi19x01234567 = _mm_cvtepi8_epi16(vi19x01234567);
        const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 152 * sizeof(int8_t)));
        const __m128i vxk19x01234567 = _mm_cvtepi8_epi16(vk19x01234567);


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi19x01234567, vxk19x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
        const __m128i vxi20x01234567 = _mm_cvtepi8_epi16(vi20x01234567);
        const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 160 * sizeof(int8_t)));
        const __m128i vxk20x01234567 = _mm_cvtepi8_epi16(vk20x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);


        const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
        const __m128i vxi21x01234567 = _mm_cvtepi8_epi16(vi21x01234567);
        const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 168 * sizeof(int8_t)));
        const __m128i vxk21x01234567 = _mm_cvtepi8_epi16(vk21x01234567);


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi21x01234567, vxk21x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
        const __m128i vxi22x01234567 = _mm_cvtepi8_epi16(vi22x01234567);
        const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 176 * sizeof(int8_t)));
        const __m128i vxk22x01234567 = _mm_cvtepi8_epi16(vk22x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);


        const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
        const __m128i vxi23x01234567 = _mm_cvtepi8_epi16(vi23x01234567);
        const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 184 * sizeof(int8_t)));
        const __m128i vxk23x01234567 = _mm_cvtepi8_epi16(vk23x01234567);


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi23x01234567, vxk23x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
        const __m128i vxi24x01234567 = _mm_cvtepi8_epi16(vi24x01234567);
        const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 192 * sizeof(int8_t)));
        const __m128i vxk24x01234567 = _mm_cvtepi8_epi16(vk24x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));


        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

        const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
        vscaled0123 = _mm_mul_ps(vscaled0123, vscale);
        vscaled4567 = _mm_mul_ps(vscaled4567, vscale);

        const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
        vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
        vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

        vacc0123 = _mm_cvtps_epi32(vscaled0123);
        vacc4567 = _mm_cvtps_epi32(vscaled4567);


        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

        vout0123456701234567 = _mm_max_epi8(vout0123456701234567, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

        if (c & 4) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (c & 2) {
          unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (c & 1) {
          *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
          output += 1;
        }
      }
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_dwconv_minmax_fp32_ukernel_9p8c__sse41_mul16_add16(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 8; c -= 8) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
      i0 += 8;


      __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);


      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
      i1 += 8;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi1x01234567, vxk1x01234567));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
      i2 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);


      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t)));
      const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);
      i3 += 8;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi3x01234567, vxk3x01234567));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(int8_t)));
      const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);
      i4 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);


      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(int8_t)));
      const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);
      i5 += 8;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi5x01234567, vxk5x01234567));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(int8_t)));
      const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);
      i6 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);


      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(int8_t)));
      const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);
      i7 += 8;


      vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi7x01234567, vxk7x01234567));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(int8_t)));
      const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);
      i8 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      w = (const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(int8_t));

      __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

      const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
      vscaled0123 = _mm_mul_ps(vscaled0123, vscale);
      vscaled4567 = _mm_mul_ps(vscaled4567, vscale);

      const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
      vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
      vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vscaled0123);
      vacc4567 = _mm_cvtps_epi32(vscaled4567);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse4.output_min);
      vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);

      _mm_storel_epi64((__m128i*) output, vout0123456701234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t)));
        const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);


        __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);


        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t)));
        const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi1x01234567, vxk1x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t)));
        const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);


        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t)));
        const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi3x01234567, vxk3x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(int8_t)));
        const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);


        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(int8_t)));
        const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi5x01234567, vxk5x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(int8_t)));
        const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);


        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(int8_t)));
        const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);


        vprod01234567 = _mm_add_epi16(vprod01234567, _mm_mullo_epi16(vxi7x01234567, vxk7x01234567));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(int8_t)));
        const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));


        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

        const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
        vscaled0123 = _mm_mul_ps(vscaled0123, vscale);
        vscaled4567 = _mm_mul_ps(vscaled4567, vscale);

        const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
        vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
        vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

        vacc0123 = _mm_cvtps_epi32(vscaled0123);
        vacc4567 = _mm_cvtps_epi32(vscaled4567);


        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

        vout0123456701234567 = _mm_max_epi8(vout0123456701234567, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

        if (c & 4) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (c & 2) {
          unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (c & 1) {
          *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
          output += 1;
        }
      }
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_f32_vcvt_ukernel__sse41_u16(
    size_t batch,
    const int8_t* input,
    float* output,
    const union xnn_qs8_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vzero_point = _mm_set1_epi32(params->scalar.zero_point);
  const __m128 vscale = _mm_set1_ps(params->scalar.scale);
  XNN_FORCE_REALIZATION(vzero_point);
  XNN_FORCE_REALIZATION(vscale);
  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    __m128i vx0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input)));
    __m128i vx4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input + 4)));
    __m128i vx89AB = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input + 8)));
    __m128i vxCDEF = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input + 12)));
    input += 16;

    vx0123 = _mm_sub_epi32(vx0123, vzero_point);
    vx4567 = _mm_sub_epi32(vx4567, vzero_point);
    vx89AB = _mm_sub_epi32(vx89AB, vzero_point);
    vxCDEF = _mm_sub_epi32(vxCDEF, vzero_point);

    __m128 vy0123 = _mm_cvtepi32_ps(vx0123);
    __m128 vy4567 = _mm_cvtepi32_ps(vx4567);
    __m128 vy89AB = _mm_cvtepi32_ps(vx89AB);
    __m128 vyCDEF = _mm_cvtepi32_ps(vxCDEF);

    vy0123 = _mm_mul_ps(vy0123, vscale);
    vy4567 = _mm_mul_ps(vy4567, vscale);
    vy89AB = _mm_mul_ps(vy89AB, vscale);
    vyCDEF = _mm_mul_ps(vyCDEF, vscale);

    _mm_storeu_ps(output, vy0123);
    _mm_storeu_ps(output + 4, vy4567);
    _mm_storeu_ps(output + 8, vy89AB);
    _mm_storeu_ps(output + 12, vyCDEF);
    output += 16;
  }
  for (; batch >= 4 * sizeof(int8_t); batch -= 4 * sizeof(int8_t)) {
    __m128i vx = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input)));
    vx = _mm_sub_epi32(vx, vzero_point);
    input += 4;

    __m128 vy = _mm_cvtepi32_ps(vx);
    vy = _mm_mul_ps(vy, vscale);

    _mm_storeu_ps(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 3 * sizeof(int8_t));

    __m128i vx = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input)));
    vx = _mm_sub_epi32(vx, vzero_point);

    __m128 vy = _mm_cvtepi32_ps(vx);
    vy = _mm_mul_ps(vy, vscale);

    if (batch & (2 * sizeof(int8_t))) {
      _mm_storel_pi((__m64*) output, vy);
      vy = _mm_movehl_ps(vy, vy);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      _mm_store_ss(output, vy);
    }
  }
}

void xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int32_t* buffer,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows > 7);
  assert(channels != 0);

  const int8_t* i0 = input;
  const int8_t* i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);
  const int8_t* i2 = (const int8_t*) ((uintptr_t) i1 + input_stride);
  const int8_t* i3 = (const int8_t*) ((uintptr_t) i2 + input_stride);
  const int8_t* i4 = (const int8_t*) ((uintptr_t) i3 + input_stride);
  const int8_t* i5 = (const int8_t*) ((uintptr_t) i4 + input_stride);
  const int8_t* i6 = (const int8_t*) ((uintptr_t) i5 + input_stride);
  const size_t input_increment = 7 * input_stride - round_up_po2(channels, 8) * sizeof(int8_t);

  const __m128i vinit_bias = _mm_load_si128((const __m128i*) params->fp32_sse4.init_bias);
  int32_t* b = buffer;
  size_t c = channels;
  for (; c != 0; c = doz(c, 8)) {
    const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i0));
    i0 += 8;
    const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i1));
    i1 += 8;

    __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
    const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i2));
    i2 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
    const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i3));
    i3 += 8;
    vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
    const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i4));
    i4 += 8;
    vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
    const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i5));
    i5 += 8;
    vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
    const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i6));
    i6 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

    __m128i vacc0123 = _mm_cvtepi16_epi32(vacc01234567);
    __m128i vacc4567 = _mm_srai_epi32(_mm_unpackhi_epi16(vacc01234567, vacc01234567), 16);

    vacc0123 = _mm_add_epi32(vacc0123, vinit_bias);
    vacc4567 = _mm_add_epi32(vacc4567, vinit_bias);

    _mm_store_si128((__m128i*) b, vacc0123);
    _mm_store_si128((__m128i*) (b + 4), vacc4567);
    b += 8;
  }

  for (rows -= 7; rows > 7; rows -= 7) {
    i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
    i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
    i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
    i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
    i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
    i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
    i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);

    int32_t* b = buffer;
    size_t c = channels;
    for (; c != 0; c = doz(c, 8)) {
      const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i0));
      i0 += 8;
      const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i1));
      i1 += 8;

      __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
      const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i2));
      i2 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
      const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i3));
      i3 += 8;
      vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
      const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i4));
      i4 += 8;
      vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
      const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i5));
      i5 += 8;
      vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
      const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i6));
      i6 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

      __m128i vacc0123 = _mm_cvtepi16_epi32(vacc01234567);
      __m128i vacc4567 = _mm_srai_epi32(_mm_unpackhi_epi16(vacc01234567, vacc01234567), 16);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_load_si128((const __m128i*) b));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_load_si128((const __m128i*) (b + 4)));

      _mm_store_si128((__m128i*) b, vacc0123);
      _mm_store_si128((__m128i*) (b + 4), vacc4567);
      b += 8;
    }
  }

  i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
  i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
  const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse4.output_min);
  for (; channels >= 8; channels -= 8) {
    const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i0));
    i0 += 8;
    const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i1));
    i1 += 8;

    __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
    const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i2));
    i2 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
    const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i3));
    i3 += 8;
    vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
    const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i4));
    i4 += 8;
    vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
    const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i5));
    i5 += 8;
    vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
    const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i6));
    i6 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

    __m128i vacc0123 = _mm_cvtepi16_epi32(vacc01234567);
    __m128i vacc4567 = _mm_srai_epi32(_mm_unpackhi_epi16(vacc01234567, vacc01234567), 16);

    vacc0123 = _mm_add_epi32(vacc0123, _mm_load_si128((const __m128i*) buffer));
    vacc4567 = _mm_add_epi32(vacc4567, _mm_load_si128((const __m128i*) (buffer + 4)));
    buffer += 8;

    __m128 vfpacc0123 = _mm_cvtepi32_ps(vacc0123);
    __m128 vfpacc4567 = _mm_cvtepi32_ps(vacc4567);

    vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
    vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

    vfpacc0123 = _mm_min_ps(vfpacc0123, voutput_max_less_zero_point);
    vfpacc4567 = _mm_min_ps(vfpacc4567, voutput_max_less_zero_point);

    vacc0123 = _mm_cvtps_epi32(vfpacc0123);
    vacc4567 = _mm_cvtps_epi32(vfpacc4567);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

    __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

    vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    {
      const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i0));
      i0 += 8;
      const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i1));
      i1 += 8;

      __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
      const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i2));
      i2 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
      const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i3));
      i3 += 8;
      vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
      const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i4));
      i4 += 8;
      vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
      const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i5));
      i5 += 8;
      vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
      const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i6));
      i6 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

      __m128i vacc0123 = _mm_cvtepi16_epi32(vacc01234567);
      __m128i vacc4567 = _mm_srai_epi32(_mm_unpackhi_epi16(vacc01234567, vacc01234567), 16);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_load_si128((const __m128i*) buffer));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_load_si128((const __m128i*) (buffer + 4)));
      buffer += 8;

      __m128 vfpacc0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vfpacc4567 = _mm_cvtepi32_ps(vacc4567);

      vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
      vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

      vfpacc0123 = _mm_min_ps(vfpacc0123, voutput_max_less_zero_point);
      vfpacc4567 = _mm_min_ps(vfpacc4567, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vfpacc0123);
      vacc4567 = _mm_cvtps_epi32(vfpacc4567);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);

      if (channels & 4) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (channels & 2) {
        unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (channels & 1) {
        *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
      }
    }
  }
}

void xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c8(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const int8_t* i0 = input;
  const int8_t* i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  const int8_t* i2 = (const int8_t*) ((uintptr_t) i1 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  const int8_t* i3 = (const int8_t*) ((uintptr_t) i2 + input_stride);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  const int8_t* i4 = (const int8_t*) ((uintptr_t) i3 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  const int8_t* i5 = (const int8_t*) ((uintptr_t) i4 + input_stride);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  const int8_t* i6 = (const int8_t*) ((uintptr_t) i5 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const __m128i vinit_bias = _mm_load_si128((const __m128i*) params->fp32_sse4.init_bias);
  const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
  const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse4.output_min);
  for (; channels >= 8; channels -= 8) {
    const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i0));
    i0 += 8;
    const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i1));
    i1 += 8;

    __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
    const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i2));
    i2 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
    const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i3));
    i3 += 8;
    vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
    const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i4));
    i4 += 8;
    vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
    const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i5));
    i5 += 8;
    vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
    const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i6));
    i6 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

    __m128i vacc0123 = _mm_cvtepi16_epi32(vacc01234567);
    __m128i vacc4567 = _mm_srai_epi32(_mm_unpackhi_epi16(vacc01234567, vacc01234567), 16);

    vacc0123 = _mm_add_epi32(vacc0123, vinit_bias);
    vacc4567 = _mm_add_epi32(vacc4567, vinit_bias);

    __m128 vfpacc0123 = _mm_cvtepi32_ps(vacc0123);
    __m128 vfpacc4567 = _mm_cvtepi32_ps(vacc4567);

    vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
    vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

    vfpacc0123 = _mm_min_ps(vfpacc0123, voutput_max_less_zero_point);
    vfpacc4567 = _mm_min_ps(vfpacc4567, voutput_max_less_zero_point);

    vacc0123 = _mm_cvtps_epi32(vfpacc0123);
    vacc4567 = _mm_cvtps_epi32(vfpacc4567);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

    __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

    vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    {
      const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i0));
      i0 += 8;
      const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i1));
      i1 += 8;

      __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
      const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i2));
      i2 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
      const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i3));
      i3 += 8;
      vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
      const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i4));
      i4 += 8;
      vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
      const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i5));
      i5 += 8;
      vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
      const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i6));
      i6 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

      __m128i vacc0123 = _mm_cvtepi16_epi32(vacc01234567);
      __m128i vacc4567 = _mm_srai_epi32(_mm_unpackhi_epi16(vacc01234567, vacc01234567), 16);

      vacc0123 = _mm_add_epi32(vacc0123, vinit_bias);
      vacc4567 = _mm_add_epi32(vacc4567, vinit_bias);

      __m128 vfpacc0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vfpacc4567 = _mm_cvtepi32_ps(vacc4567);

      vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
      vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

      vfpacc0123 = _mm_min_ps(vfpacc0123, voutput_max_less_zero_point);
      vfpacc4567 = _mm_min_ps(vfpacc4567, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vfpacc0123);
      vacc4567 = _mm_cvtps_epi32(vfpacc4567);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);

      if (channels & 4) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (channels & 2) {
        unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (channels & 1) {
        *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
      }
    }
  }
}

void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__sse41_mul16(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    const int8_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const int8_t*) ((uintptr_t) i9 + input_offset);
    }
    const int8_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const int8_t*) ((uintptr_t) i10 + input_offset);
    }
    const int8_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const int8_t*) ((uintptr_t) i11 + input_offset);
    }
    const int8_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const int8_t*) ((uintptr_t) i12 + input_offset);
    }
    const int8_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const int8_t*) ((uintptr_t) i13 + input_offset);
    }
    const int8_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const int8_t*) ((uintptr_t) i14 + input_offset);
    }
    const int8_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const int8_t*) ((uintptr_t) i15 + input_offset);
    }
    const int8_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const int8_t*) ((uintptr_t) i16 + input_offset);
    }
    const int8_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const int8_t*) ((uintptr_t) i17 + input_offset);
    }
    const int8_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const int8_t*) ((uintptr_t) i18 + input_offset);
    }
    const int8_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const int8_t*) ((uintptr_t) i19 + input_offset);
    }
    const int8_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const int8_t*) ((uintptr_t) i20 + input_offset);
    }
    const int8_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const int8_t*) ((uintptr_t) i21 + input_offset);
    }
    const int8_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const int8_t*) ((uintptr_t) i22 + input_offset);
    }
    const int8_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const int8_t*) ((uintptr_t) i23 + input_offset);
    }
    const int8_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const int8_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 8; c -= 8) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
      i0 += 8;


      __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
      i1 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
      i2 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t)));
      const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);
      i3 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(int8_t)));
      const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);
      i4 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(int8_t)));
      const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);
      i5 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(int8_t)));
      const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);
      i6 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(int8_t)));
      const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);
      i7 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(int8_t)));
      const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);
      i8 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
      const __m128i vxi9x01234567 = _mm_cvtepi8_epi16(vi9x01234567);
      const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(int8_t)));
      const __m128i vxk9x01234567 = _mm_cvtepi8_epi16(vk9x01234567);
      i9 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi9x01234567, vxk9x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
      const __m128i vxi10x01234567 = _mm_cvtepi8_epi16(vi10x01234567);
      const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 80 * sizeof(int8_t)));
      const __m128i vxk10x01234567 = _mm_cvtepi8_epi16(vk10x01234567);
      i10 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
      const __m128i vxi11x01234567 = _mm_cvtepi8_epi16(vi11x01234567);
      const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 88 * sizeof(int8_t)));
      const __m128i vxk11x01234567 = _mm_cvtepi8_epi16(vk11x01234567);
      i11 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi11x01234567, vxk11x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
      const __m128i vxi12x01234567 = _mm_cvtepi8_epi16(vi12x01234567);
      const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 96 * sizeof(int8_t)));
      const __m128i vxk12x01234567 = _mm_cvtepi8_epi16(vk12x01234567);
      i12 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
      const __m128i vxi13x01234567 = _mm_cvtepi8_epi16(vi13x01234567);
      const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 104 * sizeof(int8_t)));
      const __m128i vxk13x01234567 = _mm_cvtepi8_epi16(vk13x01234567);
      i13 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi13x01234567, vxk13x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
      const __m128i vxi14x01234567 = _mm_cvtepi8_epi16(vi14x01234567);
      const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 112 * sizeof(int8_t)));
      const __m128i vxk14x01234567 = _mm_cvtepi8_epi16(vk14x01234567);
      i14 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
      const __m128i vxi15x01234567 = _mm_cvtepi8_epi16(vi15x01234567);
      const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 120 * sizeof(int8_t)));
      const __m128i vxk15x01234567 = _mm_cvtepi8_epi16(vk15x01234567);
      i15 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi15x01234567, vxk15x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
      const __m128i vxi16x01234567 = _mm_cvtepi8_epi16(vi16x01234567);
      const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 128 * sizeof(int8_t)));
      const __m128i vxk16x01234567 = _mm_cvtepi8_epi16(vk16x01234567);
      i16 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
      const __m128i vxi17x01234567 = _mm_cvtepi8_epi16(vi17x01234567);
      const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 136 * sizeof(int8_t)));
      const __m128i vxk17x01234567 = _mm_cvtepi8_epi16(vk17x01234567);
      i17 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi17x01234567, vxk17x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
      const __m128i vxi18x01234567 = _mm_cvtepi8_epi16(vi18x01234567);
      const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 144 * sizeof(int8_t)));
      const __m128i vxk18x01234567 = _mm_cvtepi8_epi16(vk18x01234567);
      i18 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
      const __m128i vxi19x01234567 = _mm_cvtepi8_epi16(vi19x01234567);
      const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 152 * sizeof(int8_t)));
      const __m128i vxk19x01234567 = _mm_cvtepi8_epi16(vk19x01234567);
      i19 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi19x01234567, vxk19x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
      const __m128i vxi20x01234567 = _mm_cvtepi8_epi16(vi20x01234567);
      const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 160 * sizeof(int8_t)));
      const __m128i vxk20x01234567 = _mm_cvtepi8_epi16(vk20x01234567);
      i20 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
      const __m128i vxi21x01234567 = _mm_cvtepi8_epi16(vi21x01234567);
      const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 168 * sizeof(int8_t)));
      const __m128i vxk21x01234567 = _mm_cvtepi8_epi16(vk21x01234567);
      i21 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi21x01234567, vxk21x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
      const __m128i vxi22x01234567 = _mm_cvtepi8_epi16(vi22x01234567);
      const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 176 * sizeof(int8_t)));
      const __m128i vxk22x01234567 = _mm_cvtepi8_epi16(vk22x01234567);
      i22 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
      const __m128i vxi23x01234567 = _mm_cvtepi8_epi16(vi23x01234567);
      const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 184 * sizeof(int8_t)));
      const __m128i vxk23x01234567 = _mm_cvtepi8_epi16(vk23x01234567);
      i23 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi23x01234567, vxk23x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
      const __m128i vxi24x01234567 = _mm_cvtepi8_epi16(vi24x01234567);
      const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 192 * sizeof(int8_t)));
      const __m128i vxk24x01234567 = _mm_cvtepi8_epi16(vk24x01234567);
      i24 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      w = (const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 200 * sizeof(int8_t));

      __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

      const __m128 vscale0123 = _mm_loadu_ps((const float*) w);
      const __m128 vscale4567 = _mm_loadu_ps((const float*) w + 4);
      w = (const void*) ((const float*) w + 8);
      vscaled0123 = _mm_mul_ps(vscaled0123, vscale0123);
      vscaled4567 = _mm_mul_ps(vscaled4567, vscale4567);

      const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
      vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
      vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vscaled0123);
      vacc4567 = _mm_cvtps_epi32(vscaled4567);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse4.output_min);
      vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);

      _mm_storel_epi64((__m128i*) output, vout0123456701234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t)));
        const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);


        __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t)));
        const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t)));
        const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t)));
        const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(int8_t)));
        const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(int8_t)));
        const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(int8_t)));
        const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(int8_t)));
        const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(int8_t)));
        const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
        const __m128i vxi9x01234567 = _mm_cvtepi8_epi16(vi9x01234567);
        const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(int8_t)));
        const __m128i vxk9x01234567 = _mm_cvtepi8_epi16(vk9x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi9x01234567, vxk9x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
        const __m128i vxi10x01234567 = _mm_cvtepi8_epi16(vi10x01234567);
        const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 80 * sizeof(int8_t)));
        const __m128i vxk10x01234567 = _mm_cvtepi8_epi16(vk10x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
        const __m128i vxi11x01234567 = _mm_cvtepi8_epi16(vi11x01234567);
        const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 88 * sizeof(int8_t)));
        const __m128i vxk11x01234567 = _mm_cvtepi8_epi16(vk11x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi11x01234567, vxk11x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
        const __m128i vxi12x01234567 = _mm_cvtepi8_epi16(vi12x01234567);
        const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 96 * sizeof(int8_t)));
        const __m128i vxk12x01234567 = _mm_cvtepi8_epi16(vk12x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
        const __m128i vxi13x01234567 = _mm_cvtepi8_epi16(vi13x01234567);
        const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 104 * sizeof(int8_t)));
        const __m128i vxk13x01234567 = _mm_cvtepi8_epi16(vk13x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi13x01234567, vxk13x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
        const __m128i vxi14x01234567 = _mm_cvtepi8_epi16(vi14x01234567);
        const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 112 * sizeof(int8_t)));
        const __m128i vxk14x01234567 = _mm_cvtepi8_epi16(vk14x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
        const __m128i vxi15x01234567 = _mm_cvtepi8_epi16(vi15x01234567);
        const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 120 * sizeof(int8_t)));
        const __m128i vxk15x01234567 = _mm_cvtepi8_epi16(vk15x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi15x01234567, vxk15x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
        const __m128i vxi16x01234567 = _mm_cvtepi8_epi16(vi16x01234567);
        const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 128 * sizeof(int8_t)));
        const __m128i vxk16x01234567 = _mm_cvtepi8_epi16(vk16x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
        const __m128i vxi17x01234567 = _mm_cvtepi8_epi16(vi17x01234567);
        const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 136 * sizeof(int8_t)));
        const __m128i vxk17x01234567 = _mm_cvtepi8_epi16(vk17x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi17x01234567, vxk17x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
        const __m128i vxi18x01234567 = _mm_cvtepi8_epi16(vi18x01234567);
        const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 144 * sizeof(int8_t)));
        const __m128i vxk18x01234567 = _mm_cvtepi8_epi16(vk18x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
        const __m128i vxi19x01234567 = _mm_cvtepi8_epi16(vi19x01234567);
        const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 152 * sizeof(int8_t)));
        const __m128i vxk19x01234567 = _mm_cvtepi8_epi16(vk19x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi19x01234567, vxk19x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
        const __m128i vxi20x01234567 = _mm_cvtepi8_epi16(vi20x01234567);
        const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 160 * sizeof(int8_t)));
        const __m128i vxk20x01234567 = _mm_cvtepi8_epi16(vk20x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
        const __m128i vxi21x01234567 = _mm_cvtepi8_epi16(vi21x01234567);
        const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 168 * sizeof(int8_t)));
        const __m128i vxk21x01234567 = _mm_cvtepi8_epi16(vk21x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi21x01234567, vxk21x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
        const __m128i vxi22x01234567 = _mm_cvtepi8_epi16(vi22x01234567);
        const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 176 * sizeof(int8_t)));
        const __m128i vxk22x01234567 = _mm_cvtepi8_epi16(vk22x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
        const __m128i vxi23x01234567 = _mm_cvtepi8_epi16(vi23x01234567);
        const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 184 * sizeof(int8_t)));
        const __m128i vxk23x01234567 = _mm_cvtepi8_epi16(vk23x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi23x01234567, vxk23x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
        const __m128i vxi24x01234567 = _mm_cvtepi8_epi16(vi24x01234567);
        const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 192 * sizeof(int8_t)));
        const __m128i vxk24x01234567 = _mm_cvtepi8_epi16(vk24x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));


        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

        const __m128 vscale0123 = _mm_loadu_ps((const float*) ((uintptr_t) w + 8 * sizeof(int32_t) + 200 * sizeof(int8_t)));
        const __m128 vscale4567 = _mm_loadu_ps((const float*) ((uintptr_t) w + 8 * sizeof(int32_t) + 200 * sizeof(int8_t) + 4 * sizeof(float)));
        vscaled0123 = _mm_mul_ps(vscaled0123, vscale0123);
        vscaled4567 = _mm_mul_ps(vscaled4567, vscale4567);

        const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
        vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
        vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

        vacc0123 = _mm_cvtps_epi32(vscaled0123);
        vacc4567 = _mm_cvtps_epi32(vscaled4567);


        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

        vout0123456701234567 = _mm_max_epi8(vout0123456701234567, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

        if (c & 4) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (c & 2) {
          unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (c & 1) {
          *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
          output += 1;
        }
      }
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p8c__sse41_mul16(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 8; c -= 8) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
      i0 += 8;


      __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
      i1 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
      i2 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      w = (const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t));

      __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

      const __m128 vscale0123 = _mm_loadu_ps((const float*) w);
      const __m128 vscale4567 = _mm_loadu_ps((const float*) w + 4);
      w = (const void*) ((const float*) w + 8);
      vscaled0123 = _mm_mul_ps(vscaled0123, vscale0123);
      vscaled4567 = _mm_mul_ps(vscaled4567, vscale4567);

      const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
      vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
      vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vscaled0123);
      vacc4567 = _mm_cvtps_epi32(vscaled4567);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse4.output_min);
      vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);

      _mm_storel_epi64((__m128i*) output, vout0123456701234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t)));
        const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);


        __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t)));
        const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t)));
        const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));


        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

        const __m128 vscale0123 = _mm_loadu_ps((const float*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t)));
        const __m128 vscale4567 = _mm_loadu_ps((const float*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t) + 4 * sizeof(float)));
        vscaled0123 = _mm_mul_ps(vscaled0123, vscale0123);
        vscaled4567 = _mm_mul_ps(vscaled4567, vscale4567);

        const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
        vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
        vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

        vacc0123 = _mm_cvtps_epi32(vscaled0123);
        vacc4567 = _mm_cvtps_epi32(vscaled4567);


        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

        vout0123456701234567 = _mm_max_epi8(vout0123456701234567, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

        if (c & 4) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (c & 2) {
          unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (c & 1) {
          *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
          output += 1;
        }
      }
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p8c__sse41_mul16(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 8; c -= 8) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);
      i0 += 8;


      __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);
      i1 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);
      i2 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t)));
      const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);
      i3 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(int8_t)));
      const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);
      i4 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(int8_t)));
      const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);
      i5 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(int8_t)));
      const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);
      i6 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(int8_t)));
      const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);
      i7 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(int8_t)));
      const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);
      i8 += 8;


      vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

      w = (const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(int8_t));

      __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

      const __m128 vscale0123 = _mm_loadu_ps((const float*) w);
      const __m128 vscale4567 = _mm_loadu_ps((const float*) w + 4);
      w = (const void*) ((const float*) w + 8);
      vscaled0123 = _mm_mul_ps(vscaled0123, vscale0123);
      vscaled4567 = _mm_mul_ps(vscaled4567, vscale4567);

      const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
      vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
      vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vscaled0123);
      vacc4567 = _mm_cvtps_epi32(vscaled4567);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse4.output_min);
      vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);

      _mm_storel_epi64((__m128i*) output, vout0123456701234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(vi0x01234567);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t)));
        const __m128i vxk0x01234567 = _mm_cvtepi8_epi16(vk0x01234567);


        __m128i vprod01234567 = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(vi1x01234567);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t)));
        const __m128i vxk1x01234567 = _mm_cvtepi8_epi16(vk1x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(vi2x01234567);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t)));
        const __m128i vxk2x01234567 = _mm_cvtepi8_epi16(vk2x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(vi3x01234567);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t)));
        const __m128i vxk3x01234567 = _mm_cvtepi8_epi16(vk3x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(vi4x01234567);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(int8_t)));
        const __m128i vxk4x01234567 = _mm_cvtepi8_epi16(vk4x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(vi5x01234567);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(int8_t)));
        const __m128i vxk5x01234567 = _mm_cvtepi8_epi16(vk5x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(vi6x01234567);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(int8_t)));
        const __m128i vxk6x01234567 = _mm_cvtepi8_epi16(vk6x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vxi7x01234567 = _mm_cvtepi8_epi16(vi7x01234567);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(int8_t)));
        const __m128i vxk7x01234567 = _mm_cvtepi8_epi16(vk7x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vxi8x01234567 = _mm_cvtepi8_epi16(vi8x01234567);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(int8_t)));
        const __m128i vxk8x01234567 = _mm_cvtepi8_epi16(vk8x01234567);


        vprod01234567 = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vprod01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_srai_epi32(_mm_unpackhi_epi16(vprod01234567, vprod01234567), 16));


        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

        const __m128 vscale0123 = _mm_loadu_ps((const float*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(int8_t)));
        const __m128 vscale4567 = _mm_loadu_ps((const float*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(int8_t) + 4 * sizeof(float)));
        vscaled0123 = _mm_mul_ps(vscaled0123, vscale0123);
        vscaled4567 = _mm_mul_ps(vscaled4567, vscale4567);

        const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
        vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
        vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

        vacc0123 = _mm_cvtps_epi32(vscaled0123);
        vacc4567 = _mm_cvtps_epi32(vscaled4567);


        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

        vout0123456701234567 = _mm_max_epi8(vout0123456701234567, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

        if (c & 4) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (c & 2) {
          unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (c & 1) {
          *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
          output += 1;
        }
      }
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;


  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    w = (const int32_t*) w + 4;

    size_t k = kc;


    while (k >= 8 * sizeof(int8_t)) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
      a0 += 8;

      const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);

      const __m128i vxb0 = _mm_cvtepi8_epi16(vb0);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));

      const __m128i vxb1 = _mm_cvtepi8_epi16(vb1);

      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));

      const __m128i vxb2 = _mm_cvtepi8_epi16(vb2);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));

      const __m128i vxb3 = _mm_cvtepi8_epi16(vb3);

      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));

      w = (const int8_t*) w + 32;
      k -= 8 * sizeof(int8_t);
    }

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 vscale0123 = _mm_load_ps((const float*) w);
    w = (const float*) w + 4;
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale0123);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
    __m128i vacc00x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc0x0123), voutput_zero_point);


    __m128i vout = _mm_packs_epi16(vacc00x0123, vacc00x0123);

    vout = _mm_max_epi8(vout, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

    if (nc >= 4) {
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }


  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m128i vacc1x0 = vacc0x0;
    __m128i vacc1x1 = vacc0x1;
    __m128i vacc1x2 = vacc0x2;
    __m128i vacc1x3 = vacc0x3;
    __m128i vacc2x0 = vacc0x0;
    __m128i vacc2x1 = vacc0x1;
    __m128i vacc2x2 = vacc0x2;
    __m128i vacc2x3 = vacc0x3;
    w = (const int32_t*) w + 4;

    size_t k = kc;


    while (k >= 8 * sizeof(int8_t)) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
      a0 += 8;
      const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
      const __m128i vxa1 = _mm_cvtepi8_epi16(va1);
      a1 += 8;
      const __m128i va2 = _mm_loadl_epi64((const __m128i*) a2);
      const __m128i vxa2 = _mm_cvtepi8_epi16(va2);
      a2 += 8;

      const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);

      const __m128i vxb0 = _mm_cvtepi8_epi16(vb0);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
      vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2, vxb0));
      const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));

      const __m128i vxb1 = _mm_cvtepi8_epi16(vb1);

      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
      vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2, vxb1));
      const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));

      const __m128i vxb2 = _mm_cvtepi8_epi16(vb2);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
      vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2, vxb2));
      const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));

      const __m128i vxb3 = _mm_cvtepi8_epi16(vb3);

      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
      vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));
      vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2, vxb3));

      w = (const int8_t*) w + 32;
      k -= 8 * sizeof(int8_t);
    }

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);
    const __m128i vacc1x01 = _mm_hadd_epi32(vacc1x0, vacc1x1);
    const __m128i vacc1x23 = _mm_hadd_epi32(vacc1x2, vacc1x3);
    const __m128i vacc2x01 = _mm_hadd_epi32(vacc2x0, vacc2x1);
    const __m128i vacc2x23 = _mm_hadd_epi32(vacc2x2, vacc2x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
    __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);
    __m128i vacc2x0123 = _mm_hadd_epi32(vacc2x01, vacc2x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vscaled1x0123 = _mm_cvtepi32_ps(vacc1x0123);
    __m128 vscaled2x0123 = _mm_cvtepi32_ps(vacc2x0123);

    const __m128 vscale0123 = _mm_load_ps((const float*) w);
    w = (const float*) w + 4;
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale0123);
    vscaled1x0123 = _mm_mul_ps(vscaled1x0123, vscale0123);
    vscaled2x0123 = _mm_mul_ps(vscaled2x0123, vscale0123);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);
    vscaled1x0123 = _mm_min_ps(vscaled1x0123, voutput_max_less_zero_point);
    vscaled2x0123 = _mm_min_ps(vscaled2x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);
    vacc1x0123 = _mm_cvtps_epi32(vscaled1x0123);
    vacc2x0123 = _mm_cvtps_epi32(vscaled2x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
    __m128i vacc01x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);
    __m128i vacc22x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc2x0123, vacc2x0123), voutput_zero_point);


    __m128i vout = _mm_packs_epi16(vacc01x0123, vacc22x0123);

    vout = _mm_max_epi8(vout, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

    if (nc >= 4) {
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));
      unaligned_store_u32(c1, (uint32_t) _mm_extract_epi32(vout, 1));
      unaligned_store_u32(c2, (uint32_t) _mm_extract_epi32(vout, 2));

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(vout, 2));
        c1 += 2;
        unaligned_store_u16(c2, (uint16_t) _mm_extract_epi16(vout, 4));
        c2 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_extract_epi8(vout, 0);
        *c1 = (int8_t) _mm_extract_epi8(vout, 4);
        *c2 = (int8_t) _mm_extract_epi8(vout, 8);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  int8_t* c0 = c;

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
        a0 += 8;

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
        const __m128i vxb0 = _mm_cvtepi8_epi16(vb0);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));
        const __m128i vxb1 = _mm_cvtepi8_epi16(vb1);

        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vxb2 = _mm_cvtepi8_epi16(vb2);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));
        const __m128i vxb3 = _mm_cvtepi8_epi16(vb3);

        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));

        w = (const void*) ((const int8_t*) w + 32);
        k += 8 * sizeof(int8_t);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 vscale0123 = _mm_load_ps((const float*) w);
    w = (const float*) w + 4;
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale0123);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
    __m128i vacc00x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc0x0123), voutput_zero_point);


    __m128i vout = _mm_packs_epi16(vacc00x0123, vacc00x0123);

    vout = _mm_max_epi8(vout, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

    if (nc >= 4) {
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__sse41_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (3 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m128i vacc1x0 = vacc0x0;
    __m128i vacc1x1 = vacc0x1;
    __m128i vacc1x2 = vacc0x2;
    __m128i vacc1x3 = vacc0x3;
    __m128i vacc2x0 = vacc0x0;
    __m128i vacc2x1 = vacc0x1;
    __m128i vacc2x2 = vacc0x2;
    __m128i vacc2x3 = vacc0x3;
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      a += 3;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
        a0 += 8;
        const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
        const __m128i vxa1 = _mm_cvtepi8_epi16(va1);
        a1 += 8;
        const __m128i va2 = _mm_loadl_epi64((const __m128i*) a2);
        const __m128i vxa2 = _mm_cvtepi8_epi16(va2);
        a2 += 8;

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
        const __m128i vxb0 = _mm_cvtepi8_epi16(vb0);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
        vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2, vxb0));
        const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));
        const __m128i vxb1 = _mm_cvtepi8_epi16(vb1);

        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
        vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2, vxb1));
        const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vxb2 = _mm_cvtepi8_epi16(vb2);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
        vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2, vxb2));
        const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));
        const __m128i vxb3 = _mm_cvtepi8_epi16(vb3);

        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
        vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));
        vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2, vxb3));

        w = (const void*) ((const int8_t*) w + 32);
        k += 8 * sizeof(int8_t);
      }
      p -= 3 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);
    const __m128i vacc1x01 = _mm_hadd_epi32(vacc1x0, vacc1x1);
    const __m128i vacc1x23 = _mm_hadd_epi32(vacc1x2, vacc1x3);
    const __m128i vacc2x01 = _mm_hadd_epi32(vacc2x0, vacc2x1);
    const __m128i vacc2x23 = _mm_hadd_epi32(vacc2x2, vacc2x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
    __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);
    __m128i vacc2x0123 = _mm_hadd_epi32(vacc2x01, vacc2x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vscaled1x0123 = _mm_cvtepi32_ps(vacc1x0123);
    __m128 vscaled2x0123 = _mm_cvtepi32_ps(vacc2x0123);

    const __m128 vscale0123 = _mm_load_ps((const float*) w);
    w = (const float*) w + 4;
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale0123);
    vscaled1x0123 = _mm_mul_ps(vscaled1x0123, vscale0123);
    vscaled2x0123 = _mm_mul_ps(vscaled2x0123, vscale0123);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);
    vscaled1x0123 = _mm_min_ps(vscaled1x0123, voutput_max_less_zero_point);
    vscaled2x0123 = _mm_min_ps(vscaled2x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);
    vacc1x0123 = _mm_cvtps_epi32(vscaled1x0123);
    vacc2x0123 = _mm_cvtps_epi32(vscaled2x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
    __m128i vacc01x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);
    __m128i vacc22x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc2x0123, vacc2x0123), voutput_zero_point);


    __m128i vout = _mm_packs_epi16(vacc01x0123, vacc22x0123);

    vout = _mm_max_epi8(vout, _mm_load_si128((const __m128i*) params->fp32_sse4.output_min));

    if (nc >= 4) {
      unaligned_store_u32(c2, (uint32_t) _mm_extract_epi32(vout, 2));
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      unaligned_store_u32(c1, (uint32_t) _mm_extract_epi32(vout, 1));
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c2, (uint16_t) _mm_extract_epi16(vout, 4));
        c2 += 2;
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(vout, 2));
        c1 += 2;
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c2 = (int8_t) _mm_extract_epi8(vout, 8);
        *c1 = (int8_t) _mm_extract_epi8(vout, 4);
        *c0 = (int8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_u8(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128i vbias = _mm_set1_epi32(params->scalar.bias);
  const __m128i va_multiplier_lo = _mm_set1_epi16(params->scalar.a_multiplier);
  const __m128i va_multiplier_hi = _mm_set1_epi16((uint32_t)params->scalar.a_multiplier >> 16);
  const __m128i vb_multiplier_lo = _mm_set1_epi16(params->scalar.b_multiplier);
  const __m128i vb_multiplier_hi = _mm_set1_epi16((uint32_t)params->scalar.b_multiplier >> 16);
  const __m128i vshift = _mm_cvtsi32_si128((int) params->scalar.shift);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8(params->scalar.output_min);
  const __m128i voutput_max = _mm_set1_epi8(params->scalar.output_max);

  XNN_FORCE_REALIZATION(vbias);
  XNN_FORCE_REALIZATION(va_multiplier_lo);
  XNN_FORCE_REALIZATION(va_multiplier_hi);
  XNN_FORCE_REALIZATION(vb_multiplier_lo);
  XNN_FORCE_REALIZATION(vb_multiplier_hi);
  XNN_FORCE_REALIZATION(vshift);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    const __m128i va01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
    const __m128i vb01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_b));
    input_a += 8;
    input_b += 8;


    __m128i vaprod01234567hi = _mm_mulhi_epu16(va01234567, va_multiplier_lo);
    __m128i vbprod01234567hi = _mm_mulhi_epu16(vb01234567, vb_multiplier_lo);
    const __m128i vaprod01234567lo = _mm_mullo_epi16(va01234567, va_multiplier_lo);
    const __m128i vbprod01234567lo = _mm_mullo_epi16(vb01234567, vb_multiplier_lo);

    vaprod01234567hi = _mm_add_epi16(vaprod01234567hi, _mm_mullo_epi16(va01234567, va_multiplier_hi));
    vbprod01234567hi = _mm_add_epi16(vbprod01234567hi, _mm_mullo_epi16(vb01234567, vb_multiplier_hi));

    vaprod01234567hi = _mm_sub_epi16(vaprod01234567hi, _mm_and_si128(_mm_srai_epi16(va01234567, 15), va_multiplier_lo));
    vbprod01234567hi = _mm_sub_epi16(vbprod01234567hi, _mm_and_si128(_mm_srai_epi16(vb01234567, 15), vb_multiplier_lo));

    __m128i vacc0123 = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vaprod01234567lo, vaprod01234567hi));
    __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vaprod01234567lo, vaprod01234567hi));

    vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vbprod01234567lo, vbprod01234567hi));
    vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vbprod01234567lo, vbprod01234567hi));

    vacc0123 = _mm_sra_epi32(vacc0123, vshift);
    vacc4567 = _mm_sra_epi32(vacc4567, vshift);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


    __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

    vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);

    vout0123456701234567 = _mm_min_epi8(vout0123456701234567, voutput_max);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      const __m128i va01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
      const __m128i vb01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_b));


      __m128i vaprod01234567hi = _mm_mulhi_epu16(va01234567, va_multiplier_lo);
      __m128i vbprod01234567hi = _mm_mulhi_epu16(vb01234567, vb_multiplier_lo);
      const __m128i vaprod01234567lo = _mm_mullo_epi16(va01234567, va_multiplier_lo);
      const __m128i vbprod01234567lo = _mm_mullo_epi16(vb01234567, vb_multiplier_lo);

      vaprod01234567hi = _mm_add_epi16(vaprod01234567hi, _mm_mullo_epi16(va01234567, va_multiplier_hi));
      vbprod01234567hi = _mm_add_epi16(vbprod01234567hi, _mm_mullo_epi16(vb01234567, vb_multiplier_hi));

      vaprod01234567hi = _mm_sub_epi16(vaprod01234567hi, _mm_and_si128(_mm_srai_epi16(va01234567, 15), va_multiplier_lo));
      vbprod01234567hi = _mm_sub_epi16(vbprod01234567hi, _mm_and_si128(_mm_srai_epi16(vb01234567, 15), vb_multiplier_lo));

      __m128i vacc0123 = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vaprod01234567lo, vaprod01234567hi));
      __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vaprod01234567lo, vaprod01234567hi));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vbprod01234567lo, vbprod01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vbprod01234567lo, vbprod01234567hi));

      vacc0123 = _mm_sra_epi32(vacc0123, vshift);
      vacc4567 = _mm_sra_epi32(vacc4567, vshift);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epi8(vout0123456701234567, voutput_max);

      if (batch & (4 * sizeof(int8_t))) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (batch & (2 * sizeof(int8_t))) {
        unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (batch & (1 * sizeof(int8_t))) {
        *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
      }
    }
  }
}

void xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_u8(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128i vbias = _mm_set1_epi32(params->scalar.b_multiplier * (int32_t) *input_b + params->scalar.bias);
  const __m128i va_multiplier_lo = _mm_set1_epi16(params->scalar.a_multiplier);
  const __m128i va_multiplier_hi = _mm_set1_epi16((uint32_t)params->scalar.a_multiplier >> 16);
  const __m128i vshift = _mm_cvtsi32_si128((int) params->scalar.shift);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8(params->scalar.output_min);
  const __m128i voutput_max = _mm_set1_epi8(params->scalar.output_max);

  XNN_FORCE_REALIZATION(vbias);
  XNN_FORCE_REALIZATION(va_multiplier_lo);
  XNN_FORCE_REALIZATION(va_multiplier_hi);
  XNN_FORCE_REALIZATION(vshift);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    const __m128i va01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
    input_a += 8;


    __m128i vaprod01234567hi = _mm_mulhi_epu16(va01234567, va_multiplier_lo);
    const __m128i vaprod01234567lo = _mm_mullo_epi16(va01234567, va_multiplier_lo);

    vaprod01234567hi = _mm_add_epi16(vaprod01234567hi, _mm_mullo_epi16(va01234567, va_multiplier_hi));

    vaprod01234567hi = _mm_sub_epi16(vaprod01234567hi, _mm_and_si128(_mm_srai_epi16(va01234567, 15), va_multiplier_lo));

    __m128i vacc0123 = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vaprod01234567lo, vaprod01234567hi));
    __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vaprod01234567lo, vaprod01234567hi));

    vacc0123 = _mm_sra_epi32(vacc0123, vshift);
    vacc4567 = _mm_sra_epi32(vacc4567, vshift);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


    __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

    vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);

    vout0123456701234567 = _mm_min_epi8(vout0123456701234567, voutput_max);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      const __m128i va01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_a));


      __m128i vaprod01234567hi = _mm_mulhi_epu16(va01234567, va_multiplier_lo);
      const __m128i vaprod01234567lo = _mm_mullo_epi16(va01234567, va_multiplier_lo);

      vaprod01234567hi = _mm_add_epi16(vaprod01234567hi, _mm_mullo_epi16(va01234567, va_multiplier_hi));

      vaprod01234567hi = _mm_sub_epi16(vaprod01234567hi, _mm_and_si128(_mm_srai_epi16(va01234567, 15), va_multiplier_lo));

      __m128i vacc0123 = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vaprod01234567lo, vaprod01234567hi));
      __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vaprod01234567lo, vaprod01234567hi));

      vacc0123 = _mm_sra_epi32(vacc0123, vshift);
      vacc4567 = _mm_sra_epi32(vacc4567, vshift);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epi8(vout0123456701234567, voutput_max);

      if (batch & (4 * sizeof(int8_t))) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (batch & (2 * sizeof(int8_t))) {
        unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (batch & (1 * sizeof(int8_t))) {
        *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
      }
    }
  }
}

void xnn_qs8_vcvt_ukernel__sse41_u32(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vinput_zero_point = _mm_set1_epi16(params->scalar.input_zero_point);
  const __m128i vmultiplier = _mm_set1_epi16(-params->scalar.multiplier);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(vmultiplier);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    __m128i vacc0 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input));
    __m128i vacc1 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input + 8)));
    __m128i vacc2 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input + 16)));
    __m128i vacc3 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input + 24)));
    input += 32;

    vacc0 = _mm_sub_epi16(vinput_zero_point, vacc0);
    vacc1 = _mm_sub_epi16(vinput_zero_point, vacc1);
    vacc2 = _mm_sub_epi16(vinput_zero_point, vacc2);
    vacc3 = _mm_sub_epi16(vinput_zero_point, vacc3);

    vacc0 = _mm_slli_epi16(vacc0, 7);
    vacc1 = _mm_slli_epi16(vacc1, 7);
    vacc2 = _mm_slli_epi16(vacc2, 7);
    vacc3 = _mm_slli_epi16(vacc3, 7);

    vacc0 = _mm_mulhrs_epi16(vacc0, vmultiplier);
    vacc1 = _mm_mulhrs_epi16(vacc1, vmultiplier);
    vacc2 = _mm_mulhrs_epi16(vacc2, vmultiplier);
    vacc3 = _mm_mulhrs_epi16(vacc3, vmultiplier);

    vacc0 = _mm_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm_adds_epi16(vacc1, voutput_zero_point);
    vacc2 = _mm_adds_epi16(vacc2, voutput_zero_point);
    vacc3 = _mm_adds_epi16(vacc3, voutput_zero_point);

    const __m128i vy0 = _mm_packs_epi16(vacc0, vacc1);
    const __m128i vy1 = _mm_packs_epi16(vacc2, vacc3);

    _mm_storeu_si128((__m128i*) output, vy0);
    _mm_storeu_si128((__m128i*) (output + 16), vy1);
    output += 32;
  }
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    __m128i vacc = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input));
    vacc = _mm_sub_epi16(vinput_zero_point, vacc);
    vacc = _mm_slli_epi16(vacc, 7);
    vacc = _mm_mulhrs_epi16(vacc, vmultiplier);
    vacc = _mm_adds_epi16(vacc, voutput_zero_point);
    input += 8;

    const __m128i vy = _mm_packs_epi16(vacc, vacc);
    _mm_storel_epi64((__m128i*) output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    __m128i vacc = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input));
    vacc = _mm_sub_epi16(vinput_zero_point, vacc);
    vacc = _mm_slli_epi16(vacc, 7);
    vacc = _mm_mulhrs_epi16(vacc, vmultiplier);
    vacc = _mm_adds_epi16(vacc, voutput_zero_point);

    __m128i vy = _mm_packs_epi16(vacc, vacc);
    if (batch & (4 * sizeof(int8_t))) {
      _mm_storeu_si32(output, vy);
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(int8_t))) {
      _mm_storeu_si16(output, vy);
      vy = _mm_srli_epi32(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      *output = (int8_t) _mm_extract_epi8(vy, 0);
    }
  }
}

void xnn_qs8_vlrelu_ukernel__sse41_u32(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vinput_zero_point = _mm_set1_epi16(params->scalar.input_zero_point);
  const __m128i vmultiplier_diff = _mm_set1_epi16(-params->scalar.negative_multiplier ^ -params->scalar.positive_multiplier);
  const __m128i vmultiplier_base = _mm_set1_epi16(-params->scalar.negative_multiplier);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vmultiplier_diff);
  XNN_FORCE_REALIZATION(vmultiplier_base);
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    __m128i vacc0 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input));
    __m128i vacc1 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input + 8)));
    __m128i vacc2 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input + 16)));
    __m128i vacc3 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input + 24)));
    input += 32;

    __m128i vmultiplier0 = _mm_cmpgt_epi16(vacc0, vinput_zero_point);
    vacc0 = _mm_sub_epi16(vinput_zero_point, vacc0);
    __m128i vmultiplier1 = _mm_cmpgt_epi16(vacc1, vinput_zero_point);
    vacc1 = _mm_sub_epi16(vinput_zero_point, vacc1);
    __m128i vmultiplier2 = _mm_cmpgt_epi16(vacc2, vinput_zero_point);
    vacc2 = _mm_sub_epi16(vinput_zero_point, vacc2);
    __m128i vmultiplier3 = _mm_cmpgt_epi16(vacc3, vinput_zero_point);
    vacc3 = _mm_sub_epi16(vinput_zero_point, vacc3);

    vmultiplier0 = _mm_and_si128(vmultiplier0, vmultiplier_diff);
    vacc0 = _mm_slli_epi16(vacc0, 7);
    vmultiplier0 = _mm_xor_si128(vmultiplier0, vmultiplier_base);
    vmultiplier1 = _mm_and_si128(vmultiplier1, vmultiplier_diff);
    vacc1 = _mm_slli_epi16(vacc1, 7);
    vmultiplier1 = _mm_xor_si128(vmultiplier1, vmultiplier_base);
    vmultiplier2 = _mm_and_si128(vmultiplier2, vmultiplier_diff);
    vacc2 = _mm_slli_epi16(vacc2, 7);
    vmultiplier2 = _mm_xor_si128(vmultiplier2, vmultiplier_base);
    vmultiplier3 = _mm_and_si128(vmultiplier3, vmultiplier_diff);
    vacc3 = _mm_slli_epi16(vacc3, 7);
    vmultiplier3 = _mm_xor_si128(vmultiplier3, vmultiplier_base);

    vacc0 = _mm_mulhrs_epi16(vacc0, vmultiplier0);
    vacc1 = _mm_mulhrs_epi16(vacc1, vmultiplier1);
    vacc2 = _mm_mulhrs_epi16(vacc2, vmultiplier2);
    vacc3 = _mm_mulhrs_epi16(vacc3, vmultiplier3);

    vacc0 = _mm_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm_adds_epi16(vacc1, voutput_zero_point);
    vacc2 = _mm_adds_epi16(vacc2, voutput_zero_point);
    vacc3 = _mm_adds_epi16(vacc3, voutput_zero_point);

    const __m128i vy0 = _mm_packs_epi16(vacc0, vacc1);
    const __m128i vy1 = _mm_packs_epi16(vacc2, vacc3);

    _mm_storeu_si128((__m128i*) output, vy0);
    _mm_storeu_si128((__m128i*) (output + 16), vy1);
    output += 32;
  }
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    __m128i vacc = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input));
    __m128i vmultiplier = _mm_cmpgt_epi16(vacc, vinput_zero_point);
    vacc = _mm_sub_epi16(vinput_zero_point, vacc);
    vmultiplier = _mm_and_si128(vmultiplier, vmultiplier_diff);
    vacc = _mm_slli_epi16(vacc, 7);
    vmultiplier = _mm_xor_si128(vmultiplier, vmultiplier_base);
    vacc = _mm_mulhrs_epi16(vacc, vmultiplier);
    vacc = _mm_adds_epi16(vacc, voutput_zero_point);
    input += 8;

    const __m128i vy = _mm_packs_epi16(vacc, vacc);
    _mm_storel_epi64((__m128i*) output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    __m128i vacc = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input));
    __m128i vmultiplier = _mm_cmpgt_epi16(vacc, vinput_zero_point);
    vacc = _mm_sub_epi16(vinput_zero_point, vacc);
    vmultiplier = _mm_and_si128(vmultiplier, vmultiplier_diff);
    vacc = _mm_slli_epi16(vacc, 7);
    vmultiplier = _mm_xor_si128(vmultiplier, vmultiplier_base);
    vacc = _mm_mulhrs_epi16(vacc, vmultiplier);
    vacc = _mm_adds_epi16(vacc, voutput_zero_point);

    __m128i vy = _mm_packs_epi16(vacc, vacc);
    if (batch & (4 * sizeof(int8_t))) {
      _mm_storeu_si32(output, vy);
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(int8_t))) {
      _mm_storeu_si16(output, vy);
      vy = _mm_srli_epi32(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      *output = (int8_t) _mm_extract_epi8(vy, 0);
    }
  }
}

void xnn_qs8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_u16(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128i va_zero_point = _mm_set1_epi16(params->scalar.a_zero_point);
  const __m128i vb_zero_point = _mm_set1_epi16(params->scalar.b_zero_point);
  const __m128 vscale = _mm_set1_ps(params->scalar.scale);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8(params->scalar.output_min);
  const __m128i voutput_max = _mm_set1_epi8(params->scalar.output_max);

  XNN_FORCE_REALIZATION(va_zero_point);
  XNN_FORCE_REALIZATION(vb_zero_point);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    const __m128i va01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
    const __m128i vb01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_b));
    const __m128i va89ABCDEF = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input_a + 8)));
    const __m128i vb89ABCDEF = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input_b + 8)));
    input_a += 16;
    input_b += 16;


    const __m128i vxa01234567 = _mm_sub_epi16(va01234567, va_zero_point);
    const __m128i vxb01234567 = _mm_sub_epi16(vb01234567, vb_zero_point);
    const __m128i vxa89ABCDEF = _mm_sub_epi16(va89ABCDEF, va_zero_point);
    const __m128i vxb89ABCDEF = _mm_sub_epi16(vb89ABCDEF, vb_zero_point);

    const __m128i vprod01234567lo = _mm_mullo_epi16(vxa01234567, vxb01234567);
    const __m128i vprod01234567hi = _mm_mulhi_epi16(vxa01234567, vxb01234567);
    const __m128i vprod89ABCDEFlo = _mm_mullo_epi16(vxa89ABCDEF, vxb89ABCDEF);
    const __m128i vprod89ABCDEFhi = _mm_mulhi_epi16(vxa89ABCDEF, vxb89ABCDEF);

    const __m128i vprod0123 = _mm_unpacklo_epi16(vprod01234567lo, vprod01234567hi);
    const __m128i vprod4567 = _mm_unpackhi_epi16(vprod01234567lo, vprod01234567hi);
    const __m128i vprod89AB = _mm_unpacklo_epi16(vprod89ABCDEFlo, vprod89ABCDEFhi);
    const __m128i vprodCDEF = _mm_unpackhi_epi16(vprod89ABCDEFlo, vprod89ABCDEFhi);

    __m128 vfpacc0123 = _mm_cvtepi32_ps(vprod0123);
    __m128 vfpacc4567 = _mm_cvtepi32_ps(vprod4567);
    __m128 vfpacc89AB = _mm_cvtepi32_ps(vprod89AB);
    __m128 vfpaccCDEF = _mm_cvtepi32_ps(vprodCDEF);

    vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
    vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);
    vfpacc89AB = _mm_mul_ps(vfpacc89AB, vscale);
    vfpaccCDEF = _mm_mul_ps(vfpaccCDEF, vscale);

    const __m128i vacc0123 = _mm_cvtps_epi32(vfpacc0123);
    const __m128i vacc4567 = _mm_cvtps_epi32(vfpacc4567);
    const __m128i vacc89AB = _mm_cvtps_epi32(vfpacc89AB);
    const __m128i vaccCDEF = _mm_cvtps_epi32(vfpaccCDEF);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
    __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);


    __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);

    vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epi8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const __m128i va01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
      const __m128i vb01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_b));
      input_a += 8;
      input_b += 8;


      const __m128i vxa01234567 = _mm_sub_epi16(va01234567, va_zero_point);
      const __m128i vxb01234567 = _mm_sub_epi16(vb01234567, vb_zero_point);

      const __m128i vprod01234567lo = _mm_mullo_epi16(vxa01234567, vxb01234567);
      const __m128i vprod01234567hi = _mm_mulhi_epi16(vxa01234567, vxb01234567);

      const __m128i vprod0123 = _mm_unpacklo_epi16(vprod01234567lo, vprod01234567hi);
      const __m128i vprod4567 = _mm_unpackhi_epi16(vprod01234567lo, vprod01234567hi);

      __m128 vfpacc0123 = _mm_cvtepi32_ps(vprod0123);
      __m128 vfpacc4567 = _mm_cvtepi32_ps(vprod4567);

      vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
      vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

      const __m128i vacc0123 = _mm_cvtps_epi32(vfpacc0123);
      const __m128i vacc4567 = _mm_cvtps_epi32(vfpacc4567);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epi8(vout0123456701234567, voutput_max);

      if XNN_LIKELY(batch >= (8 * sizeof(int8_t))) {
        _mm_storel_epi64((__m128i*) output, vout0123456701234567);
        output += 8;
        batch -= 8 * sizeof(int8_t);
      } else {
        if (batch & (4 * sizeof(int8_t))) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (batch & (2 * sizeof(int8_t))) {
          unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (batch & (1 * sizeof(int8_t))) {
          *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
        }
        batch = 0;
      }
    } while (batch != 0);
  }
}

void xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u16(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128i va_zero_point = _mm_set1_epi16(params->scalar.a_zero_point);
  const __m128 vscale = _mm_set1_ps(params->scalar.scale);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8(params->scalar.output_min);
  const __m128i voutput_max = _mm_set1_epi8(params->scalar.output_max);

  __m128i vxb = _mm_set1_epi16((UINT32_C(0x00010001) * (uint32_t) (uint16_t) (int16_t) *input_b) - params->scalar.b_zero_point);

  XNN_FORCE_REALIZATION(va_zero_point);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    const __m128i va01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
    const __m128i va89ABCDEF = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (input_a + 8)));
    input_a += 16;


    const __m128i vxa01234567 = _mm_sub_epi16(va01234567, va_zero_point);
    const __m128i vxa89ABCDEF = _mm_sub_epi16(va89ABCDEF, va_zero_point);

    const __m128i vprod01234567lo = _mm_mullo_epi16(vxa01234567, vxb);
    const __m128i vprod01234567hi = _mm_mulhi_epi16(vxa01234567, vxb);
    const __m128i vprod89ABCDEFlo = _mm_mullo_epi16(vxa89ABCDEF, vxb);
    const __m128i vprod89ABCDEFhi = _mm_mulhi_epi16(vxa89ABCDEF, vxb);

    const __m128i vprod0123 = _mm_unpacklo_epi16(vprod01234567lo, vprod01234567hi);
    const __m128i vprod4567 = _mm_unpackhi_epi16(vprod01234567lo, vprod01234567hi);
    const __m128i vprod89AB = _mm_unpacklo_epi16(vprod89ABCDEFlo, vprod89ABCDEFhi);
    const __m128i vprodCDEF = _mm_unpackhi_epi16(vprod89ABCDEFlo, vprod89ABCDEFhi);

    __m128 vfpacc0123 = _mm_cvtepi32_ps(vprod0123);
    __m128 vfpacc4567 = _mm_cvtepi32_ps(vprod4567);
    __m128 vfpacc89AB = _mm_cvtepi32_ps(vprod89AB);
    __m128 vfpaccCDEF = _mm_cvtepi32_ps(vprodCDEF);

    vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
    vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);
    vfpacc89AB = _mm_mul_ps(vfpacc89AB, vscale);
    vfpaccCDEF = _mm_mul_ps(vfpaccCDEF, vscale);

    const __m128i vacc0123 = _mm_cvtps_epi32(vfpacc0123);
    const __m128i vacc4567 = _mm_cvtps_epi32(vfpacc4567);
    const __m128i vacc89AB = _mm_cvtps_epi32(vfpacc89AB);
    const __m128i vaccCDEF = _mm_cvtps_epi32(vfpaccCDEF);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
    __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);


    __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);

    vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epi8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const __m128i va01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
      input_a += 8;


      const __m128i vxa01234567 = _mm_sub_epi16(va01234567, va_zero_point);

      const __m128i vprod01234567lo = _mm_mullo_epi16(vxa01234567, vxb);
      const __m128i vprod01234567hi = _mm_mulhi_epi16(vxa01234567, vxb);

      const __m128i vprod0123 = _mm_unpacklo_epi16(vprod01234567lo, vprod01234567hi);
      const __m128i vprod4567 = _mm_unpackhi_epi16(vprod01234567lo, vprod01234567hi);

      __m128 vfpacc0123 = _mm_cvtepi32_ps(vprod0123);
      __m128 vfpacc4567 = _mm_cvtepi32_ps(vprod4567);

      vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
      vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

      const __m128i vacc0123 = _mm_cvtps_epi32(vfpacc0123);
      const __m128i vacc4567 = _mm_cvtps_epi32(vfpacc4567);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epi8(vout0123456701234567, voutput_max);

      if XNN_LIKELY(batch >= (8 * sizeof(int8_t))) {
        _mm_storel_epi64((__m128i*) output, vout0123456701234567);
        output += 8;
        batch -= 8 * sizeof(int8_t);
      } else {
        if (batch & (4 * sizeof(int8_t))) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (batch & (2 * sizeof(int8_t))) {
          unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (batch & (1 * sizeof(int8_t))) {
          *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
        }
        batch = 0;
      }
    } while (batch != 0);
  }
}

void xnn_qu8_dwconv_minmax_fp32_ukernel_25p8c__sse41_mul16(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const uint8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
    }
    const uint8_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const uint8_t*) ((uintptr_t) i9 + input_offset);
    }
    const uint8_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const uint8_t*) ((uintptr_t) i10 + input_offset);
    }
    const uint8_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const uint8_t*) ((uintptr_t) i11 + input_offset);
    }
    const uint8_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const uint8_t*) ((uintptr_t) i12 + input_offset);
    }
    const uint8_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const uint8_t*) ((uintptr_t) i13 + input_offset);
    }
    const uint8_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const uint8_t*) ((uintptr_t) i14 + input_offset);
    }
    const uint8_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const uint8_t*) ((uintptr_t) i15 + input_offset);
    }
    const uint8_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const uint8_t*) ((uintptr_t) i16 + input_offset);
    }
    const uint8_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const uint8_t*) ((uintptr_t) i17 + input_offset);
    }
    const uint8_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const uint8_t*) ((uintptr_t) i18 + input_offset);
    }
    const uint8_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const uint8_t*) ((uintptr_t) i19 + input_offset);
    }
    const uint8_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const uint8_t*) ((uintptr_t) i20 + input_offset);
    }
    const uint8_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const uint8_t*) ((uintptr_t) i21 + input_offset);
    }
    const uint8_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const uint8_t*) ((uintptr_t) i22 + input_offset);
    }
    const uint8_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const uint8_t*) ((uintptr_t) i23 + input_offset);
    }
    const uint8_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const uint8_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const uint8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    const __m128i vk_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.kernel_zero_point);
    for (; c >= 8; c -= 8) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vxi0x01234567 = _mm_cvtepu8_epi16(vi0x01234567);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(uint8_t)));
      const __m128i vxk0x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk0x01234567), vk_zero_point);
      i0 += 8;


      const __m128i vprod0x01234567lo = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
      const __m128i vprod0x01234567hi = _mm_mulhi_epi16(vxi0x01234567, vxk0x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod0x01234567lo, vprod0x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod0x01234567lo, vprod0x01234567hi));

      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vxi1x01234567 = _mm_cvtepu8_epi16(vi1x01234567);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(uint8_t)));
      const __m128i vxk1x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk1x01234567), vk_zero_point);
      i1 += 8;


      const __m128i vprod1x01234567lo = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);
      const __m128i vprod1x01234567hi = _mm_mulhi_epi16(vxi1x01234567, vxk1x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod1x01234567lo, vprod1x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod1x01234567lo, vprod1x01234567hi));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vxi2x01234567 = _mm_cvtepu8_epi16(vi2x01234567);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(uint8_t)));
      const __m128i vxk2x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk2x01234567), vk_zero_point);
      i2 += 8;


      const __m128i vprod2x01234567lo = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
      const __m128i vprod2x01234567hi = _mm_mulhi_epi16(vxi2x01234567, vxk2x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod2x01234567lo, vprod2x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod2x01234567lo, vprod2x01234567hi));

      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vxi3x01234567 = _mm_cvtepu8_epi16(vi3x01234567);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(uint8_t)));
      const __m128i vxk3x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk3x01234567), vk_zero_point);
      i3 += 8;


      const __m128i vprod3x01234567lo = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);
      const __m128i vprod3x01234567hi = _mm_mulhi_epi16(vxi3x01234567, vxk3x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod3x01234567lo, vprod3x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod3x01234567lo, vprod3x01234567hi));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vxi4x01234567 = _mm_cvtepu8_epi16(vi4x01234567);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(uint8_t)));
      const __m128i vxk4x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk4x01234567), vk_zero_point);
      i4 += 8;


      const __m128i vprod4x01234567lo = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
      const __m128i vprod4x01234567hi = _mm_mulhi_epi16(vxi4x01234567, vxk4x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod4x01234567lo, vprod4x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod4x01234567lo, vprod4x01234567hi));

      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vxi5x01234567 = _mm_cvtepu8_epi16(vi5x01234567);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(uint8_t)));
      const __m128i vxk5x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk5x01234567), vk_zero_point);
      i5 += 8;


      const __m128i vprod5x01234567lo = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);
      const __m128i vprod5x01234567hi = _mm_mulhi_epi16(vxi5x01234567, vxk5x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod5x01234567lo, vprod5x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod5x01234567lo, vprod5x01234567hi));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vxi6x01234567 = _mm_cvtepu8_epi16(vi6x01234567);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(uint8_t)));
      const __m128i vxk6x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk6x01234567), vk_zero_point);
      i6 += 8;


      const __m128i vprod6x01234567lo = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
      const __m128i vprod6x01234567hi = _mm_mulhi_epi16(vxi6x01234567, vxk6x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod6x01234567lo, vprod6x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod6x01234567lo, vprod6x01234567hi));

      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vxi7x01234567 = _mm_cvtepu8_epi16(vi7x01234567);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(uint8_t)));
      const __m128i vxk7x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk7x01234567), vk_zero_point);
      i7 += 8;


      const __m128i vprod7x01234567lo = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);
      const __m128i vprod7x01234567hi = _mm_mulhi_epi16(vxi7x01234567, vxk7x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod7x01234567lo, vprod7x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod7x01234567lo, vprod7x01234567hi));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vxi8x01234567 = _mm_cvtepu8_epi16(vi8x01234567);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(uint8_t)));
      const __m128i vxk8x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk8x01234567), vk_zero_point);
      i8 += 8;


      const __m128i vprod8x01234567lo = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
      const __m128i vprod8x01234567hi = _mm_mulhi_epi16(vxi8x01234567, vxk8x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod8x01234567lo, vprod8x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod8x01234567lo, vprod8x01234567hi));

      const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
      const __m128i vxi9x01234567 = _mm_cvtepu8_epi16(vi9x01234567);
      const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(uint8_t)));
      const __m128i vxk9x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk9x01234567), vk_zero_point);
      i9 += 8;


      const __m128i vprod9x01234567lo = _mm_mullo_epi16(vxi9x01234567, vxk9x01234567);
      const __m128i vprod9x01234567hi = _mm_mulhi_epi16(vxi9x01234567, vxk9x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod9x01234567lo, vprod9x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod9x01234567lo, vprod9x01234567hi));

      const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
      const __m128i vxi10x01234567 = _mm_cvtepu8_epi16(vi10x01234567);
      const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 80 * sizeof(uint8_t)));
      const __m128i vxk10x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk10x01234567), vk_zero_point);
      i10 += 8;


      const __m128i vprod10x01234567lo = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);
      const __m128i vprod10x01234567hi = _mm_mulhi_epi16(vxi10x01234567, vxk10x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod10x01234567lo, vprod10x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod10x01234567lo, vprod10x01234567hi));

      const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
      const __m128i vxi11x01234567 = _mm_cvtepu8_epi16(vi11x01234567);
      const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 88 * sizeof(uint8_t)));
      const __m128i vxk11x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk11x01234567), vk_zero_point);
      i11 += 8;


      const __m128i vprod11x01234567lo = _mm_mullo_epi16(vxi11x01234567, vxk11x01234567);
      const __m128i vprod11x01234567hi = _mm_mulhi_epi16(vxi11x01234567, vxk11x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod11x01234567lo, vprod11x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod11x01234567lo, vprod11x01234567hi));

      const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
      const __m128i vxi12x01234567 = _mm_cvtepu8_epi16(vi12x01234567);
      const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 96 * sizeof(uint8_t)));
      const __m128i vxk12x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk12x01234567), vk_zero_point);
      i12 += 8;


      const __m128i vprod12x01234567lo = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);
      const __m128i vprod12x01234567hi = _mm_mulhi_epi16(vxi12x01234567, vxk12x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod12x01234567lo, vprod12x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod12x01234567lo, vprod12x01234567hi));

      const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
      const __m128i vxi13x01234567 = _mm_cvtepu8_epi16(vi13x01234567);
      const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 104 * sizeof(uint8_t)));
      const __m128i vxk13x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk13x01234567), vk_zero_point);
      i13 += 8;


      const __m128i vprod13x01234567lo = _mm_mullo_epi16(vxi13x01234567, vxk13x01234567);
      const __m128i vprod13x01234567hi = _mm_mulhi_epi16(vxi13x01234567, vxk13x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod13x01234567lo, vprod13x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod13x01234567lo, vprod13x01234567hi));

      const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
      const __m128i vxi14x01234567 = _mm_cvtepu8_epi16(vi14x01234567);
      const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 112 * sizeof(uint8_t)));
      const __m128i vxk14x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk14x01234567), vk_zero_point);
      i14 += 8;


      const __m128i vprod14x01234567lo = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);
      const __m128i vprod14x01234567hi = _mm_mulhi_epi16(vxi14x01234567, vxk14x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod14x01234567lo, vprod14x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod14x01234567lo, vprod14x01234567hi));

      const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
      const __m128i vxi15x01234567 = _mm_cvtepu8_epi16(vi15x01234567);
      const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 120 * sizeof(uint8_t)));
      const __m128i vxk15x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk15x01234567), vk_zero_point);
      i15 += 8;


      const __m128i vprod15x01234567lo = _mm_mullo_epi16(vxi15x01234567, vxk15x01234567);
      const __m128i vprod15x01234567hi = _mm_mulhi_epi16(vxi15x01234567, vxk15x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod15x01234567lo, vprod15x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod15x01234567lo, vprod15x01234567hi));

      const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
      const __m128i vxi16x01234567 = _mm_cvtepu8_epi16(vi16x01234567);
      const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 128 * sizeof(uint8_t)));
      const __m128i vxk16x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk16x01234567), vk_zero_point);
      i16 += 8;


      const __m128i vprod16x01234567lo = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);
      const __m128i vprod16x01234567hi = _mm_mulhi_epi16(vxi16x01234567, vxk16x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod16x01234567lo, vprod16x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod16x01234567lo, vprod16x01234567hi));

      const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
      const __m128i vxi17x01234567 = _mm_cvtepu8_epi16(vi17x01234567);
      const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 136 * sizeof(uint8_t)));
      const __m128i vxk17x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk17x01234567), vk_zero_point);
      i17 += 8;


      const __m128i vprod17x01234567lo = _mm_mullo_epi16(vxi17x01234567, vxk17x01234567);
      const __m128i vprod17x01234567hi = _mm_mulhi_epi16(vxi17x01234567, vxk17x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod17x01234567lo, vprod17x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod17x01234567lo, vprod17x01234567hi));

      const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
      const __m128i vxi18x01234567 = _mm_cvtepu8_epi16(vi18x01234567);
      const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 144 * sizeof(uint8_t)));
      const __m128i vxk18x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk18x01234567), vk_zero_point);
      i18 += 8;


      const __m128i vprod18x01234567lo = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);
      const __m128i vprod18x01234567hi = _mm_mulhi_epi16(vxi18x01234567, vxk18x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod18x01234567lo, vprod18x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod18x01234567lo, vprod18x01234567hi));

      const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
      const __m128i vxi19x01234567 = _mm_cvtepu8_epi16(vi19x01234567);
      const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 152 * sizeof(uint8_t)));
      const __m128i vxk19x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk19x01234567), vk_zero_point);
      i19 += 8;


      const __m128i vprod19x01234567lo = _mm_mullo_epi16(vxi19x01234567, vxk19x01234567);
      const __m128i vprod19x01234567hi = _mm_mulhi_epi16(vxi19x01234567, vxk19x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod19x01234567lo, vprod19x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod19x01234567lo, vprod19x01234567hi));

      const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
      const __m128i vxi20x01234567 = _mm_cvtepu8_epi16(vi20x01234567);
      const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 160 * sizeof(uint8_t)));
      const __m128i vxk20x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk20x01234567), vk_zero_point);
      i20 += 8;


      const __m128i vprod20x01234567lo = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);
      const __m128i vprod20x01234567hi = _mm_mulhi_epi16(vxi20x01234567, vxk20x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod20x01234567lo, vprod20x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod20x01234567lo, vprod20x01234567hi));

      const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
      const __m128i vxi21x01234567 = _mm_cvtepu8_epi16(vi21x01234567);
      const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 168 * sizeof(uint8_t)));
      const __m128i vxk21x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk21x01234567), vk_zero_point);
      i21 += 8;


      const __m128i vprod21x01234567lo = _mm_mullo_epi16(vxi21x01234567, vxk21x01234567);
      const __m128i vprod21x01234567hi = _mm_mulhi_epi16(vxi21x01234567, vxk21x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod21x01234567lo, vprod21x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod21x01234567lo, vprod21x01234567hi));

      const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
      const __m128i vxi22x01234567 = _mm_cvtepu8_epi16(vi22x01234567);
      const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 176 * sizeof(uint8_t)));
      const __m128i vxk22x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk22x01234567), vk_zero_point);
      i22 += 8;


      const __m128i vprod22x01234567lo = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);
      const __m128i vprod22x01234567hi = _mm_mulhi_epi16(vxi22x01234567, vxk22x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod22x01234567lo, vprod22x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod22x01234567lo, vprod22x01234567hi));

      const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
      const __m128i vxi23x01234567 = _mm_cvtepu8_epi16(vi23x01234567);
      const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 184 * sizeof(uint8_t)));
      const __m128i vxk23x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk23x01234567), vk_zero_point);
      i23 += 8;


      const __m128i vprod23x01234567lo = _mm_mullo_epi16(vxi23x01234567, vxk23x01234567);
      const __m128i vprod23x01234567hi = _mm_mulhi_epi16(vxi23x01234567, vxk23x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod23x01234567lo, vprod23x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod23x01234567lo, vprod23x01234567hi));

      const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
      const __m128i vxi24x01234567 = _mm_cvtepu8_epi16(vi24x01234567);
      const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 192 * sizeof(uint8_t)));
      const __m128i vxk24x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk24x01234567), vk_zero_point);
      i24 += 8;


      const __m128i vprod24x01234567lo = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);
      const __m128i vprod24x01234567hi = _mm_mulhi_epi16(vxi24x01234567, vxk24x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod24x01234567lo, vprod24x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod24x01234567lo, vprod24x01234567hi));

      w = (const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 200 * sizeof(uint8_t));

      __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

      const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
      vscaled0123 = _mm_mul_ps(vscaled0123, vscale);
      vscaled4567 = _mm_mul_ps(vscaled4567, vscale);

      const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
      vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
      vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vscaled0123);
      vacc4567 = _mm_cvtps_epi32(vscaled4567);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

      _mm_storel_epi64((__m128i*) output, vout0123456701234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vxi0x01234567 = _mm_cvtepu8_epi16(vi0x01234567);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(uint8_t)));
        const __m128i vxk0x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk0x01234567), vk_zero_point);


        const __m128i vprod0x01234567lo = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
        const __m128i vprod0x01234567hi = _mm_mulhi_epi16(vxi0x01234567, vxk0x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod0x01234567lo, vprod0x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod0x01234567lo, vprod0x01234567hi));

        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vxi1x01234567 = _mm_cvtepu8_epi16(vi1x01234567);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(uint8_t)));
        const __m128i vxk1x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk1x01234567), vk_zero_point);


        const __m128i vprod1x01234567lo = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);
        const __m128i vprod1x01234567hi = _mm_mulhi_epi16(vxi1x01234567, vxk1x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod1x01234567lo, vprod1x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod1x01234567lo, vprod1x01234567hi));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vxi2x01234567 = _mm_cvtepu8_epi16(vi2x01234567);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(uint8_t)));
        const __m128i vxk2x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk2x01234567), vk_zero_point);


        const __m128i vprod2x01234567lo = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
        const __m128i vprod2x01234567hi = _mm_mulhi_epi16(vxi2x01234567, vxk2x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod2x01234567lo, vprod2x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod2x01234567lo, vprod2x01234567hi));

        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vxi3x01234567 = _mm_cvtepu8_epi16(vi3x01234567);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(uint8_t)));
        const __m128i vxk3x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk3x01234567), vk_zero_point);


        const __m128i vprod3x01234567lo = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);
        const __m128i vprod3x01234567hi = _mm_mulhi_epi16(vxi3x01234567, vxk3x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod3x01234567lo, vprod3x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod3x01234567lo, vprod3x01234567hi));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vxi4x01234567 = _mm_cvtepu8_epi16(vi4x01234567);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(uint8_t)));
        const __m128i vxk4x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk4x01234567), vk_zero_point);


        const __m128i vprod4x01234567lo = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
        const __m128i vprod4x01234567hi = _mm_mulhi_epi16(vxi4x01234567, vxk4x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod4x01234567lo, vprod4x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod4x01234567lo, vprod4x01234567hi));

        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vxi5x01234567 = _mm_cvtepu8_epi16(vi5x01234567);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(uint8_t)));
        const __m128i vxk5x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk5x01234567), vk_zero_point);


        const __m128i vprod5x01234567lo = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);
        const __m128i vprod5x01234567hi = _mm_mulhi_epi16(vxi5x01234567, vxk5x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod5x01234567lo, vprod5x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod5x01234567lo, vprod5x01234567hi));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vxi6x01234567 = _mm_cvtepu8_epi16(vi6x01234567);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(uint8_t)));
        const __m128i vxk6x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk6x01234567), vk_zero_point);


        const __m128i vprod6x01234567lo = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
        const __m128i vprod6x01234567hi = _mm_mulhi_epi16(vxi6x01234567, vxk6x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod6x01234567lo, vprod6x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod6x01234567lo, vprod6x01234567hi));

        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vxi7x01234567 = _mm_cvtepu8_epi16(vi7x01234567);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(uint8_t)));
        const __m128i vxk7x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk7x01234567), vk_zero_point);


        const __m128i vprod7x01234567lo = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);
        const __m128i vprod7x01234567hi = _mm_mulhi_epi16(vxi7x01234567, vxk7x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod7x01234567lo, vprod7x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod7x01234567lo, vprod7x01234567hi));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vxi8x01234567 = _mm_cvtepu8_epi16(vi8x01234567);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(uint8_t)));
        const __m128i vxk8x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk8x01234567), vk_zero_point);


        const __m128i vprod8x01234567lo = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
        const __m128i vprod8x01234567hi = _mm_mulhi_epi16(vxi8x01234567, vxk8x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod8x01234567lo, vprod8x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod8x01234567lo, vprod8x01234567hi));

        const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
        const __m128i vxi9x01234567 = _mm_cvtepu8_epi16(vi9x01234567);
        const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(uint8_t)));
        const __m128i vxk9x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk9x01234567), vk_zero_point);


        const __m128i vprod9x01234567lo = _mm_mullo_epi16(vxi9x01234567, vxk9x01234567);
        const __m128i vprod9x01234567hi = _mm_mulhi_epi16(vxi9x01234567, vxk9x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod9x01234567lo, vprod9x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod9x01234567lo, vprod9x01234567hi));

        const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
        const __m128i vxi10x01234567 = _mm_cvtepu8_epi16(vi10x01234567);
        const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 80 * sizeof(uint8_t)));
        const __m128i vxk10x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk10x01234567), vk_zero_point);


        const __m128i vprod10x01234567lo = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);
        const __m128i vprod10x01234567hi = _mm_mulhi_epi16(vxi10x01234567, vxk10x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod10x01234567lo, vprod10x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod10x01234567lo, vprod10x01234567hi));

        const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
        const __m128i vxi11x01234567 = _mm_cvtepu8_epi16(vi11x01234567);
        const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 88 * sizeof(uint8_t)));
        const __m128i vxk11x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk11x01234567), vk_zero_point);


        const __m128i vprod11x01234567lo = _mm_mullo_epi16(vxi11x01234567, vxk11x01234567);
        const __m128i vprod11x01234567hi = _mm_mulhi_epi16(vxi11x01234567, vxk11x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod11x01234567lo, vprod11x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod11x01234567lo, vprod11x01234567hi));

        const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
        const __m128i vxi12x01234567 = _mm_cvtepu8_epi16(vi12x01234567);
        const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 96 * sizeof(uint8_t)));
        const __m128i vxk12x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk12x01234567), vk_zero_point);


        const __m128i vprod12x01234567lo = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);
        const __m128i vprod12x01234567hi = _mm_mulhi_epi16(vxi12x01234567, vxk12x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod12x01234567lo, vprod12x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod12x01234567lo, vprod12x01234567hi));

        const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
        const __m128i vxi13x01234567 = _mm_cvtepu8_epi16(vi13x01234567);
        const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 104 * sizeof(uint8_t)));
        const __m128i vxk13x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk13x01234567), vk_zero_point);


        const __m128i vprod13x01234567lo = _mm_mullo_epi16(vxi13x01234567, vxk13x01234567);
        const __m128i vprod13x01234567hi = _mm_mulhi_epi16(vxi13x01234567, vxk13x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod13x01234567lo, vprod13x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod13x01234567lo, vprod13x01234567hi));

        const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
        const __m128i vxi14x01234567 = _mm_cvtepu8_epi16(vi14x01234567);
        const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 112 * sizeof(uint8_t)));
        const __m128i vxk14x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk14x01234567), vk_zero_point);


        const __m128i vprod14x01234567lo = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);
        const __m128i vprod14x01234567hi = _mm_mulhi_epi16(vxi14x01234567, vxk14x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod14x01234567lo, vprod14x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod14x01234567lo, vprod14x01234567hi));

        const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
        const __m128i vxi15x01234567 = _mm_cvtepu8_epi16(vi15x01234567);
        const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 120 * sizeof(uint8_t)));
        const __m128i vxk15x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk15x01234567), vk_zero_point);


        const __m128i vprod15x01234567lo = _mm_mullo_epi16(vxi15x01234567, vxk15x01234567);
        const __m128i vprod15x01234567hi = _mm_mulhi_epi16(vxi15x01234567, vxk15x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod15x01234567lo, vprod15x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod15x01234567lo, vprod15x01234567hi));

        const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
        const __m128i vxi16x01234567 = _mm_cvtepu8_epi16(vi16x01234567);
        const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 128 * sizeof(uint8_t)));
        const __m128i vxk16x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk16x01234567), vk_zero_point);


        const __m128i vprod16x01234567lo = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);
        const __m128i vprod16x01234567hi = _mm_mulhi_epi16(vxi16x01234567, vxk16x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod16x01234567lo, vprod16x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod16x01234567lo, vprod16x01234567hi));

        const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
        const __m128i vxi17x01234567 = _mm_cvtepu8_epi16(vi17x01234567);
        const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 136 * sizeof(uint8_t)));
        const __m128i vxk17x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk17x01234567), vk_zero_point);


        const __m128i vprod17x01234567lo = _mm_mullo_epi16(vxi17x01234567, vxk17x01234567);
        const __m128i vprod17x01234567hi = _mm_mulhi_epi16(vxi17x01234567, vxk17x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod17x01234567lo, vprod17x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod17x01234567lo, vprod17x01234567hi));

        const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
        const __m128i vxi18x01234567 = _mm_cvtepu8_epi16(vi18x01234567);
        const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 144 * sizeof(uint8_t)));
        const __m128i vxk18x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk18x01234567), vk_zero_point);


        const __m128i vprod18x01234567lo = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);
        const __m128i vprod18x01234567hi = _mm_mulhi_epi16(vxi18x01234567, vxk18x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod18x01234567lo, vprod18x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod18x01234567lo, vprod18x01234567hi));

        const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
        const __m128i vxi19x01234567 = _mm_cvtepu8_epi16(vi19x01234567);
        const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 152 * sizeof(uint8_t)));
        const __m128i vxk19x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk19x01234567), vk_zero_point);


        const __m128i vprod19x01234567lo = _mm_mullo_epi16(vxi19x01234567, vxk19x01234567);
        const __m128i vprod19x01234567hi = _mm_mulhi_epi16(vxi19x01234567, vxk19x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod19x01234567lo, vprod19x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod19x01234567lo, vprod19x01234567hi));

        const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
        const __m128i vxi20x01234567 = _mm_cvtepu8_epi16(vi20x01234567);
        const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 160 * sizeof(uint8_t)));
        const __m128i vxk20x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk20x01234567), vk_zero_point);


        const __m128i vprod20x01234567lo = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);
        const __m128i vprod20x01234567hi = _mm_mulhi_epi16(vxi20x01234567, vxk20x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod20x01234567lo, vprod20x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod20x01234567lo, vprod20x01234567hi));

        const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
        const __m128i vxi21x01234567 = _mm_cvtepu8_epi16(vi21x01234567);
        const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 168 * sizeof(uint8_t)));
        const __m128i vxk21x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk21x01234567), vk_zero_point);


        const __m128i vprod21x01234567lo = _mm_mullo_epi16(vxi21x01234567, vxk21x01234567);
        const __m128i vprod21x01234567hi = _mm_mulhi_epi16(vxi21x01234567, vxk21x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod21x01234567lo, vprod21x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod21x01234567lo, vprod21x01234567hi));

        const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
        const __m128i vxi22x01234567 = _mm_cvtepu8_epi16(vi22x01234567);
        const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 176 * sizeof(uint8_t)));
        const __m128i vxk22x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk22x01234567), vk_zero_point);


        const __m128i vprod22x01234567lo = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);
        const __m128i vprod22x01234567hi = _mm_mulhi_epi16(vxi22x01234567, vxk22x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod22x01234567lo, vprod22x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod22x01234567lo, vprod22x01234567hi));

        const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
        const __m128i vxi23x01234567 = _mm_cvtepu8_epi16(vi23x01234567);
        const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 184 * sizeof(uint8_t)));
        const __m128i vxk23x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk23x01234567), vk_zero_point);


        const __m128i vprod23x01234567lo = _mm_mullo_epi16(vxi23x01234567, vxk23x01234567);
        const __m128i vprod23x01234567hi = _mm_mulhi_epi16(vxi23x01234567, vxk23x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod23x01234567lo, vprod23x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod23x01234567lo, vprod23x01234567hi));

        const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
        const __m128i vxi24x01234567 = _mm_cvtepu8_epi16(vi24x01234567);
        const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 192 * sizeof(uint8_t)));
        const __m128i vxk24x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk24x01234567), vk_zero_point);


        const __m128i vprod24x01234567lo = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);
        const __m128i vprod24x01234567hi = _mm_mulhi_epi16(vxi24x01234567, vxk24x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod24x01234567lo, vprod24x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod24x01234567lo, vprod24x01234567hi));


        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

        const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
        vscaled0123 = _mm_mul_ps(vscaled0123, vscale);
        vscaled4567 = _mm_mul_ps(vscaled4567, vscale);

        const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
        vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
        vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

        vacc0123 = _mm_cvtps_epi32(vscaled0123);
        vacc4567 = _mm_cvtps_epi32(vscaled4567);


        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

        __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

        vout0123456701234567 = _mm_max_epu8(vout0123456701234567, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

        if (c & 4) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (c & 2) {
          unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (c & 1) {
          *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
          output += 1;
        }
      }
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qu8_dwconv_minmax_fp32_ukernel_9p8c__sse41_mul16(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const uint8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const uint8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    const __m128i vk_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.kernel_zero_point);
    for (; c >= 8; c -= 8) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vxi0x01234567 = _mm_cvtepu8_epi16(vi0x01234567);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(uint8_t)));
      const __m128i vxk0x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk0x01234567), vk_zero_point);
      i0 += 8;


      const __m128i vprod0x01234567lo = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
      const __m128i vprod0x01234567hi = _mm_mulhi_epi16(vxi0x01234567, vxk0x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod0x01234567lo, vprod0x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod0x01234567lo, vprod0x01234567hi));

      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vxi1x01234567 = _mm_cvtepu8_epi16(vi1x01234567);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(uint8_t)));
      const __m128i vxk1x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk1x01234567), vk_zero_point);
      i1 += 8;


      const __m128i vprod1x01234567lo = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);
      const __m128i vprod1x01234567hi = _mm_mulhi_epi16(vxi1x01234567, vxk1x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod1x01234567lo, vprod1x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod1x01234567lo, vprod1x01234567hi));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vxi2x01234567 = _mm_cvtepu8_epi16(vi2x01234567);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(uint8_t)));
      const __m128i vxk2x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk2x01234567), vk_zero_point);
      i2 += 8;


      const __m128i vprod2x01234567lo = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
      const __m128i vprod2x01234567hi = _mm_mulhi_epi16(vxi2x01234567, vxk2x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod2x01234567lo, vprod2x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod2x01234567lo, vprod2x01234567hi));

      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vxi3x01234567 = _mm_cvtepu8_epi16(vi3x01234567);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(uint8_t)));
      const __m128i vxk3x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk3x01234567), vk_zero_point);
      i3 += 8;


      const __m128i vprod3x01234567lo = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);
      const __m128i vprod3x01234567hi = _mm_mulhi_epi16(vxi3x01234567, vxk3x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod3x01234567lo, vprod3x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod3x01234567lo, vprod3x01234567hi));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vxi4x01234567 = _mm_cvtepu8_epi16(vi4x01234567);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(uint8_t)));
      const __m128i vxk4x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk4x01234567), vk_zero_point);
      i4 += 8;


      const __m128i vprod4x01234567lo = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
      const __m128i vprod4x01234567hi = _mm_mulhi_epi16(vxi4x01234567, vxk4x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod4x01234567lo, vprod4x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod4x01234567lo, vprod4x01234567hi));

      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vxi5x01234567 = _mm_cvtepu8_epi16(vi5x01234567);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(uint8_t)));
      const __m128i vxk5x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk5x01234567), vk_zero_point);
      i5 += 8;


      const __m128i vprod5x01234567lo = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);
      const __m128i vprod5x01234567hi = _mm_mulhi_epi16(vxi5x01234567, vxk5x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod5x01234567lo, vprod5x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod5x01234567lo, vprod5x01234567hi));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vxi6x01234567 = _mm_cvtepu8_epi16(vi6x01234567);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(uint8_t)));
      const __m128i vxk6x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk6x01234567), vk_zero_point);
      i6 += 8;


      const __m128i vprod6x01234567lo = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
      const __m128i vprod6x01234567hi = _mm_mulhi_epi16(vxi6x01234567, vxk6x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod6x01234567lo, vprod6x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod6x01234567lo, vprod6x01234567hi));

      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vxi7x01234567 = _mm_cvtepu8_epi16(vi7x01234567);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(uint8_t)));
      const __m128i vxk7x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk7x01234567), vk_zero_point);
      i7 += 8;


      const __m128i vprod7x01234567lo = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);
      const __m128i vprod7x01234567hi = _mm_mulhi_epi16(vxi7x01234567, vxk7x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod7x01234567lo, vprod7x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod7x01234567lo, vprod7x01234567hi));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vxi8x01234567 = _mm_cvtepu8_epi16(vi8x01234567);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(uint8_t)));
      const __m128i vxk8x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk8x01234567), vk_zero_point);
      i8 += 8;


      const __m128i vprod8x01234567lo = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
      const __m128i vprod8x01234567hi = _mm_mulhi_epi16(vxi8x01234567, vxk8x01234567);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod8x01234567lo, vprod8x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod8x01234567lo, vprod8x01234567hi));

      w = (const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(uint8_t));

      __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

      const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
      vscaled0123 = _mm_mul_ps(vscaled0123, vscale);
      vscaled4567 = _mm_mul_ps(vscaled4567, vscale);

      const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
      vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
      vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vscaled0123);
      vacc4567 = _mm_cvtps_epi32(vscaled4567);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

      _mm_storel_epi64((__m128i*) output, vout0123456701234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vxi0x01234567 = _mm_cvtepu8_epi16(vi0x01234567);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(uint8_t)));
        const __m128i vxk0x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk0x01234567), vk_zero_point);


        const __m128i vprod0x01234567lo = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
        const __m128i vprod0x01234567hi = _mm_mulhi_epi16(vxi0x01234567, vxk0x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod0x01234567lo, vprod0x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod0x01234567lo, vprod0x01234567hi));

        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vxi1x01234567 = _mm_cvtepu8_epi16(vi1x01234567);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(uint8_t)));
        const __m128i vxk1x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk1x01234567), vk_zero_point);


        const __m128i vprod1x01234567lo = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);
        const __m128i vprod1x01234567hi = _mm_mulhi_epi16(vxi1x01234567, vxk1x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod1x01234567lo, vprod1x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod1x01234567lo, vprod1x01234567hi));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vxi2x01234567 = _mm_cvtepu8_epi16(vi2x01234567);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(uint8_t)));
        const __m128i vxk2x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk2x01234567), vk_zero_point);


        const __m128i vprod2x01234567lo = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
        const __m128i vprod2x01234567hi = _mm_mulhi_epi16(vxi2x01234567, vxk2x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod2x01234567lo, vprod2x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod2x01234567lo, vprod2x01234567hi));

        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vxi3x01234567 = _mm_cvtepu8_epi16(vi3x01234567);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(uint8_t)));
        const __m128i vxk3x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk3x01234567), vk_zero_point);


        const __m128i vprod3x01234567lo = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);
        const __m128i vprod3x01234567hi = _mm_mulhi_epi16(vxi3x01234567, vxk3x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod3x01234567lo, vprod3x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod3x01234567lo, vprod3x01234567hi));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vxi4x01234567 = _mm_cvtepu8_epi16(vi4x01234567);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(uint8_t)));
        const __m128i vxk4x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk4x01234567), vk_zero_point);


        const __m128i vprod4x01234567lo = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
        const __m128i vprod4x01234567hi = _mm_mulhi_epi16(vxi4x01234567, vxk4x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod4x01234567lo, vprod4x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod4x01234567lo, vprod4x01234567hi));

        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vxi5x01234567 = _mm_cvtepu8_epi16(vi5x01234567);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(uint8_t)));
        const __m128i vxk5x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk5x01234567), vk_zero_point);


        const __m128i vprod5x01234567lo = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);
        const __m128i vprod5x01234567hi = _mm_mulhi_epi16(vxi5x01234567, vxk5x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod5x01234567lo, vprod5x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod5x01234567lo, vprod5x01234567hi));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vxi6x01234567 = _mm_cvtepu8_epi16(vi6x01234567);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(uint8_t)));
        const __m128i vxk6x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk6x01234567), vk_zero_point);


        const __m128i vprod6x01234567lo = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
        const __m128i vprod6x01234567hi = _mm_mulhi_epi16(vxi6x01234567, vxk6x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod6x01234567lo, vprod6x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod6x01234567lo, vprod6x01234567hi));

        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vxi7x01234567 = _mm_cvtepu8_epi16(vi7x01234567);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(uint8_t)));
        const __m128i vxk7x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk7x01234567), vk_zero_point);


        const __m128i vprod7x01234567lo = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);
        const __m128i vprod7x01234567hi = _mm_mulhi_epi16(vxi7x01234567, vxk7x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod7x01234567lo, vprod7x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod7x01234567lo, vprod7x01234567hi));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vxi8x01234567 = _mm_cvtepu8_epi16(vi8x01234567);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(uint8_t)));
        const __m128i vxk8x01234567 = _mm_sub_epi16(_mm_cvtepu8_epi16(vk8x01234567), vk_zero_point);


        const __m128i vprod8x01234567lo = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
        const __m128i vprod8x01234567hi = _mm_mulhi_epi16(vxi8x01234567, vxk8x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vprod8x01234567lo, vprod8x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vprod8x01234567lo, vprod8x01234567hi));


        __m128 vscaled0123 = _mm_cvtepi32_ps(vacc0123);
        __m128 vscaled4567 = _mm_cvtepi32_ps(vacc4567);

        const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
        vscaled0123 = _mm_mul_ps(vscaled0123, vscale);
        vscaled4567 = _mm_mul_ps(vscaled4567, vscale);

        const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
        vscaled0123 = _mm_min_ps(vscaled0123, voutput_max_less_zero_point);
        vscaled4567 = _mm_min_ps(vscaled4567, voutput_max_less_zero_point);

        vacc0123 = _mm_cvtps_epi32(vscaled0123);
        vacc4567 = _mm_cvtps_epi32(vscaled4567);


        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

        __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

        vout0123456701234567 = _mm_max_epu8(vout0123456701234567, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

        if (c & 4) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (c & 2) {
          unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (c & 1) {
          *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
          output += 1;
        }
      }
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qu8_f32_vcvt_ukernel__sse41_u16(
    size_t batch,
    const uint8_t* input,
    float* output,
    const union xnn_qu8_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vzero_point = _mm_set1_epi32(params->scalar.zero_point);
  const __m128 vscale = _mm_set1_ps(params->scalar.scale);
  XNN_FORCE_REALIZATION(vzero_point);
  XNN_FORCE_REALIZATION(vscale);
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    __m128i vx0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input)));
    __m128i vx4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input + 4)));
    __m128i vx89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input + 8)));
    __m128i vxCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input + 12)));
    input += 16;

    vx0123 = _mm_sub_epi32(vx0123, vzero_point);
    vx4567 = _mm_sub_epi32(vx4567, vzero_point);
    vx89AB = _mm_sub_epi32(vx89AB, vzero_point);
    vxCDEF = _mm_sub_epi32(vxCDEF, vzero_point);

    __m128 vy0123 = _mm_cvtepi32_ps(vx0123);
    __m128 vy4567 = _mm_cvtepi32_ps(vx4567);
    __m128 vy89AB = _mm_cvtepi32_ps(vx89AB);
    __m128 vyCDEF = _mm_cvtepi32_ps(vxCDEF);

    vy0123 = _mm_mul_ps(vy0123, vscale);
    vy4567 = _mm_mul_ps(vy4567, vscale);
    vy89AB = _mm_mul_ps(vy89AB, vscale);
    vyCDEF = _mm_mul_ps(vyCDEF, vscale);

    _mm_storeu_ps(output, vy0123);
    _mm_storeu_ps(output + 4, vy4567);
    _mm_storeu_ps(output + 8, vy89AB);
    _mm_storeu_ps(output + 12, vyCDEF);
    output += 16;
  }
  for (; batch >= 4 * sizeof(uint8_t); batch -= 4 * sizeof(uint8_t)) {
    __m128i vx = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input)));
    vx = _mm_sub_epi32(vx, vzero_point);
    input += 4;

    __m128 vy = _mm_cvtepi32_ps(vx);
    vy = _mm_mul_ps(vy, vscale);

    _mm_storeu_ps(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 3 * sizeof(uint8_t));

    __m128i vx = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_s32(input)));
    vx = _mm_sub_epi32(vx, vzero_point);

    __m128 vy = _mm_cvtepi32_ps(vx);
    vy = _mm_mul_ps(vy, vscale);

    if (batch & (2 * sizeof(uint8_t))) {
      _mm_storel_pi((__m64*) output, vy);
      vy = _mm_movehl_ps(vy, vy);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      _mm_store_ss(output, vy);
    }
  }
}

void xnn_qu8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* output,
    const union xnn_qu8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows > 7);
  assert(channels != 0);

  const uint8_t* i0 = input;
  const uint8_t* i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
  const uint8_t* i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
  const uint8_t* i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
  const uint8_t* i4 = (const uint8_t*) ((uintptr_t) i3 + input_stride);
  const uint8_t* i5 = (const uint8_t*) ((uintptr_t) i4 + input_stride);
  const uint8_t* i6 = (const uint8_t*) ((uintptr_t) i5 + input_stride);
  const size_t input_increment = 7 * input_stride - round_up_po2(channels, 8) * sizeof(uint8_t);

  const __m128i vinit_bias = _mm_load_si128((const __m128i*) params->fp32_sse4.init_bias);
  int32_t* b = buffer;
  size_t c = channels;
  for (; c != 0; c = doz(c, 8)) {
    const __m128i vxi0x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i0));
    i0 += 8;
    const __m128i vxi1x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i1));
    i1 += 8;

    __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
    const __m128i vxi2x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i2));
    i2 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
    const __m128i vxi3x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i3));
    i3 += 8;
    vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
    const __m128i vxi4x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i4));
    i4 += 8;
    vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
    const __m128i vxi5x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i5));
    i5 += 8;
    vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
    const __m128i vxi6x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i6));
    i6 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

    const __m128i vzero = _mm_setzero_si128();
    __m128i vacc0123 = _mm_cvtepu16_epi32(vacc01234567);
    __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, vzero);

    vacc0123 = _mm_add_epi32(vacc0123, vinit_bias);
    vacc4567 = _mm_add_epi32(vacc4567, vinit_bias);

    _mm_store_si128((__m128i*) b, vacc0123);
    _mm_store_si128((__m128i*) (b + 4), vacc4567);
    b += 8;
  }

  for (rows -= 7; rows > 7; rows -= 7) {
    i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
    i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
    i2 = (const uint8_t*) ((uintptr_t) i2 + input_increment);
    i3 = (const uint8_t*) ((uintptr_t) i3 + input_increment);
    i4 = (const uint8_t*) ((uintptr_t) i4 + input_increment);
    i5 = (const uint8_t*) ((uintptr_t) i5 + input_increment);
    i6 = (const uint8_t*) ((uintptr_t) i6 + input_increment);

    int32_t* b = buffer;
    size_t c = channels;
    for (; c != 0; c = doz(c, 8)) {
      const __m128i vxi0x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i0));
      i0 += 8;
      const __m128i vxi1x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i1));
      i1 += 8;

      __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
      const __m128i vxi2x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i2));
      i2 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
      const __m128i vxi3x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i3));
      i3 += 8;
      vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
      const __m128i vxi4x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i4));
      i4 += 8;
      vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
      const __m128i vxi5x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i5));
      i5 += 8;
      vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
      const __m128i vxi6x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i6));
      i6 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

      const __m128i vzero = _mm_setzero_si128();
      __m128i vacc0123 = _mm_cvtepu16_epi32(vacc01234567);
      __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, vzero);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_load_si128((const __m128i*) b));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_load_si128((const __m128i*) (b + 4)));

      _mm_store_si128((__m128i*) b, vacc0123);
      _mm_store_si128((__m128i*) (b + 4), vacc4567);
      b += 8;
    }
  }

  i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
  i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  i2 = (const uint8_t*) ((uintptr_t) i2 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  i3 = (const uint8_t*) ((uintptr_t) i3 + input_increment);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  i4 = (const uint8_t*) ((uintptr_t) i4 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  i5 = (const uint8_t*) ((uintptr_t) i5 + input_increment);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  i6 = (const uint8_t*) ((uintptr_t) i6 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
  const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse4.output_min);
  for (; channels >= 8; channels -= 8) {
    const __m128i vxi0x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i0));
    i0 += 8;
    const __m128i vxi1x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i1));
    i1 += 8;

    __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
    const __m128i vxi2x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i2));
    i2 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
    const __m128i vxi3x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i3));
    i3 += 8;
    vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
    const __m128i vxi4x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i4));
    i4 += 8;
    vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
    const __m128i vxi5x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i5));
    i5 += 8;
    vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
    const __m128i vxi6x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i6));
    i6 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

    const __m128i vzero = _mm_setzero_si128();
    __m128i vacc0123 = _mm_cvtepu16_epi32(vacc01234567);
    __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, vzero);

    vacc0123 = _mm_add_epi32(vacc0123, _mm_load_si128((const __m128i*) buffer));
    vacc4567 = _mm_add_epi32(vacc4567, _mm_load_si128((const __m128i*) (buffer + 4)));
    buffer += 8;

    __m128 vfpacc0123 = _mm_cvtepi32_ps(vacc0123);
    __m128 vfpacc4567 = _mm_cvtepi32_ps(vacc4567);

    vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
    vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

    vfpacc0123 = _mm_min_ps(vfpacc0123, voutput_max_less_zero_point);
    vfpacc4567 = _mm_min_ps(vfpacc4567, voutput_max_less_zero_point);

    vacc0123 = _mm_cvtps_epi32(vfpacc0123);
    vacc4567 = _mm_cvtps_epi32(vfpacc4567);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

    __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

    vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    {
      const __m128i vxi0x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i0));
      i0 += 8;
      const __m128i vxi1x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i1));
      i1 += 8;

      __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
      const __m128i vxi2x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i2));
      i2 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
      const __m128i vxi3x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i3));
      i3 += 8;
      vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
      const __m128i vxi4x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i4));
      i4 += 8;
      vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
      const __m128i vxi5x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i5));
      i5 += 8;
      vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
      const __m128i vxi6x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i6));
      i6 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

      __m128i vacc0123 = _mm_cvtepu16_epi32(vacc01234567);
      __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, _mm_setzero_si128());

      vacc0123 = _mm_add_epi32(vacc0123, _mm_load_si128((const __m128i*) buffer));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_load_si128((const __m128i*) (buffer + 4)));
      buffer += 8;

      __m128 vfpacc0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vfpacc4567 = _mm_cvtepi32_ps(vacc4567);

      vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
      vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

      vfpacc0123 = _mm_min_ps(vfpacc0123, voutput_max_less_zero_point);
      vfpacc4567 = _mm_min_ps(vfpacc4567, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vfpacc0123);
      vacc4567 = _mm_cvtps_epi32(vfpacc4567);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

      if (channels & 4) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (channels & 2) {
        unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (channels & 1) {
        *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
      }
    }
  }
}

void xnn_qu8_gavgpool_minmax_fp32_ukernel_7x__sse41_c8(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint8_t* output,
    const union xnn_qu8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const uint8_t* i0 = input;
  const uint8_t* i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  const uint8_t* i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  const uint8_t* i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  const uint8_t* i4 = (const uint8_t*) ((uintptr_t) i3 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  const uint8_t* i5 = (const uint8_t*) ((uintptr_t) i4 + input_stride);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  const uint8_t* i6 = (const uint8_t*) ((uintptr_t) i5 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const __m128i vinit_bias = _mm_load_si128((const __m128i*) params->fp32_sse4.init_bias);
  const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
  const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse4.output_min);
  for (; channels >= 8; channels -= 8) {
    const __m128i vxi0x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i0));
    i0 += 8;
    const __m128i vxi1x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i1));
    i1 += 8;

    __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
    const __m128i vxi2x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i2));
    i2 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
    const __m128i vxi3x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i3));
    i3 += 8;
    vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
    const __m128i vxi4x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i4));
    i4 += 8;
    vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
    const __m128i vxi5x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i5));
    i5 += 8;
    vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
    const __m128i vxi6x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i6));
    i6 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

    const __m128i vzero = _mm_setzero_si128();
    __m128i vacc0123 = _mm_cvtepu16_epi32(vacc01234567);
    __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, vzero);

    vacc0123 = _mm_add_epi32(vacc0123, vinit_bias);
    vacc4567 = _mm_add_epi32(vacc4567, vinit_bias);

    __m128 vfpacc0123 = _mm_cvtepi32_ps(vacc0123);
    __m128 vfpacc4567 = _mm_cvtepi32_ps(vacc4567);

    vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
    vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

    vfpacc0123 = _mm_min_ps(vfpacc0123, voutput_max_less_zero_point);
    vfpacc4567 = _mm_min_ps(vfpacc4567, voutput_max_less_zero_point);

    vacc0123 = _mm_cvtps_epi32(vfpacc0123);
    vacc4567 = _mm_cvtps_epi32(vfpacc4567);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

    __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

    vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    {
      const __m128i vxi0x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i0));
      i0 += 8;
      const __m128i vxi1x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i1));
      i1 += 8;

      __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
      const __m128i vxi2x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i2));
      i2 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
      const __m128i vxi3x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i3));
      i3 += 8;
      vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
      const __m128i vxi4x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i4));
      i4 += 8;
      vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
      const __m128i vxi5x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i5));
      i5 += 8;
      vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
      const __m128i vxi6x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i6));
      i6 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

      __m128i vacc0123 = _mm_cvtepu16_epi32(vacc01234567);
      __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, _mm_setzero_si128());

      vacc0123 = _mm_add_epi32(vacc0123, vinit_bias);
      vacc4567 = _mm_add_epi32(vacc4567, vinit_bias);

      __m128 vfpacc0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vfpacc4567 = _mm_cvtepi32_ps(vacc4567);

      vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
      vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

      vfpacc0123 = _mm_min_ps(vfpacc0123, voutput_max_less_zero_point);
      vfpacc4567 = _mm_min_ps(vfpacc4567, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vfpacc0123);
      vacc4567 = _mm_cvtps_epi32(vfpacc4567);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

      if (channels & 4) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (channels & 2) {
        unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (channels & 1) {
        *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
      }
    }
  }
}

void xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(uint8_t));
  const uint8_t* a0 = a;
  uint8_t* c0 = c;


  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    w = (const int32_t*) w + 4;

    const __m128i vb_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.kernel_zero_point);
    size_t k = kc;


    while (k >= 8 * sizeof(uint8_t)) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_cvtepu8_epi16(va0);
      a0 += 8;

      const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);

      const __m128i vxb0 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb0), vb_zero_point);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 8));

      const __m128i vxb1 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb1), vb_zero_point);

      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 16));

      const __m128i vxb2 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb2), vb_zero_point);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 24));

      const __m128i vxb3 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb3), vb_zero_point);

      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));

      w = (const uint8_t*) w + 32;
      k -= 8 * sizeof(uint8_t);
    }

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
    __m128i vacc00x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc0x0123), voutput_zero_point);

    __m128i vout = _mm_packus_epi16(vacc00x0123, vacc00x0123);

    vout = _mm_max_epu8(vout, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

    if (nc >= 4) {
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (uint8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(uint8_t));
  const uint8_t* a0 = a;
  uint8_t* c0 = c;
  const uint8_t* a1 = (const uint8_t*) ((uintptr_t) a0 + a_stride);
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const uint8_t* a2 = (const uint8_t*) ((uintptr_t) a1 + a_stride);
  uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }


  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m128i vacc1x0 = vacc0x0;
    __m128i vacc1x1 = vacc0x1;
    __m128i vacc1x2 = vacc0x2;
    __m128i vacc1x3 = vacc0x3;
    __m128i vacc2x0 = vacc0x0;
    __m128i vacc2x1 = vacc0x1;
    __m128i vacc2x2 = vacc0x2;
    __m128i vacc2x3 = vacc0x3;
    w = (const int32_t*) w + 4;

    const __m128i vb_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.kernel_zero_point);
    size_t k = kc;


    while (k >= 8 * sizeof(uint8_t)) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_cvtepu8_epi16(va0);
      a0 += 8;
      const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
      const __m128i vxa1 = _mm_cvtepu8_epi16(va1);
      a1 += 8;
      const __m128i va2 = _mm_loadl_epi64((const __m128i*) a2);
      const __m128i vxa2 = _mm_cvtepu8_epi16(va2);
      a2 += 8;

      const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);

      const __m128i vxb0 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb0), vb_zero_point);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
      vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2, vxb0));
      const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 8));

      const __m128i vxb1 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb1), vb_zero_point);

      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
      vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2, vxb1));
      const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 16));

      const __m128i vxb2 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb2), vb_zero_point);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
      vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2, vxb2));
      const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 24));

      const __m128i vxb3 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb3), vb_zero_point);

      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
      vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));
      vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2, vxb3));

      w = (const uint8_t*) w + 32;
      k -= 8 * sizeof(uint8_t);
    }

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);
    const __m128i vacc1x01 = _mm_hadd_epi32(vacc1x0, vacc1x1);
    const __m128i vacc1x23 = _mm_hadd_epi32(vacc1x2, vacc1x3);
    const __m128i vacc2x01 = _mm_hadd_epi32(vacc2x0, vacc2x1);
    const __m128i vacc2x23 = _mm_hadd_epi32(vacc2x2, vacc2x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
    __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);
    __m128i vacc2x0123 = _mm_hadd_epi32(vacc2x01, vacc2x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vscaled1x0123 = _mm_cvtepi32_ps(vacc1x0123);
    __m128 vscaled2x0123 = _mm_cvtepi32_ps(vacc2x0123);

    const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale);
    vscaled1x0123 = _mm_mul_ps(vscaled1x0123, vscale);
    vscaled2x0123 = _mm_mul_ps(vscaled2x0123, vscale);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);
    vscaled1x0123 = _mm_min_ps(vscaled1x0123, voutput_max_less_zero_point);
    vscaled2x0123 = _mm_min_ps(vscaled2x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);
    vacc1x0123 = _mm_cvtps_epi32(vscaled1x0123);
    vacc2x0123 = _mm_cvtps_epi32(vscaled2x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
    __m128i vacc01x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);
    __m128i vacc22x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc2x0123, vacc2x0123), voutput_zero_point);

    __m128i vout = _mm_packus_epi16(vacc01x0123, vacc22x0123);

    vout = _mm_max_epu8(vout, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

    if (nc >= 4) {
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));
      unaligned_store_u32(c1, (uint32_t) _mm_extract_epi32(vout, 1));
      unaligned_store_u32(c2, (uint32_t) _mm_extract_epi32(vout, 2));

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (uint8_t*) ((uintptr_t) c2 + cn_stride);

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint8_t*) ((uintptr_t) a1 - kc);
      a2 = (const uint8_t*) ((uintptr_t) a2 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(vout, 2));
        c1 += 2;
        unaligned_store_u16(c2, (uint16_t) _mm_extract_epi16(vout, 4));
        c2 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (uint8_t) _mm_extract_epi8(vout, 0);
        *c1 = (uint8_t) _mm_extract_epi8(vout, 4);
        *c2 = (uint8_t) _mm_extract_epi8(vout, 8);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const uint8_t** restrict a,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(uint8_t));
  uint8_t* c0 = c;

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const uint8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = 0;
      const __m128i vb_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.kernel_zero_point);
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_cvtepu8_epi16(va0);
        a0 += 8;

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
        const __m128i vxb0 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb0), vb_zero_point);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 8));
        const __m128i vxb1 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb1), vb_zero_point);

        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 16));
        const __m128i vxb2 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb2), vb_zero_point);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 24));
        const __m128i vxb3 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb3), vb_zero_point);

        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));

        w = (const void*) ((const uint8_t*) w + 32);
        k += 8 * sizeof(uint8_t);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
    __m128i vacc00x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc0x0123), voutput_zero_point);

    __m128i vout = _mm_packus_epi16(vacc00x0123, vacc00x0123);

    vout = _mm_max_epu8(vout, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

    if (nc >= 4) {
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));
      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (uint8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__sse41_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const uint8_t** restrict a,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (3 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(uint8_t));
  uint8_t* c0 = c;
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m128i vacc1x0 = vacc0x0;
    __m128i vacc1x1 = vacc0x1;
    __m128i vacc1x2 = vacc0x2;
    __m128i vacc1x3 = vacc0x3;
    __m128i vacc2x0 = vacc0x0;
    __m128i vacc2x1 = vacc0x1;
    __m128i vacc2x2 = vacc0x2;
    __m128i vacc2x3 = vacc0x3;
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const uint8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      const uint8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const uint8_t*) ((uintptr_t) a1 + a_offset);
      }
      const uint8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const uint8_t*) ((uintptr_t) a2 + a_offset);
      }
      a += 3;

      size_t k = 0;
      const __m128i vb_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.kernel_zero_point);
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_cvtepu8_epi16(va0);
        a0 += 8;
        const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
        const __m128i vxa1 = _mm_cvtepu8_epi16(va1);
        a1 += 8;
        const __m128i va2 = _mm_loadl_epi64((const __m128i*) a2);
        const __m128i vxa2 = _mm_cvtepu8_epi16(va2);
        a2 += 8;

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
        const __m128i vxb0 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb0), vb_zero_point);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
        vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2, vxb0));
        const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 8));
        const __m128i vxb1 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb1), vb_zero_point);

        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
        vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2, vxb1));
        const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 16));
        const __m128i vxb2 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb2), vb_zero_point);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
        vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2, vxb2));
        const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 24));
        const __m128i vxb3 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb3), vb_zero_point);

        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
        vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));
        vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2, vxb3));

        w = (const void*) ((const uint8_t*) w + 32);
        k += 8 * sizeof(uint8_t);
      }
      p -= 3 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);
    const __m128i vacc1x01 = _mm_hadd_epi32(vacc1x0, vacc1x1);
    const __m128i vacc1x23 = _mm_hadd_epi32(vacc1x2, vacc1x3);
    const __m128i vacc2x01 = _mm_hadd_epi32(vacc2x0, vacc2x1);
    const __m128i vacc2x23 = _mm_hadd_epi32(vacc2x2, vacc2x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
    __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);
    __m128i vacc2x0123 = _mm_hadd_epi32(vacc2x01, vacc2x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vscaled1x0123 = _mm_cvtepi32_ps(vacc1x0123);
    __m128 vscaled2x0123 = _mm_cvtepi32_ps(vacc2x0123);

    const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale);
    vscaled1x0123 = _mm_mul_ps(vscaled1x0123, vscale);
    vscaled2x0123 = _mm_mul_ps(vscaled2x0123, vscale);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);
    vscaled1x0123 = _mm_min_ps(vscaled1x0123, voutput_max_less_zero_point);
    vscaled2x0123 = _mm_min_ps(vscaled2x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);
    vacc1x0123 = _mm_cvtps_epi32(vscaled1x0123);
    vacc2x0123 = _mm_cvtps_epi32(vscaled2x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
    __m128i vacc01x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);
    __m128i vacc22x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc2x0123, vacc2x0123), voutput_zero_point);

    __m128i vout = _mm_packus_epi16(vacc01x0123, vacc22x0123);

    vout = _mm_max_epu8(vout, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

    if (nc >= 4) {
      unaligned_store_u32(c2, (uint32_t) _mm_extract_epi32(vout, 2));
      c2 = (uint8_t*) ((uintptr_t) c2 + cn_stride);
      unaligned_store_u32(c1, (uint32_t) _mm_extract_epi32(vout, 1));
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));
      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c2, (uint16_t) _mm_extract_epi16(vout, 4));
        c2 += 2;
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(vout, 2));
        c1 += 2;
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c2 = (uint8_t) _mm_extract_epi8(vout, 8);
        *c1 = (uint8_t) _mm_extract_epi8(vout, 4);
        *c0 = (uint8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_vadd_minmax_ukernel__sse41_mul16_ld64_u8(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128i vbias = _mm_set1_epi32(params->scalar.bias);
  const __m128i va_multiplier_lo = _mm_set1_epi16(params->scalar.a_multiplier);
  const __m128i va_multiplier_hi = _mm_set1_epi16((uint32_t)params->scalar.a_multiplier >> 16);
  const __m128i vb_multiplier_lo = _mm_set1_epi16(params->scalar.b_multiplier);
  const __m128i vb_multiplier_hi = _mm_set1_epi16((uint32_t)params->scalar.b_multiplier >> 16);
  const __m128i vshift = _mm_cvtsi32_si128((int) params->scalar.shift);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8(params->scalar.output_min);
  const __m128i voutput_max = _mm_set1_epi8(params->scalar.output_max);

  XNN_FORCE_REALIZATION(vbias);
  XNN_FORCE_REALIZATION(va_multiplier_lo);
  XNN_FORCE_REALIZATION(va_multiplier_hi);
  XNN_FORCE_REALIZATION(vb_multiplier_lo);
  XNN_FORCE_REALIZATION(vb_multiplier_hi);
  XNN_FORCE_REALIZATION(vshift);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    const __m128i va01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
    const __m128i vb01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input_b));
    input_a += 8;
    input_b += 8;


    __m128i vaprod01234567hi = _mm_mulhi_epu16(va01234567, va_multiplier_lo);
    __m128i vbprod01234567hi = _mm_mulhi_epu16(vb01234567, vb_multiplier_lo);
    const __m128i vaprod01234567lo = _mm_mullo_epi16(va01234567, va_multiplier_lo);
    const __m128i vbprod01234567lo = _mm_mullo_epi16(vb01234567, vb_multiplier_lo);

    vaprod01234567hi = _mm_add_epi16(vaprod01234567hi, _mm_mullo_epi16(va01234567, va_multiplier_hi));
    vbprod01234567hi = _mm_add_epi16(vbprod01234567hi, _mm_mullo_epi16(vb01234567, vb_multiplier_hi));


    __m128i vacc0123 = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vaprod01234567lo, vaprod01234567hi));
    __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vaprod01234567lo, vaprod01234567hi));

    vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vbprod01234567lo, vbprod01234567hi));
    vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vbprod01234567lo, vbprod01234567hi));

    vacc0123 = _mm_sra_epi32(vacc0123, vshift);
    vacc4567 = _mm_sra_epi32(vacc4567, vshift);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


    __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

    vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

    vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      const __m128i va01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
      const __m128i vb01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input_b));


      __m128i vaprod01234567hi = _mm_mulhi_epu16(va01234567, va_multiplier_lo);
      __m128i vbprod01234567hi = _mm_mulhi_epu16(vb01234567, vb_multiplier_lo);
      const __m128i vaprod01234567lo = _mm_mullo_epi16(va01234567, va_multiplier_lo);
      const __m128i vbprod01234567lo = _mm_mullo_epi16(vb01234567, vb_multiplier_lo);

      vaprod01234567hi = _mm_add_epi16(vaprod01234567hi, _mm_mullo_epi16(va01234567, va_multiplier_hi));
      vbprod01234567hi = _mm_add_epi16(vbprod01234567hi, _mm_mullo_epi16(vb01234567, vb_multiplier_hi));


      __m128i vacc0123 = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vaprod01234567lo, vaprod01234567hi));
      __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vaprod01234567lo, vaprod01234567hi));

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vbprod01234567lo, vbprod01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vbprod01234567lo, vbprod01234567hi));

      vacc0123 = _mm_sra_epi32(vacc0123, vshift);
      vacc4567 = _mm_sra_epi32(vacc4567, vshift);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

      if (batch & (4 * sizeof(uint8_t))) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (batch & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (batch & (1 * sizeof(uint8_t))) {
        *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
      }
    }
  }
}

void xnn_qu8_vaddc_minmax_ukernel__sse41_mul16_ld64_u8(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128i vbias = _mm_set1_epi32(params->scalar.b_multiplier * (int32_t) *input_b + params->scalar.bias);
  const __m128i va_multiplier_lo = _mm_set1_epi16(params->scalar.a_multiplier);
  const __m128i va_multiplier_hi = _mm_set1_epi16((uint32_t)params->scalar.a_multiplier >> 16);
  const __m128i vshift = _mm_cvtsi32_si128((int) params->scalar.shift);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8(params->scalar.output_min);
  const __m128i voutput_max = _mm_set1_epi8(params->scalar.output_max);

  XNN_FORCE_REALIZATION(vbias);
  XNN_FORCE_REALIZATION(va_multiplier_lo);
  XNN_FORCE_REALIZATION(va_multiplier_hi);
  XNN_FORCE_REALIZATION(vshift);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    const __m128i va01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
    input_a += 8;


    __m128i vaprod01234567hi = _mm_mulhi_epu16(va01234567, va_multiplier_lo);
    const __m128i vaprod01234567lo = _mm_mullo_epi16(va01234567, va_multiplier_lo);

    vaprod01234567hi = _mm_add_epi16(vaprod01234567hi, _mm_mullo_epi16(va01234567, va_multiplier_hi));


    __m128i vacc0123 = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vaprod01234567lo, vaprod01234567hi));
    __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vaprod01234567lo, vaprod01234567hi));

    vacc0123 = _mm_sra_epi32(vacc0123, vshift);
    vacc4567 = _mm_sra_epi32(vacc4567, vshift);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);


    __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

    vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

    vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      const __m128i va01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input_a));


      __m128i vaprod01234567hi = _mm_mulhi_epu16(va01234567, va_multiplier_lo);
      const __m128i vaprod01234567lo = _mm_mullo_epi16(va01234567, va_multiplier_lo);

      vaprod01234567hi = _mm_add_epi16(vaprod01234567hi, _mm_mullo_epi16(va01234567, va_multiplier_hi));


      __m128i vacc0123 = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vaprod01234567lo, vaprod01234567hi));
      __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vaprod01234567lo, vaprod01234567hi));

      vacc0123 = _mm_sra_epi32(vacc0123, vshift);
      vacc4567 = _mm_sra_epi32(vacc4567, vshift);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

      if (batch & (4 * sizeof(uint8_t))) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (batch & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (batch & (1 * sizeof(uint8_t))) {
        *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
      }
    }
  }
}

void xnn_qu8_vcvt_ukernel__sse41_u32(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vinput_zero_point = _mm_set1_epi16(params->scalar.input_zero_point);
  const __m128i vmultiplier = _mm_set1_epi16(-params->scalar.multiplier);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(vmultiplier);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    __m128i vacc0 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input));
    __m128i vacc1 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) (input + 8)));
    __m128i vacc2 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) (input + 16)));
    __m128i vacc3 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) (input + 24)));
    input += 32;

    vacc0 = _mm_sub_epi16(vinput_zero_point, vacc0);
    vacc1 = _mm_sub_epi16(vinput_zero_point, vacc1);
    vacc2 = _mm_sub_epi16(vinput_zero_point, vacc2);
    vacc3 = _mm_sub_epi16(vinput_zero_point, vacc3);

    vacc0 = _mm_slli_epi16(vacc0, 7);
    vacc1 = _mm_slli_epi16(vacc1, 7);
    vacc2 = _mm_slli_epi16(vacc2, 7);
    vacc3 = _mm_slli_epi16(vacc3, 7);

    vacc0 = _mm_mulhrs_epi16(vacc0, vmultiplier);
    vacc1 = _mm_mulhrs_epi16(vacc1, vmultiplier);
    vacc2 = _mm_mulhrs_epi16(vacc2, vmultiplier);
    vacc3 = _mm_mulhrs_epi16(vacc3, vmultiplier);

    vacc0 = _mm_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm_adds_epi16(vacc1, voutput_zero_point);
    vacc2 = _mm_adds_epi16(vacc2, voutput_zero_point);
    vacc3 = _mm_adds_epi16(vacc3, voutput_zero_point);

    const __m128i vy0 = _mm_packus_epi16(vacc0, vacc1);
    const __m128i vy1 = _mm_packus_epi16(vacc2, vacc3);

    _mm_storeu_si128((__m128i*) output, vy0);
    _mm_storeu_si128((__m128i*) (output + 16), vy1);
    output += 32;
  }
  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    __m128i vacc = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input));
    vacc = _mm_sub_epi16(vinput_zero_point, vacc);
    vacc = _mm_slli_epi16(vacc, 7);
    vacc = _mm_mulhrs_epi16(vacc, vmultiplier);
    vacc = _mm_adds_epi16(vacc, voutput_zero_point);
    input += 8;

    const __m128i vy = _mm_packus_epi16(vacc, vacc);
    _mm_storel_epi64((__m128i*) output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 7 * sizeof(uint8_t));

    __m128i vacc = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input));
    vacc = _mm_sub_epi16(vinput_zero_point, vacc);
    vacc = _mm_slli_epi16(vacc, 7);
    vacc = _mm_mulhrs_epi16(vacc, vmultiplier);
    vacc = _mm_adds_epi16(vacc, voutput_zero_point);

    __m128i vy = _mm_packus_epi16(vacc, vacc);
    if (batch & (4 * sizeof(uint8_t))) {
      _mm_storeu_si32(output, vy);
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(uint8_t))) {
      _mm_storeu_si16(output, vy);
      vy = _mm_srli_epi32(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      *output = (uint8_t) _mm_extract_epi8(vy, 0);
    }
  }
}

void xnn_qu8_vlrelu_ukernel__sse41_u32(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vinput_zero_point = _mm_set1_epi16(params->scalar.input_zero_point);
  const __m128i vmultiplier_diff = _mm_set1_epi16(-params->scalar.negative_multiplier ^ -params->scalar.positive_multiplier);
  const __m128i vmultiplier_base = _mm_set1_epi16(-params->scalar.negative_multiplier);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vmultiplier_diff);
  XNN_FORCE_REALIZATION(vmultiplier_base);
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    __m128i vacc0 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input));
    __m128i vacc1 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) (input + 8)));
    __m128i vacc2 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) (input + 16)));
    __m128i vacc3 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) (input + 24)));
    input += 32;

    __m128i vmultiplier0 = _mm_cmpgt_epi16(vacc0, vinput_zero_point);
    vacc0 = _mm_sub_epi16(vinput_zero_point, vacc0);
    __m128i vmultiplier1 = _mm_cmpgt_epi16(vacc1, vinput_zero_point);
    vacc1 = _mm_sub_epi16(vinput_zero_point, vacc1);
    __m128i vmultiplier2 = _mm_cmpgt_epi16(vacc2, vinput_zero_point);
    vacc2 = _mm_sub_epi16(vinput_zero_point, vacc2);
    __m128i vmultiplier3 = _mm_cmpgt_epi16(vacc3, vinput_zero_point);
    vacc3 = _mm_sub_epi16(vinput_zero_point, vacc3);

    vmultiplier0 = _mm_and_si128(vmultiplier0, vmultiplier_diff);
    vacc0 = _mm_slli_epi16(vacc0, 7);
    vmultiplier0 = _mm_xor_si128(vmultiplier0, vmultiplier_base);
    vmultiplier1 = _mm_and_si128(vmultiplier1, vmultiplier_diff);
    vacc1 = _mm_slli_epi16(vacc1, 7);
    vmultiplier1 = _mm_xor_si128(vmultiplier1, vmultiplier_base);
    vmultiplier2 = _mm_and_si128(vmultiplier2, vmultiplier_diff);
    vacc2 = _mm_slli_epi16(vacc2, 7);
    vmultiplier2 = _mm_xor_si128(vmultiplier2, vmultiplier_base);
    vmultiplier3 = _mm_and_si128(vmultiplier3, vmultiplier_diff);
    vacc3 = _mm_slli_epi16(vacc3, 7);
    vmultiplier3 = _mm_xor_si128(vmultiplier3, vmultiplier_base);

    vacc0 = _mm_mulhrs_epi16(vacc0, vmultiplier0);
    vacc1 = _mm_mulhrs_epi16(vacc1, vmultiplier1);
    vacc2 = _mm_mulhrs_epi16(vacc2, vmultiplier2);
    vacc3 = _mm_mulhrs_epi16(vacc3, vmultiplier3);

    vacc0 = _mm_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm_adds_epi16(vacc1, voutput_zero_point);
    vacc2 = _mm_adds_epi16(vacc2, voutput_zero_point);
    vacc3 = _mm_adds_epi16(vacc3, voutput_zero_point);

    const __m128i vy0 = _mm_packus_epi16(vacc0, vacc1);
    const __m128i vy1 = _mm_packus_epi16(vacc2, vacc3);

    _mm_storeu_si128((__m128i*) output, vy0);
    _mm_storeu_si128((__m128i*) (output + 16), vy1);
    output += 32;
  }
  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    __m128i vacc = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input));
    __m128i vmultiplier = _mm_cmpgt_epi16(vacc, vinput_zero_point);
    vacc = _mm_sub_epi16(vinput_zero_point, vacc);
    vmultiplier = _mm_and_si128(vmultiplier, vmultiplier_diff);
    vacc = _mm_slli_epi16(vacc, 7);
    vmultiplier = _mm_xor_si128(vmultiplier, vmultiplier_base);
    vacc = _mm_mulhrs_epi16(vacc, vmultiplier);
    vacc = _mm_adds_epi16(vacc, voutput_zero_point);
    input += 8;

    const __m128i vy = _mm_packus_epi16(vacc, vacc);
    _mm_storel_epi64((__m128i*) output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 7 * sizeof(uint8_t));

    __m128i vacc = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input));
    __m128i vmultiplier = _mm_cmpgt_epi16(vacc, vinput_zero_point);
    vacc = _mm_sub_epi16(vinput_zero_point, vacc);
    vmultiplier = _mm_and_si128(vmultiplier, vmultiplier_diff);
    vacc = _mm_slli_epi16(vacc, 7);
    vmultiplier = _mm_xor_si128(vmultiplier, vmultiplier_base);
    vacc = _mm_mulhrs_epi16(vacc, vmultiplier);
    vacc = _mm_adds_epi16(vacc, voutput_zero_point);

    __m128i vy = _mm_packus_epi16(vacc, vacc);
    if (batch & (4 * sizeof(uint8_t))) {
      _mm_storeu_si32(output, vy);
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(uint8_t))) {
      _mm_storeu_si16(output, vy);
      vy = _mm_srli_epi32(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      *output = (uint8_t) _mm_extract_epi8(vy, 0);
    }
  }
}

void xnn_qu8_vmul_minmax_fp32_ukernel__sse41_mul16_ld64_u16(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128i va_zero_point = _mm_set1_epi16(params->scalar.a_zero_point);
  const __m128i vb_zero_point = _mm_set1_epi16(params->scalar.b_zero_point);
  const __m128 vscale = _mm_set1_ps(params->scalar.scale);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8(params->scalar.output_min);
  const __m128i voutput_max = _mm_set1_epi8(params->scalar.output_max);

  XNN_FORCE_REALIZATION(va_zero_point);
  XNN_FORCE_REALIZATION(vb_zero_point);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    const __m128i va01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
    const __m128i vb01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input_b));
    const __m128i va89ABCDEF = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) (input_a + 8)));
    const __m128i vb89ABCDEF = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) (input_b + 8)));
    input_a += 16;
    input_b += 16;


    const __m128i vxa01234567 = _mm_sub_epi16(va01234567, va_zero_point);
    const __m128i vxb01234567 = _mm_sub_epi16(vb01234567, vb_zero_point);
    const __m128i vxa89ABCDEF = _mm_sub_epi16(va89ABCDEF, va_zero_point);
    const __m128i vxb89ABCDEF = _mm_sub_epi16(vb89ABCDEF, vb_zero_point);

    const __m128i vprod01234567lo = _mm_mullo_epi16(vxa01234567, vxb01234567);
    const __m128i vprod01234567hi = _mm_mulhi_epi16(vxa01234567, vxb01234567);
    const __m128i vprod89ABCDEFlo = _mm_mullo_epi16(vxa89ABCDEF, vxb89ABCDEF);
    const __m128i vprod89ABCDEFhi = _mm_mulhi_epi16(vxa89ABCDEF, vxb89ABCDEF);

    const __m128i vprod0123 = _mm_unpacklo_epi16(vprod01234567lo, vprod01234567hi);
    const __m128i vprod4567 = _mm_unpackhi_epi16(vprod01234567lo, vprod01234567hi);
    const __m128i vprod89AB = _mm_unpacklo_epi16(vprod89ABCDEFlo, vprod89ABCDEFhi);
    const __m128i vprodCDEF = _mm_unpackhi_epi16(vprod89ABCDEFlo, vprod89ABCDEFhi);

    __m128 vfpacc0123 = _mm_cvtepi32_ps(vprod0123);
    __m128 vfpacc4567 = _mm_cvtepi32_ps(vprod4567);
    __m128 vfpacc89AB = _mm_cvtepi32_ps(vprod89AB);
    __m128 vfpaccCDEF = _mm_cvtepi32_ps(vprodCDEF);

    vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
    vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);
    vfpacc89AB = _mm_mul_ps(vfpacc89AB, vscale);
    vfpaccCDEF = _mm_mul_ps(vfpaccCDEF, vscale);

    const __m128i vacc0123 = _mm_cvtps_epi32(vfpacc0123);
    const __m128i vacc4567 = _mm_cvtps_epi32(vfpacc4567);
    const __m128i vacc89AB = _mm_cvtps_epi32(vfpacc89AB);
    const __m128i vaccCDEF = _mm_cvtps_epi32(vfpaccCDEF);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
    __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);


    __m128i vout0123456789ABCDEF = _mm_packus_epi16(vout01234567, vout89ABCDEF);

    vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epu8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const __m128i va01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
      const __m128i vb01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input_b));
      input_a += 8;
      input_b += 8;


      const __m128i vxa01234567 = _mm_sub_epi16(va01234567, va_zero_point);
      const __m128i vxb01234567 = _mm_sub_epi16(vb01234567, vb_zero_point);

      const __m128i vprod01234567lo = _mm_mullo_epi16(vxa01234567, vxb01234567);
      const __m128i vprod01234567hi = _mm_mulhi_epi16(vxa01234567, vxb01234567);

      const __m128i vprod0123 = _mm_unpacklo_epi16(vprod01234567lo, vprod01234567hi);
      const __m128i vprod4567 = _mm_unpackhi_epi16(vprod01234567lo, vprod01234567hi);

      __m128 vfpacc0123 = _mm_cvtepi32_ps(vprod0123);
      __m128 vfpacc4567 = _mm_cvtepi32_ps(vprod4567);

      vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
      vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

      const __m128i vacc0123 = _mm_cvtps_epi32(vfpacc0123);
      const __m128i vacc4567 = _mm_cvtps_epi32(vfpacc4567);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

      if XNN_LIKELY(batch >= (8 * sizeof(uint8_t))) {
        _mm_storel_epi64((__m128i*) output, vout0123456701234567);
        output += 8;
        batch -= 8 * sizeof(uint8_t);
      } else {
        if (batch & (4 * sizeof(uint8_t))) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (batch & (2 * sizeof(uint8_t))) {
          unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (batch & (1 * sizeof(uint8_t))) {
          *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
        }
        batch = 0;
      }
    } while (batch != 0);
  }
}

void xnn_qu8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u16(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128i va_zero_point = _mm_set1_epi16(params->scalar.a_zero_point);
  const __m128 vscale = _mm_set1_ps(params->scalar.scale);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8(params->scalar.output_min);
  const __m128i voutput_max = _mm_set1_epi8(params->scalar.output_max);

  __m128i vxb = _mm_set1_epi16((UINT32_C(0x00010001) * (uint32_t) (uint16_t) (int16_t) *input_b) - params->scalar.b_zero_point);

  XNN_FORCE_REALIZATION(va_zero_point);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    const __m128i va01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
    const __m128i va89ABCDEF = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) (input_a + 8)));
    input_a += 16;


    const __m128i vxa01234567 = _mm_sub_epi16(va01234567, va_zero_point);
    const __m128i vxa89ABCDEF = _mm_sub_epi16(va89ABCDEF, va_zero_point);

    const __m128i vprod01234567lo = _mm_mullo_epi16(vxa01234567, vxb);
    const __m128i vprod01234567hi = _mm_mulhi_epi16(vxa01234567, vxb);
    const __m128i vprod89ABCDEFlo = _mm_mullo_epi16(vxa89ABCDEF, vxb);
    const __m128i vprod89ABCDEFhi = _mm_mulhi_epi16(vxa89ABCDEF, vxb);

    const __m128i vprod0123 = _mm_unpacklo_epi16(vprod01234567lo, vprod01234567hi);
    const __m128i vprod4567 = _mm_unpackhi_epi16(vprod01234567lo, vprod01234567hi);
    const __m128i vprod89AB = _mm_unpacklo_epi16(vprod89ABCDEFlo, vprod89ABCDEFhi);
    const __m128i vprodCDEF = _mm_unpackhi_epi16(vprod89ABCDEFlo, vprod89ABCDEFhi);

    __m128 vfpacc0123 = _mm_cvtepi32_ps(vprod0123);
    __m128 vfpacc4567 = _mm_cvtepi32_ps(vprod4567);
    __m128 vfpacc89AB = _mm_cvtepi32_ps(vprod89AB);
    __m128 vfpaccCDEF = _mm_cvtepi32_ps(vprodCDEF);

    vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
    vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);
    vfpacc89AB = _mm_mul_ps(vfpacc89AB, vscale);
    vfpaccCDEF = _mm_mul_ps(vfpaccCDEF, vscale);

    const __m128i vacc0123 = _mm_cvtps_epi32(vfpacc0123);
    const __m128i vacc4567 = _mm_cvtps_epi32(vfpacc4567);
    const __m128i vacc89AB = _mm_cvtps_epi32(vfpacc89AB);
    const __m128i vaccCDEF = _mm_cvtps_epi32(vfpaccCDEF);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
    __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);


    __m128i vout0123456789ABCDEF = _mm_packus_epi16(vout01234567, vout89ABCDEF);

    vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epu8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const __m128i va01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) input_a));
      input_a += 8;


      const __m128i vxa01234567 = _mm_sub_epi16(va01234567, va_zero_point);

      const __m128i vprod01234567lo = _mm_mullo_epi16(vxa01234567, vxb);
      const __m128i vprod01234567hi = _mm_mulhi_epi16(vxa01234567, vxb);

      const __m128i vprod0123 = _mm_unpacklo_epi16(vprod01234567lo, vprod01234567hi);
      const __m128i vprod4567 = _mm_unpackhi_epi16(vprod01234567lo, vprod01234567hi);

      __m128 vfpacc0123 = _mm_cvtepi32_ps(vprod0123);
      __m128 vfpacc4567 = _mm_cvtepi32_ps(vprod4567);

      vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
      vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

      const __m128i vacc0123 = _mm_cvtps_epi32(vfpacc0123);
      const __m128i vacc4567 = _mm_cvtps_epi32(vfpacc4567);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epu8(vout0123456701234567, voutput_max);

      if XNN_LIKELY(batch >= (8 * sizeof(uint8_t))) {
        _mm_storel_epi64((__m128i*) output, vout0123456701234567);
        output += 8;
        batch -= 8 * sizeof(uint8_t);
      } else {
        if (batch & (4 * sizeof(uint8_t))) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
          vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
          output += 4;
        }
        if (batch & (2 * sizeof(uint8_t))) {
          unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
          vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
          output += 2;
        }
        if (batch & (1 * sizeof(uint8_t))) {
          *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
        }
        batch = 0;
      }
    } while (batch != 0);
  }
}

void xnn_s8_ibilinear_ukernel__sse41_c16(
    size_t output_pixels,
    size_t channels,
    const int8_t** restrict input,
    size_t input_offset,
    const int16_t* restrict weights,
    int8_t* restrict output,
    size_t output_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(channels != 0);

  do {
    const int8_t* i0 = (const int8_t*) ((uintptr_t) input[0] + input_offset);
    const int8_t* i1 = (const int8_t*) ((uintptr_t) input[1] + input_offset);
    const int8_t* i2 = (const int8_t*) ((uintptr_t) input[2] + input_offset);
    const int8_t* i3 = (const int8_t*) ((uintptr_t) input[3] + input_offset);
    input += 4;

    const __m128i valpha = _mm_cvtsi32_si128(*((const int*) weights));
    weights += 2;
    __m128i valphah = _mm_shufflelo_epi16(valpha, _MM_SHUFFLE(0, 0, 0, 0));
    valphah = _mm_unpacklo_epi64(valphah, valphah);
    __m128i valphav = _mm_srli_epi32(valpha, 16);
    valphav = _mm_shuffle_epi32(valphav, _MM_SHUFFLE(0, 0, 0, 0));

    valphah = _mm_blend_epi16(valphah, _mm_sub_epi16(_mm_set1_epi32(0x08000000), valphah), 0xAA);

    const __m128i vrounding = _mm_set1_epi32(0x00200000);

    size_t c = channels;
    for (; c >= 16 * sizeof(int8_t); c -= 16 * sizeof(int8_t)) {
      const __m128i vtl01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i0));
      const __m128i vtr01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i1));
      const __m128i vbl01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i2));
      const __m128i vbr01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i3));
      const __m128i vtl89ABCDEF = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (i0 + 8)));
      const __m128i vtr89ABCDEF = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (i1 + 8)));
      const __m128i vbl89ABCDEF = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (i2 + 8)));
      const __m128i vbr89ABCDEF = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (i3 + 8)));
      i0 += 16;
      i1 += 16;
      i2 += 16;
      i3 += 16;


      const __m128i vdr01234567 = _mm_sub_epi16(vbr01234567, vtr01234567);
      const __m128i vt0123 = _mm_madd_epi16(_mm_unpacklo_epi16(vtr01234567, vtl01234567), valphah);
      const __m128i vdl01234567 = _mm_sub_epi16(vbl01234567, vtl01234567);
      const __m128i vt4567 = _mm_madd_epi16(_mm_unpackhi_epi16(vtr01234567, vtl01234567), valphah);
      const __m128i vdr89ABCDEF = _mm_sub_epi16(vbr89ABCDEF, vtr89ABCDEF);
      const __m128i vt89AB = _mm_madd_epi16(_mm_unpacklo_epi16(vtr89ABCDEF, vtl89ABCDEF), valphah);
      const __m128i vdl89ABCDEF = _mm_sub_epi16(vbl89ABCDEF, vtl89ABCDEF);
      const __m128i vtCDEF = _mm_madd_epi16(_mm_unpackhi_epi16(vtr89ABCDEF, vtl89ABCDEF), valphah);

      const __m128i vd0123 = _mm_madd_epi16(_mm_unpacklo_epi16(vdr01234567, vdl01234567), valphah);
      const __m128i vd4567 = _mm_madd_epi16(_mm_unpackhi_epi16(vdr01234567, vdl01234567), valphah);
      const __m128i vd89AB = _mm_madd_epi16(_mm_unpacklo_epi16(vdr89ABCDEF, vdl89ABCDEF), valphah);
      const __m128i vdCDEF = _mm_madd_epi16(_mm_unpackhi_epi16(vdr89ABCDEF, vdl89ABCDEF), valphah);

      __m128i vacc0123 = _mm_mullo_epi32(vd0123, valphav);
      __m128i vacc4567 = _mm_mullo_epi32(vd4567, valphav);
      __m128i vacc89AB = _mm_mullo_epi32(vd89AB, valphav);
      __m128i vaccCDEF = _mm_mullo_epi32(vdCDEF, valphav);

      vacc0123 = _mm_add_epi32(_mm_slli_epi32(vt0123, 11), vacc0123);
      vacc4567 = _mm_add_epi32(_mm_slli_epi32(vt4567, 11), vacc4567);
      vacc89AB = _mm_add_epi32(_mm_slli_epi32(vt89AB, 11), vacc89AB);
      vaccCDEF = _mm_add_epi32(_mm_slli_epi32(vtCDEF, 11), vaccCDEF);

      vacc0123 = _mm_srai_epi32(_mm_add_epi16(vacc0123, vrounding), 22);
      vacc4567 = _mm_srai_epi32(_mm_add_epi16(vacc4567, vrounding), 22);
      vacc89AB = _mm_srai_epi32(_mm_add_epi16(vacc89AB, vrounding), 22);
      vaccCDEF = _mm_srai_epi32(_mm_add_epi16(vaccCDEF, vrounding), 22);

      const __m128i vacc01234567 = _mm_packs_epi32(vacc0123, vacc4567);
      const __m128i vacc89ABCDEF = _mm_packs_epi32(vacc89AB, vaccCDEF);

      const __m128i vo0123456789ABCDEF = _mm_packs_epi16(vacc01234567, vacc89ABCDEF);

      _mm_storeu_si128((__m128i*) output, vo0123456789ABCDEF);
      output += 16;
    }
    for (; c >= 8 * sizeof(int8_t); c -= 8 * sizeof(int8_t)) {
      const __m128i vtl01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i0));
      i0 += 8;
      const __m128i vtr01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i1));
      i1 += 8;
      const __m128i vbl01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i2));
      i2 += 8;
      const __m128i vbr01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i3));
      i3 += 8;


      const __m128i vdr01234567 = _mm_sub_epi16(vbr01234567, vtr01234567);
      const __m128i vt0123 = _mm_madd_epi16(_mm_unpacklo_epi16(vtr01234567, vtl01234567), valphah);
      const __m128i vdl01234567 = _mm_sub_epi16(vbl01234567, vtl01234567);
      const __m128i vt4567 = _mm_madd_epi16(_mm_unpackhi_epi16(vtr01234567, vtl01234567), valphah);

      const __m128i vd0123 = _mm_madd_epi16(_mm_unpacklo_epi16(vdr01234567, vdl01234567), valphah);
      const __m128i vd4567 = _mm_madd_epi16(_mm_unpackhi_epi16(vdr01234567, vdl01234567), valphah);

      __m128i vacc0123 = _mm_mullo_epi32(vd0123, valphav);
      __m128i vacc4567 = _mm_mullo_epi32(vd4567, valphav);

      vacc0123 = _mm_add_epi32(_mm_slli_epi32(vt0123, 11), vacc0123);
      vacc4567 = _mm_add_epi32(_mm_slli_epi32(vt4567, 11), vacc4567);

      vacc0123 = _mm_srai_epi32(_mm_add_epi16(vacc0123, vrounding), 22);
      vacc4567 = _mm_srai_epi32(_mm_add_epi16(vacc4567, vrounding), 22);

      const __m128i vacc01234567 = _mm_packs_epi32(vacc0123, vacc4567);

      const __m128i vo01234567 = _mm_packs_epi16(vacc01234567, vacc01234567);

      _mm_storel_epi64((__m128i*) output, vo01234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      const __m128i vtl01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i0));
      const __m128i vtr01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i1));
      const __m128i vbl01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i2));
      const __m128i vbr01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i3));


      const __m128i vdr01234567 = _mm_sub_epi16(vbr01234567, vtr01234567);
      const __m128i vt0123 = _mm_madd_epi16(_mm_unpacklo_epi16(vtr01234567, vtl01234567), valphah);
      const __m128i vdl01234567 = _mm_sub_epi16(vbl01234567, vtl01234567);
      const __m128i vt4567 = _mm_madd_epi16(_mm_unpackhi_epi16(vtr01234567, vtl01234567), valphah);

      const __m128i vd0123 = _mm_madd_epi16(_mm_unpacklo_epi16(vdr01234567, vdl01234567), valphah);
      const __m128i vd4567 = _mm_madd_epi16(_mm_unpackhi_epi16(vdr01234567, vdl01234567), valphah);

      __m128i vacc0123 = _mm_mullo_epi32(vd0123, valphav);
      __m128i vacc4567 = _mm_mullo_epi32(vd4567, valphav);

      vacc0123 = _mm_add_epi32(_mm_slli_epi32(vt0123, 11), vacc0123);
      vacc4567 = _mm_add_epi32(_mm_slli_epi32(vt4567, 11), vacc4567);

      vacc0123 = _mm_srai_epi32(_mm_add_epi16(vacc0123, vrounding), 22);
      vacc4567 = _mm_srai_epi32(_mm_add_epi16(vacc4567, vrounding), 22);

      const __m128i vacc01234567 = _mm_packs_epi32(vacc0123, vacc4567);

      __m128i vo01234567 = _mm_packs_epi16(vacc01234567, vacc01234567);

      if (c & (4 * sizeof(int8_t))) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vo01234567));
        output += 4;
        vo01234567 = _mm_srli_epi64(vo01234567, 32);
      }
      if (c & (2 * sizeof(int8_t))) {
        unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vo01234567, 0));
        output += 2;
        vo01234567 = _mm_srli_epi32(vo01234567, 16);
      }
      if (c & (1 * sizeof(int8_t))) {
        *output++ = (uint8_t) _mm_extract_epi8(vo01234567, 0);
      }
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_s8_maxpool_minmax_ukernel_9p8x__sse41_c16(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const int8_t** input,
    size_t input_offset,
    int8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_s8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const __m128i voutput_max = _mm_set1_epi8(params->scalar.max);
  const __m128i voutput_min = _mm_set1_epi8(params->scalar.min);
  XNN_FORCE_REALIZATION(voutput_max);
  XNN_FORCE_REALIZATION(voutput_min);

  do {
    int8_t* o = output;
    {
      const int8_t* i0 = *input++;
      const int8_t* i1 = *input++;
      const int8_t* i2 = *input++;
      const int8_t* i3 = *input++;
      const int8_t* i4 = *input++;
      const int8_t* i5 = *input++;
      const int8_t* i6 = *input++;
      const int8_t* i7 = *input++;
      const int8_t* i8 = *input++;
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
      if (kernel_elements < 2) {
        i1 = i0;
      }
      if (kernel_elements <= 2) {
        i2 = i0;
      }
      if (kernel_elements < 4) {
        i3 = i0;
      }
      if (kernel_elements <= 4) {
        i4 = i0;
      }
      if (kernel_elements < 6) {
        i5 = i0;
      }
      if (kernel_elements <= 6) {
        i6 = i0;
      }
      if (kernel_elements < 8) {
        i7 = i0;
      }
      if (kernel_elements <= 8) {
        i8 = i0;
      }

      size_t c = channels;
      for (; c >= 16; c -= 16) {
        const __m128i vi0 = _mm_loadu_si128((const __m128i*) i0); i0 += 16;
        const __m128i vi1 = _mm_loadu_si128((const __m128i*) i1); i1 += 16;
        const __m128i vi2 = _mm_loadu_si128((const __m128i*) i2); i2 += 16;
        const __m128i vi3 = _mm_loadu_si128((const __m128i*) i3); i3 += 16;
        const __m128i vi4 = _mm_loadu_si128((const __m128i*) i4); i4 += 16;
        const __m128i vi5 = _mm_loadu_si128((const __m128i*) i5); i5 += 16;
        const __m128i vi6 = _mm_loadu_si128((const __m128i*) i6); i6 += 16;
        const __m128i vi7 = _mm_loadu_si128((const __m128i*) i7); i7 += 16;
        const __m128i vi8 = _mm_loadu_si128((const __m128i*) i8); i8 += 16;

        const __m128i vmax018 = _mm_max_epi8(_mm_max_epi8(vi0, vi1), vi8);
        const __m128i vmax23 = _mm_max_epi8(vi2, vi3);
        const __m128i vmax45 = _mm_max_epi8(vi4, vi5);
        const __m128i vmax67 = _mm_max_epi8(vi6, vi7);

        const __m128i vmax2345 = _mm_max_epi8(vmax23, vmax45);
        const __m128i vmax01678 = _mm_max_epi8(vmax018, vmax67);
        __m128i vout = _mm_max_epi8(vmax2345, vmax01678);
        vout = _mm_max_epi8(vout, voutput_min);
        vout = _mm_min_epi8(vout, voutput_max);

        _mm_storeu_si128((__m128i*) o, vout); o += 16;
      }
      if (c != 0) {
        const __m128i vi0 = _mm_loadu_si128((const __m128i*) i0);
        const __m128i vi1 = _mm_loadu_si128((const __m128i*) i1);
        const __m128i vi2 = _mm_loadu_si128((const __m128i*) i2);
        const __m128i vi3 = _mm_loadu_si128((const __m128i*) i3);
        const __m128i vi4 = _mm_loadu_si128((const __m128i*) i4);
        const __m128i vi5 = _mm_loadu_si128((const __m128i*) i5);
        const __m128i vi6 = _mm_loadu_si128((const __m128i*) i6);
        const __m128i vi7 = _mm_loadu_si128((const __m128i*) i7);
        const __m128i vi8 = _mm_loadu_si128((const __m128i*) i8);

        const __m128i vmax018 = _mm_max_epi8(_mm_max_epi8(vi0, vi1), vi8);
        const __m128i vmax23 = _mm_max_epi8(vi2, vi3);
        const __m128i vmax45 = _mm_max_epi8(vi4, vi5);
        const __m128i vmax67 = _mm_max_epi8(vi6, vi7);

        const __m128i vmax2345 = _mm_max_epi8(vmax23, vmax45);
        const __m128i vmax01678 = _mm_max_epi8(vmax018, vmax67);
        __m128i vout = _mm_max_epi8(vmax2345, vmax01678);
        vout = _mm_max_epi8(vout, voutput_min);
        vout = _mm_min_epi8(vout, voutput_max);

        if (c & 8) {
          _mm_storel_epi64((__m128i*) o, vout);
          vout = _mm_unpackhi_epi64(vout, vout);
          o += 8;
        }
        if (c & 4) {
          unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(vout));
          vout = _mm_srli_epi64(vout, 32);
          o += 4;
        }
        if (c & 2) {
          unaligned_store_u16(o, (uint16_t) _mm_extract_epi16(vout, 0));
          vout = _mm_srli_epi32(vout, 16);
          o += 2;
        }
        if (c & 1) {
          *o = (int8_t) _mm_cvtsi128_si32(vout);
          o += 1;
        }
      }
    }

    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
      const int8_t* i0 = *input++;
      const int8_t* i1 = *input++;
      const int8_t* i2 = *input++;
      const int8_t* i3 = *input++;
      const int8_t* i4 = *input++;
      const int8_t* i5 = *input++;
      const int8_t* i6 = *input++;
      const int8_t* i7 = *input++;
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
      if (k < 2) {
        i1 = i0;
      }
      if (k <= 2) {
        i2 = i0;
      }
      if (k < 4) {
        i3 = i0;
      }
      if (k <= 4) {
        i4 = i0;
      }
      if (k < 6) {
        i5 = i0;
      }
      if (k <= 6) {
        i6 = i0;
      }
      if (k < 8) {
        i7 = i0;
      }

      o = output;
      size_t c = channels;
      for (; c >= 16; c -= 16) {
        const __m128i vi0 = _mm_loadu_si128((const __m128i*) i0); i0 += 16;
        const __m128i vi1 = _mm_loadu_si128((const __m128i*) i1); i1 += 16;
        const __m128i vi2 = _mm_loadu_si128((const __m128i*) i2); i2 += 16;
        const __m128i vi3 = _mm_loadu_si128((const __m128i*) i3); i3 += 16;
        const __m128i vi4 = _mm_loadu_si128((const __m128i*) i4); i4 += 16;
        const __m128i vi5 = _mm_loadu_si128((const __m128i*) i5); i5 += 16;
        const __m128i vi6 = _mm_loadu_si128((const __m128i*) i6); i6 += 16;
        const __m128i vi7 = _mm_loadu_si128((const __m128i*) i7); i7 += 16;
        const __m128i vo = _mm_loadu_si128((const __m128i*) o);

        const __m128i vmax01 = _mm_max_epi8(_mm_max_epi8(vi0, vi1), vo);
        const __m128i vmax23 = _mm_max_epi8(vi2, vi3);
        const __m128i vmax45 = _mm_max_epi8(vi4, vi5);
        const __m128i vmax67 = _mm_max_epi8(vi6, vi7);

        const __m128i vmax2345 = _mm_max_epi8(vmax23, vmax45);
        const __m128i vmax0167 = _mm_max_epi8(vmax01, vmax67);
        __m128i vout = _mm_max_epi8(vmax2345, vmax0167);
        vout = _mm_max_epi8(vout, voutput_min);
        vout = _mm_min_epi8(vout, voutput_max);

        _mm_storeu_si128((__m128i*) o, vout);
        o += 16;
      }
      if (c != 0) {
        const __m128i vi0 = _mm_loadu_si128((const __m128i*) i0);
        const __m128i vi1 = _mm_loadu_si128((const __m128i*) i1);
        const __m128i vi2 = _mm_loadu_si128((const __m128i*) i2);
        const __m128i vi3 = _mm_loadu_si128((const __m128i*) i3);
        const __m128i vi4 = _mm_loadu_si128((const __m128i*) i4);
        const __m128i vi5 = _mm_loadu_si128((const __m128i*) i5);
        const __m128i vi6 = _mm_loadu_si128((const __m128i*) i6);
        const __m128i vi7 = _mm_loadu_si128((const __m128i*) i7);
        const __m128i vo = _mm_loadu_si128((const __m128i*) o);

        const __m128i vmax01 = _mm_max_epi8(_mm_max_epi8(vi0, vi1), vo);
        const __m128i vmax23 = _mm_max_epi8(vi2, vi3);
        const __m128i vmax45 = _mm_max_epi8(vi4, vi5);
        const __m128i vmax67 = _mm_max_epi8(vi6, vi7);

        const __m128i vmax2345 = _mm_max_epi8(vmax23, vmax45);
        const __m128i vmax0167 = _mm_max_epi8(vmax01, vmax67);
        __m128i vout = _mm_max_epi8(vmax2345, vmax0167);
        vout = _mm_max_epi8(vout, voutput_min);
        vout = _mm_min_epi8(vout, voutput_max);

        if (c & 8) {
          _mm_storel_epi64((__m128i*) o, vout);
          vout = _mm_unpackhi_epi64(vout, vout);
          o += 8;
        }
        if (c & 4) {
          unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(vout));
          vout = _mm_srli_epi64(vout, 32);
          o += 4;
        }
        if (c & 2) {
          unaligned_store_u16(o, (uint16_t) _mm_extract_epi16(vout, 0));
          vout = _mm_srli_epi32(vout, 16);
          o += 2;
        }
        if (c & 1) {
          *o = (int8_t) _mm_cvtsi128_si32(vout);
          o += 1;
        }
      }
    }
    input = (const int8_t**) ((uintptr_t) input + input_increment);
    output = (int8_t*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}

void xnn_s8_vclamp_ukernel__sse41_u64(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_s8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i voutput_max = _mm_set1_epi8(params->scalar.max);
  const __m128i voutput_min = _mm_set1_epi8(params->scalar.min);
  XNN_FORCE_REALIZATION(voutput_max);
  XNN_FORCE_REALIZATION(voutput_min);

  for (; batch >= 64; batch -= 64) {
    __m128i vacc0 = _mm_loadu_si128((const __m128i*) input);
    __m128i vacc1 = _mm_loadu_si128((const __m128i*) input + 1);
    __m128i vacc2 = _mm_loadu_si128((const __m128i*) input + 2);
    __m128i vacc3 = _mm_loadu_si128((const __m128i*) input + 3);
    input += 64;

    vacc0 = _mm_max_epi8(vacc0, voutput_min);
    vacc1 = _mm_max_epi8(vacc1, voutput_min);
    vacc2 = _mm_max_epi8(vacc2, voutput_min);
    vacc3 = _mm_max_epi8(vacc3, voutput_min);

    vacc0 = _mm_min_epi8(vacc0, voutput_max);
    vacc1 = _mm_min_epi8(vacc1, voutput_max);
    vacc2 = _mm_min_epi8(vacc2, voutput_max);
    vacc3 = _mm_min_epi8(vacc3, voutput_max);

    _mm_storeu_si128((__m128i*) output, vacc0);
    _mm_storeu_si128((__m128i*) output + 1, vacc1);
    _mm_storeu_si128((__m128i*) output + 2, vacc2);
    _mm_storeu_si128((__m128i*) output + 3, vacc3);
    output += 64;
  }
  for (; batch >= 16; batch -= 16) {
    __m128i vacc = _mm_loadu_si128((const __m128i*) input);
    input += 16;

    vacc = _mm_min_epi8(vacc, voutput_max);
    vacc = _mm_max_epi8(vacc, voutput_min);

    _mm_storeu_si128((__m128i*) output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m128i vacc = _mm_loadu_si128((const __m128i*) input);

    vacc = _mm_min_epi8(vacc, voutput_max);
    vacc = _mm_max_epi8(vacc, voutput_min);

    if (batch & 8) {
      _mm_storel_epi64((__m128i*) output, vacc);
      output += 8;
      vacc = _mm_unpackhi_epi64(vacc, vacc);
    }
    if (batch & 4) {
      unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vacc));
      output += 4;
      vacc = _mm_srli_epi64(vacc, 32);
    }
    if (batch & 2) {
      unaligned_store_u16(output, (uint16_t) _mm_cvtsi128_si32(vacc));
      output += 2;
      vacc = _mm_srli_epi32(vacc, 16);
    }
    if (batch & 1) {
      *output = (int8_t) _mm_cvtsi128_si32(vacc);
    }
  }
}

void xnn_u8_ibilinear_ukernel__sse41_c16(
    size_t output_pixels,
    size_t channels,
    const uint8_t** restrict input,
    size_t input_offset,
    const int16_t* restrict weights,
    uint8_t* restrict output,
    size_t output_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(channels != 0);

  do {
    const uint8_t* i0 = (const uint8_t*) ((uintptr_t) input[0] + input_offset);
    const uint8_t* i1 = (const uint8_t*) ((uintptr_t) input[1] + input_offset);
    const uint8_t* i2 = (const uint8_t*) ((uintptr_t) input[2] + input_offset);
    const uint8_t* i3 = (const uint8_t*) ((uintptr_t) input[3] + input_offset);
    input += 4;

    const __m128i valpha = _mm_cvtsi32_si128(*((const int*) weights));
    weights += 2;
    __m128i valphah = _mm_shufflelo_epi16(valpha, _MM_SHUFFLE(0, 0, 0, 0));
    valphah = _mm_unpacklo_epi64(valphah, valphah);
    __m128i valphav = _mm_srli_epi32(valpha, 16);
    valphav = _mm_shuffle_epi32(valphav, _MM_SHUFFLE(0, 0, 0, 0));

    valphah = _mm_blend_epi16(valphah, _mm_sub_epi16(_mm_set1_epi32(0x08000000), valphah), 0xAA);

    const __m128i vrounding = _mm_set1_epi32(0x00200000);

    size_t c = channels;
    for (; c >= 16 * sizeof(uint8_t); c -= 16 * sizeof(uint8_t)) {
      const __m128i vtl01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i0));
      const __m128i vtr01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i1));
      const __m128i vbl01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i2));
      const __m128i vbr01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i3));
      const __m128i vtl89ABCDEF = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) (i0 + 8)));
      const __m128i vtr89ABCDEF = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) (i1 + 8)));
      const __m128i vbl89ABCDEF = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) (i2 + 8)));
      const __m128i vbr89ABCDEF = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) (i3 + 8)));
      i0 += 16;
      i1 += 16;
      i2 += 16;
      i3 += 16;


      const __m128i vdr01234567 = _mm_sub_epi16(vbr01234567, vtr01234567);
      const __m128i vt0123 = _mm_madd_epi16(_mm_unpacklo_epi16(vtr01234567, vtl01234567), valphah);
      const __m128i vdl01234567 = _mm_sub_epi16(vbl01234567, vtl01234567);
      const __m128i vt4567 = _mm_madd_epi16(_mm_unpackhi_epi16(vtr01234567, vtl01234567), valphah);
      const __m128i vdr89ABCDEF = _mm_sub_epi16(vbr89ABCDEF, vtr89ABCDEF);
      const __m128i vt89AB = _mm_madd_epi16(_mm_unpacklo_epi16(vtr89ABCDEF, vtl89ABCDEF), valphah);
      const __m128i vdl89ABCDEF = _mm_sub_epi16(vbl89ABCDEF, vtl89ABCDEF);
      const __m128i vtCDEF = _mm_madd_epi16(_mm_unpackhi_epi16(vtr89ABCDEF, vtl89ABCDEF), valphah);

      const __m128i vd0123 = _mm_madd_epi16(_mm_unpacklo_epi16(vdr01234567, vdl01234567), valphah);
      const __m128i vd4567 = _mm_madd_epi16(_mm_unpackhi_epi16(vdr01234567, vdl01234567), valphah);
      const __m128i vd89AB = _mm_madd_epi16(_mm_unpacklo_epi16(vdr89ABCDEF, vdl89ABCDEF), valphah);
      const __m128i vdCDEF = _mm_madd_epi16(_mm_unpackhi_epi16(vdr89ABCDEF, vdl89ABCDEF), valphah);

      __m128i vacc0123 = _mm_mullo_epi32(vd0123, valphav);
      __m128i vacc4567 = _mm_mullo_epi32(vd4567, valphav);
      __m128i vacc89AB = _mm_mullo_epi32(vd89AB, valphav);
      __m128i vaccCDEF = _mm_mullo_epi32(vdCDEF, valphav);

      vacc0123 = _mm_add_epi32(_mm_slli_epi32(vt0123, 11), vacc0123);
      vacc4567 = _mm_add_epi32(_mm_slli_epi32(vt4567, 11), vacc4567);
      vacc89AB = _mm_add_epi32(_mm_slli_epi32(vt89AB, 11), vacc89AB);
      vaccCDEF = _mm_add_epi32(_mm_slli_epi32(vtCDEF, 11), vaccCDEF);

      vacc0123 = _mm_srli_epi32(_mm_add_epi16(vacc0123, vrounding), 22);
      vacc4567 = _mm_srli_epi32(_mm_add_epi16(vacc4567, vrounding), 22);
      vacc89AB = _mm_srli_epi32(_mm_add_epi16(vacc89AB, vrounding), 22);
      vaccCDEF = _mm_srli_epi32(_mm_add_epi16(vaccCDEF, vrounding), 22);

      const __m128i vacc01234567 = _mm_packs_epi32(vacc0123, vacc4567);
      const __m128i vacc89ABCDEF = _mm_packs_epi32(vacc89AB, vaccCDEF);

      const __m128i vo0123456789ABCDEF = _mm_packus_epi16(vacc01234567, vacc89ABCDEF);

      _mm_storeu_si128((__m128i*) output, vo0123456789ABCDEF);
      output += 16;
    }
    for (; c >= 8 * sizeof(uint8_t); c -= 8 * sizeof(uint8_t)) {
      const __m128i vtl01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i0));
      i0 += 8;
      const __m128i vtr01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i1));
      i1 += 8;
      const __m128i vbl01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i2));
      i2 += 8;
      const __m128i vbr01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i3));
      i3 += 8;


      const __m128i vdr01234567 = _mm_sub_epi16(vbr01234567, vtr01234567);
      const __m128i vt0123 = _mm_madd_epi16(_mm_unpacklo_epi16(vtr01234567, vtl01234567), valphah);
      const __m128i vdl01234567 = _mm_sub_epi16(vbl01234567, vtl01234567);
      const __m128i vt4567 = _mm_madd_epi16(_mm_unpackhi_epi16(vtr01234567, vtl01234567), valphah);

      const __m128i vd0123 = _mm_madd_epi16(_mm_unpacklo_epi16(vdr01234567, vdl01234567), valphah);
      const __m128i vd4567 = _mm_madd_epi16(_mm_unpackhi_epi16(vdr01234567, vdl01234567), valphah);

      __m128i vacc0123 = _mm_mullo_epi32(vd0123, valphav);
      __m128i vacc4567 = _mm_mullo_epi32(vd4567, valphav);

      vacc0123 = _mm_add_epi32(_mm_slli_epi32(vt0123, 11), vacc0123);
      vacc4567 = _mm_add_epi32(_mm_slli_epi32(vt4567, 11), vacc4567);

      vacc0123 = _mm_srli_epi32(_mm_add_epi16(vacc0123, vrounding), 22);
      vacc4567 = _mm_srli_epi32(_mm_add_epi16(vacc4567, vrounding), 22);

      const __m128i vacc01234567 = _mm_packs_epi32(vacc0123, vacc4567);

      const __m128i vo01234567 = _mm_packus_epi16(vacc01234567, vacc01234567);

      _mm_storel_epi64((__m128i*) output, vo01234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      const __m128i vtl01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i0));
      const __m128i vtr01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i1));
      const __m128i vbl01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i2));
      const __m128i vbr01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i3));


      const __m128i vdr01234567 = _mm_sub_epi16(vbr01234567, vtr01234567);
      const __m128i vt0123 = _mm_madd_epi16(_mm_unpacklo_epi16(vtr01234567, vtl01234567), valphah);
      const __m128i vdl01234567 = _mm_sub_epi16(vbl01234567, vtl01234567);
      const __m128i vt4567 = _mm_madd_epi16(_mm_unpackhi_epi16(vtr01234567, vtl01234567), valphah);

      const __m128i vd0123 = _mm_madd_epi16(_mm_unpacklo_epi16(vdr01234567, vdl01234567), valphah);
      const __m128i vd4567 = _mm_madd_epi16(_mm_unpackhi_epi16(vdr01234567, vdl01234567), valphah);

      __m128i vacc0123 = _mm_mullo_epi32(vd0123, valphav);
      __m128i vacc4567 = _mm_mullo_epi32(vd4567, valphav);

      vacc0123 = _mm_add_epi32(_mm_slli_epi32(vt0123, 11), vacc0123);
      vacc4567 = _mm_add_epi32(_mm_slli_epi32(vt4567, 11), vacc4567);

      vacc0123 = _mm_srli_epi32(_mm_add_epi16(vacc0123, vrounding), 22);
      vacc4567 = _mm_srli_epi32(_mm_add_epi16(vacc4567, vrounding), 22);

      const __m128i vacc01234567 = _mm_packs_epi32(vacc0123, vacc4567);

      __m128i vo01234567 = _mm_packus_epi16(vacc01234567, vacc01234567);

      if (c & (4 * sizeof(uint8_t))) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vo01234567));
        output += 4;
        vo01234567 = _mm_srli_epi64(vo01234567, 32);
      }
      if (c & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vo01234567, 0));
        output += 2;
        vo01234567 = _mm_srli_epi32(vo01234567, 16);
      }
      if (c & (1 * sizeof(uint8_t))) {
        *output++ = (uint8_t) _mm_extract_epi8(vo01234567, 0);
      }
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_s32_vmul_ukernel__sse41_u8(
    size_t batch,
    const int32_t* input_a,
    const int32_t* input_b,
    int32_t* output,
    const union xnn_s32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int32_t) == 0);
  assert(input_b != NULL);
  assert(input_a != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_s32 == 4);

  for (; batch >= 8 * sizeof(int32_t); batch -= 8 * sizeof(int32_t)) {
    xnn_simd_s32_t vin1_0 = xnn_loadu_s32(input_a);
    xnn_simd_s32_t vin1_1 = xnn_loadu_s32(input_a + 1 * xnn_simd_size_s32);
    input_a += 8;

    xnn_simd_s32_t vin2_0 = xnn_loadu_s32(input_b);
    xnn_simd_s32_t vin2_1 = (xnn_loadu_s32(input_b + 1 * xnn_simd_size_s32));
    input_b += 8;

    xnn_simd_s32_t vy_0 = xnn_mul_s32(vin1_0, vin2_0);
    xnn_simd_s32_t vy_1 = xnn_mul_s32(vin1_1, vin2_1);

    xnn_storeu_s32(output, vy_0);
    xnn_storeu_s32(output + 1 * xnn_simd_size_s32, vy_1);
    output += 8;
  }
  for (; batch >= xnn_simd_bytes_s32; batch -= xnn_simd_bytes_s32) {
    xnn_simd_s32_t vin1 = xnn_loadu_s32(input_a);
    input_a += xnn_simd_size_s32;

    xnn_simd_s32_t vin2 = xnn_loadu_s32(input_b);
    input_b += xnn_simd_size_s32;

    xnn_simd_s32_t vy = xnn_mul_s32(vin1, vin2);

    xnn_storeu_s32(output, vy);
    output += xnn_simd_size_s32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_s32_t vin1 = xnn_load_tail_s32(input_a, batch >> XNN_LOG2_SIZEOF_INT32_T);

    xnn_simd_s32_t vin2 = xnn_load_tail_s32(input_b, batch >> XNN_LOG2_SIZEOF_INT32_T);

    xnn_simd_s32_t vy = xnn_mul_s32(vin1, vin2);

    xnn_store_tail_s32(output, vy, batch >> XNN_LOG2_SIZEOF_INT32_T);
  }
}

void xnn_s32_vmulc_ukernel__sse41_u8(
    size_t batch,
    const int32_t* input1,
    const int32_t* input2,
    int32_t* output,
    const union xnn_s32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int32_t) == 0);
  assert(input1 != NULL);
  assert(input2 != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_s32 == 4);

  xnn_simd_s32_t vin2 = xnn_set1_s32(*input2);

  for (; batch >= 8 * sizeof(int32_t); batch -= 8 * sizeof(int32_t)) {

    xnn_simd_s32_t vin1_0 = (xnn_loadu_s32(input1));
    xnn_simd_s32_t vin1_1 = (xnn_loadu_s32(input1 + 1 * xnn_simd_size_s32));
    input1 += 8;

    xnn_simd_s32_t vy_0 = xnn_mul_s32(vin1_0, vin2);
    xnn_simd_s32_t vy_1 = xnn_mul_s32(vin1_1, vin2);

    xnn_storeu_s32(output, vy_0);
    xnn_storeu_s32(output + 1 * xnn_simd_size_s32, vy_1);
    output += 8;
  }
  for (; batch >= xnn_simd_bytes_s32; batch -= xnn_simd_bytes_s32) {
    xnn_simd_s32_t vin1 = xnn_loadu_s32(input1);
    input1 += xnn_simd_size_s32;

    xnn_simd_s32_t vy = xnn_mul_s32(vin1, vin2);

    xnn_storeu_s32(output, vy);
    output += xnn_simd_size_s32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_s32_t vin1 = (xnn_load_tail_s32(input1, batch >> XNN_LOG2_SIZEOF_INT32_T));

    xnn_simd_s32_t vy = xnn_mul_s32(vin1, vin2);

    xnn_store_tail_s32(output, vy, batch >> XNN_LOG2_SIZEOF_INT32_T);
  }
}
