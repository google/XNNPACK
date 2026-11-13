// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx4c8-ssevnni.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/prefetch.h"
#include "src/xnnpack/unaligned.h"


void xnn_qd8_f16_qc2w_gemm_minmax_ukernel_1x4c8__f16c_madd_prfm(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    xnn_float16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f16_minmax_params* restrict params,
    const float* row_sum,
    const struct xnn_qd8_quantization_params* restrict quantization_params) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const size_t original_kc = kc;
  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  uint16_t* c0 = (uint16_t*) c;

  const __m128i vmask = _mm_set1_epi8(0x03);
  const __m128i vminus_one = _mm_set1_epi16(-1);
  XNN_FORCE_REALIZATION(vmask);
  do {
    const int32_t* ksum = (const int32_t*) w;
    w = (const int32_t*) w + 8;
    __m128i vacc0x01 = _mm_setzero_si128();
    __m128i vacc0x23 = _mm_setzero_si128();

    size_t k = kc;
    while (k >= 32 * sizeof(int8_t)) {
      const __m128i vbb01234567 = _mm_load_si128((const __m128i*) w);
      const __m128i vbb89ABCDEF = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      w = (const int8_t*) w + 32;

      const __m128i vb01234567x0 = _mm_and_si128(vbb01234567, vmask);
      const __m128i vb89ABCDEFx0 = _mm_and_si128(vbb89ABCDEF, vmask);
      const __m128i vb01234567x1 = _mm_and_si128(_mm_srli_epi32(vbb01234567, 2), vmask);
      const __m128i vb89ABCDEFx1 = _mm_and_si128(_mm_srli_epi32(vbb89ABCDEF, 2), vmask);
      const __m128i vb01234567x2 = _mm_and_si128(_mm_srli_epi32(vbb01234567, 4), vmask);
      const __m128i vb89ABCDEFx2 = _mm_and_si128(_mm_srli_epi32(vbb89ABCDEF, 4), vmask);
      const __m128i vb01234567x3 = _mm_and_si128(_mm_srli_epi32(vbb01234567, 6), vmask);
      const __m128i vb89ABCDEFx3 = _mm_and_si128(_mm_srli_epi32(vbb89ABCDEF, 6), vmask);

      xnn_prefetch_to_l1((const int8_t*) w + 896);

      const __m128i va0x0 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a0));
      const __m128i va0x1 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a0 + 8));
      const __m128i va0x2 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a0 + 16));
      const __m128i va0x3 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a0 + 24));
      a0 += 32;

      __m128i vsum0x01 = _mm_add_epi16(
          _mm_maddubs_epi16(vb01234567x0, va0x0),
          _mm_maddubs_epi16(vb01234567x1, va0x1));
      __m128i vsum0x23 = _mm_add_epi16(
          _mm_maddubs_epi16(vb89ABCDEFx0, va0x0),
          _mm_maddubs_epi16(vb89ABCDEFx1, va0x1));
      vsum0x01 = _mm_add_epi16(
          vsum0x01,
          _mm_add_epi16(
              _mm_maddubs_epi16(vb01234567x2, va0x2),
              _mm_maddubs_epi16(vb01234567x3, va0x3)));
      vsum0x23 = _mm_add_epi16(
          vsum0x23,
          _mm_add_epi16(
              _mm_maddubs_epi16(vb89ABCDEFx2, va0x2),
              _mm_maddubs_epi16(vb89ABCDEFx3, va0x3)));
      vacc0x01 = _mm_sub_epi32(vacc0x01, _mm_madd_epi16(vsum0x01, vminus_one));
      vacc0x23 = _mm_sub_epi32(vacc0x23, _mm_madd_epi16(vsum0x23, vminus_one));

      k -= 32 * sizeof(int8_t);
    }
    if (k >= 16 * sizeof(int8_t)) {
      const __m128i vbb01234567 = _mm_load_si128((const __m128i*) w);
      const __m128i vbb89ABCDEF = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      w = (const int8_t*) w + 32;

      const __m128i vb01234567x0 = _mm_and_si128(vbb01234567, vmask);
      const __m128i vb89ABCDEFx0 = _mm_and_si128(vbb89ABCDEF, vmask);
      const __m128i vb01234567x1 = _mm_and_si128(_mm_srli_epi32(vbb01234567, 2), vmask);
      const __m128i vb89ABCDEFx1 = _mm_and_si128(_mm_srli_epi32(vbb89ABCDEF, 2), vmask);

      xnn_prefetch_to_l1((const int8_t*) w + 896);

      if (k > 16 * sizeof(int8_t)) {
        const __m128i vb01234567x2 = _mm_and_si128(_mm_srli_epi32(vbb01234567, 4), vmask);
        const __m128i vb89ABCDEFx2 = _mm_and_si128(_mm_srli_epi32(vbb89ABCDEF, 4), vmask);
        const __m128i va0x0 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a0));
        const __m128i va0x1 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a0 + 8));
        const __m128i va0x2 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a0 + 16));
        a0 += 24;

        __m128i vsum0x01 = _mm_add_epi16(
            _mm_maddubs_epi16(vb01234567x0, va0x0),
            _mm_maddubs_epi16(vb01234567x1, va0x1));
        __m128i vsum0x23 = _mm_add_epi16(
            _mm_maddubs_epi16(vb89ABCDEFx0, va0x0),
            _mm_maddubs_epi16(vb89ABCDEFx1, va0x1));
        vsum0x01 = _mm_add_epi16(vsum0x01, _mm_maddubs_epi16(vb01234567x2, va0x2));
        vsum0x23 = _mm_add_epi16(vsum0x23, _mm_maddubs_epi16(vb89ABCDEFx2, va0x2));
        vacc0x01 = _mm_sub_epi32(vacc0x01, _mm_madd_epi16(vsum0x01, vminus_one));
        vacc0x23 = _mm_sub_epi32(vacc0x23, _mm_madd_epi16(vsum0x23, vminus_one));
        k -= 24 * sizeof(int8_t);
      } else {
        const __m128i va0x0 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a0));
        const __m128i va0x1 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a0 + 8));
        a0 += 16;

        const __m128i vsum0x01 = _mm_add_epi16(
            _mm_maddubs_epi16(vb01234567x0, va0x0),
            _mm_maddubs_epi16(vb01234567x1, va0x1));
        const __m128i vsum0x23 = _mm_add_epi16(
            _mm_maddubs_epi16(vb89ABCDEFx0, va0x0),
            _mm_maddubs_epi16(vb89ABCDEFx1, va0x1));
        vacc0x01 = _mm_sub_epi32(vacc0x01, _mm_madd_epi16(vsum0x01, vminus_one));
        vacc0x23 = _mm_sub_epi32(vacc0x23, _mm_madd_epi16(vsum0x23, vminus_one));
        k -= 16 * sizeof(int8_t);
      }
    }
    if (k != 0) {
      const __m128i vbb01234567 = _mm_load_si128((const __m128i*) w);
      const __m128i vbb89ABCDEF = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      w = (const int8_t*) w + 32;

      const __m128i vb01234567x0 = _mm_and_si128(vbb01234567, vmask);
      const __m128i vb89ABCDEFx0 = _mm_and_si128(vbb89ABCDEF, vmask);
      const __m128i va0x0 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a0));
      a0 += 8;
      vacc0x01 = _mm_dpbusd_epi32_madd_kzp0(vacc0x01, vb01234567x0, va0x0);
      vacc0x23 = _mm_dpbusd_epi32_madd_kzp0(vacc0x23, vb89ABCDEFx0, va0x0);
      xnn_prefetch_to_l1((const int8_t*) w + 896);

      k -= 8 * sizeof(int8_t);
    }
    assert(k == 0);

    // Add adjacent pairs
    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);

    const __m128i vksum0123 = _mm_load_si128((const __m128i*) ksum);
    const __m128i vsum0x0123 = _mm_mullo_epi32(vksum0123, _mm_set1_epi32((int) quantization_params[0].zero_point));
    vacc0x0123 = _mm_add_epi32(vacc0x0123, vsum0x0123);
    __m128 vout0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 rh_zero_points_0123 = _mm_load_ps((const float*) (ksum + 4));
    const __m128 biased_rh_zero_points_0123 = _mm_add_ps(rh_zero_points_0123, _mm_set1_ps(2.0f));

    // Subtract out the scaled left-hand row sums.
    const __m128 lh_row_sum_0 = _mm_set1_ps(row_sum[0]);
    vout0x0123 = _mm_sub_ps(vout0x0123, _mm_mul_ps(biased_rh_zero_points_0123, lh_row_sum_0));
    // Add the product of left/right-hand zero points and `original_kc`.
    const __m128 vscaled_lh_zero_point_0 = _mm_set1_ps((float) original_kc * quantization_params[0].zero_point);
    vout0x0123 = _mm_add_ps(_mm_mul_ps(rh_zero_points_0123, vscaled_lh_zero_point_0), vout0x0123);
    vout0x0123 = _mm_mul_ps(vout0x0123, _mm_set1_ps(quantization_params[0].inv_scale));

    const __m128 vfilter_output_scale0123 = _mm_load_ps((const float*) w);
    const __m128 vbias0123 = _mm_load_ps((const float*) w + 4);
    w = (const float*) w + 8;

    vout0x0123 = _mm_add_ps(_mm_mul_ps(vout0x0123, vfilter_output_scale0123), vbias0123);

    const __m128 voutput_min = _mm_cvtph_ps(_mm_set1_epi16(*(const uint16_t*) &params->scalar.min));
    const __m128 voutput_max = _mm_cvtph_ps(_mm_set1_epi16(*(const uint16_t*) &params->scalar.max));
    vout0x0123 = _mm_max_ps(vout0x0123, voutput_min);

    vout0x0123 = _mm_min_ps(vout0x0123, voutput_max);

    __m128i vfp16out0x0123 = _mm_cvtps_ph(vout0x0123, _MM_FROUND_TO_NEAREST_INT);
    if XNN_LIKELY(nc >= 4) {
      _mm_storel_epi64((__m128i*) c0, vfp16out0x0123);
      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storeu_si32(c0, vfp16out0x0123);
        c0 += 2;
        vfp16out0x0123 = _mm_srli_epi64(vfp16out0x0123, 32);
      }
      if (nc & 1) {
        *c0 = (uint16_t) _mm_extract_epi16(vfp16out0x0123, 0);
      }
      nc = 0;
    }
  } while (nc != 0);
}
