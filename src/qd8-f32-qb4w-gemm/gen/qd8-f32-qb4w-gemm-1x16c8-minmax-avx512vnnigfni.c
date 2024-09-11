// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx16c8-avx512vnni.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/gemm.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/unaligned.h"


void xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_qb4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
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
  size_t bl = params->scalar.blocksize;
  assert(bl != 0);
  assert(bl <= kc);
  assert(kc % bl == 0);
  assert(bl % 32 == 0);

  const __m512i vsign_mask = _mm512_set1_epi8(0x80);
  XNN_FORCE_REALIZATION(vsign_mask);
  const __m512 vinput_zero_point0 = _mm512_set1_ps((float) quantization_params[0].zero_point + 128);
  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);
  // XNN_FORCE_REALIZATION(voutput_min);
  // XNN_FORCE_REALIZATION(voutput_max);
  const __m512i vmask = _mm512_set1_epi8(0xF0);
  XNN_FORCE_REALIZATION(vmask);
  const __m512i vshl4 = _mm512_set1_epi64(0x01020408);
  XNN_FORCE_REALIZATION(vshl4);
  do {
    const __m512 vksum0123456789ABCDEF = _mm512_loadu_ps(w);
    __m512 vscaled0x0123456789ABCDEF = _mm512_mul_ps(vksum0123456789ABCDEF, vinput_zero_point0);
    w = (const int32_t*) w + 16;

    for (size_t kb=0; kb < kc; kb+=bl) {
      __m512i vacc0x01234567 = _mm512_setzero_epi32();
      __m512i vacc0x89ABCDEF = _mm512_setzero_epi32();
      __m512i vacc1x0x01234567 = _mm512_setzero_epi32();
      __m512i vacc1x0x89ABCDEF = _mm512_setzero_epi32();
      size_t k = bl;
      while (k >= 16 * sizeof(int8_t)) {
        const __m512i va0x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a0)), vsign_mask);
        const __m512i va0x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a0 + 8)), vsign_mask);
        a0 += 16;

        const __m512i vbb01234567x01234567 = _mm512_loadu_si512(w);
        const __m512i vbb89ABCDEFx01234567 = _mm512_loadu_si512((const int8_t*) w + 64);
        const __m512i vb01234567x01234567 = _mm512_gf2p8affine_epi64_epi8(vbb01234567x01234567, vshl4, 0);
        const __m512i vb89ABCDEFx01234567 = _mm512_gf2p8affine_epi64_epi8(vbb89ABCDEFx01234567, vshl4, 0);
        const __m512i vb01234567x89ABCDEF = _mm512_and_si512(vbb01234567x01234567, vmask);
        const __m512i vb89ABCDEFx89ABCDEF = _mm512_and_si512(vbb89ABCDEFx01234567, vmask);

        vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, va0x01234567, vb01234567x01234567);
        vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, va0x01234567, vb89ABCDEFx01234567);
        vacc1x0x01234567 = _mm512_dpbusd_epi32(vacc1x0x01234567, va0x89ABCDEF, vb01234567x89ABCDEF);
        vacc1x0x89ABCDEF = _mm512_dpbusd_epi32(vacc1x0x89ABCDEF, va0x89ABCDEF, vb89ABCDEFx89ABCDEF);

        w = (const int8_t*) w + 128;
        k -= 16 * sizeof(int8_t);
      }

      if (k != 0) {
        const __m512i va0x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a0)), vsign_mask);
        a0 += 8;

        const __m512i vbb01234567x01234567 = _mm512_load_si512(w);
        const __m512i vbb89ABCDEFx01234567 = _mm512_load_si512((const int8_t*) w + 64);
        const __m512i vb01234567x01234567 = _mm512_gf2p8affine_epi64_epi8(vbb01234567x01234567, vshl4, 0);
        const __m512i vb89ABCDEFx01234567 = _mm512_gf2p8affine_epi64_epi8(vbb89ABCDEFx01234567, vshl4, 0);

        vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, va0x01234567, vb01234567x01234567);
        vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, va0x01234567, vb89ABCDEFx01234567);

        w = (const int8_t*) w + 128;
        k -= 8 * sizeof(int8_t);
      }
      const __m512 vfilter_output_scale0123456789ABCDEF = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm256_load_si256((const __m256i*) w)), 16));
      w = (const uint16_t*) w + 16;
      vacc0x01234567 = _mm512_add_epi32(vacc0x01234567, vacc1x0x01234567);
      vacc0x89ABCDEF = _mm512_add_epi32(vacc0x89ABCDEF, vacc1x0x89ABCDEF);

      // Add adjacent pairs
      const __m512i vidx = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
      const __m512i vsum0x01234567 = _mm512_add_epi32(vacc0x01234567, _mm512_srli_epi64(vacc0x01234567, 32));
      const __m512i vsum0x89ABCDEF = _mm512_add_epi32(vacc0x89ABCDEF, _mm512_srli_epi64(vacc0x89ABCDEF, 32));
      __m512i vacc0x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum0x01234567, vidx, vsum0x89ABCDEF);
      __m512 vf0x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0x0123456789ABCDEF);

      vscaled0x0123456789ABCDEF = _mm512_fmadd_ps(vf0x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vscaled0x0123456789ABCDEF);
    }

    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, _mm512_set1_ps(quantization_params[0].inv_scale));

    const __m512 vbias0123456789ABCDEF = _mm512_loadu_ps((const float*) w);
    w = (const float*) w + 16;

    vscaled0x0123456789ABCDEF = _mm512_add_ps(vscaled0x0123456789ABCDEF, vbias0123456789ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_max_ps(vscaled0x0123456789ABCDEF, voutput_min);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c0, vscaled0x0123456789ABCDEF);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((UINT32_C(1) << nc) - 1);
      _mm512_mask_storeu_ps(c0, vmask, vscaled0x0123456789ABCDEF);
      nc = 0;
    }
  } while (nc != 0);
}
