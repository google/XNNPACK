// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/c4-ace.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/ace.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/igemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/unaligned.h"


void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x32c4__ace(
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
    const struct xnn_f32_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params)
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

#if XNN_ARCH_X86_64
  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);
  const __m512i vzero_point = _mm512_set1_epi32((int) quantization_params->zero_point);
  const __m512 vinv_scale = _mm512_set1_ps(quantization_params->inv_scale);

  (void) mr;
  (void) cm_stride;

  do {
    __m512i vacc0 = _mm512_mullo_epi32(_mm512_loadu_epi32((const int32_t*) w + 0), vzero_point);
    __m512i vacc1 = _mm512_mullo_epi32(_mm512_loadu_epi32((const int32_t*) w + 16), vzero_point);
    w = (const int32_t*) w + 32;

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
      for (; k + 4 * sizeof(int8_t) <= kc; k += 4 * sizeof(int8_t)) {
        const __m512i va = _mm512_set1_epi32(unaligned_load_s32(a0 + k));
        const __m512i vb0 = _mm512_loadu_epi32((const int8_t*) w);
        vacc0 = xnn_avx10_dpbssd_epi32_t0(vacc0, va, vb0);
        if XNN_LIKELY(nc > 16) {
          const __m512i vb1 = _mm512_loadu_epi32((const int8_t*) w + 64);
          vacc1 = xnn_avx10_dpbssd_epi32_t1(vacc1, va, vb1);
        }
        w = (const int8_t*) w + 128;
      }
      if XNN_UNLIKELY(k < kc) {
        const __mmask16 vkmask = _cvtu32_mask16((UINT32_C(1) << (kc - k)) - UINT32_C(1));
        const __m512i va = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm_maskz_loadu_epi8(vkmask, a0 + k)));
        const __m512i vb0 = _mm512_loadu_epi32((const int8_t*) w);
        vacc0 = xnn_avx10_dpbssd_epi32_t0(vacc0, va, vb0);
        if XNN_LIKELY(nc > 16) {
          const __m512i vb1 = _mm512_loadu_epi32((const int8_t*) w + 64);
          vacc1 = xnn_avx10_dpbssd_epi32_t1(vacc1, va, vb1);
        }
        w = (const int8_t*) w + 128;
      }

      p -= sizeof(void*);
    } while (p != 0);

    const __m512 vfilter_output_scale0 = _mm512_loadu_ps((const float*) w + 0);
    const __m512 vfilter_output_scale1 = _mm512_loadu_ps((const float*) w + 16);
    w = (const int32_t*) w + 32;
    const __m512 vbias0 = _mm512_loadu_ps((const float*) w + 0);
    const __m512 vbias1 = _mm512_loadu_ps((const float*) w + 16);
    w = (const int32_t*) w + 32;

    __m512 vscaled0 = _mm512_cvtepi32_ps(vacc0);
    __m512 vscaled1 = _mm512_cvtepi32_ps(vacc1);
    vscaled0 = _mm512_mul_ps(vscaled0, vinv_scale);
    vscaled1 = _mm512_mul_ps(vscaled1, vinv_scale);
    vscaled0 = _mm512_fmadd_ps(vscaled0, vfilter_output_scale0, vbias0);
    vscaled1 = _mm512_fmadd_ps(vscaled1, vfilter_output_scale1, vbias1);
    vscaled0 = _mm512_max_ps(vscaled0, voutput_min);
    vscaled1 = _mm512_max_ps(vscaled1, voutput_min);
    vscaled0 = _mm512_min_ps(vscaled0, voutput_max);
    vscaled1 = _mm512_min_ps(vscaled1, voutput_max);

    if XNN_LIKELY(nc >= 32) {
      _mm512_storeu_ps(c + 0, vscaled0);
      _mm512_storeu_ps(c + 16, vscaled1);
      c = (float*) ((uintptr_t) c + cn_stride);
      a = (const int8_t** restrict) ((uintptr_t) a - ks);
      nc -= 32;
    } else {
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - UINT64_C(1)) >> 0) & 0xFFFF));
      _mm512_mask_storeu_ps(c + 0, vmask0, vscaled0);
      if (nc > 16) {
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - UINT64_C(1)) >> 16) & 0xFFFF));
      _mm512_mask_storeu_ps(c + 16, vmask1, vscaled1);
      }
      nc = 0;
    }
  } while (nc != 0);
#else
  (void) mr;
  (void) nc;
  (void) kc;
  (void) ks;
  (void) a;
  (void) w;
  (void) c;
  (void) cm_stride;
  (void) cn_stride;
  (void) a_offset;
  (void) zero;
  (void) zero_data;
  (void) params;
  (void) quantization_params;
#endif  // XNN_ARCH_X86_64
}
