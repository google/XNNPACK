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

#include <smmintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/prefetch.h"
#include "src/xnnpack/unaligned.h"


void xnn_qd8_f32_qc2w_gemm_minmax_ukernel_4x4c8__avx_madd_prfm(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* restrict params,
    const float* row_sum,
    const struct xnn_qd8_quantization_params* restrict quantization_params) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const size_t original_kc = kc;
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
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const __m128i vmask = _mm_set1_epi8(0x03);
  const __m128i vminus_one = _mm_set1_epi16(-1);
  XNN_FORCE_REALIZATION(vmask);
  do {
    const int32_t* ksum = (const int32_t*) w;
    w = (const int32_t*) w + 8;
    __m128i vacc0x01 = _mm_setzero_si128();
    __m128i vacc0x23 = _mm_setzero_si128();
    __m128i vacc1x01 = _mm_setzero_si128();
    __m128i vacc1x23 = _mm_setzero_si128();
    __m128i vacc2x01 = _mm_setzero_si128();
    __m128i vacc2x23 = _mm_setzero_si128();
    __m128i vacc3x01 = _mm_setzero_si128();
    __m128i vacc3x23 = _mm_setzero_si128();

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
      const __m128i va1x0 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a1));
      const __m128i va1x1 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a1 + 8));
      const __m128i va1x2 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a1 + 16));
      const __m128i va1x3 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a1 + 24));
      a1 += 32;

      __m128i vsum1x01 = _mm_add_epi16(
          _mm_maddubs_epi16(vb01234567x0, va1x0),
          _mm_maddubs_epi16(vb01234567x1, va1x1));
      __m128i vsum1x23 = _mm_add_epi16(
          _mm_maddubs_epi16(vb89ABCDEFx0, va1x0),
          _mm_maddubs_epi16(vb89ABCDEFx1, va1x1));
      vsum1x01 = _mm_add_epi16(
          vsum1x01,
          _mm_add_epi16(
              _mm_maddubs_epi16(vb01234567x2, va1x2),
              _mm_maddubs_epi16(vb01234567x3, va1x3)));
      vsum1x23 = _mm_add_epi16(
          vsum1x23,
          _mm_add_epi16(
              _mm_maddubs_epi16(vb89ABCDEFx2, va1x2),
              _mm_maddubs_epi16(vb89ABCDEFx3, va1x3)));
      vacc1x01 = _mm_sub_epi32(vacc1x01, _mm_madd_epi16(vsum1x01, vminus_one));
      vacc1x23 = _mm_sub_epi32(vacc1x23, _mm_madd_epi16(vsum1x23, vminus_one));
      const __m128i va2x0 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a2));
      const __m128i va2x1 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a2 + 8));
      const __m128i va2x2 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a2 + 16));
      const __m128i va2x3 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a2 + 24));
      a2 += 32;

      __m128i vsum2x01 = _mm_add_epi16(
          _mm_maddubs_epi16(vb01234567x0, va2x0),
          _mm_maddubs_epi16(vb01234567x1, va2x1));
      __m128i vsum2x23 = _mm_add_epi16(
          _mm_maddubs_epi16(vb89ABCDEFx0, va2x0),
          _mm_maddubs_epi16(vb89ABCDEFx1, va2x1));
      vsum2x01 = _mm_add_epi16(
          vsum2x01,
          _mm_add_epi16(
              _mm_maddubs_epi16(vb01234567x2, va2x2),
              _mm_maddubs_epi16(vb01234567x3, va2x3)));
      vsum2x23 = _mm_add_epi16(
          vsum2x23,
          _mm_add_epi16(
              _mm_maddubs_epi16(vb89ABCDEFx2, va2x2),
              _mm_maddubs_epi16(vb89ABCDEFx3, va2x3)));
      vacc2x01 = _mm_sub_epi32(vacc2x01, _mm_madd_epi16(vsum2x01, vminus_one));
      vacc2x23 = _mm_sub_epi32(vacc2x23, _mm_madd_epi16(vsum2x23, vminus_one));
      const __m128i va3x0 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a3));
      const __m128i va3x1 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a3 + 8));
      const __m128i va3x2 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a3 + 16));
      const __m128i va3x3 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a3 + 24));
      a3 += 32;

      __m128i vsum3x01 = _mm_add_epi16(
          _mm_maddubs_epi16(vb01234567x0, va3x0),
          _mm_maddubs_epi16(vb01234567x1, va3x1));
      __m128i vsum3x23 = _mm_add_epi16(
          _mm_maddubs_epi16(vb89ABCDEFx0, va3x0),
          _mm_maddubs_epi16(vb89ABCDEFx1, va3x1));
      vsum3x01 = _mm_add_epi16(
          vsum3x01,
          _mm_add_epi16(
              _mm_maddubs_epi16(vb01234567x2, va3x2),
              _mm_maddubs_epi16(vb01234567x3, va3x3)));
      vsum3x23 = _mm_add_epi16(
          vsum3x23,
          _mm_add_epi16(
              _mm_maddubs_epi16(vb89ABCDEFx2, va3x2),
              _mm_maddubs_epi16(vb89ABCDEFx3, va3x3)));
      vacc3x01 = _mm_sub_epi32(vacc3x01, _mm_madd_epi16(vsum3x01, vminus_one));
      vacc3x23 = _mm_sub_epi32(vacc3x23, _mm_madd_epi16(vsum3x23, vminus_one));

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
        const __m128i va1x0 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a1));
        const __m128i va1x1 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a1 + 8));
        const __m128i va1x2 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a1 + 16));
        a1 += 24;

        __m128i vsum1x01 = _mm_add_epi16(
            _mm_maddubs_epi16(vb01234567x0, va1x0),
            _mm_maddubs_epi16(vb01234567x1, va1x1));
        __m128i vsum1x23 = _mm_add_epi16(
            _mm_maddubs_epi16(vb89ABCDEFx0, va1x0),
            _mm_maddubs_epi16(vb89ABCDEFx1, va1x1));
        vsum1x01 = _mm_add_epi16(vsum1x01, _mm_maddubs_epi16(vb01234567x2, va1x2));
        vsum1x23 = _mm_add_epi16(vsum1x23, _mm_maddubs_epi16(vb89ABCDEFx2, va1x2));
        vacc1x01 = _mm_sub_epi32(vacc1x01, _mm_madd_epi16(vsum1x01, vminus_one));
        vacc1x23 = _mm_sub_epi32(vacc1x23, _mm_madd_epi16(vsum1x23, vminus_one));
        const __m128i va2x0 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a2));
        const __m128i va2x1 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a2 + 8));
        const __m128i va2x2 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a2 + 16));
        a2 += 24;

        __m128i vsum2x01 = _mm_add_epi16(
            _mm_maddubs_epi16(vb01234567x0, va2x0),
            _mm_maddubs_epi16(vb01234567x1, va2x1));
        __m128i vsum2x23 = _mm_add_epi16(
            _mm_maddubs_epi16(vb89ABCDEFx0, va2x0),
            _mm_maddubs_epi16(vb89ABCDEFx1, va2x1));
        vsum2x01 = _mm_add_epi16(vsum2x01, _mm_maddubs_epi16(vb01234567x2, va2x2));
        vsum2x23 = _mm_add_epi16(vsum2x23, _mm_maddubs_epi16(vb89ABCDEFx2, va2x2));
        vacc2x01 = _mm_sub_epi32(vacc2x01, _mm_madd_epi16(vsum2x01, vminus_one));
        vacc2x23 = _mm_sub_epi32(vacc2x23, _mm_madd_epi16(vsum2x23, vminus_one));
        const __m128i va3x0 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a3));
        const __m128i va3x1 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a3 + 8));
        const __m128i va3x2 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a3 + 16));
        a3 += 24;

        __m128i vsum3x01 = _mm_add_epi16(
            _mm_maddubs_epi16(vb01234567x0, va3x0),
            _mm_maddubs_epi16(vb01234567x1, va3x1));
        __m128i vsum3x23 = _mm_add_epi16(
            _mm_maddubs_epi16(vb89ABCDEFx0, va3x0),
            _mm_maddubs_epi16(vb89ABCDEFx1, va3x1));
        vsum3x01 = _mm_add_epi16(vsum3x01, _mm_maddubs_epi16(vb01234567x2, va3x2));
        vsum3x23 = _mm_add_epi16(vsum3x23, _mm_maddubs_epi16(vb89ABCDEFx2, va3x2));
        vacc3x01 = _mm_sub_epi32(vacc3x01, _mm_madd_epi16(vsum3x01, vminus_one));
        vacc3x23 = _mm_sub_epi32(vacc3x23, _mm_madd_epi16(vsum3x23, vminus_one));
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
        const __m128i va1x0 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a1));
        const __m128i va1x1 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a1 + 8));
        a1 += 16;

        const __m128i vsum1x01 = _mm_add_epi16(
            _mm_maddubs_epi16(vb01234567x0, va1x0),
            _mm_maddubs_epi16(vb01234567x1, va1x1));
        const __m128i vsum1x23 = _mm_add_epi16(
            _mm_maddubs_epi16(vb89ABCDEFx0, va1x0),
            _mm_maddubs_epi16(vb89ABCDEFx1, va1x1));
        vacc1x01 = _mm_sub_epi32(vacc1x01, _mm_madd_epi16(vsum1x01, vminus_one));
        vacc1x23 = _mm_sub_epi32(vacc1x23, _mm_madd_epi16(vsum1x23, vminus_one));
        const __m128i va2x0 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a2));
        const __m128i va2x1 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a2 + 8));
        a2 += 16;

        const __m128i vsum2x01 = _mm_add_epi16(
            _mm_maddubs_epi16(vb01234567x0, va2x0),
            _mm_maddubs_epi16(vb01234567x1, va2x1));
        const __m128i vsum2x23 = _mm_add_epi16(
            _mm_maddubs_epi16(vb89ABCDEFx0, va2x0),
            _mm_maddubs_epi16(vb89ABCDEFx1, va2x1));
        vacc2x01 = _mm_sub_epi32(vacc2x01, _mm_madd_epi16(vsum2x01, vminus_one));
        vacc2x23 = _mm_sub_epi32(vacc2x23, _mm_madd_epi16(vsum2x23, vminus_one));
        const __m128i va3x0 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a3));
        const __m128i va3x1 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a3 + 8));
        a3 += 16;

        const __m128i vsum3x01 = _mm_add_epi16(
            _mm_maddubs_epi16(vb01234567x0, va3x0),
            _mm_maddubs_epi16(vb01234567x1, va3x1));
        const __m128i vsum3x23 = _mm_add_epi16(
            _mm_maddubs_epi16(vb89ABCDEFx0, va3x0),
            _mm_maddubs_epi16(vb89ABCDEFx1, va3x1));
        vacc3x01 = _mm_sub_epi32(vacc3x01, _mm_madd_epi16(vsum3x01, vminus_one));
        vacc3x23 = _mm_sub_epi32(vacc3x23, _mm_madd_epi16(vsum3x23, vminus_one));
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
      const __m128i va1x0 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a1));
      a1 += 8;
      vacc1x01 = _mm_dpbusd_epi32_madd_kzp0(vacc1x01, vb01234567x0, va1x0);
      vacc1x23 = _mm_dpbusd_epi32_madd_kzp0(vacc1x23, vb89ABCDEFx0, va1x0);
      const __m128i va2x0 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a2));
      a2 += 8;
      vacc2x01 = _mm_dpbusd_epi32_madd_kzp0(vacc2x01, vb01234567x0, va2x0);
      vacc2x23 = _mm_dpbusd_epi32_madd_kzp0(vacc2x23, vb89ABCDEFx0, va2x0);
      const __m128i va3x0 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a3));
      a3 += 8;
      vacc3x01 = _mm_dpbusd_epi32_madd_kzp0(vacc3x01, vb01234567x0, va3x0);
      vacc3x23 = _mm_dpbusd_epi32_madd_kzp0(vacc3x23, vb89ABCDEFx0, va3x0);
      xnn_prefetch_to_l1((const int8_t*) w + 896);

      k -= 8 * sizeof(int8_t);
    }
    assert(k == 0);

    // Add adjacent pairs
    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
    __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);
    __m128i vacc2x0123 = _mm_hadd_epi32(vacc2x01, vacc2x23);
    __m128i vacc3x0123 = _mm_hadd_epi32(vacc3x01, vacc3x23);

    const __m128i vksum0123 = _mm_load_si128((const __m128i*) ksum);
    const __m128i vsum0x0123 = _mm_mullo_epi32(vksum0123, _mm_set1_epi32((int) quantization_params[0].zero_point));
    vacc0x0123 = _mm_add_epi32(vacc0x0123, vsum0x0123);
    const __m128i vsum1x0123 = _mm_mullo_epi32(vksum0123, _mm_set1_epi32((int) quantization_params[1].zero_point));
    vacc1x0123 = _mm_add_epi32(vacc1x0123, vsum1x0123);
    const __m128i vsum2x0123 = _mm_mullo_epi32(vksum0123, _mm_set1_epi32((int) quantization_params[2].zero_point));
    vacc2x0123 = _mm_add_epi32(vacc2x0123, vsum2x0123);
    const __m128i vsum3x0123 = _mm_mullo_epi32(vksum0123, _mm_set1_epi32((int) quantization_params[3].zero_point));
    vacc3x0123 = _mm_add_epi32(vacc3x0123, vsum3x0123);
    __m128 vout0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vout1x0123 = _mm_cvtepi32_ps(vacc1x0123);
    __m128 vout2x0123 = _mm_cvtepi32_ps(vacc2x0123);
    __m128 vout3x0123 = _mm_cvtepi32_ps(vacc3x0123);

    const __m128 rh_zero_points_0123 = _mm_load_ps((const float*) (ksum + 4));
    const __m128 biased_rh_zero_points_0123 = _mm_add_ps(rh_zero_points_0123, _mm_set1_ps(2.0f));

    // Subtract out the scaled left-hand row sums.
    const __m128 lh_row_sum_0 = _mm_set1_ps(row_sum[0]);
    vout0x0123 = _mm_sub_ps(vout0x0123, _mm_mul_ps(biased_rh_zero_points_0123, lh_row_sum_0));
    const __m128 lh_row_sum_1 = _mm_set1_ps(row_sum[1]);
    vout1x0123 = _mm_sub_ps(vout1x0123, _mm_mul_ps(biased_rh_zero_points_0123, lh_row_sum_1));
    const __m128 lh_row_sum_2 = _mm_set1_ps(row_sum[2]);
    vout2x0123 = _mm_sub_ps(vout2x0123, _mm_mul_ps(biased_rh_zero_points_0123, lh_row_sum_2));
    const __m128 lh_row_sum_3 = _mm_set1_ps(row_sum[3]);
    vout3x0123 = _mm_sub_ps(vout3x0123, _mm_mul_ps(biased_rh_zero_points_0123, lh_row_sum_3));
    // Add the product of left/right-hand zero points and `original_kc`.
    const __m128 vscaled_lh_zero_point_0 = _mm_set1_ps((float) original_kc * quantization_params[0].zero_point);
    const __m128 vscaled_lh_zero_point_1 = _mm_set1_ps((float) original_kc * quantization_params[1].zero_point);
    const __m128 vscaled_lh_zero_point_2 = _mm_set1_ps((float) original_kc * quantization_params[2].zero_point);
    const __m128 vscaled_lh_zero_point_3 = _mm_set1_ps((float) original_kc * quantization_params[3].zero_point);
    vout0x0123 = _mm_add_ps(_mm_mul_ps(rh_zero_points_0123, vscaled_lh_zero_point_0), vout0x0123);
    vout1x0123 = _mm_add_ps(_mm_mul_ps(rh_zero_points_0123, vscaled_lh_zero_point_1), vout1x0123);
    vout2x0123 = _mm_add_ps(_mm_mul_ps(rh_zero_points_0123, vscaled_lh_zero_point_2), vout2x0123);
    vout3x0123 = _mm_add_ps(_mm_mul_ps(rh_zero_points_0123, vscaled_lh_zero_point_3), vout3x0123);
    vout0x0123 = _mm_mul_ps(vout0x0123, _mm_set1_ps(quantization_params[0].inv_scale));
    vout1x0123 = _mm_mul_ps(vout1x0123, _mm_set1_ps(quantization_params[1].inv_scale));
    vout2x0123 = _mm_mul_ps(vout2x0123, _mm_set1_ps(quantization_params[2].inv_scale));
    vout3x0123 = _mm_mul_ps(vout3x0123, _mm_set1_ps(quantization_params[3].inv_scale));

    const __m128 vfilter_output_scale0123 = _mm_load_ps((const float*) w);
    const __m128 vbias0123 = _mm_load_ps((const float*) w + 4);
    w = (const float*) w + 8;

    vout0x0123 = _mm_add_ps(_mm_mul_ps(vout0x0123, vfilter_output_scale0123), vbias0123);
    vout1x0123 = _mm_add_ps(_mm_mul_ps(vout1x0123, vfilter_output_scale0123), vbias0123);
    vout2x0123 = _mm_add_ps(_mm_mul_ps(vout2x0123, vfilter_output_scale0123), vbias0123);
    vout3x0123 = _mm_add_ps(_mm_mul_ps(vout3x0123, vfilter_output_scale0123), vbias0123);

    const __m128 voutput_min = _mm_set1_ps(params->scalar.min);
    const __m128 voutput_max = _mm_set1_ps(params->scalar.max);
    vout0x0123 = _mm_max_ps(vout0x0123, voutput_min);
    vout1x0123 = _mm_max_ps(vout1x0123, voutput_min);
    vout2x0123 = _mm_max_ps(vout2x0123, voutput_min);
    vout3x0123 = _mm_max_ps(vout3x0123, voutput_min);

    vout0x0123 = _mm_min_ps(vout0x0123, voutput_max);
    vout1x0123 = _mm_min_ps(vout1x0123, voutput_max);
    vout2x0123 = _mm_min_ps(vout2x0123, voutput_max);
    vout3x0123 = _mm_min_ps(vout3x0123, voutput_max);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c0, vout0x0123);
      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      _mm_storeu_ps(c1, vout1x0123);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm_storeu_ps(c2, vout2x0123);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm_storeu_ps(c3, vout3x0123);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vout0x0123);
        c0 += 2;
        vout0x0123 = _mm_movehl_ps(vout0x0123, vout0x0123);
        _mm_storel_pi((__m64*) c1, vout1x0123);
        c1 += 2;
        vout1x0123 = _mm_movehl_ps(vout1x0123, vout1x0123);
        _mm_storel_pi((__m64*) c2, vout2x0123);
        c2 += 2;
        vout2x0123 = _mm_movehl_ps(vout2x0123, vout2x0123);
        _mm_storel_pi((__m64*) c3, vout3x0123);
        c3 += 2;
        vout3x0123 = _mm_movehl_ps(vout3x0123, vout3x0123);
      }
      if (nc & 1) {
        _mm_store_ss(c0, vout0x0123);
        _mm_store_ss(c1, vout1x0123);
        _mm_store_ss(c2, vout2x0123);
        _mm_store_ss(c3, vout3x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}
