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


void xnn_qd8_f32_qb4w_gemm_minmax_ukernel_12x16c8__avx512vnnigfni(
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
  assert(mr <= 12);
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
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const int8_t* a4 = (const int8_t*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const int8_t* a5 = (const int8_t*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const int8_t* a6 = (const int8_t*) ((uintptr_t) a5 + a_stride);
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }
  const int8_t* a7 = (const int8_t*) ((uintptr_t) a6 + a_stride);
  float* c7 = (float*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 8) {
    a7 = a6;
    c7 = c6;
  }
  const int8_t* a8 = (const int8_t*) ((uintptr_t) a7 + a_stride);
  float* c8 = (float*) ((uintptr_t) c7 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 8) {
    a8 = a7;
    c8 = c7;
  }
  const int8_t* a9 = (const int8_t*) ((uintptr_t) a8 + a_stride);
  float* c9 = (float*) ((uintptr_t) c8 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 10) {
    a9 = a8;
    c9 = c8;
  }
  const int8_t* a10 = (const int8_t*) ((uintptr_t) a9 + a_stride);
  float* c10 = (float*) ((uintptr_t) c9 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 10) {
    a10 = a9;
    c10 = c9;
  }
  const int8_t* a11 = (const int8_t*) ((uintptr_t) a10 + a_stride);
  float* c11 = (float*) ((uintptr_t) c10 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 12) {
    a11 = a10;
    c11 = c10;
  }
  size_t bl = params->avx512vnni.blocksize;
  assert(bl != 0);
  assert(bl <= kc);
  assert(kc % bl == 0);
  assert(bl % 32 == 0);

  const __m512i vsign_mask = _mm512_set1_epi8(0x80);
  XNN_FORCE_REALIZATION(vsign_mask);
  const __m512 vinput_zero_point0 = _mm512_set1_ps((float) quantization_params[0].zero_point + 128);
  const __m512 vinput_zero_point1 = _mm512_set1_ps((float) quantization_params[1].zero_point + 128);
  const __m512 vinput_zero_point2 = _mm512_set1_ps((float) quantization_params[2].zero_point + 128);
  const __m512 vinput_zero_point3 = _mm512_set1_ps((float) quantization_params[3].zero_point + 128);
  const __m512 vinput_zero_point4 = _mm512_set1_ps((float) quantization_params[4].zero_point + 128);
  const __m512 vinput_zero_point5 = _mm512_set1_ps((float) quantization_params[5].zero_point + 128);
  const __m512 vinput_zero_point6 = _mm512_set1_ps((float) quantization_params[6].zero_point + 128);
  const __m512 vinput_zero_point7 = _mm512_set1_ps((float) quantization_params[7].zero_point + 128);
  const __m512 vinput_zero_point8 = _mm512_set1_ps((float) quantization_params[8].zero_point + 128);
  const __m512 vinput_zero_point9 = _mm512_set1_ps((float) quantization_params[9].zero_point + 128);
  const __m512 vinput_zero_point10 = _mm512_set1_ps((float) quantization_params[10].zero_point + 128);
  const __m512 vinput_zero_point11 = _mm512_set1_ps((float) quantization_params[11].zero_point + 128);
  const __m512 voutput_min = _mm512_set1_ps(params->avx512vnni.min);
  const __m512 voutput_max = _mm512_set1_ps(params->avx512vnni.max);
  const __m512i vmask = _mm512_set1_epi8(0xF0);
  XNN_FORCE_REALIZATION(vmask);
  const __m512i vshl4 = _mm512_set1_epi64(0x01020408);
  XNN_FORCE_REALIZATION(vshl4);
  do {
    const __m512 vksum0123456789ABCDEF = _mm512_loadu_ps(w);
    __m512 vscaled0x0123456789ABCDEF = _mm512_mul_ps(vksum0123456789ABCDEF, vinput_zero_point0);
    __m512 vscaled1x0123456789ABCDEF = _mm512_mul_ps(vksum0123456789ABCDEF, vinput_zero_point1);
    __m512 vscaled2x0123456789ABCDEF = _mm512_mul_ps(vksum0123456789ABCDEF, vinput_zero_point2);
    __m512 vscaled3x0123456789ABCDEF = _mm512_mul_ps(vksum0123456789ABCDEF, vinput_zero_point3);
    __m512 vscaled4x0123456789ABCDEF = _mm512_mul_ps(vksum0123456789ABCDEF, vinput_zero_point4);
    __m512 vscaled5x0123456789ABCDEF = _mm512_mul_ps(vksum0123456789ABCDEF, vinput_zero_point5);
    __m512 vscaled6x0123456789ABCDEF = _mm512_mul_ps(vksum0123456789ABCDEF, vinput_zero_point6);
    __m512 vscaled7x0123456789ABCDEF = _mm512_mul_ps(vksum0123456789ABCDEF, vinput_zero_point7);
    __m512 vscaled8x0123456789ABCDEF = _mm512_mul_ps(vksum0123456789ABCDEF, vinput_zero_point8);
    __m512 vscaled9x0123456789ABCDEF = _mm512_mul_ps(vksum0123456789ABCDEF, vinput_zero_point9);
    __m512 vscaled10x0123456789ABCDEF = _mm512_mul_ps(vksum0123456789ABCDEF, vinput_zero_point10);
    __m512 vscaled11x0123456789ABCDEF = _mm512_mul_ps(vksum0123456789ABCDEF, vinput_zero_point11);
    w = (const int32_t*) w + 16;

    for (size_t kb=0; kb < kc; kb+=bl) {
      __m512i vacc0x01234567 = _mm512_setzero_epi32();
      __m512i vacc0x89ABCDEF = _mm512_setzero_epi32();
      __m512i vacc1x01234567 = _mm512_setzero_epi32();
      __m512i vacc1x89ABCDEF = _mm512_setzero_epi32();
      __m512i vacc2x01234567 = _mm512_setzero_epi32();
      __m512i vacc2x89ABCDEF = _mm512_setzero_epi32();
      __m512i vacc3x01234567 = _mm512_setzero_epi32();
      __m512i vacc3x89ABCDEF = _mm512_setzero_epi32();
      __m512i vacc4x01234567 = _mm512_setzero_epi32();
      __m512i vacc4x89ABCDEF = _mm512_setzero_epi32();
      __m512i vacc5x01234567 = _mm512_setzero_epi32();
      __m512i vacc5x89ABCDEF = _mm512_setzero_epi32();
      __m512i vacc6x01234567 = _mm512_setzero_epi32();
      __m512i vacc6x89ABCDEF = _mm512_setzero_epi32();
      __m512i vacc7x01234567 = _mm512_setzero_epi32();
      __m512i vacc7x89ABCDEF = _mm512_setzero_epi32();
      __m512i vacc8x01234567 = _mm512_setzero_epi32();
      __m512i vacc8x89ABCDEF = _mm512_setzero_epi32();
      __m512i vacc9x01234567 = _mm512_setzero_epi32();
      __m512i vacc9x89ABCDEF = _mm512_setzero_epi32();
      __m512i vacc10x01234567 = _mm512_setzero_epi32();
      __m512i vacc10x89ABCDEF = _mm512_setzero_epi32();
      __m512i vacc11x01234567 = _mm512_setzero_epi32();
      __m512i vacc11x89ABCDEF = _mm512_setzero_epi32();
      size_t k = bl;
      while (k >= 16 * sizeof(int8_t)) {
        const __m512i va0x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a0)), vsign_mask);
        const __m512i va0x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a0 + 8)), vsign_mask);
        a0 += 16;
        const __m512i va1x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a1)), vsign_mask);
        const __m512i va1x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a1 + 8)), vsign_mask);
        a1 += 16;
        const __m512i va2x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a2)), vsign_mask);
        const __m512i va2x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a2 + 8)), vsign_mask);
        a2 += 16;
        const __m512i va3x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a3)), vsign_mask);
        const __m512i va3x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a3 + 8)), vsign_mask);
        a3 += 16;
        const __m512i va4x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a4)), vsign_mask);
        const __m512i va4x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a4 + 8)), vsign_mask);
        a4 += 16;
        const __m512i va5x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a5)), vsign_mask);
        const __m512i va5x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a5 + 8)), vsign_mask);
        a5 += 16;
        const __m512i va6x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a6)), vsign_mask);
        const __m512i va6x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a6 + 8)), vsign_mask);
        a6 += 16;
        const __m512i va7x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a7)), vsign_mask);
        const __m512i va7x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a7 + 8)), vsign_mask);
        a7 += 16;
        const __m512i va8x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a8)), vsign_mask);
        const __m512i va8x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a8 + 8)), vsign_mask);
        a8 += 16;
        const __m512i va9x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a9)), vsign_mask);
        const __m512i va9x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a9 + 8)), vsign_mask);
        a9 += 16;
        const __m512i va10x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a10)), vsign_mask);
        const __m512i va10x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a10 + 8)), vsign_mask);
        a10 += 16;
        const __m512i va11x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a11)), vsign_mask);
        const __m512i va11x89ABCDEF = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a11 + 8)), vsign_mask);
        a11 += 16;

        const __m512i vbb01234567x01234567 = _mm512_loadu_si512(w);
        const __m512i vbb89ABCDEFx01234567 = _mm512_loadu_si512((const int8_t*) w + 64);
        const __m512i vb01234567x01234567 = _mm512_gf2p8affine_epi64_epi8(vbb01234567x01234567, vshl4, 0);
        const __m512i vb89ABCDEFx01234567 = _mm512_gf2p8affine_epi64_epi8(vbb89ABCDEFx01234567, vshl4, 0);
        const __m512i vb01234567x89ABCDEF = _mm512_and_si512(vbb01234567x01234567, vmask);
        const __m512i vb89ABCDEFx89ABCDEF = _mm512_and_si512(vbb89ABCDEFx01234567, vmask);

        vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, va0x01234567, vb01234567x01234567);
        vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, va0x01234567, vb89ABCDEFx01234567);
        vacc1x01234567 = _mm512_dpbusd_epi32(vacc1x01234567, va1x01234567, vb01234567x01234567);
        vacc1x89ABCDEF = _mm512_dpbusd_epi32(vacc1x89ABCDEF, va1x01234567, vb89ABCDEFx01234567);
        vacc2x01234567 = _mm512_dpbusd_epi32(vacc2x01234567, va2x01234567, vb01234567x01234567);
        vacc2x89ABCDEF = _mm512_dpbusd_epi32(vacc2x89ABCDEF, va2x01234567, vb89ABCDEFx01234567);
        vacc3x01234567 = _mm512_dpbusd_epi32(vacc3x01234567, va3x01234567, vb01234567x01234567);
        vacc3x89ABCDEF = _mm512_dpbusd_epi32(vacc3x89ABCDEF, va3x01234567, vb89ABCDEFx01234567);
        vacc4x01234567 = _mm512_dpbusd_epi32(vacc4x01234567, va4x01234567, vb01234567x01234567);
        vacc4x89ABCDEF = _mm512_dpbusd_epi32(vacc4x89ABCDEF, va4x01234567, vb89ABCDEFx01234567);
        vacc5x01234567 = _mm512_dpbusd_epi32(vacc5x01234567, va5x01234567, vb01234567x01234567);
        vacc5x89ABCDEF = _mm512_dpbusd_epi32(vacc5x89ABCDEF, va5x01234567, vb89ABCDEFx01234567);
        vacc6x01234567 = _mm512_dpbusd_epi32(vacc6x01234567, va6x01234567, vb01234567x01234567);
        vacc6x89ABCDEF = _mm512_dpbusd_epi32(vacc6x89ABCDEF, va6x01234567, vb89ABCDEFx01234567);
        vacc7x01234567 = _mm512_dpbusd_epi32(vacc7x01234567, va7x01234567, vb01234567x01234567);
        vacc7x89ABCDEF = _mm512_dpbusd_epi32(vacc7x89ABCDEF, va7x01234567, vb89ABCDEFx01234567);
        vacc8x01234567 = _mm512_dpbusd_epi32(vacc8x01234567, va8x01234567, vb01234567x01234567);
        vacc8x89ABCDEF = _mm512_dpbusd_epi32(vacc8x89ABCDEF, va8x01234567, vb89ABCDEFx01234567);
        vacc9x01234567 = _mm512_dpbusd_epi32(vacc9x01234567, va9x01234567, vb01234567x01234567);
        vacc9x89ABCDEF = _mm512_dpbusd_epi32(vacc9x89ABCDEF, va9x01234567, vb89ABCDEFx01234567);
        vacc10x01234567 = _mm512_dpbusd_epi32(vacc10x01234567, va10x01234567, vb01234567x01234567);
        vacc10x89ABCDEF = _mm512_dpbusd_epi32(vacc10x89ABCDEF, va10x01234567, vb89ABCDEFx01234567);
        vacc11x01234567 = _mm512_dpbusd_epi32(vacc11x01234567, va11x01234567, vb01234567x01234567);
        vacc11x89ABCDEF = _mm512_dpbusd_epi32(vacc11x89ABCDEF, va11x01234567, vb89ABCDEFx01234567);
        vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, va0x89ABCDEF, vb01234567x89ABCDEF);
        vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, va0x89ABCDEF, vb89ABCDEFx89ABCDEF);
        vacc1x01234567 = _mm512_dpbusd_epi32(vacc1x01234567, va1x89ABCDEF, vb01234567x89ABCDEF);
        vacc1x89ABCDEF = _mm512_dpbusd_epi32(vacc1x89ABCDEF, va1x89ABCDEF, vb89ABCDEFx89ABCDEF);
        vacc2x01234567 = _mm512_dpbusd_epi32(vacc2x01234567, va2x89ABCDEF, vb01234567x89ABCDEF);
        vacc2x89ABCDEF = _mm512_dpbusd_epi32(vacc2x89ABCDEF, va2x89ABCDEF, vb89ABCDEFx89ABCDEF);
        vacc3x01234567 = _mm512_dpbusd_epi32(vacc3x01234567, va3x89ABCDEF, vb01234567x89ABCDEF);
        vacc3x89ABCDEF = _mm512_dpbusd_epi32(vacc3x89ABCDEF, va3x89ABCDEF, vb89ABCDEFx89ABCDEF);
        vacc4x01234567 = _mm512_dpbusd_epi32(vacc4x01234567, va4x89ABCDEF, vb01234567x89ABCDEF);
        vacc4x89ABCDEF = _mm512_dpbusd_epi32(vacc4x89ABCDEF, va4x89ABCDEF, vb89ABCDEFx89ABCDEF);
        vacc5x01234567 = _mm512_dpbusd_epi32(vacc5x01234567, va5x89ABCDEF, vb01234567x89ABCDEF);
        vacc5x89ABCDEF = _mm512_dpbusd_epi32(vacc5x89ABCDEF, va5x89ABCDEF, vb89ABCDEFx89ABCDEF);
        vacc6x01234567 = _mm512_dpbusd_epi32(vacc6x01234567, va6x89ABCDEF, vb01234567x89ABCDEF);
        vacc6x89ABCDEF = _mm512_dpbusd_epi32(vacc6x89ABCDEF, va6x89ABCDEF, vb89ABCDEFx89ABCDEF);
        vacc7x01234567 = _mm512_dpbusd_epi32(vacc7x01234567, va7x89ABCDEF, vb01234567x89ABCDEF);
        vacc7x89ABCDEF = _mm512_dpbusd_epi32(vacc7x89ABCDEF, va7x89ABCDEF, vb89ABCDEFx89ABCDEF);
        vacc8x01234567 = _mm512_dpbusd_epi32(vacc8x01234567, va8x89ABCDEF, vb01234567x89ABCDEF);
        vacc8x89ABCDEF = _mm512_dpbusd_epi32(vacc8x89ABCDEF, va8x89ABCDEF, vb89ABCDEFx89ABCDEF);
        vacc9x01234567 = _mm512_dpbusd_epi32(vacc9x01234567, va9x89ABCDEF, vb01234567x89ABCDEF);
        vacc9x89ABCDEF = _mm512_dpbusd_epi32(vacc9x89ABCDEF, va9x89ABCDEF, vb89ABCDEFx89ABCDEF);
        vacc10x01234567 = _mm512_dpbusd_epi32(vacc10x01234567, va10x89ABCDEF, vb01234567x89ABCDEF);
        vacc10x89ABCDEF = _mm512_dpbusd_epi32(vacc10x89ABCDEF, va10x89ABCDEF, vb89ABCDEFx89ABCDEF);
        vacc11x01234567 = _mm512_dpbusd_epi32(vacc11x01234567, va11x89ABCDEF, vb01234567x89ABCDEF);
        vacc11x89ABCDEF = _mm512_dpbusd_epi32(vacc11x89ABCDEF, va11x89ABCDEF, vb89ABCDEFx89ABCDEF);

        w = (const int8_t*) w + 128;
        k -= 16 * sizeof(int8_t);
      }

      if (k != 0) {
        const __m512i va0x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a0)), vsign_mask);
        a0 += 8;
        const __m512i va1x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a1)), vsign_mask);
        a1 += 8;
        const __m512i va2x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a2)), vsign_mask);
        a2 += 8;
        const __m512i va3x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a3)), vsign_mask);
        a3 += 8;
        const __m512i va4x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a4)), vsign_mask);
        a4 += 8;
        const __m512i va5x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a5)), vsign_mask);
        a5 += 8;
        const __m512i va6x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a6)), vsign_mask);
        a6 += 8;
        const __m512i va7x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a7)), vsign_mask);
        a7 += 8;
        const __m512i va8x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a8)), vsign_mask);
        a8 += 8;
        const __m512i va9x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a9)), vsign_mask);
        a9 += 8;
        const __m512i va10x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a10)), vsign_mask);
        a10 += 8;
        const __m512i va11x01234567 = _mm512_xor_epi64(_mm512_set1_epi64((int64_t) unaligned_load_u64(a11)), vsign_mask);
        a11 += 8;

        const __m512i vbb01234567x01234567 = _mm512_load_si512(w);
        const __m512i vbb89ABCDEFx01234567 = _mm512_load_si512((const int8_t*) w + 64);
        const __m512i vb01234567x01234567 = _mm512_gf2p8affine_epi64_epi8(vbb01234567x01234567, vshl4, 0);
        const __m512i vb89ABCDEFx01234567 = _mm512_gf2p8affine_epi64_epi8(vbb89ABCDEFx01234567, vshl4, 0);

        vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, va0x01234567, vb01234567x01234567);
        vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, va0x01234567, vb89ABCDEFx01234567);
        vacc1x01234567 = _mm512_dpbusd_epi32(vacc1x01234567, va1x01234567, vb01234567x01234567);
        vacc1x89ABCDEF = _mm512_dpbusd_epi32(vacc1x89ABCDEF, va1x01234567, vb89ABCDEFx01234567);
        vacc2x01234567 = _mm512_dpbusd_epi32(vacc2x01234567, va2x01234567, vb01234567x01234567);
        vacc2x89ABCDEF = _mm512_dpbusd_epi32(vacc2x89ABCDEF, va2x01234567, vb89ABCDEFx01234567);
        vacc3x01234567 = _mm512_dpbusd_epi32(vacc3x01234567, va3x01234567, vb01234567x01234567);
        vacc3x89ABCDEF = _mm512_dpbusd_epi32(vacc3x89ABCDEF, va3x01234567, vb89ABCDEFx01234567);
        vacc4x01234567 = _mm512_dpbusd_epi32(vacc4x01234567, va4x01234567, vb01234567x01234567);
        vacc4x89ABCDEF = _mm512_dpbusd_epi32(vacc4x89ABCDEF, va4x01234567, vb89ABCDEFx01234567);
        vacc5x01234567 = _mm512_dpbusd_epi32(vacc5x01234567, va5x01234567, vb01234567x01234567);
        vacc5x89ABCDEF = _mm512_dpbusd_epi32(vacc5x89ABCDEF, va5x01234567, vb89ABCDEFx01234567);
        vacc6x01234567 = _mm512_dpbusd_epi32(vacc6x01234567, va6x01234567, vb01234567x01234567);
        vacc6x89ABCDEF = _mm512_dpbusd_epi32(vacc6x89ABCDEF, va6x01234567, vb89ABCDEFx01234567);
        vacc7x01234567 = _mm512_dpbusd_epi32(vacc7x01234567, va7x01234567, vb01234567x01234567);
        vacc7x89ABCDEF = _mm512_dpbusd_epi32(vacc7x89ABCDEF, va7x01234567, vb89ABCDEFx01234567);
        vacc8x01234567 = _mm512_dpbusd_epi32(vacc8x01234567, va8x01234567, vb01234567x01234567);
        vacc8x89ABCDEF = _mm512_dpbusd_epi32(vacc8x89ABCDEF, va8x01234567, vb89ABCDEFx01234567);
        vacc9x01234567 = _mm512_dpbusd_epi32(vacc9x01234567, va9x01234567, vb01234567x01234567);
        vacc9x89ABCDEF = _mm512_dpbusd_epi32(vacc9x89ABCDEF, va9x01234567, vb89ABCDEFx01234567);
        vacc10x01234567 = _mm512_dpbusd_epi32(vacc10x01234567, va10x01234567, vb01234567x01234567);
        vacc10x89ABCDEF = _mm512_dpbusd_epi32(vacc10x89ABCDEF, va10x01234567, vb89ABCDEFx01234567);
        vacc11x01234567 = _mm512_dpbusd_epi32(vacc11x01234567, va11x01234567, vb01234567x01234567);
        vacc11x89ABCDEF = _mm512_dpbusd_epi32(vacc11x89ABCDEF, va11x01234567, vb89ABCDEFx01234567);

        w = (const int8_t*) w + 128;
        k -= 8 * sizeof(int8_t);
      }
      const __m512 vfilter_output_scale0123456789ABCDEF = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm256_load_si256((const __m256i*) w)), 16));
      w = (const uint16_t*) w + 16;

      // Add adjacent pairs
      const __m512i vidx = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
      const __m512i vsum0x01234567 = _mm512_add_epi32(vacc0x01234567, _mm512_srli_epi64(vacc0x01234567, 32));
      const __m512i vsum0x89ABCDEF = _mm512_add_epi32(vacc0x89ABCDEF, _mm512_srli_epi64(vacc0x89ABCDEF, 32));
      __m512i vacc0x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum0x01234567, vidx, vsum0x89ABCDEF);
      const __m512i vsum1x01234567 = _mm512_add_epi32(vacc1x01234567, _mm512_srli_epi64(vacc1x01234567, 32));
      const __m512i vsum1x89ABCDEF = _mm512_add_epi32(vacc1x89ABCDEF, _mm512_srli_epi64(vacc1x89ABCDEF, 32));
      __m512i vacc1x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum1x01234567, vidx, vsum1x89ABCDEF);
      const __m512i vsum2x01234567 = _mm512_add_epi32(vacc2x01234567, _mm512_srli_epi64(vacc2x01234567, 32));
      const __m512i vsum2x89ABCDEF = _mm512_add_epi32(vacc2x89ABCDEF, _mm512_srli_epi64(vacc2x89ABCDEF, 32));
      __m512i vacc2x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum2x01234567, vidx, vsum2x89ABCDEF);
      const __m512i vsum3x01234567 = _mm512_add_epi32(vacc3x01234567, _mm512_srli_epi64(vacc3x01234567, 32));
      const __m512i vsum3x89ABCDEF = _mm512_add_epi32(vacc3x89ABCDEF, _mm512_srli_epi64(vacc3x89ABCDEF, 32));
      __m512i vacc3x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum3x01234567, vidx, vsum3x89ABCDEF);
      const __m512i vsum4x01234567 = _mm512_add_epi32(vacc4x01234567, _mm512_srli_epi64(vacc4x01234567, 32));
      const __m512i vsum4x89ABCDEF = _mm512_add_epi32(vacc4x89ABCDEF, _mm512_srli_epi64(vacc4x89ABCDEF, 32));
      __m512i vacc4x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum4x01234567, vidx, vsum4x89ABCDEF);
      const __m512i vsum5x01234567 = _mm512_add_epi32(vacc5x01234567, _mm512_srli_epi64(vacc5x01234567, 32));
      const __m512i vsum5x89ABCDEF = _mm512_add_epi32(vacc5x89ABCDEF, _mm512_srli_epi64(vacc5x89ABCDEF, 32));
      __m512i vacc5x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum5x01234567, vidx, vsum5x89ABCDEF);
      const __m512i vsum6x01234567 = _mm512_add_epi32(vacc6x01234567, _mm512_srli_epi64(vacc6x01234567, 32));
      const __m512i vsum6x89ABCDEF = _mm512_add_epi32(vacc6x89ABCDEF, _mm512_srli_epi64(vacc6x89ABCDEF, 32));
      __m512i vacc6x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum6x01234567, vidx, vsum6x89ABCDEF);
      const __m512i vsum7x01234567 = _mm512_add_epi32(vacc7x01234567, _mm512_srli_epi64(vacc7x01234567, 32));
      const __m512i vsum7x89ABCDEF = _mm512_add_epi32(vacc7x89ABCDEF, _mm512_srli_epi64(vacc7x89ABCDEF, 32));
      __m512i vacc7x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum7x01234567, vidx, vsum7x89ABCDEF);
      const __m512i vsum8x01234567 = _mm512_add_epi32(vacc8x01234567, _mm512_srli_epi64(vacc8x01234567, 32));
      const __m512i vsum8x89ABCDEF = _mm512_add_epi32(vacc8x89ABCDEF, _mm512_srli_epi64(vacc8x89ABCDEF, 32));
      __m512i vacc8x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum8x01234567, vidx, vsum8x89ABCDEF);
      const __m512i vsum9x01234567 = _mm512_add_epi32(vacc9x01234567, _mm512_srli_epi64(vacc9x01234567, 32));
      const __m512i vsum9x89ABCDEF = _mm512_add_epi32(vacc9x89ABCDEF, _mm512_srli_epi64(vacc9x89ABCDEF, 32));
      __m512i vacc9x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum9x01234567, vidx, vsum9x89ABCDEF);
      const __m512i vsum10x01234567 = _mm512_add_epi32(vacc10x01234567, _mm512_srli_epi64(vacc10x01234567, 32));
      const __m512i vsum10x89ABCDEF = _mm512_add_epi32(vacc10x89ABCDEF, _mm512_srli_epi64(vacc10x89ABCDEF, 32));
      __m512i vacc10x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum10x01234567, vidx, vsum10x89ABCDEF);
      const __m512i vsum11x01234567 = _mm512_add_epi32(vacc11x01234567, _mm512_srli_epi64(vacc11x01234567, 32));
      const __m512i vsum11x89ABCDEF = _mm512_add_epi32(vacc11x89ABCDEF, _mm512_srli_epi64(vacc11x89ABCDEF, 32));
      __m512i vacc11x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum11x01234567, vidx, vsum11x89ABCDEF);
      __m512 vf0x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0x0123456789ABCDEF);
      __m512 vf1x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc1x0123456789ABCDEF);
      __m512 vf2x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc2x0123456789ABCDEF);
      __m512 vf3x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc3x0123456789ABCDEF);
      __m512 vf4x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc4x0123456789ABCDEF);
      __m512 vf5x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc5x0123456789ABCDEF);
      __m512 vf6x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc6x0123456789ABCDEF);
      __m512 vf7x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc7x0123456789ABCDEF);
      __m512 vf8x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc8x0123456789ABCDEF);
      __m512 vf9x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc9x0123456789ABCDEF);
      __m512 vf10x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc10x0123456789ABCDEF);
      __m512 vf11x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc11x0123456789ABCDEF);

      vscaled0x0123456789ABCDEF = _mm512_fmadd_ps(vf0x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vscaled0x0123456789ABCDEF);
      vscaled1x0123456789ABCDEF = _mm512_fmadd_ps(vf1x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vscaled1x0123456789ABCDEF);
      vscaled2x0123456789ABCDEF = _mm512_fmadd_ps(vf2x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vscaled2x0123456789ABCDEF);
      vscaled3x0123456789ABCDEF = _mm512_fmadd_ps(vf3x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vscaled3x0123456789ABCDEF);
      vscaled4x0123456789ABCDEF = _mm512_fmadd_ps(vf4x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vscaled4x0123456789ABCDEF);
      vscaled5x0123456789ABCDEF = _mm512_fmadd_ps(vf5x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vscaled5x0123456789ABCDEF);
      vscaled6x0123456789ABCDEF = _mm512_fmadd_ps(vf6x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vscaled6x0123456789ABCDEF);
      vscaled7x0123456789ABCDEF = _mm512_fmadd_ps(vf7x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vscaled7x0123456789ABCDEF);
      vscaled8x0123456789ABCDEF = _mm512_fmadd_ps(vf8x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vscaled8x0123456789ABCDEF);
      vscaled9x0123456789ABCDEF = _mm512_fmadd_ps(vf9x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vscaled9x0123456789ABCDEF);
      vscaled10x0123456789ABCDEF = _mm512_fmadd_ps(vf10x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vscaled10x0123456789ABCDEF);
      vscaled11x0123456789ABCDEF = _mm512_fmadd_ps(vf11x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vscaled11x0123456789ABCDEF);
    }

    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, _mm512_set1_ps(quantization_params[0].inv_scale));
    vscaled1x0123456789ABCDEF = _mm512_mul_ps(vscaled1x0123456789ABCDEF, _mm512_set1_ps(quantization_params[1].inv_scale));
    vscaled2x0123456789ABCDEF = _mm512_mul_ps(vscaled2x0123456789ABCDEF, _mm512_set1_ps(quantization_params[2].inv_scale));
    vscaled3x0123456789ABCDEF = _mm512_mul_ps(vscaled3x0123456789ABCDEF, _mm512_set1_ps(quantization_params[3].inv_scale));
    vscaled4x0123456789ABCDEF = _mm512_mul_ps(vscaled4x0123456789ABCDEF, _mm512_set1_ps(quantization_params[4].inv_scale));
    vscaled5x0123456789ABCDEF = _mm512_mul_ps(vscaled5x0123456789ABCDEF, _mm512_set1_ps(quantization_params[5].inv_scale));
    vscaled6x0123456789ABCDEF = _mm512_mul_ps(vscaled6x0123456789ABCDEF, _mm512_set1_ps(quantization_params[6].inv_scale));
    vscaled7x0123456789ABCDEF = _mm512_mul_ps(vscaled7x0123456789ABCDEF, _mm512_set1_ps(quantization_params[7].inv_scale));
    vscaled8x0123456789ABCDEF = _mm512_mul_ps(vscaled8x0123456789ABCDEF, _mm512_set1_ps(quantization_params[8].inv_scale));
    vscaled9x0123456789ABCDEF = _mm512_mul_ps(vscaled9x0123456789ABCDEF, _mm512_set1_ps(quantization_params[9].inv_scale));
    vscaled10x0123456789ABCDEF = _mm512_mul_ps(vscaled10x0123456789ABCDEF, _mm512_set1_ps(quantization_params[10].inv_scale));
    vscaled11x0123456789ABCDEF = _mm512_mul_ps(vscaled11x0123456789ABCDEF, _mm512_set1_ps(quantization_params[11].inv_scale));

    const __m512 vbias0123456789ABCDEF = _mm512_loadu_ps((const float*) w);
    w = (const float*) w + 16;

    vscaled0x0123456789ABCDEF = _mm512_add_ps(vscaled0x0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled1x0123456789ABCDEF = _mm512_add_ps(vscaled1x0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled2x0123456789ABCDEF = _mm512_add_ps(vscaled2x0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled3x0123456789ABCDEF = _mm512_add_ps(vscaled3x0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled4x0123456789ABCDEF = _mm512_add_ps(vscaled4x0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled5x0123456789ABCDEF = _mm512_add_ps(vscaled5x0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled6x0123456789ABCDEF = _mm512_add_ps(vscaled6x0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled7x0123456789ABCDEF = _mm512_add_ps(vscaled7x0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled8x0123456789ABCDEF = _mm512_add_ps(vscaled8x0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled9x0123456789ABCDEF = _mm512_add_ps(vscaled9x0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled10x0123456789ABCDEF = _mm512_add_ps(vscaled10x0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled11x0123456789ABCDEF = _mm512_add_ps(vscaled11x0123456789ABCDEF, vbias0123456789ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_max_ps(vscaled0x0123456789ABCDEF, voutput_min);
    vscaled1x0123456789ABCDEF = _mm512_max_ps(vscaled1x0123456789ABCDEF, voutput_min);
    vscaled2x0123456789ABCDEF = _mm512_max_ps(vscaled2x0123456789ABCDEF, voutput_min);
    vscaled3x0123456789ABCDEF = _mm512_max_ps(vscaled3x0123456789ABCDEF, voutput_min);
    vscaled4x0123456789ABCDEF = _mm512_max_ps(vscaled4x0123456789ABCDEF, voutput_min);
    vscaled5x0123456789ABCDEF = _mm512_max_ps(vscaled5x0123456789ABCDEF, voutput_min);
    vscaled6x0123456789ABCDEF = _mm512_max_ps(vscaled6x0123456789ABCDEF, voutput_min);
    vscaled7x0123456789ABCDEF = _mm512_max_ps(vscaled7x0123456789ABCDEF, voutput_min);
    vscaled8x0123456789ABCDEF = _mm512_max_ps(vscaled8x0123456789ABCDEF, voutput_min);
    vscaled9x0123456789ABCDEF = _mm512_max_ps(vscaled9x0123456789ABCDEF, voutput_min);
    vscaled10x0123456789ABCDEF = _mm512_max_ps(vscaled10x0123456789ABCDEF, voutput_min);
    vscaled11x0123456789ABCDEF = _mm512_max_ps(vscaled11x0123456789ABCDEF, voutput_min);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max);
    vscaled1x0123456789ABCDEF = _mm512_min_ps(vscaled1x0123456789ABCDEF, voutput_max);
    vscaled2x0123456789ABCDEF = _mm512_min_ps(vscaled2x0123456789ABCDEF, voutput_max);
    vscaled3x0123456789ABCDEF = _mm512_min_ps(vscaled3x0123456789ABCDEF, voutput_max);
    vscaled4x0123456789ABCDEF = _mm512_min_ps(vscaled4x0123456789ABCDEF, voutput_max);
    vscaled5x0123456789ABCDEF = _mm512_min_ps(vscaled5x0123456789ABCDEF, voutput_max);
    vscaled6x0123456789ABCDEF = _mm512_min_ps(vscaled6x0123456789ABCDEF, voutput_max);
    vscaled7x0123456789ABCDEF = _mm512_min_ps(vscaled7x0123456789ABCDEF, voutput_max);
    vscaled8x0123456789ABCDEF = _mm512_min_ps(vscaled8x0123456789ABCDEF, voutput_max);
    vscaled9x0123456789ABCDEF = _mm512_min_ps(vscaled9x0123456789ABCDEF, voutput_max);
    vscaled10x0123456789ABCDEF = _mm512_min_ps(vscaled10x0123456789ABCDEF, voutput_max);
    vscaled11x0123456789ABCDEF = _mm512_min_ps(vscaled11x0123456789ABCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c0, vscaled0x0123456789ABCDEF);
      _mm512_storeu_ps(c1, vscaled1x0123456789ABCDEF);
      _mm512_storeu_ps(c2, vscaled2x0123456789ABCDEF);
      _mm512_storeu_ps(c3, vscaled3x0123456789ABCDEF);
      _mm512_storeu_ps(c4, vscaled4x0123456789ABCDEF);
      _mm512_storeu_ps(c5, vscaled5x0123456789ABCDEF);
      _mm512_storeu_ps(c6, vscaled6x0123456789ABCDEF);
      _mm512_storeu_ps(c7, vscaled7x0123456789ABCDEF);
      _mm512_storeu_ps(c8, vscaled8x0123456789ABCDEF);
      _mm512_storeu_ps(c9, vscaled9x0123456789ABCDEF);
      _mm512_storeu_ps(c10, vscaled10x0123456789ABCDEF);
      _mm512_storeu_ps(c11, vscaled11x0123456789ABCDEF);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);
      a4 = (const int8_t*) ((uintptr_t) a4 - kc);
      a5 = (const int8_t*) ((uintptr_t) a5 - kc);
      a6 = (const int8_t*) ((uintptr_t) a6 - kc);
      a7 = (const int8_t*) ((uintptr_t) a7 - kc);
      a8 = (const int8_t*) ((uintptr_t) a8 - kc);
      a9 = (const int8_t*) ((uintptr_t) a9 - kc);
      a10 = (const int8_t*) ((uintptr_t) a10 - kc);
      a11 = (const int8_t*) ((uintptr_t) a11 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      c7 = (float*) ((uintptr_t) c7 + cn_stride);
      c8 = (float*) ((uintptr_t) c8 + cn_stride);
      c9 = (float*) ((uintptr_t) c9 + cn_stride);
      c10 = (float*) ((uintptr_t) c10 + cn_stride);
      c11 = (float*) ((uintptr_t) c11 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((UINT32_C(1) << nc) - 1);
      _mm512_mask_storeu_ps(c0, vmask, vscaled0x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c1, vmask, vscaled1x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c2, vmask, vscaled2x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c3, vmask, vscaled3x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c4, vmask, vscaled4x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c5, vmask, vscaled5x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c6, vmask, vscaled6x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c7, vmask, vscaled7x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c8, vmask, vscaled8x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c9, vmask, vscaled9x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c10, vmask, vscaled10x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c11, vmask, vscaled11x0123456789ABCDEF);
      nc = 0;
    }
  } while (nc != 0);
}
