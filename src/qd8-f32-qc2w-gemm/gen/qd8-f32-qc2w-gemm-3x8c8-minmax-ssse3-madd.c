// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx8c8-ssevnni.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <tmmintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/unaligned.h"


void xnn_qd8_f32_qc2w_gemm_minmax_ukernel_3x8c8__ssse3_madd(
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
  assert(mr <= 3);
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

  const __m128i vinput_zero_point0 = _mm_set1_epi32((int) quantization_params[0].zero_point);
  const __m128i vinput_zero_point1 = _mm_set1_epi32((int) quantization_params[1].zero_point);
  const __m128i vinput_zero_point2 = _mm_set1_epi32((int) quantization_params[2].zero_point);
  const __m128 voutput_min = _mm_set1_ps(params->scalar.min);
  const __m128 voutput_max = _mm_set1_ps(params->scalar.max);
  const __m128i vmask = _mm_set1_epi8(0x03);
  do {
    const __m128i vksum0123 = _mm_load_si128((const __m128i*) w);
    const __m128i vksum4567 = _mm_load_si128((const __m128i*) ((const int32_t*) w + 4));
    const __m128i vksum13 = _mm_shuffle_epi32(vksum0123, 0xF5);
    const __m128i vksum57 = _mm_shuffle_epi32(vksum4567, 0xF5);
    const __m128i vsum0x02 = _mm_mul_epu32(vksum0123, vinput_zero_point0);
    const __m128i vsum0x13 = _mm_mul_epu32(vksum13, vinput_zero_point0);
    const __m128i vsum0x46 = _mm_mul_epu32(vksum4567, vinput_zero_point0);
    const __m128i vsum0x57 = _mm_mul_epu32(vksum57, vinput_zero_point0);
    const __m128i vsum0x01 = _mm_unpacklo_epi32(vsum0x02, vsum0x13);
    const __m128i vsum0x23 = _mm_unpackhi_epi32(vsum0x02, vsum0x13);
    const __m128i vsum0x45 = _mm_unpacklo_epi32(vsum0x46, vsum0x57);
    const __m128i vsum0x67 = _mm_unpackhi_epi32(vsum0x46, vsum0x57);
    const __m128i vsum0x0123 = _mm_unpacklo_epi64(vsum0x01, vsum0x23);
    const __m128i vsum0x4567 = _mm_unpacklo_epi64(vsum0x45, vsum0x67);
    const __m128i vsum1x02 = _mm_mul_epu32(vksum0123, vinput_zero_point1);
    const __m128i vsum1x13 = _mm_mul_epu32(vksum13, vinput_zero_point1);
    const __m128i vsum1x46 = _mm_mul_epu32(vksum4567, vinput_zero_point1);
    const __m128i vsum1x57 = _mm_mul_epu32(vksum57, vinput_zero_point1);
    const __m128i vsum1x01 = _mm_unpacklo_epi32(vsum1x02, vsum1x13);
    const __m128i vsum1x23 = _mm_unpackhi_epi32(vsum1x02, vsum1x13);
    const __m128i vsum1x45 = _mm_unpacklo_epi32(vsum1x46, vsum1x57);
    const __m128i vsum1x67 = _mm_unpackhi_epi32(vsum1x46, vsum1x57);
    const __m128i vsum1x0123 = _mm_unpacklo_epi64(vsum1x01, vsum1x23);
    const __m128i vsum1x4567 = _mm_unpacklo_epi64(vsum1x45, vsum1x67);
    const __m128i vsum2x02 = _mm_mul_epu32(vksum0123, vinput_zero_point2);
    const __m128i vsum2x13 = _mm_mul_epu32(vksum13, vinput_zero_point2);
    const __m128i vsum2x46 = _mm_mul_epu32(vksum4567, vinput_zero_point2);
    const __m128i vsum2x57 = _mm_mul_epu32(vksum57, vinput_zero_point2);
    const __m128i vsum2x01 = _mm_unpacklo_epi32(vsum2x02, vsum2x13);
    const __m128i vsum2x23 = _mm_unpackhi_epi32(vsum2x02, vsum2x13);
    const __m128i vsum2x45 = _mm_unpacklo_epi32(vsum2x46, vsum2x57);
    const __m128i vsum2x67 = _mm_unpackhi_epi32(vsum2x46, vsum2x57);
    const __m128i vsum2x0123 = _mm_unpacklo_epi64(vsum2x01, vsum2x23);
    const __m128i vsum2x4567 = _mm_unpacklo_epi64(vsum2x45, vsum2x67);
    __m128i vacc0x01 = _mm_setzero_si128();
    __m128i vacc0x23 = _mm_setzero_si128();
    __m128i vacc0x45 = _mm_setzero_si128();
    __m128i vacc0x67 = _mm_setzero_si128();
    __m128i vacc1x01 = _mm_setzero_si128();
    __m128i vacc1x23 = _mm_setzero_si128();
    __m128i vacc1x45 = _mm_setzero_si128();
    __m128i vacc1x67 = _mm_setzero_si128();
    __m128i vacc2x01 = _mm_setzero_si128();
    __m128i vacc2x23 = _mm_setzero_si128();
    __m128i vacc2x45 = _mm_setzero_si128();
    __m128i vacc2x67 = _mm_setzero_si128();
    w = (const int32_t*) w + 8;
    // TODO: move kernel zero point after weights
    const void* kzp = w;
    w = (const float*)w + 8;

    size_t k = kc;
    while (k >= 32 * sizeof(int8_t)) {
      const __m128i va0x0 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a0));
      const __m128i va0x1 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a0 + 8));
      const __m128i va0x2 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a0 + 16));
      const __m128i va0x3 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a0 + 24));
      a0 += 32;
      const __m128i va1x0 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a1));
      const __m128i va1x1 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a1 + 8));
      const __m128i va1x2 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a1 + 16));
      const __m128i va1x3 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a1 + 24));
      a1 += 32;
      const __m128i va2x0 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a2));
      const __m128i va2x1 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a2 + 8));
      const __m128i va2x2 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a2 + 16));
      const __m128i va2x3 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a2 + 24));
      a2 += 32;

      const __m128i vbb01_lo = _mm_load_si128((const __m128i*) w);
      const __m128i vbb23_lo = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      const __m128i vbb45_lo = _mm_load_si128((const __m128i*) ((const int8_t*) w + 32));
      const __m128i vbb67_lo = _mm_load_si128((const __m128i*) ((const int8_t*) w + 48));

      __m128i vb01_p0 = _mm_and_si128(vbb01_lo, vmask);
      __m128i vb23_p0 = _mm_and_si128(vbb23_lo, vmask);
      __m128i vb45_p0 = _mm_and_si128(vbb45_lo, vmask);
      __m128i vb67_p0 = _mm_and_si128(vbb67_lo, vmask);

      __m128i vbs01_p1 = _mm_srli_epi32(vbb01_lo, 2);
      __m128i vbs23_p1 = _mm_srli_epi32(vbb23_lo, 2);
      __m128i vbs45_p1 = _mm_srli_epi32(vbb45_lo, 2);
      __m128i vbs67_p1 = _mm_srli_epi32(vbb67_lo, 2);
      __m128i vb01_p1 = _mm_and_si128(vbs01_p1, vmask);
      __m128i vb23_p1 = _mm_and_si128(vbs23_p1, vmask);
      __m128i vb45_p1 = _mm_and_si128(vbs45_p1, vmask);
      __m128i vb67_p1 = _mm_and_si128(vbs67_p1, vmask);

      __m128i vbs01_p2 = _mm_srli_epi32(vbb01_lo, 4);
      __m128i vbs23_p2 = _mm_srli_epi32(vbb23_lo, 4);
      __m128i vbs45_p2 = _mm_srli_epi32(vbb45_lo, 4);
      __m128i vbs67_p2 = _mm_srli_epi32(vbb67_lo, 4);
      __m128i vb01_p2 = _mm_and_si128(vbs01_p2, vmask);
      __m128i vb23_p2 = _mm_and_si128(vbs23_p2, vmask);
      __m128i vb45_p2 = _mm_and_si128(vbs45_p2, vmask);
      __m128i vb67_p2 = _mm_and_si128(vbs67_p2, vmask);

      __m128i vbs01_p3 = _mm_srli_epi32(vbb01_lo, 6);
      __m128i vbs23_p3 = _mm_srli_epi32(vbb23_lo, 6);
      __m128i vbs45_p3 = _mm_srli_epi32(vbb45_lo, 6);
      __m128i vbs67_p3 = _mm_srli_epi32(vbb67_lo, 6);
      __m128i vb01_p3 = _mm_and_si128(vbs01_p3, vmask);
      __m128i vb23_p3 = _mm_and_si128(vbs23_p3, vmask);
      __m128i vb45_p3 = _mm_and_si128(vbs45_p3, vmask);
      __m128i vb67_p3 = _mm_and_si128(vbs67_p3, vmask);

      vacc0x01 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x01, vb01_p0, va0x0);
      vacc0x23 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x23, vb23_p0, va0x0);
      vacc0x45 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x45, vb45_p0, va0x0);
      vacc0x67 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x67, vb67_p0, va0x0);
      vacc0x01 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x01, vb01_p1, va0x1);
      vacc0x23 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x23, vb23_p1, va0x1);
      vacc0x45 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x45, vb45_p1, va0x1);
      vacc0x67 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x67, vb67_p1, va0x1);
      vacc0x01 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x01, vb01_p2, va0x2);
      vacc0x23 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x23, vb23_p2, va0x2);
      vacc0x45 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x45, vb45_p2, va0x2);
      vacc0x67 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x67, vb67_p2, va0x2);
      vacc0x01 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x01, vb01_p3, va0x3);
      vacc0x23 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x23, vb23_p3, va0x3);
      vacc0x45 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x45, vb45_p3, va0x3);
      vacc0x67 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x67, vb67_p3, va0x3);
      vacc1x01 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x01, vb01_p0, va1x0);
      vacc1x23 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x23, vb23_p0, va1x0);
      vacc1x45 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x45, vb45_p0, va1x0);
      vacc1x67 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x67, vb67_p0, va1x0);
      vacc1x01 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x01, vb01_p1, va1x1);
      vacc1x23 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x23, vb23_p1, va1x1);
      vacc1x45 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x45, vb45_p1, va1x1);
      vacc1x67 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x67, vb67_p1, va1x1);
      vacc1x01 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x01, vb01_p2, va1x2);
      vacc1x23 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x23, vb23_p2, va1x2);
      vacc1x45 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x45, vb45_p2, va1x2);
      vacc1x67 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x67, vb67_p2, va1x2);
      vacc1x01 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x01, vb01_p3, va1x3);
      vacc1x23 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x23, vb23_p3, va1x3);
      vacc1x45 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x45, vb45_p3, va1x3);
      vacc1x67 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x67, vb67_p3, va1x3);
      vacc2x01 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x01, vb01_p0, va2x0);
      vacc2x23 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x23, vb23_p0, va2x0);
      vacc2x45 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x45, vb45_p0, va2x0);
      vacc2x67 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x67, vb67_p0, va2x0);
      vacc2x01 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x01, vb01_p1, va2x1);
      vacc2x23 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x23, vb23_p1, va2x1);
      vacc2x45 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x45, vb45_p1, va2x1);
      vacc2x67 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x67, vb67_p1, va2x1);
      vacc2x01 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x01, vb01_p2, va2x2);
      vacc2x23 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x23, vb23_p2, va2x2);
      vacc2x45 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x45, vb45_p2, va2x2);
      vacc2x67 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x67, vb67_p2, va2x2);
      vacc2x01 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x01, vb01_p3, va2x3);
      vacc2x23 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x23, vb23_p3, va2x3);
      vacc2x45 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x45, vb45_p3, va2x3);
      vacc2x67 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x67, vb67_p3, va2x3);

      w = (const int8_t*) w + 64;
      k -= 32 * sizeof(int8_t);
    }
    while (k >= 16 * sizeof(int8_t)) {
      const __m128i va0x01234567 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a0));
      const __m128i va0x89ABCDEF = _mm_set1_epi64x((int64_t) unaligned_load_u64(a0 + 8));
      a0 += 16;
      const __m128i va1x01234567 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a1));
      const __m128i va1x89ABCDEF = _mm_set1_epi64x((int64_t) unaligned_load_u64(a1 + 8));
      a1 += 16;
      const __m128i va2x01234567 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a2));
      const __m128i va2x89ABCDEF = _mm_set1_epi64x((int64_t) unaligned_load_u64(a2 + 8));
      a2 += 16;

      const __m128i vbb01_lo = _mm_load_si128((const __m128i*) w);
      const __m128i vbb23_lo = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      const __m128i vbb45_lo = _mm_load_si128((const __m128i*) ((const int8_t*) w + 32));
      const __m128i vbb67_lo = _mm_load_si128((const __m128i*) ((const int8_t*) w + 48));

      __m128i vb01_p0 = _mm_and_si128(vbb01_lo, vmask);
      __m128i vb23_p0 = _mm_and_si128(vbb23_lo, vmask);
      __m128i vb45_p0 = _mm_and_si128(vbb45_lo, vmask);
      __m128i vb67_p0 = _mm_and_si128(vbb67_lo, vmask);

      __m128i vbs01_p1 = _mm_srli_epi32(vbb01_lo, 2);
      __m128i vbs23_p1 = _mm_srli_epi32(vbb23_lo, 2);
      __m128i vbs45_p1 = _mm_srli_epi32(vbb45_lo, 2);
      __m128i vbs67_p1 = _mm_srli_epi32(vbb67_lo, 2);
      __m128i vb01_p1 = _mm_and_si128(vbs01_p1, vmask);
      __m128i vb23_p1 = _mm_and_si128(vbs23_p1, vmask);
      __m128i vb45_p1 = _mm_and_si128(vbs45_p1, vmask);
      __m128i vb67_p1 = _mm_and_si128(vbs67_p1, vmask);

      vacc0x01 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x01, vb01_p0, va0x01234567);
      vacc0x23 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x23, vb23_p0, va0x01234567);
      vacc0x45 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x45, vb45_p0, va0x01234567);
      vacc0x67 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x67, vb67_p0, va0x01234567);
      vacc0x01 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x01, vb01_p1, va0x89ABCDEF);
      vacc0x23 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x23, vb23_p1, va0x89ABCDEF);
      vacc0x45 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x45, vb45_p1, va0x89ABCDEF);
      vacc0x67 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x67, vb67_p1, va0x89ABCDEF);
      vacc1x01 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x01, vb01_p0, va1x01234567);
      vacc1x23 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x23, vb23_p0, va1x01234567);
      vacc1x45 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x45, vb45_p0, va1x01234567);
      vacc1x67 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x67, vb67_p0, va1x01234567);
      vacc1x01 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x01, vb01_p1, va1x89ABCDEF);
      vacc1x23 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x23, vb23_p1, va1x89ABCDEF);
      vacc1x45 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x45, vb45_p1, va1x89ABCDEF);
      vacc1x67 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x67, vb67_p1, va1x89ABCDEF);
      vacc2x01 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x01, vb01_p0, va2x01234567);
      vacc2x23 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x23, vb23_p0, va2x01234567);
      vacc2x45 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x45, vb45_p0, va2x01234567);
      vacc2x67 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x67, vb67_p0, va2x01234567);
      vacc2x01 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x01, vb01_p1, va2x89ABCDEF);
      vacc2x23 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x23, vb23_p1, va2x89ABCDEF);
      vacc2x45 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x45, vb45_p1, va2x89ABCDEF);
      vacc2x67 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x67, vb67_p1, va2x89ABCDEF);

      w = (const int8_t*) w + 64;
      k -= 16 * sizeof(int8_t);
      if (k != 0) {
        const __m128i va0x3 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a0));
        a0 += 8;
        const __m128i va1x3 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a1));
        a1 += 8;
        const __m128i va2x3 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a2));
        a2 += 8;

        __m128i vbs01_p2 = _mm_srli_epi32(vbb01_lo, 4);
        __m128i vbs23_p2 = _mm_srli_epi32(vbb23_lo, 4);
        __m128i vbs45_p2 = _mm_srli_epi32(vbb45_lo, 4);
        __m128i vbs67_p2 = _mm_srli_epi32(vbb67_lo, 4);
        __m128i vb01_p2 = _mm_and_si128(vbs01_p2, vmask);
        __m128i vb23_p2 = _mm_and_si128(vbs23_p2, vmask);
        __m128i vb45_p2 = _mm_and_si128(vbs45_p2, vmask);
        __m128i vb67_p2 = _mm_and_si128(vbs67_p2, vmask);

        vacc0x01 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x01, vb01_p2, va0x3);
        vacc0x23 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x23, vb23_p2, va0x3);
        vacc0x45 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x45, vb45_p2, va0x3);
        vacc0x67 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x67, vb67_p2, va0x3);
        vacc1x01 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x01, vb01_p2, va1x3);
        vacc1x23 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x23, vb23_p2, va1x3);
        vacc1x45 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x45, vb45_p2, va1x3);
        vacc1x67 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x67, vb67_p2, va1x3);
        vacc2x01 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x01, vb01_p2, va2x3);
        vacc2x23 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x23, vb23_p2, va2x3);
        vacc2x45 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x45, vb45_p2, va2x3);
        vacc2x67 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x67, vb67_p2, va2x3);
        k -= 8 * sizeof(int8_t);
      }
    }

    if (k != 0) {
      const __m128i va0x01234567 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a0));
      a0 += 8;
      const __m128i va1x01234567 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a1));
      a1 += 8;
      const __m128i va2x01234567 = _mm_set1_epi64x((int64_t) unaligned_load_u64(a2));
      a2 += 8;

      const __m128i vbb01_lo = _mm_load_si128((const __m128i*) w);
      const __m128i vbb23_lo = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      const __m128i vbb45_lo = _mm_load_si128((const __m128i*) ((const int8_t*) w + 32));
      const __m128i vbb67_lo = _mm_load_si128((const __m128i*) ((const int8_t*) w + 48));

      __m128i vb01_p0 = _mm_and_si128(vbb01_lo, vmask);
      __m128i vb23_p0 = _mm_and_si128(vbb23_lo, vmask);
      __m128i vb45_p0 = _mm_and_si128(vbb45_lo, vmask);
      __m128i vb67_p0 = _mm_and_si128(vbb67_lo, vmask);

      vacc0x01 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x01, vb01_p0, va0x01234567);
      vacc0x23 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x23, vb23_p0, va0x01234567);
      vacc0x45 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x45, vb45_p0, va0x01234567);
      vacc0x67 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc0x67, vb67_p0, va0x01234567);
      vacc1x01 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x01, vb01_p0, va1x01234567);
      vacc1x23 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x23, vb23_p0, va1x01234567);
      vacc1x45 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x45, vb45_p0, va1x01234567);
      vacc1x67 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc1x67, vb67_p0, va1x01234567);
      vacc2x01 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x01, vb01_p0, va2x01234567);
      vacc2x23 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x23, vb23_p0, va2x01234567);
      vacc2x45 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x45, vb45_p0, va2x01234567);
      vacc2x67 = _mm_dpbusd_epi32_madd_qd8_qc2w(vacc2x67, vb67_p0, va2x01234567);

      w = (const int8_t*) w + 64;
      k -= 8 * sizeof(int8_t);
    }
    assert(k == 0);

    // Add adjacent pairs
    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
    __m128i vacc0x4567 = _mm_hadd_epi32(vacc0x45, vacc0x67);
    vacc0x0123 = _mm_add_epi32(vacc0x0123, vsum0x0123);
    vacc0x4567 = _mm_add_epi32(vacc0x4567, vsum0x4567);
    __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);
    __m128i vacc1x4567 = _mm_hadd_epi32(vacc1x45, vacc1x67);
    vacc1x0123 = _mm_add_epi32(vacc1x0123, vsum1x0123);
    vacc1x4567 = _mm_add_epi32(vacc1x4567, vsum1x4567);
    __m128i vacc2x0123 = _mm_hadd_epi32(vacc2x01, vacc2x23);
    __m128i vacc2x4567 = _mm_hadd_epi32(vacc2x45, vacc2x67);
    vacc2x0123 = _mm_add_epi32(vacc2x0123, vsum2x0123);
    vacc2x4567 = _mm_add_epi32(vacc2x4567, vsum2x4567);

    __m128 vout0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vout0x4567 = _mm_cvtepi32_ps(vacc0x4567);
    __m128 vout1x0123 = _mm_cvtepi32_ps(vacc1x0123);
    __m128 vout1x4567 = _mm_cvtepi32_ps(vacc1x4567);
    __m128 vout2x0123 = _mm_cvtepi32_ps(vacc2x0123);
    __m128 vout2x4567 = _mm_cvtepi32_ps(vacc2x4567);

    const __m128 rh_zero_points_0123 = _mm_load_ps((const float*) kzp);
    const __m128 rh_zero_points_4567 = _mm_load_ps((const float*) kzp + 4);
    kzp = (const float*)kzp + 8;
    const __m128 biased_rh_zero_points_0123 = _mm_add_ps(rh_zero_points_0123, _mm_set1_ps(2.0f));
    const __m128 biased_rh_zero_points_4567 = _mm_add_ps(rh_zero_points_4567, _mm_set1_ps(2.0f));
    // Subtract out the scaled left-hand row sums.
    const __m128 lh_row_sum_0 = _mm_set1_ps(row_sum[0]);
    vout0x0123 = _mm_sub_ps(vout0x0123, _mm_mul_ps(biased_rh_zero_points_0123, lh_row_sum_0));
    vout0x4567 = _mm_sub_ps(vout0x4567, _mm_mul_ps(biased_rh_zero_points_4567, lh_row_sum_0));
    const __m128 lh_row_sum_1 = _mm_set1_ps(row_sum[1]);
    vout1x0123 = _mm_sub_ps(vout1x0123, _mm_mul_ps(biased_rh_zero_points_0123, lh_row_sum_1));
    vout1x4567 = _mm_sub_ps(vout1x4567, _mm_mul_ps(biased_rh_zero_points_4567, lh_row_sum_1));
    const __m128 lh_row_sum_2 = _mm_set1_ps(row_sum[2]);
    vout2x0123 = _mm_sub_ps(vout2x0123, _mm_mul_ps(biased_rh_zero_points_0123, lh_row_sum_2));
    vout2x4567 = _mm_sub_ps(vout2x4567, _mm_mul_ps(biased_rh_zero_points_4567, lh_row_sum_2));
    // Add the product of left/right-hand zero points and `kc`.
    const __m128 vscaled_lh_zero_point_0 = _mm_set1_ps((float)original_kc * quantization_params[0].zero_point);
    const __m128 vscaled_lh_zero_point_1 = _mm_set1_ps((float)original_kc * quantization_params[1].zero_point);
    const __m128 vscaled_lh_zero_point_2 = _mm_set1_ps((float)original_kc * quantization_params[2].zero_point);
    vout0x0123 = _mm_add_ps(_mm_mul_ps(rh_zero_points_0123, vscaled_lh_zero_point_0), vout0x0123);
    vout0x4567 = _mm_add_ps(_mm_mul_ps(rh_zero_points_4567, vscaled_lh_zero_point_0), vout0x4567);
    vout1x0123 = _mm_add_ps(_mm_mul_ps(rh_zero_points_0123, vscaled_lh_zero_point_1), vout1x0123);
    vout1x4567 = _mm_add_ps(_mm_mul_ps(rh_zero_points_4567, vscaled_lh_zero_point_1), vout1x4567);
    vout2x0123 = _mm_add_ps(_mm_mul_ps(rh_zero_points_0123, vscaled_lh_zero_point_2), vout2x0123);
    vout2x4567 = _mm_add_ps(_mm_mul_ps(rh_zero_points_4567, vscaled_lh_zero_point_2), vout2x4567);
    vout0x0123 = _mm_mul_ps(vout0x0123, _mm_set1_ps(quantization_params[0].inv_scale));
    vout0x4567 = _mm_mul_ps(vout0x4567, _mm_set1_ps(quantization_params[0].inv_scale));
    vout1x0123 = _mm_mul_ps(vout1x0123, _mm_set1_ps(quantization_params[1].inv_scale));
    vout1x4567 = _mm_mul_ps(vout1x4567, _mm_set1_ps(quantization_params[1].inv_scale));
    vout2x0123 = _mm_mul_ps(vout2x0123, _mm_set1_ps(quantization_params[2].inv_scale));
    vout2x4567 = _mm_mul_ps(vout2x4567, _mm_set1_ps(quantization_params[2].inv_scale));

    const __m128 vfilter_output_scale0123 = _mm_load_ps((const float*) w);
    const __m128 vfilter_output_scale4567 = _mm_load_ps((const float*) w + 4);
    const __m128 vbias0123 = _mm_load_ps((const float*) w + 8);
    const __m128 vbias4567 = _mm_load_ps((const float*) w + 12);
    w = (const float*) w + 16;

    vout0x0123 = _mm_add_ps(_mm_mul_ps(vout0x0123, vfilter_output_scale0123), vbias0123);
    vout0x4567 = _mm_add_ps(_mm_mul_ps(vout0x4567, vfilter_output_scale4567), vbias4567);
    vout1x0123 = _mm_add_ps(_mm_mul_ps(vout1x0123, vfilter_output_scale0123), vbias0123);
    vout1x4567 = _mm_add_ps(_mm_mul_ps(vout1x4567, vfilter_output_scale4567), vbias4567);
    vout2x0123 = _mm_add_ps(_mm_mul_ps(vout2x0123, vfilter_output_scale0123), vbias0123);
    vout2x4567 = _mm_add_ps(_mm_mul_ps(vout2x4567, vfilter_output_scale4567), vbias4567);

    vout0x0123 = _mm_max_ps(vout0x0123, voutput_min);
    vout0x4567 = _mm_max_ps(vout0x4567, voutput_min);
    vout1x0123 = _mm_max_ps(vout1x0123, voutput_min);
    vout1x4567 = _mm_max_ps(vout1x4567, voutput_min);
    vout2x0123 = _mm_max_ps(vout2x0123, voutput_min);
    vout2x4567 = _mm_max_ps(vout2x4567, voutput_min);

    vout0x0123 = _mm_min_ps(vout0x0123, voutput_max);
    vout0x4567 = _mm_min_ps(vout0x4567, voutput_max);
    vout1x0123 = _mm_min_ps(vout1x0123, voutput_max);
    vout1x4567 = _mm_min_ps(vout1x4567, voutput_max);
    vout2x0123 = _mm_min_ps(vout2x0123, voutput_max);
    vout2x4567 = _mm_min_ps(vout2x4567, voutput_max);

    if XNN_LIKELY(nc >= 8) {
      _mm_storeu_ps(c0, vout0x0123);
      _mm_storeu_ps(c0 + 4, vout0x4567);
      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      _mm_storeu_ps(c1, vout1x0123);
      _mm_storeu_ps(c1 + 4, vout1x4567);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm_storeu_ps(c2, vout2x0123);
      _mm_storeu_ps(c2 + 4, vout2x4567);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_ps(c0, vout0x0123);
        c0 += 4;
        vout0x0123 = vout0x4567;
        _mm_storeu_ps(c1, vout1x0123);
        c1 += 4;
        vout1x0123 = vout1x4567;
        _mm_storeu_ps(c2, vout2x0123);
        c2 += 4;
        vout2x0123 = vout2x4567;
      }
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
