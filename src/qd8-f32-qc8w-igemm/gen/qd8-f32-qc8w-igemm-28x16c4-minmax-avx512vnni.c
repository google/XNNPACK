// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/MRx16c4-avx512vnni.c.in
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


void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_28x16c4__avx512vnni(
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
  assert(mr <= 28);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    c5 = c4;
  }
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    c6 = c5;
  }
  float* c7 = (float*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 8) {
    c7 = c6;
  }
  float* c8 = (float*) ((uintptr_t) c7 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 8) {
    c8 = c7;
  }
  float* c9 = (float*) ((uintptr_t) c8 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 10) {
    c9 = c8;
  }
  float* c10 = (float*) ((uintptr_t) c9 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 10) {
    c10 = c9;
  }
  float* c11 = (float*) ((uintptr_t) c10 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 12) {
    c11 = c10;
  }
  float* c12 = (float*) ((uintptr_t) c11 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 12) {
    c12 = c11;
  }
  float* c13 = (float*) ((uintptr_t) c12 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 14) {
    c13 = c12;
  }
  float* c14 = (float*) ((uintptr_t) c13 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 14) {
    c14 = c13;
  }
  float* c15 = (float*) ((uintptr_t) c14 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 16) {
    c15 = c14;
  }
  float* c16 = (float*) ((uintptr_t) c15 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 16) {
    c16 = c15;
  }
  float* c17 = (float*) ((uintptr_t) c16 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 18) {
    c17 = c16;
  }
  float* c18 = (float*) ((uintptr_t) c17 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 18) {
    c18 = c17;
  }
  float* c19 = (float*) ((uintptr_t) c18 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 20) {
    c19 = c18;
  }
  float* c20 = (float*) ((uintptr_t) c19 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 20) {
    c20 = c19;
  }
  float* c21 = (float*) ((uintptr_t) c20 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 22) {
    c21 = c20;
  }
  float* c22 = (float*) ((uintptr_t) c21 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 22) {
    c22 = c21;
  }
  float* c23 = (float*) ((uintptr_t) c22 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 24) {
    c23 = c22;
  }
  float* c24 = (float*) ((uintptr_t) c23 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 24) {
    c24 = c23;
  }
  float* c25 = (float*) ((uintptr_t) c24 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 26) {
    c25 = c24;
  }
  float* c26 = (float*) ((uintptr_t) c25 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 26) {
    c26 = c25;
  }
  float* c27 = (float*) ((uintptr_t) c26 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 28) {
    c27 = c26;
  }

  const __m512i vsign_mask = _mm512_set1_epi8(0x80);
  XNN_FORCE_REALIZATION(vsign_mask);
  const __m512i vinput_zero_point = _mm512_set1_epi32((int) quantization_params->zero_point + 128);
  const __m512 vinput_inv_scale = _mm512_set1_ps(quantization_params->inv_scale);
  const __m512 voutput_min = _mm512_set1_ps(params->avx512vnni.min);
  const __m512 voutput_max = _mm512_set1_ps(params->avx512vnni.max);
  do {
    const __m512i vksum0123456789ABCDEF = _mm512_load_epi32(w);
    __m512i vacc0x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point);
    __m512i vacc1x0x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc1x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x1x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc2x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x2x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc3x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x3x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc4x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x4x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc5x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x5x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc6x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x6x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc7x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x7x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc8x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x8x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc9x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x9x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc10x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x10x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc11x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x11x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc12x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x12x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc13x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x13x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc14x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x14x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc15x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x15x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc16x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x16x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc17x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x17x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc18x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x18x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc19x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x19x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc20x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x20x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc21x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x21x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc22x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x22x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc23x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x23x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc24x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x24x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc25x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x25x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc26x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x26x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc27x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512i vacc1x27x0123456789ABCDEF = _mm512_setzero_epi32();
    w = (const int32_t*) w + 16;

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
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      } else {
        a3 = zero_data;
      }
      const int8_t* restrict a4 = a[4];
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const int8_t*) ((uintptr_t) a4 + a_offset);
      } else {
        a4 = zero_data;
      }
      const int8_t* restrict a5 = a[5];
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const int8_t*) ((uintptr_t) a5 + a_offset);
      } else {
        a5 = zero_data;
      }
      const int8_t* restrict a6 = a[6];
      if XNN_UNPREDICTABLE(a6 != zero) {
        a6 = (const int8_t*) ((uintptr_t) a6 + a_offset);
      } else {
        a6 = zero_data;
      }
      const int8_t* restrict a7 = a[7];
      if XNN_UNPREDICTABLE(a7 != zero) {
        a7 = (const int8_t*) ((uintptr_t) a7 + a_offset);
      } else {
        a7 = zero_data;
      }
      const int8_t* restrict a8 = a[8];
      if XNN_UNPREDICTABLE(a8 != zero) {
        a8 = (const int8_t*) ((uintptr_t) a8 + a_offset);
      } else {
        a8 = zero_data;
      }
      const int8_t* restrict a9 = a[9];
      if XNN_UNPREDICTABLE(a9 != zero) {
        a9 = (const int8_t*) ((uintptr_t) a9 + a_offset);
      } else {
        a9 = zero_data;
      }
      const int8_t* restrict a10 = a[10];
      if XNN_UNPREDICTABLE(a10 != zero) {
        a10 = (const int8_t*) ((uintptr_t) a10 + a_offset);
      } else {
        a10 = zero_data;
      }
      const int8_t* restrict a11 = a[11];
      if XNN_UNPREDICTABLE(a11 != zero) {
        a11 = (const int8_t*) ((uintptr_t) a11 + a_offset);
      } else {
        a11 = zero_data;
      }
      const int8_t* restrict a12 = a[12];
      if XNN_UNPREDICTABLE(a12 != zero) {
        a12 = (const int8_t*) ((uintptr_t) a12 + a_offset);
      } else {
        a12 = zero_data;
      }
      const int8_t* restrict a13 = a[13];
      if XNN_UNPREDICTABLE(a13 != zero) {
        a13 = (const int8_t*) ((uintptr_t) a13 + a_offset);
      } else {
        a13 = zero_data;
      }
      const int8_t* restrict a14 = a[14];
      if XNN_UNPREDICTABLE(a14 != zero) {
        a14 = (const int8_t*) ((uintptr_t) a14 + a_offset);
      } else {
        a14 = zero_data;
      }
      const int8_t* restrict a15 = a[15];
      if XNN_UNPREDICTABLE(a15 != zero) {
        a15 = (const int8_t*) ((uintptr_t) a15 + a_offset);
      } else {
        a15 = zero_data;
      }
      const int8_t* restrict a16 = a[16];
      if XNN_UNPREDICTABLE(a16 != zero) {
        a16 = (const int8_t*) ((uintptr_t) a16 + a_offset);
      } else {
        a16 = zero_data;
      }
      const int8_t* restrict a17 = a[17];
      if XNN_UNPREDICTABLE(a17 != zero) {
        a17 = (const int8_t*) ((uintptr_t) a17 + a_offset);
      } else {
        a17 = zero_data;
      }
      const int8_t* restrict a18 = a[18];
      if XNN_UNPREDICTABLE(a18 != zero) {
        a18 = (const int8_t*) ((uintptr_t) a18 + a_offset);
      } else {
        a18 = zero_data;
      }
      const int8_t* restrict a19 = a[19];
      if XNN_UNPREDICTABLE(a19 != zero) {
        a19 = (const int8_t*) ((uintptr_t) a19 + a_offset);
      } else {
        a19 = zero_data;
      }
      const int8_t* restrict a20 = a[20];
      if XNN_UNPREDICTABLE(a20 != zero) {
        a20 = (const int8_t*) ((uintptr_t) a20 + a_offset);
      } else {
        a20 = zero_data;
      }
      const int8_t* restrict a21 = a[21];
      if XNN_UNPREDICTABLE(a21 != zero) {
        a21 = (const int8_t*) ((uintptr_t) a21 + a_offset);
      } else {
        a21 = zero_data;
      }
      const int8_t* restrict a22 = a[22];
      if XNN_UNPREDICTABLE(a22 != zero) {
        a22 = (const int8_t*) ((uintptr_t) a22 + a_offset);
      } else {
        a22 = zero_data;
      }
      const int8_t* restrict a23 = a[23];
      if XNN_UNPREDICTABLE(a23 != zero) {
        a23 = (const int8_t*) ((uintptr_t) a23 + a_offset);
      } else {
        a23 = zero_data;
      }
      const int8_t* restrict a24 = a[24];
      if XNN_UNPREDICTABLE(a24 != zero) {
        a24 = (const int8_t*) ((uintptr_t) a24 + a_offset);
      } else {
        a24 = zero_data;
      }
      const int8_t* restrict a25 = a[25];
      if XNN_UNPREDICTABLE(a25 != zero) {
        a25 = (const int8_t*) ((uintptr_t) a25 + a_offset);
      } else {
        a25 = zero_data;
      }
      const int8_t* restrict a26 = a[26];
      if XNN_UNPREDICTABLE(a26 != zero) {
        a26 = (const int8_t*) ((uintptr_t) a26 + a_offset);
      } else {
        a26 = zero_data;
      }
      const int8_t* restrict a27 = a[27];
      if XNN_UNPREDICTABLE(a27 != zero) {
        a27 = (const int8_t*) ((uintptr_t) a27 + a_offset);
      } else {
        a27 = zero_data;
      }
      a += 28;

      size_t k = kc;
      while (k >= 8 * sizeof(int8_t)) {
        const __m512i va0x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a0)), vsign_mask);
        const __m512i va0x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a0 + 4)), vsign_mask);
        a0 += 8;
        const __m512i va1x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a1)), vsign_mask);
        const __m512i va1x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a1 + 4)), vsign_mask);
        a1 += 8;
        const __m512i va2x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a2)), vsign_mask);
        const __m512i va2x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a2 + 4)), vsign_mask);
        a2 += 8;
        const __m512i va3x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a3)), vsign_mask);
        const __m512i va3x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a3 + 4)), vsign_mask);
        a3 += 8;
        const __m512i va4x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a4)), vsign_mask);
        const __m512i va4x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a4 + 4)), vsign_mask);
        a4 += 8;
        const __m512i va5x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a5)), vsign_mask);
        const __m512i va5x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a5 + 4)), vsign_mask);
        a5 += 8;
        const __m512i va6x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a6)), vsign_mask);
        const __m512i va6x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a6 + 4)), vsign_mask);
        a6 += 8;
        const __m512i va7x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a7)), vsign_mask);
        const __m512i va7x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a7 + 4)), vsign_mask);
        a7 += 8;
        const __m512i va8x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a8)), vsign_mask);
        const __m512i va8x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a8 + 4)), vsign_mask);
        a8 += 8;
        const __m512i va9x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a9)), vsign_mask);
        const __m512i va9x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a9 + 4)), vsign_mask);
        a9 += 8;
        const __m512i va10x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a10)), vsign_mask);
        const __m512i va10x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a10 + 4)), vsign_mask);
        a10 += 8;
        const __m512i va11x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a11)), vsign_mask);
        const __m512i va11x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a11 + 4)), vsign_mask);
        a11 += 8;
        const __m512i va12x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a12)), vsign_mask);
        const __m512i va12x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a12 + 4)), vsign_mask);
        a12 += 8;
        const __m512i va13x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a13)), vsign_mask);
        const __m512i va13x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a13 + 4)), vsign_mask);
        a13 += 8;
        const __m512i va14x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a14)), vsign_mask);
        const __m512i va14x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a14 + 4)), vsign_mask);
        a14 += 8;
        const __m512i va15x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a15)), vsign_mask);
        const __m512i va15x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a15 + 4)), vsign_mask);
        a15 += 8;
        const __m512i va16x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a16)), vsign_mask);
        const __m512i va16x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a16 + 4)), vsign_mask);
        a16 += 8;
        const __m512i va17x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a17)), vsign_mask);
        const __m512i va17x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a17 + 4)), vsign_mask);
        a17 += 8;
        const __m512i va18x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a18)), vsign_mask);
        const __m512i va18x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a18 + 4)), vsign_mask);
        a18 += 8;
        const __m512i va19x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a19)), vsign_mask);
        const __m512i va19x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a19 + 4)), vsign_mask);
        a19 += 8;
        const __m512i va20x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a20)), vsign_mask);
        const __m512i va20x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a20 + 4)), vsign_mask);
        a20 += 8;
        const __m512i va21x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a21)), vsign_mask);
        const __m512i va21x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a21 + 4)), vsign_mask);
        a21 += 8;
        const __m512i va22x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a22)), vsign_mask);
        const __m512i va22x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a22 + 4)), vsign_mask);
        a22 += 8;
        const __m512i va23x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a23)), vsign_mask);
        const __m512i va23x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a23 + 4)), vsign_mask);
        a23 += 8;
        const __m512i va24x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a24)), vsign_mask);
        const __m512i va24x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a24 + 4)), vsign_mask);
        a24 += 8;
        const __m512i va25x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a25)), vsign_mask);
        const __m512i va25x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a25 + 4)), vsign_mask);
        a25 += 8;
        const __m512i va26x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a26)), vsign_mask);
        const __m512i va26x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a26 + 4)), vsign_mask);
        a26 += 8;
        const __m512i va27x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a27)), vsign_mask);
        const __m512i va27x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a27 + 4)), vsign_mask);
        a27 += 8;

        const __m512i vb0123456789ABCDEFx0123 = _mm512_load_si512(w);
        const __m512i vb0123456789ABCDEFx4567 = _mm512_load_si512((const int8_t*) w + 64);

        vacc0x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc0x0123456789ABCDEF, va0x0123, vb0123456789ABCDEFx0123);
        vacc1x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x0123456789ABCDEF, va1x0123, vb0123456789ABCDEFx0123);
        vacc2x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc2x0123456789ABCDEF, va2x0123, vb0123456789ABCDEFx0123);
        vacc3x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc3x0123456789ABCDEF, va3x0123, vb0123456789ABCDEFx0123);
        vacc4x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc4x0123456789ABCDEF, va4x0123, vb0123456789ABCDEFx0123);
        vacc5x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc5x0123456789ABCDEF, va5x0123, vb0123456789ABCDEFx0123);
        vacc6x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc6x0123456789ABCDEF, va6x0123, vb0123456789ABCDEFx0123);
        vacc7x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc7x0123456789ABCDEF, va7x0123, vb0123456789ABCDEFx0123);
        vacc8x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc8x0123456789ABCDEF, va8x0123, vb0123456789ABCDEFx0123);
        vacc9x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc9x0123456789ABCDEF, va9x0123, vb0123456789ABCDEFx0123);
        vacc10x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc10x0123456789ABCDEF, va10x0123, vb0123456789ABCDEFx0123);
        vacc11x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc11x0123456789ABCDEF, va11x0123, vb0123456789ABCDEFx0123);
        vacc12x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc12x0123456789ABCDEF, va12x0123, vb0123456789ABCDEFx0123);
        vacc13x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc13x0123456789ABCDEF, va13x0123, vb0123456789ABCDEFx0123);
        vacc14x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc14x0123456789ABCDEF, va14x0123, vb0123456789ABCDEFx0123);
        vacc15x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc15x0123456789ABCDEF, va15x0123, vb0123456789ABCDEFx0123);
        vacc16x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc16x0123456789ABCDEF, va16x0123, vb0123456789ABCDEFx0123);
        vacc17x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc17x0123456789ABCDEF, va17x0123, vb0123456789ABCDEFx0123);
        vacc18x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc18x0123456789ABCDEF, va18x0123, vb0123456789ABCDEFx0123);
        vacc19x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc19x0123456789ABCDEF, va19x0123, vb0123456789ABCDEFx0123);
        vacc20x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc20x0123456789ABCDEF, va20x0123, vb0123456789ABCDEFx0123);
        vacc21x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc21x0123456789ABCDEF, va21x0123, vb0123456789ABCDEFx0123);
        vacc22x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc22x0123456789ABCDEF, va22x0123, vb0123456789ABCDEFx0123);
        vacc23x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc23x0123456789ABCDEF, va23x0123, vb0123456789ABCDEFx0123);
        vacc24x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc24x0123456789ABCDEF, va24x0123, vb0123456789ABCDEFx0123);
        vacc25x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc25x0123456789ABCDEF, va25x0123, vb0123456789ABCDEFx0123);
        vacc26x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc26x0123456789ABCDEF, va26x0123, vb0123456789ABCDEFx0123);
        vacc27x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc27x0123456789ABCDEF, va27x0123, vb0123456789ABCDEFx0123);
        vacc1x0x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x0x0123456789ABCDEF, va0x4567, vb0123456789ABCDEFx4567);
        vacc1x1x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x1x0123456789ABCDEF, va1x4567, vb0123456789ABCDEFx4567);
        vacc1x2x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x2x0123456789ABCDEF, va2x4567, vb0123456789ABCDEFx4567);
        vacc1x3x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x3x0123456789ABCDEF, va3x4567, vb0123456789ABCDEFx4567);
        vacc1x4x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x4x0123456789ABCDEF, va4x4567, vb0123456789ABCDEFx4567);
        vacc1x5x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x5x0123456789ABCDEF, va5x4567, vb0123456789ABCDEFx4567);
        vacc1x6x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x6x0123456789ABCDEF, va6x4567, vb0123456789ABCDEFx4567);
        vacc1x7x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x7x0123456789ABCDEF, va7x4567, vb0123456789ABCDEFx4567);
        vacc1x8x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x8x0123456789ABCDEF, va8x4567, vb0123456789ABCDEFx4567);
        vacc1x9x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x9x0123456789ABCDEF, va9x4567, vb0123456789ABCDEFx4567);
        vacc1x10x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x10x0123456789ABCDEF, va10x4567, vb0123456789ABCDEFx4567);
        vacc1x11x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x11x0123456789ABCDEF, va11x4567, vb0123456789ABCDEFx4567);
        vacc1x12x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x12x0123456789ABCDEF, va12x4567, vb0123456789ABCDEFx4567);
        vacc1x13x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x13x0123456789ABCDEF, va13x4567, vb0123456789ABCDEFx4567);
        vacc1x14x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x14x0123456789ABCDEF, va14x4567, vb0123456789ABCDEFx4567);
        vacc1x15x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x15x0123456789ABCDEF, va15x4567, vb0123456789ABCDEFx4567);
        vacc1x16x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x16x0123456789ABCDEF, va16x4567, vb0123456789ABCDEFx4567);
        vacc1x17x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x17x0123456789ABCDEF, va17x4567, vb0123456789ABCDEFx4567);
        vacc1x18x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x18x0123456789ABCDEF, va18x4567, vb0123456789ABCDEFx4567);
        vacc1x19x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x19x0123456789ABCDEF, va19x4567, vb0123456789ABCDEFx4567);
        vacc1x20x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x20x0123456789ABCDEF, va20x4567, vb0123456789ABCDEFx4567);
        vacc1x21x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x21x0123456789ABCDEF, va21x4567, vb0123456789ABCDEFx4567);
        vacc1x22x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x22x0123456789ABCDEF, va22x4567, vb0123456789ABCDEFx4567);
        vacc1x23x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x23x0123456789ABCDEF, va23x4567, vb0123456789ABCDEFx4567);
        vacc1x24x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x24x0123456789ABCDEF, va24x4567, vb0123456789ABCDEFx4567);
        vacc1x25x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x25x0123456789ABCDEF, va25x4567, vb0123456789ABCDEFx4567);
        vacc1x26x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x26x0123456789ABCDEF, va26x4567, vb0123456789ABCDEFx4567);
        vacc1x27x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x27x0123456789ABCDEF, va27x4567, vb0123456789ABCDEFx4567);

        w = (const int8_t*) w + 128;
        k -= 8 * sizeof(int8_t);
      }

      if (k != 0) {
        const __m512i va0x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a0)), vsign_mask);
        a0 += 4;
        const __m512i va1x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a1)), vsign_mask);
        a1 += 4;
        const __m512i va2x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a2)), vsign_mask);
        a2 += 4;
        const __m512i va3x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a3)), vsign_mask);
        a3 += 4;
        const __m512i va4x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a4)), vsign_mask);
        a4 += 4;
        const __m512i va5x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a5)), vsign_mask);
        a5 += 4;
        const __m512i va6x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a6)), vsign_mask);
        a6 += 4;
        const __m512i va7x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a7)), vsign_mask);
        a7 += 4;
        const __m512i va8x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a8)), vsign_mask);
        a8 += 4;
        const __m512i va9x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a9)), vsign_mask);
        a9 += 4;
        const __m512i va10x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a10)), vsign_mask);
        a10 += 4;
        const __m512i va11x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a11)), vsign_mask);
        a11 += 4;
        const __m512i va12x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a12)), vsign_mask);
        a12 += 4;
        const __m512i va13x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a13)), vsign_mask);
        a13 += 4;
        const __m512i va14x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a14)), vsign_mask);
        a14 += 4;
        const __m512i va15x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a15)), vsign_mask);
        a15 += 4;
        const __m512i va16x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a16)), vsign_mask);
        a16 += 4;
        const __m512i va17x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a17)), vsign_mask);
        a17 += 4;
        const __m512i va18x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a18)), vsign_mask);
        a18 += 4;
        const __m512i va19x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a19)), vsign_mask);
        a19 += 4;
        const __m512i va20x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a20)), vsign_mask);
        a20 += 4;
        const __m512i va21x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a21)), vsign_mask);
        a21 += 4;
        const __m512i va22x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a22)), vsign_mask);
        a22 += 4;
        const __m512i va23x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a23)), vsign_mask);
        a23 += 4;
        const __m512i va24x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a24)), vsign_mask);
        a24 += 4;
        const __m512i va25x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a25)), vsign_mask);
        a25 += 4;
        const __m512i va26x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a26)), vsign_mask);
        a26 += 4;
        const __m512i va27x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a27)), vsign_mask);
        a27 += 4;

        const __m512i vb0123456789ABCDEF = _mm512_load_si512(w);

        vacc0x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc0x0123456789ABCDEF, va0x0123, vb0123456789ABCDEF);
        vacc1x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x0123456789ABCDEF, va1x0123, vb0123456789ABCDEF);
        vacc2x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc2x0123456789ABCDEF, va2x0123, vb0123456789ABCDEF);
        vacc3x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc3x0123456789ABCDEF, va3x0123, vb0123456789ABCDEF);
        vacc4x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc4x0123456789ABCDEF, va4x0123, vb0123456789ABCDEF);
        vacc5x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc5x0123456789ABCDEF, va5x0123, vb0123456789ABCDEF);
        vacc6x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc6x0123456789ABCDEF, va6x0123, vb0123456789ABCDEF);
        vacc7x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc7x0123456789ABCDEF, va7x0123, vb0123456789ABCDEF);
        vacc8x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc8x0123456789ABCDEF, va8x0123, vb0123456789ABCDEF);
        vacc9x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc9x0123456789ABCDEF, va9x0123, vb0123456789ABCDEF);
        vacc10x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc10x0123456789ABCDEF, va10x0123, vb0123456789ABCDEF);
        vacc11x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc11x0123456789ABCDEF, va11x0123, vb0123456789ABCDEF);
        vacc12x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc12x0123456789ABCDEF, va12x0123, vb0123456789ABCDEF);
        vacc13x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc13x0123456789ABCDEF, va13x0123, vb0123456789ABCDEF);
        vacc14x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc14x0123456789ABCDEF, va14x0123, vb0123456789ABCDEF);
        vacc15x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc15x0123456789ABCDEF, va15x0123, vb0123456789ABCDEF);
        vacc16x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc16x0123456789ABCDEF, va16x0123, vb0123456789ABCDEF);
        vacc17x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc17x0123456789ABCDEF, va17x0123, vb0123456789ABCDEF);
        vacc18x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc18x0123456789ABCDEF, va18x0123, vb0123456789ABCDEF);
        vacc19x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc19x0123456789ABCDEF, va19x0123, vb0123456789ABCDEF);
        vacc20x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc20x0123456789ABCDEF, va20x0123, vb0123456789ABCDEF);
        vacc21x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc21x0123456789ABCDEF, va21x0123, vb0123456789ABCDEF);
        vacc22x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc22x0123456789ABCDEF, va22x0123, vb0123456789ABCDEF);
        vacc23x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc23x0123456789ABCDEF, va23x0123, vb0123456789ABCDEF);
        vacc24x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc24x0123456789ABCDEF, va24x0123, vb0123456789ABCDEF);
        vacc25x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc25x0123456789ABCDEF, va25x0123, vb0123456789ABCDEF);
        vacc26x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc26x0123456789ABCDEF, va26x0123, vb0123456789ABCDEF);
        vacc27x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc27x0123456789ABCDEF, va27x0123, vb0123456789ABCDEF);

        w = (const int8_t*) w + 64;
        k -= 4 * sizeof(int8_t);
      }

      p -= 28 * sizeof(void*);
    } while (p != 0);

    vacc0x0123456789ABCDEF = _mm512_add_epi32(vacc0x0123456789ABCDEF, vacc1x0x0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_add_epi32(vacc1x0123456789ABCDEF, vacc1x1x0123456789ABCDEF);
    vacc2x0123456789ABCDEF = _mm512_add_epi32(vacc2x0123456789ABCDEF, vacc1x2x0123456789ABCDEF);
    vacc3x0123456789ABCDEF = _mm512_add_epi32(vacc3x0123456789ABCDEF, vacc1x3x0123456789ABCDEF);
    vacc4x0123456789ABCDEF = _mm512_add_epi32(vacc4x0123456789ABCDEF, vacc1x4x0123456789ABCDEF);
    vacc5x0123456789ABCDEF = _mm512_add_epi32(vacc5x0123456789ABCDEF, vacc1x5x0123456789ABCDEF);
    vacc6x0123456789ABCDEF = _mm512_add_epi32(vacc6x0123456789ABCDEF, vacc1x6x0123456789ABCDEF);
    vacc7x0123456789ABCDEF = _mm512_add_epi32(vacc7x0123456789ABCDEF, vacc1x7x0123456789ABCDEF);
    vacc8x0123456789ABCDEF = _mm512_add_epi32(vacc8x0123456789ABCDEF, vacc1x8x0123456789ABCDEF);
    vacc9x0123456789ABCDEF = _mm512_add_epi32(vacc9x0123456789ABCDEF, vacc1x9x0123456789ABCDEF);
    vacc10x0123456789ABCDEF = _mm512_add_epi32(vacc10x0123456789ABCDEF, vacc1x10x0123456789ABCDEF);
    vacc11x0123456789ABCDEF = _mm512_add_epi32(vacc11x0123456789ABCDEF, vacc1x11x0123456789ABCDEF);
    vacc12x0123456789ABCDEF = _mm512_add_epi32(vacc12x0123456789ABCDEF, vacc1x12x0123456789ABCDEF);
    vacc13x0123456789ABCDEF = _mm512_add_epi32(vacc13x0123456789ABCDEF, vacc1x13x0123456789ABCDEF);
    vacc14x0123456789ABCDEF = _mm512_add_epi32(vacc14x0123456789ABCDEF, vacc1x14x0123456789ABCDEF);
    vacc15x0123456789ABCDEF = _mm512_add_epi32(vacc15x0123456789ABCDEF, vacc1x15x0123456789ABCDEF);
    vacc16x0123456789ABCDEF = _mm512_add_epi32(vacc16x0123456789ABCDEF, vacc1x16x0123456789ABCDEF);
    vacc17x0123456789ABCDEF = _mm512_add_epi32(vacc17x0123456789ABCDEF, vacc1x17x0123456789ABCDEF);
    vacc18x0123456789ABCDEF = _mm512_add_epi32(vacc18x0123456789ABCDEF, vacc1x18x0123456789ABCDEF);
    vacc19x0123456789ABCDEF = _mm512_add_epi32(vacc19x0123456789ABCDEF, vacc1x19x0123456789ABCDEF);
    vacc20x0123456789ABCDEF = _mm512_add_epi32(vacc20x0123456789ABCDEF, vacc1x20x0123456789ABCDEF);
    vacc21x0123456789ABCDEF = _mm512_add_epi32(vacc21x0123456789ABCDEF, vacc1x21x0123456789ABCDEF);
    vacc22x0123456789ABCDEF = _mm512_add_epi32(vacc22x0123456789ABCDEF, vacc1x22x0123456789ABCDEF);
    vacc23x0123456789ABCDEF = _mm512_add_epi32(vacc23x0123456789ABCDEF, vacc1x23x0123456789ABCDEF);
    vacc24x0123456789ABCDEF = _mm512_add_epi32(vacc24x0123456789ABCDEF, vacc1x24x0123456789ABCDEF);
    vacc25x0123456789ABCDEF = _mm512_add_epi32(vacc25x0123456789ABCDEF, vacc1x25x0123456789ABCDEF);
    vacc26x0123456789ABCDEF = _mm512_add_epi32(vacc26x0123456789ABCDEF, vacc1x26x0123456789ABCDEF);
    vacc27x0123456789ABCDEF = _mm512_add_epi32(vacc27x0123456789ABCDEF, vacc1x27x0123456789ABCDEF);

    __m512 vscaled0x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0x0123456789ABCDEF);
    __m512 vscaled1x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc1x0123456789ABCDEF);
    __m512 vscaled2x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc2x0123456789ABCDEF);
    __m512 vscaled3x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc3x0123456789ABCDEF);
    __m512 vscaled4x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc4x0123456789ABCDEF);
    __m512 vscaled5x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc5x0123456789ABCDEF);
    __m512 vscaled6x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc6x0123456789ABCDEF);
    __m512 vscaled7x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc7x0123456789ABCDEF);
    __m512 vscaled8x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc8x0123456789ABCDEF);
    __m512 vscaled9x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc9x0123456789ABCDEF);
    __m512 vscaled10x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc10x0123456789ABCDEF);
    __m512 vscaled11x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc11x0123456789ABCDEF);
    __m512 vscaled12x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc12x0123456789ABCDEF);
    __m512 vscaled13x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc13x0123456789ABCDEF);
    __m512 vscaled14x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc14x0123456789ABCDEF);
    __m512 vscaled15x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc15x0123456789ABCDEF);
    __m512 vscaled16x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc16x0123456789ABCDEF);
    __m512 vscaled17x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc17x0123456789ABCDEF);
    __m512 vscaled18x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc18x0123456789ABCDEF);
    __m512 vscaled19x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc19x0123456789ABCDEF);
    __m512 vscaled20x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc20x0123456789ABCDEF);
    __m512 vscaled21x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc21x0123456789ABCDEF);
    __m512 vscaled22x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc22x0123456789ABCDEF);
    __m512 vscaled23x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc23x0123456789ABCDEF);
    __m512 vscaled24x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc24x0123456789ABCDEF);
    __m512 vscaled25x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc25x0123456789ABCDEF);
    __m512 vscaled26x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc26x0123456789ABCDEF);
    __m512 vscaled27x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc27x0123456789ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, vinput_inv_scale);
    vscaled1x0123456789ABCDEF = _mm512_mul_ps(vscaled1x0123456789ABCDEF, vinput_inv_scale);
    vscaled2x0123456789ABCDEF = _mm512_mul_ps(vscaled2x0123456789ABCDEF, vinput_inv_scale);
    vscaled3x0123456789ABCDEF = _mm512_mul_ps(vscaled3x0123456789ABCDEF, vinput_inv_scale);
    vscaled4x0123456789ABCDEF = _mm512_mul_ps(vscaled4x0123456789ABCDEF, vinput_inv_scale);
    vscaled5x0123456789ABCDEF = _mm512_mul_ps(vscaled5x0123456789ABCDEF, vinput_inv_scale);
    vscaled6x0123456789ABCDEF = _mm512_mul_ps(vscaled6x0123456789ABCDEF, vinput_inv_scale);
    vscaled7x0123456789ABCDEF = _mm512_mul_ps(vscaled7x0123456789ABCDEF, vinput_inv_scale);
    vscaled8x0123456789ABCDEF = _mm512_mul_ps(vscaled8x0123456789ABCDEF, vinput_inv_scale);
    vscaled9x0123456789ABCDEF = _mm512_mul_ps(vscaled9x0123456789ABCDEF, vinput_inv_scale);
    vscaled10x0123456789ABCDEF = _mm512_mul_ps(vscaled10x0123456789ABCDEF, vinput_inv_scale);
    vscaled11x0123456789ABCDEF = _mm512_mul_ps(vscaled11x0123456789ABCDEF, vinput_inv_scale);
    vscaled12x0123456789ABCDEF = _mm512_mul_ps(vscaled12x0123456789ABCDEF, vinput_inv_scale);
    vscaled13x0123456789ABCDEF = _mm512_mul_ps(vscaled13x0123456789ABCDEF, vinput_inv_scale);
    vscaled14x0123456789ABCDEF = _mm512_mul_ps(vscaled14x0123456789ABCDEF, vinput_inv_scale);
    vscaled15x0123456789ABCDEF = _mm512_mul_ps(vscaled15x0123456789ABCDEF, vinput_inv_scale);
    vscaled16x0123456789ABCDEF = _mm512_mul_ps(vscaled16x0123456789ABCDEF, vinput_inv_scale);
    vscaled17x0123456789ABCDEF = _mm512_mul_ps(vscaled17x0123456789ABCDEF, vinput_inv_scale);
    vscaled18x0123456789ABCDEF = _mm512_mul_ps(vscaled18x0123456789ABCDEF, vinput_inv_scale);
    vscaled19x0123456789ABCDEF = _mm512_mul_ps(vscaled19x0123456789ABCDEF, vinput_inv_scale);
    vscaled20x0123456789ABCDEF = _mm512_mul_ps(vscaled20x0123456789ABCDEF, vinput_inv_scale);
    vscaled21x0123456789ABCDEF = _mm512_mul_ps(vscaled21x0123456789ABCDEF, vinput_inv_scale);
    vscaled22x0123456789ABCDEF = _mm512_mul_ps(vscaled22x0123456789ABCDEF, vinput_inv_scale);
    vscaled23x0123456789ABCDEF = _mm512_mul_ps(vscaled23x0123456789ABCDEF, vinput_inv_scale);
    vscaled24x0123456789ABCDEF = _mm512_mul_ps(vscaled24x0123456789ABCDEF, vinput_inv_scale);
    vscaled25x0123456789ABCDEF = _mm512_mul_ps(vscaled25x0123456789ABCDEF, vinput_inv_scale);
    vscaled26x0123456789ABCDEF = _mm512_mul_ps(vscaled26x0123456789ABCDEF, vinput_inv_scale);
    vscaled27x0123456789ABCDEF = _mm512_mul_ps(vscaled27x0123456789ABCDEF, vinput_inv_scale);

    const __m512 vfilter_output_scale0123456789ABCDEF = _mm512_load_ps((const float*) w);
    const __m512 vbias0123456789ABCDEF = _mm512_load_ps((const float*) w + 16);
    w = (const float*) w + 32;

    vscaled0x0123456789ABCDEF = _mm512_fmadd_ps(vscaled0x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled1x0123456789ABCDEF = _mm512_fmadd_ps(vscaled1x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled2x0123456789ABCDEF = _mm512_fmadd_ps(vscaled2x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled3x0123456789ABCDEF = _mm512_fmadd_ps(vscaled3x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled4x0123456789ABCDEF = _mm512_fmadd_ps(vscaled4x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled5x0123456789ABCDEF = _mm512_fmadd_ps(vscaled5x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled6x0123456789ABCDEF = _mm512_fmadd_ps(vscaled6x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled7x0123456789ABCDEF = _mm512_fmadd_ps(vscaled7x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled8x0123456789ABCDEF = _mm512_fmadd_ps(vscaled8x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled9x0123456789ABCDEF = _mm512_fmadd_ps(vscaled9x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled10x0123456789ABCDEF = _mm512_fmadd_ps(vscaled10x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled11x0123456789ABCDEF = _mm512_fmadd_ps(vscaled11x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled12x0123456789ABCDEF = _mm512_fmadd_ps(vscaled12x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled13x0123456789ABCDEF = _mm512_fmadd_ps(vscaled13x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled14x0123456789ABCDEF = _mm512_fmadd_ps(vscaled14x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled15x0123456789ABCDEF = _mm512_fmadd_ps(vscaled15x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled16x0123456789ABCDEF = _mm512_fmadd_ps(vscaled16x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled17x0123456789ABCDEF = _mm512_fmadd_ps(vscaled17x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled18x0123456789ABCDEF = _mm512_fmadd_ps(vscaled18x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled19x0123456789ABCDEF = _mm512_fmadd_ps(vscaled19x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled20x0123456789ABCDEF = _mm512_fmadd_ps(vscaled20x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled21x0123456789ABCDEF = _mm512_fmadd_ps(vscaled21x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled22x0123456789ABCDEF = _mm512_fmadd_ps(vscaled22x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled23x0123456789ABCDEF = _mm512_fmadd_ps(vscaled23x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled24x0123456789ABCDEF = _mm512_fmadd_ps(vscaled24x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled25x0123456789ABCDEF = _mm512_fmadd_ps(vscaled25x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled26x0123456789ABCDEF = _mm512_fmadd_ps(vscaled26x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled27x0123456789ABCDEF = _mm512_fmadd_ps(vscaled27x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);

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
    vscaled12x0123456789ABCDEF = _mm512_max_ps(vscaled12x0123456789ABCDEF, voutput_min);
    vscaled13x0123456789ABCDEF = _mm512_max_ps(vscaled13x0123456789ABCDEF, voutput_min);
    vscaled14x0123456789ABCDEF = _mm512_max_ps(vscaled14x0123456789ABCDEF, voutput_min);
    vscaled15x0123456789ABCDEF = _mm512_max_ps(vscaled15x0123456789ABCDEF, voutput_min);
    vscaled16x0123456789ABCDEF = _mm512_max_ps(vscaled16x0123456789ABCDEF, voutput_min);
    vscaled17x0123456789ABCDEF = _mm512_max_ps(vscaled17x0123456789ABCDEF, voutput_min);
    vscaled18x0123456789ABCDEF = _mm512_max_ps(vscaled18x0123456789ABCDEF, voutput_min);
    vscaled19x0123456789ABCDEF = _mm512_max_ps(vscaled19x0123456789ABCDEF, voutput_min);
    vscaled20x0123456789ABCDEF = _mm512_max_ps(vscaled20x0123456789ABCDEF, voutput_min);
    vscaled21x0123456789ABCDEF = _mm512_max_ps(vscaled21x0123456789ABCDEF, voutput_min);
    vscaled22x0123456789ABCDEF = _mm512_max_ps(vscaled22x0123456789ABCDEF, voutput_min);
    vscaled23x0123456789ABCDEF = _mm512_max_ps(vscaled23x0123456789ABCDEF, voutput_min);
    vscaled24x0123456789ABCDEF = _mm512_max_ps(vscaled24x0123456789ABCDEF, voutput_min);
    vscaled25x0123456789ABCDEF = _mm512_max_ps(vscaled25x0123456789ABCDEF, voutput_min);
    vscaled26x0123456789ABCDEF = _mm512_max_ps(vscaled26x0123456789ABCDEF, voutput_min);
    vscaled27x0123456789ABCDEF = _mm512_max_ps(vscaled27x0123456789ABCDEF, voutput_min);

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
    vscaled12x0123456789ABCDEF = _mm512_min_ps(vscaled12x0123456789ABCDEF, voutput_max);
    vscaled13x0123456789ABCDEF = _mm512_min_ps(vscaled13x0123456789ABCDEF, voutput_max);
    vscaled14x0123456789ABCDEF = _mm512_min_ps(vscaled14x0123456789ABCDEF, voutput_max);
    vscaled15x0123456789ABCDEF = _mm512_min_ps(vscaled15x0123456789ABCDEF, voutput_max);
    vscaled16x0123456789ABCDEF = _mm512_min_ps(vscaled16x0123456789ABCDEF, voutput_max);
    vscaled17x0123456789ABCDEF = _mm512_min_ps(vscaled17x0123456789ABCDEF, voutput_max);
    vscaled18x0123456789ABCDEF = _mm512_min_ps(vscaled18x0123456789ABCDEF, voutput_max);
    vscaled19x0123456789ABCDEF = _mm512_min_ps(vscaled19x0123456789ABCDEF, voutput_max);
    vscaled20x0123456789ABCDEF = _mm512_min_ps(vscaled20x0123456789ABCDEF, voutput_max);
    vscaled21x0123456789ABCDEF = _mm512_min_ps(vscaled21x0123456789ABCDEF, voutput_max);
    vscaled22x0123456789ABCDEF = _mm512_min_ps(vscaled22x0123456789ABCDEF, voutput_max);
    vscaled23x0123456789ABCDEF = _mm512_min_ps(vscaled23x0123456789ABCDEF, voutput_max);
    vscaled24x0123456789ABCDEF = _mm512_min_ps(vscaled24x0123456789ABCDEF, voutput_max);
    vscaled25x0123456789ABCDEF = _mm512_min_ps(vscaled25x0123456789ABCDEF, voutput_max);
    vscaled26x0123456789ABCDEF = _mm512_min_ps(vscaled26x0123456789ABCDEF, voutput_max);
    vscaled27x0123456789ABCDEF = _mm512_min_ps(vscaled27x0123456789ABCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c27, vscaled27x0123456789ABCDEF);
      c27 = (float*) ((uintptr_t) c27 + cn_stride);
      _mm512_storeu_ps(c26, vscaled26x0123456789ABCDEF);
      c26 = (float*) ((uintptr_t) c26 + cn_stride);
      _mm512_storeu_ps(c25, vscaled25x0123456789ABCDEF);
      c25 = (float*) ((uintptr_t) c25 + cn_stride);
      _mm512_storeu_ps(c24, vscaled24x0123456789ABCDEF);
      c24 = (float*) ((uintptr_t) c24 + cn_stride);
      _mm512_storeu_ps(c23, vscaled23x0123456789ABCDEF);
      c23 = (float*) ((uintptr_t) c23 + cn_stride);
      _mm512_storeu_ps(c22, vscaled22x0123456789ABCDEF);
      c22 = (float*) ((uintptr_t) c22 + cn_stride);
      _mm512_storeu_ps(c21, vscaled21x0123456789ABCDEF);
      c21 = (float*) ((uintptr_t) c21 + cn_stride);
      _mm512_storeu_ps(c20, vscaled20x0123456789ABCDEF);
      c20 = (float*) ((uintptr_t) c20 + cn_stride);
      _mm512_storeu_ps(c19, vscaled19x0123456789ABCDEF);
      c19 = (float*) ((uintptr_t) c19 + cn_stride);
      _mm512_storeu_ps(c18, vscaled18x0123456789ABCDEF);
      c18 = (float*) ((uintptr_t) c18 + cn_stride);
      _mm512_storeu_ps(c17, vscaled17x0123456789ABCDEF);
      c17 = (float*) ((uintptr_t) c17 + cn_stride);
      _mm512_storeu_ps(c16, vscaled16x0123456789ABCDEF);
      c16 = (float*) ((uintptr_t) c16 + cn_stride);
      _mm512_storeu_ps(c15, vscaled15x0123456789ABCDEF);
      c15 = (float*) ((uintptr_t) c15 + cn_stride);
      _mm512_storeu_ps(c14, vscaled14x0123456789ABCDEF);
      c14 = (float*) ((uintptr_t) c14 + cn_stride);
      _mm512_storeu_ps(c13, vscaled13x0123456789ABCDEF);
      c13 = (float*) ((uintptr_t) c13 + cn_stride);
      _mm512_storeu_ps(c12, vscaled12x0123456789ABCDEF);
      c12 = (float*) ((uintptr_t) c12 + cn_stride);
      _mm512_storeu_ps(c11, vscaled11x0123456789ABCDEF);
      c11 = (float*) ((uintptr_t) c11 + cn_stride);
      _mm512_storeu_ps(c10, vscaled10x0123456789ABCDEF);
      c10 = (float*) ((uintptr_t) c10 + cn_stride);
      _mm512_storeu_ps(c9, vscaled9x0123456789ABCDEF);
      c9 = (float*) ((uintptr_t) c9 + cn_stride);
      _mm512_storeu_ps(c8, vscaled8x0123456789ABCDEF);
      c8 = (float*) ((uintptr_t) c8 + cn_stride);
      _mm512_storeu_ps(c7, vscaled7x0123456789ABCDEF);
      c7 = (float*) ((uintptr_t) c7 + cn_stride);
      _mm512_storeu_ps(c6, vscaled6x0123456789ABCDEF);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      _mm512_storeu_ps(c5, vscaled5x0123456789ABCDEF);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      _mm512_storeu_ps(c4, vscaled4x0123456789ABCDEF);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm512_storeu_ps(c3, vscaled3x0123456789ABCDEF);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm512_storeu_ps(c2, vscaled2x0123456789ABCDEF);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm512_storeu_ps(c1, vscaled1x0123456789ABCDEF);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm512_storeu_ps(c0, vscaled0x0123456789ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((UINT32_C(1) << nc) - 1);
      _mm512_mask_storeu_ps(c27, vmask, vscaled27x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c26, vmask, vscaled26x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c25, vmask, vscaled25x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c24, vmask, vscaled24x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c23, vmask, vscaled23x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c22, vmask, vscaled22x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c21, vmask, vscaled21x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c20, vmask, vscaled20x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c19, vmask, vscaled19x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c18, vmask, vscaled18x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c17, vmask, vscaled17x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c16, vmask, vscaled16x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c15, vmask, vscaled15x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c14, vmask, vscaled14x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c13, vmask, vscaled13x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c12, vmask, vscaled12x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c11, vmask, vscaled11x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c10, vmask, vscaled10x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c9, vmask, vscaled9x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c8, vmask, vscaled8x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c7, vmask, vscaled7x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c6, vmask, vscaled6x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c5, vmask, vscaled5x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c4, vmask, vscaled4x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c3, vmask, vscaled3x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c2, vmask, vscaled2x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c1, vmask, vscaled1x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c0, vmask, vscaled0x0123456789ABCDEF);
      nc = 0;
    }
  } while (nc != 0);
}
