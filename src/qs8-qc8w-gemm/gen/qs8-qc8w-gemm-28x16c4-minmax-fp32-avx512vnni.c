// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx16c4-avx512vnni.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/gemm.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/unaligned.h"


void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_28x16c4__avx512vnni(
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
  assert(mr <= 28);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
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
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const int8_t* a4 = (const int8_t*) ((uintptr_t) a3 + a_stride);
  int8_t* c4 = (int8_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const int8_t* a5 = (const int8_t*) ((uintptr_t) a4 + a_stride);
  int8_t* c5 = (int8_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const int8_t* a6 = (const int8_t*) ((uintptr_t) a5 + a_stride);
  int8_t* c6 = (int8_t*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }
  const int8_t* a7 = (const int8_t*) ((uintptr_t) a6 + a_stride);
  int8_t* c7 = (int8_t*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 8) {
    a7 = a6;
    c7 = c6;
  }
  const int8_t* a8 = (const int8_t*) ((uintptr_t) a7 + a_stride);
  int8_t* c8 = (int8_t*) ((uintptr_t) c7 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 8) {
    a8 = a7;
    c8 = c7;
  }
  const int8_t* a9 = (const int8_t*) ((uintptr_t) a8 + a_stride);
  int8_t* c9 = (int8_t*) ((uintptr_t) c8 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 10) {
    a9 = a8;
    c9 = c8;
  }
  const int8_t* a10 = (const int8_t*) ((uintptr_t) a9 + a_stride);
  int8_t* c10 = (int8_t*) ((uintptr_t) c9 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 10) {
    a10 = a9;
    c10 = c9;
  }
  const int8_t* a11 = (const int8_t*) ((uintptr_t) a10 + a_stride);
  int8_t* c11 = (int8_t*) ((uintptr_t) c10 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 12) {
    a11 = a10;
    c11 = c10;
  }
  const int8_t* a12 = (const int8_t*) ((uintptr_t) a11 + a_stride);
  int8_t* c12 = (int8_t*) ((uintptr_t) c11 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 12) {
    a12 = a11;
    c12 = c11;
  }
  const int8_t* a13 = (const int8_t*) ((uintptr_t) a12 + a_stride);
  int8_t* c13 = (int8_t*) ((uintptr_t) c12 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 14) {
    a13 = a12;
    c13 = c12;
  }
  const int8_t* a14 = (const int8_t*) ((uintptr_t) a13 + a_stride);
  int8_t* c14 = (int8_t*) ((uintptr_t) c13 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 14) {
    a14 = a13;
    c14 = c13;
  }
  const int8_t* a15 = (const int8_t*) ((uintptr_t) a14 + a_stride);
  int8_t* c15 = (int8_t*) ((uintptr_t) c14 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 16) {
    a15 = a14;
    c15 = c14;
  }
  const int8_t* a16 = (const int8_t*) ((uintptr_t) a15 + a_stride);
  int8_t* c16 = (int8_t*) ((uintptr_t) c15 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 16) {
    a16 = a15;
    c16 = c15;
  }
  const int8_t* a17 = (const int8_t*) ((uintptr_t) a16 + a_stride);
  int8_t* c17 = (int8_t*) ((uintptr_t) c16 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 18) {
    a17 = a16;
    c17 = c16;
  }
  const int8_t* a18 = (const int8_t*) ((uintptr_t) a17 + a_stride);
  int8_t* c18 = (int8_t*) ((uintptr_t) c17 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 18) {
    a18 = a17;
    c18 = c17;
  }
  const int8_t* a19 = (const int8_t*) ((uintptr_t) a18 + a_stride);
  int8_t* c19 = (int8_t*) ((uintptr_t) c18 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 20) {
    a19 = a18;
    c19 = c18;
  }
  const int8_t* a20 = (const int8_t*) ((uintptr_t) a19 + a_stride);
  int8_t* c20 = (int8_t*) ((uintptr_t) c19 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 20) {
    a20 = a19;
    c20 = c19;
  }
  const int8_t* a21 = (const int8_t*) ((uintptr_t) a20 + a_stride);
  int8_t* c21 = (int8_t*) ((uintptr_t) c20 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 22) {
    a21 = a20;
    c21 = c20;
  }
  const int8_t* a22 = (const int8_t*) ((uintptr_t) a21 + a_stride);
  int8_t* c22 = (int8_t*) ((uintptr_t) c21 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 22) {
    a22 = a21;
    c22 = c21;
  }
  const int8_t* a23 = (const int8_t*) ((uintptr_t) a22 + a_stride);
  int8_t* c23 = (int8_t*) ((uintptr_t) c22 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 24) {
    a23 = a22;
    c23 = c22;
  }
  const int8_t* a24 = (const int8_t*) ((uintptr_t) a23 + a_stride);
  int8_t* c24 = (int8_t*) ((uintptr_t) c23 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 24) {
    a24 = a23;
    c24 = c23;
  }
  const int8_t* a25 = (const int8_t*) ((uintptr_t) a24 + a_stride);
  int8_t* c25 = (int8_t*) ((uintptr_t) c24 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 26) {
    a25 = a24;
    c25 = c24;
  }
  const int8_t* a26 = (const int8_t*) ((uintptr_t) a25 + a_stride);
  int8_t* c26 = (int8_t*) ((uintptr_t) c25 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 26) {
    a26 = a25;
    c26 = c25;
  }
  const int8_t* a27 = (const int8_t*) ((uintptr_t) a26 + a_stride);
  int8_t* c27 = (int8_t*) ((uintptr_t) c26 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 28) {
    a27 = a26;
    c27 = c26;
  }

  const __m512i vsign_mask =_mm512_set1_epi8(params->fp32_avx512vnni.sign_mask);  // 0x80
  const __m512 voutput_max_less_zero_point = _mm512_set1_ps(params->fp32_avx512vnni.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_set1_epi32(params->fp32_avx512vnni.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx512vnni.output_min);
  do {
    __m512i vacc0x0123456789ABCDEF = _mm512_load_epi32(w);
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

    const __m512 vscale012345678ABCDEF = _mm512_load_ps(w);
    w = (const float*) w + 16;
    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled1x0123456789ABCDEF = _mm512_mul_ps(vscaled1x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled2x0123456789ABCDEF = _mm512_mul_ps(vscaled2x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled3x0123456789ABCDEF = _mm512_mul_ps(vscaled3x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled4x0123456789ABCDEF = _mm512_mul_ps(vscaled4x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled5x0123456789ABCDEF = _mm512_mul_ps(vscaled5x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled6x0123456789ABCDEF = _mm512_mul_ps(vscaled6x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled7x0123456789ABCDEF = _mm512_mul_ps(vscaled7x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled8x0123456789ABCDEF = _mm512_mul_ps(vscaled8x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled9x0123456789ABCDEF = _mm512_mul_ps(vscaled9x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled10x0123456789ABCDEF = _mm512_mul_ps(vscaled10x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled11x0123456789ABCDEF = _mm512_mul_ps(vscaled11x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled12x0123456789ABCDEF = _mm512_mul_ps(vscaled12x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled13x0123456789ABCDEF = _mm512_mul_ps(vscaled13x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled14x0123456789ABCDEF = _mm512_mul_ps(vscaled14x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled15x0123456789ABCDEF = _mm512_mul_ps(vscaled15x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled16x0123456789ABCDEF = _mm512_mul_ps(vscaled16x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled17x0123456789ABCDEF = _mm512_mul_ps(vscaled17x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled18x0123456789ABCDEF = _mm512_mul_ps(vscaled18x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled19x0123456789ABCDEF = _mm512_mul_ps(vscaled19x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled20x0123456789ABCDEF = _mm512_mul_ps(vscaled20x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled21x0123456789ABCDEF = _mm512_mul_ps(vscaled21x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled22x0123456789ABCDEF = _mm512_mul_ps(vscaled22x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled23x0123456789ABCDEF = _mm512_mul_ps(vscaled23x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled24x0123456789ABCDEF = _mm512_mul_ps(vscaled24x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled25x0123456789ABCDEF = _mm512_mul_ps(vscaled25x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled26x0123456789ABCDEF = _mm512_mul_ps(vscaled26x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled27x0123456789ABCDEF = _mm512_mul_ps(vscaled27x0123456789ABCDEF, vscale012345678ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled1x0123456789ABCDEF = _mm512_min_ps(vscaled1x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled2x0123456789ABCDEF = _mm512_min_ps(vscaled2x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled3x0123456789ABCDEF = _mm512_min_ps(vscaled3x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled4x0123456789ABCDEF = _mm512_min_ps(vscaled4x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled5x0123456789ABCDEF = _mm512_min_ps(vscaled5x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled6x0123456789ABCDEF = _mm512_min_ps(vscaled6x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled7x0123456789ABCDEF = _mm512_min_ps(vscaled7x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled8x0123456789ABCDEF = _mm512_min_ps(vscaled8x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled9x0123456789ABCDEF = _mm512_min_ps(vscaled9x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled10x0123456789ABCDEF = _mm512_min_ps(vscaled10x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled11x0123456789ABCDEF = _mm512_min_ps(vscaled11x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled12x0123456789ABCDEF = _mm512_min_ps(vscaled12x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled13x0123456789ABCDEF = _mm512_min_ps(vscaled13x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled14x0123456789ABCDEF = _mm512_min_ps(vscaled14x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled15x0123456789ABCDEF = _mm512_min_ps(vscaled15x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled16x0123456789ABCDEF = _mm512_min_ps(vscaled16x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled17x0123456789ABCDEF = _mm512_min_ps(vscaled17x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled18x0123456789ABCDEF = _mm512_min_ps(vscaled18x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled19x0123456789ABCDEF = _mm512_min_ps(vscaled19x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled20x0123456789ABCDEF = _mm512_min_ps(vscaled20x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled21x0123456789ABCDEF = _mm512_min_ps(vscaled21x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled22x0123456789ABCDEF = _mm512_min_ps(vscaled22x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled23x0123456789ABCDEF = _mm512_min_ps(vscaled23x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled24x0123456789ABCDEF = _mm512_min_ps(vscaled24x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled25x0123456789ABCDEF = _mm512_min_ps(vscaled25x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled26x0123456789ABCDEF = _mm512_min_ps(vscaled26x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled27x0123456789ABCDEF = _mm512_min_ps(vscaled27x0123456789ABCDEF, voutput_max_less_zero_point);

    vacc0x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0x0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled1x0123456789ABCDEF);
    vacc2x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled2x0123456789ABCDEF);
    vacc3x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled3x0123456789ABCDEF);
    vacc4x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled4x0123456789ABCDEF);
    vacc5x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled5x0123456789ABCDEF);
    vacc6x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled6x0123456789ABCDEF);
    vacc7x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled7x0123456789ABCDEF);
    vacc8x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled8x0123456789ABCDEF);
    vacc9x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled9x0123456789ABCDEF);
    vacc10x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled10x0123456789ABCDEF);
    vacc11x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled11x0123456789ABCDEF);
    vacc12x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled12x0123456789ABCDEF);
    vacc13x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled13x0123456789ABCDEF);
    vacc14x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled14x0123456789ABCDEF);
    vacc15x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled15x0123456789ABCDEF);
    vacc16x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled16x0123456789ABCDEF);
    vacc17x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled17x0123456789ABCDEF);
    vacc18x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled18x0123456789ABCDEF);
    vacc19x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled19x0123456789ABCDEF);
    vacc20x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled20x0123456789ABCDEF);
    vacc21x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled21x0123456789ABCDEF);
    vacc22x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled22x0123456789ABCDEF);
    vacc23x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled23x0123456789ABCDEF);
    vacc24x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled24x0123456789ABCDEF);
    vacc25x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled25x0123456789ABCDEF);
    vacc26x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled26x0123456789ABCDEF);
    vacc27x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled27x0123456789ABCDEF);

    vacc0x0123456789ABCDEF = _mm512_add_epi32(vacc0x0123456789ABCDEF, voutput_zero_point);
    vacc1x0123456789ABCDEF = _mm512_add_epi32(vacc1x0123456789ABCDEF, voutput_zero_point);
    vacc2x0123456789ABCDEF = _mm512_add_epi32(vacc2x0123456789ABCDEF, voutput_zero_point);
    vacc3x0123456789ABCDEF = _mm512_add_epi32(vacc3x0123456789ABCDEF, voutput_zero_point);
    vacc4x0123456789ABCDEF = _mm512_add_epi32(vacc4x0123456789ABCDEF, voutput_zero_point);
    vacc5x0123456789ABCDEF = _mm512_add_epi32(vacc5x0123456789ABCDEF, voutput_zero_point);
    vacc6x0123456789ABCDEF = _mm512_add_epi32(vacc6x0123456789ABCDEF, voutput_zero_point);
    vacc7x0123456789ABCDEF = _mm512_add_epi32(vacc7x0123456789ABCDEF, voutput_zero_point);
    vacc8x0123456789ABCDEF = _mm512_add_epi32(vacc8x0123456789ABCDEF, voutput_zero_point);
    vacc9x0123456789ABCDEF = _mm512_add_epi32(vacc9x0123456789ABCDEF, voutput_zero_point);
    vacc10x0123456789ABCDEF = _mm512_add_epi32(vacc10x0123456789ABCDEF, voutput_zero_point);
    vacc11x0123456789ABCDEF = _mm512_add_epi32(vacc11x0123456789ABCDEF, voutput_zero_point);
    vacc12x0123456789ABCDEF = _mm512_add_epi32(vacc12x0123456789ABCDEF, voutput_zero_point);
    vacc13x0123456789ABCDEF = _mm512_add_epi32(vacc13x0123456789ABCDEF, voutput_zero_point);
    vacc14x0123456789ABCDEF = _mm512_add_epi32(vacc14x0123456789ABCDEF, voutput_zero_point);
    vacc15x0123456789ABCDEF = _mm512_add_epi32(vacc15x0123456789ABCDEF, voutput_zero_point);
    vacc16x0123456789ABCDEF = _mm512_add_epi32(vacc16x0123456789ABCDEF, voutput_zero_point);
    vacc17x0123456789ABCDEF = _mm512_add_epi32(vacc17x0123456789ABCDEF, voutput_zero_point);
    vacc18x0123456789ABCDEF = _mm512_add_epi32(vacc18x0123456789ABCDEF, voutput_zero_point);
    vacc19x0123456789ABCDEF = _mm512_add_epi32(vacc19x0123456789ABCDEF, voutput_zero_point);
    vacc20x0123456789ABCDEF = _mm512_add_epi32(vacc20x0123456789ABCDEF, voutput_zero_point);
    vacc21x0123456789ABCDEF = _mm512_add_epi32(vacc21x0123456789ABCDEF, voutput_zero_point);
    vacc22x0123456789ABCDEF = _mm512_add_epi32(vacc22x0123456789ABCDEF, voutput_zero_point);
    vacc23x0123456789ABCDEF = _mm512_add_epi32(vacc23x0123456789ABCDEF, voutput_zero_point);
    vacc24x0123456789ABCDEF = _mm512_add_epi32(vacc24x0123456789ABCDEF, voutput_zero_point);
    vacc25x0123456789ABCDEF = _mm512_add_epi32(vacc25x0123456789ABCDEF, voutput_zero_point);
    vacc26x0123456789ABCDEF = _mm512_add_epi32(vacc26x0123456789ABCDEF, voutput_zero_point);
    vacc27x0123456789ABCDEF = _mm512_add_epi32(vacc27x0123456789ABCDEF, voutput_zero_point);

    __m128i vout0x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc0x0123456789ABCDEF);
    __m128i vout1x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc1x0123456789ABCDEF);
    __m128i vout2x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc2x0123456789ABCDEF);
    __m128i vout3x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc3x0123456789ABCDEF);
    __m128i vout4x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc4x0123456789ABCDEF);
    __m128i vout5x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc5x0123456789ABCDEF);
    __m128i vout6x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc6x0123456789ABCDEF);
    __m128i vout7x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc7x0123456789ABCDEF);
    __m128i vout8x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc8x0123456789ABCDEF);
    __m128i vout9x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc9x0123456789ABCDEF);
    __m128i vout10x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc10x0123456789ABCDEF);
    __m128i vout11x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc11x0123456789ABCDEF);
    __m128i vout12x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc12x0123456789ABCDEF);
    __m128i vout13x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc13x0123456789ABCDEF);
    __m128i vout14x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc14x0123456789ABCDEF);
    __m128i vout15x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc15x0123456789ABCDEF);
    __m128i vout16x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc16x0123456789ABCDEF);
    __m128i vout17x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc17x0123456789ABCDEF);
    __m128i vout18x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc18x0123456789ABCDEF);
    __m128i vout19x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc19x0123456789ABCDEF);
    __m128i vout20x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc20x0123456789ABCDEF);
    __m128i vout21x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc21x0123456789ABCDEF);
    __m128i vout22x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc22x0123456789ABCDEF);
    __m128i vout23x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc23x0123456789ABCDEF);
    __m128i vout24x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc24x0123456789ABCDEF);
    __m128i vout25x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc25x0123456789ABCDEF);
    __m128i vout26x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc26x0123456789ABCDEF);
    __m128i vout27x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc27x0123456789ABCDEF);

    vout0x0123456789ABCDEF = _mm_max_epi8(vout0x0123456789ABCDEF, voutput_min);
    vout1x0123456789ABCDEF = _mm_max_epi8(vout1x0123456789ABCDEF, voutput_min);
    vout2x0123456789ABCDEF = _mm_max_epi8(vout2x0123456789ABCDEF, voutput_min);
    vout3x0123456789ABCDEF = _mm_max_epi8(vout3x0123456789ABCDEF, voutput_min);
    vout4x0123456789ABCDEF = _mm_max_epi8(vout4x0123456789ABCDEF, voutput_min);
    vout5x0123456789ABCDEF = _mm_max_epi8(vout5x0123456789ABCDEF, voutput_min);
    vout6x0123456789ABCDEF = _mm_max_epi8(vout6x0123456789ABCDEF, voutput_min);
    vout7x0123456789ABCDEF = _mm_max_epi8(vout7x0123456789ABCDEF, voutput_min);
    vout8x0123456789ABCDEF = _mm_max_epi8(vout8x0123456789ABCDEF, voutput_min);
    vout9x0123456789ABCDEF = _mm_max_epi8(vout9x0123456789ABCDEF, voutput_min);
    vout10x0123456789ABCDEF = _mm_max_epi8(vout10x0123456789ABCDEF, voutput_min);
    vout11x0123456789ABCDEF = _mm_max_epi8(vout11x0123456789ABCDEF, voutput_min);
    vout12x0123456789ABCDEF = _mm_max_epi8(vout12x0123456789ABCDEF, voutput_min);
    vout13x0123456789ABCDEF = _mm_max_epi8(vout13x0123456789ABCDEF, voutput_min);
    vout14x0123456789ABCDEF = _mm_max_epi8(vout14x0123456789ABCDEF, voutput_min);
    vout15x0123456789ABCDEF = _mm_max_epi8(vout15x0123456789ABCDEF, voutput_min);
    vout16x0123456789ABCDEF = _mm_max_epi8(vout16x0123456789ABCDEF, voutput_min);
    vout17x0123456789ABCDEF = _mm_max_epi8(vout17x0123456789ABCDEF, voutput_min);
    vout18x0123456789ABCDEF = _mm_max_epi8(vout18x0123456789ABCDEF, voutput_min);
    vout19x0123456789ABCDEF = _mm_max_epi8(vout19x0123456789ABCDEF, voutput_min);
    vout20x0123456789ABCDEF = _mm_max_epi8(vout20x0123456789ABCDEF, voutput_min);
    vout21x0123456789ABCDEF = _mm_max_epi8(vout21x0123456789ABCDEF, voutput_min);
    vout22x0123456789ABCDEF = _mm_max_epi8(vout22x0123456789ABCDEF, voutput_min);
    vout23x0123456789ABCDEF = _mm_max_epi8(vout23x0123456789ABCDEF, voutput_min);
    vout24x0123456789ABCDEF = _mm_max_epi8(vout24x0123456789ABCDEF, voutput_min);
    vout25x0123456789ABCDEF = _mm_max_epi8(vout25x0123456789ABCDEF, voutput_min);
    vout26x0123456789ABCDEF = _mm_max_epi8(vout26x0123456789ABCDEF, voutput_min);
    vout27x0123456789ABCDEF = _mm_max_epi8(vout27x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, vout0x0123456789ABCDEF);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      _mm_storeu_si128((__m128i*) c1, vout1x0123456789ABCDEF);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      _mm_storeu_si128((__m128i*) c2, vout2x0123456789ABCDEF);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      _mm_storeu_si128((__m128i*) c3, vout3x0123456789ABCDEF);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);
      _mm_storeu_si128((__m128i*) c4, vout4x0123456789ABCDEF);
      c4 = (int8_t*) ((uintptr_t) c4 + cn_stride);
      a4 = (const int8_t*) ((uintptr_t) a4 - kc);
      _mm_storeu_si128((__m128i*) c5, vout5x0123456789ABCDEF);
      c5 = (int8_t*) ((uintptr_t) c5 + cn_stride);
      a5 = (const int8_t*) ((uintptr_t) a5 - kc);
      _mm_storeu_si128((__m128i*) c6, vout6x0123456789ABCDEF);
      c6 = (int8_t*) ((uintptr_t) c6 + cn_stride);
      a6 = (const int8_t*) ((uintptr_t) a6 - kc);
      _mm_storeu_si128((__m128i*) c7, vout7x0123456789ABCDEF);
      c7 = (int8_t*) ((uintptr_t) c7 + cn_stride);
      a7 = (const int8_t*) ((uintptr_t) a7 - kc);
      _mm_storeu_si128((__m128i*) c8, vout8x0123456789ABCDEF);
      c8 = (int8_t*) ((uintptr_t) c8 + cn_stride);
      a8 = (const int8_t*) ((uintptr_t) a8 - kc);
      _mm_storeu_si128((__m128i*) c9, vout9x0123456789ABCDEF);
      c9 = (int8_t*) ((uintptr_t) c9 + cn_stride);
      a9 = (const int8_t*) ((uintptr_t) a9 - kc);
      _mm_storeu_si128((__m128i*) c10, vout10x0123456789ABCDEF);
      c10 = (int8_t*) ((uintptr_t) c10 + cn_stride);
      a10 = (const int8_t*) ((uintptr_t) a10 - kc);
      _mm_storeu_si128((__m128i*) c11, vout11x0123456789ABCDEF);
      c11 = (int8_t*) ((uintptr_t) c11 + cn_stride);
      a11 = (const int8_t*) ((uintptr_t) a11 - kc);
      _mm_storeu_si128((__m128i*) c12, vout12x0123456789ABCDEF);
      c12 = (int8_t*) ((uintptr_t) c12 + cn_stride);
      a12 = (const int8_t*) ((uintptr_t) a12 - kc);
      _mm_storeu_si128((__m128i*) c13, vout13x0123456789ABCDEF);
      c13 = (int8_t*) ((uintptr_t) c13 + cn_stride);
      a13 = (const int8_t*) ((uintptr_t) a13 - kc);
      _mm_storeu_si128((__m128i*) c14, vout14x0123456789ABCDEF);
      c14 = (int8_t*) ((uintptr_t) c14 + cn_stride);
      a14 = (const int8_t*) ((uintptr_t) a14 - kc);
      _mm_storeu_si128((__m128i*) c15, vout15x0123456789ABCDEF);
      c15 = (int8_t*) ((uintptr_t) c15 + cn_stride);
      a15 = (const int8_t*) ((uintptr_t) a15 - kc);
      _mm_storeu_si128((__m128i*) c16, vout16x0123456789ABCDEF);
      c16 = (int8_t*) ((uintptr_t) c16 + cn_stride);
      a16 = (const int8_t*) ((uintptr_t) a16 - kc);
      _mm_storeu_si128((__m128i*) c17, vout17x0123456789ABCDEF);
      c17 = (int8_t*) ((uintptr_t) c17 + cn_stride);
      a17 = (const int8_t*) ((uintptr_t) a17 - kc);
      _mm_storeu_si128((__m128i*) c18, vout18x0123456789ABCDEF);
      c18 = (int8_t*) ((uintptr_t) c18 + cn_stride);
      a18 = (const int8_t*) ((uintptr_t) a18 - kc);
      _mm_storeu_si128((__m128i*) c19, vout19x0123456789ABCDEF);
      c19 = (int8_t*) ((uintptr_t) c19 + cn_stride);
      a19 = (const int8_t*) ((uintptr_t) a19 - kc);
      _mm_storeu_si128((__m128i*) c20, vout20x0123456789ABCDEF);
      c20 = (int8_t*) ((uintptr_t) c20 + cn_stride);
      a20 = (const int8_t*) ((uintptr_t) a20 - kc);
      _mm_storeu_si128((__m128i*) c21, vout21x0123456789ABCDEF);
      c21 = (int8_t*) ((uintptr_t) c21 + cn_stride);
      a21 = (const int8_t*) ((uintptr_t) a21 - kc);
      _mm_storeu_si128((__m128i*) c22, vout22x0123456789ABCDEF);
      c22 = (int8_t*) ((uintptr_t) c22 + cn_stride);
      a22 = (const int8_t*) ((uintptr_t) a22 - kc);
      _mm_storeu_si128((__m128i*) c23, vout23x0123456789ABCDEF);
      c23 = (int8_t*) ((uintptr_t) c23 + cn_stride);
      a23 = (const int8_t*) ((uintptr_t) a23 - kc);
      _mm_storeu_si128((__m128i*) c24, vout24x0123456789ABCDEF);
      c24 = (int8_t*) ((uintptr_t) c24 + cn_stride);
      a24 = (const int8_t*) ((uintptr_t) a24 - kc);
      _mm_storeu_si128((__m128i*) c25, vout25x0123456789ABCDEF);
      c25 = (int8_t*) ((uintptr_t) c25 + cn_stride);
      a25 = (const int8_t*) ((uintptr_t) a25 - kc);
      _mm_storeu_si128((__m128i*) c26, vout26x0123456789ABCDEF);
      c26 = (int8_t*) ((uintptr_t) c26 + cn_stride);
      a26 = (const int8_t*) ((uintptr_t) a26 - kc);
      _mm_storeu_si128((__m128i*) c27, vout27x0123456789ABCDEF);
      c27 = (int8_t*) ((uintptr_t) c27 + cn_stride);
      a27 = (const int8_t*) ((uintptr_t) a27 - kc);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((UINT32_C(1) << nc) - UINT32_C(1));

      _mm_mask_storeu_epi8(c0, vmask, vout0x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c1, vmask, vout1x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c2, vmask, vout2x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c3, vmask, vout3x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c4, vmask, vout4x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c5, vmask, vout5x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c6, vmask, vout6x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c7, vmask, vout7x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c8, vmask, vout8x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c9, vmask, vout9x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c10, vmask, vout10x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c11, vmask, vout11x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c12, vmask, vout12x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c13, vmask, vout13x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c14, vmask, vout14x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c15, vmask, vout15x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c16, vmask, vout16x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c17, vmask, vout17x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c18, vmask, vout18x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c19, vmask, vout19x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c20, vmask, vout20x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c21, vmask, vout21x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c22, vmask, vout22x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c23, vmask, vout23x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c24, vmask, vout24x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c25, vmask, vout25x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c26, vmask, vout26x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c27, vmask, vout27x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}
