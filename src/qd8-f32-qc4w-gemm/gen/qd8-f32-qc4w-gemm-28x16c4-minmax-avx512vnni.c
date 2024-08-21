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

#include "xnnpack/common.h"
#include "xnnpack/gemm.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/unaligned.h"


void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_28x16c4__avx512vnni(
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
  assert(mr <= 28);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
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
  if XNN_UNPREDICTABLE(mr < 12) {
    a11 = a10;
    c11 = c10;
  }
  const int8_t* a12 = (const int8_t*) ((uintptr_t) a11 + a_stride);
  float* c12 = (float*) ((uintptr_t) c11 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 12) {
    a12 = a11;
    c12 = c11;
  }
  const int8_t* a13 = (const int8_t*) ((uintptr_t) a12 + a_stride);
  float* c13 = (float*) ((uintptr_t) c12 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 14) {
    a13 = a12;
    c13 = c12;
  }
  const int8_t* a14 = (const int8_t*) ((uintptr_t) a13 + a_stride);
  float* c14 = (float*) ((uintptr_t) c13 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 14) {
    a14 = a13;
    c14 = c13;
  }
  const int8_t* a15 = (const int8_t*) ((uintptr_t) a14 + a_stride);
  float* c15 = (float*) ((uintptr_t) c14 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 16) {
    a15 = a14;
    c15 = c14;
  }
  const int8_t* a16 = (const int8_t*) ((uintptr_t) a15 + a_stride);
  float* c16 = (float*) ((uintptr_t) c15 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 16) {
    a16 = a15;
    c16 = c15;
  }
  const int8_t* a17 = (const int8_t*) ((uintptr_t) a16 + a_stride);
  float* c17 = (float*) ((uintptr_t) c16 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 18) {
    a17 = a16;
    c17 = c16;
  }
  const int8_t* a18 = (const int8_t*) ((uintptr_t) a17 + a_stride);
  float* c18 = (float*) ((uintptr_t) c17 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 18) {
    a18 = a17;
    c18 = c17;
  }
  const int8_t* a19 = (const int8_t*) ((uintptr_t) a18 + a_stride);
  float* c19 = (float*) ((uintptr_t) c18 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 20) {
    a19 = a18;
    c19 = c18;
  }
  const int8_t* a20 = (const int8_t*) ((uintptr_t) a19 + a_stride);
  float* c20 = (float*) ((uintptr_t) c19 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 20) {
    a20 = a19;
    c20 = c19;
  }
  const int8_t* a21 = (const int8_t*) ((uintptr_t) a20 + a_stride);
  float* c21 = (float*) ((uintptr_t) c20 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 22) {
    a21 = a20;
    c21 = c20;
  }
  const int8_t* a22 = (const int8_t*) ((uintptr_t) a21 + a_stride);
  float* c22 = (float*) ((uintptr_t) c21 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 22) {
    a22 = a21;
    c22 = c21;
  }
  const int8_t* a23 = (const int8_t*) ((uintptr_t) a22 + a_stride);
  float* c23 = (float*) ((uintptr_t) c22 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 24) {
    a23 = a22;
    c23 = c22;
  }
  const int8_t* a24 = (const int8_t*) ((uintptr_t) a23 + a_stride);
  float* c24 = (float*) ((uintptr_t) c23 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 24) {
    a24 = a23;
    c24 = c23;
  }
  const int8_t* a25 = (const int8_t*) ((uintptr_t) a24 + a_stride);
  float* c25 = (float*) ((uintptr_t) c24 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 26) {
    a25 = a24;
    c25 = c24;
  }
  const int8_t* a26 = (const int8_t*) ((uintptr_t) a25 + a_stride);
  float* c26 = (float*) ((uintptr_t) c25 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 26) {
    a26 = a25;
    c26 = c25;
  }
  const int8_t* a27 = (const int8_t*) ((uintptr_t) a26 + a_stride);
  float* c27 = (float*) ((uintptr_t) c26 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 28) {
    a27 = a26;
    c27 = c26;
  }

  const __m512i vsign_mask = _mm512_set1_epi8(0x80);
  XNN_FORCE_REALIZATION(vsign_mask);
  const __m512i vinput_zero_point0 = _mm512_set1_epi32((int) quantization_params[0].zero_point + 128);
  const __m512i vinput_zero_point1 = _mm512_set1_epi32((int) quantization_params[1].zero_point + 128);
  const __m512i vinput_zero_point2 = _mm512_set1_epi32((int) quantization_params[2].zero_point + 128);
  const __m512i vinput_zero_point3 = _mm512_set1_epi32((int) quantization_params[3].zero_point + 128);
  const __m512i vinput_zero_point4 = _mm512_set1_epi32((int) quantization_params[4].zero_point + 128);
  const __m512i vinput_zero_point5 = _mm512_set1_epi32((int) quantization_params[5].zero_point + 128);
  const __m512i vinput_zero_point6 = _mm512_set1_epi32((int) quantization_params[6].zero_point + 128);
  const __m512i vinput_zero_point7 = _mm512_set1_epi32((int) quantization_params[7].zero_point + 128);
  const __m512i vinput_zero_point8 = _mm512_set1_epi32((int) quantization_params[8].zero_point + 128);
  const __m512i vinput_zero_point9 = _mm512_set1_epi32((int) quantization_params[9].zero_point + 128);
  const __m512i vinput_zero_point10 = _mm512_set1_epi32((int) quantization_params[10].zero_point + 128);
  const __m512i vinput_zero_point11 = _mm512_set1_epi32((int) quantization_params[11].zero_point + 128);
  const __m512i vinput_zero_point12 = _mm512_set1_epi32((int) quantization_params[12].zero_point + 128);
  const __m512i vinput_zero_point13 = _mm512_set1_epi32((int) quantization_params[13].zero_point + 128);
  const __m512i vinput_zero_point14 = _mm512_set1_epi32((int) quantization_params[14].zero_point + 128);
  const __m512i vinput_zero_point15 = _mm512_set1_epi32((int) quantization_params[15].zero_point + 128);
  const __m512i vinput_zero_point16 = _mm512_set1_epi32((int) quantization_params[16].zero_point + 128);
  const __m512i vinput_zero_point17 = _mm512_set1_epi32((int) quantization_params[17].zero_point + 128);
  const __m512i vinput_zero_point18 = _mm512_set1_epi32((int) quantization_params[18].zero_point + 128);
  const __m512i vinput_zero_point19 = _mm512_set1_epi32((int) quantization_params[19].zero_point + 128);
  const __m512i vinput_zero_point20 = _mm512_set1_epi32((int) quantization_params[20].zero_point + 128);
  const __m512i vinput_zero_point21 = _mm512_set1_epi32((int) quantization_params[21].zero_point + 128);
  const __m512i vinput_zero_point22 = _mm512_set1_epi32((int) quantization_params[22].zero_point + 128);
  const __m512i vinput_zero_point23 = _mm512_set1_epi32((int) quantization_params[23].zero_point + 128);
  const __m512i vinput_zero_point24 = _mm512_set1_epi32((int) quantization_params[24].zero_point + 128);
  const __m512i vinput_zero_point25 = _mm512_set1_epi32((int) quantization_params[25].zero_point + 128);
  const __m512i vinput_zero_point26 = _mm512_set1_epi32((int) quantization_params[26].zero_point + 128);
  const __m512i vinput_zero_point27 = _mm512_set1_epi32((int) quantization_params[27].zero_point + 128);
  const __m512 voutput_min = _mm512_set1_ps(params->avx512vnni.min);
  const __m512 voutput_max = _mm512_set1_ps(params->avx512vnni.max);
  const __m512i vmask = _mm512_set1_epi8(0xF0);
  XNN_FORCE_REALIZATION(vmask);
  do {
    const __m512i vksum0123456789ABCDEF = _mm512_load_epi32(w);
    __m512i vacc0x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point0);
    __m512i vacc1x0x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc1x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point1);
    __m512i vacc1x1x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc2x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point2);
    __m512i vacc1x2x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc3x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point3);
    __m512i vacc1x3x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc4x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point4);
    __m512i vacc1x4x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc5x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point5);
    __m512i vacc1x5x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc6x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point6);
    __m512i vacc1x6x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc7x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point7);
    __m512i vacc1x7x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc8x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point8);
    __m512i vacc1x8x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc9x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point9);
    __m512i vacc1x9x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc10x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point10);
    __m512i vacc1x10x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc11x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point11);
    __m512i vacc1x11x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc12x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point12);
    __m512i vacc1x12x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc13x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point13);
    __m512i vacc1x13x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc14x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point14);
    __m512i vacc1x14x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc15x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point15);
    __m512i vacc1x15x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc16x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point16);
    __m512i vacc1x16x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc17x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point17);
    __m512i vacc1x17x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc18x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point18);
    __m512i vacc1x18x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc19x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point19);
    __m512i vacc1x19x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc20x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point20);
    __m512i vacc1x20x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc21x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point21);
    __m512i vacc1x21x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc22x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point22);
    __m512i vacc1x22x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc23x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point23);
    __m512i vacc1x23x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc24x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point24);
    __m512i vacc1x24x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc25x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point25);
    __m512i vacc1x25x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc26x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point26);
    __m512i vacc1x26x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc27x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point27);
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

      const __m512i vbb0123456789ABCDEFx01234567 = _mm512_load_si512(w);
      const __m512i vbs0123456789ABCDEFx0123 = _mm512_slli_epi32(vbb0123456789ABCDEFx01234567, 4);
      const __m512i vb0123456789ABCDEFx4567 = _mm512_and_si512(vbb0123456789ABCDEFx01234567, vmask);
      const __m512i vb0123456789ABCDEFx0123 = _mm512_and_si512(vbs0123456789ABCDEFx0123, vmask);

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

      w = (const int8_t*) w + 64;
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

      const __m512i vbb0123456789ABCDEF = _mm512_load_si512(w);
      const __m512i vb0123456789ABCDEF = _mm512_slli_epi32(vbb0123456789ABCDEF, 4);

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

    vacc0x0123456789ABCDEF = _mm512_srai_epi32(vacc0x0123456789ABCDEF, 4);
    vacc1x0123456789ABCDEF = _mm512_srai_epi32(vacc1x0123456789ABCDEF, 4);
    vacc2x0123456789ABCDEF = _mm512_srai_epi32(vacc2x0123456789ABCDEF, 4);
    vacc3x0123456789ABCDEF = _mm512_srai_epi32(vacc3x0123456789ABCDEF, 4);
    vacc4x0123456789ABCDEF = _mm512_srai_epi32(vacc4x0123456789ABCDEF, 4);
    vacc5x0123456789ABCDEF = _mm512_srai_epi32(vacc5x0123456789ABCDEF, 4);
    vacc6x0123456789ABCDEF = _mm512_srai_epi32(vacc6x0123456789ABCDEF, 4);
    vacc7x0123456789ABCDEF = _mm512_srai_epi32(vacc7x0123456789ABCDEF, 4);
    vacc8x0123456789ABCDEF = _mm512_srai_epi32(vacc8x0123456789ABCDEF, 4);
    vacc9x0123456789ABCDEF = _mm512_srai_epi32(vacc9x0123456789ABCDEF, 4);
    vacc10x0123456789ABCDEF = _mm512_srai_epi32(vacc10x0123456789ABCDEF, 4);
    vacc11x0123456789ABCDEF = _mm512_srai_epi32(vacc11x0123456789ABCDEF, 4);
    vacc12x0123456789ABCDEF = _mm512_srai_epi32(vacc12x0123456789ABCDEF, 4);
    vacc13x0123456789ABCDEF = _mm512_srai_epi32(vacc13x0123456789ABCDEF, 4);
    vacc14x0123456789ABCDEF = _mm512_srai_epi32(vacc14x0123456789ABCDEF, 4);
    vacc15x0123456789ABCDEF = _mm512_srai_epi32(vacc15x0123456789ABCDEF, 4);
    vacc16x0123456789ABCDEF = _mm512_srai_epi32(vacc16x0123456789ABCDEF, 4);
    vacc17x0123456789ABCDEF = _mm512_srai_epi32(vacc17x0123456789ABCDEF, 4);
    vacc18x0123456789ABCDEF = _mm512_srai_epi32(vacc18x0123456789ABCDEF, 4);
    vacc19x0123456789ABCDEF = _mm512_srai_epi32(vacc19x0123456789ABCDEF, 4);
    vacc20x0123456789ABCDEF = _mm512_srai_epi32(vacc20x0123456789ABCDEF, 4);
    vacc21x0123456789ABCDEF = _mm512_srai_epi32(vacc21x0123456789ABCDEF, 4);
    vacc22x0123456789ABCDEF = _mm512_srai_epi32(vacc22x0123456789ABCDEF, 4);
    vacc23x0123456789ABCDEF = _mm512_srai_epi32(vacc23x0123456789ABCDEF, 4);
    vacc24x0123456789ABCDEF = _mm512_srai_epi32(vacc24x0123456789ABCDEF, 4);
    vacc25x0123456789ABCDEF = _mm512_srai_epi32(vacc25x0123456789ABCDEF, 4);
    vacc26x0123456789ABCDEF = _mm512_srai_epi32(vacc26x0123456789ABCDEF, 4);
    vacc27x0123456789ABCDEF = _mm512_srai_epi32(vacc27x0123456789ABCDEF, 4);
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
    vscaled12x0123456789ABCDEF = _mm512_mul_ps(vscaled12x0123456789ABCDEF, _mm512_set1_ps(quantization_params[12].inv_scale));
    vscaled13x0123456789ABCDEF = _mm512_mul_ps(vscaled13x0123456789ABCDEF, _mm512_set1_ps(quantization_params[13].inv_scale));
    vscaled14x0123456789ABCDEF = _mm512_mul_ps(vscaled14x0123456789ABCDEF, _mm512_set1_ps(quantization_params[14].inv_scale));
    vscaled15x0123456789ABCDEF = _mm512_mul_ps(vscaled15x0123456789ABCDEF, _mm512_set1_ps(quantization_params[15].inv_scale));
    vscaled16x0123456789ABCDEF = _mm512_mul_ps(vscaled16x0123456789ABCDEF, _mm512_set1_ps(quantization_params[16].inv_scale));
    vscaled17x0123456789ABCDEF = _mm512_mul_ps(vscaled17x0123456789ABCDEF, _mm512_set1_ps(quantization_params[17].inv_scale));
    vscaled18x0123456789ABCDEF = _mm512_mul_ps(vscaled18x0123456789ABCDEF, _mm512_set1_ps(quantization_params[18].inv_scale));
    vscaled19x0123456789ABCDEF = _mm512_mul_ps(vscaled19x0123456789ABCDEF, _mm512_set1_ps(quantization_params[19].inv_scale));
    vscaled20x0123456789ABCDEF = _mm512_mul_ps(vscaled20x0123456789ABCDEF, _mm512_set1_ps(quantization_params[20].inv_scale));
    vscaled21x0123456789ABCDEF = _mm512_mul_ps(vscaled21x0123456789ABCDEF, _mm512_set1_ps(quantization_params[21].inv_scale));
    vscaled22x0123456789ABCDEF = _mm512_mul_ps(vscaled22x0123456789ABCDEF, _mm512_set1_ps(quantization_params[22].inv_scale));
    vscaled23x0123456789ABCDEF = _mm512_mul_ps(vscaled23x0123456789ABCDEF, _mm512_set1_ps(quantization_params[23].inv_scale));
    vscaled24x0123456789ABCDEF = _mm512_mul_ps(vscaled24x0123456789ABCDEF, _mm512_set1_ps(quantization_params[24].inv_scale));
    vscaled25x0123456789ABCDEF = _mm512_mul_ps(vscaled25x0123456789ABCDEF, _mm512_set1_ps(quantization_params[25].inv_scale));
    vscaled26x0123456789ABCDEF = _mm512_mul_ps(vscaled26x0123456789ABCDEF, _mm512_set1_ps(quantization_params[26].inv_scale));
    vscaled27x0123456789ABCDEF = _mm512_mul_ps(vscaled27x0123456789ABCDEF, _mm512_set1_ps(quantization_params[27].inv_scale));

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
      _mm512_storeu_ps(c12, vscaled12x0123456789ABCDEF);
      _mm512_storeu_ps(c13, vscaled13x0123456789ABCDEF);
      _mm512_storeu_ps(c14, vscaled14x0123456789ABCDEF);
      _mm512_storeu_ps(c15, vscaled15x0123456789ABCDEF);
      _mm512_storeu_ps(c16, vscaled16x0123456789ABCDEF);
      _mm512_storeu_ps(c17, vscaled17x0123456789ABCDEF);
      _mm512_storeu_ps(c18, vscaled18x0123456789ABCDEF);
      _mm512_storeu_ps(c19, vscaled19x0123456789ABCDEF);
      _mm512_storeu_ps(c20, vscaled20x0123456789ABCDEF);
      _mm512_storeu_ps(c21, vscaled21x0123456789ABCDEF);
      _mm512_storeu_ps(c22, vscaled22x0123456789ABCDEF);
      _mm512_storeu_ps(c23, vscaled23x0123456789ABCDEF);
      _mm512_storeu_ps(c24, vscaled24x0123456789ABCDEF);
      _mm512_storeu_ps(c25, vscaled25x0123456789ABCDEF);
      _mm512_storeu_ps(c26, vscaled26x0123456789ABCDEF);
      _mm512_storeu_ps(c27, vscaled27x0123456789ABCDEF);

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
      a12 = (const int8_t*) ((uintptr_t) a12 - kc);
      a13 = (const int8_t*) ((uintptr_t) a13 - kc);
      a14 = (const int8_t*) ((uintptr_t) a14 - kc);
      a15 = (const int8_t*) ((uintptr_t) a15 - kc);
      a16 = (const int8_t*) ((uintptr_t) a16 - kc);
      a17 = (const int8_t*) ((uintptr_t) a17 - kc);
      a18 = (const int8_t*) ((uintptr_t) a18 - kc);
      a19 = (const int8_t*) ((uintptr_t) a19 - kc);
      a20 = (const int8_t*) ((uintptr_t) a20 - kc);
      a21 = (const int8_t*) ((uintptr_t) a21 - kc);
      a22 = (const int8_t*) ((uintptr_t) a22 - kc);
      a23 = (const int8_t*) ((uintptr_t) a23 - kc);
      a24 = (const int8_t*) ((uintptr_t) a24 - kc);
      a25 = (const int8_t*) ((uintptr_t) a25 - kc);
      a26 = (const int8_t*) ((uintptr_t) a26 - kc);
      a27 = (const int8_t*) ((uintptr_t) a27 - kc);

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
      c12 = (float*) ((uintptr_t) c12 + cn_stride);
      c13 = (float*) ((uintptr_t) c13 + cn_stride);
      c14 = (float*) ((uintptr_t) c14 + cn_stride);
      c15 = (float*) ((uintptr_t) c15 + cn_stride);
      c16 = (float*) ((uintptr_t) c16 + cn_stride);
      c17 = (float*) ((uintptr_t) c17 + cn_stride);
      c18 = (float*) ((uintptr_t) c18 + cn_stride);
      c19 = (float*) ((uintptr_t) c19 + cn_stride);
      c20 = (float*) ((uintptr_t) c20 + cn_stride);
      c21 = (float*) ((uintptr_t) c21 + cn_stride);
      c22 = (float*) ((uintptr_t) c22 + cn_stride);
      c23 = (float*) ((uintptr_t) c23 + cn_stride);
      c24 = (float*) ((uintptr_t) c24 + cn_stride);
      c25 = (float*) ((uintptr_t) c25 + cn_stride);
      c26 = (float*) ((uintptr_t) c26 + cn_stride);
      c27 = (float*) ((uintptr_t) c27 + cn_stride);

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
      _mm512_mask_storeu_ps(c12, vmask, vscaled12x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c13, vmask, vscaled13x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c14, vmask, vscaled14x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c15, vmask, vscaled15x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c16, vmask, vscaled16x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c17, vmask, vscaled17x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c18, vmask, vscaled18x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c19, vmask, vscaled19x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c20, vmask, vscaled20x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c21, vmask, vscaled21x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c22, vmask, vscaled22x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c23, vmask, vscaled23x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c24, vmask, vscaled24x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c25, vmask, vscaled25x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c26, vmask, vscaled26x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c27, vmask, vscaled27x0123456789ABCDEF);
      nc = 0;
    }
  } while (nc != 0);
}
