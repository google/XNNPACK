// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c4-avx512amx.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
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
#include "xnnpack/prefetch.h"


void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx_prfm(
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
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 16);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

// TODO: amxintrin.h only provide intrinsics for __x86_64__
// Update if amxintrin changes
#if defined(__x86_64__)
  __attribute__((aligned(64))) int32_t res0[16 * 16];
  __attribute__((aligned(64))) int32_t res1[16 * 16];
  __attribute__((aligned(64))) int32_t res2[16 * 16];
  __attribute__((aligned(64))) int32_t res3[16 * 16];
  __attribute__((aligned(64))) int8_t weight_buffer[16 * 64];

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  const size_t kremainder = (kc & 63) ? (kc & 63) : 64;

  // Define tile config data structure
  struct __tile_config {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[8];
    uint16_t reserved_1[8];
    uint8_t rows[8];
    uint8_t reserved_2[8];
  };

  // Load tile configuration
  __attribute__((aligned(64))) struct __tile_config tile_data = {0};
  tile_data.palette_id = 1;
  tile_data.rows[0] = mr;              // tmm0 = res 0
  tile_data.rows[1] = mr;              // tmm1 = res 1
  tile_data.rows[2] = mr;              // tmm2 = res 2
  tile_data.rows[3] = mr;              // tmm3 = res 3
  tile_data.rows[4] = mr;              // tmm4 = input
  tile_data.rows[5] = 16;              // tmm5 = weights
  tile_data.rows[6] = mr;              // tmm6 = input remainder
  tile_data.rows[7] = kremainder >> 2; // tmm7 = weights remainder

  tile_data.colsb[0] = 64;          // tmm0 = res 0
  tile_data.colsb[1] = 64;          // tmm1 = res 1
  tile_data.colsb[2] = 64;          // tmm2 = res 1
  tile_data.colsb[3] = 64;          // tmm3 = res 1
  tile_data.colsb[4] = 64;          // tmm4 = input
  tile_data.colsb[5] = 64;          // tmm5 = weights
  tile_data.colsb[6] = kremainder;  // tmm6 = input remainder
  tile_data.colsb[7] = 64;          // tmm7 = weights remainder

  //_tile_loadconfig(&tile_data);
  __asm__ volatile ("ldtilecfg %0" :: "m" (tile_data));

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
  if XNN_UNPREDICTABLE(mr != 16) {
    c15 = c14;
  }

  const __m512 voutput_min = _mm512_set1_ps(params->avx512vnni.min);
  const __m512 voutput_max = _mm512_set1_ps(params->avx512vnni.max);
  const __m512i vmask = _mm512_set1_epi8(0xF0);
  const __m512i vshl4 = _mm512_set1_epi64(0x01020408);
  XNN_FORCE_REALIZATION(vmask);
  XNN_FORCE_REALIZATION(vshl4);

  do {
    const __m512i vksum0123456789ABCDEF = _mm512_load_epi32((const int32_t*) w + 0);
    const __m512i vksumGHIJKLMNOPQRSTUV = _mm512_load_epi32((const int32_t*) w + 16);
    const __m512i vksumWXYZabcdefghijkl = _mm512_load_epi32((const int32_t*) w + 32);
    const __m512i vksummnopqrstuvwxyz01 = _mm512_load_epi32((const int32_t*) w + 48);
    w = (const int32_t*) w + 64;

    // Zero tile accumulator
    __asm__ volatile (
      "tilezero %%tmm0\n"
      "tilezero %%tmm1\n"
      "tilezero %%tmm2\n"
      "tilezero %%tmm3\n"
      ::);

    size_t k = kc;
    while (k >= 64 * sizeof(int8_t)) {
      _tile_loadd(4, a, a_stride);
      a += 64;
      const __m512i vb0x0123456789ABCDEF = _mm512_load_epi32((const int8_t*) w + 0);
      const __m512i vb1x0123456789ABCDEF = _mm512_load_epi32((const int8_t*) w + 256);
      const __m512i vb2x0123456789ABCDEF = _mm512_load_epi32((const int8_t*) w + 512);
      const __m512i vb3x0123456789ABCDEF = _mm512_load_epi32((const int8_t*) w + 768);
      const __m512i vb4x0123456789ABCDEF = _mm512_load_epi32((const int8_t*) w + 1024);
      const __m512i vb5x0123456789ABCDEF = _mm512_load_epi32((const int8_t*) w + 1280);
      const __m512i vb6x0123456789ABCDEF = _mm512_load_epi32((const int8_t*) w + 1536);
      const __m512i vb7x0123456789ABCDEF = _mm512_load_epi32((const int8_t*) w + 1792);
      const __m512i vl0x0123456789ABCDEF = _mm512_gf2p8affine_epi64_epi8(vb0x0123456789ABCDEF, vshl4, 0);
      const __m512i vh0x0123456789ABCDEF = _mm512_and_si512(vb0x0123456789ABCDEF, vmask);
      const __m512i vl1x0123456789ABCDEF = _mm512_gf2p8affine_epi64_epi8(vb1x0123456789ABCDEF, vshl4, 0);
      const __m512i vh1x0123456789ABCDEF = _mm512_and_si512(vb1x0123456789ABCDEF, vmask);
      const __m512i vl2x0123456789ABCDEF = _mm512_gf2p8affine_epi64_epi8(vb2x0123456789ABCDEF, vshl4, 0);
      const __m512i vh2x0123456789ABCDEF = _mm512_and_si512(vb2x0123456789ABCDEF, vmask);
      const __m512i vl3x0123456789ABCDEF = _mm512_gf2p8affine_epi64_epi8(vb3x0123456789ABCDEF, vshl4, 0);
      const __m512i vh3x0123456789ABCDEF = _mm512_and_si512(vb3x0123456789ABCDEF, vmask);
      const __m512i vl4x0123456789ABCDEF = _mm512_gf2p8affine_epi64_epi8(vb4x0123456789ABCDEF, vshl4, 0);
      const __m512i vh4x0123456789ABCDEF = _mm512_and_si512(vb4x0123456789ABCDEF, vmask);
      const __m512i vl5x0123456789ABCDEF = _mm512_gf2p8affine_epi64_epi8(vb5x0123456789ABCDEF, vshl4, 0);
      const __m512i vh5x0123456789ABCDEF = _mm512_and_si512(vb5x0123456789ABCDEF, vmask);
      const __m512i vl6x0123456789ABCDEF = _mm512_gf2p8affine_epi64_epi8(vb6x0123456789ABCDEF, vshl4, 0);
      const __m512i vh6x0123456789ABCDEF = _mm512_and_si512(vb6x0123456789ABCDEF, vmask);
      const __m512i vl7x0123456789ABCDEF = _mm512_gf2p8affine_epi64_epi8(vb7x0123456789ABCDEF, vshl4, 0);
      const __m512i vh7x0123456789ABCDEF = _mm512_and_si512(vb7x0123456789ABCDEF, vmask);
      _mm512_store_epi32(weight_buffer + 0, vl0x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 64, vh0x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 128, vl1x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 192, vh1x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 256, vl2x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 320, vh2x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 384, vl3x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 448, vh3x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 512, vl4x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 576, vh4x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 640, vl5x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 704, vh5x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 768, vl6x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 832, vh6x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 896, vl7x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 960, vh7x0123456789ABCDEF);
      _tile_loadd(5, weight_buffer, 64);
      _tile_dpbssd(0, 4, 5);
      const __m512i vb0xGHIJKLMNOPQRSTUV = _mm512_load_epi32((const int8_t*) w + 64);
      const __m512i vb1xGHIJKLMNOPQRSTUV = _mm512_load_epi32((const int8_t*) w + 320);
      const __m512i vb2xGHIJKLMNOPQRSTUV = _mm512_load_epi32((const int8_t*) w + 576);
      const __m512i vb3xGHIJKLMNOPQRSTUV = _mm512_load_epi32((const int8_t*) w + 832);
      const __m512i vb4xGHIJKLMNOPQRSTUV = _mm512_load_epi32((const int8_t*) w + 1088);
      const __m512i vb5xGHIJKLMNOPQRSTUV = _mm512_load_epi32((const int8_t*) w + 1344);
      const __m512i vb6xGHIJKLMNOPQRSTUV = _mm512_load_epi32((const int8_t*) w + 1600);
      const __m512i vb7xGHIJKLMNOPQRSTUV = _mm512_load_epi32((const int8_t*) w + 1856);
      const __m512i vl0xGHIJKLMNOPQRSTUV = _mm512_gf2p8affine_epi64_epi8(vb0xGHIJKLMNOPQRSTUV, vshl4, 0);
      const __m512i vh0xGHIJKLMNOPQRSTUV = _mm512_and_si512(vb0xGHIJKLMNOPQRSTUV, vmask);
      const __m512i vl1xGHIJKLMNOPQRSTUV = _mm512_gf2p8affine_epi64_epi8(vb1xGHIJKLMNOPQRSTUV, vshl4, 0);
      const __m512i vh1xGHIJKLMNOPQRSTUV = _mm512_and_si512(vb1xGHIJKLMNOPQRSTUV, vmask);
      const __m512i vl2xGHIJKLMNOPQRSTUV = _mm512_gf2p8affine_epi64_epi8(vb2xGHIJKLMNOPQRSTUV, vshl4, 0);
      const __m512i vh2xGHIJKLMNOPQRSTUV = _mm512_and_si512(vb2xGHIJKLMNOPQRSTUV, vmask);
      const __m512i vl3xGHIJKLMNOPQRSTUV = _mm512_gf2p8affine_epi64_epi8(vb3xGHIJKLMNOPQRSTUV, vshl4, 0);
      const __m512i vh3xGHIJKLMNOPQRSTUV = _mm512_and_si512(vb3xGHIJKLMNOPQRSTUV, vmask);
      const __m512i vl4xGHIJKLMNOPQRSTUV = _mm512_gf2p8affine_epi64_epi8(vb4xGHIJKLMNOPQRSTUV, vshl4, 0);
      const __m512i vh4xGHIJKLMNOPQRSTUV = _mm512_and_si512(vb4xGHIJKLMNOPQRSTUV, vmask);
      const __m512i vl5xGHIJKLMNOPQRSTUV = _mm512_gf2p8affine_epi64_epi8(vb5xGHIJKLMNOPQRSTUV, vshl4, 0);
      const __m512i vh5xGHIJKLMNOPQRSTUV = _mm512_and_si512(vb5xGHIJKLMNOPQRSTUV, vmask);
      const __m512i vl6xGHIJKLMNOPQRSTUV = _mm512_gf2p8affine_epi64_epi8(vb6xGHIJKLMNOPQRSTUV, vshl4, 0);
      const __m512i vh6xGHIJKLMNOPQRSTUV = _mm512_and_si512(vb6xGHIJKLMNOPQRSTUV, vmask);
      const __m512i vl7xGHIJKLMNOPQRSTUV = _mm512_gf2p8affine_epi64_epi8(vb7xGHIJKLMNOPQRSTUV, vshl4, 0);
      const __m512i vh7xGHIJKLMNOPQRSTUV = _mm512_and_si512(vb7xGHIJKLMNOPQRSTUV, vmask);
      _mm512_store_epi32(weight_buffer + 0, vl0xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 64, vh0xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 128, vl1xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 192, vh1xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 256, vl2xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 320, vh2xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 384, vl3xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 448, vh3xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 512, vl4xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 576, vh4xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 640, vl5xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 704, vh5xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 768, vl6xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 832, vh6xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 896, vl7xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 960, vh7xGHIJKLMNOPQRSTUV);
      _tile_loadd(5, weight_buffer, 64);
      _tile_dpbssd(1, 4, 5);
      const __m512i vb0xWXYZabcdefghijkl = _mm512_load_epi32((const int8_t*) w + 128);
      const __m512i vb1xWXYZabcdefghijkl = _mm512_load_epi32((const int8_t*) w + 384);
      const __m512i vb2xWXYZabcdefghijkl = _mm512_load_epi32((const int8_t*) w + 640);
      const __m512i vb3xWXYZabcdefghijkl = _mm512_load_epi32((const int8_t*) w + 896);
      const __m512i vb4xWXYZabcdefghijkl = _mm512_load_epi32((const int8_t*) w + 1152);
      const __m512i vb5xWXYZabcdefghijkl = _mm512_load_epi32((const int8_t*) w + 1408);
      const __m512i vb6xWXYZabcdefghijkl = _mm512_load_epi32((const int8_t*) w + 1664);
      const __m512i vb7xWXYZabcdefghijkl = _mm512_load_epi32((const int8_t*) w + 1920);
      const __m512i vl0xWXYZabcdefghijkl = _mm512_gf2p8affine_epi64_epi8(vb0xWXYZabcdefghijkl, vshl4, 0);
      const __m512i vh0xWXYZabcdefghijkl = _mm512_and_si512(vb0xWXYZabcdefghijkl, vmask);
      const __m512i vl1xWXYZabcdefghijkl = _mm512_gf2p8affine_epi64_epi8(vb1xWXYZabcdefghijkl, vshl4, 0);
      const __m512i vh1xWXYZabcdefghijkl = _mm512_and_si512(vb1xWXYZabcdefghijkl, vmask);
      const __m512i vl2xWXYZabcdefghijkl = _mm512_gf2p8affine_epi64_epi8(vb2xWXYZabcdefghijkl, vshl4, 0);
      const __m512i vh2xWXYZabcdefghijkl = _mm512_and_si512(vb2xWXYZabcdefghijkl, vmask);
      const __m512i vl3xWXYZabcdefghijkl = _mm512_gf2p8affine_epi64_epi8(vb3xWXYZabcdefghijkl, vshl4, 0);
      const __m512i vh3xWXYZabcdefghijkl = _mm512_and_si512(vb3xWXYZabcdefghijkl, vmask);
      const __m512i vl4xWXYZabcdefghijkl = _mm512_gf2p8affine_epi64_epi8(vb4xWXYZabcdefghijkl, vshl4, 0);
      const __m512i vh4xWXYZabcdefghijkl = _mm512_and_si512(vb4xWXYZabcdefghijkl, vmask);
      const __m512i vl5xWXYZabcdefghijkl = _mm512_gf2p8affine_epi64_epi8(vb5xWXYZabcdefghijkl, vshl4, 0);
      const __m512i vh5xWXYZabcdefghijkl = _mm512_and_si512(vb5xWXYZabcdefghijkl, vmask);
      const __m512i vl6xWXYZabcdefghijkl = _mm512_gf2p8affine_epi64_epi8(vb6xWXYZabcdefghijkl, vshl4, 0);
      const __m512i vh6xWXYZabcdefghijkl = _mm512_and_si512(vb6xWXYZabcdefghijkl, vmask);
      const __m512i vl7xWXYZabcdefghijkl = _mm512_gf2p8affine_epi64_epi8(vb7xWXYZabcdefghijkl, vshl4, 0);
      const __m512i vh7xWXYZabcdefghijkl = _mm512_and_si512(vb7xWXYZabcdefghijkl, vmask);
      _mm512_store_epi32(weight_buffer + 0, vl0xWXYZabcdefghijkl);
      _mm512_store_epi32(weight_buffer + 64, vh0xWXYZabcdefghijkl);
      _mm512_store_epi32(weight_buffer + 128, vl1xWXYZabcdefghijkl);
      _mm512_store_epi32(weight_buffer + 192, vh1xWXYZabcdefghijkl);
      _mm512_store_epi32(weight_buffer + 256, vl2xWXYZabcdefghijkl);
      _mm512_store_epi32(weight_buffer + 320, vh2xWXYZabcdefghijkl);
      _mm512_store_epi32(weight_buffer + 384, vl3xWXYZabcdefghijkl);
      _mm512_store_epi32(weight_buffer + 448, vh3xWXYZabcdefghijkl);
      _mm512_store_epi32(weight_buffer + 512, vl4xWXYZabcdefghijkl);
      _mm512_store_epi32(weight_buffer + 576, vh4xWXYZabcdefghijkl);
      _mm512_store_epi32(weight_buffer + 640, vl5xWXYZabcdefghijkl);
      _mm512_store_epi32(weight_buffer + 704, vh5xWXYZabcdefghijkl);
      _mm512_store_epi32(weight_buffer + 768, vl6xWXYZabcdefghijkl);
      _mm512_store_epi32(weight_buffer + 832, vh6xWXYZabcdefghijkl);
      _mm512_store_epi32(weight_buffer + 896, vl7xWXYZabcdefghijkl);
      _mm512_store_epi32(weight_buffer + 960, vh7xWXYZabcdefghijkl);
      _tile_loadd(5, weight_buffer, 64);
      _tile_dpbssd(2, 4, 5);
      const __m512i vb0xmnopqrstuvwxyz01 = _mm512_load_epi32((const int8_t*) w + 192);
      const __m512i vb1xmnopqrstuvwxyz01 = _mm512_load_epi32((const int8_t*) w + 448);
      const __m512i vb2xmnopqrstuvwxyz01 = _mm512_load_epi32((const int8_t*) w + 704);
      const __m512i vb3xmnopqrstuvwxyz01 = _mm512_load_epi32((const int8_t*) w + 960);
      const __m512i vb4xmnopqrstuvwxyz01 = _mm512_load_epi32((const int8_t*) w + 1216);
      const __m512i vb5xmnopqrstuvwxyz01 = _mm512_load_epi32((const int8_t*) w + 1472);
      const __m512i vb6xmnopqrstuvwxyz01 = _mm512_load_epi32((const int8_t*) w + 1728);
      const __m512i vb7xmnopqrstuvwxyz01 = _mm512_load_epi32((const int8_t*) w + 1984);
      const __m512i vl0xmnopqrstuvwxyz01 = _mm512_gf2p8affine_epi64_epi8(vb0xmnopqrstuvwxyz01, vshl4, 0);
      const __m512i vh0xmnopqrstuvwxyz01 = _mm512_and_si512(vb0xmnopqrstuvwxyz01, vmask);
      const __m512i vl1xmnopqrstuvwxyz01 = _mm512_gf2p8affine_epi64_epi8(vb1xmnopqrstuvwxyz01, vshl4, 0);
      const __m512i vh1xmnopqrstuvwxyz01 = _mm512_and_si512(vb1xmnopqrstuvwxyz01, vmask);
      const __m512i vl2xmnopqrstuvwxyz01 = _mm512_gf2p8affine_epi64_epi8(vb2xmnopqrstuvwxyz01, vshl4, 0);
      const __m512i vh2xmnopqrstuvwxyz01 = _mm512_and_si512(vb2xmnopqrstuvwxyz01, vmask);
      const __m512i vl3xmnopqrstuvwxyz01 = _mm512_gf2p8affine_epi64_epi8(vb3xmnopqrstuvwxyz01, vshl4, 0);
      const __m512i vh3xmnopqrstuvwxyz01 = _mm512_and_si512(vb3xmnopqrstuvwxyz01, vmask);
      const __m512i vl4xmnopqrstuvwxyz01 = _mm512_gf2p8affine_epi64_epi8(vb4xmnopqrstuvwxyz01, vshl4, 0);
      const __m512i vh4xmnopqrstuvwxyz01 = _mm512_and_si512(vb4xmnopqrstuvwxyz01, vmask);
      const __m512i vl5xmnopqrstuvwxyz01 = _mm512_gf2p8affine_epi64_epi8(vb5xmnopqrstuvwxyz01, vshl4, 0);
      const __m512i vh5xmnopqrstuvwxyz01 = _mm512_and_si512(vb5xmnopqrstuvwxyz01, vmask);
      const __m512i vl6xmnopqrstuvwxyz01 = _mm512_gf2p8affine_epi64_epi8(vb6xmnopqrstuvwxyz01, vshl4, 0);
      const __m512i vh6xmnopqrstuvwxyz01 = _mm512_and_si512(vb6xmnopqrstuvwxyz01, vmask);
      const __m512i vl7xmnopqrstuvwxyz01 = _mm512_gf2p8affine_epi64_epi8(vb7xmnopqrstuvwxyz01, vshl4, 0);
      const __m512i vh7xmnopqrstuvwxyz01 = _mm512_and_si512(vb7xmnopqrstuvwxyz01, vmask);
      _mm512_store_epi32(weight_buffer + 0, vl0xmnopqrstuvwxyz01);
      _mm512_store_epi32(weight_buffer + 64, vh0xmnopqrstuvwxyz01);
      _mm512_store_epi32(weight_buffer + 128, vl1xmnopqrstuvwxyz01);
      _mm512_store_epi32(weight_buffer + 192, vh1xmnopqrstuvwxyz01);
      _mm512_store_epi32(weight_buffer + 256, vl2xmnopqrstuvwxyz01);
      _mm512_store_epi32(weight_buffer + 320, vh2xmnopqrstuvwxyz01);
      _mm512_store_epi32(weight_buffer + 384, vl3xmnopqrstuvwxyz01);
      _mm512_store_epi32(weight_buffer + 448, vh3xmnopqrstuvwxyz01);
      _mm512_store_epi32(weight_buffer + 512, vl4xmnopqrstuvwxyz01);
      _mm512_store_epi32(weight_buffer + 576, vh4xmnopqrstuvwxyz01);
      _mm512_store_epi32(weight_buffer + 640, vl5xmnopqrstuvwxyz01);
      _mm512_store_epi32(weight_buffer + 704, vh5xmnopqrstuvwxyz01);
      _mm512_store_epi32(weight_buffer + 768, vl6xmnopqrstuvwxyz01);
      _mm512_store_epi32(weight_buffer + 832, vh6xmnopqrstuvwxyz01);
      _mm512_store_epi32(weight_buffer + 896, vl7xmnopqrstuvwxyz01);
      _mm512_store_epi32(weight_buffer + 960, vh7xmnopqrstuvwxyz01);
      _tile_loadd(5, weight_buffer, 64);
      _tile_dpbssd(3, 4, 5);
      xnn_prefetch_to_l1((const int8_t*) w + 4096);
      xnn_prefetch_to_l1((const int8_t*) w + 4160);
      xnn_prefetch_to_l1((const int8_t*) w + 4224);
      xnn_prefetch_to_l1((const int8_t*) w + 4288);
      xnn_prefetch_to_l1((const int8_t*) w + 4352);
      xnn_prefetch_to_l1((const int8_t*) w + 4416);
      xnn_prefetch_to_l1((const int8_t*) w + 4480);
      xnn_prefetch_to_l1((const int8_t*) w + 4544);
      xnn_prefetch_to_l1((const int8_t*) w + 4608);
      xnn_prefetch_to_l1((const int8_t*) w + 4672);
      xnn_prefetch_to_l1((const int8_t*) w + 4736);
      xnn_prefetch_to_l1((const int8_t*) w + 4800);
      xnn_prefetch_to_l1((const int8_t*) w + 4864);
      xnn_prefetch_to_l1((const int8_t*) w + 4928);
      xnn_prefetch_to_l1((const int8_t*) w + 4992);
      xnn_prefetch_to_l1((const int8_t*) w + 5056);
      xnn_prefetch_to_l1((const int8_t*) w + 5120);
      xnn_prefetch_to_l1((const int8_t*) w + 5184);
      xnn_prefetch_to_l1((const int8_t*) w + 5248);
      xnn_prefetch_to_l1((const int8_t*) w + 5312);
      xnn_prefetch_to_l1((const int8_t*) w + 5376);
      xnn_prefetch_to_l1((const int8_t*) w + 5440);
      xnn_prefetch_to_l1((const int8_t*) w + 5504);
      xnn_prefetch_to_l1((const int8_t*) w + 5568);
      xnn_prefetch_to_l1((const int8_t*) w + 5632);
      xnn_prefetch_to_l1((const int8_t*) w + 5696);
      xnn_prefetch_to_l1((const int8_t*) w + 5760);
      xnn_prefetch_to_l1((const int8_t*) w + 5824);
      xnn_prefetch_to_l1((const int8_t*) w + 5888);
      xnn_prefetch_to_l1((const int8_t*) w + 5952);
      xnn_prefetch_to_l1((const int8_t*) w + 6016);
      xnn_prefetch_to_l1((const int8_t*) w + 6080);
      xnn_prefetch_to_l1((const int8_t*) w + 6144);
      xnn_prefetch_to_l1((const int8_t*) w + 6208);
      xnn_prefetch_to_l1((const int8_t*) w + 6272);
      xnn_prefetch_to_l1((const int8_t*) w + 6336);
      xnn_prefetch_to_l1((const int8_t*) w + 6400);
      xnn_prefetch_to_l1((const int8_t*) w + 6464);
      xnn_prefetch_to_l1((const int8_t*) w + 6528);
      xnn_prefetch_to_l1((const int8_t*) w + 6592);
      xnn_prefetch_to_l1((const int8_t*) w + 6656);
      xnn_prefetch_to_l1((const int8_t*) w + 6720);
      xnn_prefetch_to_l1((const int8_t*) w + 6784);
      xnn_prefetch_to_l1((const int8_t*) w + 6848);
      xnn_prefetch_to_l1((const int8_t*) w + 6912);
      xnn_prefetch_to_l1((const int8_t*) w + 6976);
      xnn_prefetch_to_l1((const int8_t*) w + 7040);
      xnn_prefetch_to_l1((const int8_t*) w + 7104);
      xnn_prefetch_to_l1((const int8_t*) w + 7168);
      xnn_prefetch_to_l1((const int8_t*) w + 7232);
      xnn_prefetch_to_l1((const int8_t*) w + 7296);
      xnn_prefetch_to_l1((const int8_t*) w + 7360);
      xnn_prefetch_to_l1((const int8_t*) w + 7424);
      xnn_prefetch_to_l1((const int8_t*) w + 7488);
      xnn_prefetch_to_l1((const int8_t*) w + 7552);
      xnn_prefetch_to_l1((const int8_t*) w + 7616);
      xnn_prefetch_to_l1((const int8_t*) w + 7680);
      xnn_prefetch_to_l1((const int8_t*) w + 7744);
      xnn_prefetch_to_l1((const int8_t*) w + 7808);
      xnn_prefetch_to_l1((const int8_t*) w + 7872);
      xnn_prefetch_to_l1((const int8_t*) w + 7936);
      xnn_prefetch_to_l1((const int8_t*) w + 8000);
      xnn_prefetch_to_l1((const int8_t*) w + 8064);
      xnn_prefetch_to_l1((const int8_t*) w + 8128);
      xnn_prefetch_to_l1((const int8_t*) w + 8192);
      xnn_prefetch_to_l1((const int8_t*) w + 8256);
      xnn_prefetch_to_l1((const int8_t*) w + 8320);
      xnn_prefetch_to_l1((const int8_t*) w + 8384);
      xnn_prefetch_to_l1((const int8_t*) w + 8448);
      xnn_prefetch_to_l1((const int8_t*) w + 8512);
      xnn_prefetch_to_l1((const int8_t*) w + 8576);
      xnn_prefetch_to_l1((const int8_t*) w + 8640);
      xnn_prefetch_to_l1((const int8_t*) w + 8704);
      xnn_prefetch_to_l1((const int8_t*) w + 8768);
      xnn_prefetch_to_l1((const int8_t*) w + 8832);
      xnn_prefetch_to_l1((const int8_t*) w + 8896);
      xnn_prefetch_to_l1((const int8_t*) w + 8960);
      xnn_prefetch_to_l1((const int8_t*) w + 9024);
      xnn_prefetch_to_l1((const int8_t*) w + 9088);
      xnn_prefetch_to_l1((const int8_t*) w + 9152);

      w = (const int8_t*) w + 2048;
      k -= 64 * sizeof(int8_t);
    }

    if XNN_UNLIKELY(k != 0) {
      _tile_loadd(6, a, a_stride);
      a += kremainder;
      for (size_t k = 0; k < ((kremainder + 7) >> 3); ++k) {
        const __m512i vb0123456789ABCDEF = _mm512_load_epi32((const int8_t*) w + 0 + 256 * k);
        const __m512i vl0123456789ABCDEF = _mm512_gf2p8affine_epi64_epi8(vb0123456789ABCDEF, vshl4, 0);
        const __m512i vh0123456789ABCDEF = _mm512_and_si512(vb0123456789ABCDEF, vmask);
        _mm512_store_epi32(weight_buffer + 128 * k + 0, vl0123456789ABCDEF);
        _mm512_store_epi32(weight_buffer + 128 * k + 64, vh0123456789ABCDEF);
      }
      _tile_loadd(7, weight_buffer, 64);
      _tile_dpbssd(0, 6, 7);
      for (size_t k = 0; k < ((kremainder + 7) >> 3); ++k) {
        const __m512i vbGHIJKLMNOPQRSTUV = _mm512_load_epi32((const int8_t*) w + 64 + 256 * k);
        const __m512i vlGHIJKLMNOPQRSTUV = _mm512_gf2p8affine_epi64_epi8(vbGHIJKLMNOPQRSTUV, vshl4, 0);
        const __m512i vhGHIJKLMNOPQRSTUV = _mm512_and_si512(vbGHIJKLMNOPQRSTUV, vmask);
        _mm512_store_epi32(weight_buffer + 128 * k + 0, vlGHIJKLMNOPQRSTUV);
        _mm512_store_epi32(weight_buffer + 128 * k + 64, vhGHIJKLMNOPQRSTUV);
      }
      _tile_loadd(7, weight_buffer, 64);
      _tile_dpbssd(1, 6, 7);
      for (size_t k = 0; k < ((kremainder + 7) >> 3); ++k) {
        const __m512i vbWXYZabcdefghijkl = _mm512_load_epi32((const int8_t*) w + 128 + 256 * k);
        const __m512i vlWXYZabcdefghijkl = _mm512_gf2p8affine_epi64_epi8(vbWXYZabcdefghijkl, vshl4, 0);
        const __m512i vhWXYZabcdefghijkl = _mm512_and_si512(vbWXYZabcdefghijkl, vmask);
        _mm512_store_epi32(weight_buffer + 128 * k + 0, vlWXYZabcdefghijkl);
        _mm512_store_epi32(weight_buffer + 128 * k + 64, vhWXYZabcdefghijkl);
      }
      _tile_loadd(7, weight_buffer, 64);
      _tile_dpbssd(2, 6, 7);
      for (size_t k = 0; k < ((kremainder + 7) >> 3); ++k) {
        const __m512i vbmnopqrstuvwxyz01 = _mm512_load_epi32((const int8_t*) w + 192 + 256 * k);
        const __m512i vlmnopqrstuvwxyz01 = _mm512_gf2p8affine_epi64_epi8(vbmnopqrstuvwxyz01, vshl4, 0);
        const __m512i vhmnopqrstuvwxyz01 = _mm512_and_si512(vbmnopqrstuvwxyz01, vmask);
        _mm512_store_epi32(weight_buffer + 128 * k + 0, vlmnopqrstuvwxyz01);
        _mm512_store_epi32(weight_buffer + 128 * k + 64, vhmnopqrstuvwxyz01);
      }
      _tile_loadd(7, weight_buffer, 64);
      _tile_dpbssd(3, 6, 7);

      w = (const int8_t*) w + ((kremainder + 7) >> 3) * 256;
      k -= kremainder * sizeof(int8_t);
    }

    // Add tile to bias
    _tile_stored(0, res0, 64);
    _tile_stored(1, res1, 64);
    _tile_stored(2, res2, 64);
    _tile_stored(3, res3, 64);

    __m512i vacc0x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params[0].zero_point));
    __m512i vacc0xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params[0].zero_point));
    __m512i vacc0xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params[0].zero_point));
    __m512i vacc0xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params[0].zero_point));
    __m512i vacc1x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params[1].zero_point));
    __m512i vacc1xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params[1].zero_point));
    __m512i vacc1xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params[1].zero_point));
    __m512i vacc1xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params[1].zero_point));
    __m512i vacc2x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params[2].zero_point));
    __m512i vacc2xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params[2].zero_point));
    __m512i vacc2xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params[2].zero_point));
    __m512i vacc2xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params[2].zero_point));
    __m512i vacc3x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params[3].zero_point));
    __m512i vacc3xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params[3].zero_point));
    __m512i vacc3xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params[3].zero_point));
    __m512i vacc3xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params[3].zero_point));
    __m512i vacc4x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params[4].zero_point));
    __m512i vacc4xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params[4].zero_point));
    __m512i vacc4xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params[4].zero_point));
    __m512i vacc4xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params[4].zero_point));
    __m512i vacc5x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params[5].zero_point));
    __m512i vacc5xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params[5].zero_point));
    __m512i vacc5xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params[5].zero_point));
    __m512i vacc5xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params[5].zero_point));
    __m512i vacc6x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params[6].zero_point));
    __m512i vacc6xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params[6].zero_point));
    __m512i vacc6xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params[6].zero_point));
    __m512i vacc6xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params[6].zero_point));
    __m512i vacc7x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params[7].zero_point));
    __m512i vacc7xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params[7].zero_point));
    __m512i vacc7xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params[7].zero_point));
    __m512i vacc7xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params[7].zero_point));
    __m512i vacc8x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params[8].zero_point));
    __m512i vacc8xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params[8].zero_point));
    __m512i vacc8xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params[8].zero_point));
    __m512i vacc8xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params[8].zero_point));
    __m512i vacc9x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params[9].zero_point));
    __m512i vacc9xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params[9].zero_point));
    __m512i vacc9xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params[9].zero_point));
    __m512i vacc9xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params[9].zero_point));
    __m512i vacc10x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params[10].zero_point));
    __m512i vacc10xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params[10].zero_point));
    __m512i vacc10xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params[10].zero_point));
    __m512i vacc10xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params[10].zero_point));
    __m512i vacc11x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params[11].zero_point));
    __m512i vacc11xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params[11].zero_point));
    __m512i vacc11xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params[11].zero_point));
    __m512i vacc11xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params[11].zero_point));
    __m512i vacc12x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params[12].zero_point));
    __m512i vacc12xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params[12].zero_point));
    __m512i vacc12xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params[12].zero_point));
    __m512i vacc12xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params[12].zero_point));
    __m512i vacc13x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params[13].zero_point));
    __m512i vacc13xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params[13].zero_point));
    __m512i vacc13xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params[13].zero_point));
    __m512i vacc13xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params[13].zero_point));
    __m512i vacc14x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params[14].zero_point));
    __m512i vacc14xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params[14].zero_point));
    __m512i vacc14xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params[14].zero_point));
    __m512i vacc14xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params[14].zero_point));
    __m512i vacc15x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params[15].zero_point));
    __m512i vacc15xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params[15].zero_point));
    __m512i vacc15xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params[15].zero_point));
    __m512i vacc15xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params[15].zero_point));
    vacc0x0123456789ABCDEF = _mm512_add_epi32(vacc0x0123456789ABCDEF, _mm512_load_epi32(res0 + 0));
    vacc0xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc0xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 0));
    vacc0xWXYZabcdefghijkl = _mm512_add_epi32(vacc0xWXYZabcdefghijkl, _mm512_load_epi32(res2 + 0));
    vacc0xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc0xmnopqrstuvwxyz01, _mm512_load_epi32(res3 + 0));
    vacc1x0123456789ABCDEF = _mm512_add_epi32(vacc1x0123456789ABCDEF, _mm512_load_epi32(res0 + 16));
    vacc1xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc1xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 16));
    vacc1xWXYZabcdefghijkl = _mm512_add_epi32(vacc1xWXYZabcdefghijkl, _mm512_load_epi32(res2 + 16));
    vacc1xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc1xmnopqrstuvwxyz01, _mm512_load_epi32(res3 + 16));
    vacc2x0123456789ABCDEF = _mm512_add_epi32(vacc2x0123456789ABCDEF, _mm512_load_epi32(res0 + 32));
    vacc2xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc2xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 32));
    vacc2xWXYZabcdefghijkl = _mm512_add_epi32(vacc2xWXYZabcdefghijkl, _mm512_load_epi32(res2 + 32));
    vacc2xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc2xmnopqrstuvwxyz01, _mm512_load_epi32(res3 + 32));
    vacc3x0123456789ABCDEF = _mm512_add_epi32(vacc3x0123456789ABCDEF, _mm512_load_epi32(res0 + 48));
    vacc3xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc3xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 48));
    vacc3xWXYZabcdefghijkl = _mm512_add_epi32(vacc3xWXYZabcdefghijkl, _mm512_load_epi32(res2 + 48));
    vacc3xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc3xmnopqrstuvwxyz01, _mm512_load_epi32(res3 + 48));
    vacc4x0123456789ABCDEF = _mm512_add_epi32(vacc4x0123456789ABCDEF, _mm512_load_epi32(res0 + 64));
    vacc4xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc4xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 64));
    vacc4xWXYZabcdefghijkl = _mm512_add_epi32(vacc4xWXYZabcdefghijkl, _mm512_load_epi32(res2 + 64));
    vacc4xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc4xmnopqrstuvwxyz01, _mm512_load_epi32(res3 + 64));
    vacc5x0123456789ABCDEF = _mm512_add_epi32(vacc5x0123456789ABCDEF, _mm512_load_epi32(res0 + 80));
    vacc5xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc5xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 80));
    vacc5xWXYZabcdefghijkl = _mm512_add_epi32(vacc5xWXYZabcdefghijkl, _mm512_load_epi32(res2 + 80));
    vacc5xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc5xmnopqrstuvwxyz01, _mm512_load_epi32(res3 + 80));
    vacc6x0123456789ABCDEF = _mm512_add_epi32(vacc6x0123456789ABCDEF, _mm512_load_epi32(res0 + 96));
    vacc6xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc6xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 96));
    vacc6xWXYZabcdefghijkl = _mm512_add_epi32(vacc6xWXYZabcdefghijkl, _mm512_load_epi32(res2 + 96));
    vacc6xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc6xmnopqrstuvwxyz01, _mm512_load_epi32(res3 + 96));
    vacc7x0123456789ABCDEF = _mm512_add_epi32(vacc7x0123456789ABCDEF, _mm512_load_epi32(res0 + 112));
    vacc7xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc7xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 112));
    vacc7xWXYZabcdefghijkl = _mm512_add_epi32(vacc7xWXYZabcdefghijkl, _mm512_load_epi32(res2 + 112));
    vacc7xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc7xmnopqrstuvwxyz01, _mm512_load_epi32(res3 + 112));
    vacc8x0123456789ABCDEF = _mm512_add_epi32(vacc8x0123456789ABCDEF, _mm512_load_epi32(res0 + 128));
    vacc8xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc8xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 128));
    vacc8xWXYZabcdefghijkl = _mm512_add_epi32(vacc8xWXYZabcdefghijkl, _mm512_load_epi32(res2 + 128));
    vacc8xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc8xmnopqrstuvwxyz01, _mm512_load_epi32(res3 + 128));
    vacc9x0123456789ABCDEF = _mm512_add_epi32(vacc9x0123456789ABCDEF, _mm512_load_epi32(res0 + 144));
    vacc9xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc9xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 144));
    vacc9xWXYZabcdefghijkl = _mm512_add_epi32(vacc9xWXYZabcdefghijkl, _mm512_load_epi32(res2 + 144));
    vacc9xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc9xmnopqrstuvwxyz01, _mm512_load_epi32(res3 + 144));
    vacc10x0123456789ABCDEF = _mm512_add_epi32(vacc10x0123456789ABCDEF, _mm512_load_epi32(res0 + 160));
    vacc10xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc10xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 160));
    vacc10xWXYZabcdefghijkl = _mm512_add_epi32(vacc10xWXYZabcdefghijkl, _mm512_load_epi32(res2 + 160));
    vacc10xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc10xmnopqrstuvwxyz01, _mm512_load_epi32(res3 + 160));
    vacc11x0123456789ABCDEF = _mm512_add_epi32(vacc11x0123456789ABCDEF, _mm512_load_epi32(res0 + 176));
    vacc11xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc11xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 176));
    vacc11xWXYZabcdefghijkl = _mm512_add_epi32(vacc11xWXYZabcdefghijkl, _mm512_load_epi32(res2 + 176));
    vacc11xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc11xmnopqrstuvwxyz01, _mm512_load_epi32(res3 + 176));
    vacc12x0123456789ABCDEF = _mm512_add_epi32(vacc12x0123456789ABCDEF, _mm512_load_epi32(res0 + 192));
    vacc12xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc12xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 192));
    vacc12xWXYZabcdefghijkl = _mm512_add_epi32(vacc12xWXYZabcdefghijkl, _mm512_load_epi32(res2 + 192));
    vacc12xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc12xmnopqrstuvwxyz01, _mm512_load_epi32(res3 + 192));
    vacc13x0123456789ABCDEF = _mm512_add_epi32(vacc13x0123456789ABCDEF, _mm512_load_epi32(res0 + 208));
    vacc13xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc13xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 208));
    vacc13xWXYZabcdefghijkl = _mm512_add_epi32(vacc13xWXYZabcdefghijkl, _mm512_load_epi32(res2 + 208));
    vacc13xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc13xmnopqrstuvwxyz01, _mm512_load_epi32(res3 + 208));
    vacc14x0123456789ABCDEF = _mm512_add_epi32(vacc14x0123456789ABCDEF, _mm512_load_epi32(res0 + 224));
    vacc14xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc14xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 224));
    vacc14xWXYZabcdefghijkl = _mm512_add_epi32(vacc14xWXYZabcdefghijkl, _mm512_load_epi32(res2 + 224));
    vacc14xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc14xmnopqrstuvwxyz01, _mm512_load_epi32(res3 + 224));
    vacc15x0123456789ABCDEF = _mm512_add_epi32(vacc15x0123456789ABCDEF, _mm512_load_epi32(res0 + 240));
    vacc15xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc15xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 240));
    vacc15xWXYZabcdefghijkl = _mm512_add_epi32(vacc15xWXYZabcdefghijkl, _mm512_load_epi32(res2 + 240));
    vacc15xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc15xmnopqrstuvwxyz01, _mm512_load_epi32(res3 + 240));

    vacc0x0123456789ABCDEF = _mm512_srai_epi32(vacc0x0123456789ABCDEF, 4);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_srai_epi32(vacc0xGHIJKLMNOPQRSTUV, 4);
    vacc0xWXYZabcdefghijkl = _mm512_srai_epi32(vacc0xWXYZabcdefghijkl, 4);
    vacc0xmnopqrstuvwxyz01 = _mm512_srai_epi32(vacc0xmnopqrstuvwxyz01, 4);
    vacc1x0123456789ABCDEF = _mm512_srai_epi32(vacc1x0123456789ABCDEF, 4);
    vacc1xGHIJKLMNOPQRSTUV = _mm512_srai_epi32(vacc1xGHIJKLMNOPQRSTUV, 4);
    vacc1xWXYZabcdefghijkl = _mm512_srai_epi32(vacc1xWXYZabcdefghijkl, 4);
    vacc1xmnopqrstuvwxyz01 = _mm512_srai_epi32(vacc1xmnopqrstuvwxyz01, 4);
    vacc2x0123456789ABCDEF = _mm512_srai_epi32(vacc2x0123456789ABCDEF, 4);
    vacc2xGHIJKLMNOPQRSTUV = _mm512_srai_epi32(vacc2xGHIJKLMNOPQRSTUV, 4);
    vacc2xWXYZabcdefghijkl = _mm512_srai_epi32(vacc2xWXYZabcdefghijkl, 4);
    vacc2xmnopqrstuvwxyz01 = _mm512_srai_epi32(vacc2xmnopqrstuvwxyz01, 4);
    vacc3x0123456789ABCDEF = _mm512_srai_epi32(vacc3x0123456789ABCDEF, 4);
    vacc3xGHIJKLMNOPQRSTUV = _mm512_srai_epi32(vacc3xGHIJKLMNOPQRSTUV, 4);
    vacc3xWXYZabcdefghijkl = _mm512_srai_epi32(vacc3xWXYZabcdefghijkl, 4);
    vacc3xmnopqrstuvwxyz01 = _mm512_srai_epi32(vacc3xmnopqrstuvwxyz01, 4);
    vacc4x0123456789ABCDEF = _mm512_srai_epi32(vacc4x0123456789ABCDEF, 4);
    vacc4xGHIJKLMNOPQRSTUV = _mm512_srai_epi32(vacc4xGHIJKLMNOPQRSTUV, 4);
    vacc4xWXYZabcdefghijkl = _mm512_srai_epi32(vacc4xWXYZabcdefghijkl, 4);
    vacc4xmnopqrstuvwxyz01 = _mm512_srai_epi32(vacc4xmnopqrstuvwxyz01, 4);
    vacc5x0123456789ABCDEF = _mm512_srai_epi32(vacc5x0123456789ABCDEF, 4);
    vacc5xGHIJKLMNOPQRSTUV = _mm512_srai_epi32(vacc5xGHIJKLMNOPQRSTUV, 4);
    vacc5xWXYZabcdefghijkl = _mm512_srai_epi32(vacc5xWXYZabcdefghijkl, 4);
    vacc5xmnopqrstuvwxyz01 = _mm512_srai_epi32(vacc5xmnopqrstuvwxyz01, 4);
    vacc6x0123456789ABCDEF = _mm512_srai_epi32(vacc6x0123456789ABCDEF, 4);
    vacc6xGHIJKLMNOPQRSTUV = _mm512_srai_epi32(vacc6xGHIJKLMNOPQRSTUV, 4);
    vacc6xWXYZabcdefghijkl = _mm512_srai_epi32(vacc6xWXYZabcdefghijkl, 4);
    vacc6xmnopqrstuvwxyz01 = _mm512_srai_epi32(vacc6xmnopqrstuvwxyz01, 4);
    vacc7x0123456789ABCDEF = _mm512_srai_epi32(vacc7x0123456789ABCDEF, 4);
    vacc7xGHIJKLMNOPQRSTUV = _mm512_srai_epi32(vacc7xGHIJKLMNOPQRSTUV, 4);
    vacc7xWXYZabcdefghijkl = _mm512_srai_epi32(vacc7xWXYZabcdefghijkl, 4);
    vacc7xmnopqrstuvwxyz01 = _mm512_srai_epi32(vacc7xmnopqrstuvwxyz01, 4);
    vacc8x0123456789ABCDEF = _mm512_srai_epi32(vacc8x0123456789ABCDEF, 4);
    vacc8xGHIJKLMNOPQRSTUV = _mm512_srai_epi32(vacc8xGHIJKLMNOPQRSTUV, 4);
    vacc8xWXYZabcdefghijkl = _mm512_srai_epi32(vacc8xWXYZabcdefghijkl, 4);
    vacc8xmnopqrstuvwxyz01 = _mm512_srai_epi32(vacc8xmnopqrstuvwxyz01, 4);
    vacc9x0123456789ABCDEF = _mm512_srai_epi32(vacc9x0123456789ABCDEF, 4);
    vacc9xGHIJKLMNOPQRSTUV = _mm512_srai_epi32(vacc9xGHIJKLMNOPQRSTUV, 4);
    vacc9xWXYZabcdefghijkl = _mm512_srai_epi32(vacc9xWXYZabcdefghijkl, 4);
    vacc9xmnopqrstuvwxyz01 = _mm512_srai_epi32(vacc9xmnopqrstuvwxyz01, 4);
    vacc10x0123456789ABCDEF = _mm512_srai_epi32(vacc10x0123456789ABCDEF, 4);
    vacc10xGHIJKLMNOPQRSTUV = _mm512_srai_epi32(vacc10xGHIJKLMNOPQRSTUV, 4);
    vacc10xWXYZabcdefghijkl = _mm512_srai_epi32(vacc10xWXYZabcdefghijkl, 4);
    vacc10xmnopqrstuvwxyz01 = _mm512_srai_epi32(vacc10xmnopqrstuvwxyz01, 4);
    vacc11x0123456789ABCDEF = _mm512_srai_epi32(vacc11x0123456789ABCDEF, 4);
    vacc11xGHIJKLMNOPQRSTUV = _mm512_srai_epi32(vacc11xGHIJKLMNOPQRSTUV, 4);
    vacc11xWXYZabcdefghijkl = _mm512_srai_epi32(vacc11xWXYZabcdefghijkl, 4);
    vacc11xmnopqrstuvwxyz01 = _mm512_srai_epi32(vacc11xmnopqrstuvwxyz01, 4);
    vacc12x0123456789ABCDEF = _mm512_srai_epi32(vacc12x0123456789ABCDEF, 4);
    vacc12xGHIJKLMNOPQRSTUV = _mm512_srai_epi32(vacc12xGHIJKLMNOPQRSTUV, 4);
    vacc12xWXYZabcdefghijkl = _mm512_srai_epi32(vacc12xWXYZabcdefghijkl, 4);
    vacc12xmnopqrstuvwxyz01 = _mm512_srai_epi32(vacc12xmnopqrstuvwxyz01, 4);
    vacc13x0123456789ABCDEF = _mm512_srai_epi32(vacc13x0123456789ABCDEF, 4);
    vacc13xGHIJKLMNOPQRSTUV = _mm512_srai_epi32(vacc13xGHIJKLMNOPQRSTUV, 4);
    vacc13xWXYZabcdefghijkl = _mm512_srai_epi32(vacc13xWXYZabcdefghijkl, 4);
    vacc13xmnopqrstuvwxyz01 = _mm512_srai_epi32(vacc13xmnopqrstuvwxyz01, 4);
    vacc14x0123456789ABCDEF = _mm512_srai_epi32(vacc14x0123456789ABCDEF, 4);
    vacc14xGHIJKLMNOPQRSTUV = _mm512_srai_epi32(vacc14xGHIJKLMNOPQRSTUV, 4);
    vacc14xWXYZabcdefghijkl = _mm512_srai_epi32(vacc14xWXYZabcdefghijkl, 4);
    vacc14xmnopqrstuvwxyz01 = _mm512_srai_epi32(vacc14xmnopqrstuvwxyz01, 4);
    vacc15x0123456789ABCDEF = _mm512_srai_epi32(vacc15x0123456789ABCDEF, 4);
    vacc15xGHIJKLMNOPQRSTUV = _mm512_srai_epi32(vacc15xGHIJKLMNOPQRSTUV, 4);
    vacc15xWXYZabcdefghijkl = _mm512_srai_epi32(vacc15xWXYZabcdefghijkl, 4);
    vacc15xmnopqrstuvwxyz01 = _mm512_srai_epi32(vacc15xmnopqrstuvwxyz01, 4);
    __m512 vscaled0x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0x0123456789ABCDEF);
    __m512 vscaled0xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc0xGHIJKLMNOPQRSTUV);
    __m512 vscaled0xWXYZabcdefghijkl = _mm512_cvtepi32_ps(vacc0xWXYZabcdefghijkl);
    __m512 vscaled0xmnopqrstuvwxyz01 = _mm512_cvtepi32_ps(vacc0xmnopqrstuvwxyz01);
    __m512 vscaled1x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc1x0123456789ABCDEF);
    __m512 vscaled1xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc1xGHIJKLMNOPQRSTUV);
    __m512 vscaled1xWXYZabcdefghijkl = _mm512_cvtepi32_ps(vacc1xWXYZabcdefghijkl);
    __m512 vscaled1xmnopqrstuvwxyz01 = _mm512_cvtepi32_ps(vacc1xmnopqrstuvwxyz01);
    __m512 vscaled2x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc2x0123456789ABCDEF);
    __m512 vscaled2xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc2xGHIJKLMNOPQRSTUV);
    __m512 vscaled2xWXYZabcdefghijkl = _mm512_cvtepi32_ps(vacc2xWXYZabcdefghijkl);
    __m512 vscaled2xmnopqrstuvwxyz01 = _mm512_cvtepi32_ps(vacc2xmnopqrstuvwxyz01);
    __m512 vscaled3x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc3x0123456789ABCDEF);
    __m512 vscaled3xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc3xGHIJKLMNOPQRSTUV);
    __m512 vscaled3xWXYZabcdefghijkl = _mm512_cvtepi32_ps(vacc3xWXYZabcdefghijkl);
    __m512 vscaled3xmnopqrstuvwxyz01 = _mm512_cvtepi32_ps(vacc3xmnopqrstuvwxyz01);
    __m512 vscaled4x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc4x0123456789ABCDEF);
    __m512 vscaled4xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc4xGHIJKLMNOPQRSTUV);
    __m512 vscaled4xWXYZabcdefghijkl = _mm512_cvtepi32_ps(vacc4xWXYZabcdefghijkl);
    __m512 vscaled4xmnopqrstuvwxyz01 = _mm512_cvtepi32_ps(vacc4xmnopqrstuvwxyz01);
    __m512 vscaled5x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc5x0123456789ABCDEF);
    __m512 vscaled5xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc5xGHIJKLMNOPQRSTUV);
    __m512 vscaled5xWXYZabcdefghijkl = _mm512_cvtepi32_ps(vacc5xWXYZabcdefghijkl);
    __m512 vscaled5xmnopqrstuvwxyz01 = _mm512_cvtepi32_ps(vacc5xmnopqrstuvwxyz01);
    __m512 vscaled6x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc6x0123456789ABCDEF);
    __m512 vscaled6xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc6xGHIJKLMNOPQRSTUV);
    __m512 vscaled6xWXYZabcdefghijkl = _mm512_cvtepi32_ps(vacc6xWXYZabcdefghijkl);
    __m512 vscaled6xmnopqrstuvwxyz01 = _mm512_cvtepi32_ps(vacc6xmnopqrstuvwxyz01);
    __m512 vscaled7x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc7x0123456789ABCDEF);
    __m512 vscaled7xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc7xGHIJKLMNOPQRSTUV);
    __m512 vscaled7xWXYZabcdefghijkl = _mm512_cvtepi32_ps(vacc7xWXYZabcdefghijkl);
    __m512 vscaled7xmnopqrstuvwxyz01 = _mm512_cvtepi32_ps(vacc7xmnopqrstuvwxyz01);
    __m512 vscaled8x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc8x0123456789ABCDEF);
    __m512 vscaled8xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc8xGHIJKLMNOPQRSTUV);
    __m512 vscaled8xWXYZabcdefghijkl = _mm512_cvtepi32_ps(vacc8xWXYZabcdefghijkl);
    __m512 vscaled8xmnopqrstuvwxyz01 = _mm512_cvtepi32_ps(vacc8xmnopqrstuvwxyz01);
    __m512 vscaled9x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc9x0123456789ABCDEF);
    __m512 vscaled9xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc9xGHIJKLMNOPQRSTUV);
    __m512 vscaled9xWXYZabcdefghijkl = _mm512_cvtepi32_ps(vacc9xWXYZabcdefghijkl);
    __m512 vscaled9xmnopqrstuvwxyz01 = _mm512_cvtepi32_ps(vacc9xmnopqrstuvwxyz01);
    __m512 vscaled10x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc10x0123456789ABCDEF);
    __m512 vscaled10xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc10xGHIJKLMNOPQRSTUV);
    __m512 vscaled10xWXYZabcdefghijkl = _mm512_cvtepi32_ps(vacc10xWXYZabcdefghijkl);
    __m512 vscaled10xmnopqrstuvwxyz01 = _mm512_cvtepi32_ps(vacc10xmnopqrstuvwxyz01);
    __m512 vscaled11x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc11x0123456789ABCDEF);
    __m512 vscaled11xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc11xGHIJKLMNOPQRSTUV);
    __m512 vscaled11xWXYZabcdefghijkl = _mm512_cvtepi32_ps(vacc11xWXYZabcdefghijkl);
    __m512 vscaled11xmnopqrstuvwxyz01 = _mm512_cvtepi32_ps(vacc11xmnopqrstuvwxyz01);
    __m512 vscaled12x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc12x0123456789ABCDEF);
    __m512 vscaled12xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc12xGHIJKLMNOPQRSTUV);
    __m512 vscaled12xWXYZabcdefghijkl = _mm512_cvtepi32_ps(vacc12xWXYZabcdefghijkl);
    __m512 vscaled12xmnopqrstuvwxyz01 = _mm512_cvtepi32_ps(vacc12xmnopqrstuvwxyz01);
    __m512 vscaled13x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc13x0123456789ABCDEF);
    __m512 vscaled13xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc13xGHIJKLMNOPQRSTUV);
    __m512 vscaled13xWXYZabcdefghijkl = _mm512_cvtepi32_ps(vacc13xWXYZabcdefghijkl);
    __m512 vscaled13xmnopqrstuvwxyz01 = _mm512_cvtepi32_ps(vacc13xmnopqrstuvwxyz01);
    __m512 vscaled14x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc14x0123456789ABCDEF);
    __m512 vscaled14xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc14xGHIJKLMNOPQRSTUV);
    __m512 vscaled14xWXYZabcdefghijkl = _mm512_cvtepi32_ps(vacc14xWXYZabcdefghijkl);
    __m512 vscaled14xmnopqrstuvwxyz01 = _mm512_cvtepi32_ps(vacc14xmnopqrstuvwxyz01);
    __m512 vscaled15x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc15x0123456789ABCDEF);
    __m512 vscaled15xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc15xGHIJKLMNOPQRSTUV);
    __m512 vscaled15xWXYZabcdefghijkl = _mm512_cvtepi32_ps(vacc15xWXYZabcdefghijkl);
    __m512 vscaled15xmnopqrstuvwxyz01 = _mm512_cvtepi32_ps(vacc15xmnopqrstuvwxyz01);

    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, _mm512_set1_ps(quantization_params[0].inv_scale));
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled0xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params[0].inv_scale));
    vscaled0xWXYZabcdefghijkl = _mm512_mul_ps(vscaled0xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params[0].inv_scale));
    vscaled0xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled0xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params[0].inv_scale));
    vscaled1x0123456789ABCDEF = _mm512_mul_ps(vscaled1x0123456789ABCDEF, _mm512_set1_ps(quantization_params[1].inv_scale));
    vscaled1xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled1xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params[1].inv_scale));
    vscaled1xWXYZabcdefghijkl = _mm512_mul_ps(vscaled1xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params[1].inv_scale));
    vscaled1xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled1xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params[1].inv_scale));
    vscaled2x0123456789ABCDEF = _mm512_mul_ps(vscaled2x0123456789ABCDEF, _mm512_set1_ps(quantization_params[2].inv_scale));
    vscaled2xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled2xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params[2].inv_scale));
    vscaled2xWXYZabcdefghijkl = _mm512_mul_ps(vscaled2xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params[2].inv_scale));
    vscaled2xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled2xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params[2].inv_scale));
    vscaled3x0123456789ABCDEF = _mm512_mul_ps(vscaled3x0123456789ABCDEF, _mm512_set1_ps(quantization_params[3].inv_scale));
    vscaled3xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled3xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params[3].inv_scale));
    vscaled3xWXYZabcdefghijkl = _mm512_mul_ps(vscaled3xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params[3].inv_scale));
    vscaled3xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled3xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params[3].inv_scale));
    vscaled4x0123456789ABCDEF = _mm512_mul_ps(vscaled4x0123456789ABCDEF, _mm512_set1_ps(quantization_params[4].inv_scale));
    vscaled4xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled4xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params[4].inv_scale));
    vscaled4xWXYZabcdefghijkl = _mm512_mul_ps(vscaled4xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params[4].inv_scale));
    vscaled4xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled4xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params[4].inv_scale));
    vscaled5x0123456789ABCDEF = _mm512_mul_ps(vscaled5x0123456789ABCDEF, _mm512_set1_ps(quantization_params[5].inv_scale));
    vscaled5xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled5xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params[5].inv_scale));
    vscaled5xWXYZabcdefghijkl = _mm512_mul_ps(vscaled5xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params[5].inv_scale));
    vscaled5xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled5xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params[5].inv_scale));
    vscaled6x0123456789ABCDEF = _mm512_mul_ps(vscaled6x0123456789ABCDEF, _mm512_set1_ps(quantization_params[6].inv_scale));
    vscaled6xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled6xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params[6].inv_scale));
    vscaled6xWXYZabcdefghijkl = _mm512_mul_ps(vscaled6xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params[6].inv_scale));
    vscaled6xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled6xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params[6].inv_scale));
    vscaled7x0123456789ABCDEF = _mm512_mul_ps(vscaled7x0123456789ABCDEF, _mm512_set1_ps(quantization_params[7].inv_scale));
    vscaled7xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled7xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params[7].inv_scale));
    vscaled7xWXYZabcdefghijkl = _mm512_mul_ps(vscaled7xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params[7].inv_scale));
    vscaled7xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled7xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params[7].inv_scale));
    vscaled8x0123456789ABCDEF = _mm512_mul_ps(vscaled8x0123456789ABCDEF, _mm512_set1_ps(quantization_params[8].inv_scale));
    vscaled8xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled8xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params[8].inv_scale));
    vscaled8xWXYZabcdefghijkl = _mm512_mul_ps(vscaled8xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params[8].inv_scale));
    vscaled8xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled8xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params[8].inv_scale));
    vscaled9x0123456789ABCDEF = _mm512_mul_ps(vscaled9x0123456789ABCDEF, _mm512_set1_ps(quantization_params[9].inv_scale));
    vscaled9xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled9xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params[9].inv_scale));
    vscaled9xWXYZabcdefghijkl = _mm512_mul_ps(vscaled9xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params[9].inv_scale));
    vscaled9xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled9xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params[9].inv_scale));
    vscaled10x0123456789ABCDEF = _mm512_mul_ps(vscaled10x0123456789ABCDEF, _mm512_set1_ps(quantization_params[10].inv_scale));
    vscaled10xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled10xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params[10].inv_scale));
    vscaled10xWXYZabcdefghijkl = _mm512_mul_ps(vscaled10xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params[10].inv_scale));
    vscaled10xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled10xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params[10].inv_scale));
    vscaled11x0123456789ABCDEF = _mm512_mul_ps(vscaled11x0123456789ABCDEF, _mm512_set1_ps(quantization_params[11].inv_scale));
    vscaled11xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled11xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params[11].inv_scale));
    vscaled11xWXYZabcdefghijkl = _mm512_mul_ps(vscaled11xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params[11].inv_scale));
    vscaled11xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled11xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params[11].inv_scale));
    vscaled12x0123456789ABCDEF = _mm512_mul_ps(vscaled12x0123456789ABCDEF, _mm512_set1_ps(quantization_params[12].inv_scale));
    vscaled12xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled12xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params[12].inv_scale));
    vscaled12xWXYZabcdefghijkl = _mm512_mul_ps(vscaled12xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params[12].inv_scale));
    vscaled12xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled12xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params[12].inv_scale));
    vscaled13x0123456789ABCDEF = _mm512_mul_ps(vscaled13x0123456789ABCDEF, _mm512_set1_ps(quantization_params[13].inv_scale));
    vscaled13xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled13xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params[13].inv_scale));
    vscaled13xWXYZabcdefghijkl = _mm512_mul_ps(vscaled13xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params[13].inv_scale));
    vscaled13xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled13xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params[13].inv_scale));
    vscaled14x0123456789ABCDEF = _mm512_mul_ps(vscaled14x0123456789ABCDEF, _mm512_set1_ps(quantization_params[14].inv_scale));
    vscaled14xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled14xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params[14].inv_scale));
    vscaled14xWXYZabcdefghijkl = _mm512_mul_ps(vscaled14xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params[14].inv_scale));
    vscaled14xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled14xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params[14].inv_scale));
    vscaled15x0123456789ABCDEF = _mm512_mul_ps(vscaled15x0123456789ABCDEF, _mm512_set1_ps(quantization_params[15].inv_scale));
    vscaled15xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled15xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params[15].inv_scale));
    vscaled15xWXYZabcdefghijkl = _mm512_mul_ps(vscaled15xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params[15].inv_scale));
    vscaled15xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled15xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params[15].inv_scale));

    const __m512 vfilter_output_scale0123456789ABCDEF = _mm512_load_ps((const float*) w + 0);
    const __m512 vfilter_output_scaleGHIJKLMNOPQRSTUV = _mm512_load_ps((const float*) w + 16);
    const __m512 vfilter_output_scaleWXYZabcdefghijkl = _mm512_load_ps((const float*) w + 32);
    const __m512 vfilter_output_scalemnopqrstuvwxyz01 = _mm512_load_ps((const float*) w + 48);
    w = (const int32_t*) w + 64;
    const __m512 vbias0123456789ABCDEF = _mm512_load_ps((const float*) w + 0);
    const __m512 vbiasGHIJKLMNOPQRSTUV = _mm512_load_ps((const float*) w + 16);
    const __m512 vbiasWXYZabcdefghijkl = _mm512_load_ps((const float*) w + 32);
    const __m512 vbiasmnopqrstuvwxyz01 = _mm512_load_ps((const float*) w + 48);
    w = (const int32_t*) w + 64;

    vscaled0x0123456789ABCDEF = _mm512_fmadd_ps(vscaled0x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled0xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled0xWXYZabcdefghijkl = _mm512_fmadd_ps(vscaled0xWXYZabcdefghijkl, vfilter_output_scaleWXYZabcdefghijkl, vbiasWXYZabcdefghijkl);
    vscaled0xmnopqrstuvwxyz01 = _mm512_fmadd_ps(vscaled0xmnopqrstuvwxyz01, vfilter_output_scalemnopqrstuvwxyz01, vbiasmnopqrstuvwxyz01);
    vscaled1x0123456789ABCDEF = _mm512_fmadd_ps(vscaled1x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled1xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled1xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled1xWXYZabcdefghijkl = _mm512_fmadd_ps(vscaled1xWXYZabcdefghijkl, vfilter_output_scaleWXYZabcdefghijkl, vbiasWXYZabcdefghijkl);
    vscaled1xmnopqrstuvwxyz01 = _mm512_fmadd_ps(vscaled1xmnopqrstuvwxyz01, vfilter_output_scalemnopqrstuvwxyz01, vbiasmnopqrstuvwxyz01);
    vscaled2x0123456789ABCDEF = _mm512_fmadd_ps(vscaled2x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled2xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled2xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled2xWXYZabcdefghijkl = _mm512_fmadd_ps(vscaled2xWXYZabcdefghijkl, vfilter_output_scaleWXYZabcdefghijkl, vbiasWXYZabcdefghijkl);
    vscaled2xmnopqrstuvwxyz01 = _mm512_fmadd_ps(vscaled2xmnopqrstuvwxyz01, vfilter_output_scalemnopqrstuvwxyz01, vbiasmnopqrstuvwxyz01);
    vscaled3x0123456789ABCDEF = _mm512_fmadd_ps(vscaled3x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled3xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled3xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled3xWXYZabcdefghijkl = _mm512_fmadd_ps(vscaled3xWXYZabcdefghijkl, vfilter_output_scaleWXYZabcdefghijkl, vbiasWXYZabcdefghijkl);
    vscaled3xmnopqrstuvwxyz01 = _mm512_fmadd_ps(vscaled3xmnopqrstuvwxyz01, vfilter_output_scalemnopqrstuvwxyz01, vbiasmnopqrstuvwxyz01);
    vscaled4x0123456789ABCDEF = _mm512_fmadd_ps(vscaled4x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled4xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled4xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled4xWXYZabcdefghijkl = _mm512_fmadd_ps(vscaled4xWXYZabcdefghijkl, vfilter_output_scaleWXYZabcdefghijkl, vbiasWXYZabcdefghijkl);
    vscaled4xmnopqrstuvwxyz01 = _mm512_fmadd_ps(vscaled4xmnopqrstuvwxyz01, vfilter_output_scalemnopqrstuvwxyz01, vbiasmnopqrstuvwxyz01);
    vscaled5x0123456789ABCDEF = _mm512_fmadd_ps(vscaled5x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled5xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled5xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled5xWXYZabcdefghijkl = _mm512_fmadd_ps(vscaled5xWXYZabcdefghijkl, vfilter_output_scaleWXYZabcdefghijkl, vbiasWXYZabcdefghijkl);
    vscaled5xmnopqrstuvwxyz01 = _mm512_fmadd_ps(vscaled5xmnopqrstuvwxyz01, vfilter_output_scalemnopqrstuvwxyz01, vbiasmnopqrstuvwxyz01);
    vscaled6x0123456789ABCDEF = _mm512_fmadd_ps(vscaled6x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled6xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled6xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled6xWXYZabcdefghijkl = _mm512_fmadd_ps(vscaled6xWXYZabcdefghijkl, vfilter_output_scaleWXYZabcdefghijkl, vbiasWXYZabcdefghijkl);
    vscaled6xmnopqrstuvwxyz01 = _mm512_fmadd_ps(vscaled6xmnopqrstuvwxyz01, vfilter_output_scalemnopqrstuvwxyz01, vbiasmnopqrstuvwxyz01);
    vscaled7x0123456789ABCDEF = _mm512_fmadd_ps(vscaled7x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled7xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled7xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled7xWXYZabcdefghijkl = _mm512_fmadd_ps(vscaled7xWXYZabcdefghijkl, vfilter_output_scaleWXYZabcdefghijkl, vbiasWXYZabcdefghijkl);
    vscaled7xmnopqrstuvwxyz01 = _mm512_fmadd_ps(vscaled7xmnopqrstuvwxyz01, vfilter_output_scalemnopqrstuvwxyz01, vbiasmnopqrstuvwxyz01);
    vscaled8x0123456789ABCDEF = _mm512_fmadd_ps(vscaled8x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled8xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled8xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled8xWXYZabcdefghijkl = _mm512_fmadd_ps(vscaled8xWXYZabcdefghijkl, vfilter_output_scaleWXYZabcdefghijkl, vbiasWXYZabcdefghijkl);
    vscaled8xmnopqrstuvwxyz01 = _mm512_fmadd_ps(vscaled8xmnopqrstuvwxyz01, vfilter_output_scalemnopqrstuvwxyz01, vbiasmnopqrstuvwxyz01);
    vscaled9x0123456789ABCDEF = _mm512_fmadd_ps(vscaled9x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled9xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled9xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled9xWXYZabcdefghijkl = _mm512_fmadd_ps(vscaled9xWXYZabcdefghijkl, vfilter_output_scaleWXYZabcdefghijkl, vbiasWXYZabcdefghijkl);
    vscaled9xmnopqrstuvwxyz01 = _mm512_fmadd_ps(vscaled9xmnopqrstuvwxyz01, vfilter_output_scalemnopqrstuvwxyz01, vbiasmnopqrstuvwxyz01);
    vscaled10x0123456789ABCDEF = _mm512_fmadd_ps(vscaled10x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled10xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled10xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled10xWXYZabcdefghijkl = _mm512_fmadd_ps(vscaled10xWXYZabcdefghijkl, vfilter_output_scaleWXYZabcdefghijkl, vbiasWXYZabcdefghijkl);
    vscaled10xmnopqrstuvwxyz01 = _mm512_fmadd_ps(vscaled10xmnopqrstuvwxyz01, vfilter_output_scalemnopqrstuvwxyz01, vbiasmnopqrstuvwxyz01);
    vscaled11x0123456789ABCDEF = _mm512_fmadd_ps(vscaled11x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled11xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled11xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled11xWXYZabcdefghijkl = _mm512_fmadd_ps(vscaled11xWXYZabcdefghijkl, vfilter_output_scaleWXYZabcdefghijkl, vbiasWXYZabcdefghijkl);
    vscaled11xmnopqrstuvwxyz01 = _mm512_fmadd_ps(vscaled11xmnopqrstuvwxyz01, vfilter_output_scalemnopqrstuvwxyz01, vbiasmnopqrstuvwxyz01);
    vscaled12x0123456789ABCDEF = _mm512_fmadd_ps(vscaled12x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled12xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled12xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled12xWXYZabcdefghijkl = _mm512_fmadd_ps(vscaled12xWXYZabcdefghijkl, vfilter_output_scaleWXYZabcdefghijkl, vbiasWXYZabcdefghijkl);
    vscaled12xmnopqrstuvwxyz01 = _mm512_fmadd_ps(vscaled12xmnopqrstuvwxyz01, vfilter_output_scalemnopqrstuvwxyz01, vbiasmnopqrstuvwxyz01);
    vscaled13x0123456789ABCDEF = _mm512_fmadd_ps(vscaled13x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled13xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled13xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled13xWXYZabcdefghijkl = _mm512_fmadd_ps(vscaled13xWXYZabcdefghijkl, vfilter_output_scaleWXYZabcdefghijkl, vbiasWXYZabcdefghijkl);
    vscaled13xmnopqrstuvwxyz01 = _mm512_fmadd_ps(vscaled13xmnopqrstuvwxyz01, vfilter_output_scalemnopqrstuvwxyz01, vbiasmnopqrstuvwxyz01);
    vscaled14x0123456789ABCDEF = _mm512_fmadd_ps(vscaled14x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled14xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled14xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled14xWXYZabcdefghijkl = _mm512_fmadd_ps(vscaled14xWXYZabcdefghijkl, vfilter_output_scaleWXYZabcdefghijkl, vbiasWXYZabcdefghijkl);
    vscaled14xmnopqrstuvwxyz01 = _mm512_fmadd_ps(vscaled14xmnopqrstuvwxyz01, vfilter_output_scalemnopqrstuvwxyz01, vbiasmnopqrstuvwxyz01);
    vscaled15x0123456789ABCDEF = _mm512_fmadd_ps(vscaled15x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled15xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled15xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled15xWXYZabcdefghijkl = _mm512_fmadd_ps(vscaled15xWXYZabcdefghijkl, vfilter_output_scaleWXYZabcdefghijkl, vbiasWXYZabcdefghijkl);
    vscaled15xmnopqrstuvwxyz01 = _mm512_fmadd_ps(vscaled15xmnopqrstuvwxyz01, vfilter_output_scalemnopqrstuvwxyz01, vbiasmnopqrstuvwxyz01);

    vscaled0x0123456789ABCDEF = _mm512_max_ps(vscaled0x0123456789ABCDEF, voutput_min);
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled0xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled0xWXYZabcdefghijkl = _mm512_max_ps(vscaled0xWXYZabcdefghijkl, voutput_min);
    vscaled0xmnopqrstuvwxyz01 = _mm512_max_ps(vscaled0xmnopqrstuvwxyz01, voutput_min);
    vscaled1x0123456789ABCDEF = _mm512_max_ps(vscaled1x0123456789ABCDEF, voutput_min);
    vscaled1xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled1xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled1xWXYZabcdefghijkl = _mm512_max_ps(vscaled1xWXYZabcdefghijkl, voutput_min);
    vscaled1xmnopqrstuvwxyz01 = _mm512_max_ps(vscaled1xmnopqrstuvwxyz01, voutput_min);
    vscaled2x0123456789ABCDEF = _mm512_max_ps(vscaled2x0123456789ABCDEF, voutput_min);
    vscaled2xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled2xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled2xWXYZabcdefghijkl = _mm512_max_ps(vscaled2xWXYZabcdefghijkl, voutput_min);
    vscaled2xmnopqrstuvwxyz01 = _mm512_max_ps(vscaled2xmnopqrstuvwxyz01, voutput_min);
    vscaled3x0123456789ABCDEF = _mm512_max_ps(vscaled3x0123456789ABCDEF, voutput_min);
    vscaled3xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled3xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled3xWXYZabcdefghijkl = _mm512_max_ps(vscaled3xWXYZabcdefghijkl, voutput_min);
    vscaled3xmnopqrstuvwxyz01 = _mm512_max_ps(vscaled3xmnopqrstuvwxyz01, voutput_min);
    vscaled4x0123456789ABCDEF = _mm512_max_ps(vscaled4x0123456789ABCDEF, voutput_min);
    vscaled4xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled4xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled4xWXYZabcdefghijkl = _mm512_max_ps(vscaled4xWXYZabcdefghijkl, voutput_min);
    vscaled4xmnopqrstuvwxyz01 = _mm512_max_ps(vscaled4xmnopqrstuvwxyz01, voutput_min);
    vscaled5x0123456789ABCDEF = _mm512_max_ps(vscaled5x0123456789ABCDEF, voutput_min);
    vscaled5xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled5xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled5xWXYZabcdefghijkl = _mm512_max_ps(vscaled5xWXYZabcdefghijkl, voutput_min);
    vscaled5xmnopqrstuvwxyz01 = _mm512_max_ps(vscaled5xmnopqrstuvwxyz01, voutput_min);
    vscaled6x0123456789ABCDEF = _mm512_max_ps(vscaled6x0123456789ABCDEF, voutput_min);
    vscaled6xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled6xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled6xWXYZabcdefghijkl = _mm512_max_ps(vscaled6xWXYZabcdefghijkl, voutput_min);
    vscaled6xmnopqrstuvwxyz01 = _mm512_max_ps(vscaled6xmnopqrstuvwxyz01, voutput_min);
    vscaled7x0123456789ABCDEF = _mm512_max_ps(vscaled7x0123456789ABCDEF, voutput_min);
    vscaled7xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled7xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled7xWXYZabcdefghijkl = _mm512_max_ps(vscaled7xWXYZabcdefghijkl, voutput_min);
    vscaled7xmnopqrstuvwxyz01 = _mm512_max_ps(vscaled7xmnopqrstuvwxyz01, voutput_min);
    vscaled8x0123456789ABCDEF = _mm512_max_ps(vscaled8x0123456789ABCDEF, voutput_min);
    vscaled8xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled8xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled8xWXYZabcdefghijkl = _mm512_max_ps(vscaled8xWXYZabcdefghijkl, voutput_min);
    vscaled8xmnopqrstuvwxyz01 = _mm512_max_ps(vscaled8xmnopqrstuvwxyz01, voutput_min);
    vscaled9x0123456789ABCDEF = _mm512_max_ps(vscaled9x0123456789ABCDEF, voutput_min);
    vscaled9xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled9xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled9xWXYZabcdefghijkl = _mm512_max_ps(vscaled9xWXYZabcdefghijkl, voutput_min);
    vscaled9xmnopqrstuvwxyz01 = _mm512_max_ps(vscaled9xmnopqrstuvwxyz01, voutput_min);
    vscaled10x0123456789ABCDEF = _mm512_max_ps(vscaled10x0123456789ABCDEF, voutput_min);
    vscaled10xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled10xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled10xWXYZabcdefghijkl = _mm512_max_ps(vscaled10xWXYZabcdefghijkl, voutput_min);
    vscaled10xmnopqrstuvwxyz01 = _mm512_max_ps(vscaled10xmnopqrstuvwxyz01, voutput_min);
    vscaled11x0123456789ABCDEF = _mm512_max_ps(vscaled11x0123456789ABCDEF, voutput_min);
    vscaled11xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled11xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled11xWXYZabcdefghijkl = _mm512_max_ps(vscaled11xWXYZabcdefghijkl, voutput_min);
    vscaled11xmnopqrstuvwxyz01 = _mm512_max_ps(vscaled11xmnopqrstuvwxyz01, voutput_min);
    vscaled12x0123456789ABCDEF = _mm512_max_ps(vscaled12x0123456789ABCDEF, voutput_min);
    vscaled12xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled12xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled12xWXYZabcdefghijkl = _mm512_max_ps(vscaled12xWXYZabcdefghijkl, voutput_min);
    vscaled12xmnopqrstuvwxyz01 = _mm512_max_ps(vscaled12xmnopqrstuvwxyz01, voutput_min);
    vscaled13x0123456789ABCDEF = _mm512_max_ps(vscaled13x0123456789ABCDEF, voutput_min);
    vscaled13xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled13xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled13xWXYZabcdefghijkl = _mm512_max_ps(vscaled13xWXYZabcdefghijkl, voutput_min);
    vscaled13xmnopqrstuvwxyz01 = _mm512_max_ps(vscaled13xmnopqrstuvwxyz01, voutput_min);
    vscaled14x0123456789ABCDEF = _mm512_max_ps(vscaled14x0123456789ABCDEF, voutput_min);
    vscaled14xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled14xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled14xWXYZabcdefghijkl = _mm512_max_ps(vscaled14xWXYZabcdefghijkl, voutput_min);
    vscaled14xmnopqrstuvwxyz01 = _mm512_max_ps(vscaled14xmnopqrstuvwxyz01, voutput_min);
    vscaled15x0123456789ABCDEF = _mm512_max_ps(vscaled15x0123456789ABCDEF, voutput_min);
    vscaled15xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled15xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled15xWXYZabcdefghijkl = _mm512_max_ps(vscaled15xWXYZabcdefghijkl, voutput_min);
    vscaled15xmnopqrstuvwxyz01 = _mm512_max_ps(vscaled15xmnopqrstuvwxyz01, voutput_min);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max);
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled0xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled0xWXYZabcdefghijkl = _mm512_min_ps(vscaled0xWXYZabcdefghijkl, voutput_max);
    vscaled0xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled0xmnopqrstuvwxyz01, voutput_max);
    vscaled1x0123456789ABCDEF = _mm512_min_ps(vscaled1x0123456789ABCDEF, voutput_max);
    vscaled1xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled1xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled1xWXYZabcdefghijkl = _mm512_min_ps(vscaled1xWXYZabcdefghijkl, voutput_max);
    vscaled1xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled1xmnopqrstuvwxyz01, voutput_max);
    vscaled2x0123456789ABCDEF = _mm512_min_ps(vscaled2x0123456789ABCDEF, voutput_max);
    vscaled2xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled2xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled2xWXYZabcdefghijkl = _mm512_min_ps(vscaled2xWXYZabcdefghijkl, voutput_max);
    vscaled2xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled2xmnopqrstuvwxyz01, voutput_max);
    vscaled3x0123456789ABCDEF = _mm512_min_ps(vscaled3x0123456789ABCDEF, voutput_max);
    vscaled3xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled3xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled3xWXYZabcdefghijkl = _mm512_min_ps(vscaled3xWXYZabcdefghijkl, voutput_max);
    vscaled3xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled3xmnopqrstuvwxyz01, voutput_max);
    vscaled4x0123456789ABCDEF = _mm512_min_ps(vscaled4x0123456789ABCDEF, voutput_max);
    vscaled4xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled4xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled4xWXYZabcdefghijkl = _mm512_min_ps(vscaled4xWXYZabcdefghijkl, voutput_max);
    vscaled4xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled4xmnopqrstuvwxyz01, voutput_max);
    vscaled5x0123456789ABCDEF = _mm512_min_ps(vscaled5x0123456789ABCDEF, voutput_max);
    vscaled5xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled5xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled5xWXYZabcdefghijkl = _mm512_min_ps(vscaled5xWXYZabcdefghijkl, voutput_max);
    vscaled5xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled5xmnopqrstuvwxyz01, voutput_max);
    vscaled6x0123456789ABCDEF = _mm512_min_ps(vscaled6x0123456789ABCDEF, voutput_max);
    vscaled6xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled6xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled6xWXYZabcdefghijkl = _mm512_min_ps(vscaled6xWXYZabcdefghijkl, voutput_max);
    vscaled6xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled6xmnopqrstuvwxyz01, voutput_max);
    vscaled7x0123456789ABCDEF = _mm512_min_ps(vscaled7x0123456789ABCDEF, voutput_max);
    vscaled7xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled7xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled7xWXYZabcdefghijkl = _mm512_min_ps(vscaled7xWXYZabcdefghijkl, voutput_max);
    vscaled7xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled7xmnopqrstuvwxyz01, voutput_max);
    vscaled8x0123456789ABCDEF = _mm512_min_ps(vscaled8x0123456789ABCDEF, voutput_max);
    vscaled8xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled8xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled8xWXYZabcdefghijkl = _mm512_min_ps(vscaled8xWXYZabcdefghijkl, voutput_max);
    vscaled8xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled8xmnopqrstuvwxyz01, voutput_max);
    vscaled9x0123456789ABCDEF = _mm512_min_ps(vscaled9x0123456789ABCDEF, voutput_max);
    vscaled9xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled9xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled9xWXYZabcdefghijkl = _mm512_min_ps(vscaled9xWXYZabcdefghijkl, voutput_max);
    vscaled9xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled9xmnopqrstuvwxyz01, voutput_max);
    vscaled10x0123456789ABCDEF = _mm512_min_ps(vscaled10x0123456789ABCDEF, voutput_max);
    vscaled10xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled10xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled10xWXYZabcdefghijkl = _mm512_min_ps(vscaled10xWXYZabcdefghijkl, voutput_max);
    vscaled10xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled10xmnopqrstuvwxyz01, voutput_max);
    vscaled11x0123456789ABCDEF = _mm512_min_ps(vscaled11x0123456789ABCDEF, voutput_max);
    vscaled11xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled11xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled11xWXYZabcdefghijkl = _mm512_min_ps(vscaled11xWXYZabcdefghijkl, voutput_max);
    vscaled11xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled11xmnopqrstuvwxyz01, voutput_max);
    vscaled12x0123456789ABCDEF = _mm512_min_ps(vscaled12x0123456789ABCDEF, voutput_max);
    vscaled12xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled12xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled12xWXYZabcdefghijkl = _mm512_min_ps(vscaled12xWXYZabcdefghijkl, voutput_max);
    vscaled12xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled12xmnopqrstuvwxyz01, voutput_max);
    vscaled13x0123456789ABCDEF = _mm512_min_ps(vscaled13x0123456789ABCDEF, voutput_max);
    vscaled13xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled13xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled13xWXYZabcdefghijkl = _mm512_min_ps(vscaled13xWXYZabcdefghijkl, voutput_max);
    vscaled13xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled13xmnopqrstuvwxyz01, voutput_max);
    vscaled14x0123456789ABCDEF = _mm512_min_ps(vscaled14x0123456789ABCDEF, voutput_max);
    vscaled14xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled14xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled14xWXYZabcdefghijkl = _mm512_min_ps(vscaled14xWXYZabcdefghijkl, voutput_max);
    vscaled14xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled14xmnopqrstuvwxyz01, voutput_max);
    vscaled15x0123456789ABCDEF = _mm512_min_ps(vscaled15x0123456789ABCDEF, voutput_max);
    vscaled15xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled15xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled15xWXYZabcdefghijkl = _mm512_min_ps(vscaled15xWXYZabcdefghijkl, voutput_max);
    vscaled15xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled15xmnopqrstuvwxyz01, voutput_max);

    if XNN_LIKELY(nc >= 64) {
      _mm512_storeu_ps(c15 + 0, vscaled15x0123456789ABCDEF);
      _mm512_storeu_ps(c15 + 16, vscaled15xGHIJKLMNOPQRSTUV);
      _mm512_storeu_ps(c15 + 32, vscaled15xWXYZabcdefghijkl);
      _mm512_storeu_ps(c15 + 48, vscaled15xmnopqrstuvwxyz01);
      c15 = (float*) ((uintptr_t) c15 + cn_stride);
      _mm512_storeu_ps(c14 + 0, vscaled14x0123456789ABCDEF);
      _mm512_storeu_ps(c14 + 16, vscaled14xGHIJKLMNOPQRSTUV);
      _mm512_storeu_ps(c14 + 32, vscaled14xWXYZabcdefghijkl);
      _mm512_storeu_ps(c14 + 48, vscaled14xmnopqrstuvwxyz01);
      c14 = (float*) ((uintptr_t) c14 + cn_stride);
      _mm512_storeu_ps(c13 + 0, vscaled13x0123456789ABCDEF);
      _mm512_storeu_ps(c13 + 16, vscaled13xGHIJKLMNOPQRSTUV);
      _mm512_storeu_ps(c13 + 32, vscaled13xWXYZabcdefghijkl);
      _mm512_storeu_ps(c13 + 48, vscaled13xmnopqrstuvwxyz01);
      c13 = (float*) ((uintptr_t) c13 + cn_stride);
      _mm512_storeu_ps(c12 + 0, vscaled12x0123456789ABCDEF);
      _mm512_storeu_ps(c12 + 16, vscaled12xGHIJKLMNOPQRSTUV);
      _mm512_storeu_ps(c12 + 32, vscaled12xWXYZabcdefghijkl);
      _mm512_storeu_ps(c12 + 48, vscaled12xmnopqrstuvwxyz01);
      c12 = (float*) ((uintptr_t) c12 + cn_stride);
      _mm512_storeu_ps(c11 + 0, vscaled11x0123456789ABCDEF);
      _mm512_storeu_ps(c11 + 16, vscaled11xGHIJKLMNOPQRSTUV);
      _mm512_storeu_ps(c11 + 32, vscaled11xWXYZabcdefghijkl);
      _mm512_storeu_ps(c11 + 48, vscaled11xmnopqrstuvwxyz01);
      c11 = (float*) ((uintptr_t) c11 + cn_stride);
      _mm512_storeu_ps(c10 + 0, vscaled10x0123456789ABCDEF);
      _mm512_storeu_ps(c10 + 16, vscaled10xGHIJKLMNOPQRSTUV);
      _mm512_storeu_ps(c10 + 32, vscaled10xWXYZabcdefghijkl);
      _mm512_storeu_ps(c10 + 48, vscaled10xmnopqrstuvwxyz01);
      c10 = (float*) ((uintptr_t) c10 + cn_stride);
      _mm512_storeu_ps(c9 + 0, vscaled9x0123456789ABCDEF);
      _mm512_storeu_ps(c9 + 16, vscaled9xGHIJKLMNOPQRSTUV);
      _mm512_storeu_ps(c9 + 32, vscaled9xWXYZabcdefghijkl);
      _mm512_storeu_ps(c9 + 48, vscaled9xmnopqrstuvwxyz01);
      c9 = (float*) ((uintptr_t) c9 + cn_stride);
      _mm512_storeu_ps(c8 + 0, vscaled8x0123456789ABCDEF);
      _mm512_storeu_ps(c8 + 16, vscaled8xGHIJKLMNOPQRSTUV);
      _mm512_storeu_ps(c8 + 32, vscaled8xWXYZabcdefghijkl);
      _mm512_storeu_ps(c8 + 48, vscaled8xmnopqrstuvwxyz01);
      c8 = (float*) ((uintptr_t) c8 + cn_stride);
      _mm512_storeu_ps(c7 + 0, vscaled7x0123456789ABCDEF);
      _mm512_storeu_ps(c7 + 16, vscaled7xGHIJKLMNOPQRSTUV);
      _mm512_storeu_ps(c7 + 32, vscaled7xWXYZabcdefghijkl);
      _mm512_storeu_ps(c7 + 48, vscaled7xmnopqrstuvwxyz01);
      c7 = (float*) ((uintptr_t) c7 + cn_stride);
      _mm512_storeu_ps(c6 + 0, vscaled6x0123456789ABCDEF);
      _mm512_storeu_ps(c6 + 16, vscaled6xGHIJKLMNOPQRSTUV);
      _mm512_storeu_ps(c6 + 32, vscaled6xWXYZabcdefghijkl);
      _mm512_storeu_ps(c6 + 48, vscaled6xmnopqrstuvwxyz01);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      _mm512_storeu_ps(c5 + 0, vscaled5x0123456789ABCDEF);
      _mm512_storeu_ps(c5 + 16, vscaled5xGHIJKLMNOPQRSTUV);
      _mm512_storeu_ps(c5 + 32, vscaled5xWXYZabcdefghijkl);
      _mm512_storeu_ps(c5 + 48, vscaled5xmnopqrstuvwxyz01);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      _mm512_storeu_ps(c4 + 0, vscaled4x0123456789ABCDEF);
      _mm512_storeu_ps(c4 + 16, vscaled4xGHIJKLMNOPQRSTUV);
      _mm512_storeu_ps(c4 + 32, vscaled4xWXYZabcdefghijkl);
      _mm512_storeu_ps(c4 + 48, vscaled4xmnopqrstuvwxyz01);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm512_storeu_ps(c3 + 0, vscaled3x0123456789ABCDEF);
      _mm512_storeu_ps(c3 + 16, vscaled3xGHIJKLMNOPQRSTUV);
      _mm512_storeu_ps(c3 + 32, vscaled3xWXYZabcdefghijkl);
      _mm512_storeu_ps(c3 + 48, vscaled3xmnopqrstuvwxyz01);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm512_storeu_ps(c2 + 0, vscaled2x0123456789ABCDEF);
      _mm512_storeu_ps(c2 + 16, vscaled2xGHIJKLMNOPQRSTUV);
      _mm512_storeu_ps(c2 + 32, vscaled2xWXYZabcdefghijkl);
      _mm512_storeu_ps(c2 + 48, vscaled2xmnopqrstuvwxyz01);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm512_storeu_ps(c1 + 0, vscaled1x0123456789ABCDEF);
      _mm512_storeu_ps(c1 + 16, vscaled1xGHIJKLMNOPQRSTUV);
      _mm512_storeu_ps(c1 + 32, vscaled1xWXYZabcdefghijkl);
      _mm512_storeu_ps(c1 + 48, vscaled1xmnopqrstuvwxyz01);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm512_storeu_ps(c0 + 0, vscaled0x0123456789ABCDEF);
      _mm512_storeu_ps(c0 + 16, vscaled0xGHIJKLMNOPQRSTUV);
      _mm512_storeu_ps(c0 + 32, vscaled0xWXYZabcdefghijkl);
      _mm512_storeu_ps(c0 + 48, vscaled0xmnopqrstuvwxyz01);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a -= kc;
      nc -= 64;
    } else {
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 0) & 0xFFFF));
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 16) & 0xFFFF));
      const __mmask16 vmask2 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 32) & 0xFFFF));
      const __mmask16 vmask3 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 48) & 0xFFFF));
      _mm512_mask_storeu_ps(c15 + 0, vmask0, vscaled15x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c15 + 16, vmask1, vscaled15xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c15 + 32, vmask2, vscaled15xWXYZabcdefghijkl);
      _mm512_mask_storeu_ps(c15 + 48, vmask3, vscaled15xmnopqrstuvwxyz01);
      _mm512_mask_storeu_ps(c14 + 0, vmask0, vscaled14x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c14 + 16, vmask1, vscaled14xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c14 + 32, vmask2, vscaled14xWXYZabcdefghijkl);
      _mm512_mask_storeu_ps(c14 + 48, vmask3, vscaled14xmnopqrstuvwxyz01);
      _mm512_mask_storeu_ps(c13 + 0, vmask0, vscaled13x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c13 + 16, vmask1, vscaled13xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c13 + 32, vmask2, vscaled13xWXYZabcdefghijkl);
      _mm512_mask_storeu_ps(c13 + 48, vmask3, vscaled13xmnopqrstuvwxyz01);
      _mm512_mask_storeu_ps(c12 + 0, vmask0, vscaled12x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c12 + 16, vmask1, vscaled12xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c12 + 32, vmask2, vscaled12xWXYZabcdefghijkl);
      _mm512_mask_storeu_ps(c12 + 48, vmask3, vscaled12xmnopqrstuvwxyz01);
      _mm512_mask_storeu_ps(c11 + 0, vmask0, vscaled11x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c11 + 16, vmask1, vscaled11xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c11 + 32, vmask2, vscaled11xWXYZabcdefghijkl);
      _mm512_mask_storeu_ps(c11 + 48, vmask3, vscaled11xmnopqrstuvwxyz01);
      _mm512_mask_storeu_ps(c10 + 0, vmask0, vscaled10x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c10 + 16, vmask1, vscaled10xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c10 + 32, vmask2, vscaled10xWXYZabcdefghijkl);
      _mm512_mask_storeu_ps(c10 + 48, vmask3, vscaled10xmnopqrstuvwxyz01);
      _mm512_mask_storeu_ps(c9 + 0, vmask0, vscaled9x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c9 + 16, vmask1, vscaled9xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c9 + 32, vmask2, vscaled9xWXYZabcdefghijkl);
      _mm512_mask_storeu_ps(c9 + 48, vmask3, vscaled9xmnopqrstuvwxyz01);
      _mm512_mask_storeu_ps(c8 + 0, vmask0, vscaled8x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c8 + 16, vmask1, vscaled8xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c8 + 32, vmask2, vscaled8xWXYZabcdefghijkl);
      _mm512_mask_storeu_ps(c8 + 48, vmask3, vscaled8xmnopqrstuvwxyz01);
      _mm512_mask_storeu_ps(c7 + 0, vmask0, vscaled7x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c7 + 16, vmask1, vscaled7xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c7 + 32, vmask2, vscaled7xWXYZabcdefghijkl);
      _mm512_mask_storeu_ps(c7 + 48, vmask3, vscaled7xmnopqrstuvwxyz01);
      _mm512_mask_storeu_ps(c6 + 0, vmask0, vscaled6x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c6 + 16, vmask1, vscaled6xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c6 + 32, vmask2, vscaled6xWXYZabcdefghijkl);
      _mm512_mask_storeu_ps(c6 + 48, vmask3, vscaled6xmnopqrstuvwxyz01);
      _mm512_mask_storeu_ps(c5 + 0, vmask0, vscaled5x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c5 + 16, vmask1, vscaled5xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c5 + 32, vmask2, vscaled5xWXYZabcdefghijkl);
      _mm512_mask_storeu_ps(c5 + 48, vmask3, vscaled5xmnopqrstuvwxyz01);
      _mm512_mask_storeu_ps(c4 + 0, vmask0, vscaled4x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c4 + 16, vmask1, vscaled4xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c4 + 32, vmask2, vscaled4xWXYZabcdefghijkl);
      _mm512_mask_storeu_ps(c4 + 48, vmask3, vscaled4xmnopqrstuvwxyz01);
      _mm512_mask_storeu_ps(c3 + 0, vmask0, vscaled3x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c3 + 16, vmask1, vscaled3xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c3 + 32, vmask2, vscaled3xWXYZabcdefghijkl);
      _mm512_mask_storeu_ps(c3 + 48, vmask3, vscaled3xmnopqrstuvwxyz01);
      _mm512_mask_storeu_ps(c2 + 0, vmask0, vscaled2x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c2 + 16, vmask1, vscaled2xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c2 + 32, vmask2, vscaled2xWXYZabcdefghijkl);
      _mm512_mask_storeu_ps(c2 + 48, vmask3, vscaled2xmnopqrstuvwxyz01);
      _mm512_mask_storeu_ps(c1 + 0, vmask0, vscaled1x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c1 + 16, vmask1, vscaled1xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c1 + 32, vmask2, vscaled1xWXYZabcdefghijkl);
      _mm512_mask_storeu_ps(c1 + 48, vmask3, vscaled1xmnopqrstuvwxyz01);
      _mm512_mask_storeu_ps(c0 + 0, vmask0, vscaled0x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c0 + 16, vmask1, vscaled0xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c0 + 32, vmask2, vscaled0xWXYZabcdefghijkl);
      _mm512_mask_storeu_ps(c0 + 48, vmask3, vscaled0xmnopqrstuvwxyz01);
      nc = 0;
    }
  } while (nc != 0);
  // Release tile config
  //  _tile_release();
  __asm__ volatile ("tilerelease" ::);
  #endif  // defined(__x86_64__)
}
