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


void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_16x64c4__avx512amx_prfm(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
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

  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  int8_t* c4 = (int8_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  int8_t* c5 = (int8_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    c5 = c4;
  }
  int8_t* c6 = (int8_t*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    c6 = c5;
  }
  int8_t* c7 = (int8_t*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 8) {
    c7 = c6;
  }
  int8_t* c8 = (int8_t*) ((uintptr_t) c7 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 8) {
    c8 = c7;
  }
  int8_t* c9 = (int8_t*) ((uintptr_t) c8 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 10) {
    c9 = c8;
  }
  int8_t* c10 = (int8_t*) ((uintptr_t) c9 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 10) {
    c10 = c9;
  }
  int8_t* c11 = (int8_t*) ((uintptr_t) c10 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 12) {
    c11 = c10;
  }
  int8_t* c12 = (int8_t*) ((uintptr_t) c11 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 12) {
    c12 = c11;
  }
  int8_t* c13 = (int8_t*) ((uintptr_t) c12 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 14) {
    c13 = c12;
  }
  int8_t* c14 = (int8_t*) ((uintptr_t) c13 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 14) {
    c14 = c13;
  }
  int8_t* c15 = (int8_t*) ((uintptr_t) c14 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 16) {
    c15 = c14;
  }

  const __m512 voutput_max_less_zero_point = _mm512_set1_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_set1_epi32(params->fp32_avx512.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx512.output_min);

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
      _tile_loadd(5, (const int8_t*) w + 0, 256);
      _tile_dpbssd(0, 4, 5);
      _tile_loadd(5, (const int8_t*) w + 64, 256);
      _tile_dpbssd(1, 4, 5);
      _tile_loadd(5, (const int8_t*) w + 128, 256);
      _tile_dpbssd(2, 4, 5);
      _tile_loadd(5, (const int8_t*) w + 192, 256);
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

      w = (const int8_t*) w + 4096;
      k -= 64 * sizeof(int8_t);
    }

    if XNN_UNLIKELY(k != 0) {
      _tile_loadd(6, a, a_stride);
      a += kremainder;
      _tile_loadd(7, (const int8_t*) w + 0, 256);
      _tile_dpbssd(0, 6, 7);
      _tile_loadd(7, (const int8_t*) w + 64, 256);
      _tile_dpbssd(1, 6, 7);
      _tile_loadd(7, (const int8_t*) w + 128, 256);
      _tile_dpbssd(2, 6, 7);
      _tile_loadd(7, (const int8_t*) w + 192, 256);
      _tile_dpbssd(3, 6, 7);

      w = (const int8_t*) w + kremainder * 64;
      k -= kremainder * sizeof(int8_t);
    }

    // Add tile to bias
    _tile_stored(0, res0, 64);
    _tile_stored(1, res1, 64);
    _tile_stored(2, res2, 64);
    _tile_stored(3, res3, 64);

    __m512i vacc0x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 0));
    __m512i vacc0xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 0));
    __m512i vacc0xWXYZabcdefghijkl = _mm512_add_epi32(vksumWXYZabcdefghijkl, _mm512_load_epi32(res2 + 0));
    __m512i vacc0xmnopqrstuvwxyz01 = _mm512_add_epi32(vksummnopqrstuvwxyz01, _mm512_load_epi32(res3 + 0));
    __m512i vacc1x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 16));
    __m512i vacc1xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 16));
    __m512i vacc1xWXYZabcdefghijkl = _mm512_add_epi32(vksumWXYZabcdefghijkl, _mm512_load_epi32(res2 + 16));
    __m512i vacc1xmnopqrstuvwxyz01 = _mm512_add_epi32(vksummnopqrstuvwxyz01, _mm512_load_epi32(res3 + 16));
    __m512i vacc2x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 32));
    __m512i vacc2xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 32));
    __m512i vacc2xWXYZabcdefghijkl = _mm512_add_epi32(vksumWXYZabcdefghijkl, _mm512_load_epi32(res2 + 32));
    __m512i vacc2xmnopqrstuvwxyz01 = _mm512_add_epi32(vksummnopqrstuvwxyz01, _mm512_load_epi32(res3 + 32));
    __m512i vacc3x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 48));
    __m512i vacc3xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 48));
    __m512i vacc3xWXYZabcdefghijkl = _mm512_add_epi32(vksumWXYZabcdefghijkl, _mm512_load_epi32(res2 + 48));
    __m512i vacc3xmnopqrstuvwxyz01 = _mm512_add_epi32(vksummnopqrstuvwxyz01, _mm512_load_epi32(res3 + 48));
    __m512i vacc4x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 64));
    __m512i vacc4xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 64));
    __m512i vacc4xWXYZabcdefghijkl = _mm512_add_epi32(vksumWXYZabcdefghijkl, _mm512_load_epi32(res2 + 64));
    __m512i vacc4xmnopqrstuvwxyz01 = _mm512_add_epi32(vksummnopqrstuvwxyz01, _mm512_load_epi32(res3 + 64));
    __m512i vacc5x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 80));
    __m512i vacc5xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 80));
    __m512i vacc5xWXYZabcdefghijkl = _mm512_add_epi32(vksumWXYZabcdefghijkl, _mm512_load_epi32(res2 + 80));
    __m512i vacc5xmnopqrstuvwxyz01 = _mm512_add_epi32(vksummnopqrstuvwxyz01, _mm512_load_epi32(res3 + 80));
    __m512i vacc6x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 96));
    __m512i vacc6xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 96));
    __m512i vacc6xWXYZabcdefghijkl = _mm512_add_epi32(vksumWXYZabcdefghijkl, _mm512_load_epi32(res2 + 96));
    __m512i vacc6xmnopqrstuvwxyz01 = _mm512_add_epi32(vksummnopqrstuvwxyz01, _mm512_load_epi32(res3 + 96));
    __m512i vacc7x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 112));
    __m512i vacc7xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 112));
    __m512i vacc7xWXYZabcdefghijkl = _mm512_add_epi32(vksumWXYZabcdefghijkl, _mm512_load_epi32(res2 + 112));
    __m512i vacc7xmnopqrstuvwxyz01 = _mm512_add_epi32(vksummnopqrstuvwxyz01, _mm512_load_epi32(res3 + 112));
    __m512i vacc8x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 128));
    __m512i vacc8xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 128));
    __m512i vacc8xWXYZabcdefghijkl = _mm512_add_epi32(vksumWXYZabcdefghijkl, _mm512_load_epi32(res2 + 128));
    __m512i vacc8xmnopqrstuvwxyz01 = _mm512_add_epi32(vksummnopqrstuvwxyz01, _mm512_load_epi32(res3 + 128));
    __m512i vacc9x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 144));
    __m512i vacc9xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 144));
    __m512i vacc9xWXYZabcdefghijkl = _mm512_add_epi32(vksumWXYZabcdefghijkl, _mm512_load_epi32(res2 + 144));
    __m512i vacc9xmnopqrstuvwxyz01 = _mm512_add_epi32(vksummnopqrstuvwxyz01, _mm512_load_epi32(res3 + 144));
    __m512i vacc10x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 160));
    __m512i vacc10xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 160));
    __m512i vacc10xWXYZabcdefghijkl = _mm512_add_epi32(vksumWXYZabcdefghijkl, _mm512_load_epi32(res2 + 160));
    __m512i vacc10xmnopqrstuvwxyz01 = _mm512_add_epi32(vksummnopqrstuvwxyz01, _mm512_load_epi32(res3 + 160));
    __m512i vacc11x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 176));
    __m512i vacc11xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 176));
    __m512i vacc11xWXYZabcdefghijkl = _mm512_add_epi32(vksumWXYZabcdefghijkl, _mm512_load_epi32(res2 + 176));
    __m512i vacc11xmnopqrstuvwxyz01 = _mm512_add_epi32(vksummnopqrstuvwxyz01, _mm512_load_epi32(res3 + 176));
    __m512i vacc12x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 192));
    __m512i vacc12xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 192));
    __m512i vacc12xWXYZabcdefghijkl = _mm512_add_epi32(vksumWXYZabcdefghijkl, _mm512_load_epi32(res2 + 192));
    __m512i vacc12xmnopqrstuvwxyz01 = _mm512_add_epi32(vksummnopqrstuvwxyz01, _mm512_load_epi32(res3 + 192));
    __m512i vacc13x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 208));
    __m512i vacc13xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 208));
    __m512i vacc13xWXYZabcdefghijkl = _mm512_add_epi32(vksumWXYZabcdefghijkl, _mm512_load_epi32(res2 + 208));
    __m512i vacc13xmnopqrstuvwxyz01 = _mm512_add_epi32(vksummnopqrstuvwxyz01, _mm512_load_epi32(res3 + 208));
    __m512i vacc14x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 224));
    __m512i vacc14xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 224));
    __m512i vacc14xWXYZabcdefghijkl = _mm512_add_epi32(vksumWXYZabcdefghijkl, _mm512_load_epi32(res2 + 224));
    __m512i vacc14xmnopqrstuvwxyz01 = _mm512_add_epi32(vksummnopqrstuvwxyz01, _mm512_load_epi32(res3 + 224));
    __m512i vacc15x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 240));
    __m512i vacc15xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 240));
    __m512i vacc15xWXYZabcdefghijkl = _mm512_add_epi32(vksumWXYZabcdefghijkl, _mm512_load_epi32(res2 + 240));
    __m512i vacc15xmnopqrstuvwxyz01 = _mm512_add_epi32(vksummnopqrstuvwxyz01, _mm512_load_epi32(res3 + 240));

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

    const __m512 vscale0123456789ABCDEF = _mm512_load_ps((const float*) w + 0);
    const __m512 vscaleGHIJKLMNOPQRSTUV = _mm512_load_ps((const float*) w + 16);
    const __m512 vscaleWXYZabcdefghijkl = _mm512_load_ps((const float*) w + 32);
    const __m512 vscalemnopqrstuvwxyz01 = _mm512_load_ps((const float*) w + 48);
    w = (const int32_t*) w + 64;

    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled0xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled0xWXYZabcdefghijkl = _mm512_mul_ps(vscaled0xWXYZabcdefghijkl, vscaleWXYZabcdefghijkl);
    vscaled0xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled0xmnopqrstuvwxyz01, vscalemnopqrstuvwxyz01);
    vscaled1x0123456789ABCDEF = _mm512_mul_ps(vscaled1x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled1xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled1xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled1xWXYZabcdefghijkl = _mm512_mul_ps(vscaled1xWXYZabcdefghijkl, vscaleWXYZabcdefghijkl);
    vscaled1xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled1xmnopqrstuvwxyz01, vscalemnopqrstuvwxyz01);
    vscaled2x0123456789ABCDEF = _mm512_mul_ps(vscaled2x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled2xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled2xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled2xWXYZabcdefghijkl = _mm512_mul_ps(vscaled2xWXYZabcdefghijkl, vscaleWXYZabcdefghijkl);
    vscaled2xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled2xmnopqrstuvwxyz01, vscalemnopqrstuvwxyz01);
    vscaled3x0123456789ABCDEF = _mm512_mul_ps(vscaled3x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled3xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled3xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled3xWXYZabcdefghijkl = _mm512_mul_ps(vscaled3xWXYZabcdefghijkl, vscaleWXYZabcdefghijkl);
    vscaled3xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled3xmnopqrstuvwxyz01, vscalemnopqrstuvwxyz01);
    vscaled4x0123456789ABCDEF = _mm512_mul_ps(vscaled4x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled4xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled4xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled4xWXYZabcdefghijkl = _mm512_mul_ps(vscaled4xWXYZabcdefghijkl, vscaleWXYZabcdefghijkl);
    vscaled4xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled4xmnopqrstuvwxyz01, vscalemnopqrstuvwxyz01);
    vscaled5x0123456789ABCDEF = _mm512_mul_ps(vscaled5x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled5xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled5xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled5xWXYZabcdefghijkl = _mm512_mul_ps(vscaled5xWXYZabcdefghijkl, vscaleWXYZabcdefghijkl);
    vscaled5xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled5xmnopqrstuvwxyz01, vscalemnopqrstuvwxyz01);
    vscaled6x0123456789ABCDEF = _mm512_mul_ps(vscaled6x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled6xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled6xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled6xWXYZabcdefghijkl = _mm512_mul_ps(vscaled6xWXYZabcdefghijkl, vscaleWXYZabcdefghijkl);
    vscaled6xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled6xmnopqrstuvwxyz01, vscalemnopqrstuvwxyz01);
    vscaled7x0123456789ABCDEF = _mm512_mul_ps(vscaled7x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled7xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled7xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled7xWXYZabcdefghijkl = _mm512_mul_ps(vscaled7xWXYZabcdefghijkl, vscaleWXYZabcdefghijkl);
    vscaled7xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled7xmnopqrstuvwxyz01, vscalemnopqrstuvwxyz01);
    vscaled8x0123456789ABCDEF = _mm512_mul_ps(vscaled8x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled8xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled8xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled8xWXYZabcdefghijkl = _mm512_mul_ps(vscaled8xWXYZabcdefghijkl, vscaleWXYZabcdefghijkl);
    vscaled8xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled8xmnopqrstuvwxyz01, vscalemnopqrstuvwxyz01);
    vscaled9x0123456789ABCDEF = _mm512_mul_ps(vscaled9x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled9xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled9xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled9xWXYZabcdefghijkl = _mm512_mul_ps(vscaled9xWXYZabcdefghijkl, vscaleWXYZabcdefghijkl);
    vscaled9xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled9xmnopqrstuvwxyz01, vscalemnopqrstuvwxyz01);
    vscaled10x0123456789ABCDEF = _mm512_mul_ps(vscaled10x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled10xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled10xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled10xWXYZabcdefghijkl = _mm512_mul_ps(vscaled10xWXYZabcdefghijkl, vscaleWXYZabcdefghijkl);
    vscaled10xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled10xmnopqrstuvwxyz01, vscalemnopqrstuvwxyz01);
    vscaled11x0123456789ABCDEF = _mm512_mul_ps(vscaled11x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled11xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled11xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled11xWXYZabcdefghijkl = _mm512_mul_ps(vscaled11xWXYZabcdefghijkl, vscaleWXYZabcdefghijkl);
    vscaled11xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled11xmnopqrstuvwxyz01, vscalemnopqrstuvwxyz01);
    vscaled12x0123456789ABCDEF = _mm512_mul_ps(vscaled12x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled12xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled12xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled12xWXYZabcdefghijkl = _mm512_mul_ps(vscaled12xWXYZabcdefghijkl, vscaleWXYZabcdefghijkl);
    vscaled12xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled12xmnopqrstuvwxyz01, vscalemnopqrstuvwxyz01);
    vscaled13x0123456789ABCDEF = _mm512_mul_ps(vscaled13x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled13xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled13xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled13xWXYZabcdefghijkl = _mm512_mul_ps(vscaled13xWXYZabcdefghijkl, vscaleWXYZabcdefghijkl);
    vscaled13xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled13xmnopqrstuvwxyz01, vscalemnopqrstuvwxyz01);
    vscaled14x0123456789ABCDEF = _mm512_mul_ps(vscaled14x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled14xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled14xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled14xWXYZabcdefghijkl = _mm512_mul_ps(vscaled14xWXYZabcdefghijkl, vscaleWXYZabcdefghijkl);
    vscaled14xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled14xmnopqrstuvwxyz01, vscalemnopqrstuvwxyz01);
    vscaled15x0123456789ABCDEF = _mm512_mul_ps(vscaled15x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled15xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled15xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled15xWXYZabcdefghijkl = _mm512_mul_ps(vscaled15xWXYZabcdefghijkl, vscaleWXYZabcdefghijkl);
    vscaled15xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled15xmnopqrstuvwxyz01, vscalemnopqrstuvwxyz01);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled0xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled0xWXYZabcdefghijkl = _mm512_min_ps(vscaled0xWXYZabcdefghijkl, voutput_max_less_zero_point);
    vscaled0xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled0xmnopqrstuvwxyz01, voutput_max_less_zero_point);
    vscaled1x0123456789ABCDEF = _mm512_min_ps(vscaled1x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled1xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled1xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled1xWXYZabcdefghijkl = _mm512_min_ps(vscaled1xWXYZabcdefghijkl, voutput_max_less_zero_point);
    vscaled1xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled1xmnopqrstuvwxyz01, voutput_max_less_zero_point);
    vscaled2x0123456789ABCDEF = _mm512_min_ps(vscaled2x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled2xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled2xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled2xWXYZabcdefghijkl = _mm512_min_ps(vscaled2xWXYZabcdefghijkl, voutput_max_less_zero_point);
    vscaled2xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled2xmnopqrstuvwxyz01, voutput_max_less_zero_point);
    vscaled3x0123456789ABCDEF = _mm512_min_ps(vscaled3x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled3xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled3xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled3xWXYZabcdefghijkl = _mm512_min_ps(vscaled3xWXYZabcdefghijkl, voutput_max_less_zero_point);
    vscaled3xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled3xmnopqrstuvwxyz01, voutput_max_less_zero_point);
    vscaled4x0123456789ABCDEF = _mm512_min_ps(vscaled4x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled4xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled4xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled4xWXYZabcdefghijkl = _mm512_min_ps(vscaled4xWXYZabcdefghijkl, voutput_max_less_zero_point);
    vscaled4xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled4xmnopqrstuvwxyz01, voutput_max_less_zero_point);
    vscaled5x0123456789ABCDEF = _mm512_min_ps(vscaled5x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled5xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled5xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled5xWXYZabcdefghijkl = _mm512_min_ps(vscaled5xWXYZabcdefghijkl, voutput_max_less_zero_point);
    vscaled5xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled5xmnopqrstuvwxyz01, voutput_max_less_zero_point);
    vscaled6x0123456789ABCDEF = _mm512_min_ps(vscaled6x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled6xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled6xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled6xWXYZabcdefghijkl = _mm512_min_ps(vscaled6xWXYZabcdefghijkl, voutput_max_less_zero_point);
    vscaled6xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled6xmnopqrstuvwxyz01, voutput_max_less_zero_point);
    vscaled7x0123456789ABCDEF = _mm512_min_ps(vscaled7x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled7xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled7xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled7xWXYZabcdefghijkl = _mm512_min_ps(vscaled7xWXYZabcdefghijkl, voutput_max_less_zero_point);
    vscaled7xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled7xmnopqrstuvwxyz01, voutput_max_less_zero_point);
    vscaled8x0123456789ABCDEF = _mm512_min_ps(vscaled8x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled8xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled8xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled8xWXYZabcdefghijkl = _mm512_min_ps(vscaled8xWXYZabcdefghijkl, voutput_max_less_zero_point);
    vscaled8xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled8xmnopqrstuvwxyz01, voutput_max_less_zero_point);
    vscaled9x0123456789ABCDEF = _mm512_min_ps(vscaled9x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled9xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled9xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled9xWXYZabcdefghijkl = _mm512_min_ps(vscaled9xWXYZabcdefghijkl, voutput_max_less_zero_point);
    vscaled9xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled9xmnopqrstuvwxyz01, voutput_max_less_zero_point);
    vscaled10x0123456789ABCDEF = _mm512_min_ps(vscaled10x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled10xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled10xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled10xWXYZabcdefghijkl = _mm512_min_ps(vscaled10xWXYZabcdefghijkl, voutput_max_less_zero_point);
    vscaled10xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled10xmnopqrstuvwxyz01, voutput_max_less_zero_point);
    vscaled11x0123456789ABCDEF = _mm512_min_ps(vscaled11x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled11xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled11xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled11xWXYZabcdefghijkl = _mm512_min_ps(vscaled11xWXYZabcdefghijkl, voutput_max_less_zero_point);
    vscaled11xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled11xmnopqrstuvwxyz01, voutput_max_less_zero_point);
    vscaled12x0123456789ABCDEF = _mm512_min_ps(vscaled12x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled12xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled12xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled12xWXYZabcdefghijkl = _mm512_min_ps(vscaled12xWXYZabcdefghijkl, voutput_max_less_zero_point);
    vscaled12xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled12xmnopqrstuvwxyz01, voutput_max_less_zero_point);
    vscaled13x0123456789ABCDEF = _mm512_min_ps(vscaled13x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled13xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled13xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled13xWXYZabcdefghijkl = _mm512_min_ps(vscaled13xWXYZabcdefghijkl, voutput_max_less_zero_point);
    vscaled13xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled13xmnopqrstuvwxyz01, voutput_max_less_zero_point);
    vscaled14x0123456789ABCDEF = _mm512_min_ps(vscaled14x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled14xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled14xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled14xWXYZabcdefghijkl = _mm512_min_ps(vscaled14xWXYZabcdefghijkl, voutput_max_less_zero_point);
    vscaled14xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled14xmnopqrstuvwxyz01, voutput_max_less_zero_point);
    vscaled15x0123456789ABCDEF = _mm512_min_ps(vscaled15x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled15xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled15xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled15xWXYZabcdefghijkl = _mm512_min_ps(vscaled15xWXYZabcdefghijkl, voutput_max_less_zero_point);
    vscaled15xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled15xmnopqrstuvwxyz01, voutput_max_less_zero_point);

    vacc0x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0x0123456789ABCDEF);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled0xGHIJKLMNOPQRSTUV);
    vacc0xWXYZabcdefghijkl = _mm512_cvtps_epi32(vscaled0xWXYZabcdefghijkl);
    vacc0xmnopqrstuvwxyz01 = _mm512_cvtps_epi32(vscaled0xmnopqrstuvwxyz01);
    vacc1x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled1x0123456789ABCDEF);
    vacc1xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled1xGHIJKLMNOPQRSTUV);
    vacc1xWXYZabcdefghijkl = _mm512_cvtps_epi32(vscaled1xWXYZabcdefghijkl);
    vacc1xmnopqrstuvwxyz01 = _mm512_cvtps_epi32(vscaled1xmnopqrstuvwxyz01);
    vacc2x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled2x0123456789ABCDEF);
    vacc2xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled2xGHIJKLMNOPQRSTUV);
    vacc2xWXYZabcdefghijkl = _mm512_cvtps_epi32(vscaled2xWXYZabcdefghijkl);
    vacc2xmnopqrstuvwxyz01 = _mm512_cvtps_epi32(vscaled2xmnopqrstuvwxyz01);
    vacc3x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled3x0123456789ABCDEF);
    vacc3xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled3xGHIJKLMNOPQRSTUV);
    vacc3xWXYZabcdefghijkl = _mm512_cvtps_epi32(vscaled3xWXYZabcdefghijkl);
    vacc3xmnopqrstuvwxyz01 = _mm512_cvtps_epi32(vscaled3xmnopqrstuvwxyz01);
    vacc4x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled4x0123456789ABCDEF);
    vacc4xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled4xGHIJKLMNOPQRSTUV);
    vacc4xWXYZabcdefghijkl = _mm512_cvtps_epi32(vscaled4xWXYZabcdefghijkl);
    vacc4xmnopqrstuvwxyz01 = _mm512_cvtps_epi32(vscaled4xmnopqrstuvwxyz01);
    vacc5x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled5x0123456789ABCDEF);
    vacc5xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled5xGHIJKLMNOPQRSTUV);
    vacc5xWXYZabcdefghijkl = _mm512_cvtps_epi32(vscaled5xWXYZabcdefghijkl);
    vacc5xmnopqrstuvwxyz01 = _mm512_cvtps_epi32(vscaled5xmnopqrstuvwxyz01);
    vacc6x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled6x0123456789ABCDEF);
    vacc6xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled6xGHIJKLMNOPQRSTUV);
    vacc6xWXYZabcdefghijkl = _mm512_cvtps_epi32(vscaled6xWXYZabcdefghijkl);
    vacc6xmnopqrstuvwxyz01 = _mm512_cvtps_epi32(vscaled6xmnopqrstuvwxyz01);
    vacc7x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled7x0123456789ABCDEF);
    vacc7xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled7xGHIJKLMNOPQRSTUV);
    vacc7xWXYZabcdefghijkl = _mm512_cvtps_epi32(vscaled7xWXYZabcdefghijkl);
    vacc7xmnopqrstuvwxyz01 = _mm512_cvtps_epi32(vscaled7xmnopqrstuvwxyz01);
    vacc8x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled8x0123456789ABCDEF);
    vacc8xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled8xGHIJKLMNOPQRSTUV);
    vacc8xWXYZabcdefghijkl = _mm512_cvtps_epi32(vscaled8xWXYZabcdefghijkl);
    vacc8xmnopqrstuvwxyz01 = _mm512_cvtps_epi32(vscaled8xmnopqrstuvwxyz01);
    vacc9x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled9x0123456789ABCDEF);
    vacc9xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled9xGHIJKLMNOPQRSTUV);
    vacc9xWXYZabcdefghijkl = _mm512_cvtps_epi32(vscaled9xWXYZabcdefghijkl);
    vacc9xmnopqrstuvwxyz01 = _mm512_cvtps_epi32(vscaled9xmnopqrstuvwxyz01);
    vacc10x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled10x0123456789ABCDEF);
    vacc10xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled10xGHIJKLMNOPQRSTUV);
    vacc10xWXYZabcdefghijkl = _mm512_cvtps_epi32(vscaled10xWXYZabcdefghijkl);
    vacc10xmnopqrstuvwxyz01 = _mm512_cvtps_epi32(vscaled10xmnopqrstuvwxyz01);
    vacc11x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled11x0123456789ABCDEF);
    vacc11xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled11xGHIJKLMNOPQRSTUV);
    vacc11xWXYZabcdefghijkl = _mm512_cvtps_epi32(vscaled11xWXYZabcdefghijkl);
    vacc11xmnopqrstuvwxyz01 = _mm512_cvtps_epi32(vscaled11xmnopqrstuvwxyz01);
    vacc12x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled12x0123456789ABCDEF);
    vacc12xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled12xGHIJKLMNOPQRSTUV);
    vacc12xWXYZabcdefghijkl = _mm512_cvtps_epi32(vscaled12xWXYZabcdefghijkl);
    vacc12xmnopqrstuvwxyz01 = _mm512_cvtps_epi32(vscaled12xmnopqrstuvwxyz01);
    vacc13x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled13x0123456789ABCDEF);
    vacc13xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled13xGHIJKLMNOPQRSTUV);
    vacc13xWXYZabcdefghijkl = _mm512_cvtps_epi32(vscaled13xWXYZabcdefghijkl);
    vacc13xmnopqrstuvwxyz01 = _mm512_cvtps_epi32(vscaled13xmnopqrstuvwxyz01);
    vacc14x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled14x0123456789ABCDEF);
    vacc14xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled14xGHIJKLMNOPQRSTUV);
    vacc14xWXYZabcdefghijkl = _mm512_cvtps_epi32(vscaled14xWXYZabcdefghijkl);
    vacc14xmnopqrstuvwxyz01 = _mm512_cvtps_epi32(vscaled14xmnopqrstuvwxyz01);
    vacc15x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled15x0123456789ABCDEF);
    vacc15xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled15xGHIJKLMNOPQRSTUV);
    vacc15xWXYZabcdefghijkl = _mm512_cvtps_epi32(vscaled15xWXYZabcdefghijkl);
    vacc15xmnopqrstuvwxyz01 = _mm512_cvtps_epi32(vscaled15xmnopqrstuvwxyz01);

    vacc0x0123456789ABCDEF = _mm512_add_epi32(vacc0x0123456789ABCDEF, voutput_zero_point);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc0xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc0xWXYZabcdefghijkl = _mm512_add_epi32(vacc0xWXYZabcdefghijkl, voutput_zero_point);
    vacc0xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc0xmnopqrstuvwxyz01, voutput_zero_point);
    vacc1x0123456789ABCDEF = _mm512_add_epi32(vacc1x0123456789ABCDEF, voutput_zero_point);
    vacc1xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc1xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc1xWXYZabcdefghijkl = _mm512_add_epi32(vacc1xWXYZabcdefghijkl, voutput_zero_point);
    vacc1xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc1xmnopqrstuvwxyz01, voutput_zero_point);
    vacc2x0123456789ABCDEF = _mm512_add_epi32(vacc2x0123456789ABCDEF, voutput_zero_point);
    vacc2xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc2xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc2xWXYZabcdefghijkl = _mm512_add_epi32(vacc2xWXYZabcdefghijkl, voutput_zero_point);
    vacc2xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc2xmnopqrstuvwxyz01, voutput_zero_point);
    vacc3x0123456789ABCDEF = _mm512_add_epi32(vacc3x0123456789ABCDEF, voutput_zero_point);
    vacc3xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc3xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc3xWXYZabcdefghijkl = _mm512_add_epi32(vacc3xWXYZabcdefghijkl, voutput_zero_point);
    vacc3xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc3xmnopqrstuvwxyz01, voutput_zero_point);
    vacc4x0123456789ABCDEF = _mm512_add_epi32(vacc4x0123456789ABCDEF, voutput_zero_point);
    vacc4xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc4xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc4xWXYZabcdefghijkl = _mm512_add_epi32(vacc4xWXYZabcdefghijkl, voutput_zero_point);
    vacc4xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc4xmnopqrstuvwxyz01, voutput_zero_point);
    vacc5x0123456789ABCDEF = _mm512_add_epi32(vacc5x0123456789ABCDEF, voutput_zero_point);
    vacc5xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc5xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc5xWXYZabcdefghijkl = _mm512_add_epi32(vacc5xWXYZabcdefghijkl, voutput_zero_point);
    vacc5xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc5xmnopqrstuvwxyz01, voutput_zero_point);
    vacc6x0123456789ABCDEF = _mm512_add_epi32(vacc6x0123456789ABCDEF, voutput_zero_point);
    vacc6xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc6xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc6xWXYZabcdefghijkl = _mm512_add_epi32(vacc6xWXYZabcdefghijkl, voutput_zero_point);
    vacc6xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc6xmnopqrstuvwxyz01, voutput_zero_point);
    vacc7x0123456789ABCDEF = _mm512_add_epi32(vacc7x0123456789ABCDEF, voutput_zero_point);
    vacc7xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc7xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc7xWXYZabcdefghijkl = _mm512_add_epi32(vacc7xWXYZabcdefghijkl, voutput_zero_point);
    vacc7xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc7xmnopqrstuvwxyz01, voutput_zero_point);
    vacc8x0123456789ABCDEF = _mm512_add_epi32(vacc8x0123456789ABCDEF, voutput_zero_point);
    vacc8xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc8xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc8xWXYZabcdefghijkl = _mm512_add_epi32(vacc8xWXYZabcdefghijkl, voutput_zero_point);
    vacc8xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc8xmnopqrstuvwxyz01, voutput_zero_point);
    vacc9x0123456789ABCDEF = _mm512_add_epi32(vacc9x0123456789ABCDEF, voutput_zero_point);
    vacc9xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc9xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc9xWXYZabcdefghijkl = _mm512_add_epi32(vacc9xWXYZabcdefghijkl, voutput_zero_point);
    vacc9xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc9xmnopqrstuvwxyz01, voutput_zero_point);
    vacc10x0123456789ABCDEF = _mm512_add_epi32(vacc10x0123456789ABCDEF, voutput_zero_point);
    vacc10xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc10xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc10xWXYZabcdefghijkl = _mm512_add_epi32(vacc10xWXYZabcdefghijkl, voutput_zero_point);
    vacc10xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc10xmnopqrstuvwxyz01, voutput_zero_point);
    vacc11x0123456789ABCDEF = _mm512_add_epi32(vacc11x0123456789ABCDEF, voutput_zero_point);
    vacc11xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc11xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc11xWXYZabcdefghijkl = _mm512_add_epi32(vacc11xWXYZabcdefghijkl, voutput_zero_point);
    vacc11xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc11xmnopqrstuvwxyz01, voutput_zero_point);
    vacc12x0123456789ABCDEF = _mm512_add_epi32(vacc12x0123456789ABCDEF, voutput_zero_point);
    vacc12xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc12xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc12xWXYZabcdefghijkl = _mm512_add_epi32(vacc12xWXYZabcdefghijkl, voutput_zero_point);
    vacc12xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc12xmnopqrstuvwxyz01, voutput_zero_point);
    vacc13x0123456789ABCDEF = _mm512_add_epi32(vacc13x0123456789ABCDEF, voutput_zero_point);
    vacc13xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc13xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc13xWXYZabcdefghijkl = _mm512_add_epi32(vacc13xWXYZabcdefghijkl, voutput_zero_point);
    vacc13xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc13xmnopqrstuvwxyz01, voutput_zero_point);
    vacc14x0123456789ABCDEF = _mm512_add_epi32(vacc14x0123456789ABCDEF, voutput_zero_point);
    vacc14xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc14xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc14xWXYZabcdefghijkl = _mm512_add_epi32(vacc14xWXYZabcdefghijkl, voutput_zero_point);
    vacc14xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc14xmnopqrstuvwxyz01, voutput_zero_point);
    vacc15x0123456789ABCDEF = _mm512_add_epi32(vacc15x0123456789ABCDEF, voutput_zero_point);
    vacc15xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc15xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc15xWXYZabcdefghijkl = _mm512_add_epi32(vacc15xWXYZabcdefghijkl, voutput_zero_point);
    vacc15xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc15xmnopqrstuvwxyz01, voutput_zero_point);

    __m128i vout0x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc0x0123456789ABCDEF);
    __m128i vout0xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc0xGHIJKLMNOPQRSTUV);
    __m128i vout0xWXYZabcdefghijkl = _mm512_cvtsepi32_epi8(vacc0xWXYZabcdefghijkl);
    __m128i vout0xmnopqrstuvwxyz01 = _mm512_cvtsepi32_epi8(vacc0xmnopqrstuvwxyz01);
    __m128i vout1x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc1x0123456789ABCDEF);
    __m128i vout1xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc1xGHIJKLMNOPQRSTUV);
    __m128i vout1xWXYZabcdefghijkl = _mm512_cvtsepi32_epi8(vacc1xWXYZabcdefghijkl);
    __m128i vout1xmnopqrstuvwxyz01 = _mm512_cvtsepi32_epi8(vacc1xmnopqrstuvwxyz01);
    __m128i vout2x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc2x0123456789ABCDEF);
    __m128i vout2xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc2xGHIJKLMNOPQRSTUV);
    __m128i vout2xWXYZabcdefghijkl = _mm512_cvtsepi32_epi8(vacc2xWXYZabcdefghijkl);
    __m128i vout2xmnopqrstuvwxyz01 = _mm512_cvtsepi32_epi8(vacc2xmnopqrstuvwxyz01);
    __m128i vout3x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc3x0123456789ABCDEF);
    __m128i vout3xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc3xGHIJKLMNOPQRSTUV);
    __m128i vout3xWXYZabcdefghijkl = _mm512_cvtsepi32_epi8(vacc3xWXYZabcdefghijkl);
    __m128i vout3xmnopqrstuvwxyz01 = _mm512_cvtsepi32_epi8(vacc3xmnopqrstuvwxyz01);
    __m128i vout4x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc4x0123456789ABCDEF);
    __m128i vout4xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc4xGHIJKLMNOPQRSTUV);
    __m128i vout4xWXYZabcdefghijkl = _mm512_cvtsepi32_epi8(vacc4xWXYZabcdefghijkl);
    __m128i vout4xmnopqrstuvwxyz01 = _mm512_cvtsepi32_epi8(vacc4xmnopqrstuvwxyz01);
    __m128i vout5x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc5x0123456789ABCDEF);
    __m128i vout5xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc5xGHIJKLMNOPQRSTUV);
    __m128i vout5xWXYZabcdefghijkl = _mm512_cvtsepi32_epi8(vacc5xWXYZabcdefghijkl);
    __m128i vout5xmnopqrstuvwxyz01 = _mm512_cvtsepi32_epi8(vacc5xmnopqrstuvwxyz01);
    __m128i vout6x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc6x0123456789ABCDEF);
    __m128i vout6xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc6xGHIJKLMNOPQRSTUV);
    __m128i vout6xWXYZabcdefghijkl = _mm512_cvtsepi32_epi8(vacc6xWXYZabcdefghijkl);
    __m128i vout6xmnopqrstuvwxyz01 = _mm512_cvtsepi32_epi8(vacc6xmnopqrstuvwxyz01);
    __m128i vout7x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc7x0123456789ABCDEF);
    __m128i vout7xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc7xGHIJKLMNOPQRSTUV);
    __m128i vout7xWXYZabcdefghijkl = _mm512_cvtsepi32_epi8(vacc7xWXYZabcdefghijkl);
    __m128i vout7xmnopqrstuvwxyz01 = _mm512_cvtsepi32_epi8(vacc7xmnopqrstuvwxyz01);
    __m128i vout8x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc8x0123456789ABCDEF);
    __m128i vout8xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc8xGHIJKLMNOPQRSTUV);
    __m128i vout8xWXYZabcdefghijkl = _mm512_cvtsepi32_epi8(vacc8xWXYZabcdefghijkl);
    __m128i vout8xmnopqrstuvwxyz01 = _mm512_cvtsepi32_epi8(vacc8xmnopqrstuvwxyz01);
    __m128i vout9x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc9x0123456789ABCDEF);
    __m128i vout9xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc9xGHIJKLMNOPQRSTUV);
    __m128i vout9xWXYZabcdefghijkl = _mm512_cvtsepi32_epi8(vacc9xWXYZabcdefghijkl);
    __m128i vout9xmnopqrstuvwxyz01 = _mm512_cvtsepi32_epi8(vacc9xmnopqrstuvwxyz01);
    __m128i vout10x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc10x0123456789ABCDEF);
    __m128i vout10xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc10xGHIJKLMNOPQRSTUV);
    __m128i vout10xWXYZabcdefghijkl = _mm512_cvtsepi32_epi8(vacc10xWXYZabcdefghijkl);
    __m128i vout10xmnopqrstuvwxyz01 = _mm512_cvtsepi32_epi8(vacc10xmnopqrstuvwxyz01);
    __m128i vout11x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc11x0123456789ABCDEF);
    __m128i vout11xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc11xGHIJKLMNOPQRSTUV);
    __m128i vout11xWXYZabcdefghijkl = _mm512_cvtsepi32_epi8(vacc11xWXYZabcdefghijkl);
    __m128i vout11xmnopqrstuvwxyz01 = _mm512_cvtsepi32_epi8(vacc11xmnopqrstuvwxyz01);
    __m128i vout12x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc12x0123456789ABCDEF);
    __m128i vout12xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc12xGHIJKLMNOPQRSTUV);
    __m128i vout12xWXYZabcdefghijkl = _mm512_cvtsepi32_epi8(vacc12xWXYZabcdefghijkl);
    __m128i vout12xmnopqrstuvwxyz01 = _mm512_cvtsepi32_epi8(vacc12xmnopqrstuvwxyz01);
    __m128i vout13x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc13x0123456789ABCDEF);
    __m128i vout13xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc13xGHIJKLMNOPQRSTUV);
    __m128i vout13xWXYZabcdefghijkl = _mm512_cvtsepi32_epi8(vacc13xWXYZabcdefghijkl);
    __m128i vout13xmnopqrstuvwxyz01 = _mm512_cvtsepi32_epi8(vacc13xmnopqrstuvwxyz01);
    __m128i vout14x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc14x0123456789ABCDEF);
    __m128i vout14xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc14xGHIJKLMNOPQRSTUV);
    __m128i vout14xWXYZabcdefghijkl = _mm512_cvtsepi32_epi8(vacc14xWXYZabcdefghijkl);
    __m128i vout14xmnopqrstuvwxyz01 = _mm512_cvtsepi32_epi8(vacc14xmnopqrstuvwxyz01);
    __m128i vout15x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc15x0123456789ABCDEF);
    __m128i vout15xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc15xGHIJKLMNOPQRSTUV);
    __m128i vout15xWXYZabcdefghijkl = _mm512_cvtsepi32_epi8(vacc15xWXYZabcdefghijkl);
    __m128i vout15xmnopqrstuvwxyz01 = _mm512_cvtsepi32_epi8(vacc15xmnopqrstuvwxyz01);

    vout0x0123456789ABCDEF = _mm_max_epi8(vout0x0123456789ABCDEF, voutput_min);
    vout0xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout0xGHIJKLMNOPQRSTUV, voutput_min);
    vout0xWXYZabcdefghijkl = _mm_max_epi8(vout0xWXYZabcdefghijkl, voutput_min);
    vout0xmnopqrstuvwxyz01 = _mm_max_epi8(vout0xmnopqrstuvwxyz01, voutput_min);
    vout1x0123456789ABCDEF = _mm_max_epi8(vout1x0123456789ABCDEF, voutput_min);
    vout1xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout1xGHIJKLMNOPQRSTUV, voutput_min);
    vout1xWXYZabcdefghijkl = _mm_max_epi8(vout1xWXYZabcdefghijkl, voutput_min);
    vout1xmnopqrstuvwxyz01 = _mm_max_epi8(vout1xmnopqrstuvwxyz01, voutput_min);
    vout2x0123456789ABCDEF = _mm_max_epi8(vout2x0123456789ABCDEF, voutput_min);
    vout2xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout2xGHIJKLMNOPQRSTUV, voutput_min);
    vout2xWXYZabcdefghijkl = _mm_max_epi8(vout2xWXYZabcdefghijkl, voutput_min);
    vout2xmnopqrstuvwxyz01 = _mm_max_epi8(vout2xmnopqrstuvwxyz01, voutput_min);
    vout3x0123456789ABCDEF = _mm_max_epi8(vout3x0123456789ABCDEF, voutput_min);
    vout3xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout3xGHIJKLMNOPQRSTUV, voutput_min);
    vout3xWXYZabcdefghijkl = _mm_max_epi8(vout3xWXYZabcdefghijkl, voutput_min);
    vout3xmnopqrstuvwxyz01 = _mm_max_epi8(vout3xmnopqrstuvwxyz01, voutput_min);
    vout4x0123456789ABCDEF = _mm_max_epi8(vout4x0123456789ABCDEF, voutput_min);
    vout4xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout4xGHIJKLMNOPQRSTUV, voutput_min);
    vout4xWXYZabcdefghijkl = _mm_max_epi8(vout4xWXYZabcdefghijkl, voutput_min);
    vout4xmnopqrstuvwxyz01 = _mm_max_epi8(vout4xmnopqrstuvwxyz01, voutput_min);
    vout5x0123456789ABCDEF = _mm_max_epi8(vout5x0123456789ABCDEF, voutput_min);
    vout5xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout5xGHIJKLMNOPQRSTUV, voutput_min);
    vout5xWXYZabcdefghijkl = _mm_max_epi8(vout5xWXYZabcdefghijkl, voutput_min);
    vout5xmnopqrstuvwxyz01 = _mm_max_epi8(vout5xmnopqrstuvwxyz01, voutput_min);
    vout6x0123456789ABCDEF = _mm_max_epi8(vout6x0123456789ABCDEF, voutput_min);
    vout6xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout6xGHIJKLMNOPQRSTUV, voutput_min);
    vout6xWXYZabcdefghijkl = _mm_max_epi8(vout6xWXYZabcdefghijkl, voutput_min);
    vout6xmnopqrstuvwxyz01 = _mm_max_epi8(vout6xmnopqrstuvwxyz01, voutput_min);
    vout7x0123456789ABCDEF = _mm_max_epi8(vout7x0123456789ABCDEF, voutput_min);
    vout7xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout7xGHIJKLMNOPQRSTUV, voutput_min);
    vout7xWXYZabcdefghijkl = _mm_max_epi8(vout7xWXYZabcdefghijkl, voutput_min);
    vout7xmnopqrstuvwxyz01 = _mm_max_epi8(vout7xmnopqrstuvwxyz01, voutput_min);
    vout8x0123456789ABCDEF = _mm_max_epi8(vout8x0123456789ABCDEF, voutput_min);
    vout8xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout8xGHIJKLMNOPQRSTUV, voutput_min);
    vout8xWXYZabcdefghijkl = _mm_max_epi8(vout8xWXYZabcdefghijkl, voutput_min);
    vout8xmnopqrstuvwxyz01 = _mm_max_epi8(vout8xmnopqrstuvwxyz01, voutput_min);
    vout9x0123456789ABCDEF = _mm_max_epi8(vout9x0123456789ABCDEF, voutput_min);
    vout9xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout9xGHIJKLMNOPQRSTUV, voutput_min);
    vout9xWXYZabcdefghijkl = _mm_max_epi8(vout9xWXYZabcdefghijkl, voutput_min);
    vout9xmnopqrstuvwxyz01 = _mm_max_epi8(vout9xmnopqrstuvwxyz01, voutput_min);
    vout10x0123456789ABCDEF = _mm_max_epi8(vout10x0123456789ABCDEF, voutput_min);
    vout10xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout10xGHIJKLMNOPQRSTUV, voutput_min);
    vout10xWXYZabcdefghijkl = _mm_max_epi8(vout10xWXYZabcdefghijkl, voutput_min);
    vout10xmnopqrstuvwxyz01 = _mm_max_epi8(vout10xmnopqrstuvwxyz01, voutput_min);
    vout11x0123456789ABCDEF = _mm_max_epi8(vout11x0123456789ABCDEF, voutput_min);
    vout11xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout11xGHIJKLMNOPQRSTUV, voutput_min);
    vout11xWXYZabcdefghijkl = _mm_max_epi8(vout11xWXYZabcdefghijkl, voutput_min);
    vout11xmnopqrstuvwxyz01 = _mm_max_epi8(vout11xmnopqrstuvwxyz01, voutput_min);
    vout12x0123456789ABCDEF = _mm_max_epi8(vout12x0123456789ABCDEF, voutput_min);
    vout12xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout12xGHIJKLMNOPQRSTUV, voutput_min);
    vout12xWXYZabcdefghijkl = _mm_max_epi8(vout12xWXYZabcdefghijkl, voutput_min);
    vout12xmnopqrstuvwxyz01 = _mm_max_epi8(vout12xmnopqrstuvwxyz01, voutput_min);
    vout13x0123456789ABCDEF = _mm_max_epi8(vout13x0123456789ABCDEF, voutput_min);
    vout13xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout13xGHIJKLMNOPQRSTUV, voutput_min);
    vout13xWXYZabcdefghijkl = _mm_max_epi8(vout13xWXYZabcdefghijkl, voutput_min);
    vout13xmnopqrstuvwxyz01 = _mm_max_epi8(vout13xmnopqrstuvwxyz01, voutput_min);
    vout14x0123456789ABCDEF = _mm_max_epi8(vout14x0123456789ABCDEF, voutput_min);
    vout14xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout14xGHIJKLMNOPQRSTUV, voutput_min);
    vout14xWXYZabcdefghijkl = _mm_max_epi8(vout14xWXYZabcdefghijkl, voutput_min);
    vout14xmnopqrstuvwxyz01 = _mm_max_epi8(vout14xmnopqrstuvwxyz01, voutput_min);
    vout15x0123456789ABCDEF = _mm_max_epi8(vout15x0123456789ABCDEF, voutput_min);
    vout15xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout15xGHIJKLMNOPQRSTUV, voutput_min);
    vout15xWXYZabcdefghijkl = _mm_max_epi8(vout15xWXYZabcdefghijkl, voutput_min);
    vout15xmnopqrstuvwxyz01 = _mm_max_epi8(vout15xmnopqrstuvwxyz01, voutput_min);

    if XNN_LIKELY(nc >= 64) {
      _mm_storeu_si128((__m128i*) (c15 + 0), vout15x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c15 + 16), vout15xGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (c15 + 32), vout15xWXYZabcdefghijkl);
      _mm_storeu_si128((__m128i*) (c15 + 48), vout15xmnopqrstuvwxyz01);
      c15 = (int8_t*) ((uintptr_t) c15 + cn_stride);
      _mm_storeu_si128((__m128i*) (c14 + 0), vout14x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c14 + 16), vout14xGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (c14 + 32), vout14xWXYZabcdefghijkl);
      _mm_storeu_si128((__m128i*) (c14 + 48), vout14xmnopqrstuvwxyz01);
      c14 = (int8_t*) ((uintptr_t) c14 + cn_stride);
      _mm_storeu_si128((__m128i*) (c13 + 0), vout13x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c13 + 16), vout13xGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (c13 + 32), vout13xWXYZabcdefghijkl);
      _mm_storeu_si128((__m128i*) (c13 + 48), vout13xmnopqrstuvwxyz01);
      c13 = (int8_t*) ((uintptr_t) c13 + cn_stride);
      _mm_storeu_si128((__m128i*) (c12 + 0), vout12x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c12 + 16), vout12xGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (c12 + 32), vout12xWXYZabcdefghijkl);
      _mm_storeu_si128((__m128i*) (c12 + 48), vout12xmnopqrstuvwxyz01);
      c12 = (int8_t*) ((uintptr_t) c12 + cn_stride);
      _mm_storeu_si128((__m128i*) (c11 + 0), vout11x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c11 + 16), vout11xGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (c11 + 32), vout11xWXYZabcdefghijkl);
      _mm_storeu_si128((__m128i*) (c11 + 48), vout11xmnopqrstuvwxyz01);
      c11 = (int8_t*) ((uintptr_t) c11 + cn_stride);
      _mm_storeu_si128((__m128i*) (c10 + 0), vout10x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c10 + 16), vout10xGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (c10 + 32), vout10xWXYZabcdefghijkl);
      _mm_storeu_si128((__m128i*) (c10 + 48), vout10xmnopqrstuvwxyz01);
      c10 = (int8_t*) ((uintptr_t) c10 + cn_stride);
      _mm_storeu_si128((__m128i*) (c9 + 0), vout9x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c9 + 16), vout9xGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (c9 + 32), vout9xWXYZabcdefghijkl);
      _mm_storeu_si128((__m128i*) (c9 + 48), vout9xmnopqrstuvwxyz01);
      c9 = (int8_t*) ((uintptr_t) c9 + cn_stride);
      _mm_storeu_si128((__m128i*) (c8 + 0), vout8x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c8 + 16), vout8xGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (c8 + 32), vout8xWXYZabcdefghijkl);
      _mm_storeu_si128((__m128i*) (c8 + 48), vout8xmnopqrstuvwxyz01);
      c8 = (int8_t*) ((uintptr_t) c8 + cn_stride);
      _mm_storeu_si128((__m128i*) (c7 + 0), vout7x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c7 + 16), vout7xGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (c7 + 32), vout7xWXYZabcdefghijkl);
      _mm_storeu_si128((__m128i*) (c7 + 48), vout7xmnopqrstuvwxyz01);
      c7 = (int8_t*) ((uintptr_t) c7 + cn_stride);
      _mm_storeu_si128((__m128i*) (c6 + 0), vout6x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c6 + 16), vout6xGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (c6 + 32), vout6xWXYZabcdefghijkl);
      _mm_storeu_si128((__m128i*) (c6 + 48), vout6xmnopqrstuvwxyz01);
      c6 = (int8_t*) ((uintptr_t) c6 + cn_stride);
      _mm_storeu_si128((__m128i*) (c5 + 0), vout5x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c5 + 16), vout5xGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (c5 + 32), vout5xWXYZabcdefghijkl);
      _mm_storeu_si128((__m128i*) (c5 + 48), vout5xmnopqrstuvwxyz01);
      c5 = (int8_t*) ((uintptr_t) c5 + cn_stride);
      _mm_storeu_si128((__m128i*) (c4 + 0), vout4x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c4 + 16), vout4xGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (c4 + 32), vout4xWXYZabcdefghijkl);
      _mm_storeu_si128((__m128i*) (c4 + 48), vout4xmnopqrstuvwxyz01);
      c4 = (int8_t*) ((uintptr_t) c4 + cn_stride);
      _mm_storeu_si128((__m128i*) (c3 + 0), vout3x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c3 + 16), vout3xGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (c3 + 32), vout3xWXYZabcdefghijkl);
      _mm_storeu_si128((__m128i*) (c3 + 48), vout3xmnopqrstuvwxyz01);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      _mm_storeu_si128((__m128i*) (c2 + 0), vout2x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c2 + 16), vout2xGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (c2 + 32), vout2xWXYZabcdefghijkl);
      _mm_storeu_si128((__m128i*) (c2 + 48), vout2xmnopqrstuvwxyz01);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      _mm_storeu_si128((__m128i*) (c1 + 0), vout1x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c1 + 16), vout1xGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (c1 + 32), vout1xWXYZabcdefghijkl);
      _mm_storeu_si128((__m128i*) (c1 + 48), vout1xmnopqrstuvwxyz01);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      _mm_storeu_si128((__m128i*) (c0 + 0), vout0x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c0 + 16), vout0xGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (c0 + 32), vout0xWXYZabcdefghijkl);
      _mm_storeu_si128((__m128i*) (c0 + 48), vout0xmnopqrstuvwxyz01);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      a -= kc;
      nc -= 64;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 0) & 0xFFFF));
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 16) & 0xFFFF));
      const __mmask16 vmask2 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 32) & 0xFFFF));
      const __mmask16 vmask3 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 48) & 0xFFFF));

      _mm_mask_storeu_epi8(c15 + 0, vmask0, vout15x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c15 + 16, vmask1, vout15xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c15 + 32, vmask2, vout15xWXYZabcdefghijkl);
      _mm_mask_storeu_epi8(c15 + 48, vmask3, vout15xmnopqrstuvwxyz01);
      _mm_mask_storeu_epi8(c14 + 0, vmask0, vout14x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c14 + 16, vmask1, vout14xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c14 + 32, vmask2, vout14xWXYZabcdefghijkl);
      _mm_mask_storeu_epi8(c14 + 48, vmask3, vout14xmnopqrstuvwxyz01);
      _mm_mask_storeu_epi8(c13 + 0, vmask0, vout13x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c13 + 16, vmask1, vout13xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c13 + 32, vmask2, vout13xWXYZabcdefghijkl);
      _mm_mask_storeu_epi8(c13 + 48, vmask3, vout13xmnopqrstuvwxyz01);
      _mm_mask_storeu_epi8(c12 + 0, vmask0, vout12x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c12 + 16, vmask1, vout12xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c12 + 32, vmask2, vout12xWXYZabcdefghijkl);
      _mm_mask_storeu_epi8(c12 + 48, vmask3, vout12xmnopqrstuvwxyz01);
      _mm_mask_storeu_epi8(c11 + 0, vmask0, vout11x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c11 + 16, vmask1, vout11xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c11 + 32, vmask2, vout11xWXYZabcdefghijkl);
      _mm_mask_storeu_epi8(c11 + 48, vmask3, vout11xmnopqrstuvwxyz01);
      _mm_mask_storeu_epi8(c10 + 0, vmask0, vout10x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c10 + 16, vmask1, vout10xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c10 + 32, vmask2, vout10xWXYZabcdefghijkl);
      _mm_mask_storeu_epi8(c10 + 48, vmask3, vout10xmnopqrstuvwxyz01);
      _mm_mask_storeu_epi8(c9 + 0, vmask0, vout9x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c9 + 16, vmask1, vout9xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c9 + 32, vmask2, vout9xWXYZabcdefghijkl);
      _mm_mask_storeu_epi8(c9 + 48, vmask3, vout9xmnopqrstuvwxyz01);
      _mm_mask_storeu_epi8(c8 + 0, vmask0, vout8x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c8 + 16, vmask1, vout8xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c8 + 32, vmask2, vout8xWXYZabcdefghijkl);
      _mm_mask_storeu_epi8(c8 + 48, vmask3, vout8xmnopqrstuvwxyz01);
      _mm_mask_storeu_epi8(c7 + 0, vmask0, vout7x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c7 + 16, vmask1, vout7xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c7 + 32, vmask2, vout7xWXYZabcdefghijkl);
      _mm_mask_storeu_epi8(c7 + 48, vmask3, vout7xmnopqrstuvwxyz01);
      _mm_mask_storeu_epi8(c6 + 0, vmask0, vout6x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c6 + 16, vmask1, vout6xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c6 + 32, vmask2, vout6xWXYZabcdefghijkl);
      _mm_mask_storeu_epi8(c6 + 48, vmask3, vout6xmnopqrstuvwxyz01);
      _mm_mask_storeu_epi8(c5 + 0, vmask0, vout5x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c5 + 16, vmask1, vout5xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c5 + 32, vmask2, vout5xWXYZabcdefghijkl);
      _mm_mask_storeu_epi8(c5 + 48, vmask3, vout5xmnopqrstuvwxyz01);
      _mm_mask_storeu_epi8(c4 + 0, vmask0, vout4x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c4 + 16, vmask1, vout4xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c4 + 32, vmask2, vout4xWXYZabcdefghijkl);
      _mm_mask_storeu_epi8(c4 + 48, vmask3, vout4xmnopqrstuvwxyz01);
      _mm_mask_storeu_epi8(c3 + 0, vmask0, vout3x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c3 + 16, vmask1, vout3xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c3 + 32, vmask2, vout3xWXYZabcdefghijkl);
      _mm_mask_storeu_epi8(c3 + 48, vmask3, vout3xmnopqrstuvwxyz01);
      _mm_mask_storeu_epi8(c2 + 0, vmask0, vout2x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c2 + 16, vmask1, vout2xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c2 + 32, vmask2, vout2xWXYZabcdefghijkl);
      _mm_mask_storeu_epi8(c2 + 48, vmask3, vout2xmnopqrstuvwxyz01);
      _mm_mask_storeu_epi8(c1 + 0, vmask0, vout1x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c1 + 16, vmask1, vout1xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c1 + 32, vmask2, vout1xWXYZabcdefghijkl);
      _mm_mask_storeu_epi8(c1 + 48, vmask3, vout1xmnopqrstuvwxyz01);
      _mm_mask_storeu_epi8(c0 + 0, vmask0, vout0x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c0 + 16, vmask1, vout0xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c0 + 32, vmask2, vout0xWXYZabcdefghijkl);
      _mm_mask_storeu_epi8(c0 + 48, vmask3, vout0xmnopqrstuvwxyz01);
      nc = 0;
    }
  } while (nc != 0);
  // Release tile config
  //  _tile_release();
  __asm__ volatile ("tilerelease" ::);
  #endif  // defined(__x86_64__)
}
