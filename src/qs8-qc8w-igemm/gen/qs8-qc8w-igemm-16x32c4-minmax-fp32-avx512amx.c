// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/c4-avx512amx.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/gemm.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/unaligned.h"


void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_16x32c4__avx512amx(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
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
  __attribute__((aligned(64))) int32_t vintile[16 * 16];
  __attribute__((aligned(64))) int32_t res0[16 * 16];
  __attribute__((aligned(64))) int32_t res1[16 * 16];

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  const size_t kremainder = (kc & 63) ? (kc & 63) : 64;
  const __mmask16 kremainder_mask = _cvtu32_mask16((UINT32_C(1) << (kremainder >> 2)) - 1);

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
    const __m512i vksum0123456789ABCDEF = _mm512_loadu_epi32((const int32_t*) w + 0);
    const __m512i vksumGHIJKLMNOPQRSTUV = _mm512_loadu_epi32((const int32_t*) w + 16);
    w = (const int32_t*) w + 32;

    // Zero tile accumulator
    __asm__ volatile (
      "tilezero %%tmm0\n"
      "tilezero %%tmm1\n"
      ::);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      }
      const int8_t* restrict a4 = a[4];
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const int8_t*) ((uintptr_t) a4 + a_offset);
      }
      const int8_t* restrict a5 = a[5];
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const int8_t*) ((uintptr_t) a5 + a_offset);
      }
      const int8_t* restrict a6 = a[6];
      if XNN_UNPREDICTABLE(a6 != zero) {
        a6 = (const int8_t*) ((uintptr_t) a6 + a_offset);
      }
      const int8_t* restrict a7 = a[7];
      if XNN_UNPREDICTABLE(a7 != zero) {
        a7 = (const int8_t*) ((uintptr_t) a7 + a_offset);
      }
      const int8_t* restrict a8 = a[8];
      if XNN_UNPREDICTABLE(a8 != zero) {
        a8 = (const int8_t*) ((uintptr_t) a8 + a_offset);
      }
      const int8_t* restrict a9 = a[9];
      if XNN_UNPREDICTABLE(a9 != zero) {
        a9 = (const int8_t*) ((uintptr_t) a9 + a_offset);
      }
      const int8_t* restrict a10 = a[10];
      if XNN_UNPREDICTABLE(a10 != zero) {
        a10 = (const int8_t*) ((uintptr_t) a10 + a_offset);
      }
      const int8_t* restrict a11 = a[11];
      if XNN_UNPREDICTABLE(a11 != zero) {
        a11 = (const int8_t*) ((uintptr_t) a11 + a_offset);
      }
      const int8_t* restrict a12 = a[12];
      if XNN_UNPREDICTABLE(a12 != zero) {
        a12 = (const int8_t*) ((uintptr_t) a12 + a_offset);
      }
      const int8_t* restrict a13 = a[13];
      if XNN_UNPREDICTABLE(a13 != zero) {
        a13 = (const int8_t*) ((uintptr_t) a13 + a_offset);
      }
      const int8_t* restrict a14 = a[14];
      if XNN_UNPREDICTABLE(a14 != zero) {
        a14 = (const int8_t*) ((uintptr_t) a14 + a_offset);
      }
      const int8_t* restrict a15 = a[15];
      if XNN_UNPREDICTABLE(a15 != zero) {
        a15 = (const int8_t*) ((uintptr_t) a15 + a_offset);
      }
      a += 16;

      size_t k = kc;
      if (mr == 1)
      {
        while (k >= 64 * sizeof(int8_t)) {
          _tile_loadd(4, a0, 64);   // Directly load input for mr=1
          a15 += 64;
          _tile_loadd(5, (const int8_t*) w + 0, 128);
          _tile_dpbssd(0, 4, 5);
          _tile_loadd(5, (const int8_t*) w + 64, 128);
          _tile_dpbssd(1, 4, 5);

          w = (const int8_t*) w + 2048;
          k -= 64 * sizeof(int8_t);
        }
      }
      else {
        while (k >= 64 * sizeof(int8_t)) {
          const __m512i vin0 = _mm512_loadu_epi32(a0);
          a0 += 64;
          _mm512_store_epi32(vintile + 0, vin0);
          const __m512i vin1 = _mm512_loadu_epi32(a1);
          a1 += 64;
          _mm512_store_epi32(vintile + 16, vin1);
          const __m512i vin2 = _mm512_loadu_epi32(a2);
          a2 += 64;
          _mm512_store_epi32(vintile + 32, vin2);
          const __m512i vin3 = _mm512_loadu_epi32(a3);
          a3 += 64;
          _mm512_store_epi32(vintile + 48, vin3);
          const __m512i vin4 = _mm512_loadu_epi32(a4);
          a4 += 64;
          _mm512_store_epi32(vintile + 64, vin4);
          const __m512i vin5 = _mm512_loadu_epi32(a5);
          a5 += 64;
          _mm512_store_epi32(vintile + 80, vin5);
          const __m512i vin6 = _mm512_loadu_epi32(a6);
          a6 += 64;
          _mm512_store_epi32(vintile + 96, vin6);
          const __m512i vin7 = _mm512_loadu_epi32(a7);
          a7 += 64;
          _mm512_store_epi32(vintile + 112, vin7);
          const __m512i vin8 = _mm512_loadu_epi32(a8);
          a8 += 64;
          _mm512_store_epi32(vintile + 128, vin8);
          const __m512i vin9 = _mm512_loadu_epi32(a9);
          a9 += 64;
          _mm512_store_epi32(vintile + 144, vin9);
          const __m512i vin10 = _mm512_loadu_epi32(a10);
          a10 += 64;
          _mm512_store_epi32(vintile + 160, vin10);
          const __m512i vin11 = _mm512_loadu_epi32(a11);
          a11 += 64;
          _mm512_store_epi32(vintile + 176, vin11);
          const __m512i vin12 = _mm512_loadu_epi32(a12);
          a12 += 64;
          _mm512_store_epi32(vintile + 192, vin12);
          const __m512i vin13 = _mm512_loadu_epi32(a13);
          a13 += 64;
          _mm512_store_epi32(vintile + 208, vin13);
          const __m512i vin14 = _mm512_loadu_epi32(a14);
          a14 += 64;
          _mm512_store_epi32(vintile + 224, vin14);
          const __m512i vin15 = _mm512_loadu_epi32(a15);
          a15 += 64;
          _mm512_store_epi32(vintile + 240, vin15);
          _tile_loadd(4, vintile, 64);
          _tile_loadd(5, (const int8_t*) w + 0, 128);
          _tile_dpbssd(0, 4, 5);
          _tile_loadd(5, (const int8_t*) w + 64, 128);
          _tile_dpbssd(1, 4, 5);

          w = (const int8_t*) w + 2048;
          k -= 64 * sizeof(int8_t);
        }
      }

      if XNN_UNLIKELY(k != 0) {
        const __m512i vin0 = _mm512_maskz_loadu_epi32(kremainder_mask, a0);
        a0 += kremainder;
        _mm512_store_epi32(vintile + 0, vin0);
        const __m512i vin1 = _mm512_maskz_loadu_epi32(kremainder_mask, a1);
        a1 += kremainder;
        _mm512_store_epi32(vintile + 16, vin1);
        const __m512i vin2 = _mm512_maskz_loadu_epi32(kremainder_mask, a2);
        a2 += kremainder;
        _mm512_store_epi32(vintile + 32, vin2);
        const __m512i vin3 = _mm512_maskz_loadu_epi32(kremainder_mask, a3);
        a3 += kremainder;
        _mm512_store_epi32(vintile + 48, vin3);
        const __m512i vin4 = _mm512_maskz_loadu_epi32(kremainder_mask, a4);
        a4 += kremainder;
        _mm512_store_epi32(vintile + 64, vin4);
        const __m512i vin5 = _mm512_maskz_loadu_epi32(kremainder_mask, a5);
        a5 += kremainder;
        _mm512_store_epi32(vintile + 80, vin5);
        const __m512i vin6 = _mm512_maskz_loadu_epi32(kremainder_mask, a6);
        a6 += kremainder;
        _mm512_store_epi32(vintile + 96, vin6);
        const __m512i vin7 = _mm512_maskz_loadu_epi32(kremainder_mask, a7);
        a7 += kremainder;
        _mm512_store_epi32(vintile + 112, vin7);
        const __m512i vin8 = _mm512_maskz_loadu_epi32(kremainder_mask, a8);
        a8 += kremainder;
        _mm512_store_epi32(vintile + 128, vin8);
        const __m512i vin9 = _mm512_maskz_loadu_epi32(kremainder_mask, a9);
        a9 += kremainder;
        _mm512_store_epi32(vintile + 144, vin9);
        const __m512i vin10 = _mm512_maskz_loadu_epi32(kremainder_mask, a10);
        a10 += kremainder;
        _mm512_store_epi32(vintile + 160, vin10);
        const __m512i vin11 = _mm512_maskz_loadu_epi32(kremainder_mask, a11);
        a11 += kremainder;
        _mm512_store_epi32(vintile + 176, vin11);
        const __m512i vin12 = _mm512_maskz_loadu_epi32(kremainder_mask, a12);
        a12 += kremainder;
        _mm512_store_epi32(vintile + 192, vin12);
        const __m512i vin13 = _mm512_maskz_loadu_epi32(kremainder_mask, a13);
        a13 += kremainder;
        _mm512_store_epi32(vintile + 208, vin13);
        const __m512i vin14 = _mm512_maskz_loadu_epi32(kremainder_mask, a14);
        a14 += kremainder;
        _mm512_store_epi32(vintile + 224, vin14);
        const __m512i vin15 = _mm512_maskz_loadu_epi32(kremainder_mask, a15);
        a15 += kremainder;
        _mm512_store_epi32(vintile + 240, vin15);
        _tile_loadd(6, vintile, 64);
        _tile_loadd(7, (const int8_t*) w + 0, 128);
        _tile_dpbssd(0, 6, 7);
        _tile_loadd(7, (const int8_t*) w + 64, 128);
        _tile_dpbssd(1, 6, 7);

        w = (const int8_t*) w + kremainder * 32;
        k -= kremainder * sizeof(int8_t);
      }

      p -= 16 * sizeof(void*);
    } while (p != 0);

    // Add tile to bias
    _tile_stored(0, res0, 64);
    _tile_stored(1, res1, 64);

    __m512i vacc0x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 0));
    __m512i vacc0xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 0));
    __m512i vacc1x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 16));
    __m512i vacc1xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 16));
    __m512i vacc2x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 32));
    __m512i vacc2xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 32));
    __m512i vacc3x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 48));
    __m512i vacc3xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 48));
    __m512i vacc4x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 64));
    __m512i vacc4xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 64));
    __m512i vacc5x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 80));
    __m512i vacc5xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 80));
    __m512i vacc6x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 96));
    __m512i vacc6xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 96));
    __m512i vacc7x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 112));
    __m512i vacc7xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 112));
    __m512i vacc8x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 128));
    __m512i vacc8xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 128));
    __m512i vacc9x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 144));
    __m512i vacc9xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 144));
    __m512i vacc10x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 160));
    __m512i vacc10xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 160));
    __m512i vacc11x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 176));
    __m512i vacc11xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 176));
    __m512i vacc12x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 192));
    __m512i vacc12xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 192));
    __m512i vacc13x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 208));
    __m512i vacc13xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 208));
    __m512i vacc14x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 224));
    __m512i vacc14xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 224));
    __m512i vacc15x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 240));
    __m512i vacc15xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 240));

    __m512 vscaled0x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0x0123456789ABCDEF);
    __m512 vscaled0xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc0xGHIJKLMNOPQRSTUV);
    __m512 vscaled1x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc1x0123456789ABCDEF);
    __m512 vscaled1xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc1xGHIJKLMNOPQRSTUV);
    __m512 vscaled2x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc2x0123456789ABCDEF);
    __m512 vscaled2xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc2xGHIJKLMNOPQRSTUV);
    __m512 vscaled3x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc3x0123456789ABCDEF);
    __m512 vscaled3xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc3xGHIJKLMNOPQRSTUV);
    __m512 vscaled4x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc4x0123456789ABCDEF);
    __m512 vscaled4xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc4xGHIJKLMNOPQRSTUV);
    __m512 vscaled5x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc5x0123456789ABCDEF);
    __m512 vscaled5xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc5xGHIJKLMNOPQRSTUV);
    __m512 vscaled6x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc6x0123456789ABCDEF);
    __m512 vscaled6xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc6xGHIJKLMNOPQRSTUV);
    __m512 vscaled7x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc7x0123456789ABCDEF);
    __m512 vscaled7xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc7xGHIJKLMNOPQRSTUV);
    __m512 vscaled8x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc8x0123456789ABCDEF);
    __m512 vscaled8xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc8xGHIJKLMNOPQRSTUV);
    __m512 vscaled9x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc9x0123456789ABCDEF);
    __m512 vscaled9xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc9xGHIJKLMNOPQRSTUV);
    __m512 vscaled10x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc10x0123456789ABCDEF);
    __m512 vscaled10xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc10xGHIJKLMNOPQRSTUV);
    __m512 vscaled11x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc11x0123456789ABCDEF);
    __m512 vscaled11xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc11xGHIJKLMNOPQRSTUV);
    __m512 vscaled12x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc12x0123456789ABCDEF);
    __m512 vscaled12xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc12xGHIJKLMNOPQRSTUV);
    __m512 vscaled13x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc13x0123456789ABCDEF);
    __m512 vscaled13xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc13xGHIJKLMNOPQRSTUV);
    __m512 vscaled14x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc14x0123456789ABCDEF);
    __m512 vscaled14xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc14xGHIJKLMNOPQRSTUV);
    __m512 vscaled15x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc15x0123456789ABCDEF);
    __m512 vscaled15xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc15xGHIJKLMNOPQRSTUV);

    const __m512 vscale0123456789ABCDEF = _mm512_loadu_ps((const float*) w + 0);
    const __m512 vscaleGHIJKLMNOPQRSTUV = _mm512_loadu_ps((const float*) w + 16);
    w = (const int32_t*) w + 32;

    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled0xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled1x0123456789ABCDEF = _mm512_mul_ps(vscaled1x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled1xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled1xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled2x0123456789ABCDEF = _mm512_mul_ps(vscaled2x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled2xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled2xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled3x0123456789ABCDEF = _mm512_mul_ps(vscaled3x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled3xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled3xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled4x0123456789ABCDEF = _mm512_mul_ps(vscaled4x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled4xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled4xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled5x0123456789ABCDEF = _mm512_mul_ps(vscaled5x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled5xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled5xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled6x0123456789ABCDEF = _mm512_mul_ps(vscaled6x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled6xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled6xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled7x0123456789ABCDEF = _mm512_mul_ps(vscaled7x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled7xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled7xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled8x0123456789ABCDEF = _mm512_mul_ps(vscaled8x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled8xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled8xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled9x0123456789ABCDEF = _mm512_mul_ps(vscaled9x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled9xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled9xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled10x0123456789ABCDEF = _mm512_mul_ps(vscaled10x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled10xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled10xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled11x0123456789ABCDEF = _mm512_mul_ps(vscaled11x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled11xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled11xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled12x0123456789ABCDEF = _mm512_mul_ps(vscaled12x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled12xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled12xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled13x0123456789ABCDEF = _mm512_mul_ps(vscaled13x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled13xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled13xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled14x0123456789ABCDEF = _mm512_mul_ps(vscaled14x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled14xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled14xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled15x0123456789ABCDEF = _mm512_mul_ps(vscaled15x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled15xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled15xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled0xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled1x0123456789ABCDEF = _mm512_min_ps(vscaled1x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled1xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled1xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled2x0123456789ABCDEF = _mm512_min_ps(vscaled2x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled2xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled2xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled3x0123456789ABCDEF = _mm512_min_ps(vscaled3x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled3xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled3xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled4x0123456789ABCDEF = _mm512_min_ps(vscaled4x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled4xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled4xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled5x0123456789ABCDEF = _mm512_min_ps(vscaled5x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled5xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled5xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled6x0123456789ABCDEF = _mm512_min_ps(vscaled6x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled6xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled6xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled7x0123456789ABCDEF = _mm512_min_ps(vscaled7x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled7xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled7xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled8x0123456789ABCDEF = _mm512_min_ps(vscaled8x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled8xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled8xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled9x0123456789ABCDEF = _mm512_min_ps(vscaled9x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled9xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled9xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled10x0123456789ABCDEF = _mm512_min_ps(vscaled10x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled10xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled10xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled11x0123456789ABCDEF = _mm512_min_ps(vscaled11x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled11xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled11xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled12x0123456789ABCDEF = _mm512_min_ps(vscaled12x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled12xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled12xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled13x0123456789ABCDEF = _mm512_min_ps(vscaled13x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled13xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled13xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled14x0123456789ABCDEF = _mm512_min_ps(vscaled14x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled14xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled14xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled15x0123456789ABCDEF = _mm512_min_ps(vscaled15x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled15xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled15xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);

    vacc0x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0x0123456789ABCDEF);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled0xGHIJKLMNOPQRSTUV);
    vacc1x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled1x0123456789ABCDEF);
    vacc1xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled1xGHIJKLMNOPQRSTUV);
    vacc2x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled2x0123456789ABCDEF);
    vacc2xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled2xGHIJKLMNOPQRSTUV);
    vacc3x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled3x0123456789ABCDEF);
    vacc3xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled3xGHIJKLMNOPQRSTUV);
    vacc4x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled4x0123456789ABCDEF);
    vacc4xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled4xGHIJKLMNOPQRSTUV);
    vacc5x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled5x0123456789ABCDEF);
    vacc5xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled5xGHIJKLMNOPQRSTUV);
    vacc6x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled6x0123456789ABCDEF);
    vacc6xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled6xGHIJKLMNOPQRSTUV);
    vacc7x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled7x0123456789ABCDEF);
    vacc7xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled7xGHIJKLMNOPQRSTUV);
    vacc8x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled8x0123456789ABCDEF);
    vacc8xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled8xGHIJKLMNOPQRSTUV);
    vacc9x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled9x0123456789ABCDEF);
    vacc9xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled9xGHIJKLMNOPQRSTUV);
    vacc10x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled10x0123456789ABCDEF);
    vacc10xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled10xGHIJKLMNOPQRSTUV);
    vacc11x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled11x0123456789ABCDEF);
    vacc11xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled11xGHIJKLMNOPQRSTUV);
    vacc12x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled12x0123456789ABCDEF);
    vacc12xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled12xGHIJKLMNOPQRSTUV);
    vacc13x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled13x0123456789ABCDEF);
    vacc13xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled13xGHIJKLMNOPQRSTUV);
    vacc14x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled14x0123456789ABCDEF);
    vacc14xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled14xGHIJKLMNOPQRSTUV);
    vacc15x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled15x0123456789ABCDEF);
    vacc15xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled15xGHIJKLMNOPQRSTUV);

    vacc0x0123456789ABCDEF = _mm512_add_epi32(vacc0x0123456789ABCDEF, voutput_zero_point);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc0xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc1x0123456789ABCDEF = _mm512_add_epi32(vacc1x0123456789ABCDEF, voutput_zero_point);
    vacc1xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc1xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc2x0123456789ABCDEF = _mm512_add_epi32(vacc2x0123456789ABCDEF, voutput_zero_point);
    vacc2xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc2xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc3x0123456789ABCDEF = _mm512_add_epi32(vacc3x0123456789ABCDEF, voutput_zero_point);
    vacc3xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc3xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc4x0123456789ABCDEF = _mm512_add_epi32(vacc4x0123456789ABCDEF, voutput_zero_point);
    vacc4xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc4xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc5x0123456789ABCDEF = _mm512_add_epi32(vacc5x0123456789ABCDEF, voutput_zero_point);
    vacc5xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc5xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc6x0123456789ABCDEF = _mm512_add_epi32(vacc6x0123456789ABCDEF, voutput_zero_point);
    vacc6xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc6xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc7x0123456789ABCDEF = _mm512_add_epi32(vacc7x0123456789ABCDEF, voutput_zero_point);
    vacc7xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc7xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc8x0123456789ABCDEF = _mm512_add_epi32(vacc8x0123456789ABCDEF, voutput_zero_point);
    vacc8xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc8xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc9x0123456789ABCDEF = _mm512_add_epi32(vacc9x0123456789ABCDEF, voutput_zero_point);
    vacc9xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc9xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc10x0123456789ABCDEF = _mm512_add_epi32(vacc10x0123456789ABCDEF, voutput_zero_point);
    vacc10xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc10xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc11x0123456789ABCDEF = _mm512_add_epi32(vacc11x0123456789ABCDEF, voutput_zero_point);
    vacc11xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc11xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc12x0123456789ABCDEF = _mm512_add_epi32(vacc12x0123456789ABCDEF, voutput_zero_point);
    vacc12xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc12xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc13x0123456789ABCDEF = _mm512_add_epi32(vacc13x0123456789ABCDEF, voutput_zero_point);
    vacc13xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc13xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc14x0123456789ABCDEF = _mm512_add_epi32(vacc14x0123456789ABCDEF, voutput_zero_point);
    vacc14xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc14xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc15x0123456789ABCDEF = _mm512_add_epi32(vacc15x0123456789ABCDEF, voutput_zero_point);
    vacc15xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc15xGHIJKLMNOPQRSTUV, voutput_zero_point);

    __m128i vout0x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc0x0123456789ABCDEF);
    __m128i vout0xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc0xGHIJKLMNOPQRSTUV);
    __m128i vout1x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc1x0123456789ABCDEF);
    __m128i vout1xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc1xGHIJKLMNOPQRSTUV);
    __m128i vout2x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc2x0123456789ABCDEF);
    __m128i vout2xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc2xGHIJKLMNOPQRSTUV);
    __m128i vout3x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc3x0123456789ABCDEF);
    __m128i vout3xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc3xGHIJKLMNOPQRSTUV);
    __m128i vout4x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc4x0123456789ABCDEF);
    __m128i vout4xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc4xGHIJKLMNOPQRSTUV);
    __m128i vout5x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc5x0123456789ABCDEF);
    __m128i vout5xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc5xGHIJKLMNOPQRSTUV);
    __m128i vout6x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc6x0123456789ABCDEF);
    __m128i vout6xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc6xGHIJKLMNOPQRSTUV);
    __m128i vout7x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc7x0123456789ABCDEF);
    __m128i vout7xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc7xGHIJKLMNOPQRSTUV);
    __m128i vout8x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc8x0123456789ABCDEF);
    __m128i vout8xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc8xGHIJKLMNOPQRSTUV);
    __m128i vout9x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc9x0123456789ABCDEF);
    __m128i vout9xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc9xGHIJKLMNOPQRSTUV);
    __m128i vout10x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc10x0123456789ABCDEF);
    __m128i vout10xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc10xGHIJKLMNOPQRSTUV);
    __m128i vout11x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc11x0123456789ABCDEF);
    __m128i vout11xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc11xGHIJKLMNOPQRSTUV);
    __m128i vout12x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc12x0123456789ABCDEF);
    __m128i vout12xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc12xGHIJKLMNOPQRSTUV);
    __m128i vout13x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc13x0123456789ABCDEF);
    __m128i vout13xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc13xGHIJKLMNOPQRSTUV);
    __m128i vout14x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc14x0123456789ABCDEF);
    __m128i vout14xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc14xGHIJKLMNOPQRSTUV);
    __m128i vout15x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc15x0123456789ABCDEF);
    __m128i vout15xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc15xGHIJKLMNOPQRSTUV);

    vout0x0123456789ABCDEF = _mm_max_epi8(vout0x0123456789ABCDEF, voutput_min);
    vout0xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout0xGHIJKLMNOPQRSTUV, voutput_min);
    vout1x0123456789ABCDEF = _mm_max_epi8(vout1x0123456789ABCDEF, voutput_min);
    vout1xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout1xGHIJKLMNOPQRSTUV, voutput_min);
    vout2x0123456789ABCDEF = _mm_max_epi8(vout2x0123456789ABCDEF, voutput_min);
    vout2xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout2xGHIJKLMNOPQRSTUV, voutput_min);
    vout3x0123456789ABCDEF = _mm_max_epi8(vout3x0123456789ABCDEF, voutput_min);
    vout3xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout3xGHIJKLMNOPQRSTUV, voutput_min);
    vout4x0123456789ABCDEF = _mm_max_epi8(vout4x0123456789ABCDEF, voutput_min);
    vout4xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout4xGHIJKLMNOPQRSTUV, voutput_min);
    vout5x0123456789ABCDEF = _mm_max_epi8(vout5x0123456789ABCDEF, voutput_min);
    vout5xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout5xGHIJKLMNOPQRSTUV, voutput_min);
    vout6x0123456789ABCDEF = _mm_max_epi8(vout6x0123456789ABCDEF, voutput_min);
    vout6xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout6xGHIJKLMNOPQRSTUV, voutput_min);
    vout7x0123456789ABCDEF = _mm_max_epi8(vout7x0123456789ABCDEF, voutput_min);
    vout7xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout7xGHIJKLMNOPQRSTUV, voutput_min);
    vout8x0123456789ABCDEF = _mm_max_epi8(vout8x0123456789ABCDEF, voutput_min);
    vout8xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout8xGHIJKLMNOPQRSTUV, voutput_min);
    vout9x0123456789ABCDEF = _mm_max_epi8(vout9x0123456789ABCDEF, voutput_min);
    vout9xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout9xGHIJKLMNOPQRSTUV, voutput_min);
    vout10x0123456789ABCDEF = _mm_max_epi8(vout10x0123456789ABCDEF, voutput_min);
    vout10xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout10xGHIJKLMNOPQRSTUV, voutput_min);
    vout11x0123456789ABCDEF = _mm_max_epi8(vout11x0123456789ABCDEF, voutput_min);
    vout11xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout11xGHIJKLMNOPQRSTUV, voutput_min);
    vout12x0123456789ABCDEF = _mm_max_epi8(vout12x0123456789ABCDEF, voutput_min);
    vout12xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout12xGHIJKLMNOPQRSTUV, voutput_min);
    vout13x0123456789ABCDEF = _mm_max_epi8(vout13x0123456789ABCDEF, voutput_min);
    vout13xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout13xGHIJKLMNOPQRSTUV, voutput_min);
    vout14x0123456789ABCDEF = _mm_max_epi8(vout14x0123456789ABCDEF, voutput_min);
    vout14xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout14xGHIJKLMNOPQRSTUV, voutput_min);
    vout15x0123456789ABCDEF = _mm_max_epi8(vout15x0123456789ABCDEF, voutput_min);
    vout15xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout15xGHIJKLMNOPQRSTUV, voutput_min);

    if XNN_LIKELY(nc >= 32) {
      _mm_storeu_si128((__m128i*) (c15 + 0), vout15x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c15 + 16), vout15xGHIJKLMNOPQRSTUV);
      c15 = (int8_t*) ((uintptr_t) c15 + cn_stride);
      _mm_storeu_si128((__m128i*) (c14 + 0), vout14x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c14 + 16), vout14xGHIJKLMNOPQRSTUV);
      c14 = (int8_t*) ((uintptr_t) c14 + cn_stride);
      _mm_storeu_si128((__m128i*) (c13 + 0), vout13x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c13 + 16), vout13xGHIJKLMNOPQRSTUV);
      c13 = (int8_t*) ((uintptr_t) c13 + cn_stride);
      _mm_storeu_si128((__m128i*) (c12 + 0), vout12x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c12 + 16), vout12xGHIJKLMNOPQRSTUV);
      c12 = (int8_t*) ((uintptr_t) c12 + cn_stride);
      _mm_storeu_si128((__m128i*) (c11 + 0), vout11x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c11 + 16), vout11xGHIJKLMNOPQRSTUV);
      c11 = (int8_t*) ((uintptr_t) c11 + cn_stride);
      _mm_storeu_si128((__m128i*) (c10 + 0), vout10x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c10 + 16), vout10xGHIJKLMNOPQRSTUV);
      c10 = (int8_t*) ((uintptr_t) c10 + cn_stride);
      _mm_storeu_si128((__m128i*) (c9 + 0), vout9x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c9 + 16), vout9xGHIJKLMNOPQRSTUV);
      c9 = (int8_t*) ((uintptr_t) c9 + cn_stride);
      _mm_storeu_si128((__m128i*) (c8 + 0), vout8x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c8 + 16), vout8xGHIJKLMNOPQRSTUV);
      c8 = (int8_t*) ((uintptr_t) c8 + cn_stride);
      _mm_storeu_si128((__m128i*) (c7 + 0), vout7x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c7 + 16), vout7xGHIJKLMNOPQRSTUV);
      c7 = (int8_t*) ((uintptr_t) c7 + cn_stride);
      _mm_storeu_si128((__m128i*) (c6 + 0), vout6x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c6 + 16), vout6xGHIJKLMNOPQRSTUV);
      c6 = (int8_t*) ((uintptr_t) c6 + cn_stride);
      _mm_storeu_si128((__m128i*) (c5 + 0), vout5x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c5 + 16), vout5xGHIJKLMNOPQRSTUV);
      c5 = (int8_t*) ((uintptr_t) c5 + cn_stride);
      _mm_storeu_si128((__m128i*) (c4 + 0), vout4x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c4 + 16), vout4xGHIJKLMNOPQRSTUV);
      c4 = (int8_t*) ((uintptr_t) c4 + cn_stride);
      _mm_storeu_si128((__m128i*) (c3 + 0), vout3x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c3 + 16), vout3xGHIJKLMNOPQRSTUV);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      _mm_storeu_si128((__m128i*) (c2 + 0), vout2x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c2 + 16), vout2xGHIJKLMNOPQRSTUV);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      _mm_storeu_si128((__m128i*) (c1 + 0), vout1x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c1 + 16), vout1xGHIJKLMNOPQRSTUV);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      _mm_storeu_si128((__m128i*) (c0 + 0), vout0x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c0 + 16), vout0xGHIJKLMNOPQRSTUV);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 32;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 0) & 0xFFFF));
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 16) & 0xFFFF));

      _mm_mask_storeu_epi8(c15 + 0, vmask0, vout15x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c15 + 16, vmask1, vout15xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c14 + 0, vmask0, vout14x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c14 + 16, vmask1, vout14xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c13 + 0, vmask0, vout13x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c13 + 16, vmask1, vout13xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c12 + 0, vmask0, vout12x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c12 + 16, vmask1, vout12xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c11 + 0, vmask0, vout11x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c11 + 16, vmask1, vout11xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c10 + 0, vmask0, vout10x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c10 + 16, vmask1, vout10xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c9 + 0, vmask0, vout9x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c9 + 16, vmask1, vout9xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c8 + 0, vmask0, vout8x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c8 + 16, vmask1, vout8xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c7 + 0, vmask0, vout7x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c7 + 16, vmask1, vout7xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c6 + 0, vmask0, vout6x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c6 + 16, vmask1, vout6xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c5 + 0, vmask0, vout5x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c5 + 16, vmask1, vout5xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c4 + 0, vmask0, vout4x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c4 + 16, vmask1, vout4xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c3 + 0, vmask0, vout3x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c3 + 16, vmask1, vout3xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c2 + 0, vmask0, vout2x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c2 + 16, vmask1, vout2xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c1 + 0, vmask0, vout1x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c1 + 16, vmask1, vout1xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c0 + 0, vmask0, vout0x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c0 + 16, vmask1, vout0xGHIJKLMNOPQRSTUV);
      nc = 0;
    }
  } while (nc != 0);

  // Release tile config
  //  _tile_release();
  __asm__ volatile ("tilerelease" ::);
  #endif  // defined(__x86_64__)
}
