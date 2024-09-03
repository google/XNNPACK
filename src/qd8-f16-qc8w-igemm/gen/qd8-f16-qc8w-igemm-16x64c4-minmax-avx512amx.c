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


void xnn_qd8_f16_qc8w_igemm_minmax_ukernel_16x64c4__avx512amx(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    xnn_float16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
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
  __attribute__((aligned(64))) int32_t vintile[16 * 16];
  __attribute__((aligned(64))) int32_t res0[16 * 16];
  __attribute__((aligned(64))) int32_t res1[16 * 16];
  __attribute__((aligned(64))) int32_t res2[16 * 16];
  __attribute__((aligned(64))) int32_t res3[16 * 16];

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

  xnn_float16* c0 = c;
  xnn_float16* c1 = (xnn_float16*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  xnn_float16* c2 = (xnn_float16*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  xnn_float16* c3 = (xnn_float16*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  xnn_float16* c4 = (xnn_float16*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  xnn_float16* c5 = (xnn_float16*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    c5 = c4;
  }
  xnn_float16* c6 = (xnn_float16*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    c6 = c5;
  }
  xnn_float16* c7 = (xnn_float16*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 8) {
    c7 = c6;
  }
  xnn_float16* c8 = (xnn_float16*) ((uintptr_t) c7 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 8) {
    c8 = c7;
  }
  xnn_float16* c9 = (xnn_float16*) ((uintptr_t) c8 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 10) {
    c9 = c8;
  }
  xnn_float16* c10 = (xnn_float16*) ((uintptr_t) c9 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 10) {
    c10 = c9;
  }
  xnn_float16* c11 = (xnn_float16*) ((uintptr_t) c10 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 12) {
    c11 = c10;
  }
  xnn_float16* c12 = (xnn_float16*) ((uintptr_t) c11 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 12) {
    c12 = c11;
  }
  xnn_float16* c13 = (xnn_float16*) ((uintptr_t) c12 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 14) {
    c13 = c12;
  }
  xnn_float16* c14 = (xnn_float16*) ((uintptr_t) c13 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 14) {
    c14 = c13;
  }
  xnn_float16* c15 = (xnn_float16*) ((uintptr_t) c14 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 16) {
    c15 = c14;
  }

  const __m512 voutput_min = _mm512_cvtph_ps(_mm256_set1_epi16(*(const uint16_t*) &params->scalar.min));
  const __m512 voutput_max = _mm512_cvtph_ps(_mm256_set1_epi16(*(const uint16_t*) &params->scalar.max));
  // XNN_FORCE_REALIZATION(voutput_min);
  // XNN_FORCE_REALIZATION(voutput_max);

  do {
    const __m512i vksum0123456789ABCDEF = _mm512_loadu_epi32((const int32_t*) w + 0);
    const __m512i vksumGHIJKLMNOPQRSTUV = _mm512_loadu_epi32((const int32_t*) w + 16);
    const __m512i vksumWXYZabcdefghijkl = _mm512_loadu_epi32((const int32_t*) w + 32);
    const __m512i vksummnopqrstuvwxyz01 = _mm512_loadu_epi32((const int32_t*) w + 48);
    w = (const int32_t*) w + 64;

    // Zero tile accumulator
    __asm__ volatile (
      "tilezero %%tmm0\n"
      "tilezero %%tmm1\n"
      "tilezero %%tmm2\n"
      "tilezero %%tmm3\n"
      ::);

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
      a += 16;

      size_t k = kc;
      if (mr == 1)
      {
        while (k >= 64 * sizeof(int8_t)) {
          _tile_loadd(4, a0, 64);   // Directly load input for mr=1
          a15 += 64;
          _tile_loadd(5, (const int8_t*) w + 0, 256);
          _tile_dpbssd(0, 4, 5);
          _tile_loadd(5, (const int8_t*) w + 64, 256);
          _tile_dpbssd(1, 4, 5);
          _tile_loadd(5, (const int8_t*) w + 128, 256);
          _tile_dpbssd(2, 4, 5);
          _tile_loadd(5, (const int8_t*) w + 192, 256);
          _tile_dpbssd(3, 4, 5);

          w = (const int8_t*) w + 4096;
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
          _tile_loadd(5, (const int8_t*) w + 0, 256);
          _tile_dpbssd(0, 4, 5);
          _tile_loadd(5, (const int8_t*) w + 64, 256);
          _tile_dpbssd(1, 4, 5);
          _tile_loadd(5, (const int8_t*) w + 128, 256);
          _tile_dpbssd(2, 4, 5);
          _tile_loadd(5, (const int8_t*) w + 192, 256);
          _tile_dpbssd(3, 4, 5);

          w = (const int8_t*) w + 4096;
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

      p -= 16 * sizeof(void*);
    } while (p != 0);

    // Add tile to bias
    _tile_stored(0, res0, 64);
    _tile_stored(1, res1, 64);
    _tile_stored(2, res2, 64);
    _tile_stored(3, res3, 64);

    __m512i vacc0x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc0xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc0xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc0xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc1x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc1xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc1xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc1xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc2x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc2xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc2xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc2xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc3x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc3xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc3xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc3xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc4x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc4xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc4xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc4xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc5x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc5xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc5xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc5xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc6x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc6xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc6xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc6xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc7x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc7xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc7xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc7xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc8x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc8xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc8xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc8xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc9x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc9xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc9xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc9xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc10x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc10xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc10xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc10xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc11x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc11xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc11xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc11xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc12x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc12xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc12xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc12xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc13x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc13xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc13xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc13xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc14x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc14xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc14xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc14xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc15x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc15xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc15xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc15xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params->zero_point));
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

    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled0xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled0xWXYZabcdefghijkl = _mm512_mul_ps(vscaled0xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled0xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled0xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled1x0123456789ABCDEF = _mm512_mul_ps(vscaled1x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled1xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled1xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled1xWXYZabcdefghijkl = _mm512_mul_ps(vscaled1xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled1xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled1xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled2x0123456789ABCDEF = _mm512_mul_ps(vscaled2x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled2xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled2xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled2xWXYZabcdefghijkl = _mm512_mul_ps(vscaled2xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled2xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled2xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled3x0123456789ABCDEF = _mm512_mul_ps(vscaled3x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled3xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled3xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled3xWXYZabcdefghijkl = _mm512_mul_ps(vscaled3xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled3xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled3xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled4x0123456789ABCDEF = _mm512_mul_ps(vscaled4x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled4xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled4xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled4xWXYZabcdefghijkl = _mm512_mul_ps(vscaled4xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled4xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled4xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled5x0123456789ABCDEF = _mm512_mul_ps(vscaled5x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled5xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled5xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled5xWXYZabcdefghijkl = _mm512_mul_ps(vscaled5xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled5xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled5xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled6x0123456789ABCDEF = _mm512_mul_ps(vscaled6x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled6xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled6xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled6xWXYZabcdefghijkl = _mm512_mul_ps(vscaled6xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled6xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled6xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled7x0123456789ABCDEF = _mm512_mul_ps(vscaled7x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled7xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled7xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled7xWXYZabcdefghijkl = _mm512_mul_ps(vscaled7xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled7xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled7xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled8x0123456789ABCDEF = _mm512_mul_ps(vscaled8x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled8xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled8xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled8xWXYZabcdefghijkl = _mm512_mul_ps(vscaled8xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled8xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled8xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled9x0123456789ABCDEF = _mm512_mul_ps(vscaled9x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled9xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled9xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled9xWXYZabcdefghijkl = _mm512_mul_ps(vscaled9xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled9xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled9xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled10x0123456789ABCDEF = _mm512_mul_ps(vscaled10x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled10xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled10xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled10xWXYZabcdefghijkl = _mm512_mul_ps(vscaled10xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled10xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled10xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled11x0123456789ABCDEF = _mm512_mul_ps(vscaled11x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled11xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled11xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled11xWXYZabcdefghijkl = _mm512_mul_ps(vscaled11xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled11xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled11xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled12x0123456789ABCDEF = _mm512_mul_ps(vscaled12x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled12xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled12xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled12xWXYZabcdefghijkl = _mm512_mul_ps(vscaled12xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled12xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled12xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled13x0123456789ABCDEF = _mm512_mul_ps(vscaled13x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled13xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled13xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled13xWXYZabcdefghijkl = _mm512_mul_ps(vscaled13xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled13xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled13xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled14x0123456789ABCDEF = _mm512_mul_ps(vscaled14x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled14xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled14xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled14xWXYZabcdefghijkl = _mm512_mul_ps(vscaled14xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled14xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled14xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled15x0123456789ABCDEF = _mm512_mul_ps(vscaled15x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled15xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled15xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled15xWXYZabcdefghijkl = _mm512_mul_ps(vscaled15xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled15xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled15xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params->inv_scale));

    const __m512 vfilter_output_scale0123456789ABCDEF = _mm512_loadu_ps((const float*) w + 0);
    const __m512 vfilter_output_scaleGHIJKLMNOPQRSTUV = _mm512_loadu_ps((const float*) w + 16);
    const __m512 vfilter_output_scaleWXYZabcdefghijkl = _mm512_loadu_ps((const float*) w + 32);
    const __m512 vfilter_output_scalemnopqrstuvwxyz01 = _mm512_loadu_ps((const float*) w + 48);
    w = (const int32_t*) w + 64;
    const __m512 vbias0123456789ABCDEF = _mm512_loadu_ps((const float*) w + 0);
    const __m512 vbiasGHIJKLMNOPQRSTUV = _mm512_loadu_ps((const float*) w + 16);
    const __m512 vbiasWXYZabcdefghijkl = _mm512_loadu_ps((const float*) w + 32);
    const __m512 vbiasmnopqrstuvwxyz01 = _mm512_loadu_ps((const float*) w + 48);
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

    __m256i vfp16out0x0123456789ABCDEF = _mm512_cvtps_ph(vscaled0x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out0xGHIJKLMNOPQRSTUV = _mm512_cvtps_ph(vscaled0xGHIJKLMNOPQRSTUV, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out0xWXYZabcdefghijkl = _mm512_cvtps_ph(vscaled0xWXYZabcdefghijkl, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out0xmnopqrstuvwxyz01 = _mm512_cvtps_ph(vscaled0xmnopqrstuvwxyz01, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out1x0123456789ABCDEF = _mm512_cvtps_ph(vscaled1x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out1xGHIJKLMNOPQRSTUV = _mm512_cvtps_ph(vscaled1xGHIJKLMNOPQRSTUV, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out1xWXYZabcdefghijkl = _mm512_cvtps_ph(vscaled1xWXYZabcdefghijkl, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out1xmnopqrstuvwxyz01 = _mm512_cvtps_ph(vscaled1xmnopqrstuvwxyz01, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out2x0123456789ABCDEF = _mm512_cvtps_ph(vscaled2x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out2xGHIJKLMNOPQRSTUV = _mm512_cvtps_ph(vscaled2xGHIJKLMNOPQRSTUV, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out2xWXYZabcdefghijkl = _mm512_cvtps_ph(vscaled2xWXYZabcdefghijkl, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out2xmnopqrstuvwxyz01 = _mm512_cvtps_ph(vscaled2xmnopqrstuvwxyz01, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out3x0123456789ABCDEF = _mm512_cvtps_ph(vscaled3x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out3xGHIJKLMNOPQRSTUV = _mm512_cvtps_ph(vscaled3xGHIJKLMNOPQRSTUV, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out3xWXYZabcdefghijkl = _mm512_cvtps_ph(vscaled3xWXYZabcdefghijkl, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out3xmnopqrstuvwxyz01 = _mm512_cvtps_ph(vscaled3xmnopqrstuvwxyz01, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out4x0123456789ABCDEF = _mm512_cvtps_ph(vscaled4x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out4xGHIJKLMNOPQRSTUV = _mm512_cvtps_ph(vscaled4xGHIJKLMNOPQRSTUV, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out4xWXYZabcdefghijkl = _mm512_cvtps_ph(vscaled4xWXYZabcdefghijkl, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out4xmnopqrstuvwxyz01 = _mm512_cvtps_ph(vscaled4xmnopqrstuvwxyz01, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out5x0123456789ABCDEF = _mm512_cvtps_ph(vscaled5x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out5xGHIJKLMNOPQRSTUV = _mm512_cvtps_ph(vscaled5xGHIJKLMNOPQRSTUV, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out5xWXYZabcdefghijkl = _mm512_cvtps_ph(vscaled5xWXYZabcdefghijkl, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out5xmnopqrstuvwxyz01 = _mm512_cvtps_ph(vscaled5xmnopqrstuvwxyz01, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out6x0123456789ABCDEF = _mm512_cvtps_ph(vscaled6x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out6xGHIJKLMNOPQRSTUV = _mm512_cvtps_ph(vscaled6xGHIJKLMNOPQRSTUV, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out6xWXYZabcdefghijkl = _mm512_cvtps_ph(vscaled6xWXYZabcdefghijkl, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out6xmnopqrstuvwxyz01 = _mm512_cvtps_ph(vscaled6xmnopqrstuvwxyz01, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out7x0123456789ABCDEF = _mm512_cvtps_ph(vscaled7x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out7xGHIJKLMNOPQRSTUV = _mm512_cvtps_ph(vscaled7xGHIJKLMNOPQRSTUV, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out7xWXYZabcdefghijkl = _mm512_cvtps_ph(vscaled7xWXYZabcdefghijkl, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out7xmnopqrstuvwxyz01 = _mm512_cvtps_ph(vscaled7xmnopqrstuvwxyz01, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out8x0123456789ABCDEF = _mm512_cvtps_ph(vscaled8x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out8xGHIJKLMNOPQRSTUV = _mm512_cvtps_ph(vscaled8xGHIJKLMNOPQRSTUV, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out8xWXYZabcdefghijkl = _mm512_cvtps_ph(vscaled8xWXYZabcdefghijkl, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out8xmnopqrstuvwxyz01 = _mm512_cvtps_ph(vscaled8xmnopqrstuvwxyz01, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out9x0123456789ABCDEF = _mm512_cvtps_ph(vscaled9x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out9xGHIJKLMNOPQRSTUV = _mm512_cvtps_ph(vscaled9xGHIJKLMNOPQRSTUV, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out9xWXYZabcdefghijkl = _mm512_cvtps_ph(vscaled9xWXYZabcdefghijkl, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out9xmnopqrstuvwxyz01 = _mm512_cvtps_ph(vscaled9xmnopqrstuvwxyz01, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out10x0123456789ABCDEF = _mm512_cvtps_ph(vscaled10x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out10xGHIJKLMNOPQRSTUV = _mm512_cvtps_ph(vscaled10xGHIJKLMNOPQRSTUV, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out10xWXYZabcdefghijkl = _mm512_cvtps_ph(vscaled10xWXYZabcdefghijkl, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out10xmnopqrstuvwxyz01 = _mm512_cvtps_ph(vscaled10xmnopqrstuvwxyz01, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out11x0123456789ABCDEF = _mm512_cvtps_ph(vscaled11x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out11xGHIJKLMNOPQRSTUV = _mm512_cvtps_ph(vscaled11xGHIJKLMNOPQRSTUV, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out11xWXYZabcdefghijkl = _mm512_cvtps_ph(vscaled11xWXYZabcdefghijkl, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out11xmnopqrstuvwxyz01 = _mm512_cvtps_ph(vscaled11xmnopqrstuvwxyz01, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out12x0123456789ABCDEF = _mm512_cvtps_ph(vscaled12x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out12xGHIJKLMNOPQRSTUV = _mm512_cvtps_ph(vscaled12xGHIJKLMNOPQRSTUV, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out12xWXYZabcdefghijkl = _mm512_cvtps_ph(vscaled12xWXYZabcdefghijkl, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out12xmnopqrstuvwxyz01 = _mm512_cvtps_ph(vscaled12xmnopqrstuvwxyz01, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out13x0123456789ABCDEF = _mm512_cvtps_ph(vscaled13x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out13xGHIJKLMNOPQRSTUV = _mm512_cvtps_ph(vscaled13xGHIJKLMNOPQRSTUV, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out13xWXYZabcdefghijkl = _mm512_cvtps_ph(vscaled13xWXYZabcdefghijkl, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out13xmnopqrstuvwxyz01 = _mm512_cvtps_ph(vscaled13xmnopqrstuvwxyz01, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out14x0123456789ABCDEF = _mm512_cvtps_ph(vscaled14x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out14xGHIJKLMNOPQRSTUV = _mm512_cvtps_ph(vscaled14xGHIJKLMNOPQRSTUV, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out14xWXYZabcdefghijkl = _mm512_cvtps_ph(vscaled14xWXYZabcdefghijkl, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out14xmnopqrstuvwxyz01 = _mm512_cvtps_ph(vscaled14xmnopqrstuvwxyz01, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out15x0123456789ABCDEF = _mm512_cvtps_ph(vscaled15x0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out15xGHIJKLMNOPQRSTUV = _mm512_cvtps_ph(vscaled15xGHIJKLMNOPQRSTUV, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out15xWXYZabcdefghijkl = _mm512_cvtps_ph(vscaled15xWXYZabcdefghijkl, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out15xmnopqrstuvwxyz01 = _mm512_cvtps_ph(vscaled15xmnopqrstuvwxyz01, _MM_FROUND_TO_NEAREST_INT);
    if XNN_LIKELY(nc >= 64) {
      _mm256_storeu_si256((__m256i*) (c15 + 0), vfp16out15x0123456789ABCDEF);
      _mm256_storeu_si256((__m256i*) (c15 + 16), vfp16out15xGHIJKLMNOPQRSTUV);
      _mm256_storeu_si256((__m256i*) (c15 + 32), vfp16out15xWXYZabcdefghijkl);
      _mm256_storeu_si256((__m256i*) (c15 + 48), vfp16out15xmnopqrstuvwxyz01);
      c15 = (xnn_float16*) ((uintptr_t) c15 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c14 + 0), vfp16out14x0123456789ABCDEF);
      _mm256_storeu_si256((__m256i*) (c14 + 16), vfp16out14xGHIJKLMNOPQRSTUV);
      _mm256_storeu_si256((__m256i*) (c14 + 32), vfp16out14xWXYZabcdefghijkl);
      _mm256_storeu_si256((__m256i*) (c14 + 48), vfp16out14xmnopqrstuvwxyz01);
      c14 = (xnn_float16*) ((uintptr_t) c14 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c13 + 0), vfp16out13x0123456789ABCDEF);
      _mm256_storeu_si256((__m256i*) (c13 + 16), vfp16out13xGHIJKLMNOPQRSTUV);
      _mm256_storeu_si256((__m256i*) (c13 + 32), vfp16out13xWXYZabcdefghijkl);
      _mm256_storeu_si256((__m256i*) (c13 + 48), vfp16out13xmnopqrstuvwxyz01);
      c13 = (xnn_float16*) ((uintptr_t) c13 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c12 + 0), vfp16out12x0123456789ABCDEF);
      _mm256_storeu_si256((__m256i*) (c12 + 16), vfp16out12xGHIJKLMNOPQRSTUV);
      _mm256_storeu_si256((__m256i*) (c12 + 32), vfp16out12xWXYZabcdefghijkl);
      _mm256_storeu_si256((__m256i*) (c12 + 48), vfp16out12xmnopqrstuvwxyz01);
      c12 = (xnn_float16*) ((uintptr_t) c12 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c11 + 0), vfp16out11x0123456789ABCDEF);
      _mm256_storeu_si256((__m256i*) (c11 + 16), vfp16out11xGHIJKLMNOPQRSTUV);
      _mm256_storeu_si256((__m256i*) (c11 + 32), vfp16out11xWXYZabcdefghijkl);
      _mm256_storeu_si256((__m256i*) (c11 + 48), vfp16out11xmnopqrstuvwxyz01);
      c11 = (xnn_float16*) ((uintptr_t) c11 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c10 + 0), vfp16out10x0123456789ABCDEF);
      _mm256_storeu_si256((__m256i*) (c10 + 16), vfp16out10xGHIJKLMNOPQRSTUV);
      _mm256_storeu_si256((__m256i*) (c10 + 32), vfp16out10xWXYZabcdefghijkl);
      _mm256_storeu_si256((__m256i*) (c10 + 48), vfp16out10xmnopqrstuvwxyz01);
      c10 = (xnn_float16*) ((uintptr_t) c10 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c9 + 0), vfp16out9x0123456789ABCDEF);
      _mm256_storeu_si256((__m256i*) (c9 + 16), vfp16out9xGHIJKLMNOPQRSTUV);
      _mm256_storeu_si256((__m256i*) (c9 + 32), vfp16out9xWXYZabcdefghijkl);
      _mm256_storeu_si256((__m256i*) (c9 + 48), vfp16out9xmnopqrstuvwxyz01);
      c9 = (xnn_float16*) ((uintptr_t) c9 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c8 + 0), vfp16out8x0123456789ABCDEF);
      _mm256_storeu_si256((__m256i*) (c8 + 16), vfp16out8xGHIJKLMNOPQRSTUV);
      _mm256_storeu_si256((__m256i*) (c8 + 32), vfp16out8xWXYZabcdefghijkl);
      _mm256_storeu_si256((__m256i*) (c8 + 48), vfp16out8xmnopqrstuvwxyz01);
      c8 = (xnn_float16*) ((uintptr_t) c8 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c7 + 0), vfp16out7x0123456789ABCDEF);
      _mm256_storeu_si256((__m256i*) (c7 + 16), vfp16out7xGHIJKLMNOPQRSTUV);
      _mm256_storeu_si256((__m256i*) (c7 + 32), vfp16out7xWXYZabcdefghijkl);
      _mm256_storeu_si256((__m256i*) (c7 + 48), vfp16out7xmnopqrstuvwxyz01);
      c7 = (xnn_float16*) ((uintptr_t) c7 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c6 + 0), vfp16out6x0123456789ABCDEF);
      _mm256_storeu_si256((__m256i*) (c6 + 16), vfp16out6xGHIJKLMNOPQRSTUV);
      _mm256_storeu_si256((__m256i*) (c6 + 32), vfp16out6xWXYZabcdefghijkl);
      _mm256_storeu_si256((__m256i*) (c6 + 48), vfp16out6xmnopqrstuvwxyz01);
      c6 = (xnn_float16*) ((uintptr_t) c6 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c5 + 0), vfp16out5x0123456789ABCDEF);
      _mm256_storeu_si256((__m256i*) (c5 + 16), vfp16out5xGHIJKLMNOPQRSTUV);
      _mm256_storeu_si256((__m256i*) (c5 + 32), vfp16out5xWXYZabcdefghijkl);
      _mm256_storeu_si256((__m256i*) (c5 + 48), vfp16out5xmnopqrstuvwxyz01);
      c5 = (xnn_float16*) ((uintptr_t) c5 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c4 + 0), vfp16out4x0123456789ABCDEF);
      _mm256_storeu_si256((__m256i*) (c4 + 16), vfp16out4xGHIJKLMNOPQRSTUV);
      _mm256_storeu_si256((__m256i*) (c4 + 32), vfp16out4xWXYZabcdefghijkl);
      _mm256_storeu_si256((__m256i*) (c4 + 48), vfp16out4xmnopqrstuvwxyz01);
      c4 = (xnn_float16*) ((uintptr_t) c4 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c3 + 0), vfp16out3x0123456789ABCDEF);
      _mm256_storeu_si256((__m256i*) (c3 + 16), vfp16out3xGHIJKLMNOPQRSTUV);
      _mm256_storeu_si256((__m256i*) (c3 + 32), vfp16out3xWXYZabcdefghijkl);
      _mm256_storeu_si256((__m256i*) (c3 + 48), vfp16out3xmnopqrstuvwxyz01);
      c3 = (xnn_float16*) ((uintptr_t) c3 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c2 + 0), vfp16out2x0123456789ABCDEF);
      _mm256_storeu_si256((__m256i*) (c2 + 16), vfp16out2xGHIJKLMNOPQRSTUV);
      _mm256_storeu_si256((__m256i*) (c2 + 32), vfp16out2xWXYZabcdefghijkl);
      _mm256_storeu_si256((__m256i*) (c2 + 48), vfp16out2xmnopqrstuvwxyz01);
      c2 = (xnn_float16*) ((uintptr_t) c2 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c1 + 0), vfp16out1x0123456789ABCDEF);
      _mm256_storeu_si256((__m256i*) (c1 + 16), vfp16out1xGHIJKLMNOPQRSTUV);
      _mm256_storeu_si256((__m256i*) (c1 + 32), vfp16out1xWXYZabcdefghijkl);
      _mm256_storeu_si256((__m256i*) (c1 + 48), vfp16out1xmnopqrstuvwxyz01);
      c1 = (xnn_float16*) ((uintptr_t) c1 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c0 + 0), vfp16out0x0123456789ABCDEF);
      _mm256_storeu_si256((__m256i*) (c0 + 16), vfp16out0xGHIJKLMNOPQRSTUV);
      _mm256_storeu_si256((__m256i*) (c0 + 32), vfp16out0xWXYZabcdefghijkl);
      _mm256_storeu_si256((__m256i*) (c0 + 48), vfp16out0xmnopqrstuvwxyz01);
      c0 = (xnn_float16*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 64;
    } else {
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 0) & 0xFFFF));
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 16) & 0xFFFF));
      const __mmask16 vmask2 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 32) & 0xFFFF));
      const __mmask16 vmask3 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 48) & 0xFFFF));
      _mm256_mask_storeu_epi16(c15 + 0, vmask0, vfp16out15x0123456789ABCDEF);
      _mm256_mask_storeu_epi16(c15 + 16, vmask1, vfp16out15xGHIJKLMNOPQRSTUV);
      _mm256_mask_storeu_epi16(c15 + 32, vmask2, vfp16out15xWXYZabcdefghijkl);
      _mm256_mask_storeu_epi16(c15 + 48, vmask3, vfp16out15xmnopqrstuvwxyz01);
      _mm256_mask_storeu_epi16(c14 + 0, vmask0, vfp16out14x0123456789ABCDEF);
      _mm256_mask_storeu_epi16(c14 + 16, vmask1, vfp16out14xGHIJKLMNOPQRSTUV);
      _mm256_mask_storeu_epi16(c14 + 32, vmask2, vfp16out14xWXYZabcdefghijkl);
      _mm256_mask_storeu_epi16(c14 + 48, vmask3, vfp16out14xmnopqrstuvwxyz01);
      _mm256_mask_storeu_epi16(c13 + 0, vmask0, vfp16out13x0123456789ABCDEF);
      _mm256_mask_storeu_epi16(c13 + 16, vmask1, vfp16out13xGHIJKLMNOPQRSTUV);
      _mm256_mask_storeu_epi16(c13 + 32, vmask2, vfp16out13xWXYZabcdefghijkl);
      _mm256_mask_storeu_epi16(c13 + 48, vmask3, vfp16out13xmnopqrstuvwxyz01);
      _mm256_mask_storeu_epi16(c12 + 0, vmask0, vfp16out12x0123456789ABCDEF);
      _mm256_mask_storeu_epi16(c12 + 16, vmask1, vfp16out12xGHIJKLMNOPQRSTUV);
      _mm256_mask_storeu_epi16(c12 + 32, vmask2, vfp16out12xWXYZabcdefghijkl);
      _mm256_mask_storeu_epi16(c12 + 48, vmask3, vfp16out12xmnopqrstuvwxyz01);
      _mm256_mask_storeu_epi16(c11 + 0, vmask0, vfp16out11x0123456789ABCDEF);
      _mm256_mask_storeu_epi16(c11 + 16, vmask1, vfp16out11xGHIJKLMNOPQRSTUV);
      _mm256_mask_storeu_epi16(c11 + 32, vmask2, vfp16out11xWXYZabcdefghijkl);
      _mm256_mask_storeu_epi16(c11 + 48, vmask3, vfp16out11xmnopqrstuvwxyz01);
      _mm256_mask_storeu_epi16(c10 + 0, vmask0, vfp16out10x0123456789ABCDEF);
      _mm256_mask_storeu_epi16(c10 + 16, vmask1, vfp16out10xGHIJKLMNOPQRSTUV);
      _mm256_mask_storeu_epi16(c10 + 32, vmask2, vfp16out10xWXYZabcdefghijkl);
      _mm256_mask_storeu_epi16(c10 + 48, vmask3, vfp16out10xmnopqrstuvwxyz01);
      _mm256_mask_storeu_epi16(c9 + 0, vmask0, vfp16out9x0123456789ABCDEF);
      _mm256_mask_storeu_epi16(c9 + 16, vmask1, vfp16out9xGHIJKLMNOPQRSTUV);
      _mm256_mask_storeu_epi16(c9 + 32, vmask2, vfp16out9xWXYZabcdefghijkl);
      _mm256_mask_storeu_epi16(c9 + 48, vmask3, vfp16out9xmnopqrstuvwxyz01);
      _mm256_mask_storeu_epi16(c8 + 0, vmask0, vfp16out8x0123456789ABCDEF);
      _mm256_mask_storeu_epi16(c8 + 16, vmask1, vfp16out8xGHIJKLMNOPQRSTUV);
      _mm256_mask_storeu_epi16(c8 + 32, vmask2, vfp16out8xWXYZabcdefghijkl);
      _mm256_mask_storeu_epi16(c8 + 48, vmask3, vfp16out8xmnopqrstuvwxyz01);
      _mm256_mask_storeu_epi16(c7 + 0, vmask0, vfp16out7x0123456789ABCDEF);
      _mm256_mask_storeu_epi16(c7 + 16, vmask1, vfp16out7xGHIJKLMNOPQRSTUV);
      _mm256_mask_storeu_epi16(c7 + 32, vmask2, vfp16out7xWXYZabcdefghijkl);
      _mm256_mask_storeu_epi16(c7 + 48, vmask3, vfp16out7xmnopqrstuvwxyz01);
      _mm256_mask_storeu_epi16(c6 + 0, vmask0, vfp16out6x0123456789ABCDEF);
      _mm256_mask_storeu_epi16(c6 + 16, vmask1, vfp16out6xGHIJKLMNOPQRSTUV);
      _mm256_mask_storeu_epi16(c6 + 32, vmask2, vfp16out6xWXYZabcdefghijkl);
      _mm256_mask_storeu_epi16(c6 + 48, vmask3, vfp16out6xmnopqrstuvwxyz01);
      _mm256_mask_storeu_epi16(c5 + 0, vmask0, vfp16out5x0123456789ABCDEF);
      _mm256_mask_storeu_epi16(c5 + 16, vmask1, vfp16out5xGHIJKLMNOPQRSTUV);
      _mm256_mask_storeu_epi16(c5 + 32, vmask2, vfp16out5xWXYZabcdefghijkl);
      _mm256_mask_storeu_epi16(c5 + 48, vmask3, vfp16out5xmnopqrstuvwxyz01);
      _mm256_mask_storeu_epi16(c4 + 0, vmask0, vfp16out4x0123456789ABCDEF);
      _mm256_mask_storeu_epi16(c4 + 16, vmask1, vfp16out4xGHIJKLMNOPQRSTUV);
      _mm256_mask_storeu_epi16(c4 + 32, vmask2, vfp16out4xWXYZabcdefghijkl);
      _mm256_mask_storeu_epi16(c4 + 48, vmask3, vfp16out4xmnopqrstuvwxyz01);
      _mm256_mask_storeu_epi16(c3 + 0, vmask0, vfp16out3x0123456789ABCDEF);
      _mm256_mask_storeu_epi16(c3 + 16, vmask1, vfp16out3xGHIJKLMNOPQRSTUV);
      _mm256_mask_storeu_epi16(c3 + 32, vmask2, vfp16out3xWXYZabcdefghijkl);
      _mm256_mask_storeu_epi16(c3 + 48, vmask3, vfp16out3xmnopqrstuvwxyz01);
      _mm256_mask_storeu_epi16(c2 + 0, vmask0, vfp16out2x0123456789ABCDEF);
      _mm256_mask_storeu_epi16(c2 + 16, vmask1, vfp16out2xGHIJKLMNOPQRSTUV);
      _mm256_mask_storeu_epi16(c2 + 32, vmask2, vfp16out2xWXYZabcdefghijkl);
      _mm256_mask_storeu_epi16(c2 + 48, vmask3, vfp16out2xmnopqrstuvwxyz01);
      _mm256_mask_storeu_epi16(c1 + 0, vmask0, vfp16out1x0123456789ABCDEF);
      _mm256_mask_storeu_epi16(c1 + 16, vmask1, vfp16out1xGHIJKLMNOPQRSTUV);
      _mm256_mask_storeu_epi16(c1 + 32, vmask2, vfp16out1xWXYZabcdefghijkl);
      _mm256_mask_storeu_epi16(c1 + 48, vmask3, vfp16out1xmnopqrstuvwxyz01);
      _mm256_mask_storeu_epi16(c0 + 0, vmask0, vfp16out0x0123456789ABCDEF);
      _mm256_mask_storeu_epi16(c0 + 16, vmask1, vfp16out0xGHIJKLMNOPQRSTUV);
      _mm256_mask_storeu_epi16(c0 + 32, vmask2, vfp16out0xWXYZabcdefghijkl);
      _mm256_mask_storeu_epi16(c0 + 48, vmask3, vfp16out0xmnopqrstuvwxyz01);
      nc = 0;
    }
  } while (nc != 0);

  // Release tile config
  //  _tile_release();
  __asm__ volatile ("tilerelease" ::);
  #endif  // defined(__x86_64__)
}
