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
#include "xnnpack/prefetch.h"


void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_16x32c4__avx512amx_prfm(
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

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);

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
          _tile_loadd(5, (const int8_t*) w + 0, 128);
          _tile_dpbssd(0, 4, 5);
          _tile_loadd(5, (const int8_t*) w + 64, 128);
          _tile_dpbssd(1, 4, 5);
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

    __m512i vacc0x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc0xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc1x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc1xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc2x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc2xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc3x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc3xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc4x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc4xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc5x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc5xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc6x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc6xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc7x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc7xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc8x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc8xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc9x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc9xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc10x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc10xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc11x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc11xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc12x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc12xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc13x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc13xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc14x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc14xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc15x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc15xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params->zero_point));
    vacc0x0123456789ABCDEF = _mm512_add_epi32(vacc0x0123456789ABCDEF, _mm512_load_epi32(res0 + 0));
    vacc0xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc0xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 0));
    vacc1x0123456789ABCDEF = _mm512_add_epi32(vacc1x0123456789ABCDEF, _mm512_load_epi32(res0 + 16));
    vacc1xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc1xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 16));
    vacc2x0123456789ABCDEF = _mm512_add_epi32(vacc2x0123456789ABCDEF, _mm512_load_epi32(res0 + 32));
    vacc2xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc2xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 32));
    vacc3x0123456789ABCDEF = _mm512_add_epi32(vacc3x0123456789ABCDEF, _mm512_load_epi32(res0 + 48));
    vacc3xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc3xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 48));
    vacc4x0123456789ABCDEF = _mm512_add_epi32(vacc4x0123456789ABCDEF, _mm512_load_epi32(res0 + 64));
    vacc4xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc4xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 64));
    vacc5x0123456789ABCDEF = _mm512_add_epi32(vacc5x0123456789ABCDEF, _mm512_load_epi32(res0 + 80));
    vacc5xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc5xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 80));
    vacc6x0123456789ABCDEF = _mm512_add_epi32(vacc6x0123456789ABCDEF, _mm512_load_epi32(res0 + 96));
    vacc6xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc6xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 96));
    vacc7x0123456789ABCDEF = _mm512_add_epi32(vacc7x0123456789ABCDEF, _mm512_load_epi32(res0 + 112));
    vacc7xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc7xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 112));
    vacc8x0123456789ABCDEF = _mm512_add_epi32(vacc8x0123456789ABCDEF, _mm512_load_epi32(res0 + 128));
    vacc8xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc8xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 128));
    vacc9x0123456789ABCDEF = _mm512_add_epi32(vacc9x0123456789ABCDEF, _mm512_load_epi32(res0 + 144));
    vacc9xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc9xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 144));
    vacc10x0123456789ABCDEF = _mm512_add_epi32(vacc10x0123456789ABCDEF, _mm512_load_epi32(res0 + 160));
    vacc10xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc10xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 160));
    vacc11x0123456789ABCDEF = _mm512_add_epi32(vacc11x0123456789ABCDEF, _mm512_load_epi32(res0 + 176));
    vacc11xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc11xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 176));
    vacc12x0123456789ABCDEF = _mm512_add_epi32(vacc12x0123456789ABCDEF, _mm512_load_epi32(res0 + 192));
    vacc12xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc12xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 192));
    vacc13x0123456789ABCDEF = _mm512_add_epi32(vacc13x0123456789ABCDEF, _mm512_load_epi32(res0 + 208));
    vacc13xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc13xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 208));
    vacc14x0123456789ABCDEF = _mm512_add_epi32(vacc14x0123456789ABCDEF, _mm512_load_epi32(res0 + 224));
    vacc14xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc14xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 224));
    vacc15x0123456789ABCDEF = _mm512_add_epi32(vacc15x0123456789ABCDEF, _mm512_load_epi32(res0 + 240));
    vacc15xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc15xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 240));

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

    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled0xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled1x0123456789ABCDEF = _mm512_mul_ps(vscaled1x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled1xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled1xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled2x0123456789ABCDEF = _mm512_mul_ps(vscaled2x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled2xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled2xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled3x0123456789ABCDEF = _mm512_mul_ps(vscaled3x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled3xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled3xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled4x0123456789ABCDEF = _mm512_mul_ps(vscaled4x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled4xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled4xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled5x0123456789ABCDEF = _mm512_mul_ps(vscaled5x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled5xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled5xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled6x0123456789ABCDEF = _mm512_mul_ps(vscaled6x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled6xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled6xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled7x0123456789ABCDEF = _mm512_mul_ps(vscaled7x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled7xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled7xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled8x0123456789ABCDEF = _mm512_mul_ps(vscaled8x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled8xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled8xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled9x0123456789ABCDEF = _mm512_mul_ps(vscaled9x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled9xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled9xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled10x0123456789ABCDEF = _mm512_mul_ps(vscaled10x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled10xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled10xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled11x0123456789ABCDEF = _mm512_mul_ps(vscaled11x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled11xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled11xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled12x0123456789ABCDEF = _mm512_mul_ps(vscaled12x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled12xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled12xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled13x0123456789ABCDEF = _mm512_mul_ps(vscaled13x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled13xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled13xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled14x0123456789ABCDEF = _mm512_mul_ps(vscaled14x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled14xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled14xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled15x0123456789ABCDEF = _mm512_mul_ps(vscaled15x0123456789ABCDEF, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled15xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled15xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params->inv_scale));

    const __m512 vfilter_output_scale0123456789ABCDEF = _mm512_loadu_ps((const float*) w + 0);
    const __m512 vfilter_output_scaleGHIJKLMNOPQRSTUV = _mm512_loadu_ps((const float*) w + 16);
    w = (const int32_t*) w + 32;
    const __m512 vbias0123456789ABCDEF = _mm512_loadu_ps((const float*) w + 0);
    const __m512 vbiasGHIJKLMNOPQRSTUV = _mm512_loadu_ps((const float*) w + 16);
    w = (const int32_t*) w + 32;

    vscaled0x0123456789ABCDEF = _mm512_fmadd_ps(vscaled0x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled0xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled1x0123456789ABCDEF = _mm512_fmadd_ps(vscaled1x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled1xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled1xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled2x0123456789ABCDEF = _mm512_fmadd_ps(vscaled2x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled2xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled2xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled3x0123456789ABCDEF = _mm512_fmadd_ps(vscaled3x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled3xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled3xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled4x0123456789ABCDEF = _mm512_fmadd_ps(vscaled4x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled4xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled4xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled5x0123456789ABCDEF = _mm512_fmadd_ps(vscaled5x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled5xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled5xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled6x0123456789ABCDEF = _mm512_fmadd_ps(vscaled6x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled6xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled6xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled7x0123456789ABCDEF = _mm512_fmadd_ps(vscaled7x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled7xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled7xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled8x0123456789ABCDEF = _mm512_fmadd_ps(vscaled8x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled8xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled8xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled9x0123456789ABCDEF = _mm512_fmadd_ps(vscaled9x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled9xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled9xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled10x0123456789ABCDEF = _mm512_fmadd_ps(vscaled10x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled10xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled10xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled11x0123456789ABCDEF = _mm512_fmadd_ps(vscaled11x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled11xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled11xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled12x0123456789ABCDEF = _mm512_fmadd_ps(vscaled12x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled12xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled12xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled13x0123456789ABCDEF = _mm512_fmadd_ps(vscaled13x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled13xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled13xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled14x0123456789ABCDEF = _mm512_fmadd_ps(vscaled14x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled14xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled14xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled15x0123456789ABCDEF = _mm512_fmadd_ps(vscaled15x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled15xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled15xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);

    vscaled0x0123456789ABCDEF = _mm512_max_ps(vscaled0x0123456789ABCDEF, voutput_min);
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled0xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled1x0123456789ABCDEF = _mm512_max_ps(vscaled1x0123456789ABCDEF, voutput_min);
    vscaled1xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled1xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled2x0123456789ABCDEF = _mm512_max_ps(vscaled2x0123456789ABCDEF, voutput_min);
    vscaled2xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled2xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled3x0123456789ABCDEF = _mm512_max_ps(vscaled3x0123456789ABCDEF, voutput_min);
    vscaled3xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled3xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled4x0123456789ABCDEF = _mm512_max_ps(vscaled4x0123456789ABCDEF, voutput_min);
    vscaled4xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled4xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled5x0123456789ABCDEF = _mm512_max_ps(vscaled5x0123456789ABCDEF, voutput_min);
    vscaled5xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled5xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled6x0123456789ABCDEF = _mm512_max_ps(vscaled6x0123456789ABCDEF, voutput_min);
    vscaled6xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled6xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled7x0123456789ABCDEF = _mm512_max_ps(vscaled7x0123456789ABCDEF, voutput_min);
    vscaled7xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled7xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled8x0123456789ABCDEF = _mm512_max_ps(vscaled8x0123456789ABCDEF, voutput_min);
    vscaled8xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled8xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled9x0123456789ABCDEF = _mm512_max_ps(vscaled9x0123456789ABCDEF, voutput_min);
    vscaled9xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled9xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled10x0123456789ABCDEF = _mm512_max_ps(vscaled10x0123456789ABCDEF, voutput_min);
    vscaled10xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled10xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled11x0123456789ABCDEF = _mm512_max_ps(vscaled11x0123456789ABCDEF, voutput_min);
    vscaled11xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled11xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled12x0123456789ABCDEF = _mm512_max_ps(vscaled12x0123456789ABCDEF, voutput_min);
    vscaled12xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled12xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled13x0123456789ABCDEF = _mm512_max_ps(vscaled13x0123456789ABCDEF, voutput_min);
    vscaled13xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled13xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled14x0123456789ABCDEF = _mm512_max_ps(vscaled14x0123456789ABCDEF, voutput_min);
    vscaled14xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled14xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled15x0123456789ABCDEF = _mm512_max_ps(vscaled15x0123456789ABCDEF, voutput_min);
    vscaled15xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled15xGHIJKLMNOPQRSTUV, voutput_min);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max);
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled0xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled1x0123456789ABCDEF = _mm512_min_ps(vscaled1x0123456789ABCDEF, voutput_max);
    vscaled1xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled1xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled2x0123456789ABCDEF = _mm512_min_ps(vscaled2x0123456789ABCDEF, voutput_max);
    vscaled2xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled2xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled3x0123456789ABCDEF = _mm512_min_ps(vscaled3x0123456789ABCDEF, voutput_max);
    vscaled3xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled3xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled4x0123456789ABCDEF = _mm512_min_ps(vscaled4x0123456789ABCDEF, voutput_max);
    vscaled4xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled4xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled5x0123456789ABCDEF = _mm512_min_ps(vscaled5x0123456789ABCDEF, voutput_max);
    vscaled5xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled5xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled6x0123456789ABCDEF = _mm512_min_ps(vscaled6x0123456789ABCDEF, voutput_max);
    vscaled6xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled6xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled7x0123456789ABCDEF = _mm512_min_ps(vscaled7x0123456789ABCDEF, voutput_max);
    vscaled7xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled7xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled8x0123456789ABCDEF = _mm512_min_ps(vscaled8x0123456789ABCDEF, voutput_max);
    vscaled8xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled8xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled9x0123456789ABCDEF = _mm512_min_ps(vscaled9x0123456789ABCDEF, voutput_max);
    vscaled9xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled9xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled10x0123456789ABCDEF = _mm512_min_ps(vscaled10x0123456789ABCDEF, voutput_max);
    vscaled10xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled10xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled11x0123456789ABCDEF = _mm512_min_ps(vscaled11x0123456789ABCDEF, voutput_max);
    vscaled11xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled11xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled12x0123456789ABCDEF = _mm512_min_ps(vscaled12x0123456789ABCDEF, voutput_max);
    vscaled12xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled12xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled13x0123456789ABCDEF = _mm512_min_ps(vscaled13x0123456789ABCDEF, voutput_max);
    vscaled13xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled13xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled14x0123456789ABCDEF = _mm512_min_ps(vscaled14x0123456789ABCDEF, voutput_max);
    vscaled14xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled14xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled15x0123456789ABCDEF = _mm512_min_ps(vscaled15x0123456789ABCDEF, voutput_max);
    vscaled15xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled15xGHIJKLMNOPQRSTUV, voutput_max);

    if XNN_LIKELY(nc >= 32) {
      _mm512_storeu_ps(c15 + 0, vscaled15x0123456789ABCDEF);
      _mm512_storeu_ps(c15 + 16, vscaled15xGHIJKLMNOPQRSTUV);
      c15 = (float*) ((uintptr_t) c15 + cn_stride);
      _mm512_storeu_ps(c14 + 0, vscaled14x0123456789ABCDEF);
      _mm512_storeu_ps(c14 + 16, vscaled14xGHIJKLMNOPQRSTUV);
      c14 = (float*) ((uintptr_t) c14 + cn_stride);
      _mm512_storeu_ps(c13 + 0, vscaled13x0123456789ABCDEF);
      _mm512_storeu_ps(c13 + 16, vscaled13xGHIJKLMNOPQRSTUV);
      c13 = (float*) ((uintptr_t) c13 + cn_stride);
      _mm512_storeu_ps(c12 + 0, vscaled12x0123456789ABCDEF);
      _mm512_storeu_ps(c12 + 16, vscaled12xGHIJKLMNOPQRSTUV);
      c12 = (float*) ((uintptr_t) c12 + cn_stride);
      _mm512_storeu_ps(c11 + 0, vscaled11x0123456789ABCDEF);
      _mm512_storeu_ps(c11 + 16, vscaled11xGHIJKLMNOPQRSTUV);
      c11 = (float*) ((uintptr_t) c11 + cn_stride);
      _mm512_storeu_ps(c10 + 0, vscaled10x0123456789ABCDEF);
      _mm512_storeu_ps(c10 + 16, vscaled10xGHIJKLMNOPQRSTUV);
      c10 = (float*) ((uintptr_t) c10 + cn_stride);
      _mm512_storeu_ps(c9 + 0, vscaled9x0123456789ABCDEF);
      _mm512_storeu_ps(c9 + 16, vscaled9xGHIJKLMNOPQRSTUV);
      c9 = (float*) ((uintptr_t) c9 + cn_stride);
      _mm512_storeu_ps(c8 + 0, vscaled8x0123456789ABCDEF);
      _mm512_storeu_ps(c8 + 16, vscaled8xGHIJKLMNOPQRSTUV);
      c8 = (float*) ((uintptr_t) c8 + cn_stride);
      _mm512_storeu_ps(c7 + 0, vscaled7x0123456789ABCDEF);
      _mm512_storeu_ps(c7 + 16, vscaled7xGHIJKLMNOPQRSTUV);
      c7 = (float*) ((uintptr_t) c7 + cn_stride);
      _mm512_storeu_ps(c6 + 0, vscaled6x0123456789ABCDEF);
      _mm512_storeu_ps(c6 + 16, vscaled6xGHIJKLMNOPQRSTUV);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      _mm512_storeu_ps(c5 + 0, vscaled5x0123456789ABCDEF);
      _mm512_storeu_ps(c5 + 16, vscaled5xGHIJKLMNOPQRSTUV);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      _mm512_storeu_ps(c4 + 0, vscaled4x0123456789ABCDEF);
      _mm512_storeu_ps(c4 + 16, vscaled4xGHIJKLMNOPQRSTUV);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm512_storeu_ps(c3 + 0, vscaled3x0123456789ABCDEF);
      _mm512_storeu_ps(c3 + 16, vscaled3xGHIJKLMNOPQRSTUV);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm512_storeu_ps(c2 + 0, vscaled2x0123456789ABCDEF);
      _mm512_storeu_ps(c2 + 16, vscaled2xGHIJKLMNOPQRSTUV);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm512_storeu_ps(c1 + 0, vscaled1x0123456789ABCDEF);
      _mm512_storeu_ps(c1 + 16, vscaled1xGHIJKLMNOPQRSTUV);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm512_storeu_ps(c0 + 0, vscaled0x0123456789ABCDEF);
      _mm512_storeu_ps(c0 + 16, vscaled0xGHIJKLMNOPQRSTUV);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 32;
    } else {
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 0) & 0xFFFF));
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 16) & 0xFFFF));
      _mm512_mask_storeu_ps(c15 + 0, vmask0, vscaled15x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c15 + 16, vmask1, vscaled15xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c14 + 0, vmask0, vscaled14x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c14 + 16, vmask1, vscaled14xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c13 + 0, vmask0, vscaled13x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c13 + 16, vmask1, vscaled13xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c12 + 0, vmask0, vscaled12x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c12 + 16, vmask1, vscaled12xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c11 + 0, vmask0, vscaled11x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c11 + 16, vmask1, vscaled11xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c10 + 0, vmask0, vscaled10x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c10 + 16, vmask1, vscaled10xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c9 + 0, vmask0, vscaled9x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c9 + 16, vmask1, vscaled9xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c8 + 0, vmask0, vscaled8x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c8 + 16, vmask1, vscaled8xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c7 + 0, vmask0, vscaled7x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c7 + 16, vmask1, vscaled7xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c6 + 0, vmask0, vscaled6x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c6 + 16, vmask1, vscaled6xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c5 + 0, vmask0, vscaled5x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c5 + 16, vmask1, vscaled5xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c4 + 0, vmask0, vscaled4x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c4 + 16, vmask1, vscaled4xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c3 + 0, vmask0, vscaled3x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c3 + 16, vmask1, vscaled3xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c2 + 0, vmask0, vscaled2x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c2 + 16, vmask1, vscaled2xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c1 + 0, vmask0, vscaled1x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c1 + 16, vmask1, vscaled1xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c0 + 0, vmask0, vscaled0x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c0 + 16, vmask1, vscaled0xGHIJKLMNOPQRSTUV);
      nc = 0;
    }
  } while (nc != 0);

  // Release tile config
  //  _tile_release();
  __asm__ volatile ("tilerelease" ::);
  #endif  // defined(__x86_64__)
}
