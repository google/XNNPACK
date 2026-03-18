// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/c4-avx512amx.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#if defined(__has_feature)
  #if __has_feature(memory_sanitizer)
    #include <sanitizer/msan_interface.h>
  #endif
#endif

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/prefetch.h"


void xnn_qd8_f16_qc8w_igemm_minmax_ukernel_16x64c4__avx512amx_prfm(
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
    const struct xnn_f16_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params)
{
  assert(mr != 0);
  assert(mr <= 16);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

// AMX is only available for __x86_64__
#if XNN_ARCH_X86_64

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

  XNN_ALIGN(64) struct __tile_config tile_data = {0};
  XNN_ALIGN(64) int32_t res[4][16 * 16];
  XNN_ALIGN(64) int32_t vintile[16 * 16];

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  const size_t kremainder = (kc & 63) ? (kc & 63) : 64;
  const __mmask16 kremainder_mask = _cvtu32_mask16((UINT32_C(1) << (kremainder >> 2)) - 1);

  // Load tile configuration
  tile_data.palette_id = 1;
  tile_data.rows[0] = mr;              // tmm0 = res[0]
  tile_data.rows[1] = mr;              // tmm1 = res[1]
  tile_data.rows[2] = mr;              // tmm2 = res[2]
  tile_data.rows[3] = mr;              // tmm3 = res[3]
  tile_data.rows[4] = mr;              // tmm4 = input
  tile_data.rows[5] = 16;              // tmm5 = weights
  tile_data.rows[6] = mr;              // tmm6 = input remainder
  tile_data.rows[7] = kremainder >> 2; // tmm7 = weights remainder

  tile_data.colsb[0] = 64;          // tmm0 = res[0]
  tile_data.colsb[1] = 64;          // tmm1 = res[1]
  tile_data.colsb[2] = 64;          // tmm2 = res[2]
  tile_data.colsb[3] = 64;          // tmm3 = res[3]
  tile_data.colsb[4] = 64;          // tmm4 = input
  tile_data.colsb[5] = 64;          // tmm5 = weights
  tile_data.colsb[6] = kremainder;  // tmm6 = input remainder
  tile_data.colsb[7] = 64;          // tmm7 = weights remainder

  _tile_loadconfig(&tile_data);

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
    const __m512i vksum0 = _mm512_loadu_epi32((const int32_t*) w + 0);
    const __m512i vksum1 = _mm512_loadu_epi32((const int32_t*) w + 16);
    const __m512i vksum2 = _mm512_loadu_epi32((const int32_t*) w + 32);
    const __m512i vksum3 = _mm512_loadu_epi32((const int32_t*) w + 48);
    w = (const int32_t*) w + 64;

    // Zero tile accumulator
    _tile_zero(0);
    _tile_zero(1);
    _tile_zero(2);
    _tile_zero(3);

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
          a0 += 64;
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

    // TODO: Instead of processing up to 4 tiles (16x64) consider
    // quantizing 1 tile at a time (16 registers)
    _tile_stored(0, &res[0][0], 64);
    _tile_stored(1, &res[1][0], 64);
    _tile_stored(2, &res[2][0], 64);
    _tile_stored(3, &res[3][0], 64);

    // TODO: Fix msan for AMX
    #if defined(__has_feature)
      #if __has_feature(memory_sanitizer)
        __msan_unpoison(res, sizeof(res));
      #endif
    #endif

    // TODO: Instead of processing up to 4 tiles (16x64) consider
    // quantizing 1 row at a time.
    // Add tile to bias
    __m512i vacc0x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc0x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc0x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc0x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc1x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc1x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc1x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc1x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc2x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc2x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc2x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc2x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc3x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc3x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc3x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc3x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc4x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc4x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc4x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc4x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc5x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc5x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc5x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc5x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc6x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc6x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc6x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc6x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc7x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc7x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc7x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc7x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc8x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc8x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc8x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc8x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc9x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc9x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc9x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc9x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc10x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc10x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc10x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc10x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc11x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc11x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc11x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc11x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc12x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc12x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc12x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc12x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc13x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc13x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc13x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc13x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc14x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc14x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc14x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc14x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc15x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc15x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc15x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params->zero_point));
    __m512i vacc15x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params->zero_point));
    vacc0x0 = _mm512_add_epi32(vacc0x0, _mm512_load_epi32(&res[0][0] + 0));
    vacc0x1 = _mm512_add_epi32(vacc0x1, _mm512_load_epi32(&res[1][0] + 0));
    vacc0x2 = _mm512_add_epi32(vacc0x2, _mm512_load_epi32(&res[2][0] + 0));
    vacc0x3 = _mm512_add_epi32(vacc0x3, _mm512_load_epi32(&res[3][0] + 0));
    vacc1x0 = _mm512_add_epi32(vacc1x0, _mm512_load_epi32(&res[0][0] + 16));
    vacc1x1 = _mm512_add_epi32(vacc1x1, _mm512_load_epi32(&res[1][0] + 16));
    vacc1x2 = _mm512_add_epi32(vacc1x2, _mm512_load_epi32(&res[2][0] + 16));
    vacc1x3 = _mm512_add_epi32(vacc1x3, _mm512_load_epi32(&res[3][0] + 16));
    vacc2x0 = _mm512_add_epi32(vacc2x0, _mm512_load_epi32(&res[0][0] + 32));
    vacc2x1 = _mm512_add_epi32(vacc2x1, _mm512_load_epi32(&res[1][0] + 32));
    vacc2x2 = _mm512_add_epi32(vacc2x2, _mm512_load_epi32(&res[2][0] + 32));
    vacc2x3 = _mm512_add_epi32(vacc2x3, _mm512_load_epi32(&res[3][0] + 32));
    vacc3x0 = _mm512_add_epi32(vacc3x0, _mm512_load_epi32(&res[0][0] + 48));
    vacc3x1 = _mm512_add_epi32(vacc3x1, _mm512_load_epi32(&res[1][0] + 48));
    vacc3x2 = _mm512_add_epi32(vacc3x2, _mm512_load_epi32(&res[2][0] + 48));
    vacc3x3 = _mm512_add_epi32(vacc3x3, _mm512_load_epi32(&res[3][0] + 48));
    vacc4x0 = _mm512_add_epi32(vacc4x0, _mm512_load_epi32(&res[0][0] + 64));
    vacc4x1 = _mm512_add_epi32(vacc4x1, _mm512_load_epi32(&res[1][0] + 64));
    vacc4x2 = _mm512_add_epi32(vacc4x2, _mm512_load_epi32(&res[2][0] + 64));
    vacc4x3 = _mm512_add_epi32(vacc4x3, _mm512_load_epi32(&res[3][0] + 64));
    vacc5x0 = _mm512_add_epi32(vacc5x0, _mm512_load_epi32(&res[0][0] + 80));
    vacc5x1 = _mm512_add_epi32(vacc5x1, _mm512_load_epi32(&res[1][0] + 80));
    vacc5x2 = _mm512_add_epi32(vacc5x2, _mm512_load_epi32(&res[2][0] + 80));
    vacc5x3 = _mm512_add_epi32(vacc5x3, _mm512_load_epi32(&res[3][0] + 80));
    vacc6x0 = _mm512_add_epi32(vacc6x0, _mm512_load_epi32(&res[0][0] + 96));
    vacc6x1 = _mm512_add_epi32(vacc6x1, _mm512_load_epi32(&res[1][0] + 96));
    vacc6x2 = _mm512_add_epi32(vacc6x2, _mm512_load_epi32(&res[2][0] + 96));
    vacc6x3 = _mm512_add_epi32(vacc6x3, _mm512_load_epi32(&res[3][0] + 96));
    vacc7x0 = _mm512_add_epi32(vacc7x0, _mm512_load_epi32(&res[0][0] + 112));
    vacc7x1 = _mm512_add_epi32(vacc7x1, _mm512_load_epi32(&res[1][0] + 112));
    vacc7x2 = _mm512_add_epi32(vacc7x2, _mm512_load_epi32(&res[2][0] + 112));
    vacc7x3 = _mm512_add_epi32(vacc7x3, _mm512_load_epi32(&res[3][0] + 112));
    vacc8x0 = _mm512_add_epi32(vacc8x0, _mm512_load_epi32(&res[0][0] + 128));
    vacc8x1 = _mm512_add_epi32(vacc8x1, _mm512_load_epi32(&res[1][0] + 128));
    vacc8x2 = _mm512_add_epi32(vacc8x2, _mm512_load_epi32(&res[2][0] + 128));
    vacc8x3 = _mm512_add_epi32(vacc8x3, _mm512_load_epi32(&res[3][0] + 128));
    vacc9x0 = _mm512_add_epi32(vacc9x0, _mm512_load_epi32(&res[0][0] + 144));
    vacc9x1 = _mm512_add_epi32(vacc9x1, _mm512_load_epi32(&res[1][0] + 144));
    vacc9x2 = _mm512_add_epi32(vacc9x2, _mm512_load_epi32(&res[2][0] + 144));
    vacc9x3 = _mm512_add_epi32(vacc9x3, _mm512_load_epi32(&res[3][0] + 144));
    vacc10x0 = _mm512_add_epi32(vacc10x0, _mm512_load_epi32(&res[0][0] + 160));
    vacc10x1 = _mm512_add_epi32(vacc10x1, _mm512_load_epi32(&res[1][0] + 160));
    vacc10x2 = _mm512_add_epi32(vacc10x2, _mm512_load_epi32(&res[2][0] + 160));
    vacc10x3 = _mm512_add_epi32(vacc10x3, _mm512_load_epi32(&res[3][0] + 160));
    vacc11x0 = _mm512_add_epi32(vacc11x0, _mm512_load_epi32(&res[0][0] + 176));
    vacc11x1 = _mm512_add_epi32(vacc11x1, _mm512_load_epi32(&res[1][0] + 176));
    vacc11x2 = _mm512_add_epi32(vacc11x2, _mm512_load_epi32(&res[2][0] + 176));
    vacc11x3 = _mm512_add_epi32(vacc11x3, _mm512_load_epi32(&res[3][0] + 176));
    vacc12x0 = _mm512_add_epi32(vacc12x0, _mm512_load_epi32(&res[0][0] + 192));
    vacc12x1 = _mm512_add_epi32(vacc12x1, _mm512_load_epi32(&res[1][0] + 192));
    vacc12x2 = _mm512_add_epi32(vacc12x2, _mm512_load_epi32(&res[2][0] + 192));
    vacc12x3 = _mm512_add_epi32(vacc12x3, _mm512_load_epi32(&res[3][0] + 192));
    vacc13x0 = _mm512_add_epi32(vacc13x0, _mm512_load_epi32(&res[0][0] + 208));
    vacc13x1 = _mm512_add_epi32(vacc13x1, _mm512_load_epi32(&res[1][0] + 208));
    vacc13x2 = _mm512_add_epi32(vacc13x2, _mm512_load_epi32(&res[2][0] + 208));
    vacc13x3 = _mm512_add_epi32(vacc13x3, _mm512_load_epi32(&res[3][0] + 208));
    vacc14x0 = _mm512_add_epi32(vacc14x0, _mm512_load_epi32(&res[0][0] + 224));
    vacc14x1 = _mm512_add_epi32(vacc14x1, _mm512_load_epi32(&res[1][0] + 224));
    vacc14x2 = _mm512_add_epi32(vacc14x2, _mm512_load_epi32(&res[2][0] + 224));
    vacc14x3 = _mm512_add_epi32(vacc14x3, _mm512_load_epi32(&res[3][0] + 224));
    vacc15x0 = _mm512_add_epi32(vacc15x0, _mm512_load_epi32(&res[0][0] + 240));
    vacc15x1 = _mm512_add_epi32(vacc15x1, _mm512_load_epi32(&res[1][0] + 240));
    vacc15x2 = _mm512_add_epi32(vacc15x2, _mm512_load_epi32(&res[2][0] + 240));
    vacc15x3 = _mm512_add_epi32(vacc15x3, _mm512_load_epi32(&res[3][0] + 240));

    __m512 vscaled0x0 = _mm512_cvtepi32_ps(vacc0x0);
    __m512 vscaled0x1 = _mm512_cvtepi32_ps(vacc0x1);
    __m512 vscaled0x2 = _mm512_cvtepi32_ps(vacc0x2);
    __m512 vscaled0x3 = _mm512_cvtepi32_ps(vacc0x3);
    __m512 vscaled1x0 = _mm512_cvtepi32_ps(vacc1x0);
    __m512 vscaled1x1 = _mm512_cvtepi32_ps(vacc1x1);
    __m512 vscaled1x2 = _mm512_cvtepi32_ps(vacc1x2);
    __m512 vscaled1x3 = _mm512_cvtepi32_ps(vacc1x3);
    __m512 vscaled2x0 = _mm512_cvtepi32_ps(vacc2x0);
    __m512 vscaled2x1 = _mm512_cvtepi32_ps(vacc2x1);
    __m512 vscaled2x2 = _mm512_cvtepi32_ps(vacc2x2);
    __m512 vscaled2x3 = _mm512_cvtepi32_ps(vacc2x3);
    __m512 vscaled3x0 = _mm512_cvtepi32_ps(vacc3x0);
    __m512 vscaled3x1 = _mm512_cvtepi32_ps(vacc3x1);
    __m512 vscaled3x2 = _mm512_cvtepi32_ps(vacc3x2);
    __m512 vscaled3x3 = _mm512_cvtepi32_ps(vacc3x3);
    __m512 vscaled4x0 = _mm512_cvtepi32_ps(vacc4x0);
    __m512 vscaled4x1 = _mm512_cvtepi32_ps(vacc4x1);
    __m512 vscaled4x2 = _mm512_cvtepi32_ps(vacc4x2);
    __m512 vscaled4x3 = _mm512_cvtepi32_ps(vacc4x3);
    __m512 vscaled5x0 = _mm512_cvtepi32_ps(vacc5x0);
    __m512 vscaled5x1 = _mm512_cvtepi32_ps(vacc5x1);
    __m512 vscaled5x2 = _mm512_cvtepi32_ps(vacc5x2);
    __m512 vscaled5x3 = _mm512_cvtepi32_ps(vacc5x3);
    __m512 vscaled6x0 = _mm512_cvtepi32_ps(vacc6x0);
    __m512 vscaled6x1 = _mm512_cvtepi32_ps(vacc6x1);
    __m512 vscaled6x2 = _mm512_cvtepi32_ps(vacc6x2);
    __m512 vscaled6x3 = _mm512_cvtepi32_ps(vacc6x3);
    __m512 vscaled7x0 = _mm512_cvtepi32_ps(vacc7x0);
    __m512 vscaled7x1 = _mm512_cvtepi32_ps(vacc7x1);
    __m512 vscaled7x2 = _mm512_cvtepi32_ps(vacc7x2);
    __m512 vscaled7x3 = _mm512_cvtepi32_ps(vacc7x3);
    __m512 vscaled8x0 = _mm512_cvtepi32_ps(vacc8x0);
    __m512 vscaled8x1 = _mm512_cvtepi32_ps(vacc8x1);
    __m512 vscaled8x2 = _mm512_cvtepi32_ps(vacc8x2);
    __m512 vscaled8x3 = _mm512_cvtepi32_ps(vacc8x3);
    __m512 vscaled9x0 = _mm512_cvtepi32_ps(vacc9x0);
    __m512 vscaled9x1 = _mm512_cvtepi32_ps(vacc9x1);
    __m512 vscaled9x2 = _mm512_cvtepi32_ps(vacc9x2);
    __m512 vscaled9x3 = _mm512_cvtepi32_ps(vacc9x3);
    __m512 vscaled10x0 = _mm512_cvtepi32_ps(vacc10x0);
    __m512 vscaled10x1 = _mm512_cvtepi32_ps(vacc10x1);
    __m512 vscaled10x2 = _mm512_cvtepi32_ps(vacc10x2);
    __m512 vscaled10x3 = _mm512_cvtepi32_ps(vacc10x3);
    __m512 vscaled11x0 = _mm512_cvtepi32_ps(vacc11x0);
    __m512 vscaled11x1 = _mm512_cvtepi32_ps(vacc11x1);
    __m512 vscaled11x2 = _mm512_cvtepi32_ps(vacc11x2);
    __m512 vscaled11x3 = _mm512_cvtepi32_ps(vacc11x3);
    __m512 vscaled12x0 = _mm512_cvtepi32_ps(vacc12x0);
    __m512 vscaled12x1 = _mm512_cvtepi32_ps(vacc12x1);
    __m512 vscaled12x2 = _mm512_cvtepi32_ps(vacc12x2);
    __m512 vscaled12x3 = _mm512_cvtepi32_ps(vacc12x3);
    __m512 vscaled13x0 = _mm512_cvtepi32_ps(vacc13x0);
    __m512 vscaled13x1 = _mm512_cvtepi32_ps(vacc13x1);
    __m512 vscaled13x2 = _mm512_cvtepi32_ps(vacc13x2);
    __m512 vscaled13x3 = _mm512_cvtepi32_ps(vacc13x3);
    __m512 vscaled14x0 = _mm512_cvtepi32_ps(vacc14x0);
    __m512 vscaled14x1 = _mm512_cvtepi32_ps(vacc14x1);
    __m512 vscaled14x2 = _mm512_cvtepi32_ps(vacc14x2);
    __m512 vscaled14x3 = _mm512_cvtepi32_ps(vacc14x3);
    __m512 vscaled15x0 = _mm512_cvtepi32_ps(vacc15x0);
    __m512 vscaled15x1 = _mm512_cvtepi32_ps(vacc15x1);
    __m512 vscaled15x2 = _mm512_cvtepi32_ps(vacc15x2);
    __m512 vscaled15x3 = _mm512_cvtepi32_ps(vacc15x3);

    vscaled0x0 = _mm512_mul_ps(vscaled0x0, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled0x1 = _mm512_mul_ps(vscaled0x1, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled0x2 = _mm512_mul_ps(vscaled0x2, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled0x3 = _mm512_mul_ps(vscaled0x3, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled1x0 = _mm512_mul_ps(vscaled1x0, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled1x1 = _mm512_mul_ps(vscaled1x1, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled1x2 = _mm512_mul_ps(vscaled1x2, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled1x3 = _mm512_mul_ps(vscaled1x3, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled2x0 = _mm512_mul_ps(vscaled2x0, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled2x1 = _mm512_mul_ps(vscaled2x1, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled2x2 = _mm512_mul_ps(vscaled2x2, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled2x3 = _mm512_mul_ps(vscaled2x3, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled3x0 = _mm512_mul_ps(vscaled3x0, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled3x1 = _mm512_mul_ps(vscaled3x1, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled3x2 = _mm512_mul_ps(vscaled3x2, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled3x3 = _mm512_mul_ps(vscaled3x3, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled4x0 = _mm512_mul_ps(vscaled4x0, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled4x1 = _mm512_mul_ps(vscaled4x1, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled4x2 = _mm512_mul_ps(vscaled4x2, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled4x3 = _mm512_mul_ps(vscaled4x3, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled5x0 = _mm512_mul_ps(vscaled5x0, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled5x1 = _mm512_mul_ps(vscaled5x1, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled5x2 = _mm512_mul_ps(vscaled5x2, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled5x3 = _mm512_mul_ps(vscaled5x3, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled6x0 = _mm512_mul_ps(vscaled6x0, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled6x1 = _mm512_mul_ps(vscaled6x1, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled6x2 = _mm512_mul_ps(vscaled6x2, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled6x3 = _mm512_mul_ps(vscaled6x3, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled7x0 = _mm512_mul_ps(vscaled7x0, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled7x1 = _mm512_mul_ps(vscaled7x1, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled7x2 = _mm512_mul_ps(vscaled7x2, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled7x3 = _mm512_mul_ps(vscaled7x3, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled8x0 = _mm512_mul_ps(vscaled8x0, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled8x1 = _mm512_mul_ps(vscaled8x1, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled8x2 = _mm512_mul_ps(vscaled8x2, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled8x3 = _mm512_mul_ps(vscaled8x3, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled9x0 = _mm512_mul_ps(vscaled9x0, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled9x1 = _mm512_mul_ps(vscaled9x1, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled9x2 = _mm512_mul_ps(vscaled9x2, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled9x3 = _mm512_mul_ps(vscaled9x3, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled10x0 = _mm512_mul_ps(vscaled10x0, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled10x1 = _mm512_mul_ps(vscaled10x1, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled10x2 = _mm512_mul_ps(vscaled10x2, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled10x3 = _mm512_mul_ps(vscaled10x3, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled11x0 = _mm512_mul_ps(vscaled11x0, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled11x1 = _mm512_mul_ps(vscaled11x1, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled11x2 = _mm512_mul_ps(vscaled11x2, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled11x3 = _mm512_mul_ps(vscaled11x3, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled12x0 = _mm512_mul_ps(vscaled12x0, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled12x1 = _mm512_mul_ps(vscaled12x1, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled12x2 = _mm512_mul_ps(vscaled12x2, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled12x3 = _mm512_mul_ps(vscaled12x3, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled13x0 = _mm512_mul_ps(vscaled13x0, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled13x1 = _mm512_mul_ps(vscaled13x1, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled13x2 = _mm512_mul_ps(vscaled13x2, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled13x3 = _mm512_mul_ps(vscaled13x3, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled14x0 = _mm512_mul_ps(vscaled14x0, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled14x1 = _mm512_mul_ps(vscaled14x1, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled14x2 = _mm512_mul_ps(vscaled14x2, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled14x3 = _mm512_mul_ps(vscaled14x3, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled15x0 = _mm512_mul_ps(vscaled15x0, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled15x1 = _mm512_mul_ps(vscaled15x1, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled15x2 = _mm512_mul_ps(vscaled15x2, _mm512_set1_ps(quantization_params->inv_scale));
    vscaled15x3 = _mm512_mul_ps(vscaled15x3, _mm512_set1_ps(quantization_params->inv_scale));

    const __m512 vfilter_output_scale0 = _mm512_loadu_ps((const float*) w + 0);
    const __m512 vfilter_output_scale1 = _mm512_loadu_ps((const float*) w + 16);
    const __m512 vfilter_output_scale2 = _mm512_loadu_ps((const float*) w + 32);
    const __m512 vfilter_output_scale3 = _mm512_loadu_ps((const float*) w + 48);
    w = (const int32_t*) w + 64;
    const __m512 vbias0 = _mm512_loadu_ps((const float*) w + 0);
    const __m512 vbias1 = _mm512_loadu_ps((const float*) w + 16);
    const __m512 vbias2 = _mm512_loadu_ps((const float*) w + 32);
    const __m512 vbias3 = _mm512_loadu_ps((const float*) w + 48);
    w = (const int32_t*) w + 64;

    vscaled0x0 = _mm512_fmadd_ps(vscaled0x0, vfilter_output_scale0, vbias0);
    vscaled0x1 = _mm512_fmadd_ps(vscaled0x1, vfilter_output_scale1, vbias1);
    vscaled0x2 = _mm512_fmadd_ps(vscaled0x2, vfilter_output_scale2, vbias2);
    vscaled0x3 = _mm512_fmadd_ps(vscaled0x3, vfilter_output_scale3, vbias3);
    vscaled1x0 = _mm512_fmadd_ps(vscaled1x0, vfilter_output_scale0, vbias0);
    vscaled1x1 = _mm512_fmadd_ps(vscaled1x1, vfilter_output_scale1, vbias1);
    vscaled1x2 = _mm512_fmadd_ps(vscaled1x2, vfilter_output_scale2, vbias2);
    vscaled1x3 = _mm512_fmadd_ps(vscaled1x3, vfilter_output_scale3, vbias3);
    vscaled2x0 = _mm512_fmadd_ps(vscaled2x0, vfilter_output_scale0, vbias0);
    vscaled2x1 = _mm512_fmadd_ps(vscaled2x1, vfilter_output_scale1, vbias1);
    vscaled2x2 = _mm512_fmadd_ps(vscaled2x2, vfilter_output_scale2, vbias2);
    vscaled2x3 = _mm512_fmadd_ps(vscaled2x3, vfilter_output_scale3, vbias3);
    vscaled3x0 = _mm512_fmadd_ps(vscaled3x0, vfilter_output_scale0, vbias0);
    vscaled3x1 = _mm512_fmadd_ps(vscaled3x1, vfilter_output_scale1, vbias1);
    vscaled3x2 = _mm512_fmadd_ps(vscaled3x2, vfilter_output_scale2, vbias2);
    vscaled3x3 = _mm512_fmadd_ps(vscaled3x3, vfilter_output_scale3, vbias3);
    vscaled4x0 = _mm512_fmadd_ps(vscaled4x0, vfilter_output_scale0, vbias0);
    vscaled4x1 = _mm512_fmadd_ps(vscaled4x1, vfilter_output_scale1, vbias1);
    vscaled4x2 = _mm512_fmadd_ps(vscaled4x2, vfilter_output_scale2, vbias2);
    vscaled4x3 = _mm512_fmadd_ps(vscaled4x3, vfilter_output_scale3, vbias3);
    vscaled5x0 = _mm512_fmadd_ps(vscaled5x0, vfilter_output_scale0, vbias0);
    vscaled5x1 = _mm512_fmadd_ps(vscaled5x1, vfilter_output_scale1, vbias1);
    vscaled5x2 = _mm512_fmadd_ps(vscaled5x2, vfilter_output_scale2, vbias2);
    vscaled5x3 = _mm512_fmadd_ps(vscaled5x3, vfilter_output_scale3, vbias3);
    vscaled6x0 = _mm512_fmadd_ps(vscaled6x0, vfilter_output_scale0, vbias0);
    vscaled6x1 = _mm512_fmadd_ps(vscaled6x1, vfilter_output_scale1, vbias1);
    vscaled6x2 = _mm512_fmadd_ps(vscaled6x2, vfilter_output_scale2, vbias2);
    vscaled6x3 = _mm512_fmadd_ps(vscaled6x3, vfilter_output_scale3, vbias3);
    vscaled7x0 = _mm512_fmadd_ps(vscaled7x0, vfilter_output_scale0, vbias0);
    vscaled7x1 = _mm512_fmadd_ps(vscaled7x1, vfilter_output_scale1, vbias1);
    vscaled7x2 = _mm512_fmadd_ps(vscaled7x2, vfilter_output_scale2, vbias2);
    vscaled7x3 = _mm512_fmadd_ps(vscaled7x3, vfilter_output_scale3, vbias3);
    vscaled8x0 = _mm512_fmadd_ps(vscaled8x0, vfilter_output_scale0, vbias0);
    vscaled8x1 = _mm512_fmadd_ps(vscaled8x1, vfilter_output_scale1, vbias1);
    vscaled8x2 = _mm512_fmadd_ps(vscaled8x2, vfilter_output_scale2, vbias2);
    vscaled8x3 = _mm512_fmadd_ps(vscaled8x3, vfilter_output_scale3, vbias3);
    vscaled9x0 = _mm512_fmadd_ps(vscaled9x0, vfilter_output_scale0, vbias0);
    vscaled9x1 = _mm512_fmadd_ps(vscaled9x1, vfilter_output_scale1, vbias1);
    vscaled9x2 = _mm512_fmadd_ps(vscaled9x2, vfilter_output_scale2, vbias2);
    vscaled9x3 = _mm512_fmadd_ps(vscaled9x3, vfilter_output_scale3, vbias3);
    vscaled10x0 = _mm512_fmadd_ps(vscaled10x0, vfilter_output_scale0, vbias0);
    vscaled10x1 = _mm512_fmadd_ps(vscaled10x1, vfilter_output_scale1, vbias1);
    vscaled10x2 = _mm512_fmadd_ps(vscaled10x2, vfilter_output_scale2, vbias2);
    vscaled10x3 = _mm512_fmadd_ps(vscaled10x3, vfilter_output_scale3, vbias3);
    vscaled11x0 = _mm512_fmadd_ps(vscaled11x0, vfilter_output_scale0, vbias0);
    vscaled11x1 = _mm512_fmadd_ps(vscaled11x1, vfilter_output_scale1, vbias1);
    vscaled11x2 = _mm512_fmadd_ps(vscaled11x2, vfilter_output_scale2, vbias2);
    vscaled11x3 = _mm512_fmadd_ps(vscaled11x3, vfilter_output_scale3, vbias3);
    vscaled12x0 = _mm512_fmadd_ps(vscaled12x0, vfilter_output_scale0, vbias0);
    vscaled12x1 = _mm512_fmadd_ps(vscaled12x1, vfilter_output_scale1, vbias1);
    vscaled12x2 = _mm512_fmadd_ps(vscaled12x2, vfilter_output_scale2, vbias2);
    vscaled12x3 = _mm512_fmadd_ps(vscaled12x3, vfilter_output_scale3, vbias3);
    vscaled13x0 = _mm512_fmadd_ps(vscaled13x0, vfilter_output_scale0, vbias0);
    vscaled13x1 = _mm512_fmadd_ps(vscaled13x1, vfilter_output_scale1, vbias1);
    vscaled13x2 = _mm512_fmadd_ps(vscaled13x2, vfilter_output_scale2, vbias2);
    vscaled13x3 = _mm512_fmadd_ps(vscaled13x3, vfilter_output_scale3, vbias3);
    vscaled14x0 = _mm512_fmadd_ps(vscaled14x0, vfilter_output_scale0, vbias0);
    vscaled14x1 = _mm512_fmadd_ps(vscaled14x1, vfilter_output_scale1, vbias1);
    vscaled14x2 = _mm512_fmadd_ps(vscaled14x2, vfilter_output_scale2, vbias2);
    vscaled14x3 = _mm512_fmadd_ps(vscaled14x3, vfilter_output_scale3, vbias3);
    vscaled15x0 = _mm512_fmadd_ps(vscaled15x0, vfilter_output_scale0, vbias0);
    vscaled15x1 = _mm512_fmadd_ps(vscaled15x1, vfilter_output_scale1, vbias1);
    vscaled15x2 = _mm512_fmadd_ps(vscaled15x2, vfilter_output_scale2, vbias2);
    vscaled15x3 = _mm512_fmadd_ps(vscaled15x3, vfilter_output_scale3, vbias3);

    vscaled0x0 = _mm512_max_ps(vscaled0x0, voutput_min);
    vscaled0x1 = _mm512_max_ps(vscaled0x1, voutput_min);
    vscaled0x2 = _mm512_max_ps(vscaled0x2, voutput_min);
    vscaled0x3 = _mm512_max_ps(vscaled0x3, voutput_min);
    vscaled1x0 = _mm512_max_ps(vscaled1x0, voutput_min);
    vscaled1x1 = _mm512_max_ps(vscaled1x1, voutput_min);
    vscaled1x2 = _mm512_max_ps(vscaled1x2, voutput_min);
    vscaled1x3 = _mm512_max_ps(vscaled1x3, voutput_min);
    vscaled2x0 = _mm512_max_ps(vscaled2x0, voutput_min);
    vscaled2x1 = _mm512_max_ps(vscaled2x1, voutput_min);
    vscaled2x2 = _mm512_max_ps(vscaled2x2, voutput_min);
    vscaled2x3 = _mm512_max_ps(vscaled2x3, voutput_min);
    vscaled3x0 = _mm512_max_ps(vscaled3x0, voutput_min);
    vscaled3x1 = _mm512_max_ps(vscaled3x1, voutput_min);
    vscaled3x2 = _mm512_max_ps(vscaled3x2, voutput_min);
    vscaled3x3 = _mm512_max_ps(vscaled3x3, voutput_min);
    vscaled4x0 = _mm512_max_ps(vscaled4x0, voutput_min);
    vscaled4x1 = _mm512_max_ps(vscaled4x1, voutput_min);
    vscaled4x2 = _mm512_max_ps(vscaled4x2, voutput_min);
    vscaled4x3 = _mm512_max_ps(vscaled4x3, voutput_min);
    vscaled5x0 = _mm512_max_ps(vscaled5x0, voutput_min);
    vscaled5x1 = _mm512_max_ps(vscaled5x1, voutput_min);
    vscaled5x2 = _mm512_max_ps(vscaled5x2, voutput_min);
    vscaled5x3 = _mm512_max_ps(vscaled5x3, voutput_min);
    vscaled6x0 = _mm512_max_ps(vscaled6x0, voutput_min);
    vscaled6x1 = _mm512_max_ps(vscaled6x1, voutput_min);
    vscaled6x2 = _mm512_max_ps(vscaled6x2, voutput_min);
    vscaled6x3 = _mm512_max_ps(vscaled6x3, voutput_min);
    vscaled7x0 = _mm512_max_ps(vscaled7x0, voutput_min);
    vscaled7x1 = _mm512_max_ps(vscaled7x1, voutput_min);
    vscaled7x2 = _mm512_max_ps(vscaled7x2, voutput_min);
    vscaled7x3 = _mm512_max_ps(vscaled7x3, voutput_min);
    vscaled8x0 = _mm512_max_ps(vscaled8x0, voutput_min);
    vscaled8x1 = _mm512_max_ps(vscaled8x1, voutput_min);
    vscaled8x2 = _mm512_max_ps(vscaled8x2, voutput_min);
    vscaled8x3 = _mm512_max_ps(vscaled8x3, voutput_min);
    vscaled9x0 = _mm512_max_ps(vscaled9x0, voutput_min);
    vscaled9x1 = _mm512_max_ps(vscaled9x1, voutput_min);
    vscaled9x2 = _mm512_max_ps(vscaled9x2, voutput_min);
    vscaled9x3 = _mm512_max_ps(vscaled9x3, voutput_min);
    vscaled10x0 = _mm512_max_ps(vscaled10x0, voutput_min);
    vscaled10x1 = _mm512_max_ps(vscaled10x1, voutput_min);
    vscaled10x2 = _mm512_max_ps(vscaled10x2, voutput_min);
    vscaled10x3 = _mm512_max_ps(vscaled10x3, voutput_min);
    vscaled11x0 = _mm512_max_ps(vscaled11x0, voutput_min);
    vscaled11x1 = _mm512_max_ps(vscaled11x1, voutput_min);
    vscaled11x2 = _mm512_max_ps(vscaled11x2, voutput_min);
    vscaled11x3 = _mm512_max_ps(vscaled11x3, voutput_min);
    vscaled12x0 = _mm512_max_ps(vscaled12x0, voutput_min);
    vscaled12x1 = _mm512_max_ps(vscaled12x1, voutput_min);
    vscaled12x2 = _mm512_max_ps(vscaled12x2, voutput_min);
    vscaled12x3 = _mm512_max_ps(vscaled12x3, voutput_min);
    vscaled13x0 = _mm512_max_ps(vscaled13x0, voutput_min);
    vscaled13x1 = _mm512_max_ps(vscaled13x1, voutput_min);
    vscaled13x2 = _mm512_max_ps(vscaled13x2, voutput_min);
    vscaled13x3 = _mm512_max_ps(vscaled13x3, voutput_min);
    vscaled14x0 = _mm512_max_ps(vscaled14x0, voutput_min);
    vscaled14x1 = _mm512_max_ps(vscaled14x1, voutput_min);
    vscaled14x2 = _mm512_max_ps(vscaled14x2, voutput_min);
    vscaled14x3 = _mm512_max_ps(vscaled14x3, voutput_min);
    vscaled15x0 = _mm512_max_ps(vscaled15x0, voutput_min);
    vscaled15x1 = _mm512_max_ps(vscaled15x1, voutput_min);
    vscaled15x2 = _mm512_max_ps(vscaled15x2, voutput_min);
    vscaled15x3 = _mm512_max_ps(vscaled15x3, voutput_min);

    vscaled0x0 = _mm512_min_ps(vscaled0x0, voutput_max);
    vscaled0x1 = _mm512_min_ps(vscaled0x1, voutput_max);
    vscaled0x2 = _mm512_min_ps(vscaled0x2, voutput_max);
    vscaled0x3 = _mm512_min_ps(vscaled0x3, voutput_max);
    vscaled1x0 = _mm512_min_ps(vscaled1x0, voutput_max);
    vscaled1x1 = _mm512_min_ps(vscaled1x1, voutput_max);
    vscaled1x2 = _mm512_min_ps(vscaled1x2, voutput_max);
    vscaled1x3 = _mm512_min_ps(vscaled1x3, voutput_max);
    vscaled2x0 = _mm512_min_ps(vscaled2x0, voutput_max);
    vscaled2x1 = _mm512_min_ps(vscaled2x1, voutput_max);
    vscaled2x2 = _mm512_min_ps(vscaled2x2, voutput_max);
    vscaled2x3 = _mm512_min_ps(vscaled2x3, voutput_max);
    vscaled3x0 = _mm512_min_ps(vscaled3x0, voutput_max);
    vscaled3x1 = _mm512_min_ps(vscaled3x1, voutput_max);
    vscaled3x2 = _mm512_min_ps(vscaled3x2, voutput_max);
    vscaled3x3 = _mm512_min_ps(vscaled3x3, voutput_max);
    vscaled4x0 = _mm512_min_ps(vscaled4x0, voutput_max);
    vscaled4x1 = _mm512_min_ps(vscaled4x1, voutput_max);
    vscaled4x2 = _mm512_min_ps(vscaled4x2, voutput_max);
    vscaled4x3 = _mm512_min_ps(vscaled4x3, voutput_max);
    vscaled5x0 = _mm512_min_ps(vscaled5x0, voutput_max);
    vscaled5x1 = _mm512_min_ps(vscaled5x1, voutput_max);
    vscaled5x2 = _mm512_min_ps(vscaled5x2, voutput_max);
    vscaled5x3 = _mm512_min_ps(vscaled5x3, voutput_max);
    vscaled6x0 = _mm512_min_ps(vscaled6x0, voutput_max);
    vscaled6x1 = _mm512_min_ps(vscaled6x1, voutput_max);
    vscaled6x2 = _mm512_min_ps(vscaled6x2, voutput_max);
    vscaled6x3 = _mm512_min_ps(vscaled6x3, voutput_max);
    vscaled7x0 = _mm512_min_ps(vscaled7x0, voutput_max);
    vscaled7x1 = _mm512_min_ps(vscaled7x1, voutput_max);
    vscaled7x2 = _mm512_min_ps(vscaled7x2, voutput_max);
    vscaled7x3 = _mm512_min_ps(vscaled7x3, voutput_max);
    vscaled8x0 = _mm512_min_ps(vscaled8x0, voutput_max);
    vscaled8x1 = _mm512_min_ps(vscaled8x1, voutput_max);
    vscaled8x2 = _mm512_min_ps(vscaled8x2, voutput_max);
    vscaled8x3 = _mm512_min_ps(vscaled8x3, voutput_max);
    vscaled9x0 = _mm512_min_ps(vscaled9x0, voutput_max);
    vscaled9x1 = _mm512_min_ps(vscaled9x1, voutput_max);
    vscaled9x2 = _mm512_min_ps(vscaled9x2, voutput_max);
    vscaled9x3 = _mm512_min_ps(vscaled9x3, voutput_max);
    vscaled10x0 = _mm512_min_ps(vscaled10x0, voutput_max);
    vscaled10x1 = _mm512_min_ps(vscaled10x1, voutput_max);
    vscaled10x2 = _mm512_min_ps(vscaled10x2, voutput_max);
    vscaled10x3 = _mm512_min_ps(vscaled10x3, voutput_max);
    vscaled11x0 = _mm512_min_ps(vscaled11x0, voutput_max);
    vscaled11x1 = _mm512_min_ps(vscaled11x1, voutput_max);
    vscaled11x2 = _mm512_min_ps(vscaled11x2, voutput_max);
    vscaled11x3 = _mm512_min_ps(vscaled11x3, voutput_max);
    vscaled12x0 = _mm512_min_ps(vscaled12x0, voutput_max);
    vscaled12x1 = _mm512_min_ps(vscaled12x1, voutput_max);
    vscaled12x2 = _mm512_min_ps(vscaled12x2, voutput_max);
    vscaled12x3 = _mm512_min_ps(vscaled12x3, voutput_max);
    vscaled13x0 = _mm512_min_ps(vscaled13x0, voutput_max);
    vscaled13x1 = _mm512_min_ps(vscaled13x1, voutput_max);
    vscaled13x2 = _mm512_min_ps(vscaled13x2, voutput_max);
    vscaled13x3 = _mm512_min_ps(vscaled13x3, voutput_max);
    vscaled14x0 = _mm512_min_ps(vscaled14x0, voutput_max);
    vscaled14x1 = _mm512_min_ps(vscaled14x1, voutput_max);
    vscaled14x2 = _mm512_min_ps(vscaled14x2, voutput_max);
    vscaled14x3 = _mm512_min_ps(vscaled14x3, voutput_max);
    vscaled15x0 = _mm512_min_ps(vscaled15x0, voutput_max);
    vscaled15x1 = _mm512_min_ps(vscaled15x1, voutput_max);
    vscaled15x2 = _mm512_min_ps(vscaled15x2, voutput_max);
    vscaled15x3 = _mm512_min_ps(vscaled15x3, voutput_max);

    __m256i vfp16out0x0 = _mm512_cvtps_ph(vscaled0x0, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out0x1 = _mm512_cvtps_ph(vscaled0x1, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out0x2 = _mm512_cvtps_ph(vscaled0x2, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out0x3 = _mm512_cvtps_ph(vscaled0x3, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out1x0 = _mm512_cvtps_ph(vscaled1x0, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out1x1 = _mm512_cvtps_ph(vscaled1x1, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out1x2 = _mm512_cvtps_ph(vscaled1x2, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out1x3 = _mm512_cvtps_ph(vscaled1x3, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out2x0 = _mm512_cvtps_ph(vscaled2x0, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out2x1 = _mm512_cvtps_ph(vscaled2x1, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out2x2 = _mm512_cvtps_ph(vscaled2x2, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out2x3 = _mm512_cvtps_ph(vscaled2x3, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out3x0 = _mm512_cvtps_ph(vscaled3x0, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out3x1 = _mm512_cvtps_ph(vscaled3x1, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out3x2 = _mm512_cvtps_ph(vscaled3x2, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out3x3 = _mm512_cvtps_ph(vscaled3x3, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out4x0 = _mm512_cvtps_ph(vscaled4x0, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out4x1 = _mm512_cvtps_ph(vscaled4x1, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out4x2 = _mm512_cvtps_ph(vscaled4x2, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out4x3 = _mm512_cvtps_ph(vscaled4x3, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out5x0 = _mm512_cvtps_ph(vscaled5x0, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out5x1 = _mm512_cvtps_ph(vscaled5x1, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out5x2 = _mm512_cvtps_ph(vscaled5x2, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out5x3 = _mm512_cvtps_ph(vscaled5x3, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out6x0 = _mm512_cvtps_ph(vscaled6x0, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out6x1 = _mm512_cvtps_ph(vscaled6x1, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out6x2 = _mm512_cvtps_ph(vscaled6x2, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out6x3 = _mm512_cvtps_ph(vscaled6x3, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out7x0 = _mm512_cvtps_ph(vscaled7x0, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out7x1 = _mm512_cvtps_ph(vscaled7x1, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out7x2 = _mm512_cvtps_ph(vscaled7x2, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out7x3 = _mm512_cvtps_ph(vscaled7x3, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out8x0 = _mm512_cvtps_ph(vscaled8x0, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out8x1 = _mm512_cvtps_ph(vscaled8x1, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out8x2 = _mm512_cvtps_ph(vscaled8x2, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out8x3 = _mm512_cvtps_ph(vscaled8x3, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out9x0 = _mm512_cvtps_ph(vscaled9x0, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out9x1 = _mm512_cvtps_ph(vscaled9x1, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out9x2 = _mm512_cvtps_ph(vscaled9x2, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out9x3 = _mm512_cvtps_ph(vscaled9x3, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out10x0 = _mm512_cvtps_ph(vscaled10x0, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out10x1 = _mm512_cvtps_ph(vscaled10x1, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out10x2 = _mm512_cvtps_ph(vscaled10x2, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out10x3 = _mm512_cvtps_ph(vscaled10x3, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out11x0 = _mm512_cvtps_ph(vscaled11x0, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out11x1 = _mm512_cvtps_ph(vscaled11x1, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out11x2 = _mm512_cvtps_ph(vscaled11x2, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out11x3 = _mm512_cvtps_ph(vscaled11x3, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out12x0 = _mm512_cvtps_ph(vscaled12x0, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out12x1 = _mm512_cvtps_ph(vscaled12x1, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out12x2 = _mm512_cvtps_ph(vscaled12x2, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out12x3 = _mm512_cvtps_ph(vscaled12x3, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out13x0 = _mm512_cvtps_ph(vscaled13x0, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out13x1 = _mm512_cvtps_ph(vscaled13x1, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out13x2 = _mm512_cvtps_ph(vscaled13x2, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out13x3 = _mm512_cvtps_ph(vscaled13x3, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out14x0 = _mm512_cvtps_ph(vscaled14x0, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out14x1 = _mm512_cvtps_ph(vscaled14x1, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out14x2 = _mm512_cvtps_ph(vscaled14x2, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out14x3 = _mm512_cvtps_ph(vscaled14x3, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out15x0 = _mm512_cvtps_ph(vscaled15x0, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out15x1 = _mm512_cvtps_ph(vscaled15x1, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out15x2 = _mm512_cvtps_ph(vscaled15x2, _MM_FROUND_TO_NEAREST_INT);
    __m256i vfp16out15x3 = _mm512_cvtps_ph(vscaled15x3, _MM_FROUND_TO_NEAREST_INT);
    if XNN_LIKELY(nc >= 64) {
      _mm256_storeu_si256((__m256i*) (c15 + 0), vfp16out15x0);
      _mm256_storeu_si256((__m256i*) (c15 + 16), vfp16out15x1);
      _mm256_storeu_si256((__m256i*) (c15 + 32), vfp16out15x2);
      _mm256_storeu_si256((__m256i*) (c15 + 48), vfp16out15x3);
      c15 = (xnn_float16*) ((uintptr_t) c15 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c14 + 0), vfp16out14x0);
      _mm256_storeu_si256((__m256i*) (c14 + 16), vfp16out14x1);
      _mm256_storeu_si256((__m256i*) (c14 + 32), vfp16out14x2);
      _mm256_storeu_si256((__m256i*) (c14 + 48), vfp16out14x3);
      c14 = (xnn_float16*) ((uintptr_t) c14 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c13 + 0), vfp16out13x0);
      _mm256_storeu_si256((__m256i*) (c13 + 16), vfp16out13x1);
      _mm256_storeu_si256((__m256i*) (c13 + 32), vfp16out13x2);
      _mm256_storeu_si256((__m256i*) (c13 + 48), vfp16out13x3);
      c13 = (xnn_float16*) ((uintptr_t) c13 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c12 + 0), vfp16out12x0);
      _mm256_storeu_si256((__m256i*) (c12 + 16), vfp16out12x1);
      _mm256_storeu_si256((__m256i*) (c12 + 32), vfp16out12x2);
      _mm256_storeu_si256((__m256i*) (c12 + 48), vfp16out12x3);
      c12 = (xnn_float16*) ((uintptr_t) c12 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c11 + 0), vfp16out11x0);
      _mm256_storeu_si256((__m256i*) (c11 + 16), vfp16out11x1);
      _mm256_storeu_si256((__m256i*) (c11 + 32), vfp16out11x2);
      _mm256_storeu_si256((__m256i*) (c11 + 48), vfp16out11x3);
      c11 = (xnn_float16*) ((uintptr_t) c11 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c10 + 0), vfp16out10x0);
      _mm256_storeu_si256((__m256i*) (c10 + 16), vfp16out10x1);
      _mm256_storeu_si256((__m256i*) (c10 + 32), vfp16out10x2);
      _mm256_storeu_si256((__m256i*) (c10 + 48), vfp16out10x3);
      c10 = (xnn_float16*) ((uintptr_t) c10 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c9 + 0), vfp16out9x0);
      _mm256_storeu_si256((__m256i*) (c9 + 16), vfp16out9x1);
      _mm256_storeu_si256((__m256i*) (c9 + 32), vfp16out9x2);
      _mm256_storeu_si256((__m256i*) (c9 + 48), vfp16out9x3);
      c9 = (xnn_float16*) ((uintptr_t) c9 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c8 + 0), vfp16out8x0);
      _mm256_storeu_si256((__m256i*) (c8 + 16), vfp16out8x1);
      _mm256_storeu_si256((__m256i*) (c8 + 32), vfp16out8x2);
      _mm256_storeu_si256((__m256i*) (c8 + 48), vfp16out8x3);
      c8 = (xnn_float16*) ((uintptr_t) c8 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c7 + 0), vfp16out7x0);
      _mm256_storeu_si256((__m256i*) (c7 + 16), vfp16out7x1);
      _mm256_storeu_si256((__m256i*) (c7 + 32), vfp16out7x2);
      _mm256_storeu_si256((__m256i*) (c7 + 48), vfp16out7x3);
      c7 = (xnn_float16*) ((uintptr_t) c7 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c6 + 0), vfp16out6x0);
      _mm256_storeu_si256((__m256i*) (c6 + 16), vfp16out6x1);
      _mm256_storeu_si256((__m256i*) (c6 + 32), vfp16out6x2);
      _mm256_storeu_si256((__m256i*) (c6 + 48), vfp16out6x3);
      c6 = (xnn_float16*) ((uintptr_t) c6 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c5 + 0), vfp16out5x0);
      _mm256_storeu_si256((__m256i*) (c5 + 16), vfp16out5x1);
      _mm256_storeu_si256((__m256i*) (c5 + 32), vfp16out5x2);
      _mm256_storeu_si256((__m256i*) (c5 + 48), vfp16out5x3);
      c5 = (xnn_float16*) ((uintptr_t) c5 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c4 + 0), vfp16out4x0);
      _mm256_storeu_si256((__m256i*) (c4 + 16), vfp16out4x1);
      _mm256_storeu_si256((__m256i*) (c4 + 32), vfp16out4x2);
      _mm256_storeu_si256((__m256i*) (c4 + 48), vfp16out4x3);
      c4 = (xnn_float16*) ((uintptr_t) c4 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c3 + 0), vfp16out3x0);
      _mm256_storeu_si256((__m256i*) (c3 + 16), vfp16out3x1);
      _mm256_storeu_si256((__m256i*) (c3 + 32), vfp16out3x2);
      _mm256_storeu_si256((__m256i*) (c3 + 48), vfp16out3x3);
      c3 = (xnn_float16*) ((uintptr_t) c3 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c2 + 0), vfp16out2x0);
      _mm256_storeu_si256((__m256i*) (c2 + 16), vfp16out2x1);
      _mm256_storeu_si256((__m256i*) (c2 + 32), vfp16out2x2);
      _mm256_storeu_si256((__m256i*) (c2 + 48), vfp16out2x3);
      c2 = (xnn_float16*) ((uintptr_t) c2 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c1 + 0), vfp16out1x0);
      _mm256_storeu_si256((__m256i*) (c1 + 16), vfp16out1x1);
      _mm256_storeu_si256((__m256i*) (c1 + 32), vfp16out1x2);
      _mm256_storeu_si256((__m256i*) (c1 + 48), vfp16out1x3);
      c1 = (xnn_float16*) ((uintptr_t) c1 + cn_stride);
      _mm256_storeu_si256((__m256i*) (c0 + 0), vfp16out0x0);
      _mm256_storeu_si256((__m256i*) (c0 + 16), vfp16out0x1);
      _mm256_storeu_si256((__m256i*) (c0 + 32), vfp16out0x2);
      _mm256_storeu_si256((__m256i*) (c0 + 48), vfp16out0x3);
      c0 = (xnn_float16*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 64;
    } else {
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 0) & 0xFFFF));
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 16) & 0xFFFF));
      const __mmask16 vmask2 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 32) & 0xFFFF));
      const __mmask16 vmask3 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 48) & 0xFFFF));
      _mm256_mask_storeu_epi16(c15 + 0, vmask0, vfp16out15x0);
      _mm256_mask_storeu_epi16(c15 + 16, vmask1, vfp16out15x1);
      _mm256_mask_storeu_epi16(c15 + 32, vmask2, vfp16out15x2);
      _mm256_mask_storeu_epi16(c15 + 48, vmask3, vfp16out15x3);
      _mm256_mask_storeu_epi16(c14 + 0, vmask0, vfp16out14x0);
      _mm256_mask_storeu_epi16(c14 + 16, vmask1, vfp16out14x1);
      _mm256_mask_storeu_epi16(c14 + 32, vmask2, vfp16out14x2);
      _mm256_mask_storeu_epi16(c14 + 48, vmask3, vfp16out14x3);
      _mm256_mask_storeu_epi16(c13 + 0, vmask0, vfp16out13x0);
      _mm256_mask_storeu_epi16(c13 + 16, vmask1, vfp16out13x1);
      _mm256_mask_storeu_epi16(c13 + 32, vmask2, vfp16out13x2);
      _mm256_mask_storeu_epi16(c13 + 48, vmask3, vfp16out13x3);
      _mm256_mask_storeu_epi16(c12 + 0, vmask0, vfp16out12x0);
      _mm256_mask_storeu_epi16(c12 + 16, vmask1, vfp16out12x1);
      _mm256_mask_storeu_epi16(c12 + 32, vmask2, vfp16out12x2);
      _mm256_mask_storeu_epi16(c12 + 48, vmask3, vfp16out12x3);
      _mm256_mask_storeu_epi16(c11 + 0, vmask0, vfp16out11x0);
      _mm256_mask_storeu_epi16(c11 + 16, vmask1, vfp16out11x1);
      _mm256_mask_storeu_epi16(c11 + 32, vmask2, vfp16out11x2);
      _mm256_mask_storeu_epi16(c11 + 48, vmask3, vfp16out11x3);
      _mm256_mask_storeu_epi16(c10 + 0, vmask0, vfp16out10x0);
      _mm256_mask_storeu_epi16(c10 + 16, vmask1, vfp16out10x1);
      _mm256_mask_storeu_epi16(c10 + 32, vmask2, vfp16out10x2);
      _mm256_mask_storeu_epi16(c10 + 48, vmask3, vfp16out10x3);
      _mm256_mask_storeu_epi16(c9 + 0, vmask0, vfp16out9x0);
      _mm256_mask_storeu_epi16(c9 + 16, vmask1, vfp16out9x1);
      _mm256_mask_storeu_epi16(c9 + 32, vmask2, vfp16out9x2);
      _mm256_mask_storeu_epi16(c9 + 48, vmask3, vfp16out9x3);
      _mm256_mask_storeu_epi16(c8 + 0, vmask0, vfp16out8x0);
      _mm256_mask_storeu_epi16(c8 + 16, vmask1, vfp16out8x1);
      _mm256_mask_storeu_epi16(c8 + 32, vmask2, vfp16out8x2);
      _mm256_mask_storeu_epi16(c8 + 48, vmask3, vfp16out8x3);
      _mm256_mask_storeu_epi16(c7 + 0, vmask0, vfp16out7x0);
      _mm256_mask_storeu_epi16(c7 + 16, vmask1, vfp16out7x1);
      _mm256_mask_storeu_epi16(c7 + 32, vmask2, vfp16out7x2);
      _mm256_mask_storeu_epi16(c7 + 48, vmask3, vfp16out7x3);
      _mm256_mask_storeu_epi16(c6 + 0, vmask0, vfp16out6x0);
      _mm256_mask_storeu_epi16(c6 + 16, vmask1, vfp16out6x1);
      _mm256_mask_storeu_epi16(c6 + 32, vmask2, vfp16out6x2);
      _mm256_mask_storeu_epi16(c6 + 48, vmask3, vfp16out6x3);
      _mm256_mask_storeu_epi16(c5 + 0, vmask0, vfp16out5x0);
      _mm256_mask_storeu_epi16(c5 + 16, vmask1, vfp16out5x1);
      _mm256_mask_storeu_epi16(c5 + 32, vmask2, vfp16out5x2);
      _mm256_mask_storeu_epi16(c5 + 48, vmask3, vfp16out5x3);
      _mm256_mask_storeu_epi16(c4 + 0, vmask0, vfp16out4x0);
      _mm256_mask_storeu_epi16(c4 + 16, vmask1, vfp16out4x1);
      _mm256_mask_storeu_epi16(c4 + 32, vmask2, vfp16out4x2);
      _mm256_mask_storeu_epi16(c4 + 48, vmask3, vfp16out4x3);
      _mm256_mask_storeu_epi16(c3 + 0, vmask0, vfp16out3x0);
      _mm256_mask_storeu_epi16(c3 + 16, vmask1, vfp16out3x1);
      _mm256_mask_storeu_epi16(c3 + 32, vmask2, vfp16out3x2);
      _mm256_mask_storeu_epi16(c3 + 48, vmask3, vfp16out3x3);
      _mm256_mask_storeu_epi16(c2 + 0, vmask0, vfp16out2x0);
      _mm256_mask_storeu_epi16(c2 + 16, vmask1, vfp16out2x1);
      _mm256_mask_storeu_epi16(c2 + 32, vmask2, vfp16out2x2);
      _mm256_mask_storeu_epi16(c2 + 48, vmask3, vfp16out2x3);
      _mm256_mask_storeu_epi16(c1 + 0, vmask0, vfp16out1x0);
      _mm256_mask_storeu_epi16(c1 + 16, vmask1, vfp16out1x1);
      _mm256_mask_storeu_epi16(c1 + 32, vmask2, vfp16out1x2);
      _mm256_mask_storeu_epi16(c1 + 48, vmask3, vfp16out1x3);
      _mm256_mask_storeu_epi16(c0 + 0, vmask0, vfp16out0x0);
      _mm256_mask_storeu_epi16(c0 + 16, vmask1, vfp16out0x1);
      _mm256_mask_storeu_epi16(c0 + 32, vmask2, vfp16out0x2);
      _mm256_mask_storeu_epi16(c0 + 48, vmask3, vfp16out0x3);
      nc = 0;
    }
  } while (nc != 0);

  // Release tile config
  _tile_release();
  #endif  // defined(__x86_64__)
}
