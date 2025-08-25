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


void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x32c4__avx512amx(
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
    const union xnn_qs8_qc8w_conv_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 7);
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
  XNN_ALIGN(64) int32_t res[2][7 * 16];
  XNN_ALIGN(64) int32_t vintile[7 * 16];

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

  const __m512 voutput_max_less_zero_point = _mm512_set1_ps((int32_t) params->fp32_scalar.output_max - (int32_t) params->fp32_scalar.output_zero_point);
  const __m512i voutput_zero_point = _mm512_set1_epi32(params->fp32_scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8(params->fp32_scalar.output_min);
  // XNN_FORCE_REALIZATION(voutput_max_less_zero_point);
  // XNN_FORCE_REALIZATION(voutput_zero_point);
  // XNN_FORCE_REALIZATION(voutput_min);

  do {
    const __m512i vksum0 = _mm512_loadu_epi32((const int32_t*) w + 0);
    const __m512i vksum1 = _mm512_loadu_epi32((const int32_t*) w + 16);
    w = (const int32_t*) w + 32;

    // Zero tile accumulator
    _tile_zero(0);
    _tile_zero(1);

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
      a += 7;

      size_t k = kc;
      if (mr == 1)
      {
        while (k >= 64 * sizeof(int8_t)) {
          _tile_loadd(4, a0, 64);   // Directly load input for mr=1
          a0 += 64;
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
        _tile_loadd(6, vintile, 64);
        _tile_loadd(7, (const int8_t*) w + 0, 128);
        _tile_dpbssd(0, 6, 7);
        _tile_loadd(7, (const int8_t*) w + 64, 128);
        _tile_dpbssd(1, 6, 7);

        w = (const int8_t*) w + kremainder * 32;
        k -= kremainder * sizeof(int8_t);
      }

      p -= 7 * sizeof(void*);
    } while (p != 0);

    // TODO: Instead of processing up to 4 tiles (16x64) consider
    // quantizing 1 tile at a time (16 registers)
    _tile_stored(0, &res[0][0], 64);
    _tile_stored(1, &res[1][0], 64);

    // TODO: Fix msan for AMX
    #if defined(__has_feature)
      #if __has_feature(memory_sanitizer)
        __msan_unpoison(res, sizeof(res));
      #endif
    #endif

    // TODO: Instead of processing up to 4 tiles (16x64) consider
    // quantizing 1 row at a time.
    // Add tile to bias
    __m512i vacc0x0 = _mm512_add_epi32(vksum0, _mm512_load_epi32(&res[0][0] + 0));
    __m512i vacc0x1 = _mm512_add_epi32(vksum1, _mm512_load_epi32(&res[1][0] + 0));
    __m512i vacc1x0 = _mm512_add_epi32(vksum0, _mm512_load_epi32(&res[0][0] + 16));
    __m512i vacc1x1 = _mm512_add_epi32(vksum1, _mm512_load_epi32(&res[1][0] + 16));
    __m512i vacc2x0 = _mm512_add_epi32(vksum0, _mm512_load_epi32(&res[0][0] + 32));
    __m512i vacc2x1 = _mm512_add_epi32(vksum1, _mm512_load_epi32(&res[1][0] + 32));
    __m512i vacc3x0 = _mm512_add_epi32(vksum0, _mm512_load_epi32(&res[0][0] + 48));
    __m512i vacc3x1 = _mm512_add_epi32(vksum1, _mm512_load_epi32(&res[1][0] + 48));
    __m512i vacc4x0 = _mm512_add_epi32(vksum0, _mm512_load_epi32(&res[0][0] + 64));
    __m512i vacc4x1 = _mm512_add_epi32(vksum1, _mm512_load_epi32(&res[1][0] + 64));
    __m512i vacc5x0 = _mm512_add_epi32(vksum0, _mm512_load_epi32(&res[0][0] + 80));
    __m512i vacc5x1 = _mm512_add_epi32(vksum1, _mm512_load_epi32(&res[1][0] + 80));
    __m512i vacc6x0 = _mm512_add_epi32(vksum0, _mm512_load_epi32(&res[0][0] + 96));
    __m512i vacc6x1 = _mm512_add_epi32(vksum1, _mm512_load_epi32(&res[1][0] + 96));

    __m512 vscaled0x0 = _mm512_cvtepi32_ps(vacc0x0);
    __m512 vscaled0x1 = _mm512_cvtepi32_ps(vacc0x1);
    __m512 vscaled1x0 = _mm512_cvtepi32_ps(vacc1x0);
    __m512 vscaled1x1 = _mm512_cvtepi32_ps(vacc1x1);
    __m512 vscaled2x0 = _mm512_cvtepi32_ps(vacc2x0);
    __m512 vscaled2x1 = _mm512_cvtepi32_ps(vacc2x1);
    __m512 vscaled3x0 = _mm512_cvtepi32_ps(vacc3x0);
    __m512 vscaled3x1 = _mm512_cvtepi32_ps(vacc3x1);
    __m512 vscaled4x0 = _mm512_cvtepi32_ps(vacc4x0);
    __m512 vscaled4x1 = _mm512_cvtepi32_ps(vacc4x1);
    __m512 vscaled5x0 = _mm512_cvtepi32_ps(vacc5x0);
    __m512 vscaled5x1 = _mm512_cvtepi32_ps(vacc5x1);
    __m512 vscaled6x0 = _mm512_cvtepi32_ps(vacc6x0);
    __m512 vscaled6x1 = _mm512_cvtepi32_ps(vacc6x1);

    const __m512 vscale0 = _mm512_loadu_ps((const float*) w + 0);
    const __m512 vscale1 = _mm512_loadu_ps((const float*) w + 16);
    w = (const int32_t*) w + 32;

    vscaled0x0 = _mm512_mul_ps(vscaled0x0, vscale0);
    vscaled0x1 = _mm512_mul_ps(vscaled0x1, vscale1);
    vscaled1x0 = _mm512_mul_ps(vscaled1x0, vscale0);
    vscaled1x1 = _mm512_mul_ps(vscaled1x1, vscale1);
    vscaled2x0 = _mm512_mul_ps(vscaled2x0, vscale0);
    vscaled2x1 = _mm512_mul_ps(vscaled2x1, vscale1);
    vscaled3x0 = _mm512_mul_ps(vscaled3x0, vscale0);
    vscaled3x1 = _mm512_mul_ps(vscaled3x1, vscale1);
    vscaled4x0 = _mm512_mul_ps(vscaled4x0, vscale0);
    vscaled4x1 = _mm512_mul_ps(vscaled4x1, vscale1);
    vscaled5x0 = _mm512_mul_ps(vscaled5x0, vscale0);
    vscaled5x1 = _mm512_mul_ps(vscaled5x1, vscale1);
    vscaled6x0 = _mm512_mul_ps(vscaled6x0, vscale0);
    vscaled6x1 = _mm512_mul_ps(vscaled6x1, vscale1);

    vscaled0x0 = _mm512_min_ps(vscaled0x0, voutput_max_less_zero_point);
    vscaled0x1 = _mm512_min_ps(vscaled0x1, voutput_max_less_zero_point);
    vscaled1x0 = _mm512_min_ps(vscaled1x0, voutput_max_less_zero_point);
    vscaled1x1 = _mm512_min_ps(vscaled1x1, voutput_max_less_zero_point);
    vscaled2x0 = _mm512_min_ps(vscaled2x0, voutput_max_less_zero_point);
    vscaled2x1 = _mm512_min_ps(vscaled2x1, voutput_max_less_zero_point);
    vscaled3x0 = _mm512_min_ps(vscaled3x0, voutput_max_less_zero_point);
    vscaled3x1 = _mm512_min_ps(vscaled3x1, voutput_max_less_zero_point);
    vscaled4x0 = _mm512_min_ps(vscaled4x0, voutput_max_less_zero_point);
    vscaled4x1 = _mm512_min_ps(vscaled4x1, voutput_max_less_zero_point);
    vscaled5x0 = _mm512_min_ps(vscaled5x0, voutput_max_less_zero_point);
    vscaled5x1 = _mm512_min_ps(vscaled5x1, voutput_max_less_zero_point);
    vscaled6x0 = _mm512_min_ps(vscaled6x0, voutput_max_less_zero_point);
    vscaled6x1 = _mm512_min_ps(vscaled6x1, voutput_max_less_zero_point);

    vacc0x0 = _mm512_cvtps_epi32(vscaled0x0);
    vacc0x1 = _mm512_cvtps_epi32(vscaled0x1);
    vacc1x0 = _mm512_cvtps_epi32(vscaled1x0);
    vacc1x1 = _mm512_cvtps_epi32(vscaled1x1);
    vacc2x0 = _mm512_cvtps_epi32(vscaled2x0);
    vacc2x1 = _mm512_cvtps_epi32(vscaled2x1);
    vacc3x0 = _mm512_cvtps_epi32(vscaled3x0);
    vacc3x1 = _mm512_cvtps_epi32(vscaled3x1);
    vacc4x0 = _mm512_cvtps_epi32(vscaled4x0);
    vacc4x1 = _mm512_cvtps_epi32(vscaled4x1);
    vacc5x0 = _mm512_cvtps_epi32(vscaled5x0);
    vacc5x1 = _mm512_cvtps_epi32(vscaled5x1);
    vacc6x0 = _mm512_cvtps_epi32(vscaled6x0);
    vacc6x1 = _mm512_cvtps_epi32(vscaled6x1);

    vacc0x0 = _mm512_add_epi32(vacc0x0, voutput_zero_point);
    vacc0x1 = _mm512_add_epi32(vacc0x1, voutput_zero_point);
    vacc1x0 = _mm512_add_epi32(vacc1x0, voutput_zero_point);
    vacc1x1 = _mm512_add_epi32(vacc1x1, voutput_zero_point);
    vacc2x0 = _mm512_add_epi32(vacc2x0, voutput_zero_point);
    vacc2x1 = _mm512_add_epi32(vacc2x1, voutput_zero_point);
    vacc3x0 = _mm512_add_epi32(vacc3x0, voutput_zero_point);
    vacc3x1 = _mm512_add_epi32(vacc3x1, voutput_zero_point);
    vacc4x0 = _mm512_add_epi32(vacc4x0, voutput_zero_point);
    vacc4x1 = _mm512_add_epi32(vacc4x1, voutput_zero_point);
    vacc5x0 = _mm512_add_epi32(vacc5x0, voutput_zero_point);
    vacc5x1 = _mm512_add_epi32(vacc5x1, voutput_zero_point);
    vacc6x0 = _mm512_add_epi32(vacc6x0, voutput_zero_point);
    vacc6x1 = _mm512_add_epi32(vacc6x1, voutput_zero_point);

    __m128i vout0x0 = _mm512_cvtsepi32_epi8(vacc0x0);
    __m128i vout0x1 = _mm512_cvtsepi32_epi8(vacc0x1);
    __m128i vout1x0 = _mm512_cvtsepi32_epi8(vacc1x0);
    __m128i vout1x1 = _mm512_cvtsepi32_epi8(vacc1x1);
    __m128i vout2x0 = _mm512_cvtsepi32_epi8(vacc2x0);
    __m128i vout2x1 = _mm512_cvtsepi32_epi8(vacc2x1);
    __m128i vout3x0 = _mm512_cvtsepi32_epi8(vacc3x0);
    __m128i vout3x1 = _mm512_cvtsepi32_epi8(vacc3x1);
    __m128i vout4x0 = _mm512_cvtsepi32_epi8(vacc4x0);
    __m128i vout4x1 = _mm512_cvtsepi32_epi8(vacc4x1);
    __m128i vout5x0 = _mm512_cvtsepi32_epi8(vacc5x0);
    __m128i vout5x1 = _mm512_cvtsepi32_epi8(vacc5x1);
    __m128i vout6x0 = _mm512_cvtsepi32_epi8(vacc6x0);
    __m128i vout6x1 = _mm512_cvtsepi32_epi8(vacc6x1);

    vout0x0 = _mm_max_epi8(vout0x0, voutput_min);
    vout0x1 = _mm_max_epi8(vout0x1, voutput_min);
    vout1x0 = _mm_max_epi8(vout1x0, voutput_min);
    vout1x1 = _mm_max_epi8(vout1x1, voutput_min);
    vout2x0 = _mm_max_epi8(vout2x0, voutput_min);
    vout2x1 = _mm_max_epi8(vout2x1, voutput_min);
    vout3x0 = _mm_max_epi8(vout3x0, voutput_min);
    vout3x1 = _mm_max_epi8(vout3x1, voutput_min);
    vout4x0 = _mm_max_epi8(vout4x0, voutput_min);
    vout4x1 = _mm_max_epi8(vout4x1, voutput_min);
    vout5x0 = _mm_max_epi8(vout5x0, voutput_min);
    vout5x1 = _mm_max_epi8(vout5x1, voutput_min);
    vout6x0 = _mm_max_epi8(vout6x0, voutput_min);
    vout6x1 = _mm_max_epi8(vout6x1, voutput_min);

    if XNN_LIKELY(nc >= 32) {
      _mm_storeu_si128((__m128i*) (c6 + 0), vout6x0);
      _mm_storeu_si128((__m128i*) (c6 + 16), vout6x1);
      c6 = (int8_t*) ((uintptr_t) c6 + cn_stride);
      _mm_storeu_si128((__m128i*) (c5 + 0), vout5x0);
      _mm_storeu_si128((__m128i*) (c5 + 16), vout5x1);
      c5 = (int8_t*) ((uintptr_t) c5 + cn_stride);
      _mm_storeu_si128((__m128i*) (c4 + 0), vout4x0);
      _mm_storeu_si128((__m128i*) (c4 + 16), vout4x1);
      c4 = (int8_t*) ((uintptr_t) c4 + cn_stride);
      _mm_storeu_si128((__m128i*) (c3 + 0), vout3x0);
      _mm_storeu_si128((__m128i*) (c3 + 16), vout3x1);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      _mm_storeu_si128((__m128i*) (c2 + 0), vout2x0);
      _mm_storeu_si128((__m128i*) (c2 + 16), vout2x1);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      _mm_storeu_si128((__m128i*) (c1 + 0), vout1x0);
      _mm_storeu_si128((__m128i*) (c1 + 16), vout1x1);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      _mm_storeu_si128((__m128i*) (c0 + 0), vout0x0);
      _mm_storeu_si128((__m128i*) (c0 + 16), vout0x1);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 32;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 0) & 0xFFFF));
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 16) & 0xFFFF));

      _mm_mask_storeu_epi8(c6 + 0, vmask0, vout6x0);
      _mm_mask_storeu_epi8(c6 + 16, vmask1, vout6x1);
      _mm_mask_storeu_epi8(c5 + 0, vmask0, vout5x0);
      _mm_mask_storeu_epi8(c5 + 16, vmask1, vout5x1);
      _mm_mask_storeu_epi8(c4 + 0, vmask0, vout4x0);
      _mm_mask_storeu_epi8(c4 + 16, vmask1, vout4x1);
      _mm_mask_storeu_epi8(c3 + 0, vmask0, vout3x0);
      _mm_mask_storeu_epi8(c3 + 16, vmask1, vout3x1);
      _mm_mask_storeu_epi8(c2 + 0, vmask0, vout2x0);
      _mm_mask_storeu_epi8(c2 + 16, vmask1, vout2x1);
      _mm_mask_storeu_epi8(c1 + 0, vmask0, vout1x0);
      _mm_mask_storeu_epi8(c1 + 16, vmask1, vout1x1);
      _mm_mask_storeu_epi8(c0 + 0, vmask0, vout0x0);
      _mm_mask_storeu_epi8(c0 + 16, vmask1, vout0x1);
      nc = 0;
    }
  } while (nc != 0);

  // Release tile config
  _tile_release();
  #endif  // defined(__x86_64__)
}
