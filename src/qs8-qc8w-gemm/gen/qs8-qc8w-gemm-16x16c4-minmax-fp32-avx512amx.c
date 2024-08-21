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


void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_16x16c4__avx512amx(
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
    w = (const int32_t*) w + 16;

    // Zero tile accumulator
    __asm__ volatile (
      "tilezero %%tmm0\n"
      ::);

    size_t k = kc;
    while (k >= 64 * sizeof(int8_t)) {
      _tile_loadd(4, a, a_stride);
      a += 64;
      _tile_loadd(5, (const int8_t*) w + 0, 64);
      _tile_dpbssd(0, 4, 5);

      w = (const int8_t*) w + 1024;
      k -= 64 * sizeof(int8_t);
    }

    if XNN_UNLIKELY(k != 0) {
      _tile_loadd(6, a, a_stride);
      a += kremainder;
      _tile_loadd(7, (const int8_t*) w + 0, 64);
      _tile_dpbssd(0, 6, 7);

      w = (const int8_t*) w + kremainder * 16;
      k -= kremainder * sizeof(int8_t);
    }

    // Add tile to bias
    _tile_stored(0, res0, 64);

    __m512i vacc0x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 0));
    __m512i vacc1x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 16));
    __m512i vacc2x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 32));
    __m512i vacc3x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 48));
    __m512i vacc4x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 64));
    __m512i vacc5x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 80));
    __m512i vacc6x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 96));
    __m512i vacc7x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 112));
    __m512i vacc8x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 128));
    __m512i vacc9x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 144));
    __m512i vacc10x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 160));
    __m512i vacc11x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 176));
    __m512i vacc12x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 192));
    __m512i vacc13x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 208));
    __m512i vacc14x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 224));
    __m512i vacc15x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 240));

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

    const __m512 vscale0123456789ABCDEF = _mm512_load_ps((const float*) w + 0);
    w = (const int32_t*) w + 16;

    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled1x0123456789ABCDEF = _mm512_mul_ps(vscaled1x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled2x0123456789ABCDEF = _mm512_mul_ps(vscaled2x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled3x0123456789ABCDEF = _mm512_mul_ps(vscaled3x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled4x0123456789ABCDEF = _mm512_mul_ps(vscaled4x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled5x0123456789ABCDEF = _mm512_mul_ps(vscaled5x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled6x0123456789ABCDEF = _mm512_mul_ps(vscaled6x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled7x0123456789ABCDEF = _mm512_mul_ps(vscaled7x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled8x0123456789ABCDEF = _mm512_mul_ps(vscaled8x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled9x0123456789ABCDEF = _mm512_mul_ps(vscaled9x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled10x0123456789ABCDEF = _mm512_mul_ps(vscaled10x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled11x0123456789ABCDEF = _mm512_mul_ps(vscaled11x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled12x0123456789ABCDEF = _mm512_mul_ps(vscaled12x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled13x0123456789ABCDEF = _mm512_mul_ps(vscaled13x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled14x0123456789ABCDEF = _mm512_mul_ps(vscaled14x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled15x0123456789ABCDEF = _mm512_mul_ps(vscaled15x0123456789ABCDEF, vscale0123456789ABCDEF);

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

    if XNN_LIKELY(nc >= 16) {
      _mm_storeu_si128((__m128i*) (c15 + 0), vout15x0123456789ABCDEF);
      c15 = (int8_t*) ((uintptr_t) c15 + cn_stride);
      _mm_storeu_si128((__m128i*) (c14 + 0), vout14x0123456789ABCDEF);
      c14 = (int8_t*) ((uintptr_t) c14 + cn_stride);
      _mm_storeu_si128((__m128i*) (c13 + 0), vout13x0123456789ABCDEF);
      c13 = (int8_t*) ((uintptr_t) c13 + cn_stride);
      _mm_storeu_si128((__m128i*) (c12 + 0), vout12x0123456789ABCDEF);
      c12 = (int8_t*) ((uintptr_t) c12 + cn_stride);
      _mm_storeu_si128((__m128i*) (c11 + 0), vout11x0123456789ABCDEF);
      c11 = (int8_t*) ((uintptr_t) c11 + cn_stride);
      _mm_storeu_si128((__m128i*) (c10 + 0), vout10x0123456789ABCDEF);
      c10 = (int8_t*) ((uintptr_t) c10 + cn_stride);
      _mm_storeu_si128((__m128i*) (c9 + 0), vout9x0123456789ABCDEF);
      c9 = (int8_t*) ((uintptr_t) c9 + cn_stride);
      _mm_storeu_si128((__m128i*) (c8 + 0), vout8x0123456789ABCDEF);
      c8 = (int8_t*) ((uintptr_t) c8 + cn_stride);
      _mm_storeu_si128((__m128i*) (c7 + 0), vout7x0123456789ABCDEF);
      c7 = (int8_t*) ((uintptr_t) c7 + cn_stride);
      _mm_storeu_si128((__m128i*) (c6 + 0), vout6x0123456789ABCDEF);
      c6 = (int8_t*) ((uintptr_t) c6 + cn_stride);
      _mm_storeu_si128((__m128i*) (c5 + 0), vout5x0123456789ABCDEF);
      c5 = (int8_t*) ((uintptr_t) c5 + cn_stride);
      _mm_storeu_si128((__m128i*) (c4 + 0), vout4x0123456789ABCDEF);
      c4 = (int8_t*) ((uintptr_t) c4 + cn_stride);
      _mm_storeu_si128((__m128i*) (c3 + 0), vout3x0123456789ABCDEF);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      _mm_storeu_si128((__m128i*) (c2 + 0), vout2x0123456789ABCDEF);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      _mm_storeu_si128((__m128i*) (c1 + 0), vout1x0123456789ABCDEF);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      _mm_storeu_si128((__m128i*) (c0 + 0), vout0x0123456789ABCDEF);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      a -= kc;
      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 0) & 0xFFFF));

      _mm_mask_storeu_epi8(c15 + 0, vmask0, vout15x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c14 + 0, vmask0, vout14x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c13 + 0, vmask0, vout13x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c12 + 0, vmask0, vout12x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c11 + 0, vmask0, vout11x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c10 + 0, vmask0, vout10x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c9 + 0, vmask0, vout9x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c8 + 0, vmask0, vout8x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c7 + 0, vmask0, vout7x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c6 + 0, vmask0, vout6x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c5 + 0, vmask0, vout5x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c4 + 0, vmask0, vout4x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c3 + 0, vmask0, vout3x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c2 + 0, vmask0, vout2x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c1 + 0, vmask0, vout1x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c0 + 0, vmask0, vout0x0123456789ABCDEF);
      nc = 0;
    }
  } while (nc != 0);
  // Release tile config
  //  _tile_release();
  __asm__ volatile ("tilerelease" ::);
  #endif  // defined(__x86_64__)
}
