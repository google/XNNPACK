// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c4-avx512amx.c.in
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
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/prefetch.h"


void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_16x32c4__avx512amx_prfm(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params* restrict params)
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
  XNN_ALIGN(64) int32_t res[2][16 * 16];

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  const size_t kremainder = (kc & 63) ? (kc & 63) : 64;

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

  const __m512 voutput_max_less_zero_point = _mm512_set1_ps((int32_t) params->fp32_scalar.output_max - (int32_t) params->fp32_scalar.output_zero_point);
  const __m512i voutput_zero_point = _mm512_set1_epi32(params->fp32_scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8(params->fp32_scalar.output_min);
  // XNN_FORCE_REALIZATION(voutput_max_less_zero_point);
  // XNN_FORCE_REALIZATION(voutput_zero_point);
  // XNN_FORCE_REALIZATION(voutput_min);

  do {
    const __m512i vksum0 = _mm512_load_epi32((const int32_t*) w + 0);
    const __m512i vksum1 = _mm512_load_epi32((const int32_t*) w + 16);
    w = (const int32_t*) w + 32;

    // Zero tile accumulator
    _tile_zero(0);
    _tile_zero(1);

    size_t k = kc;
    while (k >= 64 * sizeof(int8_t)) {
      _tile_loadd(4, a, a_stride);
      a += 64;
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

    if XNN_UNLIKELY(k != 0) {
      _tile_loadd(6, a, a_stride);
      a += kremainder;
      _tile_loadd(7, (const int8_t*) w + 0, 128);
      _tile_dpbssd(0, 6, 7);
      _tile_loadd(7, (const int8_t*) w + 64, 128);
      _tile_dpbssd(1, 6, 7);

      w = (const int8_t*) w + kremainder * 32;
      k -= kremainder * sizeof(int8_t);
    }

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
    __m512i vacc7x0 = _mm512_add_epi32(vksum0, _mm512_load_epi32(&res[0][0] + 112));
    __m512i vacc7x1 = _mm512_add_epi32(vksum1, _mm512_load_epi32(&res[1][0] + 112));
    __m512i vacc8x0 = _mm512_add_epi32(vksum0, _mm512_load_epi32(&res[0][0] + 128));
    __m512i vacc8x1 = _mm512_add_epi32(vksum1, _mm512_load_epi32(&res[1][0] + 128));
    __m512i vacc9x0 = _mm512_add_epi32(vksum0, _mm512_load_epi32(&res[0][0] + 144));
    __m512i vacc9x1 = _mm512_add_epi32(vksum1, _mm512_load_epi32(&res[1][0] + 144));
    __m512i vacc10x0 = _mm512_add_epi32(vksum0, _mm512_load_epi32(&res[0][0] + 160));
    __m512i vacc10x1 = _mm512_add_epi32(vksum1, _mm512_load_epi32(&res[1][0] + 160));
    __m512i vacc11x0 = _mm512_add_epi32(vksum0, _mm512_load_epi32(&res[0][0] + 176));
    __m512i vacc11x1 = _mm512_add_epi32(vksum1, _mm512_load_epi32(&res[1][0] + 176));
    __m512i vacc12x0 = _mm512_add_epi32(vksum0, _mm512_load_epi32(&res[0][0] + 192));
    __m512i vacc12x1 = _mm512_add_epi32(vksum1, _mm512_load_epi32(&res[1][0] + 192));
    __m512i vacc13x0 = _mm512_add_epi32(vksum0, _mm512_load_epi32(&res[0][0] + 208));
    __m512i vacc13x1 = _mm512_add_epi32(vksum1, _mm512_load_epi32(&res[1][0] + 208));
    __m512i vacc14x0 = _mm512_add_epi32(vksum0, _mm512_load_epi32(&res[0][0] + 224));
    __m512i vacc14x1 = _mm512_add_epi32(vksum1, _mm512_load_epi32(&res[1][0] + 224));
    __m512i vacc15x0 = _mm512_add_epi32(vksum0, _mm512_load_epi32(&res[0][0] + 240));
    __m512i vacc15x1 = _mm512_add_epi32(vksum1, _mm512_load_epi32(&res[1][0] + 240));

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
    __m512 vscaled7x0 = _mm512_cvtepi32_ps(vacc7x0);
    __m512 vscaled7x1 = _mm512_cvtepi32_ps(vacc7x1);
    __m512 vscaled8x0 = _mm512_cvtepi32_ps(vacc8x0);
    __m512 vscaled8x1 = _mm512_cvtepi32_ps(vacc8x1);
    __m512 vscaled9x0 = _mm512_cvtepi32_ps(vacc9x0);
    __m512 vscaled9x1 = _mm512_cvtepi32_ps(vacc9x1);
    __m512 vscaled10x0 = _mm512_cvtepi32_ps(vacc10x0);
    __m512 vscaled10x1 = _mm512_cvtepi32_ps(vacc10x1);
    __m512 vscaled11x0 = _mm512_cvtepi32_ps(vacc11x0);
    __m512 vscaled11x1 = _mm512_cvtepi32_ps(vacc11x1);
    __m512 vscaled12x0 = _mm512_cvtepi32_ps(vacc12x0);
    __m512 vscaled12x1 = _mm512_cvtepi32_ps(vacc12x1);
    __m512 vscaled13x0 = _mm512_cvtepi32_ps(vacc13x0);
    __m512 vscaled13x1 = _mm512_cvtepi32_ps(vacc13x1);
    __m512 vscaled14x0 = _mm512_cvtepi32_ps(vacc14x0);
    __m512 vscaled14x1 = _mm512_cvtepi32_ps(vacc14x1);
    __m512 vscaled15x0 = _mm512_cvtepi32_ps(vacc15x0);
    __m512 vscaled15x1 = _mm512_cvtepi32_ps(vacc15x1);

    const __m512 vscale0 = _mm512_load_ps((const float*) w + 0);
    const __m512 vscale1 = _mm512_load_ps((const float*) w + 16);
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
    vscaled7x0 = _mm512_mul_ps(vscaled7x0, vscale0);
    vscaled7x1 = _mm512_mul_ps(vscaled7x1, vscale1);
    vscaled8x0 = _mm512_mul_ps(vscaled8x0, vscale0);
    vscaled8x1 = _mm512_mul_ps(vscaled8x1, vscale1);
    vscaled9x0 = _mm512_mul_ps(vscaled9x0, vscale0);
    vscaled9x1 = _mm512_mul_ps(vscaled9x1, vscale1);
    vscaled10x0 = _mm512_mul_ps(vscaled10x0, vscale0);
    vscaled10x1 = _mm512_mul_ps(vscaled10x1, vscale1);
    vscaled11x0 = _mm512_mul_ps(vscaled11x0, vscale0);
    vscaled11x1 = _mm512_mul_ps(vscaled11x1, vscale1);
    vscaled12x0 = _mm512_mul_ps(vscaled12x0, vscale0);
    vscaled12x1 = _mm512_mul_ps(vscaled12x1, vscale1);
    vscaled13x0 = _mm512_mul_ps(vscaled13x0, vscale0);
    vscaled13x1 = _mm512_mul_ps(vscaled13x1, vscale1);
    vscaled14x0 = _mm512_mul_ps(vscaled14x0, vscale0);
    vscaled14x1 = _mm512_mul_ps(vscaled14x1, vscale1);
    vscaled15x0 = _mm512_mul_ps(vscaled15x0, vscale0);
    vscaled15x1 = _mm512_mul_ps(vscaled15x1, vscale1);

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
    vscaled7x0 = _mm512_min_ps(vscaled7x0, voutput_max_less_zero_point);
    vscaled7x1 = _mm512_min_ps(vscaled7x1, voutput_max_less_zero_point);
    vscaled8x0 = _mm512_min_ps(vscaled8x0, voutput_max_less_zero_point);
    vscaled8x1 = _mm512_min_ps(vscaled8x1, voutput_max_less_zero_point);
    vscaled9x0 = _mm512_min_ps(vscaled9x0, voutput_max_less_zero_point);
    vscaled9x1 = _mm512_min_ps(vscaled9x1, voutput_max_less_zero_point);
    vscaled10x0 = _mm512_min_ps(vscaled10x0, voutput_max_less_zero_point);
    vscaled10x1 = _mm512_min_ps(vscaled10x1, voutput_max_less_zero_point);
    vscaled11x0 = _mm512_min_ps(vscaled11x0, voutput_max_less_zero_point);
    vscaled11x1 = _mm512_min_ps(vscaled11x1, voutput_max_less_zero_point);
    vscaled12x0 = _mm512_min_ps(vscaled12x0, voutput_max_less_zero_point);
    vscaled12x1 = _mm512_min_ps(vscaled12x1, voutput_max_less_zero_point);
    vscaled13x0 = _mm512_min_ps(vscaled13x0, voutput_max_less_zero_point);
    vscaled13x1 = _mm512_min_ps(vscaled13x1, voutput_max_less_zero_point);
    vscaled14x0 = _mm512_min_ps(vscaled14x0, voutput_max_less_zero_point);
    vscaled14x1 = _mm512_min_ps(vscaled14x1, voutput_max_less_zero_point);
    vscaled15x0 = _mm512_min_ps(vscaled15x0, voutput_max_less_zero_point);
    vscaled15x1 = _mm512_min_ps(vscaled15x1, voutput_max_less_zero_point);

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
    vacc7x0 = _mm512_cvtps_epi32(vscaled7x0);
    vacc7x1 = _mm512_cvtps_epi32(vscaled7x1);
    vacc8x0 = _mm512_cvtps_epi32(vscaled8x0);
    vacc8x1 = _mm512_cvtps_epi32(vscaled8x1);
    vacc9x0 = _mm512_cvtps_epi32(vscaled9x0);
    vacc9x1 = _mm512_cvtps_epi32(vscaled9x1);
    vacc10x0 = _mm512_cvtps_epi32(vscaled10x0);
    vacc10x1 = _mm512_cvtps_epi32(vscaled10x1);
    vacc11x0 = _mm512_cvtps_epi32(vscaled11x0);
    vacc11x1 = _mm512_cvtps_epi32(vscaled11x1);
    vacc12x0 = _mm512_cvtps_epi32(vscaled12x0);
    vacc12x1 = _mm512_cvtps_epi32(vscaled12x1);
    vacc13x0 = _mm512_cvtps_epi32(vscaled13x0);
    vacc13x1 = _mm512_cvtps_epi32(vscaled13x1);
    vacc14x0 = _mm512_cvtps_epi32(vscaled14x0);
    vacc14x1 = _mm512_cvtps_epi32(vscaled14x1);
    vacc15x0 = _mm512_cvtps_epi32(vscaled15x0);
    vacc15x1 = _mm512_cvtps_epi32(vscaled15x1);

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
    vacc7x0 = _mm512_add_epi32(vacc7x0, voutput_zero_point);
    vacc7x1 = _mm512_add_epi32(vacc7x1, voutput_zero_point);
    vacc8x0 = _mm512_add_epi32(vacc8x0, voutput_zero_point);
    vacc8x1 = _mm512_add_epi32(vacc8x1, voutput_zero_point);
    vacc9x0 = _mm512_add_epi32(vacc9x0, voutput_zero_point);
    vacc9x1 = _mm512_add_epi32(vacc9x1, voutput_zero_point);
    vacc10x0 = _mm512_add_epi32(vacc10x0, voutput_zero_point);
    vacc10x1 = _mm512_add_epi32(vacc10x1, voutput_zero_point);
    vacc11x0 = _mm512_add_epi32(vacc11x0, voutput_zero_point);
    vacc11x1 = _mm512_add_epi32(vacc11x1, voutput_zero_point);
    vacc12x0 = _mm512_add_epi32(vacc12x0, voutput_zero_point);
    vacc12x1 = _mm512_add_epi32(vacc12x1, voutput_zero_point);
    vacc13x0 = _mm512_add_epi32(vacc13x0, voutput_zero_point);
    vacc13x1 = _mm512_add_epi32(vacc13x1, voutput_zero_point);
    vacc14x0 = _mm512_add_epi32(vacc14x0, voutput_zero_point);
    vacc14x1 = _mm512_add_epi32(vacc14x1, voutput_zero_point);
    vacc15x0 = _mm512_add_epi32(vacc15x0, voutput_zero_point);
    vacc15x1 = _mm512_add_epi32(vacc15x1, voutput_zero_point);

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
    __m128i vout7x0 = _mm512_cvtsepi32_epi8(vacc7x0);
    __m128i vout7x1 = _mm512_cvtsepi32_epi8(vacc7x1);
    __m128i vout8x0 = _mm512_cvtsepi32_epi8(vacc8x0);
    __m128i vout8x1 = _mm512_cvtsepi32_epi8(vacc8x1);
    __m128i vout9x0 = _mm512_cvtsepi32_epi8(vacc9x0);
    __m128i vout9x1 = _mm512_cvtsepi32_epi8(vacc9x1);
    __m128i vout10x0 = _mm512_cvtsepi32_epi8(vacc10x0);
    __m128i vout10x1 = _mm512_cvtsepi32_epi8(vacc10x1);
    __m128i vout11x0 = _mm512_cvtsepi32_epi8(vacc11x0);
    __m128i vout11x1 = _mm512_cvtsepi32_epi8(vacc11x1);
    __m128i vout12x0 = _mm512_cvtsepi32_epi8(vacc12x0);
    __m128i vout12x1 = _mm512_cvtsepi32_epi8(vacc12x1);
    __m128i vout13x0 = _mm512_cvtsepi32_epi8(vacc13x0);
    __m128i vout13x1 = _mm512_cvtsepi32_epi8(vacc13x1);
    __m128i vout14x0 = _mm512_cvtsepi32_epi8(vacc14x0);
    __m128i vout14x1 = _mm512_cvtsepi32_epi8(vacc14x1);
    __m128i vout15x0 = _mm512_cvtsepi32_epi8(vacc15x0);
    __m128i vout15x1 = _mm512_cvtsepi32_epi8(vacc15x1);

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
    vout7x0 = _mm_max_epi8(vout7x0, voutput_min);
    vout7x1 = _mm_max_epi8(vout7x1, voutput_min);
    vout8x0 = _mm_max_epi8(vout8x0, voutput_min);
    vout8x1 = _mm_max_epi8(vout8x1, voutput_min);
    vout9x0 = _mm_max_epi8(vout9x0, voutput_min);
    vout9x1 = _mm_max_epi8(vout9x1, voutput_min);
    vout10x0 = _mm_max_epi8(vout10x0, voutput_min);
    vout10x1 = _mm_max_epi8(vout10x1, voutput_min);
    vout11x0 = _mm_max_epi8(vout11x0, voutput_min);
    vout11x1 = _mm_max_epi8(vout11x1, voutput_min);
    vout12x0 = _mm_max_epi8(vout12x0, voutput_min);
    vout12x1 = _mm_max_epi8(vout12x1, voutput_min);
    vout13x0 = _mm_max_epi8(vout13x0, voutput_min);
    vout13x1 = _mm_max_epi8(vout13x1, voutput_min);
    vout14x0 = _mm_max_epi8(vout14x0, voutput_min);
    vout14x1 = _mm_max_epi8(vout14x1, voutput_min);
    vout15x0 = _mm_max_epi8(vout15x0, voutput_min);
    vout15x1 = _mm_max_epi8(vout15x1, voutput_min);

    if XNN_LIKELY(nc >= 32) {
      _mm_storeu_si128((__m128i*) (c15 + 0), vout15x0);
      _mm_storeu_si128((__m128i*) (c15 + 16), vout15x1);
      c15 = (int8_t*) ((uintptr_t) c15 + cn_stride);
      _mm_storeu_si128((__m128i*) (c14 + 0), vout14x0);
      _mm_storeu_si128((__m128i*) (c14 + 16), vout14x1);
      c14 = (int8_t*) ((uintptr_t) c14 + cn_stride);
      _mm_storeu_si128((__m128i*) (c13 + 0), vout13x0);
      _mm_storeu_si128((__m128i*) (c13 + 16), vout13x1);
      c13 = (int8_t*) ((uintptr_t) c13 + cn_stride);
      _mm_storeu_si128((__m128i*) (c12 + 0), vout12x0);
      _mm_storeu_si128((__m128i*) (c12 + 16), vout12x1);
      c12 = (int8_t*) ((uintptr_t) c12 + cn_stride);
      _mm_storeu_si128((__m128i*) (c11 + 0), vout11x0);
      _mm_storeu_si128((__m128i*) (c11 + 16), vout11x1);
      c11 = (int8_t*) ((uintptr_t) c11 + cn_stride);
      _mm_storeu_si128((__m128i*) (c10 + 0), vout10x0);
      _mm_storeu_si128((__m128i*) (c10 + 16), vout10x1);
      c10 = (int8_t*) ((uintptr_t) c10 + cn_stride);
      _mm_storeu_si128((__m128i*) (c9 + 0), vout9x0);
      _mm_storeu_si128((__m128i*) (c9 + 16), vout9x1);
      c9 = (int8_t*) ((uintptr_t) c9 + cn_stride);
      _mm_storeu_si128((__m128i*) (c8 + 0), vout8x0);
      _mm_storeu_si128((__m128i*) (c8 + 16), vout8x1);
      c8 = (int8_t*) ((uintptr_t) c8 + cn_stride);
      _mm_storeu_si128((__m128i*) (c7 + 0), vout7x0);
      _mm_storeu_si128((__m128i*) (c7 + 16), vout7x1);
      c7 = (int8_t*) ((uintptr_t) c7 + cn_stride);
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
      a -= kc;
      nc -= 32;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 0) & 0xFFFF));
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 16) & 0xFFFF));

      _mm_mask_storeu_epi8(c15 + 0, vmask0, vout15x0);
      _mm_mask_storeu_epi8(c15 + 16, vmask1, vout15x1);
      _mm_mask_storeu_epi8(c14 + 0, vmask0, vout14x0);
      _mm_mask_storeu_epi8(c14 + 16, vmask1, vout14x1);
      _mm_mask_storeu_epi8(c13 + 0, vmask0, vout13x0);
      _mm_mask_storeu_epi8(c13 + 16, vmask1, vout13x1);
      _mm_mask_storeu_epi8(c12 + 0, vmask0, vout12x0);
      _mm_mask_storeu_epi8(c12 + 16, vmask1, vout12x1);
      _mm_mask_storeu_epi8(c11 + 0, vmask0, vout11x0);
      _mm_mask_storeu_epi8(c11 + 16, vmask1, vout11x1);
      _mm_mask_storeu_epi8(c10 + 0, vmask0, vout10x0);
      _mm_mask_storeu_epi8(c10 + 16, vmask1, vout10x1);
      _mm_mask_storeu_epi8(c9 + 0, vmask0, vout9x0);
      _mm_mask_storeu_epi8(c9 + 16, vmask1, vout9x1);
      _mm_mask_storeu_epi8(c8 + 0, vmask0, vout8x0);
      _mm_mask_storeu_epi8(c8 + 16, vmask1, vout8x1);
      _mm_mask_storeu_epi8(c7 + 0, vmask0, vout7x0);
      _mm_mask_storeu_epi8(c7 + 16, vmask1, vout7x1);
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
