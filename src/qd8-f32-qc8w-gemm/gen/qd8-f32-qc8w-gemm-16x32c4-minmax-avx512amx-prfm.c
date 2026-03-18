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


void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_16x32c4__avx512amx_prfm(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* restrict params,
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
  // XNN_FORCE_REALIZATION(voutput_min);
  // XNN_FORCE_REALIZATION(voutput_max);

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
    __m512i vacc0x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[0].zero_point));
    __m512i vacc0x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[0].zero_point));
    __m512i vacc1x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[1].zero_point));
    __m512i vacc1x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[1].zero_point));
    __m512i vacc2x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[2].zero_point));
    __m512i vacc2x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[2].zero_point));
    __m512i vacc3x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[3].zero_point));
    __m512i vacc3x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[3].zero_point));
    __m512i vacc4x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[4].zero_point));
    __m512i vacc4x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[4].zero_point));
    __m512i vacc5x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[5].zero_point));
    __m512i vacc5x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[5].zero_point));
    __m512i vacc6x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[6].zero_point));
    __m512i vacc6x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[6].zero_point));
    __m512i vacc7x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[7].zero_point));
    __m512i vacc7x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[7].zero_point));
    __m512i vacc8x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[8].zero_point));
    __m512i vacc8x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[8].zero_point));
    __m512i vacc9x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[9].zero_point));
    __m512i vacc9x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[9].zero_point));
    __m512i vacc10x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[10].zero_point));
    __m512i vacc10x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[10].zero_point));
    __m512i vacc11x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[11].zero_point));
    __m512i vacc11x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[11].zero_point));
    __m512i vacc12x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[12].zero_point));
    __m512i vacc12x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[12].zero_point));
    __m512i vacc13x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[13].zero_point));
    __m512i vacc13x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[13].zero_point));
    __m512i vacc14x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[14].zero_point));
    __m512i vacc14x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[14].zero_point));
    __m512i vacc15x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[15].zero_point));
    __m512i vacc15x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[15].zero_point));
    // Add tile to bias
    vacc0x0 = _mm512_add_epi32(vacc0x0, _mm512_load_epi32(&res[0][0] + 0));
    vacc0x1 = _mm512_add_epi32(vacc0x1, _mm512_load_epi32(&res[1][0] + 0));
    vacc1x0 = _mm512_add_epi32(vacc1x0, _mm512_load_epi32(&res[0][0] + 16));
    vacc1x1 = _mm512_add_epi32(vacc1x1, _mm512_load_epi32(&res[1][0] + 16));
    vacc2x0 = _mm512_add_epi32(vacc2x0, _mm512_load_epi32(&res[0][0] + 32));
    vacc2x1 = _mm512_add_epi32(vacc2x1, _mm512_load_epi32(&res[1][0] + 32));
    vacc3x0 = _mm512_add_epi32(vacc3x0, _mm512_load_epi32(&res[0][0] + 48));
    vacc3x1 = _mm512_add_epi32(vacc3x1, _mm512_load_epi32(&res[1][0] + 48));
    vacc4x0 = _mm512_add_epi32(vacc4x0, _mm512_load_epi32(&res[0][0] + 64));
    vacc4x1 = _mm512_add_epi32(vacc4x1, _mm512_load_epi32(&res[1][0] + 64));
    vacc5x0 = _mm512_add_epi32(vacc5x0, _mm512_load_epi32(&res[0][0] + 80));
    vacc5x1 = _mm512_add_epi32(vacc5x1, _mm512_load_epi32(&res[1][0] + 80));
    vacc6x0 = _mm512_add_epi32(vacc6x0, _mm512_load_epi32(&res[0][0] + 96));
    vacc6x1 = _mm512_add_epi32(vacc6x1, _mm512_load_epi32(&res[1][0] + 96));
    vacc7x0 = _mm512_add_epi32(vacc7x0, _mm512_load_epi32(&res[0][0] + 112));
    vacc7x1 = _mm512_add_epi32(vacc7x1, _mm512_load_epi32(&res[1][0] + 112));
    vacc8x0 = _mm512_add_epi32(vacc8x0, _mm512_load_epi32(&res[0][0] + 128));
    vacc8x1 = _mm512_add_epi32(vacc8x1, _mm512_load_epi32(&res[1][0] + 128));
    vacc9x0 = _mm512_add_epi32(vacc9x0, _mm512_load_epi32(&res[0][0] + 144));
    vacc9x1 = _mm512_add_epi32(vacc9x1, _mm512_load_epi32(&res[1][0] + 144));
    vacc10x0 = _mm512_add_epi32(vacc10x0, _mm512_load_epi32(&res[0][0] + 160));
    vacc10x1 = _mm512_add_epi32(vacc10x1, _mm512_load_epi32(&res[1][0] + 160));
    vacc11x0 = _mm512_add_epi32(vacc11x0, _mm512_load_epi32(&res[0][0] + 176));
    vacc11x1 = _mm512_add_epi32(vacc11x1, _mm512_load_epi32(&res[1][0] + 176));
    vacc12x0 = _mm512_add_epi32(vacc12x0, _mm512_load_epi32(&res[0][0] + 192));
    vacc12x1 = _mm512_add_epi32(vacc12x1, _mm512_load_epi32(&res[1][0] + 192));
    vacc13x0 = _mm512_add_epi32(vacc13x0, _mm512_load_epi32(&res[0][0] + 208));
    vacc13x1 = _mm512_add_epi32(vacc13x1, _mm512_load_epi32(&res[1][0] + 208));
    vacc14x0 = _mm512_add_epi32(vacc14x0, _mm512_load_epi32(&res[0][0] + 224));
    vacc14x1 = _mm512_add_epi32(vacc14x1, _mm512_load_epi32(&res[1][0] + 224));
    vacc15x0 = _mm512_add_epi32(vacc15x0, _mm512_load_epi32(&res[0][0] + 240));
    vacc15x1 = _mm512_add_epi32(vacc15x1, _mm512_load_epi32(&res[1][0] + 240));

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

    vscaled0x0 = _mm512_mul_ps(vscaled0x0, _mm512_set1_ps(quantization_params[0].inv_scale));
    vscaled0x1 = _mm512_mul_ps(vscaled0x1, _mm512_set1_ps(quantization_params[0].inv_scale));
    vscaled1x0 = _mm512_mul_ps(vscaled1x0, _mm512_set1_ps(quantization_params[1].inv_scale));
    vscaled1x1 = _mm512_mul_ps(vscaled1x1, _mm512_set1_ps(quantization_params[1].inv_scale));
    vscaled2x0 = _mm512_mul_ps(vscaled2x0, _mm512_set1_ps(quantization_params[2].inv_scale));
    vscaled2x1 = _mm512_mul_ps(vscaled2x1, _mm512_set1_ps(quantization_params[2].inv_scale));
    vscaled3x0 = _mm512_mul_ps(vscaled3x0, _mm512_set1_ps(quantization_params[3].inv_scale));
    vscaled3x1 = _mm512_mul_ps(vscaled3x1, _mm512_set1_ps(quantization_params[3].inv_scale));
    vscaled4x0 = _mm512_mul_ps(vscaled4x0, _mm512_set1_ps(quantization_params[4].inv_scale));
    vscaled4x1 = _mm512_mul_ps(vscaled4x1, _mm512_set1_ps(quantization_params[4].inv_scale));
    vscaled5x0 = _mm512_mul_ps(vscaled5x0, _mm512_set1_ps(quantization_params[5].inv_scale));
    vscaled5x1 = _mm512_mul_ps(vscaled5x1, _mm512_set1_ps(quantization_params[5].inv_scale));
    vscaled6x0 = _mm512_mul_ps(vscaled6x0, _mm512_set1_ps(quantization_params[6].inv_scale));
    vscaled6x1 = _mm512_mul_ps(vscaled6x1, _mm512_set1_ps(quantization_params[6].inv_scale));
    vscaled7x0 = _mm512_mul_ps(vscaled7x0, _mm512_set1_ps(quantization_params[7].inv_scale));
    vscaled7x1 = _mm512_mul_ps(vscaled7x1, _mm512_set1_ps(quantization_params[7].inv_scale));
    vscaled8x0 = _mm512_mul_ps(vscaled8x0, _mm512_set1_ps(quantization_params[8].inv_scale));
    vscaled8x1 = _mm512_mul_ps(vscaled8x1, _mm512_set1_ps(quantization_params[8].inv_scale));
    vscaled9x0 = _mm512_mul_ps(vscaled9x0, _mm512_set1_ps(quantization_params[9].inv_scale));
    vscaled9x1 = _mm512_mul_ps(vscaled9x1, _mm512_set1_ps(quantization_params[9].inv_scale));
    vscaled10x0 = _mm512_mul_ps(vscaled10x0, _mm512_set1_ps(quantization_params[10].inv_scale));
    vscaled10x1 = _mm512_mul_ps(vscaled10x1, _mm512_set1_ps(quantization_params[10].inv_scale));
    vscaled11x0 = _mm512_mul_ps(vscaled11x0, _mm512_set1_ps(quantization_params[11].inv_scale));
    vscaled11x1 = _mm512_mul_ps(vscaled11x1, _mm512_set1_ps(quantization_params[11].inv_scale));
    vscaled12x0 = _mm512_mul_ps(vscaled12x0, _mm512_set1_ps(quantization_params[12].inv_scale));
    vscaled12x1 = _mm512_mul_ps(vscaled12x1, _mm512_set1_ps(quantization_params[12].inv_scale));
    vscaled13x0 = _mm512_mul_ps(vscaled13x0, _mm512_set1_ps(quantization_params[13].inv_scale));
    vscaled13x1 = _mm512_mul_ps(vscaled13x1, _mm512_set1_ps(quantization_params[13].inv_scale));
    vscaled14x0 = _mm512_mul_ps(vscaled14x0, _mm512_set1_ps(quantization_params[14].inv_scale));
    vscaled14x1 = _mm512_mul_ps(vscaled14x1, _mm512_set1_ps(quantization_params[14].inv_scale));
    vscaled15x0 = _mm512_mul_ps(vscaled15x0, _mm512_set1_ps(quantization_params[15].inv_scale));
    vscaled15x1 = _mm512_mul_ps(vscaled15x1, _mm512_set1_ps(quantization_params[15].inv_scale));

    const __m512 vfilter_output_scale0 = _mm512_load_ps((const float*) w + 0);
    const __m512 vfilter_output_scale1 = _mm512_load_ps((const float*) w + 16);
    w = (const int32_t*) w + 32;
    const __m512 vbias0 = _mm512_load_ps((const float*) w + 0);
    const __m512 vbias1 = _mm512_load_ps((const float*) w + 16);
    w = (const int32_t*) w + 32;

    vscaled0x0 = _mm512_fmadd_ps(vscaled0x0, vfilter_output_scale0, vbias0);
    vscaled0x1 = _mm512_fmadd_ps(vscaled0x1, vfilter_output_scale1, vbias1);
    vscaled1x0 = _mm512_fmadd_ps(vscaled1x0, vfilter_output_scale0, vbias0);
    vscaled1x1 = _mm512_fmadd_ps(vscaled1x1, vfilter_output_scale1, vbias1);
    vscaled2x0 = _mm512_fmadd_ps(vscaled2x0, vfilter_output_scale0, vbias0);
    vscaled2x1 = _mm512_fmadd_ps(vscaled2x1, vfilter_output_scale1, vbias1);
    vscaled3x0 = _mm512_fmadd_ps(vscaled3x0, vfilter_output_scale0, vbias0);
    vscaled3x1 = _mm512_fmadd_ps(vscaled3x1, vfilter_output_scale1, vbias1);
    vscaled4x0 = _mm512_fmadd_ps(vscaled4x0, vfilter_output_scale0, vbias0);
    vscaled4x1 = _mm512_fmadd_ps(vscaled4x1, vfilter_output_scale1, vbias1);
    vscaled5x0 = _mm512_fmadd_ps(vscaled5x0, vfilter_output_scale0, vbias0);
    vscaled5x1 = _mm512_fmadd_ps(vscaled5x1, vfilter_output_scale1, vbias1);
    vscaled6x0 = _mm512_fmadd_ps(vscaled6x0, vfilter_output_scale0, vbias0);
    vscaled6x1 = _mm512_fmadd_ps(vscaled6x1, vfilter_output_scale1, vbias1);
    vscaled7x0 = _mm512_fmadd_ps(vscaled7x0, vfilter_output_scale0, vbias0);
    vscaled7x1 = _mm512_fmadd_ps(vscaled7x1, vfilter_output_scale1, vbias1);
    vscaled8x0 = _mm512_fmadd_ps(vscaled8x0, vfilter_output_scale0, vbias0);
    vscaled8x1 = _mm512_fmadd_ps(vscaled8x1, vfilter_output_scale1, vbias1);
    vscaled9x0 = _mm512_fmadd_ps(vscaled9x0, vfilter_output_scale0, vbias0);
    vscaled9x1 = _mm512_fmadd_ps(vscaled9x1, vfilter_output_scale1, vbias1);
    vscaled10x0 = _mm512_fmadd_ps(vscaled10x0, vfilter_output_scale0, vbias0);
    vscaled10x1 = _mm512_fmadd_ps(vscaled10x1, vfilter_output_scale1, vbias1);
    vscaled11x0 = _mm512_fmadd_ps(vscaled11x0, vfilter_output_scale0, vbias0);
    vscaled11x1 = _mm512_fmadd_ps(vscaled11x1, vfilter_output_scale1, vbias1);
    vscaled12x0 = _mm512_fmadd_ps(vscaled12x0, vfilter_output_scale0, vbias0);
    vscaled12x1 = _mm512_fmadd_ps(vscaled12x1, vfilter_output_scale1, vbias1);
    vscaled13x0 = _mm512_fmadd_ps(vscaled13x0, vfilter_output_scale0, vbias0);
    vscaled13x1 = _mm512_fmadd_ps(vscaled13x1, vfilter_output_scale1, vbias1);
    vscaled14x0 = _mm512_fmadd_ps(vscaled14x0, vfilter_output_scale0, vbias0);
    vscaled14x1 = _mm512_fmadd_ps(vscaled14x1, vfilter_output_scale1, vbias1);
    vscaled15x0 = _mm512_fmadd_ps(vscaled15x0, vfilter_output_scale0, vbias0);
    vscaled15x1 = _mm512_fmadd_ps(vscaled15x1, vfilter_output_scale1, vbias1);

    vscaled0x0 = _mm512_max_ps(vscaled0x0, voutput_min);
    vscaled0x1 = _mm512_max_ps(vscaled0x1, voutput_min);
    vscaled1x0 = _mm512_max_ps(vscaled1x0, voutput_min);
    vscaled1x1 = _mm512_max_ps(vscaled1x1, voutput_min);
    vscaled2x0 = _mm512_max_ps(vscaled2x0, voutput_min);
    vscaled2x1 = _mm512_max_ps(vscaled2x1, voutput_min);
    vscaled3x0 = _mm512_max_ps(vscaled3x0, voutput_min);
    vscaled3x1 = _mm512_max_ps(vscaled3x1, voutput_min);
    vscaled4x0 = _mm512_max_ps(vscaled4x0, voutput_min);
    vscaled4x1 = _mm512_max_ps(vscaled4x1, voutput_min);
    vscaled5x0 = _mm512_max_ps(vscaled5x0, voutput_min);
    vscaled5x1 = _mm512_max_ps(vscaled5x1, voutput_min);
    vscaled6x0 = _mm512_max_ps(vscaled6x0, voutput_min);
    vscaled6x1 = _mm512_max_ps(vscaled6x1, voutput_min);
    vscaled7x0 = _mm512_max_ps(vscaled7x0, voutput_min);
    vscaled7x1 = _mm512_max_ps(vscaled7x1, voutput_min);
    vscaled8x0 = _mm512_max_ps(vscaled8x0, voutput_min);
    vscaled8x1 = _mm512_max_ps(vscaled8x1, voutput_min);
    vscaled9x0 = _mm512_max_ps(vscaled9x0, voutput_min);
    vscaled9x1 = _mm512_max_ps(vscaled9x1, voutput_min);
    vscaled10x0 = _mm512_max_ps(vscaled10x0, voutput_min);
    vscaled10x1 = _mm512_max_ps(vscaled10x1, voutput_min);
    vscaled11x0 = _mm512_max_ps(vscaled11x0, voutput_min);
    vscaled11x1 = _mm512_max_ps(vscaled11x1, voutput_min);
    vscaled12x0 = _mm512_max_ps(vscaled12x0, voutput_min);
    vscaled12x1 = _mm512_max_ps(vscaled12x1, voutput_min);
    vscaled13x0 = _mm512_max_ps(vscaled13x0, voutput_min);
    vscaled13x1 = _mm512_max_ps(vscaled13x1, voutput_min);
    vscaled14x0 = _mm512_max_ps(vscaled14x0, voutput_min);
    vscaled14x1 = _mm512_max_ps(vscaled14x1, voutput_min);
    vscaled15x0 = _mm512_max_ps(vscaled15x0, voutput_min);
    vscaled15x1 = _mm512_max_ps(vscaled15x1, voutput_min);

    vscaled0x0 = _mm512_min_ps(vscaled0x0, voutput_max);
    vscaled0x1 = _mm512_min_ps(vscaled0x1, voutput_max);
    vscaled1x0 = _mm512_min_ps(vscaled1x0, voutput_max);
    vscaled1x1 = _mm512_min_ps(vscaled1x1, voutput_max);
    vscaled2x0 = _mm512_min_ps(vscaled2x0, voutput_max);
    vscaled2x1 = _mm512_min_ps(vscaled2x1, voutput_max);
    vscaled3x0 = _mm512_min_ps(vscaled3x0, voutput_max);
    vscaled3x1 = _mm512_min_ps(vscaled3x1, voutput_max);
    vscaled4x0 = _mm512_min_ps(vscaled4x0, voutput_max);
    vscaled4x1 = _mm512_min_ps(vscaled4x1, voutput_max);
    vscaled5x0 = _mm512_min_ps(vscaled5x0, voutput_max);
    vscaled5x1 = _mm512_min_ps(vscaled5x1, voutput_max);
    vscaled6x0 = _mm512_min_ps(vscaled6x0, voutput_max);
    vscaled6x1 = _mm512_min_ps(vscaled6x1, voutput_max);
    vscaled7x0 = _mm512_min_ps(vscaled7x0, voutput_max);
    vscaled7x1 = _mm512_min_ps(vscaled7x1, voutput_max);
    vscaled8x0 = _mm512_min_ps(vscaled8x0, voutput_max);
    vscaled8x1 = _mm512_min_ps(vscaled8x1, voutput_max);
    vscaled9x0 = _mm512_min_ps(vscaled9x0, voutput_max);
    vscaled9x1 = _mm512_min_ps(vscaled9x1, voutput_max);
    vscaled10x0 = _mm512_min_ps(vscaled10x0, voutput_max);
    vscaled10x1 = _mm512_min_ps(vscaled10x1, voutput_max);
    vscaled11x0 = _mm512_min_ps(vscaled11x0, voutput_max);
    vscaled11x1 = _mm512_min_ps(vscaled11x1, voutput_max);
    vscaled12x0 = _mm512_min_ps(vscaled12x0, voutput_max);
    vscaled12x1 = _mm512_min_ps(vscaled12x1, voutput_max);
    vscaled13x0 = _mm512_min_ps(vscaled13x0, voutput_max);
    vscaled13x1 = _mm512_min_ps(vscaled13x1, voutput_max);
    vscaled14x0 = _mm512_min_ps(vscaled14x0, voutput_max);
    vscaled14x1 = _mm512_min_ps(vscaled14x1, voutput_max);
    vscaled15x0 = _mm512_min_ps(vscaled15x0, voutput_max);
    vscaled15x1 = _mm512_min_ps(vscaled15x1, voutput_max);

    if XNN_LIKELY(nc >= 32) {
      _mm512_storeu_ps(c15 + 0, vscaled15x0);
      _mm512_storeu_ps(c15 + 16, vscaled15x1);
      c15 = (float*) ((uintptr_t) c15 + cn_stride);
      _mm512_storeu_ps(c14 + 0, vscaled14x0);
      _mm512_storeu_ps(c14 + 16, vscaled14x1);
      c14 = (float*) ((uintptr_t) c14 + cn_stride);
      _mm512_storeu_ps(c13 + 0, vscaled13x0);
      _mm512_storeu_ps(c13 + 16, vscaled13x1);
      c13 = (float*) ((uintptr_t) c13 + cn_stride);
      _mm512_storeu_ps(c12 + 0, vscaled12x0);
      _mm512_storeu_ps(c12 + 16, vscaled12x1);
      c12 = (float*) ((uintptr_t) c12 + cn_stride);
      _mm512_storeu_ps(c11 + 0, vscaled11x0);
      _mm512_storeu_ps(c11 + 16, vscaled11x1);
      c11 = (float*) ((uintptr_t) c11 + cn_stride);
      _mm512_storeu_ps(c10 + 0, vscaled10x0);
      _mm512_storeu_ps(c10 + 16, vscaled10x1);
      c10 = (float*) ((uintptr_t) c10 + cn_stride);
      _mm512_storeu_ps(c9 + 0, vscaled9x0);
      _mm512_storeu_ps(c9 + 16, vscaled9x1);
      c9 = (float*) ((uintptr_t) c9 + cn_stride);
      _mm512_storeu_ps(c8 + 0, vscaled8x0);
      _mm512_storeu_ps(c8 + 16, vscaled8x1);
      c8 = (float*) ((uintptr_t) c8 + cn_stride);
      _mm512_storeu_ps(c7 + 0, vscaled7x0);
      _mm512_storeu_ps(c7 + 16, vscaled7x1);
      c7 = (float*) ((uintptr_t) c7 + cn_stride);
      _mm512_storeu_ps(c6 + 0, vscaled6x0);
      _mm512_storeu_ps(c6 + 16, vscaled6x1);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      _mm512_storeu_ps(c5 + 0, vscaled5x0);
      _mm512_storeu_ps(c5 + 16, vscaled5x1);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      _mm512_storeu_ps(c4 + 0, vscaled4x0);
      _mm512_storeu_ps(c4 + 16, vscaled4x1);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm512_storeu_ps(c3 + 0, vscaled3x0);
      _mm512_storeu_ps(c3 + 16, vscaled3x1);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm512_storeu_ps(c2 + 0, vscaled2x0);
      _mm512_storeu_ps(c2 + 16, vscaled2x1);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm512_storeu_ps(c1 + 0, vscaled1x0);
      _mm512_storeu_ps(c1 + 16, vscaled1x1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm512_storeu_ps(c0 + 0, vscaled0x0);
      _mm512_storeu_ps(c0 + 16, vscaled0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a -= kc;
      nc -= 32;
    } else {
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 0) & 0xFFFF));
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 16) & 0xFFFF));
      _mm512_mask_storeu_ps(c15 + 0, vmask0, vscaled15x0);
      _mm512_mask_storeu_ps(c15 + 16, vmask1, vscaled15x1);
      _mm512_mask_storeu_ps(c14 + 0, vmask0, vscaled14x0);
      _mm512_mask_storeu_ps(c14 + 16, vmask1, vscaled14x1);
      _mm512_mask_storeu_ps(c13 + 0, vmask0, vscaled13x0);
      _mm512_mask_storeu_ps(c13 + 16, vmask1, vscaled13x1);
      _mm512_mask_storeu_ps(c12 + 0, vmask0, vscaled12x0);
      _mm512_mask_storeu_ps(c12 + 16, vmask1, vscaled12x1);
      _mm512_mask_storeu_ps(c11 + 0, vmask0, vscaled11x0);
      _mm512_mask_storeu_ps(c11 + 16, vmask1, vscaled11x1);
      _mm512_mask_storeu_ps(c10 + 0, vmask0, vscaled10x0);
      _mm512_mask_storeu_ps(c10 + 16, vmask1, vscaled10x1);
      _mm512_mask_storeu_ps(c9 + 0, vmask0, vscaled9x0);
      _mm512_mask_storeu_ps(c9 + 16, vmask1, vscaled9x1);
      _mm512_mask_storeu_ps(c8 + 0, vmask0, vscaled8x0);
      _mm512_mask_storeu_ps(c8 + 16, vmask1, vscaled8x1);
      _mm512_mask_storeu_ps(c7 + 0, vmask0, vscaled7x0);
      _mm512_mask_storeu_ps(c7 + 16, vmask1, vscaled7x1);
      _mm512_mask_storeu_ps(c6 + 0, vmask0, vscaled6x0);
      _mm512_mask_storeu_ps(c6 + 16, vmask1, vscaled6x1);
      _mm512_mask_storeu_ps(c5 + 0, vmask0, vscaled5x0);
      _mm512_mask_storeu_ps(c5 + 16, vmask1, vscaled5x1);
      _mm512_mask_storeu_ps(c4 + 0, vmask0, vscaled4x0);
      _mm512_mask_storeu_ps(c4 + 16, vmask1, vscaled4x1);
      _mm512_mask_storeu_ps(c3 + 0, vmask0, vscaled3x0);
      _mm512_mask_storeu_ps(c3 + 16, vmask1, vscaled3x1);
      _mm512_mask_storeu_ps(c2 + 0, vmask0, vscaled2x0);
      _mm512_mask_storeu_ps(c2 + 16, vmask1, vscaled2x1);
      _mm512_mask_storeu_ps(c1 + 0, vmask0, vscaled1x0);
      _mm512_mask_storeu_ps(c1 + 16, vmask1, vscaled1x1);
      _mm512_mask_storeu_ps(c0 + 0, vmask0, vscaled0x0);
      _mm512_mask_storeu_ps(c0 + 16, vmask1, vscaled0x1);
      nc = 0;
    }
  } while (nc != 0);
  // Release tile config
  _tile_release();
  #endif  // defined(__x86_64__)
}
