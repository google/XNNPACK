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


void xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x64c4__avx512amx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    xnn_float16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f16_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params)
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
  XNN_ALIGN(64) int32_t res[4][7 * 16];

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

  const __m512 voutput_min = _mm512_cvtph_ps(_mm256_set1_epi16(*(const uint16_t*) &params->scalar.min));
  const __m512 voutput_max = _mm512_cvtph_ps(_mm256_set1_epi16(*(const uint16_t*) &params->scalar.max));
  // XNN_FORCE_REALIZATION(voutput_min);
  // XNN_FORCE_REALIZATION(voutput_max);

  do {
    const __m512i vksum0 = _mm512_load_epi32((const int32_t*) w + 0);
    const __m512i vksum1 = _mm512_load_epi32((const int32_t*) w + 16);
    const __m512i vksum2 = _mm512_load_epi32((const int32_t*) w + 32);
    const __m512i vksum3 = _mm512_load_epi32((const int32_t*) w + 48);
    w = (const int32_t*) w + 64;

    // Zero tile accumulator
    _tile_zero(0);
    _tile_zero(1);
    _tile_zero(2);
    _tile_zero(3);

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
    __m512i vacc0x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[0].zero_point));
    __m512i vacc0x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[0].zero_point));
    __m512i vacc0x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params[0].zero_point));
    __m512i vacc0x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params[0].zero_point));
    __m512i vacc1x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[1].zero_point));
    __m512i vacc1x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[1].zero_point));
    __m512i vacc1x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params[1].zero_point));
    __m512i vacc1x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params[1].zero_point));
    __m512i vacc2x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[2].zero_point));
    __m512i vacc2x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[2].zero_point));
    __m512i vacc2x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params[2].zero_point));
    __m512i vacc2x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params[2].zero_point));
    __m512i vacc3x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[3].zero_point));
    __m512i vacc3x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[3].zero_point));
    __m512i vacc3x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params[3].zero_point));
    __m512i vacc3x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params[3].zero_point));
    __m512i vacc4x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[4].zero_point));
    __m512i vacc4x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[4].zero_point));
    __m512i vacc4x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params[4].zero_point));
    __m512i vacc4x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params[4].zero_point));
    __m512i vacc5x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[5].zero_point));
    __m512i vacc5x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[5].zero_point));
    __m512i vacc5x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params[5].zero_point));
    __m512i vacc5x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params[5].zero_point));
    __m512i vacc6x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[6].zero_point));
    __m512i vacc6x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[6].zero_point));
    __m512i vacc6x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params[6].zero_point));
    __m512i vacc6x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params[6].zero_point));
    // Add tile to bias
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

    vscaled0x0 = _mm512_mul_ps(vscaled0x0, _mm512_set1_ps(quantization_params[0].inv_scale));
    vscaled0x1 = _mm512_mul_ps(vscaled0x1, _mm512_set1_ps(quantization_params[0].inv_scale));
    vscaled0x2 = _mm512_mul_ps(vscaled0x2, _mm512_set1_ps(quantization_params[0].inv_scale));
    vscaled0x3 = _mm512_mul_ps(vscaled0x3, _mm512_set1_ps(quantization_params[0].inv_scale));
    vscaled1x0 = _mm512_mul_ps(vscaled1x0, _mm512_set1_ps(quantization_params[1].inv_scale));
    vscaled1x1 = _mm512_mul_ps(vscaled1x1, _mm512_set1_ps(quantization_params[1].inv_scale));
    vscaled1x2 = _mm512_mul_ps(vscaled1x2, _mm512_set1_ps(quantization_params[1].inv_scale));
    vscaled1x3 = _mm512_mul_ps(vscaled1x3, _mm512_set1_ps(quantization_params[1].inv_scale));
    vscaled2x0 = _mm512_mul_ps(vscaled2x0, _mm512_set1_ps(quantization_params[2].inv_scale));
    vscaled2x1 = _mm512_mul_ps(vscaled2x1, _mm512_set1_ps(quantization_params[2].inv_scale));
    vscaled2x2 = _mm512_mul_ps(vscaled2x2, _mm512_set1_ps(quantization_params[2].inv_scale));
    vscaled2x3 = _mm512_mul_ps(vscaled2x3, _mm512_set1_ps(quantization_params[2].inv_scale));
    vscaled3x0 = _mm512_mul_ps(vscaled3x0, _mm512_set1_ps(quantization_params[3].inv_scale));
    vscaled3x1 = _mm512_mul_ps(vscaled3x1, _mm512_set1_ps(quantization_params[3].inv_scale));
    vscaled3x2 = _mm512_mul_ps(vscaled3x2, _mm512_set1_ps(quantization_params[3].inv_scale));
    vscaled3x3 = _mm512_mul_ps(vscaled3x3, _mm512_set1_ps(quantization_params[3].inv_scale));
    vscaled4x0 = _mm512_mul_ps(vscaled4x0, _mm512_set1_ps(quantization_params[4].inv_scale));
    vscaled4x1 = _mm512_mul_ps(vscaled4x1, _mm512_set1_ps(quantization_params[4].inv_scale));
    vscaled4x2 = _mm512_mul_ps(vscaled4x2, _mm512_set1_ps(quantization_params[4].inv_scale));
    vscaled4x3 = _mm512_mul_ps(vscaled4x3, _mm512_set1_ps(quantization_params[4].inv_scale));
    vscaled5x0 = _mm512_mul_ps(vscaled5x0, _mm512_set1_ps(quantization_params[5].inv_scale));
    vscaled5x1 = _mm512_mul_ps(vscaled5x1, _mm512_set1_ps(quantization_params[5].inv_scale));
    vscaled5x2 = _mm512_mul_ps(vscaled5x2, _mm512_set1_ps(quantization_params[5].inv_scale));
    vscaled5x3 = _mm512_mul_ps(vscaled5x3, _mm512_set1_ps(quantization_params[5].inv_scale));
    vscaled6x0 = _mm512_mul_ps(vscaled6x0, _mm512_set1_ps(quantization_params[6].inv_scale));
    vscaled6x1 = _mm512_mul_ps(vscaled6x1, _mm512_set1_ps(quantization_params[6].inv_scale));
    vscaled6x2 = _mm512_mul_ps(vscaled6x2, _mm512_set1_ps(quantization_params[6].inv_scale));
    vscaled6x3 = _mm512_mul_ps(vscaled6x3, _mm512_set1_ps(quantization_params[6].inv_scale));

    const __m512 vfilter_output_scale0 = _mm512_load_ps((const float*) w + 0);
    const __m512 vfilter_output_scale1 = _mm512_load_ps((const float*) w + 16);
    const __m512 vfilter_output_scale2 = _mm512_load_ps((const float*) w + 32);
    const __m512 vfilter_output_scale3 = _mm512_load_ps((const float*) w + 48);
    w = (const int32_t*) w + 64;
    const __m512 vbias0 = _mm512_load_ps((const float*) w + 0);
    const __m512 vbias1 = _mm512_load_ps((const float*) w + 16);
    const __m512 vbias2 = _mm512_load_ps((const float*) w + 32);
    const __m512 vbias3 = _mm512_load_ps((const float*) w + 48);
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
    if XNN_LIKELY(nc >= 64) {
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

      a -= kc;
      nc -= 64;
    } else {
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 0) & 0xFFFF));
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 16) & 0xFFFF));
      const __mmask16 vmask2 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 32) & 0xFFFF));
      const __mmask16 vmask3 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 48) & 0xFFFF));
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
