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


void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_qc4w_minmax_params* restrict params,
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
  XNN_ALIGN(64) int8_t weight_buffer[16 * 64];

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
  const __m512i vmask = _mm512_set1_epi8(0xF0);
  const __m512i vshl4 = _mm512_set1_epi64(0x01020408);
  XNN_FORCE_REALIZATION(vmask);
  XNN_FORCE_REALIZATION(vshl4);

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
      const __m512i vb0x0 = _mm512_load_epi32((const int8_t*) w + 0);
      const __m512i vb1x0 = _mm512_load_epi32((const int8_t*) w + 256);
      const __m512i vb2x0 = _mm512_load_epi32((const int8_t*) w + 512);
      const __m512i vb3x0 = _mm512_load_epi32((const int8_t*) w + 768);
      const __m512i vb4x0 = _mm512_load_epi32((const int8_t*) w + 1024);
      const __m512i vb5x0 = _mm512_load_epi32((const int8_t*) w + 1280);
      const __m512i vb6x0 = _mm512_load_epi32((const int8_t*) w + 1536);
      const __m512i vb7x0 = _mm512_load_epi32((const int8_t*) w + 1792);
      const __m512i vl0x0 = _mm512_gf2p8affine_epi64_epi8(vb0x0, vshl4, 0);
      const __m512i vh0x0 = _mm512_and_si512(vb0x0, vmask);
      const __m512i vl1x0 = _mm512_gf2p8affine_epi64_epi8(vb1x0, vshl4, 0);
      const __m512i vh1x0 = _mm512_and_si512(vb1x0, vmask);
      const __m512i vl2x0 = _mm512_gf2p8affine_epi64_epi8(vb2x0, vshl4, 0);
      const __m512i vh2x0 = _mm512_and_si512(vb2x0, vmask);
      const __m512i vl3x0 = _mm512_gf2p8affine_epi64_epi8(vb3x0, vshl4, 0);
      const __m512i vh3x0 = _mm512_and_si512(vb3x0, vmask);
      const __m512i vl4x0 = _mm512_gf2p8affine_epi64_epi8(vb4x0, vshl4, 0);
      const __m512i vh4x0 = _mm512_and_si512(vb4x0, vmask);
      const __m512i vl5x0 = _mm512_gf2p8affine_epi64_epi8(vb5x0, vshl4, 0);
      const __m512i vh5x0 = _mm512_and_si512(vb5x0, vmask);
      const __m512i vl6x0 = _mm512_gf2p8affine_epi64_epi8(vb6x0, vshl4, 0);
      const __m512i vh6x0 = _mm512_and_si512(vb6x0, vmask);
      const __m512i vl7x0 = _mm512_gf2p8affine_epi64_epi8(vb7x0, vshl4, 0);
      const __m512i vh7x0 = _mm512_and_si512(vb7x0, vmask);
      _mm512_store_epi32(weight_buffer + 0, vl0x0);
      _mm512_store_epi32(weight_buffer + 64, vh0x0);
      _mm512_store_epi32(weight_buffer + 128, vl1x0);
      _mm512_store_epi32(weight_buffer + 192, vh1x0);
      _mm512_store_epi32(weight_buffer + 256, vl2x0);
      _mm512_store_epi32(weight_buffer + 320, vh2x0);
      _mm512_store_epi32(weight_buffer + 384, vl3x0);
      _mm512_store_epi32(weight_buffer + 448, vh3x0);
      _mm512_store_epi32(weight_buffer + 512, vl4x0);
      _mm512_store_epi32(weight_buffer + 576, vh4x0);
      _mm512_store_epi32(weight_buffer + 640, vl5x0);
      _mm512_store_epi32(weight_buffer + 704, vh5x0);
      _mm512_store_epi32(weight_buffer + 768, vl6x0);
      _mm512_store_epi32(weight_buffer + 832, vh6x0);
      _mm512_store_epi32(weight_buffer + 896, vl7x0);
      _mm512_store_epi32(weight_buffer + 960, vh7x0);
      _tile_loadd(5, weight_buffer, 64);
      _tile_dpbssd(0, 4, 5);
      const __m512i vb0x1 = _mm512_load_epi32((const int8_t*) w + 64);
      const __m512i vb1x1 = _mm512_load_epi32((const int8_t*) w + 320);
      const __m512i vb2x1 = _mm512_load_epi32((const int8_t*) w + 576);
      const __m512i vb3x1 = _mm512_load_epi32((const int8_t*) w + 832);
      const __m512i vb4x1 = _mm512_load_epi32((const int8_t*) w + 1088);
      const __m512i vb5x1 = _mm512_load_epi32((const int8_t*) w + 1344);
      const __m512i vb6x1 = _mm512_load_epi32((const int8_t*) w + 1600);
      const __m512i vb7x1 = _mm512_load_epi32((const int8_t*) w + 1856);
      const __m512i vl0x1 = _mm512_gf2p8affine_epi64_epi8(vb0x1, vshl4, 0);
      const __m512i vh0x1 = _mm512_and_si512(vb0x1, vmask);
      const __m512i vl1x1 = _mm512_gf2p8affine_epi64_epi8(vb1x1, vshl4, 0);
      const __m512i vh1x1 = _mm512_and_si512(vb1x1, vmask);
      const __m512i vl2x1 = _mm512_gf2p8affine_epi64_epi8(vb2x1, vshl4, 0);
      const __m512i vh2x1 = _mm512_and_si512(vb2x1, vmask);
      const __m512i vl3x1 = _mm512_gf2p8affine_epi64_epi8(vb3x1, vshl4, 0);
      const __m512i vh3x1 = _mm512_and_si512(vb3x1, vmask);
      const __m512i vl4x1 = _mm512_gf2p8affine_epi64_epi8(vb4x1, vshl4, 0);
      const __m512i vh4x1 = _mm512_and_si512(vb4x1, vmask);
      const __m512i vl5x1 = _mm512_gf2p8affine_epi64_epi8(vb5x1, vshl4, 0);
      const __m512i vh5x1 = _mm512_and_si512(vb5x1, vmask);
      const __m512i vl6x1 = _mm512_gf2p8affine_epi64_epi8(vb6x1, vshl4, 0);
      const __m512i vh6x1 = _mm512_and_si512(vb6x1, vmask);
      const __m512i vl7x1 = _mm512_gf2p8affine_epi64_epi8(vb7x1, vshl4, 0);
      const __m512i vh7x1 = _mm512_and_si512(vb7x1, vmask);
      _mm512_store_epi32(weight_buffer + 0, vl0x1);
      _mm512_store_epi32(weight_buffer + 64, vh0x1);
      _mm512_store_epi32(weight_buffer + 128, vl1x1);
      _mm512_store_epi32(weight_buffer + 192, vh1x1);
      _mm512_store_epi32(weight_buffer + 256, vl2x1);
      _mm512_store_epi32(weight_buffer + 320, vh2x1);
      _mm512_store_epi32(weight_buffer + 384, vl3x1);
      _mm512_store_epi32(weight_buffer + 448, vh3x1);
      _mm512_store_epi32(weight_buffer + 512, vl4x1);
      _mm512_store_epi32(weight_buffer + 576, vh4x1);
      _mm512_store_epi32(weight_buffer + 640, vl5x1);
      _mm512_store_epi32(weight_buffer + 704, vh5x1);
      _mm512_store_epi32(weight_buffer + 768, vl6x1);
      _mm512_store_epi32(weight_buffer + 832, vh6x1);
      _mm512_store_epi32(weight_buffer + 896, vl7x1);
      _mm512_store_epi32(weight_buffer + 960, vh7x1);
      _tile_loadd(5, weight_buffer, 64);
      _tile_dpbssd(1, 4, 5);
      const __m512i vb0x2 = _mm512_load_epi32((const int8_t*) w + 128);
      const __m512i vb1x2 = _mm512_load_epi32((const int8_t*) w + 384);
      const __m512i vb2x2 = _mm512_load_epi32((const int8_t*) w + 640);
      const __m512i vb3x2 = _mm512_load_epi32((const int8_t*) w + 896);
      const __m512i vb4x2 = _mm512_load_epi32((const int8_t*) w + 1152);
      const __m512i vb5x2 = _mm512_load_epi32((const int8_t*) w + 1408);
      const __m512i vb6x2 = _mm512_load_epi32((const int8_t*) w + 1664);
      const __m512i vb7x2 = _mm512_load_epi32((const int8_t*) w + 1920);
      const __m512i vl0x2 = _mm512_gf2p8affine_epi64_epi8(vb0x2, vshl4, 0);
      const __m512i vh0x2 = _mm512_and_si512(vb0x2, vmask);
      const __m512i vl1x2 = _mm512_gf2p8affine_epi64_epi8(vb1x2, vshl4, 0);
      const __m512i vh1x2 = _mm512_and_si512(vb1x2, vmask);
      const __m512i vl2x2 = _mm512_gf2p8affine_epi64_epi8(vb2x2, vshl4, 0);
      const __m512i vh2x2 = _mm512_and_si512(vb2x2, vmask);
      const __m512i vl3x2 = _mm512_gf2p8affine_epi64_epi8(vb3x2, vshl4, 0);
      const __m512i vh3x2 = _mm512_and_si512(vb3x2, vmask);
      const __m512i vl4x2 = _mm512_gf2p8affine_epi64_epi8(vb4x2, vshl4, 0);
      const __m512i vh4x2 = _mm512_and_si512(vb4x2, vmask);
      const __m512i vl5x2 = _mm512_gf2p8affine_epi64_epi8(vb5x2, vshl4, 0);
      const __m512i vh5x2 = _mm512_and_si512(vb5x2, vmask);
      const __m512i vl6x2 = _mm512_gf2p8affine_epi64_epi8(vb6x2, vshl4, 0);
      const __m512i vh6x2 = _mm512_and_si512(vb6x2, vmask);
      const __m512i vl7x2 = _mm512_gf2p8affine_epi64_epi8(vb7x2, vshl4, 0);
      const __m512i vh7x2 = _mm512_and_si512(vb7x2, vmask);
      _mm512_store_epi32(weight_buffer + 0, vl0x2);
      _mm512_store_epi32(weight_buffer + 64, vh0x2);
      _mm512_store_epi32(weight_buffer + 128, vl1x2);
      _mm512_store_epi32(weight_buffer + 192, vh1x2);
      _mm512_store_epi32(weight_buffer + 256, vl2x2);
      _mm512_store_epi32(weight_buffer + 320, vh2x2);
      _mm512_store_epi32(weight_buffer + 384, vl3x2);
      _mm512_store_epi32(weight_buffer + 448, vh3x2);
      _mm512_store_epi32(weight_buffer + 512, vl4x2);
      _mm512_store_epi32(weight_buffer + 576, vh4x2);
      _mm512_store_epi32(weight_buffer + 640, vl5x2);
      _mm512_store_epi32(weight_buffer + 704, vh5x2);
      _mm512_store_epi32(weight_buffer + 768, vl6x2);
      _mm512_store_epi32(weight_buffer + 832, vh6x2);
      _mm512_store_epi32(weight_buffer + 896, vl7x2);
      _mm512_store_epi32(weight_buffer + 960, vh7x2);
      _tile_loadd(5, weight_buffer, 64);
      _tile_dpbssd(2, 4, 5);
      const __m512i vb0x3 = _mm512_load_epi32((const int8_t*) w + 192);
      const __m512i vb1x3 = _mm512_load_epi32((const int8_t*) w + 448);
      const __m512i vb2x3 = _mm512_load_epi32((const int8_t*) w + 704);
      const __m512i vb3x3 = _mm512_load_epi32((const int8_t*) w + 960);
      const __m512i vb4x3 = _mm512_load_epi32((const int8_t*) w + 1216);
      const __m512i vb5x3 = _mm512_load_epi32((const int8_t*) w + 1472);
      const __m512i vb6x3 = _mm512_load_epi32((const int8_t*) w + 1728);
      const __m512i vb7x3 = _mm512_load_epi32((const int8_t*) w + 1984);
      const __m512i vl0x3 = _mm512_gf2p8affine_epi64_epi8(vb0x3, vshl4, 0);
      const __m512i vh0x3 = _mm512_and_si512(vb0x3, vmask);
      const __m512i vl1x3 = _mm512_gf2p8affine_epi64_epi8(vb1x3, vshl4, 0);
      const __m512i vh1x3 = _mm512_and_si512(vb1x3, vmask);
      const __m512i vl2x3 = _mm512_gf2p8affine_epi64_epi8(vb2x3, vshl4, 0);
      const __m512i vh2x3 = _mm512_and_si512(vb2x3, vmask);
      const __m512i vl3x3 = _mm512_gf2p8affine_epi64_epi8(vb3x3, vshl4, 0);
      const __m512i vh3x3 = _mm512_and_si512(vb3x3, vmask);
      const __m512i vl4x3 = _mm512_gf2p8affine_epi64_epi8(vb4x3, vshl4, 0);
      const __m512i vh4x3 = _mm512_and_si512(vb4x3, vmask);
      const __m512i vl5x3 = _mm512_gf2p8affine_epi64_epi8(vb5x3, vshl4, 0);
      const __m512i vh5x3 = _mm512_and_si512(vb5x3, vmask);
      const __m512i vl6x3 = _mm512_gf2p8affine_epi64_epi8(vb6x3, vshl4, 0);
      const __m512i vh6x3 = _mm512_and_si512(vb6x3, vmask);
      const __m512i vl7x3 = _mm512_gf2p8affine_epi64_epi8(vb7x3, vshl4, 0);
      const __m512i vh7x3 = _mm512_and_si512(vb7x3, vmask);
      _mm512_store_epi32(weight_buffer + 0, vl0x3);
      _mm512_store_epi32(weight_buffer + 64, vh0x3);
      _mm512_store_epi32(weight_buffer + 128, vl1x3);
      _mm512_store_epi32(weight_buffer + 192, vh1x3);
      _mm512_store_epi32(weight_buffer + 256, vl2x3);
      _mm512_store_epi32(weight_buffer + 320, vh2x3);
      _mm512_store_epi32(weight_buffer + 384, vl3x3);
      _mm512_store_epi32(weight_buffer + 448, vh3x3);
      _mm512_store_epi32(weight_buffer + 512, vl4x3);
      _mm512_store_epi32(weight_buffer + 576, vh4x3);
      _mm512_store_epi32(weight_buffer + 640, vl5x3);
      _mm512_store_epi32(weight_buffer + 704, vh5x3);
      _mm512_store_epi32(weight_buffer + 768, vl6x3);
      _mm512_store_epi32(weight_buffer + 832, vh6x3);
      _mm512_store_epi32(weight_buffer + 896, vl7x3);
      _mm512_store_epi32(weight_buffer + 960, vh7x3);
      _tile_loadd(5, weight_buffer, 64);
      _tile_dpbssd(3, 4, 5);

      w = (const int8_t*) w + 2048;
      k -= 64 * sizeof(int8_t);
    }

    if XNN_UNLIKELY(k != 0) {
      _tile_loadd(6, a, a_stride);
      a += kremainder;
      for (size_t k = 0; k < ((kremainder + 7) >> 3); ++k) {
        const __m512i vb0 = _mm512_load_epi32((const int8_t*) w + 0 + 256 * k);
        const __m512i vl0 = _mm512_gf2p8affine_epi64_epi8(vb0, vshl4, 0);
        const __m512i vh0 = _mm512_and_si512(vb0, vmask);
        _mm512_store_epi32(weight_buffer + 128 * k + 0, vl0);
        _mm512_store_epi32(weight_buffer + 128 * k + 64, vh0);
      }
      _tile_loadd(7, weight_buffer, 64);
      _tile_dpbssd(0, 6, 7);
      for (size_t k = 0; k < ((kremainder + 7) >> 3); ++k) {
        const __m512i vb1 = _mm512_load_epi32((const int8_t*) w + 64 + 256 * k);
        const __m512i vl1 = _mm512_gf2p8affine_epi64_epi8(vb1, vshl4, 0);
        const __m512i vh1 = _mm512_and_si512(vb1, vmask);
        _mm512_store_epi32(weight_buffer + 128 * k + 0, vl1);
        _mm512_store_epi32(weight_buffer + 128 * k + 64, vh1);
      }
      _tile_loadd(7, weight_buffer, 64);
      _tile_dpbssd(1, 6, 7);
      for (size_t k = 0; k < ((kremainder + 7) >> 3); ++k) {
        const __m512i vb2 = _mm512_load_epi32((const int8_t*) w + 128 + 256 * k);
        const __m512i vl2 = _mm512_gf2p8affine_epi64_epi8(vb2, vshl4, 0);
        const __m512i vh2 = _mm512_and_si512(vb2, vmask);
        _mm512_store_epi32(weight_buffer + 128 * k + 0, vl2);
        _mm512_store_epi32(weight_buffer + 128 * k + 64, vh2);
      }
      _tile_loadd(7, weight_buffer, 64);
      _tile_dpbssd(2, 6, 7);
      for (size_t k = 0; k < ((kremainder + 7) >> 3); ++k) {
        const __m512i vb3 = _mm512_load_epi32((const int8_t*) w + 192 + 256 * k);
        const __m512i vl3 = _mm512_gf2p8affine_epi64_epi8(vb3, vshl4, 0);
        const __m512i vh3 = _mm512_and_si512(vb3, vmask);
        _mm512_store_epi32(weight_buffer + 128 * k + 0, vl3);
        _mm512_store_epi32(weight_buffer + 128 * k + 64, vh3);
      }
      _tile_loadd(7, weight_buffer, 64);
      _tile_dpbssd(3, 6, 7);

      w = (const int8_t*) w + ((kremainder + 7) >> 3) * 256;
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
    __m512i vacc7x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[7].zero_point));
    __m512i vacc7x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[7].zero_point));
    __m512i vacc7x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params[7].zero_point));
    __m512i vacc7x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params[7].zero_point));
    __m512i vacc8x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[8].zero_point));
    __m512i vacc8x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[8].zero_point));
    __m512i vacc8x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params[8].zero_point));
    __m512i vacc8x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params[8].zero_point));
    __m512i vacc9x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[9].zero_point));
    __m512i vacc9x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[9].zero_point));
    __m512i vacc9x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params[9].zero_point));
    __m512i vacc9x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params[9].zero_point));
    __m512i vacc10x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[10].zero_point));
    __m512i vacc10x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[10].zero_point));
    __m512i vacc10x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params[10].zero_point));
    __m512i vacc10x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params[10].zero_point));
    __m512i vacc11x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[11].zero_point));
    __m512i vacc11x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[11].zero_point));
    __m512i vacc11x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params[11].zero_point));
    __m512i vacc11x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params[11].zero_point));
    __m512i vacc12x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[12].zero_point));
    __m512i vacc12x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[12].zero_point));
    __m512i vacc12x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params[12].zero_point));
    __m512i vacc12x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params[12].zero_point));
    __m512i vacc13x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[13].zero_point));
    __m512i vacc13x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[13].zero_point));
    __m512i vacc13x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params[13].zero_point));
    __m512i vacc13x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params[13].zero_point));
    __m512i vacc14x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[14].zero_point));
    __m512i vacc14x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[14].zero_point));
    __m512i vacc14x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params[14].zero_point));
    __m512i vacc14x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params[14].zero_point));
    __m512i vacc15x0 = _mm512_mullo_epi32(vksum0, _mm512_set1_epi32((int) quantization_params[15].zero_point));
    __m512i vacc15x1 = _mm512_mullo_epi32(vksum1, _mm512_set1_epi32((int) quantization_params[15].zero_point));
    __m512i vacc15x2 = _mm512_mullo_epi32(vksum2, _mm512_set1_epi32((int) quantization_params[15].zero_point));
    __m512i vacc15x3 = _mm512_mullo_epi32(vksum3, _mm512_set1_epi32((int) quantization_params[15].zero_point));
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

    vacc0x0 = _mm512_srai_epi32(vacc0x0, 4);
    vacc0x1 = _mm512_srai_epi32(vacc0x1, 4);
    vacc0x2 = _mm512_srai_epi32(vacc0x2, 4);
    vacc0x3 = _mm512_srai_epi32(vacc0x3, 4);
    vacc1x0 = _mm512_srai_epi32(vacc1x0, 4);
    vacc1x1 = _mm512_srai_epi32(vacc1x1, 4);
    vacc1x2 = _mm512_srai_epi32(vacc1x2, 4);
    vacc1x3 = _mm512_srai_epi32(vacc1x3, 4);
    vacc2x0 = _mm512_srai_epi32(vacc2x0, 4);
    vacc2x1 = _mm512_srai_epi32(vacc2x1, 4);
    vacc2x2 = _mm512_srai_epi32(vacc2x2, 4);
    vacc2x3 = _mm512_srai_epi32(vacc2x3, 4);
    vacc3x0 = _mm512_srai_epi32(vacc3x0, 4);
    vacc3x1 = _mm512_srai_epi32(vacc3x1, 4);
    vacc3x2 = _mm512_srai_epi32(vacc3x2, 4);
    vacc3x3 = _mm512_srai_epi32(vacc3x3, 4);
    vacc4x0 = _mm512_srai_epi32(vacc4x0, 4);
    vacc4x1 = _mm512_srai_epi32(vacc4x1, 4);
    vacc4x2 = _mm512_srai_epi32(vacc4x2, 4);
    vacc4x3 = _mm512_srai_epi32(vacc4x3, 4);
    vacc5x0 = _mm512_srai_epi32(vacc5x0, 4);
    vacc5x1 = _mm512_srai_epi32(vacc5x1, 4);
    vacc5x2 = _mm512_srai_epi32(vacc5x2, 4);
    vacc5x3 = _mm512_srai_epi32(vacc5x3, 4);
    vacc6x0 = _mm512_srai_epi32(vacc6x0, 4);
    vacc6x1 = _mm512_srai_epi32(vacc6x1, 4);
    vacc6x2 = _mm512_srai_epi32(vacc6x2, 4);
    vacc6x3 = _mm512_srai_epi32(vacc6x3, 4);
    vacc7x0 = _mm512_srai_epi32(vacc7x0, 4);
    vacc7x1 = _mm512_srai_epi32(vacc7x1, 4);
    vacc7x2 = _mm512_srai_epi32(vacc7x2, 4);
    vacc7x3 = _mm512_srai_epi32(vacc7x3, 4);
    vacc8x0 = _mm512_srai_epi32(vacc8x0, 4);
    vacc8x1 = _mm512_srai_epi32(vacc8x1, 4);
    vacc8x2 = _mm512_srai_epi32(vacc8x2, 4);
    vacc8x3 = _mm512_srai_epi32(vacc8x3, 4);
    vacc9x0 = _mm512_srai_epi32(vacc9x0, 4);
    vacc9x1 = _mm512_srai_epi32(vacc9x1, 4);
    vacc9x2 = _mm512_srai_epi32(vacc9x2, 4);
    vacc9x3 = _mm512_srai_epi32(vacc9x3, 4);
    vacc10x0 = _mm512_srai_epi32(vacc10x0, 4);
    vacc10x1 = _mm512_srai_epi32(vacc10x1, 4);
    vacc10x2 = _mm512_srai_epi32(vacc10x2, 4);
    vacc10x3 = _mm512_srai_epi32(vacc10x3, 4);
    vacc11x0 = _mm512_srai_epi32(vacc11x0, 4);
    vacc11x1 = _mm512_srai_epi32(vacc11x1, 4);
    vacc11x2 = _mm512_srai_epi32(vacc11x2, 4);
    vacc11x3 = _mm512_srai_epi32(vacc11x3, 4);
    vacc12x0 = _mm512_srai_epi32(vacc12x0, 4);
    vacc12x1 = _mm512_srai_epi32(vacc12x1, 4);
    vacc12x2 = _mm512_srai_epi32(vacc12x2, 4);
    vacc12x3 = _mm512_srai_epi32(vacc12x3, 4);
    vacc13x0 = _mm512_srai_epi32(vacc13x0, 4);
    vacc13x1 = _mm512_srai_epi32(vacc13x1, 4);
    vacc13x2 = _mm512_srai_epi32(vacc13x2, 4);
    vacc13x3 = _mm512_srai_epi32(vacc13x3, 4);
    vacc14x0 = _mm512_srai_epi32(vacc14x0, 4);
    vacc14x1 = _mm512_srai_epi32(vacc14x1, 4);
    vacc14x2 = _mm512_srai_epi32(vacc14x2, 4);
    vacc14x3 = _mm512_srai_epi32(vacc14x3, 4);
    vacc15x0 = _mm512_srai_epi32(vacc15x0, 4);
    vacc15x1 = _mm512_srai_epi32(vacc15x1, 4);
    vacc15x2 = _mm512_srai_epi32(vacc15x2, 4);
    vacc15x3 = _mm512_srai_epi32(vacc15x3, 4);
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
    vscaled7x0 = _mm512_mul_ps(vscaled7x0, _mm512_set1_ps(quantization_params[7].inv_scale));
    vscaled7x1 = _mm512_mul_ps(vscaled7x1, _mm512_set1_ps(quantization_params[7].inv_scale));
    vscaled7x2 = _mm512_mul_ps(vscaled7x2, _mm512_set1_ps(quantization_params[7].inv_scale));
    vscaled7x3 = _mm512_mul_ps(vscaled7x3, _mm512_set1_ps(quantization_params[7].inv_scale));
    vscaled8x0 = _mm512_mul_ps(vscaled8x0, _mm512_set1_ps(quantization_params[8].inv_scale));
    vscaled8x1 = _mm512_mul_ps(vscaled8x1, _mm512_set1_ps(quantization_params[8].inv_scale));
    vscaled8x2 = _mm512_mul_ps(vscaled8x2, _mm512_set1_ps(quantization_params[8].inv_scale));
    vscaled8x3 = _mm512_mul_ps(vscaled8x3, _mm512_set1_ps(quantization_params[8].inv_scale));
    vscaled9x0 = _mm512_mul_ps(vscaled9x0, _mm512_set1_ps(quantization_params[9].inv_scale));
    vscaled9x1 = _mm512_mul_ps(vscaled9x1, _mm512_set1_ps(quantization_params[9].inv_scale));
    vscaled9x2 = _mm512_mul_ps(vscaled9x2, _mm512_set1_ps(quantization_params[9].inv_scale));
    vscaled9x3 = _mm512_mul_ps(vscaled9x3, _mm512_set1_ps(quantization_params[9].inv_scale));
    vscaled10x0 = _mm512_mul_ps(vscaled10x0, _mm512_set1_ps(quantization_params[10].inv_scale));
    vscaled10x1 = _mm512_mul_ps(vscaled10x1, _mm512_set1_ps(quantization_params[10].inv_scale));
    vscaled10x2 = _mm512_mul_ps(vscaled10x2, _mm512_set1_ps(quantization_params[10].inv_scale));
    vscaled10x3 = _mm512_mul_ps(vscaled10x3, _mm512_set1_ps(quantization_params[10].inv_scale));
    vscaled11x0 = _mm512_mul_ps(vscaled11x0, _mm512_set1_ps(quantization_params[11].inv_scale));
    vscaled11x1 = _mm512_mul_ps(vscaled11x1, _mm512_set1_ps(quantization_params[11].inv_scale));
    vscaled11x2 = _mm512_mul_ps(vscaled11x2, _mm512_set1_ps(quantization_params[11].inv_scale));
    vscaled11x3 = _mm512_mul_ps(vscaled11x3, _mm512_set1_ps(quantization_params[11].inv_scale));
    vscaled12x0 = _mm512_mul_ps(vscaled12x0, _mm512_set1_ps(quantization_params[12].inv_scale));
    vscaled12x1 = _mm512_mul_ps(vscaled12x1, _mm512_set1_ps(quantization_params[12].inv_scale));
    vscaled12x2 = _mm512_mul_ps(vscaled12x2, _mm512_set1_ps(quantization_params[12].inv_scale));
    vscaled12x3 = _mm512_mul_ps(vscaled12x3, _mm512_set1_ps(quantization_params[12].inv_scale));
    vscaled13x0 = _mm512_mul_ps(vscaled13x0, _mm512_set1_ps(quantization_params[13].inv_scale));
    vscaled13x1 = _mm512_mul_ps(vscaled13x1, _mm512_set1_ps(quantization_params[13].inv_scale));
    vscaled13x2 = _mm512_mul_ps(vscaled13x2, _mm512_set1_ps(quantization_params[13].inv_scale));
    vscaled13x3 = _mm512_mul_ps(vscaled13x3, _mm512_set1_ps(quantization_params[13].inv_scale));
    vscaled14x0 = _mm512_mul_ps(vscaled14x0, _mm512_set1_ps(quantization_params[14].inv_scale));
    vscaled14x1 = _mm512_mul_ps(vscaled14x1, _mm512_set1_ps(quantization_params[14].inv_scale));
    vscaled14x2 = _mm512_mul_ps(vscaled14x2, _mm512_set1_ps(quantization_params[14].inv_scale));
    vscaled14x3 = _mm512_mul_ps(vscaled14x3, _mm512_set1_ps(quantization_params[14].inv_scale));
    vscaled15x0 = _mm512_mul_ps(vscaled15x0, _mm512_set1_ps(quantization_params[15].inv_scale));
    vscaled15x1 = _mm512_mul_ps(vscaled15x1, _mm512_set1_ps(quantization_params[15].inv_scale));
    vscaled15x2 = _mm512_mul_ps(vscaled15x2, _mm512_set1_ps(quantization_params[15].inv_scale));
    vscaled15x3 = _mm512_mul_ps(vscaled15x3, _mm512_set1_ps(quantization_params[15].inv_scale));

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

    if XNN_LIKELY(nc >= 64) {
      _mm512_storeu_ps(c15 + 0, vscaled15x0);
      _mm512_storeu_ps(c15 + 16, vscaled15x1);
      _mm512_storeu_ps(c15 + 32, vscaled15x2);
      _mm512_storeu_ps(c15 + 48, vscaled15x3);
      c15 = (float*) ((uintptr_t) c15 + cn_stride);
      _mm512_storeu_ps(c14 + 0, vscaled14x0);
      _mm512_storeu_ps(c14 + 16, vscaled14x1);
      _mm512_storeu_ps(c14 + 32, vscaled14x2);
      _mm512_storeu_ps(c14 + 48, vscaled14x3);
      c14 = (float*) ((uintptr_t) c14 + cn_stride);
      _mm512_storeu_ps(c13 + 0, vscaled13x0);
      _mm512_storeu_ps(c13 + 16, vscaled13x1);
      _mm512_storeu_ps(c13 + 32, vscaled13x2);
      _mm512_storeu_ps(c13 + 48, vscaled13x3);
      c13 = (float*) ((uintptr_t) c13 + cn_stride);
      _mm512_storeu_ps(c12 + 0, vscaled12x0);
      _mm512_storeu_ps(c12 + 16, vscaled12x1);
      _mm512_storeu_ps(c12 + 32, vscaled12x2);
      _mm512_storeu_ps(c12 + 48, vscaled12x3);
      c12 = (float*) ((uintptr_t) c12 + cn_stride);
      _mm512_storeu_ps(c11 + 0, vscaled11x0);
      _mm512_storeu_ps(c11 + 16, vscaled11x1);
      _mm512_storeu_ps(c11 + 32, vscaled11x2);
      _mm512_storeu_ps(c11 + 48, vscaled11x3);
      c11 = (float*) ((uintptr_t) c11 + cn_stride);
      _mm512_storeu_ps(c10 + 0, vscaled10x0);
      _mm512_storeu_ps(c10 + 16, vscaled10x1);
      _mm512_storeu_ps(c10 + 32, vscaled10x2);
      _mm512_storeu_ps(c10 + 48, vscaled10x3);
      c10 = (float*) ((uintptr_t) c10 + cn_stride);
      _mm512_storeu_ps(c9 + 0, vscaled9x0);
      _mm512_storeu_ps(c9 + 16, vscaled9x1);
      _mm512_storeu_ps(c9 + 32, vscaled9x2);
      _mm512_storeu_ps(c9 + 48, vscaled9x3);
      c9 = (float*) ((uintptr_t) c9 + cn_stride);
      _mm512_storeu_ps(c8 + 0, vscaled8x0);
      _mm512_storeu_ps(c8 + 16, vscaled8x1);
      _mm512_storeu_ps(c8 + 32, vscaled8x2);
      _mm512_storeu_ps(c8 + 48, vscaled8x3);
      c8 = (float*) ((uintptr_t) c8 + cn_stride);
      _mm512_storeu_ps(c7 + 0, vscaled7x0);
      _mm512_storeu_ps(c7 + 16, vscaled7x1);
      _mm512_storeu_ps(c7 + 32, vscaled7x2);
      _mm512_storeu_ps(c7 + 48, vscaled7x3);
      c7 = (float*) ((uintptr_t) c7 + cn_stride);
      _mm512_storeu_ps(c6 + 0, vscaled6x0);
      _mm512_storeu_ps(c6 + 16, vscaled6x1);
      _mm512_storeu_ps(c6 + 32, vscaled6x2);
      _mm512_storeu_ps(c6 + 48, vscaled6x3);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      _mm512_storeu_ps(c5 + 0, vscaled5x0);
      _mm512_storeu_ps(c5 + 16, vscaled5x1);
      _mm512_storeu_ps(c5 + 32, vscaled5x2);
      _mm512_storeu_ps(c5 + 48, vscaled5x3);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      _mm512_storeu_ps(c4 + 0, vscaled4x0);
      _mm512_storeu_ps(c4 + 16, vscaled4x1);
      _mm512_storeu_ps(c4 + 32, vscaled4x2);
      _mm512_storeu_ps(c4 + 48, vscaled4x3);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm512_storeu_ps(c3 + 0, vscaled3x0);
      _mm512_storeu_ps(c3 + 16, vscaled3x1);
      _mm512_storeu_ps(c3 + 32, vscaled3x2);
      _mm512_storeu_ps(c3 + 48, vscaled3x3);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm512_storeu_ps(c2 + 0, vscaled2x0);
      _mm512_storeu_ps(c2 + 16, vscaled2x1);
      _mm512_storeu_ps(c2 + 32, vscaled2x2);
      _mm512_storeu_ps(c2 + 48, vscaled2x3);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm512_storeu_ps(c1 + 0, vscaled1x0);
      _mm512_storeu_ps(c1 + 16, vscaled1x1);
      _mm512_storeu_ps(c1 + 32, vscaled1x2);
      _mm512_storeu_ps(c1 + 48, vscaled1x3);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm512_storeu_ps(c0 + 0, vscaled0x0);
      _mm512_storeu_ps(c0 + 16, vscaled0x1);
      _mm512_storeu_ps(c0 + 32, vscaled0x2);
      _mm512_storeu_ps(c0 + 48, vscaled0x3);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a -= kc;
      nc -= 64;
    } else {
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 0) & 0xFFFF));
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 16) & 0xFFFF));
      const __mmask16 vmask2 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 32) & 0xFFFF));
      const __mmask16 vmask3 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 48) & 0xFFFF));
      _mm512_mask_storeu_ps(c15 + 0, vmask0, vscaled15x0);
      _mm512_mask_storeu_ps(c15 + 16, vmask1, vscaled15x1);
      _mm512_mask_storeu_ps(c15 + 32, vmask2, vscaled15x2);
      _mm512_mask_storeu_ps(c15 + 48, vmask3, vscaled15x3);
      _mm512_mask_storeu_ps(c14 + 0, vmask0, vscaled14x0);
      _mm512_mask_storeu_ps(c14 + 16, vmask1, vscaled14x1);
      _mm512_mask_storeu_ps(c14 + 32, vmask2, vscaled14x2);
      _mm512_mask_storeu_ps(c14 + 48, vmask3, vscaled14x3);
      _mm512_mask_storeu_ps(c13 + 0, vmask0, vscaled13x0);
      _mm512_mask_storeu_ps(c13 + 16, vmask1, vscaled13x1);
      _mm512_mask_storeu_ps(c13 + 32, vmask2, vscaled13x2);
      _mm512_mask_storeu_ps(c13 + 48, vmask3, vscaled13x3);
      _mm512_mask_storeu_ps(c12 + 0, vmask0, vscaled12x0);
      _mm512_mask_storeu_ps(c12 + 16, vmask1, vscaled12x1);
      _mm512_mask_storeu_ps(c12 + 32, vmask2, vscaled12x2);
      _mm512_mask_storeu_ps(c12 + 48, vmask3, vscaled12x3);
      _mm512_mask_storeu_ps(c11 + 0, vmask0, vscaled11x0);
      _mm512_mask_storeu_ps(c11 + 16, vmask1, vscaled11x1);
      _mm512_mask_storeu_ps(c11 + 32, vmask2, vscaled11x2);
      _mm512_mask_storeu_ps(c11 + 48, vmask3, vscaled11x3);
      _mm512_mask_storeu_ps(c10 + 0, vmask0, vscaled10x0);
      _mm512_mask_storeu_ps(c10 + 16, vmask1, vscaled10x1);
      _mm512_mask_storeu_ps(c10 + 32, vmask2, vscaled10x2);
      _mm512_mask_storeu_ps(c10 + 48, vmask3, vscaled10x3);
      _mm512_mask_storeu_ps(c9 + 0, vmask0, vscaled9x0);
      _mm512_mask_storeu_ps(c9 + 16, vmask1, vscaled9x1);
      _mm512_mask_storeu_ps(c9 + 32, vmask2, vscaled9x2);
      _mm512_mask_storeu_ps(c9 + 48, vmask3, vscaled9x3);
      _mm512_mask_storeu_ps(c8 + 0, vmask0, vscaled8x0);
      _mm512_mask_storeu_ps(c8 + 16, vmask1, vscaled8x1);
      _mm512_mask_storeu_ps(c8 + 32, vmask2, vscaled8x2);
      _mm512_mask_storeu_ps(c8 + 48, vmask3, vscaled8x3);
      _mm512_mask_storeu_ps(c7 + 0, vmask0, vscaled7x0);
      _mm512_mask_storeu_ps(c7 + 16, vmask1, vscaled7x1);
      _mm512_mask_storeu_ps(c7 + 32, vmask2, vscaled7x2);
      _mm512_mask_storeu_ps(c7 + 48, vmask3, vscaled7x3);
      _mm512_mask_storeu_ps(c6 + 0, vmask0, vscaled6x0);
      _mm512_mask_storeu_ps(c6 + 16, vmask1, vscaled6x1);
      _mm512_mask_storeu_ps(c6 + 32, vmask2, vscaled6x2);
      _mm512_mask_storeu_ps(c6 + 48, vmask3, vscaled6x3);
      _mm512_mask_storeu_ps(c5 + 0, vmask0, vscaled5x0);
      _mm512_mask_storeu_ps(c5 + 16, vmask1, vscaled5x1);
      _mm512_mask_storeu_ps(c5 + 32, vmask2, vscaled5x2);
      _mm512_mask_storeu_ps(c5 + 48, vmask3, vscaled5x3);
      _mm512_mask_storeu_ps(c4 + 0, vmask0, vscaled4x0);
      _mm512_mask_storeu_ps(c4 + 16, vmask1, vscaled4x1);
      _mm512_mask_storeu_ps(c4 + 32, vmask2, vscaled4x2);
      _mm512_mask_storeu_ps(c4 + 48, vmask3, vscaled4x3);
      _mm512_mask_storeu_ps(c3 + 0, vmask0, vscaled3x0);
      _mm512_mask_storeu_ps(c3 + 16, vmask1, vscaled3x1);
      _mm512_mask_storeu_ps(c3 + 32, vmask2, vscaled3x2);
      _mm512_mask_storeu_ps(c3 + 48, vmask3, vscaled3x3);
      _mm512_mask_storeu_ps(c2 + 0, vmask0, vscaled2x0);
      _mm512_mask_storeu_ps(c2 + 16, vmask1, vscaled2x1);
      _mm512_mask_storeu_ps(c2 + 32, vmask2, vscaled2x2);
      _mm512_mask_storeu_ps(c2 + 48, vmask3, vscaled2x3);
      _mm512_mask_storeu_ps(c1 + 0, vmask0, vscaled1x0);
      _mm512_mask_storeu_ps(c1 + 16, vmask1, vscaled1x1);
      _mm512_mask_storeu_ps(c1 + 32, vmask2, vscaled1x2);
      _mm512_mask_storeu_ps(c1 + 48, vmask3, vscaled1x3);
      _mm512_mask_storeu_ps(c0 + 0, vmask0, vscaled0x0);
      _mm512_mask_storeu_ps(c0 + 16, vmask1, vscaled0x1);
      _mm512_mask_storeu_ps(c0 + 32, vmask2, vscaled0x2);
      _mm512_mask_storeu_ps(c0 + 48, vmask3, vscaled0x3);
      nc = 0;
    }
  } while (nc != 0);
  // Release tile config
  _tile_release();
  #endif  // defined(__x86_64__)
}
