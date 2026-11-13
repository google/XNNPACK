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


void xnn_qd8_f16_qc2w_gemm_minmax_ukernel_16x64c4__avx512amx_prfm(
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
    const float* row_sum,
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
  assert(row_sum != NULL);

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
  XNN_ALIGN(64) uint8_t weight_buffer[4][16 * 64];
  kc = round_up_po2(kc, 4 * sizeof(int8_t));

  const size_t kremainder = (kc & 63) ? (kc & 63) : 64;
  const size_t kremainder_colsb = round_up_po2(kremainder, 4 * sizeof(int8_t));
  const __mmask64 kremainder_mask = _cvtu64_mask64((UINT64_C(1) << (kc & 63)) - 1);

  tile_data.palette_id = 1;
  tile_data.rows[0] = mr;              // tmm0 = res[0]
  tile_data.rows[1] = mr;              // tmm1 = res[1]
  tile_data.rows[2] = mr;              // tmm2 = res[2]
  tile_data.rows[3] = mr;              // tmm3 = res[3]
  tile_data.rows[4] = mr;              // tmm4 = input
  tile_data.rows[5] = 16;                    // tmm5 = weights
  tile_data.rows[6] = mr;              // tmm6 = input remainder
  tile_data.rows[7] = kremainder_colsb >> 2; // tmm7 = weights remainder
  tile_data.colsb[0] = 64;                   // tmm0 = res[0]
  tile_data.colsb[1] = 64;                   // tmm1 = res[1]
  tile_data.colsb[2] = 64;                   // tmm2 = res[2]
  tile_data.colsb[3] = 64;                   // tmm3 = res[3]
  tile_data.colsb[4] = 64;                   // tmm4 = input
  tile_data.colsb[5] = 64;                   // tmm5 = weights
  tile_data.colsb[6] = kremainder_colsb;     // tmm6 = input remainder
  tile_data.colsb[7] = 64;                   // tmm7 = weights remainder
  _tile_loadconfig(&tile_data);

  const __m512 voutput_min = _mm512_cvtph_ps(_mm256_set1_epi16(*(const uint16_t*) &params->scalar.min));
  const __m512 voutput_max = _mm512_cvtph_ps(_mm256_set1_epi16(*(const uint16_t*) &params->scalar.max));
  // XNN_FORCE_REALIZATION(voutput_min);
  // XNN_FORCE_REALIZATION(voutput_max);

  do {
    const int32_t* ksum = (const int32_t*) w;
    w = (const int32_t*) w + 64;
    const float* kzp = (const float*) w;
    w = (const float*) w + 64;

    // Zero tile accumulator
    _tile_zero(0);
    _tile_zero(1);
    _tile_zero(2);
    _tile_zero(3);

    size_t k = kc;
    const __m512i vmask = _mm512_set1_epi8(0x03);
    const __m512i vshr2 = _mm512_set1_epi64(0x0408000000000000);
    const __m512i vshr4 = _mm512_set1_epi64(0x1020000000000000);
    const __m512i vshr6 = _mm512_set1_epi64(0x4080000000000000);
    if XNN_LIKELY(k >= 64 * sizeof(int8_t)) {
      const __m512i vb0x0 = _mm512_load_epi32((const int8_t*) w + 0);
      const __m512i vb1x0 = _mm512_load_epi32((const int8_t*) w + 256);
      const __m512i vb2x0 = _mm512_load_epi32((const int8_t*) w + 512);
      const __m512i vb3x0 = _mm512_load_epi32((const int8_t*) w + 768);
      const __m512i vc0_0x0 = _mm512_and_si512(vb0x0, vmask);
      const __m512i vc1_0x0 = _mm512_gf2p8affine_epi64_epi8(vb0x0, vshr2, 0);
      const __m512i vc2_0x0 = _mm512_gf2p8affine_epi64_epi8(vb0x0, vshr4, 0);
      const __m512i vc3_0x0 = _mm512_gf2p8affine_epi64_epi8(vb0x0, vshr6, 0);
      _mm512_store_epi32(weight_buffer[0] + 0, vc0_0x0);
      _mm512_store_epi32(weight_buffer[0] + 64, vc1_0x0);
      _mm512_store_epi32(weight_buffer[0] + 128, vc2_0x0);
      _mm512_store_epi32(weight_buffer[0] + 192, vc3_0x0);
      const __m512i vc0_1x0 = _mm512_and_si512(vb1x0, vmask);
      const __m512i vc1_1x0 = _mm512_gf2p8affine_epi64_epi8(vb1x0, vshr2, 0);
      const __m512i vc2_1x0 = _mm512_gf2p8affine_epi64_epi8(vb1x0, vshr4, 0);
      const __m512i vc3_1x0 = _mm512_gf2p8affine_epi64_epi8(vb1x0, vshr6, 0);
      _mm512_store_epi32(weight_buffer[0] + 256, vc0_1x0);
      _mm512_store_epi32(weight_buffer[0] + 320, vc1_1x0);
      _mm512_store_epi32(weight_buffer[0] + 384, vc2_1x0);
      _mm512_store_epi32(weight_buffer[0] + 448, vc3_1x0);
      const __m512i vc0_2x0 = _mm512_and_si512(vb2x0, vmask);
      const __m512i vc1_2x0 = _mm512_gf2p8affine_epi64_epi8(vb2x0, vshr2, 0);
      const __m512i vc2_2x0 = _mm512_gf2p8affine_epi64_epi8(vb2x0, vshr4, 0);
      const __m512i vc3_2x0 = _mm512_gf2p8affine_epi64_epi8(vb2x0, vshr6, 0);
      _mm512_store_epi32(weight_buffer[0] + 512, vc0_2x0);
      _mm512_store_epi32(weight_buffer[0] + 576, vc1_2x0);
      _mm512_store_epi32(weight_buffer[0] + 640, vc2_2x0);
      _mm512_store_epi32(weight_buffer[0] + 704, vc3_2x0);
      const __m512i vc0_3x0 = _mm512_and_si512(vb3x0, vmask);
      const __m512i vc1_3x0 = _mm512_gf2p8affine_epi64_epi8(vb3x0, vshr2, 0);
      const __m512i vc2_3x0 = _mm512_gf2p8affine_epi64_epi8(vb3x0, vshr4, 0);
      const __m512i vc3_3x0 = _mm512_gf2p8affine_epi64_epi8(vb3x0, vshr6, 0);
      _mm512_store_epi32(weight_buffer[0] + 768, vc0_3x0);
      _mm512_store_epi32(weight_buffer[0] + 832, vc1_3x0);
      _mm512_store_epi32(weight_buffer[0] + 896, vc2_3x0);
      _mm512_store_epi32(weight_buffer[0] + 960, vc3_3x0);
    }
    while (k >= 64 * sizeof(int8_t)) {
      _tile_loadd(4, a, a_stride);
      a += 64;
      if XNN_LIKELY(nc > 16) {
        const __m512i vb0x1 = _mm512_load_epi32((const int8_t*) w + 64);
        const __m512i vb1x1 = _mm512_load_epi32((const int8_t*) w + 320);
        const __m512i vb2x1 = _mm512_load_epi32((const int8_t*) w + 576);
        const __m512i vb3x1 = _mm512_load_epi32((const int8_t*) w + 832);
        const __m512i vc0_0x1 = _mm512_and_si512(vb0x1, vmask);
        const __m512i vc1_0x1 = _mm512_gf2p8affine_epi64_epi8(vb0x1, vshr2, 0);
        const __m512i vc2_0x1 = _mm512_gf2p8affine_epi64_epi8(vb0x1, vshr4, 0);
        const __m512i vc3_0x1 = _mm512_gf2p8affine_epi64_epi8(vb0x1, vshr6, 0);
        _mm512_store_epi32(weight_buffer[1] + 0, vc0_0x1);
        _mm512_store_epi32(weight_buffer[1] + 64, vc1_0x1);
        _mm512_store_epi32(weight_buffer[1] + 128, vc2_0x1);
        _mm512_store_epi32(weight_buffer[1] + 192, vc3_0x1);
        const __m512i vc0_1x1 = _mm512_and_si512(vb1x1, vmask);
        const __m512i vc1_1x1 = _mm512_gf2p8affine_epi64_epi8(vb1x1, vshr2, 0);
        const __m512i vc2_1x1 = _mm512_gf2p8affine_epi64_epi8(vb1x1, vshr4, 0);
        const __m512i vc3_1x1 = _mm512_gf2p8affine_epi64_epi8(vb1x1, vshr6, 0);
        _mm512_store_epi32(weight_buffer[1] + 256, vc0_1x1);
        _mm512_store_epi32(weight_buffer[1] + 320, vc1_1x1);
        _mm512_store_epi32(weight_buffer[1] + 384, vc2_1x1);
        _mm512_store_epi32(weight_buffer[1] + 448, vc3_1x1);
        const __m512i vc0_2x1 = _mm512_and_si512(vb2x1, vmask);
        const __m512i vc1_2x1 = _mm512_gf2p8affine_epi64_epi8(vb2x1, vshr2, 0);
        const __m512i vc2_2x1 = _mm512_gf2p8affine_epi64_epi8(vb2x1, vshr4, 0);
        const __m512i vc3_2x1 = _mm512_gf2p8affine_epi64_epi8(vb2x1, vshr6, 0);
        _mm512_store_epi32(weight_buffer[1] + 512, vc0_2x1);
        _mm512_store_epi32(weight_buffer[1] + 576, vc1_2x1);
        _mm512_store_epi32(weight_buffer[1] + 640, vc2_2x1);
        _mm512_store_epi32(weight_buffer[1] + 704, vc3_2x1);
        const __m512i vc0_3x1 = _mm512_and_si512(vb3x1, vmask);
        const __m512i vc1_3x1 = _mm512_gf2p8affine_epi64_epi8(vb3x1, vshr2, 0);
        const __m512i vc2_3x1 = _mm512_gf2p8affine_epi64_epi8(vb3x1, vshr4, 0);
        const __m512i vc3_3x1 = _mm512_gf2p8affine_epi64_epi8(vb3x1, vshr6, 0);
        _mm512_store_epi32(weight_buffer[1] + 768, vc0_3x1);
        _mm512_store_epi32(weight_buffer[1] + 832, vc1_3x1);
        _mm512_store_epi32(weight_buffer[1] + 896, vc2_3x1);
        _mm512_store_epi32(weight_buffer[1] + 960, vc3_3x1);
      }
      _tile_loadd(5, weight_buffer[0], 64);
      _tile_dpbsud(0, 4, 5);
      if XNN_LIKELY(nc > 32) {
        const __m512i vb0x2 = _mm512_load_epi32((const int8_t*) w + 128);
        const __m512i vb1x2 = _mm512_load_epi32((const int8_t*) w + 384);
        const __m512i vb2x2 = _mm512_load_epi32((const int8_t*) w + 640);
        const __m512i vb3x2 = _mm512_load_epi32((const int8_t*) w + 896);
        const __m512i vc0_0x2 = _mm512_and_si512(vb0x2, vmask);
        const __m512i vc1_0x2 = _mm512_gf2p8affine_epi64_epi8(vb0x2, vshr2, 0);
        const __m512i vc2_0x2 = _mm512_gf2p8affine_epi64_epi8(vb0x2, vshr4, 0);
        const __m512i vc3_0x2 = _mm512_gf2p8affine_epi64_epi8(vb0x2, vshr6, 0);
        _mm512_store_epi32(weight_buffer[2] + 0, vc0_0x2);
        _mm512_store_epi32(weight_buffer[2] + 64, vc1_0x2);
        _mm512_store_epi32(weight_buffer[2] + 128, vc2_0x2);
        _mm512_store_epi32(weight_buffer[2] + 192, vc3_0x2);
        const __m512i vc0_1x2 = _mm512_and_si512(vb1x2, vmask);
        const __m512i vc1_1x2 = _mm512_gf2p8affine_epi64_epi8(vb1x2, vshr2, 0);
        const __m512i vc2_1x2 = _mm512_gf2p8affine_epi64_epi8(vb1x2, vshr4, 0);
        const __m512i vc3_1x2 = _mm512_gf2p8affine_epi64_epi8(vb1x2, vshr6, 0);
        _mm512_store_epi32(weight_buffer[2] + 256, vc0_1x2);
        _mm512_store_epi32(weight_buffer[2] + 320, vc1_1x2);
        _mm512_store_epi32(weight_buffer[2] + 384, vc2_1x2);
        _mm512_store_epi32(weight_buffer[2] + 448, vc3_1x2);
        const __m512i vc0_2x2 = _mm512_and_si512(vb2x2, vmask);
        const __m512i vc1_2x2 = _mm512_gf2p8affine_epi64_epi8(vb2x2, vshr2, 0);
        const __m512i vc2_2x2 = _mm512_gf2p8affine_epi64_epi8(vb2x2, vshr4, 0);
        const __m512i vc3_2x2 = _mm512_gf2p8affine_epi64_epi8(vb2x2, vshr6, 0);
        _mm512_store_epi32(weight_buffer[2] + 512, vc0_2x2);
        _mm512_store_epi32(weight_buffer[2] + 576, vc1_2x2);
        _mm512_store_epi32(weight_buffer[2] + 640, vc2_2x2);
        _mm512_store_epi32(weight_buffer[2] + 704, vc3_2x2);
        const __m512i vc0_3x2 = _mm512_and_si512(vb3x2, vmask);
        const __m512i vc1_3x2 = _mm512_gf2p8affine_epi64_epi8(vb3x2, vshr2, 0);
        const __m512i vc2_3x2 = _mm512_gf2p8affine_epi64_epi8(vb3x2, vshr4, 0);
        const __m512i vc3_3x2 = _mm512_gf2p8affine_epi64_epi8(vb3x2, vshr6, 0);
        _mm512_store_epi32(weight_buffer[2] + 768, vc0_3x2);
        _mm512_store_epi32(weight_buffer[2] + 832, vc1_3x2);
        _mm512_store_epi32(weight_buffer[2] + 896, vc2_3x2);
        _mm512_store_epi32(weight_buffer[2] + 960, vc3_3x2);
      }
      if XNN_LIKELY(nc > 16) {
        _tile_loadd(5, weight_buffer[1], 64);
        _tile_dpbsud(1, 4, 5);
      }
      if XNN_LIKELY(nc > 48) {
        const __m512i vb0x3 = _mm512_load_epi32((const int8_t*) w + 192);
        const __m512i vb1x3 = _mm512_load_epi32((const int8_t*) w + 448);
        const __m512i vb2x3 = _mm512_load_epi32((const int8_t*) w + 704);
        const __m512i vb3x3 = _mm512_load_epi32((const int8_t*) w + 960);
        const __m512i vc0_0x3 = _mm512_and_si512(vb0x3, vmask);
        const __m512i vc1_0x3 = _mm512_gf2p8affine_epi64_epi8(vb0x3, vshr2, 0);
        const __m512i vc2_0x3 = _mm512_gf2p8affine_epi64_epi8(vb0x3, vshr4, 0);
        const __m512i vc3_0x3 = _mm512_gf2p8affine_epi64_epi8(vb0x3, vshr6, 0);
        _mm512_store_epi32(weight_buffer[3] + 0, vc0_0x3);
        _mm512_store_epi32(weight_buffer[3] + 64, vc1_0x3);
        _mm512_store_epi32(weight_buffer[3] + 128, vc2_0x3);
        _mm512_store_epi32(weight_buffer[3] + 192, vc3_0x3);
        const __m512i vc0_1x3 = _mm512_and_si512(vb1x3, vmask);
        const __m512i vc1_1x3 = _mm512_gf2p8affine_epi64_epi8(vb1x3, vshr2, 0);
        const __m512i vc2_1x3 = _mm512_gf2p8affine_epi64_epi8(vb1x3, vshr4, 0);
        const __m512i vc3_1x3 = _mm512_gf2p8affine_epi64_epi8(vb1x3, vshr6, 0);
        _mm512_store_epi32(weight_buffer[3] + 256, vc0_1x3);
        _mm512_store_epi32(weight_buffer[3] + 320, vc1_1x3);
        _mm512_store_epi32(weight_buffer[3] + 384, vc2_1x3);
        _mm512_store_epi32(weight_buffer[3] + 448, vc3_1x3);
        const __m512i vc0_2x3 = _mm512_and_si512(vb2x3, vmask);
        const __m512i vc1_2x3 = _mm512_gf2p8affine_epi64_epi8(vb2x3, vshr2, 0);
        const __m512i vc2_2x3 = _mm512_gf2p8affine_epi64_epi8(vb2x3, vshr4, 0);
        const __m512i vc3_2x3 = _mm512_gf2p8affine_epi64_epi8(vb2x3, vshr6, 0);
        _mm512_store_epi32(weight_buffer[3] + 512, vc0_2x3);
        _mm512_store_epi32(weight_buffer[3] + 576, vc1_2x3);
        _mm512_store_epi32(weight_buffer[3] + 640, vc2_2x3);
        _mm512_store_epi32(weight_buffer[3] + 704, vc3_2x3);
        const __m512i vc0_3x3 = _mm512_and_si512(vb3x3, vmask);
        const __m512i vc1_3x3 = _mm512_gf2p8affine_epi64_epi8(vb3x3, vshr2, 0);
        const __m512i vc2_3x3 = _mm512_gf2p8affine_epi64_epi8(vb3x3, vshr4, 0);
        const __m512i vc3_3x3 = _mm512_gf2p8affine_epi64_epi8(vb3x3, vshr6, 0);
        _mm512_store_epi32(weight_buffer[3] + 768, vc0_3x3);
        _mm512_store_epi32(weight_buffer[3] + 832, vc1_3x3);
        _mm512_store_epi32(weight_buffer[3] + 896, vc2_3x3);
        _mm512_store_epi32(weight_buffer[3] + 960, vc3_3x3);
      }
      if XNN_LIKELY(nc > 32) {
        _tile_loadd(5, weight_buffer[2], 64);
        _tile_dpbsud(2, 4, 5);
      }
      xnn_prefetch_to_l1((const int8_t*) w + 2048);
      xnn_prefetch_to_l1((const int8_t*) w + 2112);
      xnn_prefetch_to_l1((const int8_t*) w + 2176);
      xnn_prefetch_to_l1((const int8_t*) w + 2240);
      xnn_prefetch_to_l1((const int8_t*) w + 2304);
      xnn_prefetch_to_l1((const int8_t*) w + 2368);
      xnn_prefetch_to_l1((const int8_t*) w + 2432);
      xnn_prefetch_to_l1((const int8_t*) w + 2496);
      xnn_prefetch_to_l1((const int8_t*) w + 2560);
      xnn_prefetch_to_l1((const int8_t*) w + 2624);
      xnn_prefetch_to_l1((const int8_t*) w + 2688);
      xnn_prefetch_to_l1((const int8_t*) w + 2752);
      xnn_prefetch_to_l1((const int8_t*) w + 2816);
      xnn_prefetch_to_l1((const int8_t*) w + 2880);
      xnn_prefetch_to_l1((const int8_t*) w + 2944);
      xnn_prefetch_to_l1((const int8_t*) w + 3008);
      w = (const int8_t*) w + 1024;
      k -= 64 * sizeof(int8_t);
      if XNN_LIKELY(k >= 64 * sizeof(int8_t)) {
        const __m512i vb0x0 = _mm512_load_epi32((const int8_t*) w + 0);
        const __m512i vb1x0 = _mm512_load_epi32((const int8_t*) w + 256);
        const __m512i vb2x0 = _mm512_load_epi32((const int8_t*) w + 512);
        const __m512i vb3x0 = _mm512_load_epi32((const int8_t*) w + 768);
        const __m512i vc0_0x0 = _mm512_and_si512(vb0x0, vmask);
        const __m512i vc1_0x0 = _mm512_gf2p8affine_epi64_epi8(vb0x0, vshr2, 0);
        const __m512i vc2_0x0 = _mm512_gf2p8affine_epi64_epi8(vb0x0, vshr4, 0);
        const __m512i vc3_0x0 = _mm512_gf2p8affine_epi64_epi8(vb0x0, vshr6, 0);
        _mm512_store_epi32(weight_buffer[0] + 0, vc0_0x0);
        _mm512_store_epi32(weight_buffer[0] + 64, vc1_0x0);
        _mm512_store_epi32(weight_buffer[0] + 128, vc2_0x0);
        _mm512_store_epi32(weight_buffer[0] + 192, vc3_0x0);
        const __m512i vc0_1x0 = _mm512_and_si512(vb1x0, vmask);
        const __m512i vc1_1x0 = _mm512_gf2p8affine_epi64_epi8(vb1x0, vshr2, 0);
        const __m512i vc2_1x0 = _mm512_gf2p8affine_epi64_epi8(vb1x0, vshr4, 0);
        const __m512i vc3_1x0 = _mm512_gf2p8affine_epi64_epi8(vb1x0, vshr6, 0);
        _mm512_store_epi32(weight_buffer[0] + 256, vc0_1x0);
        _mm512_store_epi32(weight_buffer[0] + 320, vc1_1x0);
        _mm512_store_epi32(weight_buffer[0] + 384, vc2_1x0);
        _mm512_store_epi32(weight_buffer[0] + 448, vc3_1x0);
        const __m512i vc0_2x0 = _mm512_and_si512(vb2x0, vmask);
        const __m512i vc1_2x0 = _mm512_gf2p8affine_epi64_epi8(vb2x0, vshr2, 0);
        const __m512i vc2_2x0 = _mm512_gf2p8affine_epi64_epi8(vb2x0, vshr4, 0);
        const __m512i vc3_2x0 = _mm512_gf2p8affine_epi64_epi8(vb2x0, vshr6, 0);
        _mm512_store_epi32(weight_buffer[0] + 512, vc0_2x0);
        _mm512_store_epi32(weight_buffer[0] + 576, vc1_2x0);
        _mm512_store_epi32(weight_buffer[0] + 640, vc2_2x0);
        _mm512_store_epi32(weight_buffer[0] + 704, vc3_2x0);
        const __m512i vc0_3x0 = _mm512_and_si512(vb3x0, vmask);
        const __m512i vc1_3x0 = _mm512_gf2p8affine_epi64_epi8(vb3x0, vshr2, 0);
        const __m512i vc2_3x0 = _mm512_gf2p8affine_epi64_epi8(vb3x0, vshr4, 0);
        const __m512i vc3_3x0 = _mm512_gf2p8affine_epi64_epi8(vb3x0, vshr6, 0);
        _mm512_store_epi32(weight_buffer[0] + 768, vc0_3x0);
        _mm512_store_epi32(weight_buffer[0] + 832, vc1_3x0);
        _mm512_store_epi32(weight_buffer[0] + 896, vc2_3x0);
        _mm512_store_epi32(weight_buffer[0] + 960, vc3_3x0);
      }
      if XNN_LIKELY(nc > 48) {
        _tile_loadd(5, weight_buffer[3], 64);
        _tile_dpbsud(3, 4, 5);
      }
    }

    if XNN_UNLIKELY(k != 0) {
      const int8_t* a0 = a;
      for (size_t m = 0; m < mr; ++m) {
        const __m512i vin = _mm512_maskz_loadu_epi8(kremainder_mask, a0);
        a0 = (const int8_t*) ((uintptr_t) a0 + a_stride);
        _mm512_store_epi32(&res[0][m * 16], vin);
      }
      _tile_loadd(6, &res[0][0], 64);
      a += kremainder;
      for (size_t k_blk = 0; k_blk < ((kremainder_colsb + 15) >> 4); ++k_blk) {
        const __m512i vb0 = _mm512_load_epi32((const int8_t*) w + 256 * k_blk);
        const __m512i vc0_0 = _mm512_and_si512(vb0, vmask);
        const __m512i vc1_0 = _mm512_gf2p8affine_epi64_epi8(vb0, vshr2, 0);
        const __m512i vc2_0 = _mm512_gf2p8affine_epi64_epi8(vb0, vshr4, 0);
        const __m512i vc3_0 = _mm512_gf2p8affine_epi64_epi8(vb0, vshr6, 0);
        _mm512_store_epi32(weight_buffer[0] + 256 * k_blk + 0, vc0_0);
        _mm512_store_epi32(weight_buffer[0] + 256 * k_blk + 64, vc1_0);
        _mm512_store_epi32(weight_buffer[0] + 256 * k_blk + 128, vc2_0);
        _mm512_store_epi32(weight_buffer[0] + 256 * k_blk + 192, vc3_0);
      }
      _tile_loadd(7, weight_buffer[0], 64);
      _tile_dpbsud(0, 6, 7);
      if XNN_LIKELY(nc > 16) {
        for (size_t k_blk = 0; k_blk < ((kremainder_colsb + 15) >> 4); ++k_blk) {
          const __m512i vb1 = _mm512_load_epi32((const int8_t*) w + 64 + 256 * k_blk);
          const __m512i vc0_1 = _mm512_and_si512(vb1, vmask);
          const __m512i vc1_1 = _mm512_gf2p8affine_epi64_epi8(vb1, vshr2, 0);
          const __m512i vc2_1 = _mm512_gf2p8affine_epi64_epi8(vb1, vshr4, 0);
          const __m512i vc3_1 = _mm512_gf2p8affine_epi64_epi8(vb1, vshr6, 0);
          _mm512_store_epi32(weight_buffer[1] + 256 * k_blk + 0, vc0_1);
          _mm512_store_epi32(weight_buffer[1] + 256 * k_blk + 64, vc1_1);
          _mm512_store_epi32(weight_buffer[1] + 256 * k_blk + 128, vc2_1);
          _mm512_store_epi32(weight_buffer[1] + 256 * k_blk + 192, vc3_1);
        }
        _tile_loadd(7, weight_buffer[1], 64);
        _tile_dpbsud(1, 6, 7);
      }
      if XNN_LIKELY(nc > 32) {
        for (size_t k_blk = 0; k_blk < ((kremainder_colsb + 15) >> 4); ++k_blk) {
          const __m512i vb2 = _mm512_load_epi32((const int8_t*) w + 128 + 256 * k_blk);
          const __m512i vc0_2 = _mm512_and_si512(vb2, vmask);
          const __m512i vc1_2 = _mm512_gf2p8affine_epi64_epi8(vb2, vshr2, 0);
          const __m512i vc2_2 = _mm512_gf2p8affine_epi64_epi8(vb2, vshr4, 0);
          const __m512i vc3_2 = _mm512_gf2p8affine_epi64_epi8(vb2, vshr6, 0);
          _mm512_store_epi32(weight_buffer[2] + 256 * k_blk + 0, vc0_2);
          _mm512_store_epi32(weight_buffer[2] + 256 * k_blk + 64, vc1_2);
          _mm512_store_epi32(weight_buffer[2] + 256 * k_blk + 128, vc2_2);
          _mm512_store_epi32(weight_buffer[2] + 256 * k_blk + 192, vc3_2);
        }
        _tile_loadd(7, weight_buffer[2], 64);
        _tile_dpbsud(2, 6, 7);
      }
      if XNN_LIKELY(nc > 48) {
        for (size_t k_blk = 0; k_blk < ((kremainder_colsb + 15) >> 4); ++k_blk) {
          const __m512i vb3 = _mm512_load_epi32((const int8_t*) w + 192 + 256 * k_blk);
          const __m512i vc0_3 = _mm512_and_si512(vb3, vmask);
          const __m512i vc1_3 = _mm512_gf2p8affine_epi64_epi8(vb3, vshr2, 0);
          const __m512i vc2_3 = _mm512_gf2p8affine_epi64_epi8(vb3, vshr4, 0);
          const __m512i vc3_3 = _mm512_gf2p8affine_epi64_epi8(vb3, vshr6, 0);
          _mm512_store_epi32(weight_buffer[3] + 256 * k_blk + 0, vc0_3);
          _mm512_store_epi32(weight_buffer[3] + 256 * k_blk + 64, vc1_3);
          _mm512_store_epi32(weight_buffer[3] + 256 * k_blk + 128, vc2_3);
          _mm512_store_epi32(weight_buffer[3] + 256 * k_blk + 192, vc3_3);
        }
        _tile_loadd(7, weight_buffer[3], 64);
        _tile_dpbsud(3, 6, 7);
      }

      w = (const int8_t*) w + ((kremainder_colsb + 15) >> 4) * 256;
      k -= kremainder * sizeof(int8_t);
    }

    // TODO: Instead of processing up to 4 tiles (16x64) consider
    // quantizing 1 tile at a time (16 registers)
    _tile_stored(0, &res[0][0], 64);
    if XNN_LIKELY(nc > 16) {
      _tile_stored(1, &res[1][0], 64);
    }
    if XNN_LIKELY(nc > 32) {
      _tile_stored(2, &res[2][0], 64);
    }
    if XNN_LIKELY(nc > 48) {
      _tile_stored(3, &res[3][0], 64);
    }

    // TODO: Fix msan for AMX
    #if defined(__has_feature)
      #if __has_feature(memory_sanitizer)
        __msan_unpoison(res, sizeof(res));
      #endif
    #endif

    // Quantize 1 row at a time across all tiles.
    const __m512 vtwo = _mm512_set1_ps(2.0f);
    const __m512i vksum0 = _mm512_load_epi32(ksum + 0);
    const __m512 rh_zero_points_0 = _mm512_load_ps(kzp + 0);
    const __m512 biased_rh_zero_points_0 = _mm512_add_ps(rh_zero_points_0, vtwo);
    const __m512i vksum1 = _mm512_load_epi32(ksum + 16);
    const __m512 rh_zero_points_1 = _mm512_load_ps(kzp + 16);
    const __m512 biased_rh_zero_points_1 = _mm512_add_ps(rh_zero_points_1, vtwo);
    const __m512i vksum2 = _mm512_load_epi32(ksum + 32);
    const __m512 rh_zero_points_2 = _mm512_load_ps(kzp + 32);
    const __m512 biased_rh_zero_points_2 = _mm512_add_ps(rh_zero_points_2, vtwo);
    const __m512i vksum3 = _mm512_load_epi32(ksum + 48);
    const __m512 rh_zero_points_3 = _mm512_load_ps(kzp + 48);
    const __m512 biased_rh_zero_points_3 = _mm512_add_ps(rh_zero_points_3, vtwo);
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

    xnn_float16* c0 = c;
    for (size_t m = 0; m < mr; ++m)
    {
      const __m512i vzero_point = _mm512_set1_epi32((int) quantization_params[m].zero_point);
      const __m512 vinv_scale = _mm512_set1_ps(quantization_params[m].inv_scale);
      const __m512 vlh_row_sum = _mm512_set1_ps(row_sum[m]);
      const __m512 vscaled_lh_zero_point = _mm512_set1_ps((float) kc * quantization_params[m].zero_point);
      if XNN_LIKELY(nc >= 64) {
        __m512i vacc0 = _mm512_mullo_epi32(vksum0, vzero_point);
        __m512i vacc1 = _mm512_mullo_epi32(vksum1, vzero_point);
        __m512i vacc2 = _mm512_mullo_epi32(vksum2, vzero_point);
        __m512i vacc3 = _mm512_mullo_epi32(vksum3, vzero_point);
        // Add tile to bias
        vacc0 = _mm512_add_epi32(vacc0, _mm512_load_epi32(&res[0][0] + m * 16));
        vacc1 = _mm512_add_epi32(vacc1, _mm512_load_epi32(&res[1][0] + m * 16));
        vacc2 = _mm512_add_epi32(vacc2, _mm512_load_epi32(&res[2][0] + m * 16));
        vacc3 = _mm512_add_epi32(vacc3, _mm512_load_epi32(&res[3][0] + m * 16));

        __m512 vscaled0 = _mm512_cvtepi32_ps(vacc0);
        __m512 vscaled1 = _mm512_cvtepi32_ps(vacc1);
        __m512 vscaled2 = _mm512_cvtepi32_ps(vacc2);
        __m512 vscaled3 = _mm512_cvtepi32_ps(vacc3);

        vscaled0 = _mm512_fnmadd_ps(biased_rh_zero_points_0, vlh_row_sum, vscaled0);
        vscaled1 = _mm512_fnmadd_ps(biased_rh_zero_points_1, vlh_row_sum, vscaled1);
        vscaled2 = _mm512_fnmadd_ps(biased_rh_zero_points_2, vlh_row_sum, vscaled2);
        vscaled3 = _mm512_fnmadd_ps(biased_rh_zero_points_3, vlh_row_sum, vscaled3);
        vscaled0 = _mm512_fmadd_ps(rh_zero_points_0, vscaled_lh_zero_point, vscaled0);
        vscaled1 = _mm512_fmadd_ps(rh_zero_points_1, vscaled_lh_zero_point, vscaled1);
        vscaled2 = _mm512_fmadd_ps(rh_zero_points_2, vscaled_lh_zero_point, vscaled2);
        vscaled3 = _mm512_fmadd_ps(rh_zero_points_3, vscaled_lh_zero_point, vscaled3);
        vscaled0 = _mm512_mul_ps(vscaled0, vinv_scale);
        vscaled1 = _mm512_mul_ps(vscaled1, vinv_scale);
        vscaled2 = _mm512_mul_ps(vscaled2, vinv_scale);
        vscaled3 = _mm512_mul_ps(vscaled3, vinv_scale);
        vscaled0 = _mm512_fmadd_ps(vscaled0, vfilter_output_scale0, vbias0);
        vscaled1 = _mm512_fmadd_ps(vscaled1, vfilter_output_scale1, vbias1);
        vscaled2 = _mm512_fmadd_ps(vscaled2, vfilter_output_scale2, vbias2);
        vscaled3 = _mm512_fmadd_ps(vscaled3, vfilter_output_scale3, vbias3);
        vscaled0 = _mm512_max_ps(vscaled0, voutput_min);
        vscaled1 = _mm512_max_ps(vscaled1, voutput_min);
        vscaled2 = _mm512_max_ps(vscaled2, voutput_min);
        vscaled3 = _mm512_max_ps(vscaled3, voutput_min);
        vscaled0 = _mm512_min_ps(vscaled0, voutput_max);
        vscaled1 = _mm512_min_ps(vscaled1, voutput_max);
        vscaled2 = _mm512_min_ps(vscaled2, voutput_max);
        vscaled3 = _mm512_min_ps(vscaled3, voutput_max);

        __m256i vfp16out0 = _mm512_cvtps_ph(vscaled0, _MM_FROUND_TO_NEAREST_INT);
        __m256i vfp16out1 = _mm512_cvtps_ph(vscaled1, _MM_FROUND_TO_NEAREST_INT);
        __m256i vfp16out2 = _mm512_cvtps_ph(vscaled2, _MM_FROUND_TO_NEAREST_INT);
        __m256i vfp16out3 = _mm512_cvtps_ph(vscaled3, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i*) (c0 + 0), vfp16out0);
        _mm256_storeu_si256((__m256i*) (c0 + 16), vfp16out1);
        _mm256_storeu_si256((__m256i*) (c0 + 32), vfp16out2);
        _mm256_storeu_si256((__m256i*) (c0 + 48), vfp16out3);
      } else {
        __m512i vacc0 = _mm512_mullo_epi32(vksum0, vzero_point);
        vacc0 = _mm512_add_epi32(vacc0, _mm512_load_epi32(&res[0][0] + m * 16));
        __m512 vscaled0 = _mm512_cvtepi32_ps(vacc0);
        vscaled0 = _mm512_fnmadd_ps(biased_rh_zero_points_0, vlh_row_sum, vscaled0);
        vscaled0 = _mm512_fmadd_ps(rh_zero_points_0, vscaled_lh_zero_point, vscaled0);
        vscaled0 = _mm512_mul_ps(vscaled0, vinv_scale);
        vscaled0 = _mm512_fmadd_ps(vscaled0, vfilter_output_scale0, vbias0);
        vscaled0 = _mm512_max_ps(vscaled0, voutput_min);
        vscaled0 = _mm512_min_ps(vscaled0, voutput_max);
        const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 0) & 0xFFFF));
        __m256i vfp16out0 = _mm512_cvtps_ph(vscaled0, _MM_FROUND_TO_NEAREST_INT);
        _mm256_mask_storeu_epi16(c0 + 0, vmask0, vfp16out0);
        if (nc > 16) {
        __m512i vacc1 = _mm512_mullo_epi32(vksum1, vzero_point);
        vacc1 = _mm512_add_epi32(vacc1, _mm512_load_epi32(&res[1][0] + m * 16));
        __m512 vscaled1 = _mm512_cvtepi32_ps(vacc1);
        vscaled1 = _mm512_fnmadd_ps(biased_rh_zero_points_1, vlh_row_sum, vscaled1);
        vscaled1 = _mm512_fmadd_ps(rh_zero_points_1, vscaled_lh_zero_point, vscaled1);
        vscaled1 = _mm512_mul_ps(vscaled1, vinv_scale);
        vscaled1 = _mm512_fmadd_ps(vscaled1, vfilter_output_scale1, vbias1);
        vscaled1 = _mm512_max_ps(vscaled1, voutput_min);
        vscaled1 = _mm512_min_ps(vscaled1, voutput_max);
        const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 16) & 0xFFFF));
        __m256i vfp16out1 = _mm512_cvtps_ph(vscaled1, _MM_FROUND_TO_NEAREST_INT);
        _mm256_mask_storeu_epi16(c0 + 16, vmask1, vfp16out1);
        }
        if (nc > 32) {
        __m512i vacc2 = _mm512_mullo_epi32(vksum2, vzero_point);
        vacc2 = _mm512_add_epi32(vacc2, _mm512_load_epi32(&res[2][0] + m * 16));
        __m512 vscaled2 = _mm512_cvtepi32_ps(vacc2);
        vscaled2 = _mm512_fnmadd_ps(biased_rh_zero_points_2, vlh_row_sum, vscaled2);
        vscaled2 = _mm512_fmadd_ps(rh_zero_points_2, vscaled_lh_zero_point, vscaled2);
        vscaled2 = _mm512_mul_ps(vscaled2, vinv_scale);
        vscaled2 = _mm512_fmadd_ps(vscaled2, vfilter_output_scale2, vbias2);
        vscaled2 = _mm512_max_ps(vscaled2, voutput_min);
        vscaled2 = _mm512_min_ps(vscaled2, voutput_max);
        const __mmask16 vmask2 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 32) & 0xFFFF));
        __m256i vfp16out2 = _mm512_cvtps_ph(vscaled2, _MM_FROUND_TO_NEAREST_INT);
        _mm256_mask_storeu_epi16(c0 + 32, vmask2, vfp16out2);
        }
        if (nc > 48) {
        __m512i vacc3 = _mm512_mullo_epi32(vksum3, vzero_point);
        vacc3 = _mm512_add_epi32(vacc3, _mm512_load_epi32(&res[3][0] + m * 16));
        __m512 vscaled3 = _mm512_cvtepi32_ps(vacc3);
        vscaled3 = _mm512_fnmadd_ps(biased_rh_zero_points_3, vlh_row_sum, vscaled3);
        vscaled3 = _mm512_fmadd_ps(rh_zero_points_3, vscaled_lh_zero_point, vscaled3);
        vscaled3 = _mm512_mul_ps(vscaled3, vinv_scale);
        vscaled3 = _mm512_fmadd_ps(vscaled3, vfilter_output_scale3, vbias3);
        vscaled3 = _mm512_max_ps(vscaled3, voutput_min);
        vscaled3 = _mm512_min_ps(vscaled3, voutput_max);
        const __mmask16 vmask3 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 48) & 0xFFFF));
        __m256i vfp16out3 = _mm512_cvtps_ph(vscaled3, _MM_FROUND_TO_NEAREST_INT);
        _mm256_mask_storeu_epi16(c0 + 48, vmask3, vfp16out3);
        }
      }
      c0 = (xnn_float16*) ((uintptr_t) c0 + cm_stride);
    }
    if XNN_LIKELY(nc >= 64) {
      c = (xnn_float16*) ((uintptr_t) c + cn_stride);
      a -= kc;
      nc -= 64;
    } else {
      nc = 0;
    }
  } while (nc != 0);
  // Release tile config
  _tile_release();
  #endif  // defined(__x86_64__)
}
