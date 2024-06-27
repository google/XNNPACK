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


void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x64c4__avx512amx(
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
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 7);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

// TODO: amxintrin.h only provide intrinsics for __x86_64__
// Update if amxintrin changes
#if defined(__x86_64__)
  __attribute__((aligned(64))) int32_t vintile[7 * 16];
  __attribute__((aligned(64))) int32_t res0[7 * 16];
  __attribute__((aligned(64))) int32_t res1[7 * 16];
  __attribute__((aligned(64))) int32_t res2[7 * 16];
  __attribute__((aligned(64))) int32_t res3[7 * 16];

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

  const __m512 voutput_max_less_zero_point = _mm512_set1_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_set1_epi32(params->fp32_avx512.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx512.output_min);

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
          a6 += 64;
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

      p -= 7 * sizeof(void*);
    } while (p != 0);

    // Add tile to bias
    _tile_stored(0, res0, 64);
    _tile_stored(1, res1, 64);
    _tile_stored(2, res2, 64);
    _tile_stored(3, res3, 64);

    __m512i vacc0x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 0));
    __m512i vacc0xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 0));
    __m512i vacc0xWXYZabcdefghijkl = _mm512_add_epi32(vksumWXYZabcdefghijkl, _mm512_load_epi32(res2 + 0));
    __m512i vacc0xmnopqrstuvwxyz01 = _mm512_add_epi32(vksummnopqrstuvwxyz01, _mm512_load_epi32(res3 + 0));
    __m512i vacc1x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 16));
    __m512i vacc1xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 16));
    __m512i vacc1xWXYZabcdefghijkl = _mm512_add_epi32(vksumWXYZabcdefghijkl, _mm512_load_epi32(res2 + 16));
    __m512i vacc1xmnopqrstuvwxyz01 = _mm512_add_epi32(vksummnopqrstuvwxyz01, _mm512_load_epi32(res3 + 16));
    __m512i vacc2x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 32));
    __m512i vacc2xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 32));
    __m512i vacc2xWXYZabcdefghijkl = _mm512_add_epi32(vksumWXYZabcdefghijkl, _mm512_load_epi32(res2 + 32));
    __m512i vacc2xmnopqrstuvwxyz01 = _mm512_add_epi32(vksummnopqrstuvwxyz01, _mm512_load_epi32(res3 + 32));
    __m512i vacc3x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 48));
    __m512i vacc3xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 48));
    __m512i vacc3xWXYZabcdefghijkl = _mm512_add_epi32(vksumWXYZabcdefghijkl, _mm512_load_epi32(res2 + 48));
    __m512i vacc3xmnopqrstuvwxyz01 = _mm512_add_epi32(vksummnopqrstuvwxyz01, _mm512_load_epi32(res3 + 48));
    __m512i vacc4x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 64));
    __m512i vacc4xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 64));
    __m512i vacc4xWXYZabcdefghijkl = _mm512_add_epi32(vksumWXYZabcdefghijkl, _mm512_load_epi32(res2 + 64));
    __m512i vacc4xmnopqrstuvwxyz01 = _mm512_add_epi32(vksummnopqrstuvwxyz01, _mm512_load_epi32(res3 + 64));
    __m512i vacc5x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 80));
    __m512i vacc5xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 80));
    __m512i vacc5xWXYZabcdefghijkl = _mm512_add_epi32(vksumWXYZabcdefghijkl, _mm512_load_epi32(res2 + 80));
    __m512i vacc5xmnopqrstuvwxyz01 = _mm512_add_epi32(vksummnopqrstuvwxyz01, _mm512_load_epi32(res3 + 80));
    __m512i vacc6x0123456789ABCDEF = _mm512_add_epi32(vksum0123456789ABCDEF, _mm512_load_epi32(res0 + 96));
    __m512i vacc6xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 96));
    __m512i vacc6xWXYZabcdefghijkl = _mm512_add_epi32(vksumWXYZabcdefghijkl, _mm512_load_epi32(res2 + 96));
    __m512i vacc6xmnopqrstuvwxyz01 = _mm512_add_epi32(vksummnopqrstuvwxyz01, _mm512_load_epi32(res3 + 96));

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

    const __m512 vscale0123456789ABCDEF = _mm512_loadu_ps((const float*) w + 0);
    const __m512 vscaleGHIJKLMNOPQRSTUV = _mm512_loadu_ps((const float*) w + 16);
    const __m512 vscaleWXYZabcdefghijkl = _mm512_loadu_ps((const float*) w + 32);
    const __m512 vscalemnopqrstuvwxyz01 = _mm512_loadu_ps((const float*) w + 48);
    w = (const int32_t*) w + 64;

    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled0xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled0xWXYZabcdefghijkl = _mm512_mul_ps(vscaled0xWXYZabcdefghijkl, vscaleWXYZabcdefghijkl);
    vscaled0xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled0xmnopqrstuvwxyz01, vscalemnopqrstuvwxyz01);
    vscaled1x0123456789ABCDEF = _mm512_mul_ps(vscaled1x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled1xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled1xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled1xWXYZabcdefghijkl = _mm512_mul_ps(vscaled1xWXYZabcdefghijkl, vscaleWXYZabcdefghijkl);
    vscaled1xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled1xmnopqrstuvwxyz01, vscalemnopqrstuvwxyz01);
    vscaled2x0123456789ABCDEF = _mm512_mul_ps(vscaled2x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled2xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled2xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled2xWXYZabcdefghijkl = _mm512_mul_ps(vscaled2xWXYZabcdefghijkl, vscaleWXYZabcdefghijkl);
    vscaled2xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled2xmnopqrstuvwxyz01, vscalemnopqrstuvwxyz01);
    vscaled3x0123456789ABCDEF = _mm512_mul_ps(vscaled3x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled3xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled3xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled3xWXYZabcdefghijkl = _mm512_mul_ps(vscaled3xWXYZabcdefghijkl, vscaleWXYZabcdefghijkl);
    vscaled3xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled3xmnopqrstuvwxyz01, vscalemnopqrstuvwxyz01);
    vscaled4x0123456789ABCDEF = _mm512_mul_ps(vscaled4x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled4xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled4xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled4xWXYZabcdefghijkl = _mm512_mul_ps(vscaled4xWXYZabcdefghijkl, vscaleWXYZabcdefghijkl);
    vscaled4xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled4xmnopqrstuvwxyz01, vscalemnopqrstuvwxyz01);
    vscaled5x0123456789ABCDEF = _mm512_mul_ps(vscaled5x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled5xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled5xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled5xWXYZabcdefghijkl = _mm512_mul_ps(vscaled5xWXYZabcdefghijkl, vscaleWXYZabcdefghijkl);
    vscaled5xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled5xmnopqrstuvwxyz01, vscalemnopqrstuvwxyz01);
    vscaled6x0123456789ABCDEF = _mm512_mul_ps(vscaled6x0123456789ABCDEF, vscale0123456789ABCDEF);
    vscaled6xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled6xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vscaled6xWXYZabcdefghijkl = _mm512_mul_ps(vscaled6xWXYZabcdefghijkl, vscaleWXYZabcdefghijkl);
    vscaled6xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled6xmnopqrstuvwxyz01, vscalemnopqrstuvwxyz01);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled0xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled0xWXYZabcdefghijkl = _mm512_min_ps(vscaled0xWXYZabcdefghijkl, voutput_max_less_zero_point);
    vscaled0xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled0xmnopqrstuvwxyz01, voutput_max_less_zero_point);
    vscaled1x0123456789ABCDEF = _mm512_min_ps(vscaled1x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled1xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled1xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled1xWXYZabcdefghijkl = _mm512_min_ps(vscaled1xWXYZabcdefghijkl, voutput_max_less_zero_point);
    vscaled1xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled1xmnopqrstuvwxyz01, voutput_max_less_zero_point);
    vscaled2x0123456789ABCDEF = _mm512_min_ps(vscaled2x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled2xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled2xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled2xWXYZabcdefghijkl = _mm512_min_ps(vscaled2xWXYZabcdefghijkl, voutput_max_less_zero_point);
    vscaled2xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled2xmnopqrstuvwxyz01, voutput_max_less_zero_point);
    vscaled3x0123456789ABCDEF = _mm512_min_ps(vscaled3x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled3xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled3xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled3xWXYZabcdefghijkl = _mm512_min_ps(vscaled3xWXYZabcdefghijkl, voutput_max_less_zero_point);
    vscaled3xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled3xmnopqrstuvwxyz01, voutput_max_less_zero_point);
    vscaled4x0123456789ABCDEF = _mm512_min_ps(vscaled4x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled4xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled4xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled4xWXYZabcdefghijkl = _mm512_min_ps(vscaled4xWXYZabcdefghijkl, voutput_max_less_zero_point);
    vscaled4xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled4xmnopqrstuvwxyz01, voutput_max_less_zero_point);
    vscaled5x0123456789ABCDEF = _mm512_min_ps(vscaled5x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled5xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled5xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled5xWXYZabcdefghijkl = _mm512_min_ps(vscaled5xWXYZabcdefghijkl, voutput_max_less_zero_point);
    vscaled5xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled5xmnopqrstuvwxyz01, voutput_max_less_zero_point);
    vscaled6x0123456789ABCDEF = _mm512_min_ps(vscaled6x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled6xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled6xGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);
    vscaled6xWXYZabcdefghijkl = _mm512_min_ps(vscaled6xWXYZabcdefghijkl, voutput_max_less_zero_point);
    vscaled6xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled6xmnopqrstuvwxyz01, voutput_max_less_zero_point);

    vacc0x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0x0123456789ABCDEF);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled0xGHIJKLMNOPQRSTUV);
    vacc0xWXYZabcdefghijkl = _mm512_cvtps_epi32(vscaled0xWXYZabcdefghijkl);
    vacc0xmnopqrstuvwxyz01 = _mm512_cvtps_epi32(vscaled0xmnopqrstuvwxyz01);
    vacc1x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled1x0123456789ABCDEF);
    vacc1xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled1xGHIJKLMNOPQRSTUV);
    vacc1xWXYZabcdefghijkl = _mm512_cvtps_epi32(vscaled1xWXYZabcdefghijkl);
    vacc1xmnopqrstuvwxyz01 = _mm512_cvtps_epi32(vscaled1xmnopqrstuvwxyz01);
    vacc2x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled2x0123456789ABCDEF);
    vacc2xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled2xGHIJKLMNOPQRSTUV);
    vacc2xWXYZabcdefghijkl = _mm512_cvtps_epi32(vscaled2xWXYZabcdefghijkl);
    vacc2xmnopqrstuvwxyz01 = _mm512_cvtps_epi32(vscaled2xmnopqrstuvwxyz01);
    vacc3x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled3x0123456789ABCDEF);
    vacc3xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled3xGHIJKLMNOPQRSTUV);
    vacc3xWXYZabcdefghijkl = _mm512_cvtps_epi32(vscaled3xWXYZabcdefghijkl);
    vacc3xmnopqrstuvwxyz01 = _mm512_cvtps_epi32(vscaled3xmnopqrstuvwxyz01);
    vacc4x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled4x0123456789ABCDEF);
    vacc4xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled4xGHIJKLMNOPQRSTUV);
    vacc4xWXYZabcdefghijkl = _mm512_cvtps_epi32(vscaled4xWXYZabcdefghijkl);
    vacc4xmnopqrstuvwxyz01 = _mm512_cvtps_epi32(vscaled4xmnopqrstuvwxyz01);
    vacc5x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled5x0123456789ABCDEF);
    vacc5xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled5xGHIJKLMNOPQRSTUV);
    vacc5xWXYZabcdefghijkl = _mm512_cvtps_epi32(vscaled5xWXYZabcdefghijkl);
    vacc5xmnopqrstuvwxyz01 = _mm512_cvtps_epi32(vscaled5xmnopqrstuvwxyz01);
    vacc6x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled6x0123456789ABCDEF);
    vacc6xGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaled6xGHIJKLMNOPQRSTUV);
    vacc6xWXYZabcdefghijkl = _mm512_cvtps_epi32(vscaled6xWXYZabcdefghijkl);
    vacc6xmnopqrstuvwxyz01 = _mm512_cvtps_epi32(vscaled6xmnopqrstuvwxyz01);

    vacc0x0123456789ABCDEF = _mm512_add_epi32(vacc0x0123456789ABCDEF, voutput_zero_point);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc0xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc0xWXYZabcdefghijkl = _mm512_add_epi32(vacc0xWXYZabcdefghijkl, voutput_zero_point);
    vacc0xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc0xmnopqrstuvwxyz01, voutput_zero_point);
    vacc1x0123456789ABCDEF = _mm512_add_epi32(vacc1x0123456789ABCDEF, voutput_zero_point);
    vacc1xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc1xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc1xWXYZabcdefghijkl = _mm512_add_epi32(vacc1xWXYZabcdefghijkl, voutput_zero_point);
    vacc1xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc1xmnopqrstuvwxyz01, voutput_zero_point);
    vacc2x0123456789ABCDEF = _mm512_add_epi32(vacc2x0123456789ABCDEF, voutput_zero_point);
    vacc2xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc2xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc2xWXYZabcdefghijkl = _mm512_add_epi32(vacc2xWXYZabcdefghijkl, voutput_zero_point);
    vacc2xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc2xmnopqrstuvwxyz01, voutput_zero_point);
    vacc3x0123456789ABCDEF = _mm512_add_epi32(vacc3x0123456789ABCDEF, voutput_zero_point);
    vacc3xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc3xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc3xWXYZabcdefghijkl = _mm512_add_epi32(vacc3xWXYZabcdefghijkl, voutput_zero_point);
    vacc3xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc3xmnopqrstuvwxyz01, voutput_zero_point);
    vacc4x0123456789ABCDEF = _mm512_add_epi32(vacc4x0123456789ABCDEF, voutput_zero_point);
    vacc4xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc4xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc4xWXYZabcdefghijkl = _mm512_add_epi32(vacc4xWXYZabcdefghijkl, voutput_zero_point);
    vacc4xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc4xmnopqrstuvwxyz01, voutput_zero_point);
    vacc5x0123456789ABCDEF = _mm512_add_epi32(vacc5x0123456789ABCDEF, voutput_zero_point);
    vacc5xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc5xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc5xWXYZabcdefghijkl = _mm512_add_epi32(vacc5xWXYZabcdefghijkl, voutput_zero_point);
    vacc5xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc5xmnopqrstuvwxyz01, voutput_zero_point);
    vacc6x0123456789ABCDEF = _mm512_add_epi32(vacc6x0123456789ABCDEF, voutput_zero_point);
    vacc6xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc6xGHIJKLMNOPQRSTUV, voutput_zero_point);
    vacc6xWXYZabcdefghijkl = _mm512_add_epi32(vacc6xWXYZabcdefghijkl, voutput_zero_point);
    vacc6xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc6xmnopqrstuvwxyz01, voutput_zero_point);

    __m128i vout0x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc0x0123456789ABCDEF);
    __m128i vout0xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc0xGHIJKLMNOPQRSTUV);
    __m128i vout0xWXYZabcdefghijkl = _mm512_cvtsepi32_epi8(vacc0xWXYZabcdefghijkl);
    __m128i vout0xmnopqrstuvwxyz01 = _mm512_cvtsepi32_epi8(vacc0xmnopqrstuvwxyz01);
    __m128i vout1x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc1x0123456789ABCDEF);
    __m128i vout1xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc1xGHIJKLMNOPQRSTUV);
    __m128i vout1xWXYZabcdefghijkl = _mm512_cvtsepi32_epi8(vacc1xWXYZabcdefghijkl);
    __m128i vout1xmnopqrstuvwxyz01 = _mm512_cvtsepi32_epi8(vacc1xmnopqrstuvwxyz01);
    __m128i vout2x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc2x0123456789ABCDEF);
    __m128i vout2xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc2xGHIJKLMNOPQRSTUV);
    __m128i vout2xWXYZabcdefghijkl = _mm512_cvtsepi32_epi8(vacc2xWXYZabcdefghijkl);
    __m128i vout2xmnopqrstuvwxyz01 = _mm512_cvtsepi32_epi8(vacc2xmnopqrstuvwxyz01);
    __m128i vout3x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc3x0123456789ABCDEF);
    __m128i vout3xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc3xGHIJKLMNOPQRSTUV);
    __m128i vout3xWXYZabcdefghijkl = _mm512_cvtsepi32_epi8(vacc3xWXYZabcdefghijkl);
    __m128i vout3xmnopqrstuvwxyz01 = _mm512_cvtsepi32_epi8(vacc3xmnopqrstuvwxyz01);
    __m128i vout4x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc4x0123456789ABCDEF);
    __m128i vout4xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc4xGHIJKLMNOPQRSTUV);
    __m128i vout4xWXYZabcdefghijkl = _mm512_cvtsepi32_epi8(vacc4xWXYZabcdefghijkl);
    __m128i vout4xmnopqrstuvwxyz01 = _mm512_cvtsepi32_epi8(vacc4xmnopqrstuvwxyz01);
    __m128i vout5x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc5x0123456789ABCDEF);
    __m128i vout5xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc5xGHIJKLMNOPQRSTUV);
    __m128i vout5xWXYZabcdefghijkl = _mm512_cvtsepi32_epi8(vacc5xWXYZabcdefghijkl);
    __m128i vout5xmnopqrstuvwxyz01 = _mm512_cvtsepi32_epi8(vacc5xmnopqrstuvwxyz01);
    __m128i vout6x0123456789ABCDEF = _mm512_cvtsepi32_epi8(vacc6x0123456789ABCDEF);
    __m128i vout6xGHIJKLMNOPQRSTUV = _mm512_cvtsepi32_epi8(vacc6xGHIJKLMNOPQRSTUV);
    __m128i vout6xWXYZabcdefghijkl = _mm512_cvtsepi32_epi8(vacc6xWXYZabcdefghijkl);
    __m128i vout6xmnopqrstuvwxyz01 = _mm512_cvtsepi32_epi8(vacc6xmnopqrstuvwxyz01);

    vout0x0123456789ABCDEF = _mm_max_epi8(vout0x0123456789ABCDEF, voutput_min);
    vout0xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout0xGHIJKLMNOPQRSTUV, voutput_min);
    vout0xWXYZabcdefghijkl = _mm_max_epi8(vout0xWXYZabcdefghijkl, voutput_min);
    vout0xmnopqrstuvwxyz01 = _mm_max_epi8(vout0xmnopqrstuvwxyz01, voutput_min);
    vout1x0123456789ABCDEF = _mm_max_epi8(vout1x0123456789ABCDEF, voutput_min);
    vout1xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout1xGHIJKLMNOPQRSTUV, voutput_min);
    vout1xWXYZabcdefghijkl = _mm_max_epi8(vout1xWXYZabcdefghijkl, voutput_min);
    vout1xmnopqrstuvwxyz01 = _mm_max_epi8(vout1xmnopqrstuvwxyz01, voutput_min);
    vout2x0123456789ABCDEF = _mm_max_epi8(vout2x0123456789ABCDEF, voutput_min);
    vout2xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout2xGHIJKLMNOPQRSTUV, voutput_min);
    vout2xWXYZabcdefghijkl = _mm_max_epi8(vout2xWXYZabcdefghijkl, voutput_min);
    vout2xmnopqrstuvwxyz01 = _mm_max_epi8(vout2xmnopqrstuvwxyz01, voutput_min);
    vout3x0123456789ABCDEF = _mm_max_epi8(vout3x0123456789ABCDEF, voutput_min);
    vout3xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout3xGHIJKLMNOPQRSTUV, voutput_min);
    vout3xWXYZabcdefghijkl = _mm_max_epi8(vout3xWXYZabcdefghijkl, voutput_min);
    vout3xmnopqrstuvwxyz01 = _mm_max_epi8(vout3xmnopqrstuvwxyz01, voutput_min);
    vout4x0123456789ABCDEF = _mm_max_epi8(vout4x0123456789ABCDEF, voutput_min);
    vout4xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout4xGHIJKLMNOPQRSTUV, voutput_min);
    vout4xWXYZabcdefghijkl = _mm_max_epi8(vout4xWXYZabcdefghijkl, voutput_min);
    vout4xmnopqrstuvwxyz01 = _mm_max_epi8(vout4xmnopqrstuvwxyz01, voutput_min);
    vout5x0123456789ABCDEF = _mm_max_epi8(vout5x0123456789ABCDEF, voutput_min);
    vout5xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout5xGHIJKLMNOPQRSTUV, voutput_min);
    vout5xWXYZabcdefghijkl = _mm_max_epi8(vout5xWXYZabcdefghijkl, voutput_min);
    vout5xmnopqrstuvwxyz01 = _mm_max_epi8(vout5xmnopqrstuvwxyz01, voutput_min);
    vout6x0123456789ABCDEF = _mm_max_epi8(vout6x0123456789ABCDEF, voutput_min);
    vout6xGHIJKLMNOPQRSTUV = _mm_max_epi8(vout6xGHIJKLMNOPQRSTUV, voutput_min);
    vout6xWXYZabcdefghijkl = _mm_max_epi8(vout6xWXYZabcdefghijkl, voutput_min);
    vout6xmnopqrstuvwxyz01 = _mm_max_epi8(vout6xmnopqrstuvwxyz01, voutput_min);

    if XNN_LIKELY(nc >= 64) {
      _mm_storeu_si128((__m128i*) (c6 + 0), vout6x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c6 + 16), vout6xGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (c6 + 32), vout6xWXYZabcdefghijkl);
      _mm_storeu_si128((__m128i*) (c6 + 48), vout6xmnopqrstuvwxyz01);
      c6 = (int8_t*) ((uintptr_t) c6 + cn_stride);
      _mm_storeu_si128((__m128i*) (c5 + 0), vout5x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c5 + 16), vout5xGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (c5 + 32), vout5xWXYZabcdefghijkl);
      _mm_storeu_si128((__m128i*) (c5 + 48), vout5xmnopqrstuvwxyz01);
      c5 = (int8_t*) ((uintptr_t) c5 + cn_stride);
      _mm_storeu_si128((__m128i*) (c4 + 0), vout4x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c4 + 16), vout4xGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (c4 + 32), vout4xWXYZabcdefghijkl);
      _mm_storeu_si128((__m128i*) (c4 + 48), vout4xmnopqrstuvwxyz01);
      c4 = (int8_t*) ((uintptr_t) c4 + cn_stride);
      _mm_storeu_si128((__m128i*) (c3 + 0), vout3x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c3 + 16), vout3xGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (c3 + 32), vout3xWXYZabcdefghijkl);
      _mm_storeu_si128((__m128i*) (c3 + 48), vout3xmnopqrstuvwxyz01);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      _mm_storeu_si128((__m128i*) (c2 + 0), vout2x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c2 + 16), vout2xGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (c2 + 32), vout2xWXYZabcdefghijkl);
      _mm_storeu_si128((__m128i*) (c2 + 48), vout2xmnopqrstuvwxyz01);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      _mm_storeu_si128((__m128i*) (c1 + 0), vout1x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c1 + 16), vout1xGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (c1 + 32), vout1xWXYZabcdefghijkl);
      _mm_storeu_si128((__m128i*) (c1 + 48), vout1xmnopqrstuvwxyz01);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      _mm_storeu_si128((__m128i*) (c0 + 0), vout0x0123456789ABCDEF);
      _mm_storeu_si128((__m128i*) (c0 + 16), vout0xGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (c0 + 32), vout0xWXYZabcdefghijkl);
      _mm_storeu_si128((__m128i*) (c0 + 48), vout0xmnopqrstuvwxyz01);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 64;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 0) & 0xFFFF));
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 16) & 0xFFFF));
      const __mmask16 vmask2 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 32) & 0xFFFF));
      const __mmask16 vmask3 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 48) & 0xFFFF));

      _mm_mask_storeu_epi8(c6 + 0, vmask0, vout6x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c6 + 16, vmask1, vout6xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c6 + 32, vmask2, vout6xWXYZabcdefghijkl);
      _mm_mask_storeu_epi8(c6 + 48, vmask3, vout6xmnopqrstuvwxyz01);
      _mm_mask_storeu_epi8(c5 + 0, vmask0, vout5x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c5 + 16, vmask1, vout5xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c5 + 32, vmask2, vout5xWXYZabcdefghijkl);
      _mm_mask_storeu_epi8(c5 + 48, vmask3, vout5xmnopqrstuvwxyz01);
      _mm_mask_storeu_epi8(c4 + 0, vmask0, vout4x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c4 + 16, vmask1, vout4xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c4 + 32, vmask2, vout4xWXYZabcdefghijkl);
      _mm_mask_storeu_epi8(c4 + 48, vmask3, vout4xmnopqrstuvwxyz01);
      _mm_mask_storeu_epi8(c3 + 0, vmask0, vout3x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c3 + 16, vmask1, vout3xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c3 + 32, vmask2, vout3xWXYZabcdefghijkl);
      _mm_mask_storeu_epi8(c3 + 48, vmask3, vout3xmnopqrstuvwxyz01);
      _mm_mask_storeu_epi8(c2 + 0, vmask0, vout2x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c2 + 16, vmask1, vout2xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c2 + 32, vmask2, vout2xWXYZabcdefghijkl);
      _mm_mask_storeu_epi8(c2 + 48, vmask3, vout2xmnopqrstuvwxyz01);
      _mm_mask_storeu_epi8(c1 + 0, vmask0, vout1x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c1 + 16, vmask1, vout1xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c1 + 32, vmask2, vout1xWXYZabcdefghijkl);
      _mm_mask_storeu_epi8(c1 + 48, vmask3, vout1xmnopqrstuvwxyz01);
      _mm_mask_storeu_epi8(c0 + 0, vmask0, vout0x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c0 + 16, vmask1, vout0xGHIJKLMNOPQRSTUV);
      _mm_mask_storeu_epi8(c0 + 32, vmask2, vout0xWXYZabcdefghijkl);
      _mm_mask_storeu_epi8(c0 + 48, vmask3, vout0xmnopqrstuvwxyz01);
      nc = 0;
    }
  } while (nc != 0);

  // Release tile config
  //  _tile_release();
  __asm__ volatile ("tilerelease" ::);
  #endif  // defined(__x86_64__)
}
