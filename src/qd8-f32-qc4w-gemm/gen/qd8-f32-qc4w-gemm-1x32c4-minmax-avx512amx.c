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


void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x32c4__avx512amx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

// TODO: amxintrin.h only provide intrinsics for __x86_64__
// Update if amxintrin changes
#if defined(__x86_64__)
  __attribute__((aligned(64))) int32_t res0[1 * 16];
  __attribute__((aligned(64))) int32_t res1[1 * 16];
  __attribute__((aligned(64))) int8_t weight_buffer[16 * 64];

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

  float* c0 = c;

  const __m512 voutput_min = _mm512_set1_ps(params->avx512vnni.min);
  const __m512 voutput_max = _mm512_set1_ps(params->avx512vnni.max);
  const __m512i vmask = _mm512_set1_epi8(0xF0);
  const __m512i vshl4 = _mm512_set1_epi64(0x01020408);
  XNN_FORCE_REALIZATION(vmask);
  XNN_FORCE_REALIZATION(vshl4);

  do {
    const __m512i vksum0123456789ABCDEF = _mm512_load_epi32((const int32_t*) w + 0);
    const __m512i vksumGHIJKLMNOPQRSTUV = _mm512_load_epi32((const int32_t*) w + 16);
    w = (const int32_t*) w + 32;

    // Zero tile accumulator
    __asm__ volatile (
      "tilezero %%tmm0\n"
      "tilezero %%tmm1\n"
      ::);

    size_t k = kc;
    while (k >= 64 * sizeof(int8_t)) {
      _tile_loadd(4, a, a_stride);
      a += 64;
      const __m512i vb0x0123456789ABCDEF = _mm512_load_epi32((const int8_t*) w + 0);
      const __m512i vb1x0123456789ABCDEF = _mm512_load_epi32((const int8_t*) w + 128);
      const __m512i vb2x0123456789ABCDEF = _mm512_load_epi32((const int8_t*) w + 256);
      const __m512i vb3x0123456789ABCDEF = _mm512_load_epi32((const int8_t*) w + 384);
      const __m512i vb4x0123456789ABCDEF = _mm512_load_epi32((const int8_t*) w + 512);
      const __m512i vb5x0123456789ABCDEF = _mm512_load_epi32((const int8_t*) w + 640);
      const __m512i vb6x0123456789ABCDEF = _mm512_load_epi32((const int8_t*) w + 768);
      const __m512i vb7x0123456789ABCDEF = _mm512_load_epi32((const int8_t*) w + 896);
      const __m512i vl0x0123456789ABCDEF = _mm512_gf2p8affine_epi64_epi8(vb0x0123456789ABCDEF, vshl4, 0);
      const __m512i vh0x0123456789ABCDEF = _mm512_and_si512(vb0x0123456789ABCDEF, vmask);
      const __m512i vl1x0123456789ABCDEF = _mm512_gf2p8affine_epi64_epi8(vb1x0123456789ABCDEF, vshl4, 0);
      const __m512i vh1x0123456789ABCDEF = _mm512_and_si512(vb1x0123456789ABCDEF, vmask);
      const __m512i vl2x0123456789ABCDEF = _mm512_gf2p8affine_epi64_epi8(vb2x0123456789ABCDEF, vshl4, 0);
      const __m512i vh2x0123456789ABCDEF = _mm512_and_si512(vb2x0123456789ABCDEF, vmask);
      const __m512i vl3x0123456789ABCDEF = _mm512_gf2p8affine_epi64_epi8(vb3x0123456789ABCDEF, vshl4, 0);
      const __m512i vh3x0123456789ABCDEF = _mm512_and_si512(vb3x0123456789ABCDEF, vmask);
      const __m512i vl4x0123456789ABCDEF = _mm512_gf2p8affine_epi64_epi8(vb4x0123456789ABCDEF, vshl4, 0);
      const __m512i vh4x0123456789ABCDEF = _mm512_and_si512(vb4x0123456789ABCDEF, vmask);
      const __m512i vl5x0123456789ABCDEF = _mm512_gf2p8affine_epi64_epi8(vb5x0123456789ABCDEF, vshl4, 0);
      const __m512i vh5x0123456789ABCDEF = _mm512_and_si512(vb5x0123456789ABCDEF, vmask);
      const __m512i vl6x0123456789ABCDEF = _mm512_gf2p8affine_epi64_epi8(vb6x0123456789ABCDEF, vshl4, 0);
      const __m512i vh6x0123456789ABCDEF = _mm512_and_si512(vb6x0123456789ABCDEF, vmask);
      const __m512i vl7x0123456789ABCDEF = _mm512_gf2p8affine_epi64_epi8(vb7x0123456789ABCDEF, vshl4, 0);
      const __m512i vh7x0123456789ABCDEF = _mm512_and_si512(vb7x0123456789ABCDEF, vmask);
      _mm512_store_epi32(weight_buffer + 0, vl0x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 64, vh0x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 128, vl1x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 192, vh1x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 256, vl2x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 320, vh2x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 384, vl3x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 448, vh3x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 512, vl4x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 576, vh4x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 640, vl5x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 704, vh5x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 768, vl6x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 832, vh6x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 896, vl7x0123456789ABCDEF);
      _mm512_store_epi32(weight_buffer + 960, vh7x0123456789ABCDEF);
      _tile_loadd(5, weight_buffer, 64);
      _tile_dpbssd(0, 4, 5);
      const __m512i vb0xGHIJKLMNOPQRSTUV = _mm512_load_epi32((const int8_t*) w + 64);
      const __m512i vb1xGHIJKLMNOPQRSTUV = _mm512_load_epi32((const int8_t*) w + 192);
      const __m512i vb2xGHIJKLMNOPQRSTUV = _mm512_load_epi32((const int8_t*) w + 320);
      const __m512i vb3xGHIJKLMNOPQRSTUV = _mm512_load_epi32((const int8_t*) w + 448);
      const __m512i vb4xGHIJKLMNOPQRSTUV = _mm512_load_epi32((const int8_t*) w + 576);
      const __m512i vb5xGHIJKLMNOPQRSTUV = _mm512_load_epi32((const int8_t*) w + 704);
      const __m512i vb6xGHIJKLMNOPQRSTUV = _mm512_load_epi32((const int8_t*) w + 832);
      const __m512i vb7xGHIJKLMNOPQRSTUV = _mm512_load_epi32((const int8_t*) w + 960);
      const __m512i vl0xGHIJKLMNOPQRSTUV = _mm512_gf2p8affine_epi64_epi8(vb0xGHIJKLMNOPQRSTUV, vshl4, 0);
      const __m512i vh0xGHIJKLMNOPQRSTUV = _mm512_and_si512(vb0xGHIJKLMNOPQRSTUV, vmask);
      const __m512i vl1xGHIJKLMNOPQRSTUV = _mm512_gf2p8affine_epi64_epi8(vb1xGHIJKLMNOPQRSTUV, vshl4, 0);
      const __m512i vh1xGHIJKLMNOPQRSTUV = _mm512_and_si512(vb1xGHIJKLMNOPQRSTUV, vmask);
      const __m512i vl2xGHIJKLMNOPQRSTUV = _mm512_gf2p8affine_epi64_epi8(vb2xGHIJKLMNOPQRSTUV, vshl4, 0);
      const __m512i vh2xGHIJKLMNOPQRSTUV = _mm512_and_si512(vb2xGHIJKLMNOPQRSTUV, vmask);
      const __m512i vl3xGHIJKLMNOPQRSTUV = _mm512_gf2p8affine_epi64_epi8(vb3xGHIJKLMNOPQRSTUV, vshl4, 0);
      const __m512i vh3xGHIJKLMNOPQRSTUV = _mm512_and_si512(vb3xGHIJKLMNOPQRSTUV, vmask);
      const __m512i vl4xGHIJKLMNOPQRSTUV = _mm512_gf2p8affine_epi64_epi8(vb4xGHIJKLMNOPQRSTUV, vshl4, 0);
      const __m512i vh4xGHIJKLMNOPQRSTUV = _mm512_and_si512(vb4xGHIJKLMNOPQRSTUV, vmask);
      const __m512i vl5xGHIJKLMNOPQRSTUV = _mm512_gf2p8affine_epi64_epi8(vb5xGHIJKLMNOPQRSTUV, vshl4, 0);
      const __m512i vh5xGHIJKLMNOPQRSTUV = _mm512_and_si512(vb5xGHIJKLMNOPQRSTUV, vmask);
      const __m512i vl6xGHIJKLMNOPQRSTUV = _mm512_gf2p8affine_epi64_epi8(vb6xGHIJKLMNOPQRSTUV, vshl4, 0);
      const __m512i vh6xGHIJKLMNOPQRSTUV = _mm512_and_si512(vb6xGHIJKLMNOPQRSTUV, vmask);
      const __m512i vl7xGHIJKLMNOPQRSTUV = _mm512_gf2p8affine_epi64_epi8(vb7xGHIJKLMNOPQRSTUV, vshl4, 0);
      const __m512i vh7xGHIJKLMNOPQRSTUV = _mm512_and_si512(vb7xGHIJKLMNOPQRSTUV, vmask);
      _mm512_store_epi32(weight_buffer + 0, vl0xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 64, vh0xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 128, vl1xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 192, vh1xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 256, vl2xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 320, vh2xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 384, vl3xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 448, vh3xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 512, vl4xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 576, vh4xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 640, vl5xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 704, vh5xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 768, vl6xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 832, vh6xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 896, vl7xGHIJKLMNOPQRSTUV);
      _mm512_store_epi32(weight_buffer + 960, vh7xGHIJKLMNOPQRSTUV);
      _tile_loadd(5, weight_buffer, 64);
      _tile_dpbssd(1, 4, 5);

      w = (const int8_t*) w + 1024;
      k -= 64 * sizeof(int8_t);
    }

    if XNN_UNLIKELY(k != 0) {
      _tile_loadd(6, a, a_stride);
      a += kremainder;
      for (size_t k = 0; k < ((kremainder + 7) >> 3); ++k) {
        const __m512i vb0123456789ABCDEF = _mm512_load_epi32((const int8_t*) w + 0 + 128 * k);
        const __m512i vl0123456789ABCDEF = _mm512_gf2p8affine_epi64_epi8(vb0123456789ABCDEF, vshl4, 0);
        const __m512i vh0123456789ABCDEF = _mm512_and_si512(vb0123456789ABCDEF, vmask);
        _mm512_store_epi32(weight_buffer + 128 * k + 0, vl0123456789ABCDEF);
        _mm512_store_epi32(weight_buffer + 128 * k + 64, vh0123456789ABCDEF);
      }
      _tile_loadd(7, weight_buffer, 64);
      _tile_dpbssd(0, 6, 7);
      for (size_t k = 0; k < ((kremainder + 7) >> 3); ++k) {
        const __m512i vbGHIJKLMNOPQRSTUV = _mm512_load_epi32((const int8_t*) w + 64 + 128 * k);
        const __m512i vlGHIJKLMNOPQRSTUV = _mm512_gf2p8affine_epi64_epi8(vbGHIJKLMNOPQRSTUV, vshl4, 0);
        const __m512i vhGHIJKLMNOPQRSTUV = _mm512_and_si512(vbGHIJKLMNOPQRSTUV, vmask);
        _mm512_store_epi32(weight_buffer + 128 * k + 0, vlGHIJKLMNOPQRSTUV);
        _mm512_store_epi32(weight_buffer + 128 * k + 64, vhGHIJKLMNOPQRSTUV);
      }
      _tile_loadd(7, weight_buffer, 64);
      _tile_dpbssd(1, 6, 7);

      w = (const int8_t*) w + ((kremainder + 7) >> 3) * 128;
      k -= kremainder * sizeof(int8_t);
    }

    // Add tile to bias
    _tile_stored(0, res0, 64);
    _tile_stored(1, res1, 64);

    __m512i vacc0x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params[0].zero_point));
    __m512i vacc0xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params[0].zero_point));
    vacc0x0123456789ABCDEF = _mm512_add_epi32(vacc0x0123456789ABCDEF, _mm512_load_epi32(res0 + 0));
    vacc0xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc0xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 0));

    vacc0x0123456789ABCDEF = _mm512_srai_epi32(vacc0x0123456789ABCDEF, 4);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_srai_epi32(vacc0xGHIJKLMNOPQRSTUV, 4);
    __m512 vscaled0x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0x0123456789ABCDEF);
    __m512 vscaled0xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc0xGHIJKLMNOPQRSTUV);

    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, _mm512_set1_ps(quantization_params[0].inv_scale));
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled0xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params[0].inv_scale));

    const __m512 vfilter_output_scale0123456789ABCDEF = _mm512_load_ps((const float*) w + 0);
    const __m512 vfilter_output_scaleGHIJKLMNOPQRSTUV = _mm512_load_ps((const float*) w + 16);
    w = (const int32_t*) w + 32;
    const __m512 vbias0123456789ABCDEF = _mm512_load_ps((const float*) w + 0);
    const __m512 vbiasGHIJKLMNOPQRSTUV = _mm512_load_ps((const float*) w + 16);
    w = (const int32_t*) w + 32;

    vscaled0x0123456789ABCDEF = _mm512_fmadd_ps(vscaled0x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled0xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);

    vscaled0x0123456789ABCDEF = _mm512_max_ps(vscaled0x0123456789ABCDEF, voutput_min);
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled0xGHIJKLMNOPQRSTUV, voutput_min);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max);
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled0xGHIJKLMNOPQRSTUV, voutput_max);

    if XNN_LIKELY(nc >= 32) {
      _mm512_storeu_ps(c0 + 0, vscaled0x0123456789ABCDEF);
      _mm512_storeu_ps(c0 + 16, vscaled0xGHIJKLMNOPQRSTUV);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a -= kc;
      nc -= 32;
    } else {
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 0) & 0xFFFF));
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 16) & 0xFFFF));
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
