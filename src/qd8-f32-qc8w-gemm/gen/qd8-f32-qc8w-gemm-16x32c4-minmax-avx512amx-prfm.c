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
      if XNN_LIKELY(nc > 16) {
        _tile_loadd(5, (const int8_t*) w + 64, 128);
        _tile_dpbssd(1, 4, 5);
      }
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
      if XNN_LIKELY(nc > 16) {
        _tile_loadd(7, (const int8_t*) w + 64, 128);
        _tile_dpbssd(1, 6, 7);
      }

      w = (const int8_t*) w + kremainder * 32;
      k -= kremainder * sizeof(int8_t);
    }

    // TODO: Instead of processing up to 4 tiles (16x64) consider
    // quantizing 1 tile at a time (16 registers)
    _tile_stored(0, &res[0][0], 64);
    if XNN_LIKELY(nc > 16) {
      _tile_stored(1, &res[1][0], 64);
    }

    // TODO: Fix msan for AMX
    #if defined(__has_feature)
      #if __has_feature(memory_sanitizer)
        __msan_unpoison(res, sizeof(res));
      #endif
    #endif

    // Quantize 1 row at a time across all tiles.
    const __m512 vfilter_output_scale0 = _mm512_load_ps((const float*) w + 0);
    const __m512 vfilter_output_scale1 = _mm512_load_ps((const float*) w + 16);
    w = (const int32_t*) w + 32;
    const __m512 vbias0 = _mm512_load_ps((const float*) w + 0);
    const __m512 vbias1 = _mm512_load_ps((const float*) w + 16);
    w = (const int32_t*) w + 32;

    float* c0 = c;
    for (size_t m = 0; m < mr; ++m)
    {
      const __m512i vzero_point = _mm512_set1_epi32((int) quantization_params[m].zero_point);
      const __m512 vinv_scale = _mm512_set1_ps(quantization_params[m].inv_scale);
      if XNN_LIKELY(nc >= 32) {
        __m512i vacc0 = _mm512_mullo_epi32(vksum0, vzero_point);
        __m512i vacc1 = _mm512_mullo_epi32(vksum1, vzero_point);
        // Add tile to bias
        vacc0 = _mm512_add_epi32(vacc0, _mm512_load_epi32(&res[0][0] + m * 16));
        vacc1 = _mm512_add_epi32(vacc1, _mm512_load_epi32(&res[1][0] + m * 16));

        __m512 vscaled0 = _mm512_cvtepi32_ps(vacc0);
        __m512 vscaled1 = _mm512_cvtepi32_ps(vacc1);

        vscaled0 = _mm512_mul_ps(vscaled0, vinv_scale);
        vscaled1 = _mm512_mul_ps(vscaled1, vinv_scale);
        vscaled0 = _mm512_fmadd_ps(vscaled0, vfilter_output_scale0, vbias0);
        vscaled1 = _mm512_fmadd_ps(vscaled1, vfilter_output_scale1, vbias1);
        vscaled0 = _mm512_max_ps(vscaled0, voutput_min);
        vscaled1 = _mm512_max_ps(vscaled1, voutput_min);
        vscaled0 = _mm512_min_ps(vscaled0, voutput_max);
        vscaled1 = _mm512_min_ps(vscaled1, voutput_max);

        _mm512_storeu_ps(c0 + 0, vscaled0);
        _mm512_storeu_ps(c0 + 16, vscaled1);
      } else {
        __m512i vacc0 = _mm512_mullo_epi32(vksum0, vzero_point);
        vacc0 = _mm512_add_epi32(vacc0, _mm512_load_epi32(&res[0][0] + m * 16));
        __m512 vscaled0 = _mm512_cvtepi32_ps(vacc0);
        vscaled0 = _mm512_mul_ps(vscaled0, vinv_scale);
        vscaled0 = _mm512_fmadd_ps(vscaled0, vfilter_output_scale0, vbias0);
        vscaled0 = _mm512_max_ps(vscaled0, voutput_min);
        vscaled0 = _mm512_min_ps(vscaled0, voutput_max);
        const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 0) & 0xFFFF));
        _mm512_mask_storeu_ps(c0 + 0, vmask0, vscaled0);
        if (nc > 16) {
        __m512i vacc1 = _mm512_mullo_epi32(vksum1, vzero_point);
        vacc1 = _mm512_add_epi32(vacc1, _mm512_load_epi32(&res[1][0] + m * 16));
        __m512 vscaled1 = _mm512_cvtepi32_ps(vacc1);
        vscaled1 = _mm512_mul_ps(vscaled1, vinv_scale);
        vscaled1 = _mm512_fmadd_ps(vscaled1, vfilter_output_scale1, vbias1);
        vscaled1 = _mm512_max_ps(vscaled1, voutput_min);
        vscaled1 = _mm512_min_ps(vscaled1, voutput_max);
        const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 16) & 0xFFFF));
        _mm512_mask_storeu_ps(c0 + 16, vmask1, vscaled1);
        }
      }
      c0 = (float*) ((uintptr_t) c0 + cm_stride);
    }
    if XNN_LIKELY(nc >= 32) {
      c = (float*) ((uintptr_t) c + cn_stride);
      a -= kc;
      nc -= 32;
    } else {
      nc = 0;
    }
  } while (nc != 0);
  // Release tile config
  _tile_release();
  #endif  // defined(__x86_64__)
}
