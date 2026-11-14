// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c4-avx512vnnigfni.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/prefetch.h"
#include "src/xnnpack/unaligned.h"


void xnn_qd8_f32_qc2w_gemm_minmax_ukernel_1x64c4__avx512vnnigfni_prfm(
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
    const float* row_sum,
    const struct xnn_qd8_quantization_params* restrict quantization_params) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);
  assert(row_sum != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);

  const __m512i vmask = _mm512_set1_epi8(0x03);
  const __m512i vshr2 = _mm512_set1_epi64(0x0408000000000000);
  const __m512i vshr4 = _mm512_set1_epi64(0x1020000000000000);
  const __m512i vshr6 = _mm512_set1_epi64(0x4080000000000000);
  const __m512 vtwo = _mm512_set1_ps(2.0f);
  const __m512i vinput_zero_point0 = _mm512_set1_epi32((int) quantization_params[0].zero_point);
  const __m512 vlh_row_sum_0 = _mm512_set1_ps(row_sum[0]);
  const __m512 vscaled_lh_zero_point_0 = _mm512_set1_ps((float) kc * quantization_params[0].zero_point);
  const __m512 vinv_scale_0 = _mm512_set1_ps(quantization_params[0].inv_scale);

  do {
    const int32_t* ksum = (const int32_t*) w;
    w = (const int32_t*) w + 64;
    const float* kzp = (const float*) w;
    w = (const float*) w + 64;
    __m512i vacc0x0 = _mm512_setzero_epi32();
    __m512i vacc0x1 = _mm512_setzero_epi32();
    __m512i vacc0x2 = _mm512_setzero_epi32();
    __m512i vacc0x3 = _mm512_setzero_epi32();
    __m512i vacc1x0x0 = _mm512_setzero_epi32();
    __m512i vacc1x0x1 = _mm512_setzero_epi32();
    __m512i vacc1x0x2 = _mm512_setzero_epi32();
    __m512i vacc1x0x3 = _mm512_setzero_epi32();

    size_t k = kc;
    while (k >= 16 * sizeof(int8_t)) {
      const __m512i vbb0 = _mm512_load_si512((const int8_t*) w + 0);
      const __m512i vbb1 = _mm512_load_si512((const int8_t*) w + 64);
      const __m512i vbb2 = _mm512_load_si512((const int8_t*) w + 128);
      const __m512i vbb3 = _mm512_load_si512((const int8_t*) w + 192);

      const __m512i vb0x0 = _mm512_and_si512(vbb0, vmask);
      const __m512i vb1x0 = _mm512_and_si512(vbb1, vmask);
      const __m512i vb2x0 = _mm512_and_si512(vbb2, vmask);
      const __m512i vb3x0 = _mm512_and_si512(vbb3, vmask);
      const __m512i va0x0 = _mm512_set1_epi32((int) unaligned_load_u32(a0));
      vacc0x0 = _mm512_dpbusd_epi32(vacc0x0, vb0x0, va0x0);
      vacc0x1 = _mm512_dpbusd_epi32(vacc0x1, vb1x0, va0x0);
      vacc0x2 = _mm512_dpbusd_epi32(vacc0x2, vb2x0, va0x0);
      vacc0x3 = _mm512_dpbusd_epi32(vacc0x3, vb3x0, va0x0);

      const __m512i vb0x1 = _mm512_gf2p8affine_epi64_epi8(vbb0, vshr2, 0);
      const __m512i vb1x1 = _mm512_gf2p8affine_epi64_epi8(vbb1, vshr2, 0);
      const __m512i vb2x1 = _mm512_gf2p8affine_epi64_epi8(vbb2, vshr2, 0);
      const __m512i vb3x1 = _mm512_gf2p8affine_epi64_epi8(vbb3, vshr2, 0);
      const __m512i va0x1 = _mm512_set1_epi32((int) unaligned_load_u32(a0 + 4));
      vacc1x0x0 = _mm512_dpbusd_epi32(vacc1x0x0, vb0x1, va0x1);
      vacc1x0x1 = _mm512_dpbusd_epi32(vacc1x0x1, vb1x1, va0x1);
      vacc1x0x2 = _mm512_dpbusd_epi32(vacc1x0x2, vb2x1, va0x1);
      vacc1x0x3 = _mm512_dpbusd_epi32(vacc1x0x3, vb3x1, va0x1);

      xnn_prefetch_to_l1((const int8_t*) w + 1536);
      xnn_prefetch_to_l1((const int8_t*) w + 1600);
      xnn_prefetch_to_l1((const int8_t*) w + 1664);
      xnn_prefetch_to_l1((const int8_t*) w + 1728);

      const __m512i vb0x2 = _mm512_gf2p8affine_epi64_epi8(vbb0, vshr4, 0);
      const __m512i vb1x2 = _mm512_gf2p8affine_epi64_epi8(vbb1, vshr4, 0);
      const __m512i vb2x2 = _mm512_gf2p8affine_epi64_epi8(vbb2, vshr4, 0);
      const __m512i vb3x2 = _mm512_gf2p8affine_epi64_epi8(vbb3, vshr4, 0);
      const __m512i va0x2 = _mm512_set1_epi32((int) unaligned_load_u32(a0 + 8));
      vacc0x0 = _mm512_dpbusd_epi32(vacc0x0, vb0x2, va0x2);
      vacc0x1 = _mm512_dpbusd_epi32(vacc0x1, vb1x2, va0x2);
      vacc0x2 = _mm512_dpbusd_epi32(vacc0x2, vb2x2, va0x2);
      vacc0x3 = _mm512_dpbusd_epi32(vacc0x3, vb3x2, va0x2);

      const __m512i vb0x3 = _mm512_gf2p8affine_epi64_epi8(vbb0, vshr6, 0);
      const __m512i vb1x3 = _mm512_gf2p8affine_epi64_epi8(vbb1, vshr6, 0);
      const __m512i vb2x3 = _mm512_gf2p8affine_epi64_epi8(vbb2, vshr6, 0);
      const __m512i vb3x3 = _mm512_gf2p8affine_epi64_epi8(vbb3, vshr6, 0);
      const __m512i va0x3 = _mm512_set1_epi32((int) unaligned_load_u32(a0 + 12));
      a0 += 16;
      vacc1x0x0 = _mm512_dpbusd_epi32(vacc1x0x0, vb0x3, va0x3);
      vacc1x0x1 = _mm512_dpbusd_epi32(vacc1x0x1, vb1x3, va0x3);
      vacc1x0x2 = _mm512_dpbusd_epi32(vacc1x0x2, vb2x3, va0x3);
      vacc1x0x3 = _mm512_dpbusd_epi32(vacc1x0x3, vb3x3, va0x3);

      w = (const int8_t*) w + 256;
      k -= 16 * sizeof(int8_t);
    }

    if XNN_UNLIKELY(k != 0) {
      const __m512i vbb0 = _mm512_load_si512((const int8_t*) w + 0);
      const __m512i vbb1 = _mm512_load_si512((const int8_t*) w + 64);
      const __m512i vbb2 = _mm512_load_si512((const int8_t*) w + 128);
      const __m512i vbb3 = _mm512_load_si512((const int8_t*) w + 192);
      const __m512i vb0x0 = _mm512_and_si512(vbb0, vmask);
      const __m512i vb1x0 = _mm512_and_si512(vbb1, vmask);
      const __m512i vb2x0 = _mm512_and_si512(vbb2, vmask);
      const __m512i vb3x0 = _mm512_and_si512(vbb3, vmask);
      const __m512i va0x0 = _mm512_set1_epi32((int) unaligned_load_u32(a0));
      a0 += 4;
      vacc0x0 = _mm512_dpbusd_epi32(vacc0x0, vb0x0, va0x0);
      vacc0x1 = _mm512_dpbusd_epi32(vacc0x1, vb1x0, va0x0);
      vacc0x2 = _mm512_dpbusd_epi32(vacc0x2, vb2x0, va0x0);
      vacc0x3 = _mm512_dpbusd_epi32(vacc0x3, vb3x0, va0x0);

      if (k > 4 * sizeof(int8_t)) {
        const __m512i vb0x1 = _mm512_gf2p8affine_epi64_epi8(vbb0, vshr2, 0);
        const __m512i vb1x1 = _mm512_gf2p8affine_epi64_epi8(vbb1, vshr2, 0);
        const __m512i vb2x1 = _mm512_gf2p8affine_epi64_epi8(vbb2, vshr2, 0);
        const __m512i vb3x1 = _mm512_gf2p8affine_epi64_epi8(vbb3, vshr2, 0);
        const __m512i va0x1 = _mm512_set1_epi32((int) unaligned_load_u32(a0));
        a0 += 4;
        vacc1x0x0 = _mm512_dpbusd_epi32(vacc1x0x0, vb0x1, va0x1);
        vacc1x0x1 = _mm512_dpbusd_epi32(vacc1x0x1, vb1x1, va0x1);
        vacc1x0x2 = _mm512_dpbusd_epi32(vacc1x0x2, vb2x1, va0x1);
        vacc1x0x3 = _mm512_dpbusd_epi32(vacc1x0x3, vb3x1, va0x1);

        if (k > 8 * sizeof(int8_t)) {
          const __m512i vb0x2 = _mm512_gf2p8affine_epi64_epi8(vbb0, vshr4, 0);
          const __m512i vb1x2 = _mm512_gf2p8affine_epi64_epi8(vbb1, vshr4, 0);
          const __m512i vb2x2 = _mm512_gf2p8affine_epi64_epi8(vbb2, vshr4, 0);
          const __m512i vb3x2 = _mm512_gf2p8affine_epi64_epi8(vbb3, vshr4, 0);
          const __m512i va0x2 = _mm512_set1_epi32((int) unaligned_load_u32(a0));
          a0 += 4;
          vacc0x0 = _mm512_dpbusd_epi32(vacc0x0, vb0x2, va0x2);
          vacc0x1 = _mm512_dpbusd_epi32(vacc0x1, vb1x2, va0x2);
          vacc0x2 = _mm512_dpbusd_epi32(vacc0x2, vb2x2, va0x2);
          vacc0x3 = _mm512_dpbusd_epi32(vacc0x3, vb3x2, va0x2);
        }
      }
      w = (const int8_t*) w + 256;
    }

    vacc0x0 = _mm512_add_epi32(vacc0x0, vacc1x0x0);
    vacc0x1 = _mm512_add_epi32(vacc0x1, vacc1x0x1);
    vacc0x2 = _mm512_add_epi32(vacc0x2, vacc1x0x2);
    vacc0x3 = _mm512_add_epi32(vacc0x3, vacc1x0x3);

    const __m512i vksum0 = _mm512_load_epi32(ksum + 0);
    const __m512i vksum1 = _mm512_load_epi32(ksum + 16);
    const __m512i vksum2 = _mm512_load_epi32(ksum + 32);
    const __m512i vksum3 = _mm512_load_epi32(ksum + 48);
    vacc0x0 = _mm512_add_epi32(vacc0x0, _mm512_mullo_epi32(vksum0, vinput_zero_point0));
    vacc0x1 = _mm512_add_epi32(vacc0x1, _mm512_mullo_epi32(vksum1, vinput_zero_point0));
    vacc0x2 = _mm512_add_epi32(vacc0x2, _mm512_mullo_epi32(vksum2, vinput_zero_point0));
    vacc0x3 = _mm512_add_epi32(vacc0x3, _mm512_mullo_epi32(vksum3, vinput_zero_point0));

    __m512 vscaled0x0 = _mm512_cvtepi32_ps(vacc0x0);
    __m512 vscaled0x1 = _mm512_cvtepi32_ps(vacc0x1);
    __m512 vscaled0x2 = _mm512_cvtepi32_ps(vacc0x2);
    __m512 vscaled0x3 = _mm512_cvtepi32_ps(vacc0x3);

    const __m512 rh_zero_points_0 = _mm512_load_ps(kzp + 0);
    const __m512 biased_rh_zero_points_0 = _mm512_add_ps(rh_zero_points_0, vtwo);
    const __m512 rh_zero_points_1 = _mm512_load_ps(kzp + 16);
    const __m512 biased_rh_zero_points_1 = _mm512_add_ps(rh_zero_points_1, vtwo);
    const __m512 rh_zero_points_2 = _mm512_load_ps(kzp + 32);
    const __m512 biased_rh_zero_points_2 = _mm512_add_ps(rh_zero_points_2, vtwo);
    const __m512 rh_zero_points_3 = _mm512_load_ps(kzp + 48);
    const __m512 biased_rh_zero_points_3 = _mm512_add_ps(rh_zero_points_3, vtwo);

    vscaled0x0 = _mm512_fnmadd_ps(biased_rh_zero_points_0, vlh_row_sum_0, vscaled0x0);
    vscaled0x1 = _mm512_fnmadd_ps(biased_rh_zero_points_1, vlh_row_sum_0, vscaled0x1);
    vscaled0x2 = _mm512_fnmadd_ps(biased_rh_zero_points_2, vlh_row_sum_0, vscaled0x2);
    vscaled0x3 = _mm512_fnmadd_ps(biased_rh_zero_points_3, vlh_row_sum_0, vscaled0x3);

    vscaled0x0 = _mm512_fmadd_ps(rh_zero_points_0, vscaled_lh_zero_point_0, vscaled0x0);
    vscaled0x1 = _mm512_fmadd_ps(rh_zero_points_1, vscaled_lh_zero_point_0, vscaled0x1);
    vscaled0x2 = _mm512_fmadd_ps(rh_zero_points_2, vscaled_lh_zero_point_0, vscaled0x2);
    vscaled0x3 = _mm512_fmadd_ps(rh_zero_points_3, vscaled_lh_zero_point_0, vscaled0x3);

    vscaled0x0 = _mm512_mul_ps(vscaled0x0, vinv_scale_0);
    vscaled0x1 = _mm512_mul_ps(vscaled0x1, vinv_scale_0);
    vscaled0x2 = _mm512_mul_ps(vscaled0x2, vinv_scale_0);
    vscaled0x3 = _mm512_mul_ps(vscaled0x3, vinv_scale_0);

    const __m512 vfilter_output_scale0 = _mm512_load_ps((const float*) w + 0);
    const __m512 vfilter_output_scale1 = _mm512_load_ps((const float*) w + 16);
    const __m512 vfilter_output_scale2 = _mm512_load_ps((const float*) w + 32);
    const __m512 vfilter_output_scale3 = _mm512_load_ps((const float*) w + 48);
    w = (const float*) w + 64;
    const __m512 vbias0 = _mm512_load_ps((const float*) w + 0);
    const __m512 vbias1 = _mm512_load_ps((const float*) w + 16);
    const __m512 vbias2 = _mm512_load_ps((const float*) w + 32);
    const __m512 vbias3 = _mm512_load_ps((const float*) w + 48);
    w = (const float*) w + 64;

    vscaled0x0 = _mm512_fmadd_ps(vscaled0x0, vfilter_output_scale0, vbias0);
    vscaled0x1 = _mm512_fmadd_ps(vscaled0x1, vfilter_output_scale1, vbias1);
    vscaled0x2 = _mm512_fmadd_ps(vscaled0x2, vfilter_output_scale2, vbias2);
    vscaled0x3 = _mm512_fmadd_ps(vscaled0x3, vfilter_output_scale3, vbias3);

    vscaled0x0 = _mm512_max_ps(vscaled0x0, voutput_min);
    vscaled0x1 = _mm512_max_ps(vscaled0x1, voutput_min);
    vscaled0x2 = _mm512_max_ps(vscaled0x2, voutput_min);
    vscaled0x3 = _mm512_max_ps(vscaled0x3, voutput_min);

    vscaled0x0 = _mm512_min_ps(vscaled0x0, voutput_max);
    vscaled0x1 = _mm512_min_ps(vscaled0x1, voutput_max);
    vscaled0x2 = _mm512_min_ps(vscaled0x2, voutput_max);
    vscaled0x3 = _mm512_min_ps(vscaled0x3, voutput_max);

    if XNN_LIKELY(nc >= 64) {
      _mm512_storeu_ps(c0 + 0, vscaled0x0);
      _mm512_storeu_ps(c0 + 16, vscaled0x1);
      _mm512_storeu_ps(c0 + 32, vscaled0x2);
      _mm512_storeu_ps(c0 + 48, vscaled0x3);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 64;
    } else {
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 0) & 0xFFFF));
      _mm512_mask_storeu_ps(c0 + 0, vmask0, vscaled0x0);
      if (nc > 16) {
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 16) & 0xFFFF));
      _mm512_mask_storeu_ps(c0 + 16, vmask1, vscaled0x1);
      }
      if (nc > 32) {
      const __mmask16 vmask2 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 32) & 0xFFFF));
      _mm512_mask_storeu_ps(c0 + 32, vmask2, vscaled0x2);
      }
      if (nc > 48) {
      const __mmask16 vmask3 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 48) & 0xFFFF));
      _mm512_mask_storeu_ps(c0 + 48, vmask3, vscaled0x3);
      }
      nc = 0;
    }
  } while (nc != 0);
}
