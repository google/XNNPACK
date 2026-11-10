// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/ace.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/unaligned.h"

// Uses the same RHS packing as the 16x32c4 ACE kernel with exact int32
// accumulation, so results are bit-identical to it. LHS is packed with
// mr_packed=1: one row of K bytes followed by the int32 negated zero point and
// the float scale.
void xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x32c4__ace(
    size_t m, size_t n, size_t k, const void* restrict lhs_packed,
    const void* restrict rhs_packed, float* restrict dst,
    size_t dst_stride_row, size_t dst_stride_col, const void* params) {
  assert(m == 1);
  assert(n != 0);
  assert(k != 0);
  assert(lhs_packed != NULL);
  assert(rhs_packed != NULL);
  assert(dst != NULL);
  assert(dst_stride_col == sizeof(float));
  (void) m;
  (void) dst_stride_row;
  (void) dst_stride_col;

  const struct xnn_f32_minmax_params* minmax_params =
      (const struct xnn_f32_minmax_params*) params;
  const __m512 voutput_min = _mm512_set1_ps(minmax_params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(minmax_params->scalar.max);

  const size_t k_internal = round_up_po2(k, 32);
  const int8_t* a = (const int8_t*) lhs_packed;
  const __m512i vneg_zp =
      _mm512_set1_epi32(unaligned_load_s32(a + k_internal));
  const __m512 vlhs_scale = _mm512_set1_ps(
      unaligned_load_f32(a + k_internal + sizeof(int32_t)));
  const int8_t* w = (const int8_t*) rhs_packed;

  do {
    // Two accumulators per 16 columns break the VPDPBSSD dependency chain.
    __m512i vacc0 = _mm512_setzero_si512();
    __m512i vacc1 = _mm512_setzero_si512();
    __m512i vacc2 = _mm512_setzero_si512();
    __m512i vacc3 = _mm512_setzero_si512();

    const int8_t* a_ptr = a;
    for (size_t kk = k_internal; kk != 0; kk -= 8) {
      const __m512i va0 = _mm512_set1_epi32(unaligned_load_s32(a_ptr + 0));
      vacc0 = xnn_avx10_dpbssd_epi32_t0(vacc0, va0, _mm512_loadu_si512(w + 0));
      vacc1 = xnn_avx10_dpbssd_epi32_t1(vacc1, va0, _mm512_loadu_si512(w + 64));
      const __m512i va1 = _mm512_set1_epi32(unaligned_load_s32(a_ptr + 4));
      vacc2 = xnn_avx10_dpbssd_epi32_t2(vacc2, va1, _mm512_loadu_si512(w + 128));
      vacc3 = xnn_avx10_dpbssd_epi32_t3(vacc3, va1, _mm512_loadu_si512(w + 192));
      a_ptr += 8;
      w += 256;
    }
    vacc0 = _mm512_add_epi32(vacc0, vacc2);
    vacc1 = _mm512_add_epi32(vacc1, vacc3);

    const __m512i vksum0 = _mm512_loadu_si512(w + 0);
    const __m512i vksum1 = _mm512_loadu_si512(w + 64);
    const __m512 vfilter_scale0 = _mm512_loadu_ps((const float*) (w + 128));
    const __m512 vfilter_scale1 = _mm512_loadu_ps((const float*) (w + 192));
    const __m512 vbias0 = _mm512_loadu_ps((const float*) (w + 256));
    const __m512 vbias1 = _mm512_loadu_ps((const float*) (w + 320));
    w += 384;

    vacc0 = _mm512_add_epi32(_mm512_mullo_epi32(vksum0, vneg_zp), vacc0);
    vacc1 = _mm512_add_epi32(_mm512_mullo_epi32(vksum1, vneg_zp), vacc1);
    __m512 vout0 = _mm512_cvtepi32_ps(vacc0);
    __m512 vout1 = _mm512_cvtepi32_ps(vacc1);
    vout0 = _mm512_mul_ps(vout0, vlhs_scale);
    vout1 = _mm512_mul_ps(vout1, vlhs_scale);
    vout0 = _mm512_fmadd_ps(vout0, vfilter_scale0, vbias0);
    vout1 = _mm512_fmadd_ps(vout1, vfilter_scale1, vbias1);
    vout0 = _mm512_max_ps(vout0, voutput_min);
    vout1 = _mm512_max_ps(vout1, voutput_min);
    vout0 = _mm512_min_ps(vout0, voutput_max);
    vout1 = _mm512_min_ps(vout1, voutput_max);

    if XNN_LIKELY(n >= 32) {
      _mm512_storeu_ps(dst + 0, vout0);
      _mm512_storeu_ps(dst + 16, vout1);
      dst += 32;
      n -= 32;
    } else {
      const __mmask16 vmask0 = _cvtu32_mask16(
          n >= 16 ? UINT32_C(0xFFFF) : (UINT32_C(1) << n) - UINT32_C(1));
      const __mmask16 vmask1 = _cvtu32_mask16(
          n > 16 ? (UINT32_C(1) << (n - 16)) - UINT32_C(1) : 0);
      _mm512_mask_storeu_ps(dst + 0, vmask0, vout0);
      _mm512_mask_storeu_ps(dst + 16, vmask1, vout1);
      n = 0;
    }
  } while (n != 0);
}
