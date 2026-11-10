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

// LHS is packed by `x8-packq` with mr_packed=16, kr=4, sr=1: every 4-byte K
// step stores the 16 rows contiguously (64 bytes), so each K step is already
// the transposed operand for TOP4BSSD and no in-kernel transpose is needed.
// The block is followed by 16 int32 negated zero points and 16 float scales.
//
// RHS is packed with nr=32, kr=4: each K step stores 32 columns x 4 bytes,
// followed by 32 int32 row sums, 32 float scales and 32 float biases.
void xnn_qp8_f32_qc8w_gemm_minmax_ukernel_16x32c4__ace(
    size_t m, size_t n, size_t k, const void* restrict lhs_packed,
    const void* restrict rhs_packed, float* restrict dst,
    size_t dst_stride_row, size_t dst_stride_col, const void* params) {
  assert(m != 0);
  assert(m <= 16);
  assert(n != 0);
  assert(k != 0);
  assert(lhs_packed != NULL);
  assert(rhs_packed != NULL);
  assert(dst != NULL);
  assert(dst_stride_col == sizeof(float));
  (void) dst_stride_col;

  const struct xnn_f32_minmax_params* minmax_params =
      (const struct xnn_f32_minmax_params*) params;
  const __m512 voutput_min = _mm512_set1_ps(minmax_params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(minmax_params->scalar.max);

  const size_t k_internal = round_up_po2(k, 32);
  const int8_t* a = (const int8_t*) lhs_packed;
  const int32_t* lhs_neg_zp = (const int32_t*) (a + 16 * k_internal);
  const float* lhs_scale = (const float*) (lhs_neg_zp + 16);
  const int8_t* w = (const int8_t*) rhs_packed;

  xnn_ace_load_palette2_config();

  do {
    _tile_zero(0);
    _tile_zero(1);

    const int8_t* a_ptr = a;
    for (size_t kk = k_internal; kk != 0; kk -= 32) {
      __m512i va = _mm512_loadu_si512(a_ptr + 0);
      xnn_ace_top4bssd_tmm0(va, _mm512_loadu_si512(w + 0));
      xnn_ace_top4bssd_tmm1(va, _mm512_loadu_si512(w + 64));
      va = _mm512_loadu_si512(a_ptr + 64);
      xnn_ace_top4bssd_tmm0(va, _mm512_loadu_si512(w + 128));
      xnn_ace_top4bssd_tmm1(va, _mm512_loadu_si512(w + 192));
      va = _mm512_loadu_si512(a_ptr + 128);
      xnn_ace_top4bssd_tmm0(va, _mm512_loadu_si512(w + 256));
      xnn_ace_top4bssd_tmm1(va, _mm512_loadu_si512(w + 320));
      va = _mm512_loadu_si512(a_ptr + 192);
      xnn_ace_top4bssd_tmm0(va, _mm512_loadu_si512(w + 384));
      xnn_ace_top4bssd_tmm1(va, _mm512_loadu_si512(w + 448));
      va = _mm512_loadu_si512(a_ptr + 256);
      xnn_ace_top4bssd_tmm0(va, _mm512_loadu_si512(w + 512));
      xnn_ace_top4bssd_tmm1(va, _mm512_loadu_si512(w + 576));
      va = _mm512_loadu_si512(a_ptr + 320);
      xnn_ace_top4bssd_tmm0(va, _mm512_loadu_si512(w + 640));
      xnn_ace_top4bssd_tmm1(va, _mm512_loadu_si512(w + 704));
      va = _mm512_loadu_si512(a_ptr + 384);
      xnn_ace_top4bssd_tmm0(va, _mm512_loadu_si512(w + 768));
      xnn_ace_top4bssd_tmm1(va, _mm512_loadu_si512(w + 832));
      va = _mm512_loadu_si512(a_ptr + 448);
      xnn_ace_top4bssd_tmm0(va, _mm512_loadu_si512(w + 896));
      xnn_ace_top4bssd_tmm1(va, _mm512_loadu_si512(w + 960));
      a_ptr += 512;
      w += 1024;
    }

    const __m512i vksum0 = _mm512_loadu_si512(w + 0);
    const __m512i vksum1 = _mm512_loadu_si512(w + 64);
    const __m512 vfilter_scale0 = _mm512_loadu_ps((const float*) (w + 128));
    const __m512 vfilter_scale1 = _mm512_loadu_ps((const float*) (w + 192));
    const __m512 vbias0 = _mm512_loadu_ps((const float*) (w + 256));
    const __m512 vbias1 = _mm512_loadu_ps((const float*) (w + 320));
    w += 384;

    const __mmask16 vmask0 = _cvtu32_mask16(
        n >= 16 ? UINT32_C(0xFFFF) : (UINT32_C(1) << n) - UINT32_C(1));
    const __mmask16 vmask1 = _cvtu32_mask16(
        n >= 32 ? UINT32_C(0xFFFF)
                : n > 16 ? (UINT32_C(1) << (n - 16)) - UINT32_C(1) : 0);

    float* c = dst;
    for (size_t row = 0; row < m; ++row) {
      const __m512i vneg_zp = _mm512_set1_epi32(lhs_neg_zp[row]);
      const __m512 vlhs_scale = _mm512_set1_ps(lhs_scale[row]);
      __m512i vacc0 = _mm512_mullo_epi32(vksum0, vneg_zp);
      __m512i vacc1 = _mm512_mullo_epi32(vksum1, vneg_zp);
      vacc0 = _mm512_add_epi32(vacc0, xnn_ace_tilemovrow_tmm0((uint32_t) row));
      vacc1 = _mm512_add_epi32(vacc1, xnn_ace_tilemovrow_tmm1((uint32_t) row));
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
      _mm512_mask_storeu_ps(c + 0, vmask0, vout0);
      _mm512_mask_storeu_ps(c + 16, vmask1, vout1);
      c = (float*) ((uintptr_t) c + dst_stride_row);
    }

    if XNN_LIKELY(n > 32) {
      dst += 32;
      n -= 32;
    } else {
      n = 0;
    }
  } while (n != 0);

  _tile_release();
}
