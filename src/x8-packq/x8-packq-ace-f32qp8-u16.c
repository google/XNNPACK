// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <immintrin.h>

#include "src/xnnpack/ace.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/packq.h"

// Matches the scalar reference packer bit-exactly.
static XNN_INLINE void xnn_packq_row_params(const float* src, size_t k,
                                            float* scale_out,
                                            int32_t* zero_point_out,
                                            float* recip_scale_out) {
  __m512 vmax = _mm512_setzero_ps();
  __m512 vmin = _mm512_setzero_ps();
  size_t kk = k;
  for (; kk >= 16; kk -= 16) {
    const __m512 vx = _mm512_loadu_ps(src);
    src += 16;
    vmax = _mm512_max_ps(vmax, vx);
    vmin = _mm512_min_ps(vmin, vx);
  }
  if (kk != 0) {
    const __mmask16 vmask =
        _cvtu32_mask16((UINT32_C(1) << kk) - UINT32_C(1));
    const __m512 vx = _mm512_maskz_loadu_ps(vmask, src);
    vmax = _mm512_max_ps(vmax, vx);
    vmin = _mm512_min_ps(vmin, vx);
  }
  const float max0 = _mm512_reduce_max_ps(vmax);
  const float min0 = _mm512_reduce_min_ps(vmin);

  const float qmin = (float) INT8_MIN;
  const float qmax = (float) INT8_MAX;
  const float scale0 = min0 == max0 ? 1.0f : (qmax - qmin) / (max0 - min0);
  const float recip_scale0 = scale0 ? 1.0f / scale0 : 0.0f;
  const float descaled_min0 = min0 * scale0;
  const float descaled_max0 = max0 * scale0;
  const float zero_point_from_min_error0 = qmin + descaled_min0;
  const float zero_point_from_max_error0 = qmax + descaled_max0;
  float zero_point0 =
      zero_point_from_min_error0 + zero_point_from_max_error0 > 0
          ? qmin - descaled_min0
          : qmax - descaled_max0;
  zero_point0 = math_max_f32(zero_point0, qmin);
  zero_point0 = math_min_f32(zero_point0, qmax);

  *scale_out = scale0;
  *zero_point_out = (int32_t) rintf(zero_point0);
  *recip_scale_out = recip_scale0;
}

// Computes saturate_s8(roundf(x * scale) + zero_point) for 16 values, where
// roundf rounds half away from zero.
static XNN_INLINE __m128i xnn_packq_quantize16(__m512 vx, __m512 vscale,
                                               __m512i vzero_point) {
  const __m512 vprod = _mm512_mul_ps(vx, vscale);
  __m512 vt =
      _mm512_roundscale_ps(vprod, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
  const __m512 vfrac = _mm512_sub_ps(vprod, vt);
  const __mmask16 vround_away = _mm512_cmp_ps_mask(
      _mm512_abs_ps(vfrac), _mm512_set1_ps(0.5f), _CMP_GE_OQ);
  const __m512 vsigned_one = _mm512_castsi512_ps(_mm512_or_si512(
      _mm512_and_si512(_mm512_castps_si512(vprod),
                       _mm512_set1_epi32(INT32_C(0x80000000))),
      _mm512_set1_epi32(INT32_C(0x3F800000))));
  vt = _mm512_mask_add_ps(vt, vround_away, vt, vsigned_one);
  const __m512i vq = _mm512_add_epi32(_mm512_cvttps_epi32(vt), vzero_point);
  return _mm512_cvtsepi32_epi8(vq);
}

// Quantizes 64 values starting at `src`. Positions at or beyond `num_valid`
// replicate `vlast` (the last element of the row), matching the K padding of
// the reference packer.
static XNN_INLINE __m512i xnn_packq_quantize64(const float* src,
                                               size_t num_valid, __m512 vlast,
                                               __m512 vscale,
                                               __m512i vzero_point) {
  __m128i vq0, vq1, vq2, vq3;
  if XNN_LIKELY(num_valid >= 64) {
    vq0 = xnn_packq_quantize16(_mm512_loadu_ps(src + 0), vscale, vzero_point);
    vq1 = xnn_packq_quantize16(_mm512_loadu_ps(src + 16), vscale, vzero_point);
    vq2 = xnn_packq_quantize16(_mm512_loadu_ps(src + 32), vscale, vzero_point);
    vq3 = xnn_packq_quantize16(_mm512_loadu_ps(src + 48), vscale, vzero_point);
  } else {
    const uint64_t vmask_bits =
        (UINT64_C(1) << num_valid) - UINT64_C(1);
    vq0 = xnn_packq_quantize16(
        _mm512_mask_loadu_ps(vlast, _cvtu32_mask16((uint32_t) vmask_bits & 0xFFFF), src + 0),
        vscale, vzero_point);
    vq1 = xnn_packq_quantize16(
        _mm512_mask_loadu_ps(vlast, _cvtu32_mask16((uint32_t) (vmask_bits >> 16) & 0xFFFF), src + 16),
        vscale, vzero_point);
    vq2 = xnn_packq_quantize16(
        _mm512_mask_loadu_ps(vlast, _cvtu32_mask16((uint32_t) (vmask_bits >> 32) & 0xFFFF), src + 32),
        vscale, vzero_point);
    vq3 = xnn_packq_quantize16(
        _mm512_mask_loadu_ps(vlast, _cvtu32_mask16((uint32_t) (vmask_bits >> 48) & 0xFFFF), src + 48),
        vscale, vzero_point);
  }
  __m512i vout = _mm512_castsi128_si512(vq0);
  vout = _mm512_inserti32x4(vout, vq1, 1);
  vout = _mm512_inserti32x4(vout, vq2, 2);
  vout = _mm512_inserti32x4(vout, vq3, 3);
  return vout;
}

// Packs up to 16 rows into one mr=16, kr=4, sr=1 block. Each row is quantized
// 64 bytes at a time into a column of tmm7 (TILEMOVCOL) and read back by rows
// (TILEMOVROW), so the LHS transpose is done once here instead of in every
// NC tile of the GEMM.
static void xnn_packq_block16(size_t rows, size_t k, size_t k_internal,
                              const float* lhs, size_t lhs_stride,
                              int8_t* dst) {
  XNN_ALIGN(64) float scale[16];
  XNN_ALIGN(64) int32_t zero_point[16];
  XNN_ALIGN(64) float last[16];
  const float* src[16];
  int8_t* dst_params = dst + 16 * k_internal;
  for (size_t r = 0; r < 16; ++r) {
    src[r] = (const float*) ((uintptr_t) lhs + (r < rows ? r : 0) * lhs_stride);
    if (r < rows) {
      float recip_scale;
      xnn_packq_row_params(src[r], k, &scale[r], &zero_point[r], &recip_scale);
      const int32_t neg_zero_point = -zero_point[r];
      memcpy(dst_params + r * sizeof(int32_t), &neg_zero_point,
             sizeof(int32_t));
      memcpy(dst_params + 16 * sizeof(int32_t) + r * sizeof(float),
             &recip_scale, sizeof(float));
      last[r] = src[r][k - 1];
    }
  }

  for (size_t kb = 0; kb < k_internal; kb += 64) {
    const size_t num_valid = k > kb ? k - kb : 0;
#define XNN_PACKQ_ROW(r)                                                   \
    xnn_ace_tilemovcol_tmm7(                                               \
        (r) < rows ? xnn_packq_quantize64(src[r] + kb, num_valid,         \
                                          _mm512_set1_ps(last[r]),         \
                                          _mm512_set1_ps(scale[r]),        \
                                          _mm512_set1_epi32(zero_point[r])) \
                   : _mm512_setzero_si512(),                               \
        r)
    XNN_PACKQ_ROW(0);
    XNN_PACKQ_ROW(1);
    XNN_PACKQ_ROW(2);
    XNN_PACKQ_ROW(3);
    XNN_PACKQ_ROW(4);
    XNN_PACKQ_ROW(5);
    XNN_PACKQ_ROW(6);
    XNN_PACKQ_ROW(7);
    XNN_PACKQ_ROW(8);
    XNN_PACKQ_ROW(9);
    XNN_PACKQ_ROW(10);
    XNN_PACKQ_ROW(11);
    XNN_PACKQ_ROW(12);
    XNN_PACKQ_ROW(13);
    XNN_PACKQ_ROW(14);
    XNN_PACKQ_ROW(15);
#undef XNN_PACKQ_ROW

    // k_internal is a multiple of 32, so the last block has 8 or 16 K steps.
    const size_t num_ksteps = (k_internal - kb >= 64 ? 64 : 32) / 4;
    int8_t* out = dst + kb * 16;
    for (uint32_t s = 0; s < num_ksteps; ++s) {
      _mm512_storeu_si512(out + s * 64, xnn_ace_tilemovrow_tmm7(s));
    }
  }
}

void xnn_x8_packq_f32qp8_ukernel__ace_u16(size_t m, size_t k, size_t mr,
                                          size_t kr, size_t sr,
                                          size_t m_idx_start,
                                          const float* XNN_RESTRICT lhs,
                                          size_t lhs_stride,
                                          void* XNN_RESTRICT lhs_packed) {
  assert(kr % sr == 0);
  assert(kr / sr <= 32);
  if (m == 0 || k == 0) {
    return;
  }

  const size_t k_block_len = kr / sr;
  const size_t k_internal = k_roundedup(k, kr, sr);
  const size_t dst_stride = lhs_packed_stride(k, mr, kr, sr);
  int8_t* dst = (int8_t*) lhs_packed;

  if (mr == 16 && k_block_len == 4 && m_idx_start % 16 == 0) {
    xnn_ace_load_palette2_config();
    do {
      const size_t rows = m < 16 ? m : 16;
      xnn_packq_block16(rows, k, k_internal, lhs, lhs_stride, dst);
      lhs = (const float*) ((uintptr_t) lhs + rows * lhs_stride);
      dst += dst_stride;
      m -= rows;
    } while (m != 0);
    _tile_release();
    return;
  }

  for (size_t row = 0; row < m; ++row) {
    const float* src = (const float*) ((uintptr_t) lhs + row * lhs_stride);
    float scale;
    int32_t zero_point;
    float recip_scale;
    xnn_packq_row_params(src, k, &scale, &zero_point, &recip_scale);
    const __m512 vscale = _mm512_set1_ps(scale);
    const __m512i vzero_point = _mm512_set1_epi32(zero_point);
    const __m512 vlast = _mm512_set1_ps(src[k - 1]);
    const size_t dst_x = (row + m_idx_start) % mr;

    for (size_t kb = 0; kb < k_internal; kb += 64) {
      const size_t num_valid = k > kb ? k - kb : 0;
      const size_t kn = k_internal - kb >= 64 ? 64 : 32;
      const __m512i vq =
          xnn_packq_quantize64(src + kb, num_valid, vlast, vscale, vzero_point);
      if (mr == 1) {
        const __mmask64 vmask = _cvtu64_mask64(
            kn == 64 ? UINT64_MAX : (UINT64_C(1) << kn) - UINT64_C(1));
        _mm512_mask_storeu_epi8(dst + kb, vmask, vq);
      } else {
        XNN_ALIGN(64) int8_t tmp[64];
        _mm512_store_si512(tmp, vq);
        for (size_t c = 0; c < kn; c += k_block_len) {
          memcpy(dst + dst_x * k_block_len + (kb + c) * mr, tmp + c,
                 k_block_len);
        }
      }
    }

    const int32_t neg_zero_point = -zero_point;
    int8_t* dst_params = dst + mr * k_internal;
    memcpy(dst_params + dst_x * sizeof(int32_t), &neg_zero_point,
           sizeof(int32_t));
    memcpy(dst_params + mr * sizeof(int32_t) + dst_x * sizeof(float),
           &recip_scale, sizeof(float));

    if (dst_x == mr - 1) {
      dst += dst_stride;
    }
  }
}
