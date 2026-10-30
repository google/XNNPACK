// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#if XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
#include <arm_sme.h>
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"

size_t xnn_pf32_gemm_minmax_ukernel_32x32__sme2_get_mr() {
  return 2 * svcntsw();
}

size_t xnn_pf32_gemm_minmax_ukernel_32x32__sme2_get_nr() {
  return 2 * svcntsw();
}

#define PREFETCH_W_MATRIX

__arm_locally_streaming __arm_new("za")
static void pf32_gemm_32x32_sme2_mr16(
    size_t mr,
    size_t nc,
    size_t kc,
    const void*  a,
    const float* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    float min,
    float max)
{
#ifdef PREFETCH_W_MATRIX
#if defined(__APPLE__)
  const size_t prfm_min_w_size = (16 * 1024 * 1024);
#else
  const size_t prfm_min_w_size = (1024 * 1024);
#endif
#endif
  size_t vl = svcntw();
  cn_stride = 2 * vl * cn_stride; //32:nr

  svbool_t pg_32_all    = svptrue_b32();
  svcount_t pcnt_32_all = svptrue_c32();

  svfloat32_t vmin = svdup_f32(min);
  svfloat32_t vmax = svdup_f32(max);

  const float* a_base = (const float*) a;

  size_t w_offset_nr = 2 * vl * (kc + 1);
  const float* w0 = w;
  const float* w1 = (nc > 2 * vl) ? w0 + w_offset_nr : w0;
  size_t nc_rem = nc;
  float* c0 = c;

#ifdef PREFETCH_W_MATRIX
  const size_t kPrefetchDistance = 16;
  const size_t kBytesPerKLoop = 32 * vl;
  const size_t kCacheLineSize = 64;
  const size_t kCacheLinesPerLoop = kBytesPerKLoop / kCacheLineSize;
  const size_t kPrfmStride = kBytesPerKLoop / 8;
  const size_t kTrailerPrfmStride = 8 * vl;

  size_t block_size = w_offset_nr * sizeof(float);
  size_t nc_rup_nr = ((nc + 2 * vl - 1) / (2 * vl)) * (2 * vl);
  size_t total_w_size = ((kc * nc_rup_nr) + nc_rup_nr) * sizeof(float);
  bool need_prfm_0 = (block_size >= kPrefetchDistance * kBytesPerKLoop) && (total_w_size > prfm_min_w_size);
  bool need_prfm_1 = (nc_rem > 2 * vl) && need_prfm_0;

  size_t prfm_acc_w_size_0 = 0;
  size_t prfm_acc_w_size_1 = 0;

  const int8_t* prfm_ptr_0 = (const int8_t*) w0;
  const int8_t* prfm_ptr_1 = (const int8_t*) w1;

  if (need_prfm_0) {
    prfm_acc_w_size_0 = kPrefetchDistance * kBytesPerKLoop;
    int i = 0;
    for (; i < (kPrefetchDistance * kCacheLinesPerLoop); i++) {
      __builtin_prefetch(prfm_ptr_0 + kCacheLineSize * i, 0, 2);
    }
    prfm_ptr_0 += kCacheLineSize * i;
  }
  if (need_prfm_1) {
    prfm_acc_w_size_1 = kPrefetchDistance * kBytesPerKLoop;
    int i = 0;
    for (; i < (kPrefetchDistance * kCacheLinesPerLoop); i++) {
      __builtin_prefetch(prfm_ptr_1 + kCacheLineSize * i, 0, 2);
    }
    prfm_ptr_1 += kCacheLineSize * i;
  }
#endif

  do {
    svcount_t pcnt_32_next_nr32 = (nc_rem > 2 * vl) ? pcnt_32_all : svpfalse_c();

    svfloat32x2_t vb_bias_x2_0 = svld1_f32_x2(pcnt_32_all, w0); w0 += 2*vl;
    svfloat32x2_t vb_bias_x2_1 = svld1_f32_x2(pcnt_32_next_nr32, w1); w1 += 2*vl;

    svfloat32_t f32_one = svdup_f32(1.0f);
    svzero_za();
    svmopa_za32_f32_m(0, pg_32_all, pg_32_all, f32_one, svget2(vb_bias_x2_0, 0));
    svmopa_za32_f32_m(1, pg_32_all, pg_32_all, f32_one, svget2(vb_bias_x2_0, 1));
    svmopa_za32_f32_m(2, pg_32_all, pg_32_all, f32_one, svget2(vb_bias_x2_1, 0));
    svmopa_za32_f32_m(3, pg_32_all, pg_32_all, f32_one, svget2(vb_bias_x2_1, 1));

    size_t k = kc;
    const float* a0 = a_base;

    size_t k_chunks = k / 4;
    if (k_chunks > 0) {
      svfloat32_t va0 = svld1_f32(pg_32_all, a0);
      svfloat32_t va1 = svld1_f32(pg_32_all, a0 + 2*vl);
      svfloat32_t va2 = svld1_f32(pg_32_all, a0 + 4*vl);
      svfloat32_t va3 = svld1_f32(pg_32_all, a0 + 6*vl);
      a0 += 8*vl;
      svfloat32x4_t vb_x4_0 = svld1_f32_x4(pcnt_32_all, w0);
      svfloat32x4_t vb_x4_1 = svld1_f32_x4(pcnt_32_all, w0 + 4*vl);
      w0 += 8*vl;
      svfloat32x4_t vb_x4_2 = svld1_f32_x4(pcnt_32_next_nr32, w1);
      svfloat32x4_t vb_x4_3 = svld1_f32_x4(pcnt_32_next_nr32, w1 + 4*vl);
      w1 += 8*vl;

#ifdef PREFETCH_W_MATRIX
      if (need_prfm_0 && (prfm_acc_w_size_0 < (block_size - kBytesPerKLoop))) {
        prfm_acc_w_size_0 += kBytesPerKLoop;
        __builtin_prefetch(prfm_ptr_0 + 0 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr_0 + 1 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr_0 + 2 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr_0 + 3 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr_0 + 4 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr_0 + 5 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr_0 + 6 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr_0 + 7 * kPrfmStride, 0, 2);
        prfm_ptr_0 += kBytesPerKLoop;
      }
      if (need_prfm_1 && (prfm_acc_w_size_1 < (block_size - kBytesPerKLoop))) {
        prfm_acc_w_size_1 += kBytesPerKLoop;
        __builtin_prefetch(prfm_ptr_1 + 0 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr_1 + 1 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr_1 + 2 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr_1 + 3 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr_1 + 4 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr_1 + 5 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr_1 + 6 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr_1 + 7 * kPrfmStride, 0, 2);
        prfm_ptr_1 += kBytesPerKLoop;
      }
#endif

      k_chunks--;

      while (k_chunks > 0) {
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va0, svget4_f32(vb_x4_0, 0));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va0, svget4_f32(vb_x4_0, 1));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va0, svget4_f32(vb_x4_2, 0));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va0, svget4_f32(vb_x4_2, 1));
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va1, svget4_f32(vb_x4_0, 2));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va1, svget4_f32(vb_x4_0, 3));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va1, svget4_f32(vb_x4_2, 2));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va1, svget4_f32(vb_x4_2, 3));

        va0 = svld1_f32(pg_32_all, a0);
        vb_x4_0 = svld1_f32_x4(pcnt_32_all, w0);
        va1 = svld1_f32(pg_32_all, a0 + 2*vl);
        vb_x4_2 = svld1_f32_x4(pcnt_32_next_nr32, w1);

        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va2, svget4_f32(vb_x4_1, 0));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va2, svget4_f32(vb_x4_1, 1));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va2, svget4_f32(vb_x4_3, 0));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va2, svget4_f32(vb_x4_3, 1));
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va3, svget4_f32(vb_x4_1, 2));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va3, svget4_f32(vb_x4_1, 3));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va3, svget4_f32(vb_x4_3, 2));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va3, svget4_f32(vb_x4_3, 3));

        va2 = svld1_f32(pg_32_all, a0 + 4*vl);
        vb_x4_1 = svld1_f32_x4(pcnt_32_all, w0 + 4*vl);
        va3 = svld1_f32(pg_32_all, a0 + 6*vl);
        vb_x4_3 = svld1_f32_x4(pcnt_32_next_nr32, w1 + 4*vl);
        a0 += 8*vl;
        w0 += 8*vl;
        w1 += 8*vl;

#ifdef PREFETCH_W_MATRIX
        if (need_prfm_0 && (prfm_acc_w_size_0 < (block_size - kBytesPerKLoop))) {
          prfm_acc_w_size_0 += kBytesPerKLoop;
          __builtin_prefetch(prfm_ptr_0 + 0 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 1 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 2 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 3 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 4 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 5 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 6 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 7 * kPrfmStride, 0, 2);
          prfm_ptr_0 += kBytesPerKLoop;
        }
        if (need_prfm_1 && (prfm_acc_w_size_1 < (block_size - kBytesPerKLoop))) {
          prfm_acc_w_size_1 += kBytesPerKLoop;
          __builtin_prefetch(prfm_ptr_1 + 0 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 1 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 2 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 3 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 4 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 5 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 6 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 7 * kPrfmStride, 0, 2);
          prfm_ptr_1 += kBytesPerKLoop;
        }
#endif

        k_chunks--;
      }

      svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va0, svget4_f32(vb_x4_0, 0));
      svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va0, svget4_f32(vb_x4_0, 1));
      svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va0, svget4_f32(vb_x4_2, 0));
      svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va0, svget4_f32(vb_x4_2, 1));
      svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va1, svget4_f32(vb_x4_0, 2));
      svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va1, svget4_f32(vb_x4_0, 3));
      svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va1, svget4_f32(vb_x4_2, 2));
      svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va1, svget4_f32(vb_x4_2, 3));

      svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va2, svget4_f32(vb_x4_1, 0));
      svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va2, svget4_f32(vb_x4_1, 1));
      svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va2, svget4_f32(vb_x4_3, 0));
      svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va2, svget4_f32(vb_x4_3, 1));
      svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va3, svget4_f32(vb_x4_1, 2));
      svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va3, svget4_f32(vb_x4_1, 3));
      svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va3, svget4_f32(vb_x4_3, 2));
      svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va3, svget4_f32(vb_x4_3, 3));
    }
    k %= 4;

    XNN_PRAGMA_CLANG("clang loop unroll(disable)")
    while (k != 0) {
      svfloat32_t va0 = svld1_f32(pg_32_all, a0); a0 += 2*vl;

      svfloat32x2_t vb_x2_0 = svld1_f32_x2(pcnt_32_all, w0); w0 += 2*vl;
      svfloat32x2_t vb_x2_1 = svld1_f32_x2(pcnt_32_next_nr32, w1); w1 += 2*vl;

      svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va0, svget2(vb_x2_0, 0));
      svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va0, svget2(vb_x2_0, 1));
      svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va0, svget2(vb_x2_1, 0));
      svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va0, svget2(vb_x2_1, 1));
      k -= 1;
    }

#ifdef PREFETCH_W_MATRIX
    if (need_prfm_0) {
      if (prfm_acc_w_size_0 <= (block_size - 3 * (8 * vl))) {
        prfm_acc_w_size_0 += 3 * (8 * vl);
        __builtin_prefetch(prfm_ptr_0 + 0 * kTrailerPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr_0 + 1 * kTrailerPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr_0 + 2 * kTrailerPrfmStride, 0, 2);
        prfm_ptr_0 += 3 * (8 * vl);
      } else if (prfm_acc_w_size_0 <= (block_size - 2 * (8 * vl))) {
        prfm_acc_w_size_0 += 2 * (8 * vl);
        __builtin_prefetch(prfm_ptr_0 + 0 * kTrailerPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr_0 + 1 * kTrailerPrfmStride, 0, 2);
        prfm_ptr_0 += 2 * (8 * vl);
      } else if (prfm_acc_w_size_0 <= (block_size - 1 * (8 * vl))) {
        prfm_acc_w_size_0 += 1 * (8 * vl);
        __builtin_prefetch(prfm_ptr_0 + 0 * kTrailerPrfmStride, 0, 2);
        prfm_ptr_0 += 1 * (8 * vl);
      }
    }
    if (need_prfm_1) {
      if (prfm_acc_w_size_1 <= (block_size - 3 * (8 * vl))) {
        prfm_acc_w_size_1 += 3 * (8 * vl);
        __builtin_prefetch(prfm_ptr_1 + 0 * kTrailerPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr_1 + 1 * kTrailerPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr_1 + 2 * kTrailerPrfmStride, 0, 2);
        prfm_ptr_1 += 3 * (8 * vl);
      } else if (prfm_acc_w_size_1 <= (block_size - 2 * (8 * vl))) {
        prfm_acc_w_size_1 += 2 * (8 * vl);
        __builtin_prefetch(prfm_ptr_1 + 0 * kTrailerPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr_1 + 1 * kTrailerPrfmStride, 0, 2);
        prfm_ptr_1 += 2 * (8 * vl);
      } else if (prfm_acc_w_size_1 <= (block_size - 1 * (8 * vl))) {
        prfm_acc_w_size_1 += 1 * (8 * vl);
        __builtin_prefetch(prfm_ptr_1 + 0 * kTrailerPrfmStride, 0, 2);
        prfm_ptr_1 += 1 * (8 * vl);
      }
    }
#endif

    size_t n_len = (nc_rem >= 4 * vl) ? 4 * vl : nc_rem;
    svcount_t pn = svwhilelt_c32_u64(0, n_len, 4);

    float* c_inc = c0;
    size_t mr_idx = 0;
    XNN_PRAGMA_CLANG("clang loop unroll(disable)")
    for (size_t i = 0; i < mr; i++) {
      svfloat32x4_t za0x4 = svreinterpret_f32(svread_hor_za8_u8_vg4(0, mr_idx));
      za0x4 = svclamp_single_f32_x4(za0x4, vmin, vmax);
      svst1_f32_x4(pn, c_inc, za0x4);
      c_inc = (float*) ((uintptr_t)c_inc + cm_stride);
      mr_idx += 4;
    }

    if (nc_rem > 4 * vl) {
      c0 = (float*)((uintptr_t)c0 + 2 * cn_stride);
      nc_rem -= 4 * vl;
      w0 = w1;
      w1 = (nc_rem > 2 * vl) ? w0 + w_offset_nr : w0;
#ifdef PREFETCH_W_MATRIX
      need_prfm_1 = (nc_rem > 2 * vl) && need_prfm_0;
      if (need_prfm_0 && nc_rem > 0) {
        prfm_ptr_0 = (const int8_t*) w0;
        prfm_acc_w_size_0 = kPrefetchDistance * kBytesPerKLoop;
        int i = 0;
        for (; i < (kPrefetchDistance * kCacheLinesPerLoop); i++) {
          __builtin_prefetch(prfm_ptr_0 + kCacheLineSize * i, 0, 2);
        }
        prfm_ptr_0 += kCacheLineSize * i;
      }
      if (need_prfm_1) {
        prfm_ptr_1 = (const int8_t*) w1;
        prfm_acc_w_size_1 = kPrefetchDistance * kBytesPerKLoop;
        int i = 0;
        for (; i < (kPrefetchDistance * kCacheLinesPerLoop); i++) {
          __builtin_prefetch(prfm_ptr_1 + kCacheLineSize * i, 0, 2);
        }
        prfm_ptr_1 += kCacheLineSize * i;
      }
#endif
    } else {
      nc_rem = 0;
    }

  } while (nc_rem != 0);
}

__arm_locally_streaming __arm_new("za")
static void pf32_gemm_32x32_sme2_mr32(
    size_t mr,
    size_t nc,
    size_t kc,
    const void*  a,
    const float* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    float min,
    float max)
{
#ifdef PREFETCH_W_MATRIX
#if defined(__APPLE__)
  const size_t prfm_min_w_size = (16 * 1024 * 1024);
#else
  const size_t prfm_min_w_size = (1024 * 1024);
#endif
#endif
  size_t vl = svcntw();
  cn_stride = 2 * vl * cn_stride; //32:nr

  svbool_t pg_32_all    = svptrue_b32();
  svcount_t pcnt_32_all = svptrue_c32();

  svfloat32_t vmin = svdup_f32(min);
  svfloat32_t vmax = svdup_f32(max);

  const float* a_base = (const float*) a;

  // Outer M loop
  for (size_t m_start = 0; m_start < mr; m_start += 2 * vl) {
    size_t m_chunk = (mr - m_start >= 2 * vl) ? 2 * vl : (mr - m_start);

    float* c_base = (float*)((uintptr_t)c + m_start * cm_stride);
    const float* w0 = w;
    size_t nc_rem = nc;
    float* c0 = c_base;

#ifdef PREFETCH_W_MATRIX
    const size_t kPrefetchDistance = 16;
    const size_t kBytesPerKLoop = 32 * vl;
    const size_t kCacheLineSize = 64;
    const size_t kCacheLinesPerLoop = kBytesPerKLoop / kCacheLineSize;
    const size_t kPrfmStride = kBytesPerKLoop / 8;
    const size_t kTrailerPrfmStride = 8 * vl;

    size_t nc_rup_nr = ((nc + 2 * vl - 1) / (2 * vl)) * (2 * vl);
    size_t total_w_size = ((kc * nc_rup_nr) + nc_rup_nr) * sizeof(float);
    bool need_prfm = (total_w_size >= kPrefetchDistance * kBytesPerKLoop) && (total_w_size > prfm_min_w_size);

    size_t prfm_acc_w_size = 0;
    const int8_t* prfm_ptr = (const int8_t*) w;

    if (need_prfm) {
      prfm_acc_w_size = kPrefetchDistance * kBytesPerKLoop;
      int i = 0;
      for (; i < (kPrefetchDistance * kCacheLinesPerLoop); i++) {
        __builtin_prefetch(prfm_ptr + kCacheLineSize * i, 0, 2);
      }
      prfm_ptr += kCacheLineSize * i;
    }
#endif

    do {
      svfloat32x2_t vb_bias_x2 = svld1_f32_x2(pcnt_32_all, w0); w0 += 2*vl;

      svfloat32_t f32_bias_0 = svget2(vb_bias_x2, 0);
      svfloat32_t f32_bias_1 = svget2(vb_bias_x2, 1);
      svfloat32_t f32_one = svdup_f32(1.0f);
      svzero_za();
      svmopa_za32_f32_m(0, pg_32_all, pg_32_all, f32_one, f32_bias_0);
      svmopa_za32_f32_m(1, pg_32_all, pg_32_all, f32_one, f32_bias_1);
      svmopa_za32_f32_m(2, pg_32_all, pg_32_all, f32_one, f32_bias_0);
      svmopa_za32_f32_m(3, pg_32_all, pg_32_all, f32_one, f32_bias_1);

      size_t k = kc;
      const float* a0 = a_base + m_start * kc;

      size_t k_chunks = k / 4;
      if (k_chunks > 0) {
        svfloat32x4_t va_x4_0 = svld1_f32_x4(pcnt_32_all, a0);
        svfloat32x4_t va_x4_1 = svld1_f32_x4(pcnt_32_all, a0 + 4*vl);
        a0 += 8*vl;
        svfloat32x4_t vb_x4_0 = svld1_f32_x4(pcnt_32_all, w0);
        svfloat32x4_t vb_x4_1 = svld1_f32_x4(pcnt_32_all, w0 + 4*vl);
        w0 += 8*vl;

#ifdef PREFETCH_W_MATRIX
        if (need_prfm && (prfm_acc_w_size < (total_w_size - kBytesPerKLoop))) {
          prfm_acc_w_size += kBytesPerKLoop;
          __builtin_prefetch(prfm_ptr + 0 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 1 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 2 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 3 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 4 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 5 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 6 * kPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr + 7 * kPrfmStride, 0, 2);
          prfm_ptr += kBytesPerKLoop;
        }
#endif

        k_chunks--;

        while (k_chunks > 0) {
          svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 0), svget4_f32(vb_x4_0, 0));
          svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 0), svget4_f32(vb_x4_0, 1));
          svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 1), svget4_f32(vb_x4_0, 0));
          svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 1), svget4_f32(vb_x4_0, 1));
          svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 2), svget4_f32(vb_x4_0, 2));
          svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 2), svget4_f32(vb_x4_0, 3));
          svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 3), svget4_f32(vb_x4_0, 2));
          svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 3), svget4_f32(vb_x4_0, 3));

          va_x4_0 = svld1_f32_x4(pcnt_32_all, a0);
          svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 0), svget4_f32(vb_x4_1, 0));
          vb_x4_0 = svld1_f32_x4(pcnt_32_all, w0);
          svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 0), svget4_f32(vb_x4_1, 1));
          svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 1), svget4_f32(vb_x4_1, 0));
          svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 1), svget4_f32(vb_x4_1, 1));
          svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 2), svget4_f32(vb_x4_1, 2));
          svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 2), svget4_f32(vb_x4_1, 3));
          svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 3), svget4_f32(vb_x4_1, 2));
          svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 3), svget4_f32(vb_x4_1, 3));

          va_x4_1 = svld1_f32_x4(pcnt_32_all, a0 + 4*vl);
          vb_x4_1 = svld1_f32_x4(pcnt_32_all, w0 + 4*vl);
          a0 += 8*vl;
          w0 += 8*vl;

#ifdef PREFETCH_W_MATRIX
          if (need_prfm && (prfm_acc_w_size < (total_w_size - kBytesPerKLoop))) {
            prfm_acc_w_size += kBytesPerKLoop;
            __builtin_prefetch(prfm_ptr + 0 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr + 1 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr + 2 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr + 3 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr + 4 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr + 5 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr + 6 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr + 7 * kPrfmStride, 0, 2);
            prfm_ptr += kBytesPerKLoop;
          }
#endif

          k_chunks--;
        }

        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 0), svget4_f32(vb_x4_0, 0));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 0), svget4_f32(vb_x4_0, 1));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 1), svget4_f32(vb_x4_0, 0));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 1), svget4_f32(vb_x4_0, 1));
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 2), svget4_f32(vb_x4_0, 2));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 2), svget4_f32(vb_x4_0, 3));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 3), svget4_f32(vb_x4_0, 2));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 3), svget4_f32(vb_x4_0, 3));

        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 0), svget4_f32(vb_x4_1, 0));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 0), svget4_f32(vb_x4_1, 1));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 1), svget4_f32(vb_x4_1, 0));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 1), svget4_f32(vb_x4_1, 1));
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 2), svget4_f32(vb_x4_1, 2));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 2), svget4_f32(vb_x4_1, 3));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 3), svget4_f32(vb_x4_1, 2));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 3), svget4_f32(vb_x4_1, 3));
      }
      k %= 4;

      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      while (k != 0) {
        svfloat32x2_t va_x2_0 = svld1_f32_x2(pcnt_32_all, a0); a0 += 2*vl;
        svfloat32_t va0 = svget2(va_x2_0, 0);
        svfloat32_t va1 = svget2(va_x2_0, 1);

        svfloat32x2_t vb_x2 = svld1_f32_x2(pcnt_32_all, w0); w0 += 2*vl;
        svfloat32_t vb0 = svget2(vb_x2, 0);
        svfloat32_t vb1 = svget2(vb_x2, 1);

      svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va0, vb0);
      svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va0, vb1);
      svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va1, vb0);
      svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va1, vb1);
      k -= 1;
    }


      float* c_inc = c0;
      svcount_t pn = svwhilelt_c32_u64(0, nc_rem, 2);

      size_t m_chunk_0 = (m_chunk < vl) ? m_chunk : vl;
      size_t m_chunk_0_vec4 = m_chunk_0 & ~3;
      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (size_t i = 0; i < m_chunk_0_vec4; i += 4) {
          svfloat32x4_t z0x4 = svread_hor_za32_f32_vg4(0, i);
          svfloat32x4_t z1x4 = svread_hor_za32_f32_vg4(1, i);
          z0x4 = svclamp_single_f32_x4(z0x4, vmin, vmax);
          z1x4 = svclamp_single_f32_x4(z1x4, vmin, vmax);

          svfloat32x2_t z01 = svundef2_f32();
          z01 = svcreate2(svget4_f32(z0x4, 0), svget4_f32(z1x4, 0));
          svst1_f32_x2(pn, c_inc + 0, z01);

          z01 = svcreate2(svget4_f32(z0x4, 1), svget4_f32(z1x4, 1));
          svst1_f32_x2(pn, (float*)((uintptr_t)c_inc + cm_stride), z01);

          z01 = svcreate2(svget4_f32(z0x4, 2), svget4_f32(z1x4, 2));
          svst1_f32_x2(pn, (float*)((uintptr_t)c_inc + 2 * cm_stride), z01);

          z01 = svcreate2(svget4_f32(z0x4, 3), svget4_f32(z1x4, 3));
          svst1_f32_x2(pn, (float*)((uintptr_t)c_inc + 3 * cm_stride), z01);

          c_inc = (float*) ((uintptr_t)c_inc + 4 * cm_stride);
      }

      if (m_chunk_0_vec4 < m_chunk_0) {
          size_t i = m_chunk_0_vec4;
          svfloat32x4_t z0x4 = svread_hor_za32_f32_vg4(0, i);
          svfloat32x4_t z1x4 = svread_hor_za32_f32_vg4(1, i);
          z0x4 = svclamp_single_f32_x4(z0x4, vmin, vmax);
          z1x4 = svclamp_single_f32_x4(z1x4, vmin, vmax);

          svfloat32x2_t z01 = svundef2_f32();
          z01 = svcreate2(svget4_f32(z0x4, 0), svget4_f32(z1x4, 0));
          svst1_f32_x2(pn, c_inc + 0, z01);

          if (i + 1 < m_chunk_0) {
              z01 = svcreate2(svget4_f32(z0x4, 1), svget4_f32(z1x4, 1));
              svst1_f32_x2(pn, (float*)((uintptr_t)c_inc + cm_stride), z01);
          }
          if (i + 2 < m_chunk_0) {
              z01 = svcreate2(svget4_f32(z0x4, 2), svget4_f32(z1x4, 2));
              svst1_f32_x2(pn, (float*)((uintptr_t)c_inc + 2 * cm_stride), z01);
          }
          c_inc = (float*) ((uintptr_t)c_inc + (m_chunk_0 - m_chunk_0_vec4) * cm_stride);
      }

      c_inc = (float*)((uintptr_t)c0 + vl * cm_stride);

      if (m_chunk > vl) {
          size_t m_chunk_1_rem = m_chunk - vl;
          size_t m_chunk_1_vec4 = m_chunk_1_rem & ~3;
          XNN_PRAGMA_CLANG("clang loop unroll(disable)")
          for (size_t i = 0; i < m_chunk_1_vec4; i += 4) {
              svfloat32x4_t z2x4 = svread_hor_za32_f32_vg4(2, i);
              svfloat32x4_t z3x4 = svread_hor_za32_f32_vg4(3, i);
              z2x4 = svclamp_single_f32_x4(z2x4, vmin, vmax);
              z3x4 = svclamp_single_f32_x4(z3x4, vmin, vmax);

              svfloat32x2_t z23 = svundef2_f32();
              z23 = svcreate2(svget4_f32(z2x4, 0), svget4_f32(z3x4, 0));
              svst1_f32_x2(pn, c_inc + 0, z23);

              z23 = svcreate2(svget4_f32(z2x4, 1), svget4_f32(z3x4, 1));
              svst1_f32_x2(pn, (float*)((uintptr_t)c_inc + cm_stride), z23);

              z23 = svcreate2(svget4_f32(z2x4, 2), svget4_f32(z3x4, 2));
              svst1_f32_x2(pn, (float*)((uintptr_t)c_inc + 2 * cm_stride), z23);

              z23 = svcreate2(svget4_f32(z2x4, 3), svget4_f32(z3x4, 3));
              svst1_f32_x2(pn, (float*)((uintptr_t)c_inc + 3 * cm_stride), z23);

              c_inc = (float*) ((uintptr_t)c_inc + 4 * cm_stride);
          }

          if (m_chunk_1_vec4 < m_chunk_1_rem) {
              size_t i = m_chunk_1_vec4;
              svfloat32x4_t z2x4 = svread_hor_za32_f32_vg4(2, i);
              svfloat32x4_t z3x4 = svread_hor_za32_f32_vg4(3, i);
              z2x4 = svclamp_single_f32_x4(z2x4, vmin, vmax);
              z3x4 = svclamp_single_f32_x4(z3x4, vmin, vmax);

              svfloat32x2_t z23 = svundef2_f32();
              z23 = svcreate2(svget4_f32(z2x4, 0), svget4_f32(z3x4, 0));
              svst1_f32_x2(pn, c_inc + 0, z23);

              if (i + 1 < m_chunk_1_rem) {
                  z23 = svcreate2(svget4_f32(z2x4, 1), svget4_f32(z3x4, 1));
                  svst1_f32_x2(pn, (float*)((uintptr_t)c_inc + cm_stride), z23);
              }
              if (i + 2 < m_chunk_1_rem) {
                  z23 = svcreate2(svget4_f32(z2x4, 2), svget4_f32(z3x4, 2));
                  svst1_f32_x2(pn, (float*)((uintptr_t)c_inc + 2 * cm_stride), z23);
              }
              c_inc = (float*) ((uintptr_t)c_inc + (m_chunk_1_rem - m_chunk_1_vec4) * cm_stride);
          }
      }

      if (nc_rem > 2 * vl) {
          c0 = (float*)((uintptr_t)c0 + cn_stride);
          nc_rem -= 2 * vl;
      } else {
          nc_rem = 0;
      }

    } while (nc_rem != 0);

#ifdef PREFETCH_W_MATRIX
    if (need_prfm) {
      if (prfm_acc_w_size <= (total_w_size - 3 * (8 * vl))) {
        prfm_acc_w_size += 3 * (8 * vl);
        __builtin_prefetch(prfm_ptr + 0 * kTrailerPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr + 1 * kTrailerPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr + 2 * kTrailerPrfmStride, 0, 2);
        prfm_ptr += 3 * (8 * vl);
      } else if (prfm_acc_w_size <= (total_w_size - 2 * (8 * vl))) {
        prfm_acc_w_size += 2 * (8 * vl);
        __builtin_prefetch(prfm_ptr + 0 * kTrailerPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr + 1 * kTrailerPrfmStride, 0, 2);
        prfm_ptr += 2 * (8 * vl);
      } else if (prfm_acc_w_size <= (total_w_size - 1 * (8 * vl))) {
        prfm_acc_w_size += 1 * (8 * vl);
        __builtin_prefetch(prfm_ptr + 0 * kTrailerPrfmStride, 0, 2);
        prfm_ptr += 1 * (8 * vl);
      }
    }
#endif
  }
}

__arm_locally_streaming __arm_new("za")
static void pf32_gemm_32x32_sme2_mr32_n16(
    size_t mr,
    size_t nc,
    size_t kc,
    const void*  a,
    const float* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    float min,
    float max)
{
#ifdef PREFETCH_W_MATRIX
#if defined(__APPLE__)
  const size_t prfm_min_w_size = (16 * 1024 * 1024);
#else
  const size_t prfm_min_w_size = (1024 * 1024);
#endif
#endif
  (void) cn_stride;
  size_t vl = svcntw();

  svbool_t pg_32_all    = svptrue_b32();
  svcount_t pcnt_32_all = svptrue_c32();

  svfloat32_t vmin = svdup_f32(min);
  svfloat32_t vmax = svdup_f32(max);

  const float* a0 = (const float*) a;
  const float* w0 = w;

#ifdef PREFETCH_W_MATRIX
  const size_t kPrefetchDistance = 16;
  const size_t kBytesPerKLoop = 32 * vl;
  const size_t kCacheLineSize = 64;
  const size_t kCacheLinesPerLoop = kBytesPerKLoop / kCacheLineSize;
  const size_t kPrfmStride = kBytesPerKLoop / 8;
  const size_t kTrailerPrfmStride = 8 * vl;

  size_t total_w_size = (kc + 1) * (8 * vl);
  bool need_prfm = (total_w_size >= kPrefetchDistance * kBytesPerKLoop) && (total_w_size > prfm_min_w_size);

  size_t prfm_acc_w_size = 0;
  const int8_t* prfm_ptr = (const int8_t*) w;

  if (need_prfm) {
    prfm_acc_w_size = kPrefetchDistance * kBytesPerKLoop;
    int i = 0;
    for (; i < (kPrefetchDistance * kCacheLinesPerLoop); i++) {
      __builtin_prefetch(prfm_ptr + kCacheLineSize * i, 0, 2);
    }
    prfm_ptr += kCacheLineSize * i;
  }
#endif

  svfloat32_t vb_bias = svld1_f32(pg_32_all, w0); w0 += 2*vl;

  svfloat32_t f32_one = svdup_f32(1.0f);
  svzero_za();
  svmopa_za32_f32_m(0, pg_32_all, pg_32_all, f32_one, vb_bias);
  svmopa_za32_f32_m(2, pg_32_all, pg_32_all, f32_one, vb_bias);

  size_t k = kc;
  size_t k_chunks = k / 4;
  if (k_chunks > 0) {
    svfloat32x4_t va_x4_0 = svld1_f32_x4(pcnt_32_all, a0);
    svfloat32x4_t va_x4_1 = svld1_f32_x4(pcnt_32_all, a0 + 4*vl);
    a0 += 8*vl;
    svfloat32_t vb0 = svld1_f32(pg_32_all, w0);
    svfloat32_t vb1 = svld1_f32(pg_32_all, w0 + 2*vl);
    svfloat32_t vb2 = svld1_f32(pg_32_all, w0 + 4*vl);
    svfloat32_t vb3 = svld1_f32(pg_32_all, w0 + 6*vl);
    w0 += 8*vl;

#ifdef PREFETCH_W_MATRIX
    if (need_prfm && (prfm_acc_w_size < (total_w_size - kBytesPerKLoop))) {
      prfm_acc_w_size += kBytesPerKLoop;
      __builtin_prefetch(prfm_ptr + 0 * kPrfmStride, 0, 2);
      __builtin_prefetch(prfm_ptr + 1 * kPrfmStride, 0, 2);
      __builtin_prefetch(prfm_ptr + 2 * kPrfmStride, 0, 2);
      __builtin_prefetch(prfm_ptr + 3 * kPrfmStride, 0, 2);
      __builtin_prefetch(prfm_ptr + 4 * kPrfmStride, 0, 2);
      __builtin_prefetch(prfm_ptr + 5 * kPrfmStride, 0, 2);
      __builtin_prefetch(prfm_ptr + 6 * kPrfmStride, 0, 2);
      __builtin_prefetch(prfm_ptr + 7 * kPrfmStride, 0, 2);
      prfm_ptr += kBytesPerKLoop;
    }
#endif

    k_chunks--;

    while (k_chunks > 0) {
      svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 0), vb0);
      svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 1), vb0);
      svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 2), vb1);
      svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 3), vb1);

      va_x4_0 = svld1_f32_x4(pcnt_32_all, a0);
      vb0 = svld1_f32(pg_32_all, w0);
      vb1 = svld1_f32(pg_32_all, w0 + 2*vl);

      svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 0), vb2);
      svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 1), vb2);
      svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 2), vb3);
      svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 3), vb3);

      va_x4_1 = svld1_f32_x4(pcnt_32_all, a0 + 4*vl);
      vb2 = svld1_f32(pg_32_all, w0 + 4*vl);
      vb3 = svld1_f32(pg_32_all, w0 + 6*vl);
      a0 += 8*vl;
      w0 += 8*vl;

#ifdef PREFETCH_W_MATRIX
      if (need_prfm && (prfm_acc_w_size < (total_w_size - kBytesPerKLoop))) {
        prfm_acc_w_size += kBytesPerKLoop;
        __builtin_prefetch(prfm_ptr + 0 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr + 1 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr + 2 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr + 3 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr + 4 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr + 5 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr + 6 * kPrfmStride, 0, 2);
        __builtin_prefetch(prfm_ptr + 7 * kPrfmStride, 0, 2);
        prfm_ptr += kBytesPerKLoop;
      }
#endif

      k_chunks--;
    }

    svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 0), vb0);
    svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 1), vb0);
    svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 2), vb1);
    svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 3), vb1);

    svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 0), vb2);
    svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 1), vb2);
    svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 2), vb3);
    svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 3), vb3);
  }
  k %= 4;

  XNN_PRAGMA_CLANG("clang loop unroll(disable)")
  while (k != 0) {
    svfloat32x2_t va_x2_0 = svld1_f32_x2(pcnt_32_all, a0); a0 += 2*vl;
    svfloat32_t vb0 = svld1_f32(pg_32_all, w0); w0 += 2*vl;

    svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget2(va_x2_0, 0), vb0);
    svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget2(va_x2_0, 1), vb0);
    k -= 1;
  }

#ifdef PREFETCH_W_MATRIX
  if (need_prfm) {
    if (prfm_acc_w_size <= (total_w_size - 3 * (8 * vl))) {
      prfm_acc_w_size += 3 * (8 * vl);
      __builtin_prefetch(prfm_ptr + 0 * kTrailerPrfmStride, 0, 2);
      __builtin_prefetch(prfm_ptr + 1 * kTrailerPrfmStride, 0, 2);
      __builtin_prefetch(prfm_ptr + 2 * kTrailerPrfmStride, 0, 2);
      prfm_ptr += 3 * (8 * vl);
    } else if (prfm_acc_w_size <= (total_w_size - 2 * (8 * vl))) {
      prfm_acc_w_size += 2 * (8 * vl);
      __builtin_prefetch(prfm_ptr + 0 * kTrailerPrfmStride, 0, 2);
      __builtin_prefetch(prfm_ptr + 1 * kTrailerPrfmStride, 0, 2);
      prfm_ptr += 2 * (8 * vl);
    } else if (prfm_acc_w_size <= (total_w_size - 1 * (8 * vl))) {
      prfm_acc_w_size += 1 * (8 * vl);
      __builtin_prefetch(prfm_ptr + 0 * kTrailerPrfmStride, 0, 2);
      prfm_ptr += 1 * (8 * vl);
    }
  }
#endif

  svbool_t pn = svwhilelt_b32_u64(0, nc);
  float* c_inc = c;
  XNN_PRAGMA_CLANG("clang loop unroll(disable)")
  for (size_t i = 0; i < vl; i += 4) {
    svfloat32x4_t z0x4 = svread_hor_za32_f32_vg4(0, i);
    svfloat32x4_t z1x4 = svread_hor_za32_f32_vg4(1, i);
    svfloat32x4_t zx4 = svcreate4(svadd_f32_x(pg_32_all, svget4_f32(z0x4, 0), svget4_f32(z1x4, 0)),
                                  svadd_f32_x(pg_32_all, svget4_f32(z0x4, 1), svget4_f32(z1x4, 1)),
                                  svadd_f32_x(pg_32_all, svget4_f32(z0x4, 2), svget4_f32(z1x4, 2)),
                                  svadd_f32_x(pg_32_all, svget4_f32(z0x4, 3), svget4_f32(z1x4, 3)));
    zx4 = svclamp_single_f32_x4(zx4, vmin, vmax);
    svst1_f32(pn, c_inc, svget4_f32(zx4, 0));
    svst1_f32(pn, (float*)((uintptr_t)c_inc + cm_stride), svget4_f32(zx4, 1));
    svst1_f32(pn, (float*)((uintptr_t)c_inc + 2 * cm_stride), svget4_f32(zx4, 2));
    svst1_f32(pn, (float*)((uintptr_t)c_inc + 3 * cm_stride), svget4_f32(zx4, 3));
    c_inc = (float*) ((uintptr_t)c_inc + 4 * cm_stride);
  }

  size_t m_chunk_1 = mr - vl;
  XNN_PRAGMA_CLANG("clang loop unroll(disable)")
  for (size_t i = 0; i < m_chunk_1; i += 4) {
    svfloat32x4_t z2x4 = svread_hor_za32_f32_vg4(2, i);
    svfloat32x4_t z3x4 = svread_hor_za32_f32_vg4(3, i);
    svfloat32x4_t zx4 = svcreate4(svadd_f32_x(pg_32_all, svget4_f32(z2x4, 0), svget4_f32(z3x4, 0)),
                                  svadd_f32_x(pg_32_all, svget4_f32(z2x4, 1), svget4_f32(z3x4, 1)),
                                  svadd_f32_x(pg_32_all, svget4_f32(z2x4, 2), svget4_f32(z3x4, 2)),
                                  svadd_f32_x(pg_32_all, svget4_f32(z2x4, 3), svget4_f32(z3x4, 3)));
    zx4 = svclamp_single_f32_x4(zx4, vmin, vmax);
    svst1_f32(pn, c_inc, svget4_f32(zx4, 0));
    if (i + 1 < m_chunk_1) {
      svst1_f32(pn, (float*)((uintptr_t)c_inc + cm_stride), svget4_f32(zx4, 1));
    }
    if (i + 2 < m_chunk_1) {
      svst1_f32(pn, (float*)((uintptr_t)c_inc + 2 * cm_stride), svget4_f32(zx4, 2));
    }
    if (i + 3 < m_chunk_1) {
      svst1_f32(pn, (float*)((uintptr_t)c_inc + 3 * cm_stride), svget4_f32(zx4, 3));
    }
    c_inc = (float*) ((uintptr_t)c_inc + 4 * cm_stride);
  }
}

void xnn_pf32_gemm_minmax_ukernel_32x32__sme2(
    size_t mr,
    size_t nc,
    size_t kc,
    const void*  a,
    const float* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* restrict params)
{
  if XNN_UNLIKELY(mr == 0 || nc == 0) {
    return;
  }
  const size_t vl = svcntsw();
  if (mr <= vl) {
    pf32_gemm_32x32_sme2_mr16(mr, nc, kc / sizeof(float), a, w, c, cm_stride, sizeof(float),
                                                  params->scalar.min, params->scalar.max);
  } else if (nc <= vl && mr <= 2 * vl) {
    pf32_gemm_32x32_sme2_mr32_n16(mr, nc, kc / sizeof(float), a, w, c, cm_stride, sizeof(float),
                                                      params->scalar.min, params->scalar.max);
  } else {
    pf32_gemm_32x32_sme2_mr32(mr, nc, kc / sizeof(float), a, w, c, cm_stride, sizeof(float),
                                                  params->scalar.min, params->scalar.max);
  }
}

#undef PREFETCH_W_MATRIX

#endif  // XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
