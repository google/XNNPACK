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

size_t xnn_pf32_gemm_minmax_ukernel_1x32__sme2_get_mr() {
  return 1;
}

size_t xnn_pf32_gemm_minmax_ukernel_1x32__sme2_get_nr() {
  return 2 * svcntsw();
}

#define PREFETCH_W_MATRIX

__arm_locally_streaming __arm_new("za")
static void pf32_gemm_1x32_sme2_impl(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* a,
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
  const size_t nr = 2 * vl;
  cn_stride = nr * cn_stride;
  const float* a0 = a;
  float* c0 = c;

  svbool_t pg_32_msk_4 = svwhilelt_b32_u64(0, 4);

#ifdef PREFETCH_W_MATRIX
  const size_t kPrefetchDistance = 16;
  const size_t kBytesPerKLoop = 4 * nr * sizeof(float);
  const size_t kCacheLineSize = 64;
  const size_t kCacheLinesPerLoop = kBytesPerKLoop / kCacheLineSize;
  const size_t kPrfmStride = kBytesPerKLoop / 8;
  const size_t kTrailerPrfmStride = 8 * vl;

  size_t nc_rup_nr = ((nc + nr - 1) / nr) * nr;
  size_t total_w_size = ((kc * nc_rup_nr) + nc_rup_nr) * sizeof(float);
  bool need_prfm = (total_w_size > prfm_min_w_size);
#endif

  while (nc != 0) {
    size_t blocks_processed = (nc + nr - 1) / nr;

    const size_t stream_stride = nr + kc * nr;

#ifdef PREFETCH_W_MATRIX
    size_t w_block_bytes = stream_stride * sizeof(float);
#endif

    if (blocks_processed >= 4) {
      blocks_processed = 4;
      // Branch 2: Focused on N = 128 (blocks_processed == 4)
      svcount_t pcnt_0 = svwhilelt_c32_u64(0, nc, 2);
      svcount_t pcnt_1 = svwhilelt_c32_u64(nr, nc, 2);
      svcount_t pcnt_2 = svwhilelt_c32_u64(2 * nr, nc, 2);
      svcount_t pcnt_3 = svwhilelt_c32_u64(3 * nr, nc, 2);

      const float* w0 = w;
      const float* w1 = w0 + stream_stride;
      const float* w2 = w1 + stream_stride;
      const float* w3 = w2 + stream_stride;
      w = w + 4 * stream_stride;

      const float* wb0 = w0;
      const float* wb1 = w1;
      const float* wb2 = w2;
      const float* wb3 = w3;
      w0 += nr; w1 += nr; w2 += nr; w3 += nr;

      svzero_za();
      size_t k = kc;
      a0 = a;

#ifdef PREFETCH_W_MATRIX
      bool need_prfm_0 = (w_block_bytes >= kPrefetchDistance * kBytesPerKLoop) && need_prfm && (nc > 0);
      bool need_prfm_1 = (w_block_bytes >= kPrefetchDistance * kBytesPerKLoop) && need_prfm && (nc > 1 * nr);
      bool need_prfm_2 = (w_block_bytes >= kPrefetchDistance * kBytesPerKLoop) && need_prfm && (nc > 2 * nr);
      bool need_prfm_3 = (w_block_bytes >= kPrefetchDistance * kBytesPerKLoop) && need_prfm && (nc > 3 * nr);

      size_t prfm_acc_w_size_0 = 0;
      size_t prfm_acc_w_size_1 = 0;
      size_t prfm_acc_w_size_2 = 0;
      size_t prfm_acc_w_size_3 = 0;

      const int8_t* prfm_ptr_0 = (const int8_t*) wb0;
      const int8_t* prfm_ptr_1 = (const int8_t*) wb1;
      const int8_t* prfm_ptr_2 = (const int8_t*) wb2;
      const int8_t* prfm_ptr_3 = (const int8_t*) wb3;

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
      if (need_prfm_2) {
        prfm_acc_w_size_2 = kPrefetchDistance * kBytesPerKLoop;
        int i = 0;
        for (; i < (kPrefetchDistance * kCacheLinesPerLoop); i++) {
          __builtin_prefetch(prfm_ptr_2 + kCacheLineSize * i, 0, 2);
        }
        prfm_ptr_2 += kCacheLineSize * i;
      }
      if (need_prfm_3) {
        prfm_acc_w_size_3 = kPrefetchDistance * kBytesPerKLoop;
        int i = 0;
        for (; i < (kPrefetchDistance * kCacheLinesPerLoop); i++) {
          __builtin_prefetch(prfm_ptr_3 + kCacheLineSize * i, 0, 2);
        }
        prfm_ptr_3 += kCacheLineSize * i;
      }
#endif

      size_t k_chunks = kc / 4;
      if (k_chunks > 0) {
        svfloat32_t va0 = svld1rq_f32(pg_32_msk_4, a0); a0 += 4;

        // Prequel: Preload W for Lane 0 AND Lane 1 (2 lanes * 4 streams * 2 registers = 16 vector registers)
        svfloat32x2_t vb_s0_l0 = svld1_f32_x2(pcnt_0, w0); w0 += nr;
        svfloat32x2_t vb_s1_l0 = svld1_f32_x2(pcnt_1, w1); w1 += nr;
        svfloat32x2_t vb_s2_l0 = svld1_f32_x2(pcnt_2, w2); w2 += nr;
        svfloat32x2_t vb_s3_l0 = svld1_f32_x2(pcnt_3, w3); w3 += nr;

        svfloat32x2_t vb_s0_l1 = svld1_f32_x2(pcnt_0, w0); w0 += nr;
        svfloat32x2_t vb_s1_l1 = svld1_f32_x2(pcnt_1, w1); w1 += nr;
        svfloat32x2_t vb_s2_l1 = svld1_f32_x2(pcnt_2, w2); w2 += nr;
        svfloat32x2_t vb_s3_l1 = svld1_f32_x2(pcnt_3, w3); w3 += nr;

        k_chunks--;

        while (k_chunks > 0) {
          svmla_lane_za32_f32_vg1x2(0, vb_s0_l0, va0, 0);
          svmla_lane_za32_f32_vg1x2(1, vb_s1_l0, va0, 0);
          svmla_lane_za32_f32_vg1x2(2, vb_s2_l0, va0, 0);
          svmla_lane_za32_f32_vg1x2(3, vb_s3_l0, va0, 0);

          svfloat32x2_t vb_s0_l2 = svld1_f32_x2(pcnt_0, w0); w0 += nr;
          svfloat32x2_t vb_s1_l2 = svld1_f32_x2(pcnt_1, w1); w1 += nr;
          svfloat32x2_t vb_s2_l2 = svld1_f32_x2(pcnt_2, w2); w2 += nr;
          svfloat32x2_t vb_s3_l2 = svld1_f32_x2(pcnt_3, w3); w3 += nr;

          svmla_lane_za32_f32_vg1x2(4, vb_s0_l1, va0, 1);
          svmla_lane_za32_f32_vg1x2(5, vb_s1_l1, va0, 1);
          svmla_lane_za32_f32_vg1x2(6, vb_s2_l1, va0, 1);
          svmla_lane_za32_f32_vg1x2(7, vb_s3_l1, va0, 1);

          svfloat32x2_t vb_s0_l3 = svld1_f32_x2(pcnt_0, w0); w0 += nr;
          svfloat32x2_t vb_s1_l3 = svld1_f32_x2(pcnt_1, w1); w1 += nr;
          svfloat32x2_t vb_s2_l3 = svld1_f32_x2(pcnt_2, w2); w2 += nr;
          svfloat32x2_t vb_s3_l3 = svld1_f32_x2(pcnt_3, w3); w3 += nr;

          svmla_lane_za32_f32_vg1x2(0, vb_s0_l2, va0, 2);
          svmla_lane_za32_f32_vg1x2(1, vb_s1_l2, va0, 2);
          svmla_lane_za32_f32_vg1x2(2, vb_s2_l2, va0, 2);
          svmla_lane_za32_f32_vg1x2(3, vb_s3_l2, va0, 2);

          vb_s0_l0 = svld1_f32_x2(pcnt_0, w0); w0 += nr;
          vb_s1_l0 = svld1_f32_x2(pcnt_1, w1); w1 += nr;
          vb_s2_l0 = svld1_f32_x2(pcnt_2, w2); w2 += nr;
          vb_s3_l0 = svld1_f32_x2(pcnt_3, w3); w3 += nr;

          svmla_lane_za32_f32_vg1x2(4, vb_s0_l3, va0, 3);
          svmla_lane_za32_f32_vg1x2(5, vb_s1_l3, va0, 3);
          svmla_lane_za32_f32_vg1x2(6, vb_s2_l3, va0, 3);
          svmla_lane_za32_f32_vg1x2(7, vb_s3_l3, va0, 3);

          vb_s0_l1 = svld1_f32_x2(pcnt_0, w0); w0 += nr;
          vb_s1_l1 = svld1_f32_x2(pcnt_1, w1); w1 += nr;
          vb_s2_l1 = svld1_f32_x2(pcnt_2, w2); w2 += nr;
          vb_s3_l1 = svld1_f32_x2(pcnt_3, w3); w3 += nr;

          va0 = svld1rq_f32(pg_32_msk_4, a0); a0 += 4;

#ifdef PREFETCH_W_MATRIX
          if (need_prfm_0 && (prfm_acc_w_size_0 < (w_block_bytes - kBytesPerKLoop))) {
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
          if (need_prfm_1 && (prfm_acc_w_size_1 < (w_block_bytes - kBytesPerKLoop))) {
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
          if (need_prfm_2 && (prfm_acc_w_size_2 < (w_block_bytes - kBytesPerKLoop))) {
            prfm_acc_w_size_2 += kBytesPerKLoop;
            __builtin_prefetch(prfm_ptr_2 + 0 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_2 + 1 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_2 + 2 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_2 + 3 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_2 + 4 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_2 + 5 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_2 + 6 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_2 + 7 * kPrfmStride, 0, 2);
            prfm_ptr_2 += kBytesPerKLoop;
          }
          if (need_prfm_3 && (prfm_acc_w_size_3 < (w_block_bytes - kBytesPerKLoop))) {
            prfm_acc_w_size_3 += kBytesPerKLoop;
            __builtin_prefetch(prfm_ptr_3 + 0 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_3 + 1 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_3 + 2 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_3 + 3 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_3 + 4 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_3 + 5 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_3 + 6 * kPrfmStride, 0, 2);
            __builtin_prefetch(prfm_ptr_3 + 7 * kPrfmStride, 0, 2);
            prfm_ptr_3 += kBytesPerKLoop;
          }
#endif
          k_chunks--;
        }

        // Epilogue: Compute last preloaded W:
        svmla_lane_za32_f32_vg1x2(0, vb_s0_l0, va0, 0);
        svmla_lane_za32_f32_vg1x2(1, vb_s1_l0, va0, 0);
        svmla_lane_za32_f32_vg1x2(2, vb_s2_l0, va0, 0);
        svmla_lane_za32_f32_vg1x2(3, vb_s3_l0, va0, 0);

        svmla_lane_za32_f32_vg1x2(4, vb_s0_l1, va0, 1);
        svmla_lane_za32_f32_vg1x2(5, vb_s1_l1, va0, 1);
        svmla_lane_za32_f32_vg1x2(6, vb_s2_l1, va0, 1);
        svmla_lane_za32_f32_vg1x2(7, vb_s3_l1, va0, 1);

        svfloat32x2_t vb_s0_l2 = svld1_f32_x2(pcnt_0, w0); w0 += nr;
        svfloat32x2_t vb_s1_l2 = svld1_f32_x2(pcnt_1, w1); w1 += nr;
        svfloat32x2_t vb_s2_l2 = svld1_f32_x2(pcnt_2, w2); w2 += nr;
        svfloat32x2_t vb_s3_l2 = svld1_f32_x2(pcnt_3, w3); w3 += nr;

        svmla_lane_za32_f32_vg1x2(0, vb_s0_l2, va0, 2);
        svmla_lane_za32_f32_vg1x2(1, vb_s1_l2, va0, 2);
        svmla_lane_za32_f32_vg1x2(2, vb_s2_l2, va0, 2);
        svmla_lane_za32_f32_vg1x2(3, vb_s3_l2, va0, 2);

        svfloat32x2_t vb_s0_l3 = svld1_f32_x2(pcnt_0, w0); w0 += nr;
        svfloat32x2_t vb_s1_l3 = svld1_f32_x2(pcnt_1, w1); w1 += nr;
        svfloat32x2_t vb_s2_l3 = svld1_f32_x2(pcnt_2, w2); w2 += nr;
        svfloat32x2_t vb_s3_l3 = svld1_f32_x2(pcnt_3, w3); w3 += nr;

        svmla_lane_za32_f32_vg1x2(4, vb_s0_l3, va0, 3);
        svmla_lane_za32_f32_vg1x2(5, vb_s1_l3, va0, 3);
        svmla_lane_za32_f32_vg1x2(6, vb_s2_l3, va0, 3);
        svmla_lane_za32_f32_vg1x2(7, vb_s3_l3, va0, 3);
      }
      k = kc % 4;

      if (k >= 2) {
        svfloat32_t va0 = svld1rq_f32(svwhilelt_b32_u64(0, 2), a0);
        a0 += 2;
        svfloat32x2_t vb_s0 = svld1_f32_x2(pcnt_0, w0);
        svfloat32x2_t vb_s1 = svld1_f32_x2(pcnt_1, w1);
        svfloat32x2_t vb_s2 = svld1_f32_x2(pcnt_2, w2);
        svfloat32x2_t vb_s3 = svld1_f32_x2(pcnt_3, w3);
        svmla_lane_za32_f32_vg1x2(0, vb_s0, va0, 0);
        svmla_lane_za32_f32_vg1x2(1, vb_s1, va0, 0);
        svmla_lane_za32_f32_vg1x2(2, vb_s2, va0, 0);
        svmla_lane_za32_f32_vg1x2(3, vb_s3, va0, 0);

        vb_s0 = svld1_f32_x2(pcnt_0, w0 + nr);
        vb_s1 = svld1_f32_x2(pcnt_1, w1 + nr);
        vb_s2 = svld1_f32_x2(pcnt_2, w2 + nr);
        vb_s3 = svld1_f32_x2(pcnt_3, w3 + nr);
        svmla_lane_za32_f32_vg1x2(4, vb_s0, va0, 1);
        svmla_lane_za32_f32_vg1x2(5, vb_s1, va0, 1);
        svmla_lane_za32_f32_vg1x2(6, vb_s2, va0, 1);
        svmla_lane_za32_f32_vg1x2(7, vb_s3, va0, 1);

        w0 += 2 * nr; w1 += 2 * nr; w2 += 2 * nr; w3 += 2 * nr;
        k -= 2;
      }

      if XNN_UNLIKELY(k != 0) {
        svfloat32_t va0 = svld1rq_f32(svwhilelt_b32_u64(0, 1), a0);
        a0 += 1;
        svfloat32x2_t vb_s0 = svld1_f32_x2(pcnt_0, w0);
        svfloat32x2_t vb_s1 = svld1_f32_x2(pcnt_1, w1);
        svfloat32x2_t vb_s2 = svld1_f32_x2(pcnt_2, w2);
        svfloat32x2_t vb_s3 = svld1_f32_x2(pcnt_3, w3);
        svmla_lane_za32_f32_vg1x2(0, vb_s0, va0, 0);
        svmla_lane_za32_f32_vg1x2(1, vb_s1, va0, 0);
        svmla_lane_za32_f32_vg1x2(2, vb_s2, va0, 0);
        svmla_lane_za32_f32_vg1x2(3, vb_s3, va0, 0);
        w0 += nr; w1 += nr; w2 += nr; w3 += nr;
      }

#ifdef PREFETCH_W_MATRIX
      if (need_prfm_0) {
        if (prfm_acc_w_size_0 <= (w_block_bytes - 3 * (8 * vl))) {
          prfm_acc_w_size_0 += 3 * (8 * vl);
          __builtin_prefetch(prfm_ptr_0 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 2 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_0 += 3 * (8 * vl);
        } else if (prfm_acc_w_size_0 <= (w_block_bytes - 2 * (8 * vl))) {
          prfm_acc_w_size_0 += 2 * (8 * vl);
          __builtin_prefetch(prfm_ptr_0 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 1 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_0 += 2 * (8 * vl);
        } else if (prfm_acc_w_size_0 <= (w_block_bytes - 1 * (8 * vl))) {
          prfm_acc_w_size_0 += 1 * (8 * vl);
          __builtin_prefetch(prfm_ptr_0 + 0 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_0 += 1 * (8 * vl);
        }
      }
      if (need_prfm_1) {
        if (prfm_acc_w_size_1 <= (w_block_bytes - 3 * (8 * vl))) {
          prfm_acc_w_size_1 += 3 * (8 * vl);
          __builtin_prefetch(prfm_ptr_1 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 2 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_1 += 3 * (8 * vl);
        } else if (prfm_acc_w_size_1 <= (w_block_bytes - 2 * (8 * vl))) {
          prfm_acc_w_size_1 += 2 * (8 * vl);
          __builtin_prefetch(prfm_ptr_1 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 1 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_1 += 2 * (8 * vl);
        } else if (prfm_acc_w_size_1 <= (w_block_bytes - 1 * (8 * vl))) {
          prfm_acc_w_size_1 += 1 * (8 * vl);
          __builtin_prefetch(prfm_ptr_1 + 0 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_1 += 1 * (8 * vl);
        }
      }
      if (need_prfm_2) {
        if (prfm_acc_w_size_2 <= (w_block_bytes - 3 * (8 * vl))) {
          prfm_acc_w_size_2 += 3 * (8 * vl);
          __builtin_prefetch(prfm_ptr_2 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_2 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_2 + 2 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_2 += 3 * (8 * vl);
        } else if (prfm_acc_w_size_2 <= (w_block_bytes - 2 * (8 * vl))) {
          prfm_acc_w_size_2 += 2 * (8 * vl);
          __builtin_prefetch(prfm_ptr_2 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_2 + 1 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_2 += 2 * (8 * vl);
        } else if (prfm_acc_w_size_2 <= (w_block_bytes - 1 * (8 * vl))) {
          prfm_acc_w_size_2 += 1 * (8 * vl);
          __builtin_prefetch(prfm_ptr_2 + 0 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_2 += 1 * (8 * vl);
        }
      }
      if (need_prfm_3) {
        if (prfm_acc_w_size_3 <= (w_block_bytes - 3 * (8 * vl))) {
          prfm_acc_w_size_3 += 3 * (8 * vl);
          __builtin_prefetch(prfm_ptr_3 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_3 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_3 + 2 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_3 += 3 * (8 * vl);
        } else if (prfm_acc_w_size_3 <= (w_block_bytes - 2 * (8 * vl))) {
          prfm_acc_w_size_3 += 2 * (8 * vl);
          __builtin_prefetch(prfm_ptr_3 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_3 + 1 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_3 += 2 * (8 * vl);
        } else if (prfm_acc_w_size_3 <= (w_block_bytes - 1 * (8 * vl))) {
          prfm_acc_w_size_3 += 1 * (8 * vl);
          __builtin_prefetch(prfm_ptr_3 + 0 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_3 += 1 * (8 * vl);
        }
      }
#endif

      svfloat32_t vmin = svdup_f32(min);
      svfloat32_t vmax = svdup_f32(max);

      svfloat32x2_t f32_bias_0 = svld1_f32_x2(pcnt_0, wb0);
      svfloat32x2_t f32_bias_1 = svld1_f32_x2(pcnt_1, wb1);
      svfloat32x2_t f32_bias_2 = svld1_f32_x2(pcnt_2, wb2);
      svfloat32x2_t f32_bias_3 = svld1_f32_x2(pcnt_3, wb3);

      svbool_t pg_32_all = svptrue_b32();
      svfloat32x2_t za0_acco = svread_za32_f32_vg1x2(0);
      svfloat32x2_t za1_acco = svread_za32_f32_vg1x2(1);
      svfloat32x2_t za2_acco = svread_za32_f32_vg1x2(2);
      svfloat32x2_t za3_acco = svread_za32_f32_vg1x2(3);
      svfloat32x2_t za4_acco = svread_za32_f32_vg1x2(4);
      svfloat32x2_t za5_acco = svread_za32_f32_vg1x2(5);
      svfloat32x2_t za6_acco = svread_za32_f32_vg1x2(6);
      svfloat32x2_t za7_acco = svread_za32_f32_vg1x2(7);

      za0_acco = svcreate2(svadd_f32_x(pg_32_all, svadd_f32_x(pg_32_all, svget2(za0_acco, 0), svget2(za4_acco, 0)), svget2(f32_bias_0, 0)),
                           svadd_f32_x(pg_32_all, svadd_f32_x(pg_32_all, svget2(za0_acco, 1), svget2(za4_acco, 1)), svget2(f32_bias_0, 1)));
      za1_acco = svcreate2(svadd_f32_x(pg_32_all, svadd_f32_x(pg_32_all, svget2(za1_acco, 0), svget2(za5_acco, 0)), svget2(f32_bias_1, 0)),
                           svadd_f32_x(pg_32_all, svadd_f32_x(pg_32_all, svget2(za1_acco, 1), svget2(za5_acco, 1)), svget2(f32_bias_1, 1)));
      za2_acco = svcreate2(svadd_f32_x(pg_32_all, svadd_f32_x(pg_32_all, svget2(za2_acco, 0), svget2(za6_acco, 0)), svget2(f32_bias_2, 0)),
                           svadd_f32_x(pg_32_all, svadd_f32_x(pg_32_all, svget2(za2_acco, 1), svget2(za6_acco, 1)), svget2(f32_bias_2, 1)));
      za3_acco = svcreate2(svadd_f32_x(pg_32_all, svadd_f32_x(pg_32_all, svget2(za3_acco, 0), svget2(za7_acco, 0)), svget2(f32_bias_3, 0)),
                           svadd_f32_x(pg_32_all, svadd_f32_x(pg_32_all, svget2(za3_acco, 1), svget2(za7_acco, 1)), svget2(f32_bias_3, 1)));

      za0_acco = svclamp(za0_acco, vmin, vmax);
      za1_acco = svclamp(za1_acco, vmin, vmax);
      za2_acco = svclamp(za2_acco, vmin, vmax);
      za3_acco = svclamp(za3_acco, vmin, vmax);

      svst1_f32_x2(pcnt_0, c0, za0_acco);
      svst1_f32_x2(pcnt_1, c0 + nr, za1_acco);
      svst1_f32_x2(pcnt_2, c0 + 2 * nr, za2_acco);
      svst1_f32_x2(pcnt_3, c0 + 3 * nr, za3_acco);

      c0 = (float*) ((uintptr_t) c0 + blocks_processed * cn_stride);
      if (nc >= blocks_processed * nr) {
        nc -= blocks_processed * nr;
      } else {
        nc = 0;
      }
    } else if (blocks_processed >= 2) {
      blocks_processed = 2;
      // Branch 1: Focused on N = 64 (blocks_processed == 2)
      svcount_t pcnt_0 = svwhilelt_c32_u64(0, nc, 2);
      svcount_t pcnt_1 = svwhilelt_c32_u64(nr, nc, 2);

      const float* w0 = w;
      const float* w1 = w0 + stream_stride;
      w = w + 2 * stream_stride;

      const float* wb0 = w0;
      const float* wb1 = w1;
      w0 += nr; w1 += nr;

      svzero_za();
      a0 = a;

#ifdef PREFETCH_W_MATRIX
      bool need_prfm_0 = (w_block_bytes >= kPrefetchDistance * kBytesPerKLoop) && need_prfm && (nc > 0);
      bool need_prfm_1 = (w_block_bytes >= kPrefetchDistance * kBytesPerKLoop) && need_prfm && (nc > 1 * nr);

      size_t prfm_acc_w_size_0 = 0;
      size_t prfm_acc_w_size_1 = 0;

      const int8_t* prfm_ptr_0 = (const int8_t*) wb0;
      const int8_t* prfm_ptr_1 = (const int8_t*) wb1;

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

      size_t k_chunks = kc / 4;
      if (k_chunks > 0) {
        svfloat32_t va0 = svld1rq_f32(pg_32_msk_4, a0); a0 += 4;

        // Prequel: Preload W for all 4 lanes (K=4)
        svfloat32x2_t vb_s0_l0 = svld1_f32_x2(pcnt_0, w0); w0 += nr;
        svfloat32x2_t vb_s1_l0 = svld1_f32_x2(pcnt_1, w1); w1 += nr;

        svfloat32x2_t vb_s0_l1 = svld1_f32_x2(pcnt_0, w0); w0 += nr;
        svfloat32x2_t vb_s1_l1 = svld1_f32_x2(pcnt_1, w1); w1 += nr;

        svfloat32x2_t vb_s0_l2 = svld1_f32_x2(pcnt_0, w0); w0 += nr;
        svfloat32x2_t vb_s1_l2 = svld1_f32_x2(pcnt_1, w1); w1 += nr;

        svfloat32x2_t vb_s0_l3 = svld1_f32_x2(pcnt_0, w0); w0 += nr;
        svfloat32x2_t vb_s1_l3 = svld1_f32_x2(pcnt_1, w1); w1 += nr;

        k_chunks--;

        while (k_chunks > 0) {
          svmla_lane_za32_f32_vg1x2(0, vb_s0_l0, va0, 0);
          svmla_lane_za32_f32_vg1x2(1, vb_s1_l0, va0, 0);

          vb_s0_l0 = svld1_f32_x2(pcnt_0, w0); w0 += nr;
          vb_s1_l0 = svld1_f32_x2(pcnt_1, w1); w1 += nr;

          svmla_lane_za32_f32_vg1x2(2, vb_s0_l1, va0, 1);
          svmla_lane_za32_f32_vg1x2(3, vb_s1_l1, va0, 1);

          vb_s0_l1 = svld1_f32_x2(pcnt_0, w0); w0 += nr;
          vb_s1_l1 = svld1_f32_x2(pcnt_1, w1); w1 += nr;

          svmla_lane_za32_f32_vg1x2(0, vb_s0_l2, va0, 2);
          svmla_lane_za32_f32_vg1x2(1, vb_s1_l2, va0, 2);

          vb_s0_l2 = svld1_f32_x2(pcnt_0, w0); w0 += nr;
          vb_s1_l2 = svld1_f32_x2(pcnt_1, w1); w1 += nr;

          svmla_lane_za32_f32_vg1x2(2, vb_s0_l3, va0, 3);
          svmla_lane_za32_f32_vg1x2(3, vb_s1_l3, va0, 3);

          vb_s0_l3 = svld1_f32_x2(pcnt_0, w0); w0 += nr;
          vb_s1_l3 = svld1_f32_x2(pcnt_1, w1); w1 += nr;

          va0 = svld1rq_f32(pg_32_msk_4, a0); a0 += 4;

#ifdef PREFETCH_W_MATRIX
          if (need_prfm_0 && (prfm_acc_w_size_0 < (w_block_bytes - kBytesPerKLoop))) {
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
          if (need_prfm_1 && (prfm_acc_w_size_1 < (w_block_bytes - kBytesPerKLoop))) {
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

        // Epilogue:
        svmla_lane_za32_f32_vg1x2(0, vb_s0_l0, va0, 0);
        svmla_lane_za32_f32_vg1x2(1, vb_s1_l0, va0, 0);

        svmla_lane_za32_f32_vg1x2(2, vb_s0_l1, va0, 1);
        svmla_lane_za32_f32_vg1x2(3, vb_s1_l1, va0, 1);

        svmla_lane_za32_f32_vg1x2(0, vb_s0_l2, va0, 2);
        svmla_lane_za32_f32_vg1x2(1, vb_s1_l2, va0, 2);

        svmla_lane_za32_f32_vg1x2(2, vb_s0_l3, va0, 3);
        svmla_lane_za32_f32_vg1x2(3, vb_s1_l3, va0, 3);
      }
      size_t k = kc % 4;

      if (k >= 2) {
        svfloat32_t va0 = svld1rq_f32(svwhilelt_b32_u64(0, 2), a0);
        a0 += 2;
        svfloat32x2_t vb_s0 = svld1_f32_x2(pcnt_0, w0);
        svfloat32x2_t vb_s1 = svld1_f32_x2(pcnt_1, w1);
        svmla_lane_za32_f32_vg1x2(0, vb_s0, va0, 0);
        svmla_lane_za32_f32_vg1x2(1, vb_s1, va0, 0);

        vb_s0 = svld1_f32_x2(pcnt_0, w0 + nr);
        vb_s1 = svld1_f32_x2(pcnt_1, w1 + nr);
        svmla_lane_za32_f32_vg1x2(2, vb_s0, va0, 1);
        svmla_lane_za32_f32_vg1x2(3, vb_s1, va0, 1);

        w0 += 2 * nr; w1 += 2 * nr;
        k -= 2;
      }

      if XNN_UNLIKELY(k != 0) {
        svfloat32_t va0 = svld1rq_f32(svwhilelt_b32_u64(0, 1), a0);
        a0 += 1;
        svfloat32x2_t vb_s0 = svld1_f32_x2(pcnt_0, w0);
        svfloat32x2_t vb_s1 = svld1_f32_x2(pcnt_1, w1);
        svmla_lane_za32_f32_vg1x2(0, vb_s0, va0, 0);
        svmla_lane_za32_f32_vg1x2(1, vb_s1, va0, 0);
        w0 += nr; w1 += nr;
      }

#ifdef PREFETCH_W_MATRIX
      if (need_prfm_0) {
        if (prfm_acc_w_size_0 <= (w_block_bytes - 3 * (8 * vl))) {
          prfm_acc_w_size_0 += 3 * (8 * vl);
          __builtin_prefetch(prfm_ptr_0 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 2 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_0 += 3 * (8 * vl);
        } else if (prfm_acc_w_size_0 <= (w_block_bytes - 2 * (8 * vl))) {
          prfm_acc_w_size_0 += 2 * (8 * vl);
          __builtin_prefetch(prfm_ptr_0 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 1 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_0 += 2 * (8 * vl);
        } else if (prfm_acc_w_size_0 <= (w_block_bytes - 1 * (8 * vl))) {
          prfm_acc_w_size_0 += 1 * (8 * vl);
          __builtin_prefetch(prfm_ptr_0 + 0 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_0 += 1 * (8 * vl);
        }
      }
      if (need_prfm_1) {
        if (prfm_acc_w_size_1 <= (w_block_bytes - 3 * (8 * vl))) {
          prfm_acc_w_size_1 += 3 * (8 * vl);
          __builtin_prefetch(prfm_ptr_1 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 2 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_1 += 3 * (8 * vl);
        } else if (prfm_acc_w_size_1 <= (w_block_bytes - 2 * (8 * vl))) {
          prfm_acc_w_size_1 += 2 * (8 * vl);
          __builtin_prefetch(prfm_ptr_1 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_1 + 1 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_1 += 2 * (8 * vl);
        } else if (prfm_acc_w_size_1 <= (w_block_bytes - 1 * (8 * vl))) {
          prfm_acc_w_size_1 += 1 * (8 * vl);
          __builtin_prefetch(prfm_ptr_1 + 0 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_1 += 1 * (8 * vl);
        }
      }
#endif

      svfloat32_t vmin = svdup_f32(min);
      svfloat32_t vmax = svdup_f32(max);

      svfloat32x2_t f32_bias_0 = svld1_f32_x2(pcnt_0, wb0);
      svfloat32x2_t f32_bias_1 = svld1_f32_x2(pcnt_1, wb1);

      svadd_za32_f32_vg1x2(0, svread_za32_f32_vg1x2(2));
      svadd_za32_f32_vg1x2(1, svread_za32_f32_vg1x2(3));
      svadd_za32_f32_vg1x2(0, f32_bias_0);
      svadd_za32_f32_vg1x2(1, f32_bias_1);

      svfloat32x2_t za0_acco = svread_za32_f32_vg1x2(0);
      za0_acco = svclamp(za0_acco, vmin, vmax);
      svst1_f32_x2(pcnt_0, c0, za0_acco);

      svfloat32x2_t za1_acco = svread_za32_f32_vg1x2(1);
      za1_acco = svclamp(za1_acco, vmin, vmax);
      svst1_f32_x2(pcnt_1, c0 + nr, za1_acco);

      c0 = (float*) ((uintptr_t) c0 + blocks_processed * cn_stride);
      if (nc >= blocks_processed * nr) {
        nc -= blocks_processed * nr;
      } else {
        nc = 0;
      }
    } else {
      blocks_processed = 1;
      // Branch 0: Focused on N = 32 (blocks_processed == 1)
      svcount_t pcnt_0 = svwhilelt_c32_u64(0, nc, 2);

      const float* w0 = w;
      w = w + 1 * stream_stride;

      const float* wb0 = w0;
      w0 += nr;

      svzero_za();
      a0 = a;

#ifdef PREFETCH_W_MATRIX
      bool need_prfm_0 = (w_block_bytes >= kPrefetchDistance * kBytesPerKLoop) && need_prfm && (nc > 0);

      size_t prfm_acc_w_size_0 = 0;

      const int8_t* prfm_ptr_0 = (const int8_t*) wb0;

      if (need_prfm_0) {
        prfm_acc_w_size_0 = kPrefetchDistance * kBytesPerKLoop;
        int i = 0;
        for (; i < (kPrefetchDistance * kCacheLinesPerLoop); i++) {
          __builtin_prefetch(prfm_ptr_0 + kCacheLineSize * i, 0, 2);
        }
        prfm_ptr_0 += kCacheLineSize * i;
      }
#endif

      size_t k_chunks = kc / 4;
      if (k_chunks > 0) {
        svfloat32_t va0 = svld1rq_f32(pg_32_msk_4, a0); a0 += 4;

        // Prequel: Preload W for all 4 lanes (K=4)
        svfloat32x2_t vb_s0_l0 = svld1_f32_x2(pcnt_0, w0); w0 += nr;
        svfloat32x2_t vb_s0_l1 = svld1_f32_x2(pcnt_0, w0); w0 += nr;
        svfloat32x2_t vb_s0_l2 = svld1_f32_x2(pcnt_0, w0); w0 += nr;
        svfloat32x2_t vb_s0_l3 = svld1_f32_x2(pcnt_0, w0); w0 += nr;

        k_chunks--;

        while (k_chunks > 0) {
          svmla_lane_za32_f32_vg1x2(0, vb_s0_l0, va0, 0);
          vb_s0_l0 = svld1_f32_x2(pcnt_0, w0); w0 += nr;
          svmla_lane_za32_f32_vg1x2(1, vb_s0_l1, va0, 1);
          vb_s0_l1 = svld1_f32_x2(pcnt_0, w0); w0 += nr;
          svmla_lane_za32_f32_vg1x2(2, vb_s0_l2, va0, 2);
          vb_s0_l2 = svld1_f32_x2(pcnt_0, w0); w0 += nr;
          svmla_lane_za32_f32_vg1x2(3, vb_s0_l3, va0, 3);
          vb_s0_l3 = svld1_f32_x2(pcnt_0, w0); w0 += nr;
          va0 = svld1rq_f32(pg_32_msk_4, a0); a0 += 4;

#ifdef PREFETCH_W_MATRIX
          if (need_prfm_0 && (prfm_acc_w_size_0 < (w_block_bytes - kBytesPerKLoop))) {
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
#endif
          k_chunks--;
        }

        // Epilogue:
        svmla_lane_za32_f32_vg1x2(0, vb_s0_l0, va0, 0);
        svmla_lane_za32_f32_vg1x2(1, vb_s0_l1, va0, 1);
        svmla_lane_za32_f32_vg1x2(2, vb_s0_l2, va0, 2);
        svmla_lane_za32_f32_vg1x2(3, vb_s0_l3, va0, 3);
      }
      size_t k = kc % 4;

      if (k >= 2) {
        svfloat32_t va0 = svld1rq_f32(svwhilelt_b32_u64(0, 2), a0);
        a0 += 2;
        svfloat32x2_t vb_s0 = svld1_f32_x2(pcnt_0, w0);
        svmla_lane_za32_f32_vg1x2(0, vb_s0, va0, 0);

        vb_s0 = svld1_f32_x2(pcnt_0, w0 + nr);
        svmla_lane_za32_f32_vg1x2(1, vb_s0, va0, 1);

        w0 += 2 * nr;
        k -= 2;
      }

      if XNN_UNLIKELY(k != 0) {
        svfloat32_t va0 = svld1rq_f32(svwhilelt_b32_u64(0, 1), a0);
        a0 += 1;
        svfloat32x2_t vb_s0 = svld1_f32_x2(pcnt_0, w0);
        svmla_lane_za32_f32_vg1x2(0, vb_s0, va0, 0);
        w0 += nr;
      }

#ifdef PREFETCH_W_MATRIX
      if (need_prfm_0) {
        if (prfm_acc_w_size_0 <= (w_block_bytes - 3 * (8 * vl))) {
          prfm_acc_w_size_0 += 3 * (8 * vl);
          __builtin_prefetch(prfm_ptr_0 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 1 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 2 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_0 += 3 * (8 * vl);
        } else if (prfm_acc_w_size_0 <= (w_block_bytes - 2 * (8 * vl))) {
          prfm_acc_w_size_0 += 2 * (8 * vl);
          __builtin_prefetch(prfm_ptr_0 + 0 * kTrailerPrfmStride, 0, 2);
          __builtin_prefetch(prfm_ptr_0 + 1 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_0 += 2 * (8 * vl);
        } else if (prfm_acc_w_size_0 <= (w_block_bytes - 1 * (8 * vl))) {
          prfm_acc_w_size_0 += 1 * (8 * vl);
          __builtin_prefetch(prfm_ptr_0 + 0 * kTrailerPrfmStride, 0, 2);
          prfm_ptr_0 += 1 * (8 * vl);
        }
      }
#endif

      svfloat32_t vmin = svdup_f32(min);
      svfloat32_t vmax = svdup_f32(max);

      svfloat32x2_t f32_bias_0 = svld1_f32_x2(pcnt_0, wb0);

      svadd_za32_f32_vg1x2(0, svread_za32_f32_vg1x2(2));
      svadd_za32_f32_vg1x2(1, svread_za32_f32_vg1x2(3));
      svadd_za32_f32_vg1x2(0, svread_za32_f32_vg1x2(1));
      svadd_za32_f32_vg1x2(0, f32_bias_0);

      svfloat32x2_t za0_acco = svread_za32_f32_vg1x2(0);
      za0_acco = svclamp(za0_acco, vmin, vmax);
      svst1_f32_x2(pcnt_0, c0, za0_acco);

      c0 = (float*) ((uintptr_t) c0 + blocks_processed * cn_stride);
      if (nc >= blocks_processed * nr) {
        nc -= blocks_processed * nr;
      } else {
        nc = 0;
      }
    }
  }
}

void xnn_pf32_gemm_minmax_ukernel_1x32__sme2(
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
  assert(mr == 1);
  pf32_gemm_1x32_sme2_impl(mr, nc, kc / sizeof(float), a, w, c, cm_stride, sizeof(float),
                                              params->scalar.min, params->scalar.max);
}

#undef PREFETCH_W_MATRIX

#endif  // XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
