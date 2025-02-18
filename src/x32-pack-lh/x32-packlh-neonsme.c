// Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/pack-lh.h"
#include "pf32-gemm/gen/pf32-gemm-minmax-utils.h"

#define GET_MIN(a, b) ((a) < (b) ? (a) : (b))
#define roundup(x, y)   ((((x) + ((y) - 1)) / (y)) * (y))

void xnn_pack_f32_run_pack_lhs(size_t m, size_t k, size_t mr, size_t kr, size_t sr,
                  size_t m_idx_start, const void *lhs, size_t lhs_stride,
                  void *lhs_packed) {

  if ((mr != xnn_pf32_gemm_minmax_mr * xnn_pf32_get_word_sme_vl__asm_aarch64_neonsme()) || (kr != xnn_pf32_gemm_minmax_kr) ||
      (sr != xnn_pf32_gemm_minmax_sr) || (lhs == NULL) || (lhs_packed == NULL) ||
      (m_idx_start != 0)) {
    exit(EXIT_FAILURE);
  }

  const size_t block_height = xnn_pf32_gemm_minmax_mr * xnn_pf32_get_word_sme_vl__asm_aarch64_neonsme();
  const size_t width = k;
  const size_t row_offset = 0;

  const void *in[block_height];

  for (size_t block_y = 0; block_y < m; block_y += block_height) {
    const size_t height = GET_MIN(m - block_y, block_height);
    void *out = (void *)((char *)lhs_packed + block_y * k * sizeof(float));

    for (size_t y = 0; y < height; y++) {
      in[y] = (void *)((char *)lhs + (block_y + y) * lhs_stride);
    }

    xnn_pf32_pack_lhs__asm_aarch64_neonsme(in, out, height, row_offset, width);
  }
}

void xnn_x32_pack_lhs_ukernel__neonsme(size_t m, size_t k, size_t mr,
                                          size_t kr, size_t sr,
                                          size_t m_idx_start,
                                          const float* XNN_RESTRICT lhs,
                                          size_t lhs_stride,
                                          void* XNN_RESTRICT lhs_packed) {
  if (m == 1) {
    memcpy(lhs_packed, lhs, sizeof(float) * k);
  } else {
    xnn_pack_f32_run_pack_lhs(m, k, mr, kr, sr, 0, lhs, lhs_stride, lhs_packed);
  }
}

size_t xnn_x32_pack_lhs_size__neonsme(size_t m, size_t k, size_t mr, size_t kr, size_t sr) {
  if ((mr != xnn_pf32_gemm_minmax_mr * xnn_pf32_get_word_sme_vl__asm_aarch64_neonsme()) || (kr != xnn_pf32_gemm_minmax_kr) ||
      (sr != xnn_pf32_gemm_minmax_sr)) {
    exit(EXIT_FAILURE);
  }

  (void)mr;
  (void)kr;
  (void)sr;

  return roundup(m, xnn_pf32_gemm_minmax_mr * xnn_pf32_get_word_sme_vl__asm_aarch64_neonsme()) * k * sizeof(float);
  }

size_t xnn_x32_pack_lhs_offset__neonsme(size_t m, size_t k, size_t mr, size_t kr, size_t sr) {
    if (((m % (xnn_pf32_gemm_minmax_mr * xnn_pf32_get_word_sme_vl__asm_aarch64_neonsme())) != 0) || (mr != (xnn_pf32_gemm_minmax_mr * xnn_pf32_get_word_sme_vl__asm_aarch64_neonsme())) | (kr != xnn_pf32_gemm_minmax_kr) | (sr != xnn_pf32_gemm_minmax_sr))
    {
        exit(EXIT_FAILURE);
    }

  (void)mr;
  (void)kr;
  (void)sr;

  return m * k;
}