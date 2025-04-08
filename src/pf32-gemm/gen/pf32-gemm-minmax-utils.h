// Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stdint.h>
#include "src/xnnpack/common.h"

#ifdef __cplusplus
extern "C" {
#endif

static const size_t xnn_pf32_gemm_minmax_mr = 2;
static const size_t xnn_pf32_gemm_minmax_nr = 2;
static const size_t xnn_pf32_gemm_minmax_kr = 1;
static const size_t xnn_pf32_gemm_minmax_sr = 1;

XNN_INTERNAL uint64_t xnn_pf32_get_word_sme_vl__asm_aarch64_neonsme(void);

XNN_INTERNAL void xnn_pf32_pack_lhs__asm_aarch64_neonsme(
  const void* in, void* out, const size_t height, const size_t row_offset, const size_t width);

XNN_INTERNAL void xnn_pf32_pack_rhs__asm_aarch64_neonsme(
  const void* in,
  void* out,
  size_t height,
  const size_t in_stride,
  const void* bias,
  size_t out_stride,
  const size_t width);

XNN_INTERNAL void xnn_pf32_gemm_minmax__asm_aarch64_neonsme(
  const void* A,
  const void* B,
  void* C,
  uint64_t K,
  void* max,
  void* min,
  uint64_t M,
  uint64_t N,
  void* accumulator_buffer,
  uint64_t flags,
  uint64_t ldcb);

#ifdef __cplusplus
}  // extern "C"
#endif
