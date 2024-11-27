// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>

#include "xnnpack/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// When parallelizing GEMMs, try to tile the computation such that we have at
// least this many tiles per thread.
#define XNN_GEMM_TILES_PER_THREAD 5

// Computes the largest `nc`, the largest multiple of `nr` such that there are
// at least five tiles per thread (if `num_threads > 1`).
size_t xnn_gemm_best_nc(size_t num_groups, size_t m, size_t n, size_t mr,
                        size_t nr, size_t num_threads);

// The total tile size needed to cover kernel_size.
XNN_INTERNAL size_t xnn_dwconv_multipass_tile_size(
  size_t kernel_size,
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile);

// The total count of weights (in bytes) needed for multipass dwconv.
size_t xnn_dwconv_multipass_weights_size(
  size_t tile_size,
  size_t channels,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  size_t bias_element_size,
  size_t log2_filter_element_size,
  size_t extra_weights_byte);

// Calculate the number of bytes read.
size_t xnn_dwconv_multipass_bytes_read(
  size_t kernel_size,
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t channels,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  size_t log2_input_size,
  size_t log2_filter_size,
  size_t bias_element_size,
  size_t log2_accumulator_size);

// Calculate the number of bytes written.
size_t xnn_dwconv_multipass_bytes_written(
  size_t kernel_size,
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t channels,
  size_t channel_round,
  size_t log2_accumulator_size,
  size_t log2_output_size);

#ifdef __cplusplus
}
#endif
