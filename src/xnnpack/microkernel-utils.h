// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>

#include <xnnpack/common.h>

#ifdef __cplusplus
extern "C" {
#endif

// The total tile size needed to cover kernel_size.
XNN_INTERNAL size_t xnn_dwconv_multipass_tile_size(
  size_t kernel_size,
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile);

// The total count of weights (in elements) needed for multipass dwconv.
XNN_INTERNAL size_t xnn_dwconv_multipass_weights_count(
  size_t tile_size,
  size_t channels,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round);

#ifdef __cplusplus
}
#endif
