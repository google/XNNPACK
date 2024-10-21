// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"

#ifdef __cplusplus
extern "C" {
#endif


#define XNN_UKERNEL(arch_flags, ukernel, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, qmin, qmax) \
  XNN_INTERNAL void ukernel(                                                                                            \
      size_t output_pixels,                                                                                             \
      size_t kernel_size,                                                                                               \
      size_t channels,                                                                                                  \
      const xnn_float16** input,                                                                               \
      size_t input_offset,                                                                                              \
      xnn_float16* output,                                                                                     \
      size_t input_increment,                                                                                           \
      size_t output_increment,                                                                                          \
      const union xnn_f16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

#include "f16-maxpool/f16-maxpool-minmax.h"

#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, ukernel, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, qmin, qmax) \
  XNN_INTERNAL void ukernel(                                                                                            \
      size_t output_pixels,                                                                                             \
      size_t kernel_size,                                                                                               \
      size_t channels,                                                                                                  \
      const float** input,                                                                                              \
      size_t input_offset,                                                                                              \
      float* output,                                                                                                    \
      size_t input_increment,                                                                                           \
      size_t output_increment,                                                                                          \
      const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

#include "f32-maxpool/f32-maxpool-minmax.h"

#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, ukernel, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, qmin, qmax) \
  XNN_INTERNAL void ukernel(                                                                                            \
      size_t output_pixels,                                                                                             \
      size_t kernel_size,                                                                                               \
      size_t channels,                                                                                                  \
      const uint8_t** input,                                                                                            \
      size_t input_offset,                                                                                              \
      uint8_t* output,                                                                                                  \
      size_t input_increment,                                                                                           \
      size_t output_increment,                                                                                          \
      const struct xnn_u8_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);


#include "u8-maxpool/u8-maxpool-minmax.h"

#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, ukernel, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, qmin, qmax) \
  XNN_INTERNAL void ukernel(                                                                                            \
      size_t output_pixels,                                                                                             \
      size_t kernel_size,                                                                                               \
      size_t channels,                                                                                                  \
      const int8_t** input,                                                                                             \
      size_t input_offset,                                                                                              \
      int8_t* output,                                                                                                   \
      size_t input_increment,                                                                                           \
      size_t output_increment,                                                                                          \
      const struct xnn_s8_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);


#include "s8-maxpool/s8-maxpool-minmax.h"

#undef XNN_UKERNEL

#ifdef __cplusplus
}  // extern "C"
#endif
