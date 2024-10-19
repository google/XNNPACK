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


#define XNN_UKERNEL_MULTIPASS(arch_flags, ukernel, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, init_params) \
  XNN_INTERNAL void ukernel(                                           \
      size_t output_pixels,                                            \
      size_t kernel_elements,                                          \
      size_t channels,                                                 \
      const xnn_float16** input,                              \
      size_t input_offset,                                             \
      const xnn_float16* zero,                                \
      xnn_float16* buffer,                                    \
      xnn_float16* output,                                    \
      size_t input_increment,                                          \
      size_t output_increment,                                         \
      const struct xnn_f16_scaleminmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

#define XNN_UKERNEL_UNIPASS(arch_flags, ukernel, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, init_params) \
  XNN_INTERNAL void ukernel(                                         \
      size_t output_pixels,                                          \
      size_t kernel_elements,                                        \
      size_t channels,                                               \
      const xnn_float16** input,                            \
      size_t input_offset,                                           \
      const xnn_float16* zero,                              \
      xnn_float16* output,                                  \
      size_t input_increment,                                        \
      size_t output_increment,                                       \
      const struct xnn_f16_scaleminmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

#include "f16-avgpool/f16-avgpool-minmax.h"

#undef XNN_UKERNEL_MULTIPASS
#undef XNN_UKERNEL_UNIPASS

#define XNN_UKERNEL_MULTIPASS(arch_flags, ukernel, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, init_params) \
  XNN_INTERNAL void ukernel(                                           \
      size_t output_pixels,                                            \
      size_t kernel_elements,                                          \
      size_t channels,                                                 \
      const float** input,                                             \
      size_t input_offset,                                             \
      const float* zero,                                               \
      float* buffer,                                                   \
      float* output,                                                   \
      size_t input_increment,                                          \
      size_t output_increment,                                         \
      const struct xnn_f32_scaleminmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

#define XNN_UKERNEL_UNIPASS(arch_flags, ukernel, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, init_params) \
  XNN_INTERNAL void ukernel(                                         \
      size_t output_pixels,                                          \
      size_t kernel_elements,                                        \
      size_t channels,                                               \
      const float** input,                                           \
      size_t input_offset,                                           \
      const float* zero,                                             \
      float* output,                                                 \
      size_t input_increment,                                        \
      size_t output_increment,                                       \
      const struct xnn_f32_scaleminmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

#include "f32-avgpool/f32-avgpool-minmax.h"

#undef XNN_UKERNEL_MULTIPASS
#undef XNN_UKERNEL_UNIPASS


#define XNN_UKERNEL_MULTIPASS(arch_flags, ukernel, requantize, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, init_params) \
  XNN_INTERNAL void ukernel(                                           \
      size_t output_pixels,                                            \
      size_t kernel_elements,                                          \
      size_t channels,                                                 \
      const uint8_t** input,                                           \
      size_t input_offset,                                             \
      const uint8_t* zero,                                             \
      int32_t* buffer,                                                 \
      uint8_t* output,                                                 \
      size_t input_increment,                                          \
      size_t output_increment,                                         \
      const struct xnn_qu8_avgpool_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

#define XNN_UKERNEL_UNIPASS(arch_flags, ukernel, requantize, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, init_params) \
  XNN_INTERNAL void ukernel(                                         \
      size_t output_pixels,                                          \
      size_t kernel_elements,                                        \
      size_t channels,                                               \
      const uint8_t** input,                                         \
      size_t input_offset,                                           \
      const uint8_t* zero,                                           \
      uint8_t* output,                                               \
      size_t input_increment,                                        \
      size_t output_increment,                                       \
      const struct xnn_qu8_avgpool_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

#include "qu8-avgpool/qu8-avgpool-minmax.h"

#undef XNN_UKERNEL_MULTIPASS
#undef XNN_UKERNEL_UNIPASS



#ifdef __cplusplus
}  // extern "C"
#endif
