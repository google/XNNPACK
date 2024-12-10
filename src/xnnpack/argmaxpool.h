// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"

#ifdef __cplusplus
extern "C" {
#endif

                    
#define XNN_UKERNEL_UNIPASS(arch_flags,fn_name, primary_tile, incremental_tile, channel_tile, vector_tile, datatype) \
  XNN_INTERNAL void fn_name(                                     \
      size_t output_pixels,                                      \
      size_t kernel_elements,                                    \
      size_t channels,                                           \
      const datatype** input,                                    \
      size_t input_offset,                                       \
      datatype* output,                                          \
      uint32_t* index,                                           \
      size_t input_increment,                                    \
      size_t output_increment);
#include "f32-argmaxpool/f32-argmaxpool-unipass.h"
#undef XNN_UKERNEL_UNIPASS

#define XNN_UKERNEL_MULTIPASS(arch_flags,fn_name, primary_tile, incremental_tile, channel_tile, vector_tile, datatype) \
  XNN_INTERNAL void fn_name(                                       \
      size_t output_pixels,                                        \
      size_t kernel_elements,                                      \
      size_t channels,                                             \
      const datatype** input,                                      \
      size_t input_offset,                                         \
      datatype* accumulation_buffer,                               \
      uint32_t* index_buffer,                                      \
      datatype* output,                                            \
      uint32_t* index,                                             \
      size_t input_increment,                                      \
      size_t output_increment);
#include "f32-argmaxpool/f32-argmaxpool-multipass.h"
#undef XNN_UKERNEL_MULTIPASS
#ifdef __cplusplus
}  // extern "C"
#endif
