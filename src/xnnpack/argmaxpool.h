// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <xnnpack/common.h>

#ifdef __cplusplus
extern "C" {
#endif


#define DECLARE_F32_ARGMAXPOOL_UNIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                     \
      size_t output_pixels,                                      \
      size_t kernel_elements,                                    \
      size_t channels,                                           \
      const float** input,                                       \
      size_t input_offset,                                       \
      float* output,                                             \
      uint32_t* index,                                           \
      size_t input_increment,                                    \
      size_t output_increment);

DECLARE_F32_ARGMAXPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_f32_argmaxpool_ukernel_4x__neon_c4)
DECLARE_F32_ARGMAXPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_f32_argmaxpool_ukernel_4x__scalar_c1)
DECLARE_F32_ARGMAXPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_f32_argmaxpool_ukernel_4x__sse2_c4)
DECLARE_F32_ARGMAXPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_f32_argmaxpool_ukernel_4x__wasmsimd_c4)

DECLARE_F32_ARGMAXPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_f32_argmaxpool_ukernel_9x__neon_c4)
DECLARE_F32_ARGMAXPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_f32_argmaxpool_ukernel_9x__scalar_c1)
DECLARE_F32_ARGMAXPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_f32_argmaxpool_ukernel_9x__sse2_c4)
DECLARE_F32_ARGMAXPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_f32_argmaxpool_ukernel_9x__wasmsimd_c4)


#define DECLARE_F32_ARGMAXPOOL_MULTIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                       \
      size_t output_pixels,                                        \
      size_t kernel_elements,                                      \
      size_t channels,                                             \
      const float** input,                                         \
      size_t input_offset,                                         \
      float* accumulation_buffer,                                  \
      uint32_t* index_buffer,                                      \
      float* output,                                               \
      uint32_t* index,                                             \
      size_t input_increment,                                      \
      size_t output_increment);

DECLARE_F32_ARGMAXPOOL_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_argmaxpool_ukernel_9p8x__neon_c4)
DECLARE_F32_ARGMAXPOOL_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1)
DECLARE_F32_ARGMAXPOOL_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4)
DECLARE_F32_ARGMAXPOOL_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_argmaxpool_ukernel_9p8x__wasmsimd_c4)


#ifdef __cplusplus
}  // extern "C"
#endif
