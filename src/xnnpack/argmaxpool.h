// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <xnnpack/params.h>
#include <xnnpack/common.h>

#ifdef __cplusplus
extern "C" {
#endif


#define DECLARE_F32_ARGMAXPOOL_UNIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                     \
      size_t n,                                                  \
      size_t ks,                                                 \
      size_t kc,                                                 \
      const float** x,                                           \
      float* y,                                                  \
      uint32_t* i,                                               \
      size_t x_increment,                                        \
      size_t y_increment,                                        \
      const union xnn_f32_output_params* params);

DECLARE_F32_ARGMAXPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_f32_argmaxpool_ukernel_up4__psimd)
DECLARE_F32_ARGMAXPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_f32_argmaxpool_ukernel_up4__scalar)
DECLARE_F32_ARGMAXPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_f32_argmaxpool_ukernel_up4__sse2)
DECLARE_F32_ARGMAXPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_f32_argmaxpool_ukernel_up9__psimd)
DECLARE_F32_ARGMAXPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_f32_argmaxpool_ukernel_up9__scalar)
DECLARE_F32_ARGMAXPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_f32_argmaxpool_ukernel_up9__sse2)


#define DECLARE_F32_ARGMAXPOOL_MULTIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                       \
      size_t n,                                                    \
      size_t ks,                                                   \
      size_t kc,                                                   \
      const float** x,                                             \
      float* ab,                                                   \
      uint32_t* ib,                                                \
      float* y,                                                    \
      uint32_t* i,                                                 \
      size_t x_increment,                                          \
      size_t y_increment,                                          \
      const union xnn_f32_output_params* params);

DECLARE_F32_ARGMAXPOOL_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_argmaxpool_ukernel_mp9p8q__psimd)
DECLARE_F32_ARGMAXPOOL_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_argmaxpool_ukernel_mp9p8q__scalar)
DECLARE_F32_ARGMAXPOOL_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_argmaxpool_ukernel_mp9p8q__sse2)


#ifdef __cplusplus
} /* extern "C" */
#endif
