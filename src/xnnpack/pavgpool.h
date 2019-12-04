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


#define DECLARE_F32_PAVGPOOL_MULTIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                    \
      size_t n,                                                 \
      size_t ks,                                                \
      size_t kc,                                                \
      const float** x,                                          \
      const float* zero,                                        \
      const float* multiplier,                                  \
      float* buffer,                                            \
      float* y,                                                 \
      size_t x_increment,                                       \
      size_t y_increment,                                       \
      const union xnn_f32_output_params* params);

DECLARE_F32_PAVGPOOL_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_pavgpool_ukernel_mp9p8q__neon)
DECLARE_F32_PAVGPOOL_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_pavgpool_ukernel_mp9p8q__sse)
DECLARE_F32_PAVGPOOL_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_pavgpool_ukernel_mp9p8q__psimd)
DECLARE_F32_PAVGPOOL_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_pavgpool_ukernel_mp9p8q__wasm)
DECLARE_F32_PAVGPOOL_MULTIPASS_UKERNEL_FUNCTION(xnn_f32_pavgpool_ukernel_mp9p8q__scalar)


#define DECLARE_F32_PAVGPOOL_UNIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                  \
      size_t n,                                               \
      size_t ks,                                              \
      size_t kc,                                              \
      const float** x,                                        \
      const float* zero,                                      \
      const float* multiplier,                                  \
      float* y,                                               \
      size_t x_increment,                                     \
      size_t y_increment,                                     \
      const union xnn_f32_output_params* params);

DECLARE_F32_PAVGPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_f32_pavgpool_ukernel_up9__neon)
DECLARE_F32_PAVGPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_f32_pavgpool_ukernel_up9__sse)
DECLARE_F32_PAVGPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_f32_pavgpool_ukernel_up9__psimd)
DECLARE_F32_PAVGPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_f32_pavgpool_ukernel_up9__wasm)
DECLARE_F32_PAVGPOOL_UNIPASS_UKERNEL_FUNCTION(xnn_f32_pavgpool_ukernel_up9__scalar)


#ifdef __cplusplus
}  // extern "C"
#endif
