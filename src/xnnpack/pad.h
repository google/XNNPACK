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


#define DECLARE_PAD_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                  \
    size_t m,                                 \
    size_t n,                                 \
    size_t l,                                 \
    size_t r,                                 \
    uint32_t c,                               \
    const void* input,                        \
    size_t input_stride,                      \
    void* output,                             \
    size_t output_stride);

DECLARE_PAD_UKERNEL_FUNCTION(xnn_x32_pad_x2__neon)
DECLARE_PAD_UKERNEL_FUNCTION(xnn_x32_pad_x2__psimd)
DECLARE_PAD_UKERNEL_FUNCTION(xnn_x32_pad_x2__scalar)
DECLARE_PAD_UKERNEL_FUNCTION(xnn_x32_pad_x2__sse2)


#ifdef __cplusplus
}  // extern "C"
#endif
