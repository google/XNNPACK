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


#define DECLARE_X32_PAD_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                      \
    size_t rows,                                  \
    size_t channels,                              \
    size_t pre_padding,                           \
    size_t post_padding,                          \
    const uint32_t* fill_value,                   \
    const uint32_t* input,                        \
    size_t input_stride,                          \
    uint32_t* output,                             \
    size_t output_stride);

DECLARE_X32_PAD_UKERNEL_FUNCTION(xnn_x32_pad_ukernel__neon)
DECLARE_X32_PAD_UKERNEL_FUNCTION(xnn_x32_pad_ukernel__wasmsimd)
DECLARE_X32_PAD_UKERNEL_FUNCTION(xnn_x32_pad_ukernel__sse)
DECLARE_X32_PAD_UKERNEL_FUNCTION(xnn_x32_pad_ukernel__scalar_float)
DECLARE_X32_PAD_UKERNEL_FUNCTION(xnn_x32_pad_ukernel__scalar_int)


#ifdef __cplusplus
}  // extern "C"
#endif
