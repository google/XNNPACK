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


#define DECLARE_PAD_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                  \
    size_t rows,                              \
    size_t channels,                          \
    size_t pre_padding,                       \
    size_t post_padding,                      \
    const void* input,                        \
    size_t input_stride,                      \
    void* output,                             \
    size_t output_stride,                     \
    const uint32_t fill_pattern);

DECLARE_PAD_UKERNEL_FUNCTION(xnn_xx_pad_ukernel_p4__scalar_u16)
DECLARE_PAD_UKERNEL_FUNCTION(xnn_xx_pad_ukernel_p16__neon_u16)
DECLARE_PAD_UKERNEL_FUNCTION(xnn_xx_pad_ukernel_p16__sse2_u16)
DECLARE_PAD_UKERNEL_FUNCTION(xnn_xx_pad_ukernel_p16__wasmsimd_u16)


#ifdef __cplusplus
}  // extern "C"
#endif
