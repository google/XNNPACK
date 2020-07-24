// Copyright 2020 Google LLC
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


#define DECLARE_FILL_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                   \
    size_t kernel_elements,                    \
    size_t channels,                           \
    uint32_t* output,                          \
    size_t output_stride,                      \
    const uint32_t* fill_value);

DECLARE_FILL_UKERNEL_FUNCTION(xnn_x32_fill_ukernel__sse)
DECLARE_FILL_UKERNEL_FUNCTION(xnn_x32_fill_ukernel__neon)
DECLARE_FILL_UKERNEL_FUNCTION(xnn_x32_fill_ukernel__wasmsimd)
DECLARE_FILL_UKERNEL_FUNCTION(xnn_x32_fill_ukernel__scalar_float)
DECLARE_FILL_UKERNEL_FUNCTION(xnn_x32_fill_ukernel__scalar_int)


#ifdef __cplusplus
}  // extern "C"
#endif
