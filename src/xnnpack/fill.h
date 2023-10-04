// Copyright 2020 Google LLC
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


#define DECLARE_FILL_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                   \
    size_t kernel_elements,                    \
    size_t channels,                           \
    void* output,                              \
    size_t output_stride,                      \
    const uint32_t fill_pattern);

DECLARE_FILL_UKERNEL_FUNCTION(xnn_xx_fill_ukernel__neon_u64)
DECLARE_FILL_UKERNEL_FUNCTION(xnn_xx_fill_ukernel__scalar_u16)
DECLARE_FILL_UKERNEL_FUNCTION(xnn_xx_fill_ukernel__sse2_u64)
DECLARE_FILL_UKERNEL_FUNCTION(xnn_xx_fill_ukernel__wasmsimd_u64)


#ifdef __cplusplus
}  // extern "C"
#endif
