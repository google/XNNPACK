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


#define DECLARE_X32_UNPOOL_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                         \
    size_t p,                                        \
    size_t c,                                        \
    uint32_t f,                                      \
    const uint32_t* input,                           \
    const uint32_t* index,                           \
    uint32_t** output);

DECLARE_X32_UNPOOL_UKERNEL_FUNCTION(xnn_x32_unpool_ukernel__neon)
DECLARE_X32_UNPOOL_UKERNEL_FUNCTION(xnn_x32_unpool_ukernel__scalar)
DECLARE_X32_UNPOOL_UKERNEL_FUNCTION(xnn_x32_unpool_ukernel__sse2)
DECLARE_X32_UNPOOL_UKERNEL_FUNCTION(xnn_x32_unpool_ukernel__wasmsimd)


#ifdef __cplusplus
}  // extern "C"
#endif
