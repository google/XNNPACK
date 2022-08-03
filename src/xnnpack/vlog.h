// Copyright 2022 Google LLC
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


#define DECLARE_U32_VLOG_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
    size_t batch_size,                             \
    const uint32_t* input,                         \
    uint32_t input_lshift,                         \
    uint32_t output_scale,                         \
    uint16_t* output);


DECLARE_U32_VLOG_UKERNEL_FUNCTION(xnn_u32_vlog_ukernel__scalar_x1)
DECLARE_U32_VLOG_UKERNEL_FUNCTION(xnn_u32_vlog_ukernel__scalar_x2)
DECLARE_U32_VLOG_UKERNEL_FUNCTION(xnn_u32_vlog_ukernel__scalar_x3)
DECLARE_U32_VLOG_UKERNEL_FUNCTION(xnn_u32_vlog_ukernel__scalar_x4)

#ifdef __cplusplus
}  // extern "C"
#endif
