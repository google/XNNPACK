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


#define DECLARE_I16_VLSHIFT_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                          \
    size_t batch,                                     \
    const uint16_t* input,                            \
    uint16_t* output,                                 \
    uint32_t shift);


DECLARE_I16_VLSHIFT_UKERNEL_FUNCTION(xnn_i16_vlshift_ukernel__neon_u8)
DECLARE_I16_VLSHIFT_UKERNEL_FUNCTION(xnn_i16_vlshift_ukernel__neon_u16)
DECLARE_I16_VLSHIFT_UKERNEL_FUNCTION(xnn_i16_vlshift_ukernel__neon_u24)
DECLARE_I16_VLSHIFT_UKERNEL_FUNCTION(xnn_i16_vlshift_ukernel__neon_u32)

DECLARE_I16_VLSHIFT_UKERNEL_FUNCTION(xnn_i16_vlshift_ukernel__scalar_u1)
DECLARE_I16_VLSHIFT_UKERNEL_FUNCTION(xnn_i16_vlshift_ukernel__scalar_u2)
DECLARE_I16_VLSHIFT_UKERNEL_FUNCTION(xnn_i16_vlshift_ukernel__scalar_u3)
DECLARE_I16_VLSHIFT_UKERNEL_FUNCTION(xnn_i16_vlshift_ukernel__scalar_u4)

#ifdef __cplusplus
}  // extern "C"
#endif
