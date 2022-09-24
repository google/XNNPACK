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


#define DECLARE_CS16_VSQUAREABS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                              \
    size_t batch_size,                                    \
    const int16_t* input,                                 \
    uint32_t* output);


DECLARE_CS16_VSQUAREABS_UKERNEL_FUNCTION(xnn_cs16_vsquareabs_ukernel__scalar_x1)
DECLARE_CS16_VSQUAREABS_UKERNEL_FUNCTION(xnn_cs16_vsquareabs_ukernel__scalar_x2)
DECLARE_CS16_VSQUAREABS_UKERNEL_FUNCTION(xnn_cs16_vsquareabs_ukernel__scalar_x3)
DECLARE_CS16_VSQUAREABS_UKERNEL_FUNCTION(xnn_cs16_vsquareabs_ukernel__scalar_x4)

DECLARE_CS16_VSQUAREABS_UKERNEL_FUNCTION(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x4)
DECLARE_CS16_VSQUAREABS_UKERNEL_FUNCTION(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x8)
DECLARE_CS16_VSQUAREABS_UKERNEL_FUNCTION(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x12)
DECLARE_CS16_VSQUAREABS_UKERNEL_FUNCTION(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x16)

DECLARE_CS16_VSQUAREABS_UKERNEL_FUNCTION(xnn_cs16_vsquareabs_ukernel__hexagon_x2)
DECLARE_CS16_VSQUAREABS_UKERNEL_FUNCTION(xnn_cs16_vsquareabs_ukernel__hexagon_x4)
DECLARE_CS16_VSQUAREABS_UKERNEL_FUNCTION(xnn_cs16_vsquareabs_ukernel__hexagon_x6)
DECLARE_CS16_VSQUAREABS_UKERNEL_FUNCTION(xnn_cs16_vsquareabs_ukernel__hexagon_x8)
DECLARE_CS16_VSQUAREABS_UKERNEL_FUNCTION(xnn_cs16_vsquareabs_ukernel__hexagon_x10)
DECLARE_CS16_VSQUAREABS_UKERNEL_FUNCTION(xnn_cs16_vsquareabs_ukernel__hexagon_x12)

#ifdef __cplusplus
}  // extern "C"
#endif
