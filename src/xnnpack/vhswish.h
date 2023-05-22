// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <xnnpack/common.h>
#include <xnnpack/microparams.h>

#ifdef __cplusplus
extern "C" {
#endif


#define DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                          \
      size_t n,                                       \
      const int8_t* input,                            \
      int8_t* output,                                 \
      const union xnn_qs8_hswish_params* params);

DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__neon_x8)
DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__neon_x16)
DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__neon_x32)

DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__scalar_x1)
DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__scalar_x2)
DECLARE_QS8_VHSWISH_UKERNEL_FUNCTION(xnn_qs8_vhswish_ukernel__scalar_x4)

#define DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                          \
      size_t n,                                       \
      const uint8_t* input,                            \
      uint8_t* output,                                 \
      const union xnn_qu8_hswish_params* params);

DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__neon_x8)
DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__neon_x16)
DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__neon_x32)

DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__scalar_x1)
DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__scalar_x2)
DECLARE_QU8_VHSWISH_UKERNEL_FUNCTION(xnn_qu8_vhswish_ukernel__scalar_x4)

#ifdef __cplusplus
}  // extern "C"
#endif