// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
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


#define DECLARE_X8_LUT_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                     \
      size_t n,                                  \
      const uint8_t* x,                          \
      const uint8_t* t,                          \
      uint8_t* y);

DECLARE_X8_LUT_UKERNEL_FUNCTION(xnn_x8_lut_ukernel__scalar)


#define DECLARE_U8_LUT32NORM_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                           \
      size_t n,                                        \
      const uint8_t* x,                                \
      const uint32_t* t,                               \
      uint8_t* y);

DECLARE_U8_LUT32NORM_UKERNEL_FUNCTION(xnn_u8_lut32norm_ukernel__scalar)


#ifdef __cplusplus
}  // extern "C"
#endif
