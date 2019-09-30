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


#define DECLARE_X8_ZIPC_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                      \
      size_t n,                                   \
      const uint8_t* x,                           \
      uint8_t* y);

DECLARE_X8_ZIPC_UKERNEL_FUNCTION(xnn_x8_zip_x2_ukernel__neon)
DECLARE_X8_ZIPC_UKERNEL_FUNCTION(xnn_x8_zip_x2_ukernel__sse2)
DECLARE_X8_ZIPC_UKERNEL_FUNCTION(xnn_x8_zip_x2_ukernel__scalar)
DECLARE_X8_ZIPC_UKERNEL_FUNCTION(xnn_x8_zip_x3_ukernel__neon)
DECLARE_X8_ZIPC_UKERNEL_FUNCTION(xnn_x8_zip_x3_ukernel__sse2)
DECLARE_X8_ZIPC_UKERNEL_FUNCTION(xnn_x8_zip_x3_ukernel__scalar)
DECLARE_X8_ZIPC_UKERNEL_FUNCTION(xnn_x8_zip_x4_ukernel__neon)
DECLARE_X8_ZIPC_UKERNEL_FUNCTION(xnn_x8_zip_x4_ukernel__sse2)
DECLARE_X8_ZIPC_UKERNEL_FUNCTION(xnn_x8_zip_x4_ukernel__scalar)


#define DECLARE_X32_ZIPC_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t n,                                    \
      const uint32_t* x,                           \
      uint32_t* y);

DECLARE_X32_ZIPC_UKERNEL_FUNCTION(xnn_x32_zip_x2_ukernel__neon)
DECLARE_X32_ZIPC_UKERNEL_FUNCTION(xnn_x32_zip_x2_ukernel__psimd)
DECLARE_X32_ZIPC_UKERNEL_FUNCTION(xnn_x32_zip_x2_ukernel__scalar)
DECLARE_X32_ZIPC_UKERNEL_FUNCTION(xnn_x32_zip_x2_ukernel__sse2)
DECLARE_X32_ZIPC_UKERNEL_FUNCTION(xnn_x32_zip_x3_ukernel__neon)
DECLARE_X32_ZIPC_UKERNEL_FUNCTION(xnn_x32_zip_x3_ukernel__psimd)
DECLARE_X32_ZIPC_UKERNEL_FUNCTION(xnn_x32_zip_x3_ukernel__scalar)
DECLARE_X32_ZIPC_UKERNEL_FUNCTION(xnn_x32_zip_x3_ukernel__sse2)
DECLARE_X32_ZIPC_UKERNEL_FUNCTION(xnn_x32_zip_x4_ukernel__neon)
DECLARE_X32_ZIPC_UKERNEL_FUNCTION(xnn_x32_zip_x4_ukernel__psimd)
DECLARE_X32_ZIPC_UKERNEL_FUNCTION(xnn_x32_zip_x4_ukernel__scalar)
DECLARE_X32_ZIPC_UKERNEL_FUNCTION(xnn_x32_zip_x4_ukernel__sse2)


#define DECLARE_X8_ZIPV_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                      \
      size_t n,                                   \
      size_t m,                                   \
      const uint8_t* x,                           \
      uint8_t* y);

DECLARE_X8_ZIPV_UKERNEL_FUNCTION(xnn_x8_zip_xm_ukernel__neon)
DECLARE_X8_ZIPV_UKERNEL_FUNCTION(xnn_x8_zip_xm_ukernel__sse2)
DECLARE_X8_ZIPV_UKERNEL_FUNCTION(xnn_x8_zip_xm_ukernel__scalar)


#define DECLARE_X32_ZIPV_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t n,                                    \
      size_t m,                                    \
      const uint32_t* x,                           \
      uint32_t* y);

DECLARE_X32_ZIPV_UKERNEL_FUNCTION(xnn_x32_zip_xm_ukernel__neon)
DECLARE_X32_ZIPV_UKERNEL_FUNCTION(xnn_x32_zip_xm_ukernel__psimd)
DECLARE_X32_ZIPV_UKERNEL_FUNCTION(xnn_x32_zip_xm_ukernel__scalar)
DECLARE_X32_ZIPV_UKERNEL_FUNCTION(xnn_x32_zip_xm_ukernel__sse2)


#ifdef __cplusplus
}  // extern "C"
#endif
