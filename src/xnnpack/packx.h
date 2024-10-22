// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"

#ifdef __cplusplus
extern "C" {
#endif


#define XNN_UKERNEL(arch_flags, ukernel, k_, mr_)    \
  XNN_INTERNAL void ukernel(                         \
      size_t m,                                      \
      size_t k,                                      \
      const uint32_t* x,                             \
      size_t x_stride,                               \
      uint32_t* y);

#include "x32-packx/x32-packx.h"

#undef XNN_UKERNEL


#ifdef __cplusplus
}  // extern "C"
#endif
