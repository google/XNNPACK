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

#define XNN_UKERNEL(arch_flags, fn_name, element_tile, datatype) \
  XNN_INTERNAL void fn_name(                               \
      size_t n,                                            \
      const float* input,                                  \
      float* output,                                       \
      float scale_mantissa,                                \
      float scale_exponent);
#include "f32-vscaleextexp/f32-vscaleextexp.h"
#undef XNN_UKERNEL


#ifdef __cplusplus
} /* extern "C" */
#endif
