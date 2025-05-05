// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"

#ifdef __cplusplus
extern "C" {
#endif

#define XNN_UKERNEL(arch_flags, fn_name, row_tile, channel_tile, datatype, \
                    params_type)                                           \
  XNN_INTERNAL void fn_name(                                               \
      size_t m, size_t c, const datatype* x, size_t x_stride,              \
      const datatype* w, datatype* y, size_t y_stride,                     \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "src/f16-vmulcaddc/f16-vmulcaddc.h"
#include "src/f32-vmulcaddc/f32-vmulcaddc.h"
#undef XNN_UKERNEL

#ifdef __cplusplus
}  // extern "C"
#endif
