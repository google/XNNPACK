// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"

#ifdef __cplusplus
extern "C" {
#endif


#define XNN_UKERNEL(arch_flags, fn_name, row_tile, channel_tile, datatype) \
  XNN_INTERNAL void fn_name(                                   \
      size_t m,                                                \
      size_t c,                                                \
      const float* x,                                          \
      size_t x_stride,                                         \
      const float* w,                                          \
      float* y,                                                \
      size_t y_stride,                                         \
      const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "f32-vmulcaddc/f32-vmulcaddc.h"
#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, fn_name, row_tile, channel_tile, datatype) \
  XNN_INTERNAL void fn_name(                                   \
      size_t m,                                                \
      size_t c,                                                \
      const xnn_float16* x,                                    \
      size_t x_stride,                                         \
      const xnn_float16* w,                                    \
      xnn_float16* y,                                          \
      size_t y_stride,                                         \
      const union xnn_f16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "f16-vmulcaddc/f16-vmulcaddc.h"
#undef XNN_UKERNEL

#ifdef __cplusplus
}  // extern "C"
#endif
