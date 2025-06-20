// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_VMULCADDC_H_
#define XNNPACK_SRC_XNNPACK_VMULCADDC_H_

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"

#ifdef __cplusplus
extern "C" {
#endif

#define XNN_UKERNEL(arch_flags, fn_name, row_tile, channel_tile, datatype,   \
                    params_type, init_params)                                \
  XNN_INTERNAL void fn_name(size_t m, size_t c, const datatype* x,           \
                            size_t x_stride, const datatype* w, datatype* y, \
                            size_t y_stride, const params_type* params);
#include "src/f16-vmulcaddc/f16-vmulcaddc.inc"
#include "src/f32-vmulcaddc/f32-vmulcaddc.inc"
#undef XNN_UKERNEL

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_SRC_XNNPACK_VMULCADDC_H_
