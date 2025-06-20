// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_VSCALEEXPMINUSMAX_H_
#define XNNPACK_SRC_XNNPACK_VSCALEEXPMINUSMAX_H_

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define XNN_UKERNEL(arch_flags, fn_name, element_tile, datatype, params_type, \
                    init_params)                                              \
  XNN_INTERNAL void fn_name(size_t n, const float* input, float* output,      \
                            float scale, float max);
#include "src/f32-vscaleexpminusmax/f32-vscaleexpminusmax.inc"
#undef XNN_UKERNEL

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // XNNPACK_SRC_XNNPACK_VSCALEEXPMINUSMAX_H_
