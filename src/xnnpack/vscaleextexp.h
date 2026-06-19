// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_VSCALEEXTEXP_H_
#define XNNPACK_SRC_XNNPACK_VSCALEEXTEXP_H_

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define XNN_UKERNEL(arch_flags, fn_name, element_tile, datatype, params_type, \
                    init_params)                                              \
  XNN_INTERNAL void fn_name(size_t n, const float* input, float* output,      \
                            float scale_mantissa, float scale_exponent);
#include "src/f32-vscaleextexp/f32-vscaleextexp.inc"
#undef XNN_UKERNEL

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // XNNPACK_SRC_XNNPACK_VSCALEEXTEXP_H_
