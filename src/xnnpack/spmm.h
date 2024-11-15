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


#define XNN_UKERNEL(arch_flags, fn_name, mr, nr, pipelined, kblock, datatype, params_fn) \
  XNN_INTERNAL void fn_name(                              \
    size_t mc,                                            \
    size_t nc,                                            \
    const datatype* input,                                \
    const datatype* weights,                              \
    const int32_t* widx_dmap,                             \
    const uint32_t* nidx_nnzmap,                          \
    datatype* output,                                     \
    size_t output_stride,                                 \
    const params_fn params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "src/f32-spmm/f32-spmm-minmax.h"
#include "src/f16-spmm/f16-spmm-minmax.h"
#undef XNN_UKERNEL

#ifdef __cplusplus
}  // extern "C"
#endif
