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


#define XNN_UKERNEL(arch_flags, fn_name, mr, nr, pipelined, kblock, datatype) \
  XNN_INTERNAL void fn_name(                              \
    size_t mc,                                            \
    size_t nc,                                            \
    const float* input,                                   \
    const float* weights,                                 \
    const int32_t* widx_dmap,                             \
    const uint32_t* nidx_nnzmap,                          \
    float* output,                                        \
    size_t output_stride,                                 \
    const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "f32-spmm/f32-spmm-minmax.h"
#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, fn_name, mr, nr, pipelined, kblock, datatype) \
  XNN_INTERNAL void fn_name(                              \
    size_t mc,                                            \
    size_t nc,                                            \
    const xnn_float16* input,                             \
    const xnn_float16* weights,                           \
    const int32_t* widx_dmap,                             \
    const uint32_t* nidx_nnzmap,                          \
    xnn_float16* output,                                  \
    size_t output_stride,                                 \
    const union xnn_f16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "f16-spmm/f16-spmm-minmax.h"
#undef XNN_UKERNEL

#ifdef __cplusplus
}  // extern "C"
#endif
