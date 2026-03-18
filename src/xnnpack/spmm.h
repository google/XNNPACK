// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_SPMM_H_
#define XNNPACK_SRC_XNNPACK_SPMM_H_

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"

#ifdef __cplusplus
extern "C" {
#endif

#define XNN_UKERNEL(arch_flags, ukernel, mr, nr, k_block, vector_tile,         \
                    pipelined, datatype, params_type, init_params)             \
  XNN_INTERNAL void ukernel(size_t mc, size_t nc, const datatype* input,       \
                            const datatype* weights, const int32_t* widx_dmap, \
                            const uint32_t* nidx_nnzmap, datatype* output,     \
                            size_t output_stride, const params_type* params);
#include "src/f16-spmm/f16-spmm-minmax.inc"
#include "src/f32-spmm/f32-spmm-minmax.inc"
#undef XNN_UKERNEL

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_SRC_XNNPACK_SPMM_H_
