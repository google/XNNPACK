// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_REDUCE_H_
#define XNNPACK_SRC_XNNPACK_REDUCE_H_

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"  // IWYU pragma: keep

#ifdef __cplusplus
extern "C" {
#endif

#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype_in, \
                    datatype_out, params_type, init_params)                    \
  XNN_INTERNAL void ukernel(size_t batch, const datatype_in* input,            \
                            datatype_out* output, const params_type* params);
#include "src/f16-f32acc-rsum/f16-f32acc-rsum.inc"
#include "src/f16-f32acc-rsum2/f16-f32acc-rsum2.inc"
#include "src/f16-rminmax/f16-rmax.inc"
#include "src/f16-rminmax/f16-rmin.inc"
#include "src/f16-rminmax/f16-rminmax.inc"
#include "src/f16-rsum/f16-rsum.inc"
#include "src/f32-rminmax/f32-rmax.inc"
#include "src/f32-rminmax/f32-rmin.inc"
#include "src/f32-rminmax/f32-rminmax.inc"
#include "src/f32-rsum/f32-rsum.inc"
#include "src/f32-rsum2/f32-rsum2.inc"
#include "src/qs8-rsum/qs8-rsum.inc"
#include "src/qu8-rsum/qu8-rsum.inc"
#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype_in, \
                    datatype_out, params_type, init_params)                    \
  XNN_INTERNAL void ukernel(size_t batch, const datatype_in* input,            \
                            datatype_out* output, const void* params);
#include "src/s8-rminmax/s8-rmax.inc"
#include "src/s8-rminmax/s8-rmin.inc"
#include "src/s8-rminmax/s8-rminmax.inc"
#include "src/u8-rminmax/u8-rmax.inc"
#include "src/u8-rminmax/u8-rmin.inc"
#include "src/u8-rminmax/u8-rminmax.inc"
#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, ukernel, row_tile, batch_tile, vector_tile,   \
                    datatype_in, datatype_out, params_type, init_params)      \
  XNN_INTERNAL void ukernel(size_t channels, size_t k1, size_t k2, size_t k3, \
                            const datatype_in* input, size_t input_stride1,   \
                            size_t input_stride2, size_t input_stride3,       \
                            const datatype_in* zero, datatype_out* output,    \
                            const params_type* params);
#include "src/f16-f32acc-rdsum/f16-f32acc-rdsum.inc"
#include "src/f16-f32acc-rdsum2/f16-f32acc-rdsum2.inc"
#include "src/f32-rdsum/f32-rdsum.inc"
#include "src/f32-rdsum2/f32-rdsum2.inc"
#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, ukernel, row_tile, batch_tile, vector_tile, \
                    datatype_in, datatype_out, params_type, init_params)    \
  XNN_INTERNAL void ukernel(size_t rows, size_t channels,                   \
                            const datatype_in* input, size_t input_stride,  \
                            const datatype_in* zero, datatype_out* output,  \
                            const params_type* params);
#include "src/qs8-rdsum/qs8-rdsum-minmax-fp32.inc"
#include "src/qu8-rdsum/qu8-rdsum.inc"
#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, ukernel, row_tile, batch_tile, vector_tile, \
                    datatype_in, datatype_out, params_type, init_params)    \
  XNN_INTERNAL void ukernel(size_t rows, size_t channels,                   \
                            const datatype_in* input, size_t input_stride,  \
                            const datatype_in* zero, datatype_out* output,  \
                            const void* params);
#include "src/f16-rdminmax/f16-rdmax.inc"
#include "src/f16-rdminmax/f16-rdmin.inc"
#include "src/f32-rdminmax/f32-rdmax.inc"
#include "src/f32-rdminmax/f32-rdmin.inc"
#include "src/s8-rdminmax/s8-rdmax.inc"
#include "src/s8-rdminmax/s8-rdmin.inc"
#include "src/u8-rdminmax/u8-rdmax.inc"
#include "src/u8-rdminmax/u8-rdmin.inc"
#undef XNN_UKERNEL

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_SRC_XNNPACK_REDUCE_H_
