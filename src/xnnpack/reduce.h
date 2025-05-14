// Copyright 2023 Google LLC
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

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, \
                                datatype_in, datatype_out, params_type,       \
                                init_params)                                  \
  XNN_INTERNAL void ukernel(                                                  \
      size_t batch, const datatype_in* input, datatype_out* output,           \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "src/f16-f32acc-rsum/f16-f32acc-rsum.h"
#include "src/f16-rminmax/f16-rmax.h"
#include "src/f16-rminmax/f16-rmin.h"
#include "src/f16-rminmax/f16-rminmax.h"
#include "src/f16-rsum/f16-rsum.h"
#include "src/f32-rminmax/f32-rmax.h"
#include "src/f32-rminmax/f32-rmin.h"
#include "src/f32-rminmax/f32-rminmax.h"
#include "src/f32-rsum/f32-rsum.h"
#include "src/f32-rsum/f32-rsum.h"
#include "src/qs8-rsum/qs8-rsum.h"
#include "src/qu8-rsum/qu8-rsum.h"
#undef XNN_UKERNEL_WITH_PARAMS

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, \
                                datatype_in, datatype_out, params_type,       \
                                init_params)                                  \
  XNN_INTERNAL void ukernel(                                                  \
      size_t batch, const datatype_in* input, datatype_out* output,           \
      const void* params);
#include "src/u8-rminmax/u8-rminmax.h"
#include "src/u8-rminmax/u8-rmax.h"
#include "src/u8-rminmax/u8-rmin.h"
#include "src/s8-rminmax/s8-rminmax.h"
#include "src/s8-rminmax/s8-rmax.h"
#include "src/s8-rminmax/s8-rmin.h"
#undef XNN_UKERNEL_WITH_PARAMS

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, row_tile, batch_tile, \
                                vector_tile, datatype_in, datatype_out,    \
                                params_type, init_params)                  \
  XNN_INTERNAL void ukernel(                                               \
      size_t rows, size_t channels, const datatype_in* input,              \
      size_t input_stride, const datatype_in* zero, datatype_out* output,  \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "src/f16-f32acc-rdsum/f16-f32acc-rdsum.h"
#include "src/f32-rdsum/f32-rdsum.h"
#include "src/qs8-rdsum/qs8-rdsum-minmax-fp32.h"
#include "src/qu8-rdsum/qu8-rdsum.h"
#undef XNN_UKERNEL_WITH_PARAMS

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, row_tile, batch_tile, \
                                vector_tile, datatype_in, datatype_out,    \
                                params_type, init_params)                  \
  XNN_INTERNAL void ukernel(size_t rows, size_t channels,                  \
                            const datatype_in* input, size_t input_stride, \
                            const datatype_in* zero, datatype_out* output, \
                            const void* params);
#include "src/f16-rdminmax/f16-rdmax.h"
#include "src/f16-rdminmax/f16-rdmin.h"
#include "src/f32-rdminmax/f32-rdmax.h"
#include "src/f32-rdminmax/f32-rdmin.h"
#include "src/s8-rdminmax/s8-rdmax.h"
#include "src/s8-rdminmax/s8-rdmin.h"
#include "src/u8-rdminmax/u8-rdmax.h"
#include "src/u8-rdminmax/u8-rdmin.h"
#undef XNN_UKERNEL_WITH_PARAMS

#ifdef __cplusplus
}  // extern "C"
#endif
