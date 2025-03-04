// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
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

#define XNN_UKERNEL(arch_flags, fn_name, c_block, adj_c_block, cr, kr,         \
                    datatype, weights_type, params_type, init_fn)              \
  XNN_INTERNAL void fn_name(                                                   \
      size_t channels, size_t output_width, const datatype** input,            \
      const weights_type* weights, datatype* output, intptr_t input_stride,    \
      size_t output_increment, size_t input_offset, size_t input_pixel_stride, \
      const datatype* zero,                                                    \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "src/f16-dwconv/f16-dwconv-minmax.h"
#include "src/f32-dwconv/f32-dwconv-minmax.h"
#include "src/f32-dwconv/f32-dwconv.h"
#include "src/qs8-dwconv/qs8-dwconv-minmax-fp32.h"
#include "src/qs8-dwconv/qs8-dwconv-minmax-rndnu.h"
#include "src/qs8-qc8w-dwconv/qs8-qc8w-dwconv-minmax-fp32.h"
#include "src/qu8-dwconv/qu8-dwconv-minmax-fp32.h"
#include "src/qu8-dwconv/qu8-dwconv-minmax-rndnu.h"
#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, fn_name, _kernel_height_, _kernel_width_, _subsampling_, _padding_, _height_tile_, _width_tile_, datatype, params_type, init_params) \
  XNN_INTERNAL void fn_name(                                      \
    size_t input_height,                                          \
    size_t input_width,                                           \
    const datatype* input,                                        \
    const datatype* weights,                                      \
    const datatype* zero,                                         \
    datatype* output,                                             \
    uint32_t padding_top,                                         \
    const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "src/f16-dwconv2d-chw/f16-dwconv2d-chw.h"
#include "src/f32-dwconv2d-chw/f32-dwconv2d-chw.h"
#undef XNN_UKERNEL

#ifdef __cplusplus
}  // extern "C"
#endif
