// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_AVGPOOL_H_
#define XNNPACK_SRC_XNNPACK_AVGPOOL_H_

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef xnn_float16 xnn_float16_avgpool_multiplier_t;
typedef float float_avgpool_multiplier_t;
typedef float int8_t_avgpool_multiplier_t;
typedef float uint8_t_avgpool_multiplier_t;

#define XNN_UKERNEL(arch_flags, ukernel, channel_tile, primary_tile, datatype, \
                    params_type, init_params)                                  \
  XNN_INTERNAL void ukernel(                                                   \
      size_t output_pixels, size_t kernel_elements, size_t channels,           \
      const datatype** input, size_t input_offset, size_t input_pixel_stride,  \
      const datatype* zero, const datatype##_avgpool_multiplier_t* multiplier, \
      datatype* output, size_t input_increment, size_t output_increment,       \
      const params_type* params);

#include "src/f16-avgpool/f16-avgpool-minmax.inc"
#include "src/f32-avgpool/f32-avgpool-minmax.inc"
#include "src/qs8-avgpool/qs8-avgpool-minmax-fp32.inc"
#include "src/qu8-avgpool/qu8-avgpool-minmax-fp32.inc"

#undef XNN_UKERNEL

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_SRC_XNNPACK_AVGPOOL_H_
