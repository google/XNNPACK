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

#define XNN_UKERNEL(arch_flags, ukernel, channel_tile, primary_tile, datatype, \
                    params_type, init_params)                                  \
  XNN_INTERNAL void ukernel(                                                   \
      size_t output_pixels, size_t kernel_elements, size_t channels,           \
      const datatype** input, size_t input_offset, size_t input_pixel_stride,  \
      const datatype* zero, const datatype* multiplier, datatype* output,      \
      size_t input_increment, size_t output_increment,                         \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

#include "src/f16-avgpool/f16-avgpool-minmax.h"
#include "src/f32-avgpool/f32-avgpool-minmax.h"

#undef XNN_UKERNEL

#ifdef __cplusplus
}  // extern "C"
#endif
