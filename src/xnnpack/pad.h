// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"

#ifdef __cplusplus
extern "C" {
#endif


#define XNN_PAD_UKERNEL(arch_flags, fn_name, tile_size) \
  XNN_INTERNAL void fn_name(                  \
    size_t rows,                              \
    size_t channels,                          \
    size_t pre_padding,                       \
    size_t post_padding,                      \
    const void* input,                        \
    size_t input_stride,                      \
    void* output,                             \
    size_t output_stride,                     \
    const uint32_t fill_pattern);
#include "xx-pad/xx-pad.h"
#undef XNN_PAD_UKERNEL


#ifdef __cplusplus
}  // extern "C"
#endif
