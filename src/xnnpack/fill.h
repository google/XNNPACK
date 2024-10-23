// Copyright 2020 Google LLC
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


#define XNN_FILL_UKERNEL(arch_flags, fn_name) \
  XNN_INTERNAL void fn_name(                   \
    size_t kernel_elements,                    \
    size_t channels,                           \
    void* output,                              \
    size_t output_stride,                      \
    const uint32_t fill_pattern);
#include "xx-fill/xx-fill.h"
#undef XNN_FILL_UKERNEL

#ifdef __cplusplus
}  // extern "C"
#endif
