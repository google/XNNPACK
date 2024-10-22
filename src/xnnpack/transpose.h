// Copyright 2021 Google LLC
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

#define XNN_TRANSPOSE_UKERNEL(arch_flags, fn_name, datasize, datatype, ...) \
  XNN_INTERNAL void fn_name(                            \
      const void* input,                                \
      void* output,                                     \
      size_t input_row_stride,                          \
      size_t output_row_stride,                         \
      size_t input_element_stride,                      \
      size_t output_element_stride,                     \
      size_t element_size,                              \
      size_t block_width,                               \
      size_t block_height);
#include "xx-transposev/xx-transposev.h"
#undef XNN_TRANSPOSE_UKERNEL

#define XNN_TRANSPOSE_UKERNEL(arch_flags, fn_name, datasize, datatype, ...) \
  XNN_INTERNAL void fn_name(                             \
      const datatype* input,                             \
      datatype* output,                                  \
      size_t input_stride,                               \
      size_t output_stride,                              \
      size_t block_width,                                \
      size_t block_height);
#include "x8-transposec/x8-transposec.h"
#include "x16-transposec/x16-transposec.h"
#include "x24-transposec/x24-transposec.h"
#include "x32-transposec/x32-transposec.h"
#include "x64-transposec/x64-transposec.h"
#undef XNN_TRANSPOSE_UKERNEL

#ifdef __cplusplus
}  // extern "C"
#endif
