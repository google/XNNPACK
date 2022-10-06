// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>

// Operators that can be applied post convolution.
enum xnn_post_operation_type {
  xnn_post_operation_type_none,
  xnn_post_operation_type_hardswish,
};

// Struct representing a post operation and its associated data. For example,
// an addition with constant will specify the constant in the arg1 field, a
// clamp will specify min in arg1, and max in arg2.
struct xnn_post_operation {
  enum xnn_post_operation_type op_type;
  float arg1;
  float arg2;
};


#ifdef __cplusplus
extern "C" {
#endif

// Allocate space for params required for post_operations and initialize all params.
// This allocation will be freed when the operator holding these params is deleted.
char* allocate_and_initialize_post_operation_params(
    size_t num_post_operations,
    const struct xnn_post_operation* post_operations);

#ifdef __cplusplus
}  // extern "C"
#endif
