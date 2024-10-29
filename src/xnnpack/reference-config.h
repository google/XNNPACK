// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdbool.h>
#include <stddef.h>

#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/config-types.h"

#ifdef __cplusplus
extern "C" {
#endif

XNN_INTERNAL const struct xnn_unary_elementwise_config*
xnn_init_unary_reference_config(enum xnn_unary_operator op,
                                enum xnn_datatype input_datatype,
                                enum xnn_datatype output_datatype);
XNN_INTERNAL const struct xnn_binary_elementwise_config*
xnn_init_binary_reference_config(enum xnn_binary_operator op,
                                 enum xnn_datatype datatype);

#ifdef __cplusplus
}  // extern "C"
#endif
