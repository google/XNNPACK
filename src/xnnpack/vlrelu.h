// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"

#ifdef __cplusplus
extern "C" {
#endif

#define XNN_UKERNEL(arch_flags, fn_name, batch_tile, vector_tile, datatype) \
  XNN_INTERNAL void fn_name(size_t n, const int8_t* input, int8_t* output,  \
                            const struct xnn_qs8_lrelu_params                \
                                params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "src/qs8-vlrelu/qs8-vlrelu.h"
#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, fn_name, batch_tile, vector_tile, datatype)  \
  XNN_INTERNAL void fn_name(size_t n, const uint8_t* input, uint8_t* output, \
                            const struct xnn_qu8_lrelu_params                 \
                                params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "src/qu8-vlrelu/qu8-vlrelu.h"
#undef XNN_UKERNEL

#ifdef __cplusplus
}  // extern "C"
#endif
