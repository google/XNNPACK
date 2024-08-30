// Copyright 2023 Google LLC
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


#define XNN_UKERNEL_WITH_PARAMS(arch_flags, fn_name, batch_tile, vector_tile, \
                                datatype, params_type, init_params)           \
  XNN_INTERNAL void fn_name(                          \
      size_t n,                                       \
      const int8_t* input,                            \
      int8_t* output,                                 \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "src/qs8-vhswish/qs8-vhswish.h"
#undef XNN_UKERNEL
#undef XNN_UKERNEL_WITH_PARAMS

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, fn_name, batch_tile, vector_tile, \
                                datatype, params_type, init_params)           \
  XNN_INTERNAL void fn_name(                          \
      size_t n,                                       \
      const uint8_t* input,                            \
      uint8_t* output,                                 \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "src/qu8-vhswish/qu8-vhswish.h"
#undef XNN_UKERNEL
#undef XNN_UKERNEL_WITH_PARAMS


#ifdef __cplusplus
}  // extern "C"
#endif
