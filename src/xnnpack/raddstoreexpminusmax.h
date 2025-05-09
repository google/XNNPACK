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

#define XNN_UKERNEL(arch_flags, fn_name, element_tile, datatype) \
  XNN_INTERNAL void fn_name(                                   \
      size_t n,                                                \
      const datatype* input,                                   \
      const datatype* max,                                     \
      datatype* output,                                        \
      datatype* sum,                                           \
      const void* params);
#include "f16-raddstoreexpminusmax/f16-raddstoreexpminusmax.h"
#include "f32-raddstoreexpminusmax/f32-raddstoreexpminusmax.h"
#undef XNN_UKERNEL

#ifdef __cplusplus
} /* extern "C" */
#endif
