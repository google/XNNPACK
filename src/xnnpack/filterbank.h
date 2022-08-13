// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <xnnpack/common.h>

#ifdef __cplusplus
extern "C" {
#endif


#define DECLARE_U32_FILTERBANK_ACCUMULATE_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                        \
    size_t rows,                                                    \
    size_t batch_size,                                              \
    const uint32_t* input,                                          \
    const uint16_t* input_offset,                                   \
    const uint16_t* weight_offset,                                  \
    const uint16_t* weight_widths,                                  \
    const uint16_t* weights,                                        \
    const uint16_t* unweights,                                      \
    uint64_t* output);


DECLARE_U32_FILTERBANK_ACCUMULATE_UKERNEL_FUNCTION(xnn_u32_filterbank_accumulate_ukernel__neon_x1)
DECLARE_U32_FILTERBANK_ACCUMULATE_UKERNEL_FUNCTION(xnn_u32_filterbank_accumulate_ukernel__scalar_x1)

#ifdef __cplusplus
}  // extern "C"
#endif
