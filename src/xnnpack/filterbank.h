// Copyright 2022 Google LLC
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


#define DECLARE_U32_FILTERBANK_ACCUMULATE_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                        \
    size_t rows,                                                    \
    const uint32_t* input,                                          \
    const uint8_t* weight_widths,                                   \
    const uint16_t* weights,                                        \
    uint64_t* output);

DECLARE_U32_FILTERBANK_ACCUMULATE_UKERNEL_FUNCTION(xnn_u32_filterbank_accumulate_ukernel__asm_aarch32_arm_x1)
DECLARE_U32_FILTERBANK_ACCUMULATE_UKERNEL_FUNCTION(xnn_u32_filterbank_accumulate_ukernel__asm_aarch32_neon_x1)
DECLARE_U32_FILTERBANK_ACCUMULATE_UKERNEL_FUNCTION(xnn_u32_filterbank_accumulate_ukernel__asm_aarch32_neon_x2)
DECLARE_U32_FILTERBANK_ACCUMULATE_UKERNEL_FUNCTION(xnn_u32_filterbank_accumulate_ukernel__neon_x1)
DECLARE_U32_FILTERBANK_ACCUMULATE_UKERNEL_FUNCTION(xnn_u32_filterbank_accumulate_ukernel__neon_x2)
DECLARE_U32_FILTERBANK_ACCUMULATE_UKERNEL_FUNCTION(xnn_u32_filterbank_accumulate_ukernel__scalar_x1)


#define DECLARE_U32_FILTERBANK_SUBTRACT_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                      \
    size_t batch_size,                                            \
    const uint32_t* input,                                        \
    uint32_t smoothing,                                           \
    uint32_t alternate_smoothing,                                 \
    uint32_t one_minus_smoothing,                                 \
    uint32_t alternate_one_minus_smoothing,                       \
    uint32_t min_signal_remaining,                                \
    uint32_t smoothing_bits,                                      \
    uint32_t spectral_subtraction_bits,                           \
    uint32_t* noise_estimate,                                     \
    uint32_t* output);


DECLARE_U32_FILTERBANK_SUBTRACT_UKERNEL_FUNCTION(xnn_u32_filterbank_subtract_ukernel__scalar_x2)

#ifdef __cplusplus
}  // extern "C"
#endif
