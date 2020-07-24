// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdint.h>
#include <stddef.h>

#include <xnnpack/params.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*requantization_function)(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output);

#define DECLARE_REQUANTIZATION_FUNCTION(fn_name) \
    void fn_name( \
        size_t n, \
        const int32_t* input, \
        float scale, \
        uint8_t zero_point, \
        uint8_t qmin, \
        uint8_t qmax, \
        uint8_t* output);

DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_precise__scalar_unsigned32)
DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_precise__scalar_unsigned64)
DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_precise__scalar_signed64)
DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_precise__sse2)
DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_precise__ssse3)
DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_precise__sse4)
DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_precise__neon)
DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_precise__psimd)

DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_fp32__scalar_lrintf)
DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_fp32__scalar_magic)
DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_fp32__sse2)
DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_fp32__neon)
DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_fp32__psimd)
DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_fp32__wasmsimd)

DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_q31__scalar)
DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_q31__sse2)
DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_q31__ssse3)
DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_q31__sse4)
DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_q31__neon)
DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_q31__psimd)

DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_gemmlowp__scalar)
DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_gemmlowp__sse2)
DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_gemmlowp__ssse3)
DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_gemmlowp__sse4)
DECLARE_REQUANTIZATION_FUNCTION(xnn_requantize_gemmlowp__neon)

#ifdef __cplusplus
}  // extern "C"
#endif
