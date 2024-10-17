// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef void (*xnn_qu8_requantization_fn)(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output);

#define DECLARE_QU8_REQUANTIZATION_FUNCTION(fn_name) \
    void fn_name(                                    \
        size_t n,                                    \
        const int32_t* input,                        \
        float scale,                                 \
        uint8_t zero_point,                          \
        uint8_t qmin,                                \
        uint8_t qmax,                                \
        uint8_t* output);

DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_fp32__neon)
DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_fp32__scalar_fmagic)
DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_fp32__scalar_lrintf)
DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_fp32__sse2)
DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_fp32__wasmsimd)

DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_gemmlowp__neon)
DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_gemmlowp__scalar)
DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_gemmlowp__sse2)
DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_gemmlowp__sse41)
DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_gemmlowp__ssse3)
DECLARE_QU8_REQUANTIZATION_FUNCTION(xnn_qu8_requantize_gemmlowp__wasmsimd)


typedef void (*xnn_qs8_requantization_fn)(
    size_t n,
    const int32_t* input,
    float scale,
    int8_t zero_point,
    int8_t qmin,
    int8_t qmax,
    int8_t* output);

#define DECLARE_QS8_REQUANTIZATION_FUNCTION(fn_name) \
    void fn_name(                                    \
        size_t n,                                    \
        const int32_t* input,                        \
        float scale,                                 \
        int8_t zero_point,                           \
        int8_t qmin,                                 \
        int8_t qmax,                                 \
        int8_t* output);

DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_fp32__neon)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_fp32__scalar_fmagic)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_fp32__scalar_lrintf)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_fp32__sse2)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_fp32__sse41)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_fp32__wasmsimd)

DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_gemmlowp__neon)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_gemmlowp__scalar)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_gemmlowp__sse2)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_gemmlowp__sse41)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_gemmlowp__ssse3)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_gemmlowp__wasmsimd)

DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_rndnu__neon_mull)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_rndnu__neon_qdmulh)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_rndnu__scalar)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_rndnu__sse41_sra)
DECLARE_QS8_REQUANTIZATION_FUNCTION(xnn_qs8_requantize_rndnu__sse41_srl)


#ifdef __cplusplus
}  // extern "C"
#endif
