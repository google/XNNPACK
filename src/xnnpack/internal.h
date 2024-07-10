// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// This file contains internal functions that are not part of the public API.

#ifndef THIRD_PARTY_XNNPACK_SRC_XNNPACK_INTERNAL_H_
#define THIRD_PARTY_XNNPACK_SRC_XNNPACK_INTERNAL_H_

#include <stddef.h>
#include <stdint.h>

#include "xnnpack.h"
#include "pthreadpool.h"

#ifdef __cplusplus
extern "C" {
#endif

/// If set, try to pack the quantized values for use by a GEMM.
#define XNN_FLAG_MAYBE_PACK_FOR_GEMM 0x00000080

enum xnn_status xnn_create_fully_connected_nc_qp8_f32_qc4w(
    size_t input_channels,              //
    size_t output_channels,             //
    size_t input_stride,                //
    size_t output_stride,               //
    uint8_t kernel_zero_point,          //
    const float* kernel_scale,          //
    const void* kernel,                 //
    const float* bias,                  //
    float output_min,                   //
    float output_max,                   //
    uint32_t flags,                     //
    xnn_code_cache_t code_cache,        //
    xnn_weights_cache_t weights_cache,  //
    xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_setup_fully_connected_nc_qp8_f32_qc4w(
    xnn_operator_t fully_connected_op,  //
    const int8_t* input,                //
    float* output);

enum xnn_status xnn_reshape_fully_connected_nc_qp8_f32_qc4w(
    xnn_operator_t fully_connected_op,  //
    size_t batch_size,                  //
    pthreadpool_t threadpool);

enum xnn_status xnn_create_convert_nc_f32_qp8(uint32_t flags,  //
                                              xnn_operator_t* convert_op_out);

enum xnn_status xnn_reshape_convert_nc_f32_qp8(xnn_operator_t convert_op,  //
                                               size_t batch_size,          //
                                               size_t channels,            //
                                               size_t input_stride,        //
                                               pthreadpool_t threadpool);

enum xnn_status xnn_setup_convert_nc_f32_qp8(xnn_operator_t convert_op,  //
                                             const float* input,         //
                                             int8_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_XNNPACK_SRC_XNNPACK_INTERNAL_H_
