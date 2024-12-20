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
#include "xnnpack/config-types.h"
#include "pthreadpool.h"

#ifdef __cplusplus
extern "C" {
#endif

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

enum xnn_status xnn_create_fully_connected_nc_qp8_f32_qc8w(
    size_t input_channels,              //
    size_t output_channels,             //
    size_t input_stride,                //
    size_t output_stride,               //
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

enum xnn_status xnn_setup_fully_connected_nc_qp8_f32_qc8w(
    xnn_operator_t fully_connected_op,  //
    const int8_t* input,                //
    float* output);

enum xnn_status xnn_reshape_fully_connected_nc_qp8_f32_qc4w(
    xnn_operator_t fully_connected_op,  //
    size_t batch_size,                  //
    pthreadpool_t threadpool);

enum xnn_status xnn_reshape_fully_connected_nc_qp8_f32_qc8w(
    xnn_operator_t fully_connected_op,  //
    size_t batch_size,                  //
    pthreadpool_t threadpool);

enum xnn_status xnn_create_batch_matrix_multiply_nc_qp8_f32_qc8w(
    size_t batch_size_b,   //
    size_t k,              //
    size_t n,              //
    const int8_t* data_b,  //
    const float* scale_b,  //
    uint32_t flags, xnn_operator_t* batch_matrix_multiply_op_out);

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_qp8_f32_qc8w(
    xnn_operator_t batch_matrix_multiply_op,  //
    size_t num_batch_dims,                    //
    const size_t* batch_dims_a,               //
    const size_t* batch_dims_b,               //
    size_t m,                                 //
    size_t k,                                 //
    size_t n,                                 //
    pthreadpool_t threadpool);

enum xnn_status xnn_setup_batch_matrix_multiply_nc_qp8_f32_qc8w(
    xnn_operator_t batch_matrix_multiply_op,  //
    const int8_t* input_a,                    //
    float* output);

enum xnn_status xnn_create_convert_nc_f32_qp8(
    uint32_t flags,                             //
    const struct xnn_gemm_config* gemm_config,  //
    xnn_operator_t* convert_op_out);

enum xnn_status xnn_reshape_convert_nc_f32_qp8(xnn_operator_t convert_op,  //
                                               size_t num_groups,          //
                                               size_t batch_size,          //
                                               size_t channels,            //
                                               size_t input_stride,        //
                                               pthreadpool_t threadpool);

enum xnn_status xnn_setup_convert_nc_f32_qp8(xnn_operator_t convert_op,  //
                                             const float* input,         //
                                             int8_t* output);

enum xnn_status xnn_create_pack_lh_x32(uint32_t flags,
                                       xnn_operator_t* pack_lh_op_out);

enum xnn_status xnn_reshape_pack_lh_x32(xnn_operator_t pack_lh_op,
                                        size_t batch_size, size_t channels,
                                        size_t* output_size_bytes,
                                        pthreadpool_t threadpool);

enum xnn_status xnn_setup_pack_lh_x32(xnn_operator_t pack_lh_op,
                                      const void* input, void* output);

enum xnn_status xnn_define_pack_lh(xnn_subgraph_t subgraph, uint32_t input_id,
                                   uint32_t output_id, uint32_t flags);

enum xnn_status xnn_create_fully_connected_nc_qp8_f32_qb4w(
    size_t input_channels,              //
    size_t output_channels,             //
    size_t input_stride,                //
    size_t output_stride,               //
    size_t block_size,                  //
    uint8_t kernel_zero_point,          //
    const uint16_t* kernel_scale,       //
    const void* kernel,                 //
    const float* bias,                  //
    float output_min,                   //
    float output_max,                   //
    uint32_t flags,                     //
    xnn_code_cache_t code_cache,        //
    xnn_weights_cache_t weights_cache,  //
    xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_setup_fully_connected_nc_qp8_f32_qb4w(
    xnn_operator_t fully_connected_op,  //
    const int8_t* input,                //
    float* output);

enum xnn_status xnn_reshape_fully_connected_nc_qp8_f32_qb4w(
    xnn_operator_t fully_connected_op,  //
    size_t batch_size,                  //
    pthreadpool_t threadpool);

enum xnn_status xnn_create_fully_connected_nc_pf32(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_code_cache_t code_cache, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_create_convolution2d_nchw_f32_f16(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const void* kernel, const void* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_code_cache_t code_cache, xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out);

enum xnn_status xnn_create_convolution2d_nhwc_pf32(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const float* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_code_cache_t code_cache, xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out);

// quantization_params must be padded with at least
// XNN_EXTRA_QUANTIZATION_PARAMS entries.
enum xnn_status xnn_setup_convert_nc_f16_qdu8(
    xnn_operator_t convert_op, const void* input, uint8_t* output,
    struct xnn_quantization_params* quantization_params);

enum xnn_status xnn_create_convert_nc_f16_qdu8(uint32_t flags,
                                               xnn_operator_t* convert_op_out);

enum xnn_status xnn_reshape_convert_nc_f16_qdu8(
    xnn_operator_t convert_op, size_t batch_size, size_t channels,
    size_t input_stride, size_t output_stride, pthreadpool_t threadpool);

enum xnn_status xnn_create_convert_nc_f32_qdu8(uint32_t flags,
                                               xnn_operator_t* convert_op_out);

enum xnn_status xnn_reshape_convert_nc_f32_qdu8(
    xnn_operator_t convert_op, size_t batch_size, size_t channels,
    size_t input_stride, size_t output_stride, pthreadpool_t threadpool);

// quantization_params must be padded with at least
// XNN_EXTRA_QUANTIZATION_PARAMS entries.
enum xnn_status xnn_setup_convert_nc_f32_qdu8(
    xnn_operator_t convert_op, const float* input, uint8_t* output,
    struct xnn_quantization_params* quantization_params);

enum xnn_status xnn_create_fully_connected_nc_qdu8_f16_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel_scale, const int8_t* kernel,
    const float* bias, float output_min, float output_max, uint32_t flags,
    xnn_code_cache_t code_cache, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_reshape_fully_connected_nc_qdu8_f16_qc8w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    pthreadpool_t threadpool);

enum xnn_status xnn_setup_fully_connected_nc_qdu8_f16_qc8w(
    xnn_operator_t fully_connected_op, const int8_t* input, float* output,
    const struct xnn_quantization_params* quantization_params);

enum xnn_status xnn_create_fully_connected_nc_qdu8_f32_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel_scale, const int8_t* kernel,
    const float* bias, float output_min, float output_max, uint32_t flags,
    xnn_code_cache_t code_cache, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_reshape_fully_connected_nc_qdu8_f32_qc8w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    pthreadpool_t threadpool);

enum xnn_status xnn_setup_fully_connected_nc_qdu8_f32_qc8w(
    xnn_operator_t fully_connected_op, const int8_t* input, float* output,
    const struct xnn_quantization_params* quantization_params);

enum xnn_status xnn_create_convolution2d_nhwc_qdu8_f32_qc8w(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const float* kernel_scale,
    const int8_t* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache, xnn_operator_t* convolution_op_out);

enum xnn_status xnn_reshape_convolution2d_nhwc_qdu8_f32_qc8w(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* workspace_alignment,
    size_t* output_height_out, size_t* output_width_out,
    pthreadpool_t threadpool);

enum xnn_status xnn_setup_convolution2d_nhwc_qdu8_f32_qc8w(
    xnn_operator_t convolution_op, void* workspace, const uint8_t* input,
    float* output, const struct xnn_quantization_params* quantization_params);

enum xnn_status xnn_create_convolution2d_nhwc_qdu8_f16_qc8w(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const float* kernel_scale,
    const int8_t* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache, xnn_operator_t* convolution_op_out);

enum xnn_status xnn_reshape_convolution2d_nhwc_qdu8_f16_qc8w(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* workspace_alignment,
    size_t* output_height_out, size_t* output_width_out,
    pthreadpool_t threadpool);

enum xnn_status xnn_setup_convolution2d_nhwc_qdu8_f16_qc8w(
    xnn_operator_t convolution_op, void* workspace, const int8_t* input,
    void* output, const struct xnn_quantization_params* quantization_params);

enum xnn_status xnn_create_fully_connected_nc_qdu8_f32_qc4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, uint8_t kernel_zero_point, const float* kernel_scale,
    const void* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_reshape_fully_connected_nc_qdu8_f32_qc4w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    pthreadpool_t threadpool);

enum xnn_status xnn_setup_fully_connected_nc_qdu8_f32_qc4w(
    xnn_operator_t fully_connected_op, const uint8_t* input, float* output,
    const struct xnn_quantization_params* quantization_params);

enum xnn_status xnn_create_deconvolution2d_nhwc_qdu8_f32_qc8w(
    uint32_t output_padding_top, uint32_t output_padding_right,
    uint32_t output_padding_bottom, uint32_t output_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t stride_height,
    uint32_t stride_width, uint32_t dilation_height, uint32_t dilation_width,
    uint32_t groups, size_t group_input_channels, size_t group_output_channels,
    size_t input_pixel_stride, size_t output_pixel_stride,
    const float* kernel_scale, const int8_t* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_code_cache_t code_cache, xnn_weights_cache_t weights_cache,
    xnn_operator_t* deconvolution_op_out);

enum xnn_status xnn_reshape_deconvolution2d_nhwc_qdu8_f32_qc8w(
    xnn_operator_t deconvolution_op, size_t batch_size, size_t input_height,
    size_t input_width, uint32_t adjustment_height, uint32_t adjustment_width,
    size_t* output_height_out, size_t* output_width_out,
    pthreadpool_t threadpool);

enum xnn_status xnn_setup_deconvolution2d_nhwc_qdu8_f32_qc8w(
    xnn_operator_t deconvolution_op, const int8_t* input, float* output,
    const struct xnn_quantization_params* quantization_params);

enum xnn_status xnn_create_fully_connected_nc_qdu8_f32_qb4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, size_t block_size, uint8_t kernel_zero_point,
    const uint16_t* kernel_scale, const void* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_code_cache_t code_cache, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_reshape_fully_connected_nc_qdu8_f32_qb4w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    pthreadpool_t threadpool);

enum xnn_status xnn_setup_fully_connected_nc_qdu8_f32_qb4w(
    xnn_operator_t fully_connected_op, const int8_t* input, float* output,
    const struct xnn_quantization_params* quantization_params);

enum xnn_status xnn_create_fully_connected_nc_qdu8_f16_qc4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, uint8_t kernel_zero_point, const float* kernel_scale,
    const void* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_setup_fully_connected_nc_qdu8_f16_qc4w(
    xnn_operator_t fully_connected_op, const int8_t* input, void* output,
    const struct xnn_quantization_params* quantization_params);

enum xnn_status xnn_reshape_fully_connected_nc_qdu8_f16_qc4w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    pthreadpool_t threadpool);

enum xnn_status xnn_create_batch_matrix_multiply_nc_qdu8_f32_qc8w(
    size_t batch_size_b, size_t k, size_t n, const int8_t* data_b,
    const float* scale_b, uint32_t flags,
    xnn_operator_t* batch_matrix_multiply_op);

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_qdu8_f32_qc8w(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, pthreadpool_t threadpool);

enum xnn_status xnn_setup_batch_matrix_multiply_nc_qdu8_f32_qc8w(
    xnn_operator_t batch_matrix_multiply_op, const int8_t* input_a,
    const struct xnn_quantization_params* quantization_params, float* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_XNNPACK_SRC_XNNPACK_INTERNAL_H_
