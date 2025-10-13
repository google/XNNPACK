// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "include/xnnpack.h"
#include "ynnpack/base/log.h"
#include <pthreadpool.h>

extern "C" {

xnn_status xnn_run_operator(xnn_operator_t op, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_delete_operator(xnn_operator_t op) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_binary_elementwise_nd(
    xnn_binary_operator type, xnn_datatype datatype,
    const xnn_quantization_params* a_quantization,
    const xnn_quantization_params* b_quantization,
    const xnn_quantization_params* output_quantization, uint32_t flags,
    xnn_operator_t* binary_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_binary_elementwise_nd(xnn_operator_t binary_op,
                                             size_t num_input1_dims,
                                             const size_t* input1_shape,
                                             size_t num_input2_dims,
                                             const size_t* input2_shape,
                                             pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_binary_elementwise_nd(xnn_operator_t binary_op,
                                           const void* input1,
                                           const void* input2, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_run_binary_elementwise_nd(
    xnn_binary_operator type, xnn_datatype datatype,
    const xnn_quantization_params* input1_quantization,
    const xnn_quantization_params* input2_quantization,
    const xnn_quantization_params* output_quantization, uint32_t flags,
    size_t num_input1_dims, const size_t* input1_shape, size_t num_input2_dims,
    const size_t* input2_shape, const void* input1, const void* input2,
    void* output, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_unary_elementwise_nc(
    xnn_unary_operator op_type, xnn_datatype input_datatype,
    xnn_datatype output_datatype, const union xnn_unary_params* params,
    const void* lut, const xnn_quantization_params* input_quantization,
    const xnn_quantization_params* output_quantization, uint32_t flags,
    xnn_operator_t* op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_unary_elementwise_nc(xnn_operator_t op,
                                            size_t batch_size, size_t channels,
                                            size_t input_stride,
                                            size_t output_stride,
                                            pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_unary_elementwise_nc(xnn_operator_t op, const void* input,
                                          void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_run_unary_elementwise_nc(
    xnn_unary_operator op_type, xnn_datatype input_datatype,
    xnn_datatype output_datatype, const union xnn_unary_params* params,
    const xnn_quantization_params* input_quantization,
    const xnn_quantization_params* output_quantization, uint32_t flags,
    size_t batch_size, size_t channels, size_t input_stride,
    size_t output_stride, pthreadpool_t threadpool, const void* input,
    void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_argmax_pooling2d_nhwc_f32(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t pooling_height, uint32_t pooling_width, uint32_t flags,
    xnn_operator_t* argmax_pooling_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_argmax_pooling2d_nhwc_f32(
    xnn_operator_t argmax_pooling_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t channels, size_t input_pixel_stride,
    size_t output_pixel_stride, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_argmax_pooling2d_nhwc_f32(xnn_operator_t argmax_pooling_op,
                                               const float* input,
                                               float* output, uint32_t* index) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_average_pooling2d_nhwc_f16(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t pooling_height, uint32_t pooling_width, uint32_t stride_height,
    uint32_t stride_width, float output_min, float output_max, uint32_t flags,
    xnn_operator_t* average_pooling_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_average_pooling2d_nhwc_f16(
    xnn_operator_t average_pooling_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t channels, size_t input_pixel_stride,
    size_t output_pixel_stride, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_average_pooling2d_nhwc_f16(
    xnn_operator_t average_pooling_op, const void* input, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_average_pooling2d_nhwc_f32(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t pooling_height, uint32_t pooling_width, uint32_t stride_height,
    uint32_t stride_width, float output_min, float output_max, uint32_t flags,
    xnn_operator_t* average_pooling_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_average_pooling2d_nhwc_f32(
    xnn_operator_t average_pooling_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t channels, size_t input_pixel_stride,
    size_t output_pixel_stride, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_average_pooling2d_nhwc_f32(
    xnn_operator_t average_pooling_op, const float* input, float* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_batch_matrix_multiply_nc_f16(
    uint32_t flags, xnn_operator_t* batch_matrix_multiply_op) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_batch_matrix_multiply_nc_bf16_f32(
    uint32_t flags, xnn_operator_t* batch_matrix_multiply_op) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_batch_matrix_multiply_nc_f16_const_weights(
    size_t batch_size_b, size_t k, size_t n, const void* data_b, uint32_t flags,
    xnn_operator_t* batch_matrix_multiply_op) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_batch_matrix_multiply_nc_f16(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, size_t* workspace_size, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_batch_matrix_multiply_nc_f16(
    xnn_operator_t batch_matrix_multiply_op, void* workspace,
    const void* input_a, const void* input_b, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_batch_matrix_multiply_nc_bf16_f32(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, size_t* workspace_size, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_batch_matrix_multiply_nc_bf16_f32(
    xnn_operator_t batch_matrix_multiply_op, void* workspace,
    const void* input_a, const void* input_b, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_batch_matrix_multiply_nc_f32(
    uint32_t flags, xnn_operator_t* batch_matrix_multiply_op) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_batch_matrix_multiply_nc_f32_const_weights(
    size_t batch_size_b, size_t k, size_t n, const float* data_b,
    uint32_t flags, xnn_operator_t* batch_matrix_multiply_op) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_batch_matrix_multiply_nc_f32(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, size_t* workspace_size, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_batch_matrix_multiply_nc_f32(
    xnn_operator_t batch_matrix_multiply_op, void* workspace,
    const float* input_a, const float* input_b, float* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_batch_matrix_multiply_nc_qs8_const_weights(
    size_t batch_size_b, size_t k, size_t n, const void* data_b,
    int8_t input_zero_point, int8_t output_zero_point, int8_t output_min,
    int8_t output_max, float requantization_scale, uint32_t flags,
    xnn_operator_t* batch_matrix_multiply_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_batch_matrix_multiply_nc_qs8(
    int8_t input_zero_point, int8_t output_zero_point, int8_t output_min,
    int8_t output_max, const float* scale_b, uint32_t flags,
    xnn_operator_t* batch_matrix_multiply_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_batch_matrix_multiply_nc_qs8_const_weights(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_batch_matrix_multiply_nc_qs8(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, size_t* workspace_size, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_batch_matrix_multiply_nc_qs8(
    xnn_operator_t batch_matrix_multiply_op, void* workspace,
    const int8_t* input_a, const int8_t* input_b, int8_t* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_batch_matrix_multiply_nc_qd8_f32_qc8w(
    size_t batch_size_b, size_t k, size_t n, const int8_t* data_b,
    const float* scale_b, uint32_t flags,
    xnn_operator_t* batch_matrix_multiply_op) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_batch_matrix_multiply_nc_qd8_f32_qc8w(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, size_t* workspace_size, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_batch_matrix_multiply_nc_qd8_f32_qc8w(
    xnn_operator_t batch_matrix_multiply_op, void* workspace,
    const int8_t* input_a, const int8_t* input_b,
    const xnn_quantization_params* quantization_params, float* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_constant_pad_nd_x8(const void* padding_value,
                                         uint32_t flags,
                                         xnn_operator_t* constant_pad_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_constant_pad_nd_x8(xnn_operator_t constant_pad_op,
                                          size_t num_dims,
                                          const size_t* input_shape,
                                          const size_t* pre_padding,
                                          const size_t* post_padding,
                                          pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_constant_pad_nd_x8(xnn_operator_t constant_pad_op,
                                        const void* input, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_run_constant_pad_nd_x8(
    uint32_t flags, size_t num_dims, const size_t* input_shape,
    const size_t* pre_paddings, const size_t* post_paddings, const void* input,
    void* output, const void* padding_value, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_constant_pad_nd_x16(const void* padding_value,
                                          uint32_t flags,
                                          xnn_operator_t* constant_pad_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_constant_pad_nd_x16(xnn_operator_t constant_pad_op,
                                           size_t num_dims,
                                           const size_t* input_shape,
                                           const size_t* pre_padding,
                                           const size_t* post_padding,
                                           pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_constant_pad_nd_x16(xnn_operator_t constant_pad_op,
                                         const void* input, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_run_constant_pad_nd_x16(
    uint32_t flags, size_t num_dims, const size_t* input_shape,
    const size_t* pre_paddings, const size_t* post_paddings, const void* input,
    void* output, const void* padding_value, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_constant_pad_nd_x32(const void* padding_value,
                                          uint32_t flags,
                                          xnn_operator_t* constant_pad_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_constant_pad_nd_x32(xnn_operator_t constant_pad_op,
                                           size_t num_dims,
                                           const size_t* input_shape,
                                           const size_t* pre_padding,
                                           const size_t* post_padding,
                                           pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_constant_pad_nd_x32(xnn_operator_t constant_pad_op,
                                         const void* input, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_run_constant_pad_nd_x32(
    uint32_t flags, size_t num_dims, const size_t* input_shape,
    const size_t* pre_paddings, const size_t* post_paddings, const void* input,
    void* output, const void* padding_value, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_convert_nc_f16_qd8(uint32_t flags,
                                         xnn_operator_t* convert_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_convert_nc_f16_qd8(xnn_operator_t convert_op,
                                          size_t batch_size, size_t channels,
                                          size_t input_stride,
                                          size_t output_stride,
                                          pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_convert_nc_f16_qd8(
    xnn_operator_t convert_op, const void* input, int8_t* output,
    xnn_quantization_params* quantization_params) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_convert_nc_f32_qd8(uint32_t flags,
                                         xnn_operator_t* convert_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_convert_nc_f32_qd8(xnn_operator_t convert_op,
                                          size_t batch_size, size_t channels,
                                          size_t input_stride,
                                          size_t output_stride,
                                          pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_convert_nc_f32_qd8(
    xnn_operator_t convert_op, const float* input, int8_t* output,
    xnn_quantization_params* quantization_params) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

XNN_DEPRECATED xnn_status xnn_run_convert_nc_f32_f16(
    size_t channels, size_t input_stride, size_t output_stride,
    size_t batch_size, const float* input, void* output, uint32_t flags,
    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_convolution2d_nchw_f16(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const void* kernel, const void* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* convolution_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_convolution2d_nchw_f16(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* output_height_out, size_t* output_width_out,
    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_convolution2d_nchw_f16(xnn_operator_t convolution_op,
                                            const void* input, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_convolution2d_nchw_f32(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const float* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* convolution_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_convolution2d_nchw_f32(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* output_height_out, size_t* output_width_out,
    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_convolution2d_nchw_f32(xnn_operator_t convolution_op,
                                            const float* input, float* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_convolution2d_nhwc_f16(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const void* kernel, const void* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* convolution_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_convolution2d_nhwc_f16(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_convolution2d_nhwc_f16(xnn_operator_t convolution_op,
                                            void* workspace, const void* input,
                                            void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_convolution2d_nhwc_f32(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const float* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* convolution_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_convolution2d_nhwc_f32_f16(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const void* kernel, const void* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* convolution_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_fused_convolution2d_nhwc_f32(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const float* kernel, const float* bias,
    size_t num_post_operations, xnn_post_operation* post_operations,
    uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_convolution2d_nhwc_f32(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_convolution2d_nhwc_f32(xnn_operator_t convolution_op,
                                            void* workspace, const float* input,
                                            float* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_convolution2d_nhwc_qd8_f16_qc8w(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const float* kernel_scale,
    const int8_t* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_convolution2d_nhwc_qd8_f32_qc8w(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const float* kernel_scale,
    const int8_t* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_convolution2d_nhwc_qs8(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, int8_t input_zero_point, float input_scale,
    float kernel_scale, const int8_t* kernel, const int32_t* bias,
    int8_t output_zero_point, float output_scale, int8_t output_min,
    int8_t output_max, uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_convolution2d_nhwc_qd8_f16_qc8w(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_convolution2d_nhwc_qd8_f32_qc8w(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_convolution2d_nhwc_qs8(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_convolution2d_nhwc_qd8_f16_qc8w(
    xnn_operator_t convolution_op, void* workspace, const int8_t* input,
    void* output, const xnn_quantization_params* quantization_params) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_convolution2d_nhwc_qd8_f32_qc8w(
    xnn_operator_t convolution_op, void* workspace, const int8_t* input,
    float* output, const xnn_quantization_params* quantization_params) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_convolution2d_nhwc_qs8(xnn_operator_t convolution_op,
                                            void* workspace,
                                            const int8_t* input,
                                            int8_t* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_convolution2d_nhwc_qs8_qc8w(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, int8_t input_zero_point, float input_scale,
    const float* kernel_scale, const int8_t* kernel, const int32_t* bias,
    int8_t output_zero_point, float output_scale, int8_t output_min,
    int8_t output_max, uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_convolution2d_nhwc_qs8_qc8w(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_convolution2d_nhwc_qs8_qc8w(xnn_operator_t convolution_op,
                                                 void* workspace,
                                                 const int8_t* input,
                                                 int8_t* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_convolution2d_nhwc_qu8(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, uint8_t input_zero_point, float input_scale,
    uint8_t kernel_zero_point, float kernel_scale, const uint8_t* kernel,
    const int32_t* bias, uint8_t output_zero_point, float output_scale,
    uint8_t output_min, uint8_t output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* convolution_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_convolution2d_nhwc_qu8(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_convolution2d_nhwc_qu8(xnn_operator_t convolution_op,
                                            void* workspace,
                                            const uint8_t* input,
                                            uint8_t* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_copy_nc_x8(uint32_t flags, xnn_operator_t* copy_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_copy_nc_x8(xnn_operator_t copy_op, size_t batch_size,
                                  size_t channels, size_t input_stride,
                                  size_t output_stride,
                                  pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_copy_nc_x8(xnn_operator_t copy_op, const void* input,
                                void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_copy_nc_x16(uint32_t flags, xnn_operator_t* copy_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_copy_nc_x16(xnn_operator_t copy_op, size_t batch_size,
                                   size_t channels, size_t input_stride,
                                   size_t output_stride,
                                   pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_copy_nc_x16(xnn_operator_t copy_op, const void* input,
                                 void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_copy_nc_x32(uint32_t flags, xnn_operator_t* copy_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_copy_nc_x32(xnn_operator_t copy_op, size_t batch_size,
                                   size_t channels, size_t input_stride,
                                   size_t output_stride,
                                   pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_copy_nc_x32(xnn_operator_t copy_op, const void* input,
                                 void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_run_copy_nc_x32(size_t channels, size_t input_stride,
                               size_t output_stride, size_t batch_size,
                               const uint32_t* input, uint32_t* output,
                               uint32_t flags, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_deconvolution2d_nhwc_f16(
    uint32_t output_padding_top, uint32_t output_padding_right,
    uint32_t output_padding_bottom, uint32_t output_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t stride_height,
    uint32_t stride_width, uint32_t dilation_height, uint32_t dilation_width,
    uint32_t groups, size_t group_input_channels, size_t group_output_channels,
    size_t input_pixel_stride, size_t output_pixel_stride, const void* kernel,
    const void* bias, float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* deconvolution_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_deconvolution2d_nhwc_f16(
    xnn_operator_t deconvolution_op, size_t batch_size, size_t input_height,
    size_t input_width, uint32_t adjustment_height, uint32_t adjustment_width,
    size_t* output_height_out, size_t* output_width_out,
    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_deconvolution2d_nhwc_f16(xnn_operator_t deconvolution_op,
                                              const void* input, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_deconvolution2d_nhwc_f32(
    uint32_t output_padding_top, uint32_t output_padding_right,
    uint32_t output_padding_bottom, uint32_t output_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t stride_height,
    uint32_t stride_width, uint32_t dilation_height, uint32_t dilation_width,
    uint32_t groups, size_t group_input_channels, size_t group_output_channels,
    size_t input_pixel_stride, size_t output_pixel_stride, const float* kernel,
    const float* bias, float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* deconvolution_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_deconvolution2d_nhwc_f32_f16(
    uint32_t output_padding_top, uint32_t output_padding_right,
    uint32_t output_padding_bottom, uint32_t output_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t stride_height,
    uint32_t stride_width, uint32_t dilation_height, uint32_t dilation_width,
    uint32_t groups, size_t group_input_channels, size_t group_output_channels,
    size_t input_pixel_stride, size_t output_pixel_stride, const void* kernel,
    const void* bias, float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* deconvolution_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_deconvolution2d_nhwc_f32(
    xnn_operator_t deconvolution_op, size_t batch_size, size_t input_height,
    size_t input_width, uint32_t adjustment_height, uint32_t adjustment_width,
    size_t* output_height_out, size_t* output_width_out,
    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_deconvolution2d_nhwc_f32(xnn_operator_t deconvolution_op,
                                              const float* input,
                                              float* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_deconvolution2d_nhwc_qd8_f32_qc8w(
    uint32_t output_padding_top, uint32_t output_padding_right,
    uint32_t output_padding_bottom, uint32_t output_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t stride_height,
    uint32_t stride_width, uint32_t dilation_height, uint32_t dilation_width,
    uint32_t groups, size_t group_input_channels, size_t group_output_channels,
    size_t input_pixel_stride, size_t output_pixel_stride,
    const float* kernel_scale, const int8_t* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* deconvolution_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_deconvolution2d_nhwc_qd8_f32_qc8w(
    xnn_operator_t deconvolution_op, size_t batch_size, size_t input_height,
    size_t input_width, uint32_t adjustment_height, uint32_t adjustment_width,
    size_t* output_height_out, size_t* output_width_out,
    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_deconvolution2d_nhwc_qd8_f32_qc8w(
    xnn_operator_t deconvolution_op, const int8_t* input, float* output,
    const xnn_quantization_params* quantization_params) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_deconvolution2d_nhwc_qs8(
    uint32_t output_padding_top, uint32_t output_padding_right,
    uint32_t output_padding_bottom, uint32_t output_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t stride_height,
    uint32_t stride_width, uint32_t dilation_height, uint32_t dilation_width,
    uint32_t groups, size_t group_input_channels, size_t group_output_channels,
    size_t input_pixel_stride, size_t output_pixel_stride,
    int8_t input_zero_point, float input_scale, float kernel_scale,
    const int8_t* kernel, const int32_t* bias, int8_t output_zero_point,
    float output_scale, int8_t output_min, int8_t output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* deconvolution_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_deconvolution2d_nhwc_qs8(
    xnn_operator_t deconvolution_op, size_t batch_size, size_t input_height,
    size_t input_width, uint32_t adjustment_height, uint32_t adjustment_width,
    size_t* output_height_out, size_t* output_width_out,
    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_deconvolution2d_nhwc_qs8(xnn_operator_t deconvolution_op,
                                              const int8_t* input,
                                              int8_t* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_deconvolution2d_nhwc_qs8_qc8w(
    uint32_t output_padding_top, uint32_t output_padding_right,
    uint32_t output_padding_bottom, uint32_t output_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t stride_height,
    uint32_t stride_width, uint32_t dilation_height, uint32_t dilation_width,
    uint32_t groups, size_t group_input_channels, size_t group_output_channels,
    size_t input_pixel_stride, size_t output_pixel_stride,
    int8_t input_zero_point, float input_scale, const float* kernel_scale,
    const int8_t* kernel, const int32_t* bias, int8_t output_zero_point,
    float output_scale, int8_t output_min, int8_t output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* deconvolution_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_deconvolution2d_nhwc_qs8_qc8w(
    xnn_operator_t deconvolution_op, size_t batch_size, size_t input_height,
    size_t input_width, uint32_t adjustment_height, uint32_t adjustment_width,
    size_t* output_height_out, size_t* output_width_out,
    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_deconvolution2d_nhwc_qs8_qc8w(
    xnn_operator_t deconvolution_op, const int8_t* input, int8_t* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_deconvolution2d_nhwc_qu8(
    uint32_t output_padding_top, uint32_t output_padding_right,
    uint32_t output_padding_bottom, uint32_t output_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t stride_height,
    uint32_t stride_width, uint32_t dilation_height, uint32_t dilation_width,
    uint32_t groups, size_t group_input_channels, size_t group_output_channels,
    size_t input_pixel_stride, size_t output_pixel_stride,
    uint8_t input_zero_point, float input_scale, uint8_t kernel_zero_point,
    float kernel_scale, const uint8_t* kernel, const int32_t* bias,
    uint8_t output_zero_point, float output_scale, uint8_t output_min,
    uint8_t output_max, uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* deconvolution_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_deconvolution2d_nhwc_qu8(
    xnn_operator_t deconvolution_op, size_t batch_size, size_t input_height,
    size_t input_width, uint32_t adjustment_height, uint32_t adjustment_width,
    size_t* output_height_out, size_t* output_width_out,
    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_deconvolution2d_nhwc_qu8(xnn_operator_t deconvolution_op,
                                              const uint8_t* input,
                                              uint8_t* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_depth_to_space_nchw2nhwc_x16(
    uint32_t block_size, uint32_t flags,
    xnn_operator_t* depth_to_space_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_depth_to_space_nchw2nhwc_x16(
    xnn_operator_t depth_to_space_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t input_channels, size_t* output_height_out,
    size_t* output_width_out, size_t* output_channels_out,
    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_depth_to_space_nchw2nhwc_x16(
    xnn_operator_t depth_to_space_op, const void* input, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_depth_to_space_nchw2nhwc_x32(
    uint32_t block_size, uint32_t flags,
    xnn_operator_t* depth_to_space_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_depth_to_space_nchw2nhwc_x32(
    xnn_operator_t depth_to_space_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t input_channels, size_t* output_height_out,
    size_t* output_width_out, size_t* output_channels_out,
    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_depth_to_space_nchw2nhwc_x32(
    xnn_operator_t depth_to_space_op, const void* input, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_depth_to_space_nhwc_x8(
    uint32_t block_size, uint32_t flags,
    xnn_operator_t* depth_to_space_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_depth_to_space_nhwc_x8(
    xnn_operator_t depth_to_space_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t input_channels, size_t* output_height_out,
    size_t* output_width_out, size_t* output_channels_out,
    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_depth_to_space_nhwc_x8(xnn_operator_t depth_to_space_op,
                                            const void* input, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_depth_to_space_nhwc_x16(
    uint32_t block_size, uint32_t flags,
    xnn_operator_t* depth_to_space_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_depth_to_space_nhwc_x16(
    xnn_operator_t depth_to_space_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t input_channels, size_t* output_height_out,
    size_t* output_width_out, size_t* output_channels_out,
    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_depth_to_space_nhwc_x16(xnn_operator_t depth_to_space_op,
                                             const void* input, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_depth_to_space_nhwc_x32(
    uint32_t block_size, uint32_t flags,
    xnn_operator_t* depth_to_space_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_depth_to_space_nhwc_x32(
    xnn_operator_t depth_to_space_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t input_channels, size_t* output_height_out,
    size_t* output_width_out, size_t* output_channels_out,
    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_depth_to_space_nhwc_x32(xnn_operator_t depth_to_space_op,
                                             const void* input, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_dynamic_fully_connected_nc_f16(
    float output_min, float output_max, uint32_t flags,
    xnn_operator_t* dynamic_fully_connected_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_dynamic_fully_connected_nc_f16(
    xnn_operator_t dynamic_fully_connected_op, size_t batch_size,
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, size_t* workspace_size, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_dynamic_fully_connected_nc_f16(
    xnn_operator_t dynamic_fully_connected_op, void* workspace,
    const void* input, const void* kernel, const void* bias, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_dynamic_fully_connected_nc_f32(
    float output_min, float output_max, uint32_t flags,
    xnn_operator_t* dynamic_fully_connected_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_dynamic_fully_connected_nc_f32(
    xnn_operator_t dynamic_fully_connected_op, size_t batch_size,
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, size_t* workspace_size, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_dynamic_fully_connected_nc_f32(
    xnn_operator_t dynamic_fully_connected_op, void* workspace,
    const float* input, const float* kernel, const float* bias, float* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_fully_connected_nc_f16(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const void* kernel, const void* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_fully_connected_nc_f16(xnn_operator_t fully_connected_op,
                                              size_t batch_size,
                                              pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_fully_connected_nc_f16(xnn_operator_t fully_connected_op,
                                            const void* input, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_fully_connected_nc_f32_f16(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const void* kernel, const void* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_fully_connected_nc_bf16_f32(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const void* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_fully_connected_nc_f32(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_fully_connected_nc_f32_f16(
    xnn_operator_t fully_connected_op, size_t batch_size,
    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_fully_connected_nc_bf16_f32(
    xnn_operator_t fully_connected_op, size_t batch_size,
    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_fully_connected_nc_f32(xnn_operator_t fully_connected_op,
                                              size_t batch_size,
                                              pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_fully_connected_nc_f32_f16(
    xnn_operator_t fully_connected_op, const float* input, float* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_fully_connected_nc_bf16_f32(
    xnn_operator_t fully_connected_op, const void* input, float* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_fully_connected_nc_f32(xnn_operator_t fully_connected_op,
                                            const float* input, float* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_fully_connected_nc_f32_qc4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, uint8_t kernel_zero_point, const float* kernel_scale,
    const uint8_t* kernel, const float* bias, float output_min,
    float output_max, uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_fully_connected_nc_f32_qc4w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_fully_connected_nc_f32_qc4w(
    xnn_operator_t fully_connected_op, const float* input, float* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_fully_connected_nc_f32_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel_scale, const int8_t* kernel,
    const float* bias, float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_fully_connected_nc_f32_qc8w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_fully_connected_nc_f32_qc8w(
    xnn_operator_t fully_connected_op, const float* input, float* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_fully_connected_nc_qd8_f16_qc4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, uint8_t kernel_zero_point, const float* kernel_scale,
    const void* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_fully_connected_nc_qd8_f16_qc4w(
    xnn_operator_t fully_connected_op, const int8_t* input, void* output,
    void* workspace, const xnn_quantization_params* quantization_params) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_fully_connected_nc_qd8_f16_qc4w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_fully_connected_nc_qd8_f16_qb4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, size_t block_size, uint8_t kernel_zero_point,
    const uint16_t* kernel_scale, const void* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_fully_connected_nc_qd8_f16_qb4w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_fully_connected_nc_qd8_f16_qb4w(
    xnn_operator_t fully_connected_op, const int8_t* input, void* output,
    void* workspace, const xnn_quantization_params* quantization_params) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_fully_connected_nc_qd8_f32_qc4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, uint8_t kernel_zero_point, const float* kernel_scale,
    const void* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_fully_connected_nc_qd8_f32_qc4w(
    xnn_operator_t fully_connected_op, const int8_t* input, float* output,
    void* workspace, const xnn_quantization_params* quantization_params) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_fully_connected_nc_qd8_f32_qc4w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_fully_connected_nc_qd8_f32_qb4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, size_t block_size, uint8_t kernel_zero_point,
    const uint16_t* kernel_scale, const void* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_fully_connected_nc_qd8_f32_qb4w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_fully_connected_nc_qd8_f32_qb4w(
    xnn_operator_t fully_connected_op, const int8_t* input, float* output,
    void* workspace, const xnn_quantization_params* quantization_params) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_fully_connected_nc_qd8_f16_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel_scale, const int8_t* kernel,
    const float* bias, float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_fully_connected_nc_qd8_f16_qc8w(
    xnn_operator_t fully_connected_op, const int8_t* input, void* output,
    void* workspace, const xnn_quantization_params* quantization_params) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_fully_connected_nc_qd8_f16_qc8w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_fully_connected_nc_qd8_f32_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel_scale, const int8_t* kernel,
    const float* bias, float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_fully_connected_nc_qd8_f32_qc8w(
    xnn_operator_t fully_connected_op, const int8_t* input, float* output,
    void* workspace, const xnn_quantization_params* quantization_params) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_fully_connected_nc_qd8_f32_qc8w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    size_t* workspace_size, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_fully_connected_nc_qs8(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, int8_t input_zero_point, float input_scale,
    float kernel_scale, const int8_t* kernel, const int32_t* bias,
    int8_t output_zero_point, float output_scale, int8_t output_min,
    int8_t output_max, uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_fully_connected_nc_qs8(xnn_operator_t fully_connected_op,
                                              size_t batch_size,
                                              pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_fully_connected_nc_qs8(xnn_operator_t fully_connected_op,
                                            const int8_t* input,
                                            int8_t* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_fully_connected_nc_qs8_qc4w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, int8_t input_zero_point, float input_scale,
    uint8_t kernel_zero_point, const float* kernel_scale, const void* kernel,
    const int32_t* bias, int8_t output_zero_point, float output_scale,
    int8_t output_min, int8_t output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_fully_connected_nc_qs8_qc4w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_fully_connected_nc_qs8_qc4w(
    xnn_operator_t fully_connected_op, const int8_t* input, int8_t* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_fully_connected_nc_qs8_qc8w(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, int8_t input_zero_point, float input_scale,
    const float* kernel_scale, const int8_t* kernel, const int32_t* bias,
    int8_t output_zero_point, float output_scale, int8_t output_min,
    int8_t output_max, uint32_t flags, xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_fully_connected_nc_qs8_qc8w(
    xnn_operator_t fully_connected_op, size_t batch_size,
    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_fully_connected_nc_qs8_qc8w(
    xnn_operator_t fully_connected_op, const int8_t* input, int8_t* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_fully_connected_nc_qu8(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, uint8_t input_zero_point, float input_scale,
    uint8_t kernel_zero_point, float kernel_scale, const uint8_t* kernel,
    const int32_t* bias, uint8_t output_zero_point, float output_scale,
    uint8_t output_min, uint8_t output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* fully_connected_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_fully_connected_nc_qu8(xnn_operator_t fully_connected_op,
                                              size_t batch_size,
                                              pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_fully_connected_nc_qu8(xnn_operator_t fully_connected_op,
                                            const uint8_t* input,
                                            uint8_t* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_max_pooling2d_nhwc_f16(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t pooling_height, uint32_t pooling_width, uint32_t stride_height,
    uint32_t stride_width, uint32_t dilation_height, uint32_t dilation_width,
    float output_min, float output_max, uint32_t flags,
    xnn_operator_t* max_pooling_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_max_pooling2d_nhwc_f16(
    xnn_operator_t max_pooling_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t channels, size_t input_pixel_stride,
    size_t output_pixel_stride, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_max_pooling2d_nhwc_f16(xnn_operator_t max_pooling_op,
                                            const void* input, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_max_pooling2d_nhwc_f32(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t pooling_height, uint32_t pooling_width, uint32_t stride_height,
    uint32_t stride_width, uint32_t dilation_height, uint32_t dilation_width,
    float output_min, float output_max, uint32_t flags,
    xnn_operator_t* max_pooling_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_max_pooling2d_nhwc_f32(
    xnn_operator_t max_pooling_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t channels, size_t input_pixel_stride,
    size_t output_pixel_stride, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_max_pooling2d_nhwc_f32(xnn_operator_t max_pooling_op,
                                            const float* input, float* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_max_pooling2d_nhwc_s8(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t pooling_height, uint32_t pooling_width, uint32_t stride_height,
    uint32_t stride_width, uint32_t dilation_height, uint32_t dilation_width,
    int8_t output_min, int8_t output_max, uint32_t flags,
    xnn_operator_t* max_pooling_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_max_pooling2d_nhwc_s8(
    xnn_operator_t max_pooling_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t channels, size_t input_pixel_stride,
    size_t output_pixel_stride, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_max_pooling2d_nhwc_s8(xnn_operator_t max_pooling_op,
                                           const int8_t* input,
                                           int8_t* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_max_pooling2d_nhwc_u8(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t pooling_height, uint32_t pooling_width, uint32_t stride_height,
    uint32_t stride_width, uint32_t dilation_height, uint32_t dilation_width,
    uint8_t output_min, uint8_t output_max, uint32_t flags,
    xnn_operator_t* max_pooling_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_max_pooling2d_nhwc_u8(
    xnn_operator_t max_pooling_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t channels, size_t input_pixel_stride,
    size_t output_pixel_stride, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_max_pooling2d_nhwc_u8(xnn_operator_t max_pooling_op,
                                           const uint8_t* input,
                                           uint8_t* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_reduce_nd(
    xnn_reduce_operator reduce_operator_type, xnn_datatype datatype,
    const xnn_quantization_params* input_quantization,
    const xnn_quantization_params* output_quantization, uint32_t flags,
    xnn_operator_t* reduce_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_reduce_nd(xnn_operator_t reduce_op,
                                 size_t num_reduction_axes,
                                 const int64_t* reduction_axes,
                                 size_t num_input_dims,
                                 const size_t* input_shape,
                                 size_t* workspace_size,
                                 pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_reduce_nd(xnn_operator_t reduce_op, void* workspace,
                               const void* input, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_resize_bilinear2d_nchw(xnn_datatype datatype,
                                             size_t output_height,
                                             size_t output_width,
                                             uint32_t flags,
                                             xnn_operator_t* resize_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_resize_bilinear2d_nchw(
    xnn_operator_t resize_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t channels, size_t input_pixel_stride,
    size_t output_pixel_stride, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_resize_bilinear2d_nchw(xnn_operator_t resize_op,
                                            const void* input, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_resize_bilinear2d_nhwc(xnn_datatype datatype,
                                             size_t output_height,
                                             size_t output_width,
                                             uint32_t flags,
                                             xnn_operator_t* resize_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_resize_bilinear2d_nhwc(
    xnn_operator_t resize_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t channels, size_t input_pixel_stride,
    size_t output_pixel_stride, size_t* workspace_size,
    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_resize_bilinear2d_nhwc(xnn_operator_t resize_op,
                                            void* workspace, const void* input,
                                            void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_rope_nthc_f16(uint32_t flags,
                                    xnn_operator_t* rope_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_rope_nthc_f16(xnn_operator_t rope_op, size_t batch_size,
                                     size_t tokens, size_t heads,
                                     size_t channels,
                                     pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_rope_nthc_f16(xnn_operator_t rope_op, const void* input,
                                   const void* weights, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_rope_nthc_f32(uint32_t flags,
                                    xnn_operator_t* rope_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_rope_nthc_f32(xnn_operator_t rope_op, size_t batch_size,
                                     size_t tokens, size_t heads,
                                     size_t channels,
                                     pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_rope_nthc_f32(xnn_operator_t rope_op, const float* input,
                                   const float* weights, float* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_slice_nd_x16(uint32_t flags,
                                   xnn_operator_t* slice_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_slice_nd_x16(xnn_operator_t slice_op, size_t num_dims,
                                    const size_t* input_shape,
                                    const size_t* offsets, const size_t* sizes,
                                    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_slice_nd_x16(xnn_operator_t slice_op, const void* input,
                                  void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_slice_nd_x32(uint32_t flags,
                                   xnn_operator_t* slice_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_slice_nd_x32(xnn_operator_t slice_op, size_t num_dims,
                                    const size_t* input_shape,
                                    const size_t* offsets, const size_t* sizes,
                                    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_slice_nd_x32(xnn_operator_t slice_op, const void* input,
                                  void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_run_slice_nd_x32(size_t num_dims, const size_t* input_shape,
                                const size_t* offsets, const size_t* sizes,
                                const void* input, void* output, uint32_t flags,
                                pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_softmax_nc_f16(uint32_t flags,
                                     xnn_operator_t* softmax_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_softmax_nc_f16(xnn_operator_t softmax_op,
                                      size_t channels, size_t input_stride,
                                      size_t output_stride, size_t batch_size,
                                      pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_softmax_nc_f16(xnn_operator_t softmax_op,
                                    const void* input, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_softmax_nc_f32(uint32_t flags,
                                     xnn_operator_t* softmax_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_softmax_nc_f32(xnn_operator_t softmax_op,
                                      size_t channels, size_t input_stride,
                                      size_t output_stride, size_t batch_size,
                                      pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_softmax_nc_f32(xnn_operator_t softmax_op,
                                    const float* input, float* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_softmax_nc_qu8(float input_scale,
                                     uint8_t output_zero_point,
                                     float output_scale, uint32_t flags,
                                     xnn_operator_t* softmax_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_softmax_nc_qu8(xnn_operator_t softmax_op,
                                      size_t channels, size_t input_stride,
                                      size_t output_stride, size_t batch_size,
                                      pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_softmax_nc_qu8(xnn_operator_t softmax_op,
                                    const uint8_t* input, uint8_t* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_space_to_depth_nhwc_x16(
    uint32_t block_size, uint32_t flags,
    xnn_operator_t* space_to_depth_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_space_to_depth_nhwc_x16(
    xnn_operator_t space_to_depth_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t input_channels, size_t* output_height_out,
    size_t* output_width_out, size_t* output_channels_out,
    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_space_to_depth_nhwc_x16(xnn_operator_t space_to_depth_op,
                                             const void* input, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_space_to_depth_nhwc_x32(
    uint32_t block_size, uint32_t flags,
    xnn_operator_t* space_to_depth_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_space_to_depth_nhwc_x32(
    xnn_operator_t space_to_depth_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t input_channels, size_t* output_height_out,
    size_t* output_width_out, size_t* output_channels_out,
    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_space_to_depth_nhwc_x32(xnn_operator_t space_to_depth_op,
                                             const void* input, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_transpose_nd_x8(uint32_t flags,
                                      xnn_operator_t* transpose_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_transpose_nd_x8(xnn_operator_t transpose_op,
                                       size_t num_dims,
                                       const size_t* input_shape,
                                       const size_t* output_perm,
                                       pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_transpose_nd_x8(xnn_operator_t transpose_op,
                                     const void* input, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_run_transpose_nd_x8(const void* input, void* output,
                                   size_t num_dims, const size_t* input_shape,
                                   const size_t* output_perm, uint32_t flags,
                                   pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_transpose_nd_x16(uint32_t flags,
                                       xnn_operator_t* transpose_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_transpose_nd_x16(xnn_operator_t transpose_op,
                                        size_t num_dims,
                                        const size_t* input_shape,
                                        const size_t* output_perm,
                                        pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_transpose_nd_x16(xnn_operator_t transpose_op,
                                      const void* input, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_run_transpose_nd_x16(const void* input, void* output,
                                    size_t num_dims, const size_t* input_shape,
                                    const size_t* output_perm, uint32_t flags,
                                    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_transpose_nd_x32(uint32_t flags,
                                       xnn_operator_t* transpose_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_transpose_nd_x32(xnn_operator_t transpose_op,
                                        size_t num_dims,
                                        const size_t* input_shape,
                                        const size_t* output_perm,
                                        pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_transpose_nd_x32(xnn_operator_t transpose_op,
                                      const void* input, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_run_transpose_nd_x32(const void* input, void* output,
                                    size_t num_dims, const size_t* input_shape,
                                    const size_t* output_perm, uint32_t flags,
                                    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_transpose_nd_x64(uint32_t flags,
                                       xnn_operator_t* transpose_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_transpose_nd_x64(xnn_operator_t transpose_op,
                                        size_t num_dims,
                                        const size_t* input_shape,
                                        const size_t* output_perm,
                                        pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_transpose_nd_x64(xnn_operator_t transpose_op,
                                      const void* input, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_run_transpose_nd_x64(const void* input, void* output,
                                    size_t num_dims, const size_t* input_shape,
                                    const size_t* output_perm, uint32_t flags,
                                    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_unpooling2d_nhwc_x32(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t pooling_height, uint32_t pooling_width, uint32_t flags,
    xnn_operator_t* unpooling_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_unpooling2d_nhwc_x32(
    xnn_operator_t unpooling_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t channels, size_t input_pixel_stride,
    size_t output_pixel_stride, size_t* output_height_out,
    size_t* output_width_out, pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_unpooling2d_nhwc_x32(xnn_operator_t unpooling_op,
                                          const void* input,
                                          const uint32_t* index, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_slice_nd_x8(uint32_t flags,
                                  xnn_operator_t* slice_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_slice_nd_x8(xnn_operator_t slice_op, size_t num_dims,
                                   const size_t* input_shape,
                                   const size_t* offsets, const size_t* sizes,
                                   pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_slice_nd_x8(xnn_operator_t slice_op, const void* input,
                                 void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_create_space_to_depth_nhwc_x8(
    uint32_t block_size, uint32_t flags,
    xnn_operator_t* space_to_depth_op_out) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_reshape_space_to_depth_nhwc_x8(
    xnn_operator_t space_to_depth_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t input_channels, size_t* output_height_out,
    size_t* output_width_out, size_t* output_channels_out,
    pthreadpool_t threadpool) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_setup_space_to_depth_nhwc_x8(xnn_operator_t space_to_depth_op,
                                            const void* input, void* output) {
  YNN_LOG_FATAL() << "operator API is not supported";
  return xnn_status_deprecated;
}

}  // extern "C"
