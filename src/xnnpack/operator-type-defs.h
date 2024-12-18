// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_ENUM_ITEM_0
#define XNN_ENUM_ITEM_0(enum_name, enum_string) XNN_ENUM_ITEM(enum_name, enum_string)
#define XNN_DEFINED_ENUM_ITEM_0
#endif

XNN_ENUM_ITEM_0(xnn_operator_type_invalid, "Invalid")
XNN_ENUM_ITEM(xnn_operator_type_argmax_pooling_nhwc_f32, "ArgMax Pooling (NHWC, F32)")
XNN_ENUM_ITEM(xnn_operator_type_average_pooling_nhwc_f16, "Average Pooling (NHWC, F16)")
XNN_ENUM_ITEM(xnn_operator_type_average_pooling_nhwc_f32, "Average Pooling (NHWC, F32)")
XNN_ENUM_ITEM(xnn_operator_type_average_pooling_nhwc_qu8, "Average Pooling (NHWC, QU8)")
XNN_ENUM_ITEM(xnn_operator_type_batch_matrix_multiply_nc_f16, "Batch Matrix Multiply (NC, F16)")
XNN_ENUM_ITEM(xnn_operator_type_batch_matrix_multiply_nc_f32, "Batch Matrix Multiply (NC, F32)")
XNN_ENUM_ITEM(xnn_operator_type_batch_matrix_multiply_nc_qd8_f32_qc8w, "Batch Matrix Multiply (NC, QD8, F32, QC8W)")
XNN_ENUM_ITEM(xnn_operator_type_batch_matrix_multiply_nc_qdu8_f32_qc8w, "Batch Matrix Multiply (NC, QDU8, F32, QC8W)")
XNN_ENUM_ITEM(xnn_operator_type_batch_matrix_multiply_nc_qp8_f32_qc8w,
              "Batch Matrix Multiply (NC, QP8, F32, QC8W)")
XNN_ENUM_ITEM(xnn_operator_type_binary_elementwise, "Binary Elementwise (ND)")
XNN_ENUM_ITEM(xnn_operator_type_constant_pad_nd_x8, "Constant Pad (ND, X8)")
XNN_ENUM_ITEM(xnn_operator_type_constant_pad_nd_x16, "Constant Pad (ND, X16)")
XNN_ENUM_ITEM(xnn_operator_type_constant_pad_nd_x32, "Constant Pad (ND, X32)")
XNN_ENUM_ITEM(xnn_operator_type_convert_nc_f16_qd8, "Convert (NC, F16, QD8)")
XNN_ENUM_ITEM(xnn_operator_type_convert_nc_f16_qdu8, "Convert (NC, F16, QDU8)")
XNN_ENUM_ITEM(xnn_operator_type_convert_nc_f32_qd8, "Convert (NC, F32, QD8)")
XNN_ENUM_ITEM(xnn_operator_type_convert_nc_f32_qdu8, "Convert (NC, F32, QDU8)")
XNN_ENUM_ITEM(xnn_operator_type_convert_nc_f32_qp8, "Convert (NC, F32, QP8)")
XNN_ENUM_ITEM(xnn_operator_type_convolution_nchw_f16, "Convolution (NCHW, F16)")
XNN_ENUM_ITEM(xnn_operator_type_convolution_nchw_f32, "Convolution (NCHW, F32)")
XNN_ENUM_ITEM(xnn_operator_type_convolution_nhwc_f16, "Convolution (NHWC, F16)")
XNN_ENUM_ITEM(xnn_operator_type_convolution_nhwc_f32, "Convolution (NHWC, F32)")
XNN_ENUM_ITEM(xnn_operator_type_convolution_nhwc_qdu8_f16_qc8w,
              "Convolution (NHWC, QD8, F16, QC8W)")
XNN_ENUM_ITEM(xnn_operator_type_convolution_nhwc_qd8_f16_qc8w, "Convolution (NHWC, QD8, F16, QC8W)")
XNN_ENUM_ITEM(xnn_operator_type_convolution_nhwc_qdu8_f32_qc8w,
              "Convolution (NHWC, QDU8, F32, QC8W)")
XNN_ENUM_ITEM(xnn_operator_type_convolution_nhwc_qd8_f32_qc8w, "Convolution (NHWC, QD8, F32, QC8W)")
XNN_ENUM_ITEM(xnn_operator_type_convolution_nhwc_qc8, "Convolution (NHWC, QC8)")
XNN_ENUM_ITEM(xnn_operator_type_convolution_nhwc_qs8, "Convolution (NHWC, QS8)")
XNN_ENUM_ITEM(xnn_operator_type_convolution_nhwc_qu8, "Convolution (NHWC, QU8)")
XNN_ENUM_ITEM(xnn_operator_type_copy_nc_x8, "Copy (NC, X8)")
XNN_ENUM_ITEM(xnn_operator_type_copy_nc_x16, "Copy (NC, X16)")
XNN_ENUM_ITEM(xnn_operator_type_copy_nc_x32, "Copy (NC, X32)")
XNN_ENUM_ITEM(xnn_operator_type_deconvolution_nhwc_f16, "Deconvolution (NHWC, F16)")
XNN_ENUM_ITEM(xnn_operator_type_deconvolution_nhwc_f32, "Deconvolution (NHWC, F32)")
XNN_ENUM_ITEM(xnn_operator_type_deconvolution_nhwc_qd8_f32_qc8w, "Deconvolution (NHWC, QD8, F32, QC8W)")
XNN_ENUM_ITEM(xnn_operator_type_deconvolution_nhwc_qdu8_f32_qc8w,
              "Deconvolution (NHWC, QDU8, F32, QC8W)")
XNN_ENUM_ITEM(xnn_operator_type_deconvolution_nhwc_qs8, "Deconvolution (NHWC, QS8)")
XNN_ENUM_ITEM(xnn_operator_type_deconvolution_nhwc_qs8_qc8w, "Deconvolution (NC, QS8, QC8W)")
XNN_ENUM_ITEM(xnn_operator_type_deconvolution_nhwc_qu8, "Deconvolution (NHWC, QU8)")
XNN_ENUM_ITEM(xnn_operator_type_depth_to_space_nchw2nhwc_x16, "Depth To Space (NCHW2NHWC, X16)")
XNN_ENUM_ITEM(xnn_operator_type_depth_to_space_nchw2nhwc_x32, "Depth To Space (NCHW2NHWC, X32)")
XNN_ENUM_ITEM(xnn_operator_type_depth_to_space_nhwc_x8, "Depth To Space (NHWC, X8)")
XNN_ENUM_ITEM(xnn_operator_type_depth_to_space_nhwc_x16, "Depth To Space (NHWC, X16)")
XNN_ENUM_ITEM(xnn_operator_type_depth_to_space_nhwc_x32, "Depth To Space (NHWC, X32)")
XNN_ENUM_ITEM(xnn_operator_type_dynamic_fully_connected_nc_f16, "Dynamic Fully Connected (NC, F16)")
XNN_ENUM_ITEM(xnn_operator_type_dynamic_fully_connected_nc_f32, "Dynamic Fully Connected (NC, F32)")
XNN_ENUM_ITEM(xnn_operator_type_fully_connected_nc_f16, "Fully Connected (NC, F16)")
XNN_ENUM_ITEM(xnn_operator_type_fully_connected_nc_f32, "Fully Connected (NC, F32)")
XNN_ENUM_ITEM(xnn_operator_type_fully_connected_nc_f32_qc4w, "Fully Connected (NC, F32, QC4W)")
XNN_ENUM_ITEM(xnn_operator_type_fully_connected_nc_f32_qc8w, "Fully Connected (NC, F32, QC8W)")
XNN_ENUM_ITEM(xnn_operator_type_fully_connected_nc_pf32,
              "Fully Connected (NC, PF32)")
XNN_ENUM_ITEM(xnn_operator_type_fully_connected_nc_qd8_f16_qb4w, "Fully Connected (NC, QD8, F16, QB4W)")
XNN_ENUM_ITEM(xnn_operator_type_fully_connected_nc_qd8_f16_qc4w, "Fully Connected (NC, QD8, F16, QC4W)")
XNN_ENUM_ITEM(xnn_operator_type_fully_connected_nc_qdu8_f16_qc4w,
              "Fully Connected (NC, QDU8, F16, QC4W)")
XNN_ENUM_ITEM(xnn_operator_type_fully_connected_nc_qd8_f16_qc8w, "Fully Connected (NC, QD8, F16, QC8W)")
XNN_ENUM_ITEM(xnn_operator_type_fully_connected_nc_qdu8_f16_qc8w,
              "Fully Connected (NC, QDU8, F16, QC8W)")
XNN_ENUM_ITEM(xnn_operator_type_fully_connected_nc_qd8_f32_qb4w, "Fully Connected (NC, QD8, F32, QB4W)")
XNN_ENUM_ITEM(xnn_operator_type_fully_connected_nc_qdu8_f32_qb4w,
              "Fully Connected (NC, QDU8, F32, QB4W)")
XNN_ENUM_ITEM(xnn_operator_type_fully_connected_nc_qd8_f32_qc4w, "Fully Connected (NC, QD8, F32, QC4W)")
XNN_ENUM_ITEM(xnn_operator_type_fully_connected_nc_qdu8_f32_qc4w,
              "Fully Connected (NC, QDU8, F32, QC4W)")
XNN_ENUM_ITEM(xnn_operator_type_fully_connected_nc_qd8_f32_qc8w, "Fully Connected (NC, QD8, F32, QC8W)")
XNN_ENUM_ITEM(xnn_operator_type_fully_connected_nc_qdu8_f32_qc8w,
              "Fully Connected (NC, QDU8, F32, QC8W)")
XNN_ENUM_ITEM(xnn_operator_type_fully_connected_nc_qp8_f32_qc4w, "Fully Connected (NC, QP8, F32, QC4W)")
XNN_ENUM_ITEM(xnn_operator_type_fully_connected_nc_qp8_f32_qc8w,
              "Fully Connected (NC, QP8, F32, QC8W)")
XNN_ENUM_ITEM(xnn_operator_type_fully_connected_nc_qp8_f32_qb4w, "Fully Connected (NC, QP8, F32, QB4W)")
XNN_ENUM_ITEM(xnn_operator_type_fully_connected_nc_qs8, "Fully Connected (NC, QS8)")
XNN_ENUM_ITEM(xnn_operator_type_fully_connected_nc_qs8_qc8w, "Fully Connected (NC, QS8, QC8W)")
XNN_ENUM_ITEM(xnn_operator_type_fully_connected_nc_qu8, "Fully Connected (NC, QU8)")
XNN_ENUM_ITEM(xnn_operator_type_max_pooling_nhwc_f16, "Max Pooling (NHWC, F16)")
XNN_ENUM_ITEM(xnn_operator_type_max_pooling_nhwc_f32, "Max Pooling (NHWC, F32)")
XNN_ENUM_ITEM(xnn_operator_type_max_pooling_nhwc_s8, "Max Pooling (NHWC, S8)")
XNN_ENUM_ITEM(xnn_operator_type_max_pooling_nhwc_u8, "Max Pooling (NHWC, U8)")
XNN_ENUM_ITEM(xnn_operator_type_mean_nd, "Mean (ND)")
XNN_ENUM_ITEM(xnn_operator_type_pack_lh_x32, "Pack LH (X32)")
XNN_ENUM_ITEM(xnn_operator_type_reciprocal_square_root, "Reciprocal Square Root (NC)")
XNN_ENUM_ITEM(xnn_operator_type_resize_bilinear_nchw_f16, "Resize Bilinear (NCHW, F16)")
XNN_ENUM_ITEM(xnn_operator_type_resize_bilinear_nchw_f32, "Resize Bilinear (NCHW, F32)")
XNN_ENUM_ITEM(xnn_operator_type_resize_bilinear_nhwc_f16, "Resize Bilinear (NHWC, F16)")
XNN_ENUM_ITEM(xnn_operator_type_resize_bilinear_nhwc_f32, "Resize Bilinear (NHWC, F32)")
XNN_ENUM_ITEM(xnn_operator_type_resize_bilinear_nhwc_s8, "Resize Bilinear (NHWC, S8)")
XNN_ENUM_ITEM(xnn_operator_type_resize_bilinear_nhwc_u8, "Resize Bilinear (NHWC, U8)")
XNN_ENUM_ITEM(xnn_operator_type_rope_nthc_f16, "RoPE (NTHC, F16)")
XNN_ENUM_ITEM(xnn_operator_type_rope_nthc_f32, "RoPE (NTHC, F32)")
XNN_ENUM_ITEM(xnn_operator_type_scaled_dot_product_attention_nhtc_f16, "Scaled Dot-Product Attention (NHTC, F16)")
XNN_ENUM_ITEM(xnn_operator_type_scaled_dot_product_attention_nhtc_f32, "Scaled Dot-Product Attention (NHTC, F32)")
XNN_ENUM_ITEM(xnn_operator_type_slice_nd_x8, "Slice (ND, X8)")
XNN_ENUM_ITEM(xnn_operator_type_slice_nd_x16, "Slice (ND, X16)")
XNN_ENUM_ITEM(xnn_operator_type_slice_nd_x32, "Slice (ND, X32)")
XNN_ENUM_ITEM(xnn_operator_type_softmax_nc_f16, "Softmax (NC, F16)")
XNN_ENUM_ITEM(xnn_operator_type_softmax_nc_f32, "Softmax (NC, F32)")
XNN_ENUM_ITEM(xnn_operator_type_softmax_nc_qu8, "Softmax (NC, QU8)")
XNN_ENUM_ITEM(xnn_operator_type_space_to_depth_nhwc_x8, "Space To Depth (NHWC, X8)")
XNN_ENUM_ITEM(xnn_operator_type_space_to_depth_nhwc_x16, "Space To Depth (NHWC, X16)")
XNN_ENUM_ITEM(xnn_operator_type_space_to_depth_nhwc_x32, "Space To Depth (NHWC, X32)")
XNN_ENUM_ITEM(xnn_operator_type_sum_nd, "Sum (ND)")
XNN_ENUM_ITEM(xnn_operator_type_transpose_nd_x8, "Transpose (ND, X8)")
XNN_ENUM_ITEM(xnn_operator_type_transpose_nd_x16, "Transpose (ND, X16)")
XNN_ENUM_ITEM(xnn_operator_type_transpose_nd_x32, "Transpose (ND, X32)")
XNN_ENUM_ITEM(xnn_operator_type_transpose_nd_x64, "Transpose (ND, X64)")
XNN_ENUM_ITEM(xnn_operator_type_unary_elementwise, "Unary Elementwise (NC)")
XNN_ENUM_ITEM(xnn_operator_type_unpooling_nhwc_x32, "Unpooling (NHWC, X32)")


#ifdef XNN_DEFINED_ENUM_ITEM_0
#undef XNN_DEFINED_ENUM_ITEM_0
#undef XNN_ENUM_ITEM_0
#endif
