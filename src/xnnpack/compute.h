// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef THIRD_PARTY_XNNPACK_SRC_XNNPACK_COMPUTE_H_
#define THIRD_PARTY_XNNPACK_SRC_XNNPACK_COMPUTE_H_

#include <stddef.h>
#include <stdint.h>

#include "include/xnnpack.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/operator-type.h"
#include <pthreadpool.h>

enum xnn_parallelization_type {
  xnn_parallelization_type_invalid = 0,
  xnn_parallelization_type_1d,
  xnn_parallelization_type_1d_with_thread,
  xnn_parallelization_type_1d_tile_1d,
  xnn_parallelization_type_1d_tile_1d_dynamic,
  xnn_parallelization_type_1d_tile_1d_dynamic_with_thread,
  xnn_parallelization_type_2d,
  xnn_parallelization_type_2d_with_thread,
  xnn_parallelization_type_2d_tile_1d,
  xnn_parallelization_type_2d_tile_2d,
  xnn_parallelization_type_2d_tile_1d_dynamic,
  xnn_parallelization_type_2d_tile_1d_dynamic_with_thread,
  xnn_parallelization_type_2d_tile_2d_dynamic,
  xnn_parallelization_type_2d_tile_2d_dynamic_with_thread,
  xnn_parallelization_type_3d,
  xnn_parallelization_type_3d_tile_1d,
  xnn_parallelization_type_3d_tile_1d_with_thread,
  xnn_parallelization_type_3d_tile_1d_dynamic_with_thread,
  xnn_parallelization_type_3d_tile_2d,
  xnn_parallelization_type_3d_tile_2d_dynamic,
  xnn_parallelization_type_3d_tile_2d_dynamic_with_thread,
  xnn_parallelization_type_4d,
  xnn_parallelization_type_4d_tile_2d,
  xnn_parallelization_type_4d_tile_2d_dynamic,
  xnn_parallelization_type_5d,
  xnn_parallelization_type_5d_tile_2d,
  xnn_parallelization_type_6d_tile_2d,
#if XNN_MAX_UARCH_TYPES > 1
  xnn_parallelization_type_1d_tile_1d_dynamic_with_uarch_with_thread,
  xnn_parallelization_type_2d_tile_1d_with_uarch,
  xnn_parallelization_type_2d_tile_1d_dynamic_with_uarch_with_thread,
  xnn_parallelization_type_2d_tile_2d_with_uarch,
  xnn_parallelization_type_2d_tile_2d_dynamic_with_uarch,
  xnn_parallelization_type_3d_tile_1d_with_uarch,
  xnn_parallelization_type_3d_tile_1d_with_uarch_with_thread,
  xnn_parallelization_type_3d_tile_2d_with_uarch,
  xnn_parallelization_type_3d_tile_2d_dynamic_with_uarch,
  xnn_parallelization_type_4d_tile_2d_with_uarch,
  xnn_parallelization_type_4d_tile_2d_dynamic_with_uarch,
#endif  // XNN_MAX_UARCH_TYPES > 1
};

struct compute_parameters {
  enum xnn_parallelization_type type;
  union {
    pthreadpool_task_1d_t task_1d;
    pthreadpool_task_1d_with_thread_t task_1d_with_thread;
    pthreadpool_task_1d_tile_1d_t task_1d_tile_1d;
    pthreadpool_task_1d_tile_1d_t task_1d_tile_1d_dynamic;
    pthreadpool_task_1d_tile_1d_dynamic_with_id_t
        task_1d_tile_1d_dynamic_with_id;
    pthreadpool_task_2d_t task_2d;
    pthreadpool_task_2d_with_thread_t task_2d_with_thread;
    pthreadpool_task_2d_tile_1d_t task_2d_tile_1d;
    pthreadpool_task_2d_tile_2d_t task_2d_tile_2d;
    pthreadpool_task_2d_tile_1d_dynamic_t task_2d_tile_1d_dynamic;
    pthreadpool_task_2d_tile_1d_dynamic_with_id_t
        task_2d_tile_1d_dynamic_with_id;
    pthreadpool_task_2d_tile_2d_dynamic_t task_2d_tile_2d_dynamic;
    pthreadpool_task_2d_tile_2d_dynamic_with_id_t
        task_2d_tile_2d_dynamic_with_id;
    pthreadpool_task_3d_t task_3d;
    pthreadpool_task_3d_tile_1d_t task_3d_tile_1d;
    pthreadpool_task_3d_tile_1d_with_thread_t task_3d_tile_1d_with_thread;
    pthreadpool_task_3d_tile_1d_dynamic_with_id_t
        task_3d_tile_1d_dynamic_with_id;
    pthreadpool_task_3d_tile_2d_t task_3d_tile_2d;
    pthreadpool_task_3d_tile_2d_dynamic_t task_3d_tile_2d_dynamic;
    pthreadpool_task_3d_tile_2d_dynamic_with_id_t
        task_3d_tile_2d_dynamic_with_id;
    pthreadpool_task_4d_t task_4d;
    pthreadpool_task_4d_tile_2d_t task_4d_tile_2d;
    pthreadpool_task_4d_tile_2d_dynamic_t task_4d_tile_2d_dynamic;
    pthreadpool_task_5d_t task_5d;
    pthreadpool_task_5d_tile_2d_t task_5d_tile_2d;
    pthreadpool_task_6d_tile_2d_t task_6d_tile_2d;
#if XNN_MAX_UARCH_TYPES > 1
    pthreadpool_task_1d_tile_1d_dynamic_with_id_with_thread_t
        task_1d_tile_1d_dynamic_with_id_with_thread;
    pthreadpool_task_2d_tile_1d_with_id_t task_2d_tile_1d_with_id;
    pthreadpool_task_2d_tile_1d_dynamic_with_id_with_thread_t
        task_2d_tile_1d_dynamic_with_id_with_thread;
    pthreadpool_task_2d_tile_2d_with_id_t task_2d_tile_2d_with_id;
    pthreadpool_task_3d_tile_1d_with_id_t task_3d_tile_1d_with_id;
    pthreadpool_task_3d_tile_1d_with_id_with_thread_t
        task_3d_tile_1d_with_id_with_thread;
    pthreadpool_task_3d_tile_2d_with_id_t task_3d_tile_2d_with_id;
    pthreadpool_task_4d_tile_2d_with_id_t task_4d_tile_2d_with_id;
    pthreadpool_task_4d_tile_2d_dynamic_with_id_t
        task_4d_tile_2d_dynamic_with_id;
#endif  // XNN_MAX_UARCH_TYPES > 1
  };
  // Offset of the invocation context w.r.t. xnn_operator.context. Typically 0,
  // but can be non-zero when an operator does multiple invocations.
  size_t context_offset;
  size_t range[6];
  size_t tile[2];
};

struct transpose_context {
  const void* x;
  void* y;
  union {
    xnn_transposec_ukernel_fn const_size_ukernel;
    xnn_transposev_ukernel_fn variable_size_ukernel;
  };
  size_t input_stride[XNN_MAX_TENSOR_DIMS];
  size_t output_stride[XNN_MAX_TENSOR_DIMS];
};

XNN_PRIVATE void xnn_compute_transposec_2d(
    const struct transpose_context* context, size_t i, size_t j, size_t tile_i,
    size_t tile_j);

XNN_PRIVATE void xnn_compute_transposec_3d(
    const struct transpose_context* context, size_t i, size_t j, size_t k,
    size_t tile_j, size_t tile_k);

XNN_PRIVATE void xnn_compute_transposec_4d(
    const struct transpose_context* context, size_t i, size_t j, size_t k,
    size_t l, size_t tile_k, size_t tile_l);

XNN_PRIVATE void xnn_compute_transposec_5d(
    const struct transpose_context* context, size_t i, size_t j, size_t k,
    size_t l, size_t m, size_t tile_l, size_t tile_m);

XNN_PRIVATE void xnn_compute_transposec_6d(
    const struct transpose_context* context, size_t i, size_t j, size_t k,
    size_t l, size_t m, size_t n, size_t tile_m, size_t tile_n);

XNN_PRIVATE void xnn_compute_transposev_2d(
    const struct transpose_context* context, size_t i, size_t j, size_t tile_i,
    size_t tile_j);

XNN_PRIVATE void xnn_compute_transposev_3d(
    const struct transpose_context* context, size_t i, size_t j, size_t k,
    size_t tile_j, size_t tile_k);

XNN_PRIVATE void xnn_compute_transposev_4d(
    const struct transpose_context* context, size_t i, size_t j, size_t k,
    size_t l, size_t tile_k, size_t tile_l);

XNN_PRIVATE void xnn_compute_transposev_5d(
    const struct transpose_context* context, size_t i, size_t j, size_t k,
    size_t l, size_t m, size_t tile_l, size_t tile_m);

XNN_PRIVATE void xnn_compute_transposev_6d(
    const struct transpose_context* context, size_t i, size_t j, size_t k,
    size_t l, size_t m, size_t n, size_t tile_m, size_t tile_n);

// Context for Packing Weights (packw) for GEMM microkernels in
// Group-OutputChannels-InputChannels layout. Kernel has shape GxNxK, bias has
// shape GxN.
struct packw_gemm_goi_context {
  // Number of input channels.
  size_t kc;
  // Number of output channels the GEMM is optimized for.
  size_t nr;
  size_t kr;
  size_t sr;
  // Pointer to kernel.
  const void* kernel;
  // Stride, in bytes, between each N of the kernel.
  size_t k_stride;
  // Pointer to bias.
  const void* bias;
  // Stride, in bytes, between each bias.
  size_t b_stride;
  // Output pointer to write packed kernel and bias.
  void* packed_weights;
  // Stride, in bytes, between each packed kernel and bias.
  size_t w_stride;

  // Strides used for batched packw.
  // Stride, in bytes, between each group of kernel
  size_t gk_stride;
  // Stride, in bytes, between each group of bias.
  size_t gb_stride;
  // Stride, in bytes, between each group of packed weights.
  size_t gc_stride;

  // Packing params passed to the packing microkernel.
  const void* params;

  // Microkernel to preform packing.
  xnn_packw_gemm_goi_ukernel_fn packw_gemm_goi;

  // Optional wrapped packing function.
  xnn_pack_weights_and_biases_fn pack_weights_and_biases;

  // A pointer to the underlying `xnn_gemm_config`.
  const struct xnn_gemm_config* gemm_config;
};

XNN_PRIVATE void xnn_compute_packw_gemm_goi(
    const struct packw_gemm_goi_context* context, size_t n_block_start,
    size_t n_block_size);
XNN_PRIVATE void xnn_compute_batched_packw_gemm_goi(
    const struct packw_gemm_goi_context* context, size_t batch_index,
    size_t n_block_start, size_t n_block_size);

// Context for Packing Weights (packw) for GEMM microkernels in
// Groups-InputChannels-OutputChannels layout. Kernel has shape GxKxN, bias has
// shape GxN.
struct packw_gemm_gio_context {
  // Number of input channels.
  size_t kc;
  // Number of output channels the GEMM is optimized for.
  size_t nr;
  size_t kr;
  size_t sr;
  // Pointer to kernel.
  const void* kernel;
  // Pointer to bias.
  const void* bias;
  // Stride, in bytes, between each bias.
  size_t b_stride;
  // Output pointer to write packed kernel and bias.
  void* packed_weights;
  // Stride, in bytes, between each packed kernel and bias.
  size_t w_stride;
  // Stride, in number of elements, between each k of the kernel.
  size_t k_stride_elements;
  // Stride, in bytes, between each n of the kernel.
  size_t n_stride;

  // Strides used for batched packw.
  // Stride, in bytes, between each group of kernel
  size_t gk_stride;
  // Stride, in bytes, between each group of bias.
  size_t gb_stride;
  // Stride, in bytes, between each group of of packed weights.
  size_t gc_stride;

  // Microkernel to preform packing.
  xnn_packw_gemm_gio_ukernel_fn packw_gemm_gio;

  // Optional wrapped packing function.
  xnn_pack_weights_and_biases_fn pack_weights_and_biases;

  // A pointer to the underlying `xnn_gemm_config`.
  const struct xnn_gemm_config* gemm_config;
};

XNN_PRIVATE void xnn_compute_packw_gemm_gio(
    const struct packw_gemm_gio_context* context, size_t n_block_start,
    size_t n_block_size);
XNN_PRIVATE void xnn_compute_batched_packw_gemm_gio(
    const struct packw_gemm_gio_context* context, size_t batch_index,
    size_t n_block_start, size_t n_block_size);

// Context for Dense Matrix Multiplication.
// C [GxMxN] := A [GxMxK] * B[GxKxN] + bias [GxN]
// Where B and bias have been packed into packed_w.
struct gemm_context {
  // K dimension of matrix A, scaled by size of an element in A.
  // Corresponds to the number of input channels.
  size_t k_scaled;
  // Input matrix A.
  const void* a;
  // Stride, in bytes, between each row (M) of A.
  size_t a_stride;
  // Stride, in bytes, between each group (G) of A.
  size_t ga_stride;
  // Pointer to weights (kernel and bias) that have been packed.
  const void* packed_w;
  // Stride, in bytes, between output channel (N) of weights.
  size_t w_stride;
  // Stride, in bytes, between each group (G) of weights.
  size_t gw_stride;
  // Output matrix C.
  void* c;
  // Stride, in bytes, between each row (M) of C.
  size_t cm_stride;
  // Stride, in bytes, between columns (N) of C written.
  size_t cn_stride;
  // Stride, in bytes, between each group (G) of C.
  size_t gc_stride;
  // Pointer to additional workspace, if required.
  void* workspace;
  // Offset of the per-thread chunks from the start of the workspace pointer.
  size_t workspace_offset;
  // Size, in bytes, of each element of C.
  uint32_t log2_csize;
  // Number of batch dimensions in A, B, and C.
  uint32_t num_batch_dims;
  // Batch dimensions of the input A.
  size_t batch_dims_a[XNN_MAX_TENSOR_DIMS];
  // Batch dimensions of the input B.
  size_t batch_dims_b[XNN_MAX_TENSOR_DIMS];
  // Strides of each batch dimension of the output C.
  size_t batch_strides_c[XNN_MAX_TENSOR_DIMS];
  // The `mr` size of the current GEMM microkernel.
  size_t mr;
  // The `kr` size of the current GEMM microkernel.
  size_t kr;
  // The `sr` size of the current GEMM microkernel.
  size_t sr;
  // The `mr_packed` size of the current GEMM microkernel.
  size_t mr_packed;
  // The inner dimension of the matrix product.
  size_t kc;
  // The number of columns.
  size_t nc;
  // GEMM microkernels.
  union {
    struct xnn_hmp_gemm_ukernel ukernel;
    struct xnn_hmp_dqgemm_ukernel dq_ukernel;
    struct xnn_hmp_qp8gemm_ukernel qp8_ukernel;
  };
  // Parameters for dynamically quantized inputs.
  const struct xnn_qd8_quantization_params* quantization_params;
  // Stride between each group of quantization params.
  size_t gq_stride;
  // Parameters for fused GEMM.
  void* fused_params;
  // Parameters for fused activations.
  union {
    union xnn_qs8_conv_minmax_params qs8;
    union xnn_qu8_conv_minmax_params qu8;
    struct xnn_f16_scaleminmax_params f16;
    struct xnn_f32_minmax_params f32;
  } params;
  const struct xnn_pack_lh_config* packed_lh_config;
  // Whether to use the `dq_kernel` or not.
  bool dynamic_quantization;
};

XNN_PRIVATE void xnn_compute_grouped_gemm(const struct gemm_context* context,
                                          size_t group_index,
                                          size_t nr_block_start,
                                          size_t mr_block_start,
                                          size_t nr_block_size,
                                          size_t mr_block_size);

XNN_PRIVATE void xnn_compute_grouped_qp8gemm(const struct gemm_context* context,
                                             size_t group_index,
                                             size_t nr_block_start,
                                             size_t mr_block_start,
                                             size_t nr_block_size,
                                             size_t mr_block_size);

XNN_PRIVATE void xnn_compute_dqgemm(const struct gemm_context* context,
                                    size_t nr_block_start,
                                    size_t mr_block_start, size_t nr_block_size,
                                    size_t mr_block_size);

XNN_PRIVATE void xnn_compute_gemm(const struct gemm_context* context,
                                  size_t nr_block_start, size_t mr_block_start,
                                  size_t nr_block_size, size_t mr_block_size);

XNN_PRIVATE void xnn_compute_qp8gemm(const struct gemm_context* context,
                                     size_t nr_block_start,
                                     size_t mr_block_start,
                                     size_t nr_block_size,
                                     size_t mr_block_size);

XNN_PRIVATE void xnn_compute_inline_packed_qp8gemm(
    const struct gemm_context* context, uint32_t thread_id,
    size_t mr_block_start, size_t mr_block_size);

#if XNN_MAX_UARCH_TYPES > 1
XNN_PRIVATE void xnn_compute_hmp_grouped_gemm(
    const struct gemm_context* context, uint32_t uarch_index,
    size_t group_index, size_t nr_block_start, size_t mr_block_start,
    size_t nr_block_size, size_t mr_block_size);

XNN_PRIVATE void xnn_compute_hmp_grouped_qp8gemm(
    const struct gemm_context* context, uint32_t uarch_index,
    size_t group_index, size_t mr_block_start, size_t nr_block_start,
    size_t mr_block_size, size_t nr_block_size);

XNN_PRIVATE void xnn_compute_hmp_gemm(const struct gemm_context* context,
                                      uint32_t uarch_index,
                                      size_t nr_block_start,
                                      size_t mr_block_start,
                                      size_t nr_block_size,
                                      size_t mr_block_size);

XNN_PRIVATE void xnn_compute_qp8gemm(const struct gemm_context* context,
                                     size_t nr_block_start,
                                     size_t mr_block_start,
                                     size_t nr_block_size,
                                     size_t mr_block_size);

XNN_PRIVATE void xnn_compute_hmp_grouped_gemm(
    const struct gemm_context* context, uint32_t uarch_index,
    size_t group_index, size_t nr_block_start, size_t mr_block_start,
    size_t nr_block_size, size_t mr_block_size);

XNN_PRIVATE void xnn_compute_hmp_dqgemm(const struct gemm_context* context,
                                        uint32_t uarch_index,
                                        size_t nr_block_start,
                                        size_t mr_block_start,
                                        size_t nr_block_size,
                                        size_t mr_block_size);

XNN_PRIVATE void xnn_compute_hmp_dqgemm(const struct gemm_context* context,
                                        uint32_t uarch_index,
                                        size_t nr_block_start,
                                        size_t mr_block_start,
                                        size_t nr_block_size,
                                        size_t mr_block_size);

XNN_PRIVATE void xnn_compute_hmp_qp8gemm(const struct gemm_context* context,
                                         uint32_t uarch_index,
                                         size_t nr_block_start,
                                         size_t mr_block_start,
                                         size_t nr_block_size,
                                         size_t mr_block_size);

XNN_PRIVATE void xnn_compute_hmp_inline_packed_qp8gemm(
    const struct gemm_context* context, uint32_t uarch_index, size_t thread_id,
    size_t mr_block_start, size_t mr_block_size);
#endif  // XNN_MAX_UARCH_TYPES > 1

// Context for Sparse Matrix-Dense Matrix Multiplication.
// C [MxN] := A [MxK] * B [KxN] + bias [N]
// A and C are dense matrices with row-major storage, B is a sparse matrix.
struct spmm_context {
  // N dimension of the B and C matrices.
  // Corresponds to number of output channels in 1x1 convolution.
  size_t n;
  // M dimension of the A and C matrices, pre-scaled by sizeof(element
  // size). Corresponds to the stride, in bytes, between adjacent rows of C
  // matrix.
  size_t scaled_m;
  // Input matrix A.
  const void* input;
  // Packed bias elements and non-zero filter elements.
  const void* nonzero_weights;
  // Input pointer increments, in bytes, after each processed non-zero
  // weight.
  const int32_t* input_increments;
  // Number of non-zero filter elements per each N (output channel)
  // dimension.
  const uint32_t* output_channel_nonzeros;
  // Output matrix C.
  void* output;
  // Stride, in bytes, between matrices A corresponding to different images
  // in batched 1x1 Convolution
  size_t batched_input_stride;
  // Stride, in bytes, between matrices C corresponding to different images
  // in batched 1x1 Convolution
  size_t batched_output_stride;
  // Micro-kernel function pointer.
  xnn_spmm_ukernel_fn ukernel;
  // Output activation parameters.
  union {
    struct xnn_f32_minmax_params f32;
  } params;
};

XNN_PRIVATE void xnn_compute_spmm(const struct spmm_context* context,
                                  size_t batch_index, size_t mr_block_start,
                                  size_t mr_block_size);

// Context for initializing the indirection buffer for conv2d igemm.
struct conv2d_igemm_indirection_init_context {
  const void** indirection_buffer;
  const void* input;
  const void* zero_buffer;
  size_t input_pixel_stride;
  size_t input_height;
  size_t input_width;
  size_t output_height;
  size_t output_width;
  size_t kernel_height;
  size_t kernel_width;
  size_t stride_height;
  size_t stride_width;
  size_t dilation_height;
  size_t dilation_width;
  size_t input_padding_top;
  size_t input_padding_left;
};

// Context for Indirect Dense Matrix Multiplication.
// C [BxGxMxN] := A [BxGxMxK] * B[BxGxKxN] + bias [BxGxN]
// Where B and bias have been packed into packed_w.
struct igemm_context {
  size_t ks;
  size_t ks_scaled;
  // Number of input channels (K).
  size_t kc;
  // Stride, in bytes, between output channel (N) of weights.
  size_t w_stride;
  // Indirection buffer for input matrix A.
  const void** indirect_a;
  // Offset of each pointer in indirection buffer.
  size_t a_offset;
  // Zero buffer.
  void* zero;
  // Zero buffers.
  void** zero_buffers;
  // Pointer to weights (kernel and bias) that have been packed.
  const void* packed_w;
  // Output matrix C.
  void* c;
  // Stride, in bytes, between each row (M) of C.
  size_t cm_stride;
  // Stride, in bytes, between columns (N) of C written.
  size_t cn_stride;
  // Stride, in bytes, between each group (G) of A.
  size_t ga_stride;
  // Stride, in bytes, between each group (G) of packed weights.
  size_t gw_stride;
  // Stride, in bytes, between each group (G) of C.
  size_t gc_stride;
  // Stride, in bytes, between each batch (B) of A.
  size_t ba_stride;
  // Stride, in bytes, between each batch (B) of C.
  size_t bc_stride;
  // Size, in bytes, of each element of C.
  uint32_t log2_csize;
  // Size, in bytes, of the zero buffer.
  size_t zero_size;
  // Value of mr in the microkernel.
  size_t mr;
  // Value of nc in the weights.
  size_t nc;
  // IGEMM microkernels.
  union {
    struct xnn_hmp_igemm_ukernel ukernel;
    struct xnn_hmp_dqigemm_ukernel dq_ukernel;
  };
  // Parameters for dynamically quantized inputs.
  const struct xnn_qd8_quantization_params* quantization_params;
  // Parameters for fused activations.
  union {
    union xnn_qs8_conv_minmax_params qs8;
    union xnn_qu8_conv_minmax_params qu8;
    struct xnn_f16_scaleminmax_params f16;
    struct xnn_f32_minmax_params f32;
  } params;
};

XNN_PRIVATE void xnn_compute_grouped_dqigemm(
    const struct igemm_context* context, size_t group_index,
    size_t nr_block_start, size_t mr_block_start, size_t nr_block_size,
    size_t mr_block_size);

XNN_PRIVATE void xnn_compute_grouped_igemm(const struct igemm_context* context,
                                           size_t group_index,
                                           size_t nr_block_start,
                                           size_t mr_block_start,
                                           size_t nr_block_size,
                                           size_t mr_block_size);

XNN_PRIVATE void xnn_compute_grouped_batch_dqigemm(
    const struct igemm_context* context, size_t batch_index, size_t group_index,
    size_t nr_block_start, size_t mr_block_start, size_t nr_block_size,
    size_t mr_block_size);

XNN_PRIVATE void xnn_compute_grouped_batch_igemm(
    const struct igemm_context* context, size_t batch_index, size_t group_index,
    size_t nr_block_start, size_t mr_block_start, size_t nr_block_size,
    size_t mr_block_size);

XNN_PRIVATE void xnn_compute_dq_zero_buffer_igemm(
    const struct igemm_context* context, size_t size);

XNN_PRIVATE void xnn_compute_dqigemm(const struct igemm_context* context,
                                     size_t nr_block_start,
                                     size_t mr_block_start,
                                     size_t nr_block_size,
                                     size_t mr_block_size);

XNN_PRIVATE void xnn_compute_igemm(const struct igemm_context* context,
                                   size_t nr_block_start, size_t mr_block_start,
                                   size_t nr_block_size, size_t mr_block_size);

XNN_PRIVATE void xnn_compute_conv2d_igemm_indirection(
    const struct conv2d_igemm_indirection_init_context* context,
    size_t output_tile_start, size_t output_tile_size);

XNN_PRIVATE void xnn_compute_batch_dqigemm(const struct igemm_context* context,
                                           size_t batch_index,
                                           size_t nr_block_start,
                                           size_t mr_block_start,
                                           size_t nr_block_size,
                                           size_t mr_block_size);

XNN_PRIVATE void xnn_compute_batch_igemm(const struct igemm_context* context,
                                         size_t batch_index,
                                         size_t nr_block_start,
                                         size_t mr_block_start,
                                         size_t nr_block_size,
                                         size_t mr_block_size);

#if XNN_MAX_UARCH_TYPES > 1
XNN_PRIVATE void xnn_compute_hmp_grouped_igemm(
    const struct igemm_context* context, uint32_t uarch_index,
    size_t group_index, size_t nr_block_start, size_t mr_block_start,
    size_t nr_block_size, size_t mr_block_size);

XNN_PRIVATE void xnn_compute_hmp_grouped_dqigemm(
    const struct igemm_context* context, uint32_t uarch_index,
    size_t group_index, size_t nr_block_start, size_t mr_block_start,
    size_t nr_block_size, size_t mr_block_size);

XNN_PRIVATE void xnn_compute_hmp_grouped_batch_dqigemm(
    const struct igemm_context* context, uint32_t uarch_index,
    size_t batch_index, size_t group_index, size_t nr_block_start,
    size_t mr_block_start, size_t nr_block_size, size_t mr_block_size);

XNN_PRIVATE void xnn_compute_hmp_grouped_batch_igemm(
    const struct igemm_context* context, uint32_t uarch_index,
    size_t batch_index, size_t group_index, size_t nr_block_start,
    size_t mr_block_start, size_t nr_block_size, size_t mr_block_size);

XNN_PRIVATE void xnn_compute_hmp_dqigemm(const struct igemm_context* context,
                                         uint32_t uarch_index,
                                         size_t nr_block_start,
                                         size_t mr_block_start,
                                         size_t nr_block_size,
                                         size_t mr_block_size);

XNN_PRIVATE void xnn_compute_hmp_igemm(const struct igemm_context* context,
                                       uint32_t uarch_index,
                                       size_t nr_block_start,
                                       size_t mr_block_start,
                                       size_t nr_block_size,
                                       size_t mr_block_size);

XNN_PRIVATE void xnn_compute_batch_hmp_dqigemm(
    const struct igemm_context* context, uint32_t uarch_index,
    size_t batch_index, size_t nr_block_start, size_t mr_block_start,
    size_t nr_block_size, size_t mr_block_size);

XNN_PRIVATE void xnn_compute_batch_hmp_igemm(
    const struct igemm_context* context, uint32_t uarch_index,
    size_t batch_index, size_t nr_block_start, size_t mr_block_start,
    size_t nr_block_size, size_t mr_block_size);
#endif  // XNN_MAX_UARCH_TYPES > 1

struct subgemm_context {
  const struct subconvolution_params* subconvolution_params;
  size_t kc;
  const void* a;
  size_t ax_stride;
  size_t ay_stride;
  size_t cx_stride;
  size_t cy_stride;
  size_t cn_stride;
  size_t ga_stride;
  size_t gw_stride;
  size_t gc_stride;
  size_t ba_stride;
  size_t bc_stride;
  uint32_t log2_csize;
  struct xnn_hmp_gemm_ukernel ukernel;
  union {
    union xnn_qs8_conv_minmax_params qs8;
    union xnn_qu8_conv_minmax_params qu8;
    struct xnn_f16_scaleminmax_params f16;
    struct xnn_f32_minmax_params f32;
  } params;
};

XNN_PRIVATE void xnn_compute_grouped_subgemm2d(
    const struct subgemm_context* context, size_t batch_index,
    size_t group_index, size_t subkernel_index, size_t slice_y,
    size_t slice_x_start, size_t nc_block_start, size_t slice_x_max,
    size_t nc_block_size);

struct subconv_context {
  const struct subconvolution_params* subconvolution_params;
  size_t kc;
  size_t a_offset;
  void* zero;
  // Zero buffers.
  void** zero_buffers;
  size_t cx_stride;
  size_t cy_stride;
  size_t cn_stride;
  size_t ga_stride;
  size_t gw_stride;
  size_t gc_stride;
  size_t ba_stride;
  size_t bc_stride;
  uint32_t log2_csize;
  // Size, in bytes, of the zero buffer.
  size_t zero_size;
  union {
    struct xnn_hmp_igemm_ukernel ukernel;
    struct xnn_hmp_dqigemm_ukernel dq_ukernel;
  };
  // Parameters for dynamically quantized inputs.
  const struct xnn_qd8_quantization_params* quantization_params;
  union {
    union xnn_qs8_conv_minmax_params qs8;
    union xnn_qu8_conv_minmax_params qu8;
    struct xnn_f16_scaleminmax_params f16;
    struct xnn_f32_minmax_params f32;
  } params;
};

XNN_PRIVATE void xnn_compute_dq_zero_buffer_subconv(
    const struct subconv_context* context, size_t size);

XNN_PRIVATE void xnn_compute_grouped_subconv2d(
    const struct subconv_context* context, size_t batch_index,
    size_t group_index, size_t subkernel_index, size_t slice_y,
    size_t slice_x_start, size_t nr_block_start, size_t slice_x_max,
    size_t nr_block_size);

XNN_PRIVATE void xnn_compute_grouped_dqsubconv2d(
    const struct subconv_context* context, size_t batch_index,
    size_t group_index, size_t subkernel_index, size_t slice_y,
    size_t slice_x_start, size_t nr_block_start, size_t slice_x_max,
    size_t nr_block_size);

XNN_PRIVATE void xnn_compute_subconv2d(
    const struct subconv_context* context, size_t batch_index,
    size_t subkernel_index, size_t slice_y, size_t slice_x_start,
    size_t nr_block_start, size_t slice_x_max, size_t nr_block_size);

XNN_PRIVATE void xnn_compute_dqsubconv2d(
    const struct subconv_context* context, size_t batch_index,
    size_t subkernel_index, size_t slice_y, size_t slice_x_start,
    size_t nr_block_start, size_t slice_x_max, size_t nr_block_size);

struct conv2d_context {
  size_t input_height;
  size_t input_width;
  const void* input;
  size_t input_batch_stride;
  const void* zero;
  const void* packed_weights;
  void* output;
  size_t output_batch_stride;
  size_t input_padding_top;
  size_t output_channels;
  size_t output_height_stride;
  size_t output_channel_stride;
  union {
    xnn_conv_hwc2chw_ukernel_fn hwc2chw_ukernel;
  };
  union {
    struct xnn_f32_minmax_params f32;
  } params;
};

XNN_PRIVATE void xnn_compute_conv2d_hwc2chw(
    const struct conv2d_context* context, size_t batch_index,
    size_t output_y_start, size_t output_y_slice);

// Context for initializing the indirection buffer for dwconv.
struct dwconv_indirection_init_context {
  const void** indirection_buffer;
  const void* input;
  const void* zero_buffer;
  size_t input_pixel_stride;
  size_t input_height;
  size_t input_width;
  size_t output_height;
  size_t output_width;
  size_t kernel_height;
  size_t kernel_width;
  size_t stride_height;
  size_t stride_width;
  size_t dilation_height;
  size_t dilation_width;
  size_t input_padding_top;
  size_t input_padding_left;
  size_t step_height;
  size_t step_width;
  size_t tile_size;
};

struct dwconv_context {
  size_t kernel_size;
  const void** indirect_input;
  intptr_t indirect_input_width_stride;
  size_t indirect_input_height_stride;
  size_t input_offset;
  size_t input_batch_stride;
  size_t input_channel_stride;
  const void* packed_weights;
  size_t weights_channel_stride;
  void* output;
  size_t output_batch_stride;
  size_t output_height_stride;
  size_t output_pixel_stride;
  size_t output_channel_stride;
  size_t output_height;
  size_t output_width;
  size_t groups;
  const void* zero;
  union {
    union xnn_qs8_conv_minmax_params qs8;
    union xnn_qu8_conv_minmax_params qu8;
    struct xnn_f16_minmax_params f16;
    struct xnn_f32_minmax_params f32;
  } params;
  xnn_dwconv_ukernel_fn ukernel;
};

XNN_PRIVATE void xnn_compute_dwconv_indirection(
    const struct dwconv_indirection_init_context* context,
    size_t output_y_start, size_t output_y_tile);
XNN_PRIVATE void xnn_compute_dwconv_unipass(
    const struct dwconv_context* context, size_t batch_index, size_t output_y,
    size_t output_c_start, size_t output_c_tile);

struct dwconv2d_context {
  size_t input_height;
  size_t input_width;
  const void* input;
  const void* zero;
  uint32_t input_padding_top;
  size_t input_channel_stride;
  size_t input_batch_stride;
  const void* packed_weights;
  size_t weights_channel_stride;
  void* output;
  size_t output_channel_stride;
  size_t output_batch_stride;
  union {
    struct xnn_f32_minmax_params f32;
  } params;
  union {
    xnn_dwconv2d_chw_ukernel_fn chw_ukernel;
  };
};

XNN_PRIVATE void xnn_compute_dwconv2d_chw(
    const struct dwconv2d_context* context, size_t batch_index, size_t channel);

struct max_pooling_context {
  const void** indirect_input;
  size_t indirect_input_height_stride;
  size_t input_offset;
  size_t input_batch_stride;
  void* output;
  size_t output_batch_stride;
  size_t output_height_stride;
  size_t output_width;
  size_t pooling_size;
  size_t channels;
  size_t input_increment;
  size_t output_increment;
  union {
    struct xnn_u8_minmax_params u8;
    struct xnn_s8_minmax_params s8;
    struct xnn_f16_minmax_params f16;
    struct xnn_f32_minmax_params f32;
  } params;
  xnn_maxpool_ukernel_fn ukernel;
};

XNN_PRIVATE void xnn_compute_max_pooling(
    const struct max_pooling_context* context, size_t batch_index,
    size_t output_y);

struct unpooling_context {
  const void* input;
  size_t input_height_stride;
  size_t input_width_stride;
  const uint32_t* index;
  size_t index_height_stride;
  size_t index_width_stride;
  const void** indirect_output;
  size_t indirect_output_height_stride;
  size_t indirect_output_width_stride;
  size_t pooling_size;
  size_t channels;
  uint32_t fill_value;
  xnn_unpool_ukernel_fn ukernel;
};

XNN_PRIVATE void xnn_compute_unpooling(const struct unpooling_context* context,
                                       size_t input_y, size_t input_x);

struct argmax_pooling_context {
  const void** indirect_input;
  size_t indirect_input_height_stride;
  size_t input_offset;
  size_t input_batch_stride;
  void* output;
  size_t output_batch_stride;
  size_t output_height_stride;
  size_t output_height;
  size_t output_width;
  uint32_t* index;
  size_t index_batch_stride;
  size_t index_height_stride;
  size_t pooling_size;
  size_t channels;
  size_t input_increment;
  size_t output_increment;
  size_t index_increment;
  xnn_argmaxpool_unipass_ukernel_fn ukernel;
};

XNN_PRIVATE void xnn_compute_argmax_pooling(
    const struct argmax_pooling_context* context, size_t batch_index,
    size_t output_y);

struct average_pooling_context {
  const void** indirect_input;
  size_t indirect_input_height_stride;
  size_t input_offset;
  size_t input_batch_stride;

  // Stride to get to the next y of input. Used when we have compressed
  // indirection buffers (i.e. indirection buffers contain only pointers to the
  // first row of input).
  size_t input_y_stride;
  size_t indirect_top_height;  // Number of output rows that form the top
                               // section of indirection buffer.
  size_t indirect_bot_start;  // Smallest output row y for the bottom section of
                              // indirection buffer.

  const void* pixelwise_buffer;
  size_t pixelwise_buffer_height_stride;
  void* output;
  size_t output_batch_stride;
  size_t output_height_stride;
  size_t output_width;
  size_t pooling_size;
  size_t channels;
  const void* zero;
  size_t input_increment;
  size_t output_increment;
  union {
    struct xnn_f16_scaleminmax_params f16;
    struct xnn_f32_scaleminmax_params f32;
  } params;
  xnn_avgpool_ukernel_fn ukernel;
};

XNN_PRIVATE void xnn_compute_average_pooling(
    const struct average_pooling_context* context, size_t batch_index,
    size_t output_y);

struct resize_bilinear_nhwc_indirection_init_context {
  const void** buffer;
  const void* input;
  size_t indirect_input_offset;
  size_t input_pixel_stride;
  size_t input_offset;
  size_t input_height;
  size_t input_width;
  size_t output_height;
  size_t output_width;
  bool align_corners;
  bool tensorflow_legacy_mode;
  xnn_indirection_init_resize_bilinear2d_hwc_fn indirection_init;
};

struct resize_bilinear_context {
  // Number of channels multiplied by sizeof(input element).
  size_t scaled_channels;
  // Indirection buffer with pointers related to rows of input pixels.
  const void** indirect_input;
  // Offset, in bytes, to be added to pointers in indirection buffer.
  size_t input_offset;
  // Stride, in bytes, between images of consecutive batches in the input.
  size_t input_batch_stride;
  // Packed pairs of (x, y) linear interpolation coefficients.
  const void* packed_weights;
  // Pointer to the output tensor.
  void* output;
  // Stride, in bytes, between adjacent pixels in the output.
  size_t output_pixel_stride;
  // Stride, in bytes, between images of consecutive batches in the output.
  size_t output_batch_stride;
  // log2(sizeof(weight element)).
  uint32_t log2_wsize;
  // Pointer to BILINEAR micro-kernel function.
  xnn_ibilinear_ukernel_fn ukernel;
};

struct resize_bilinear_chw_context {
  // Number of pixels per output image plane.
  size_t output_pixels;
  // Number of channels multiplied by sizeof(input element).
  size_t channels;
  // Stride, in bytes, between adjacent channels in the input.
  size_t input_channel_stride;
  // Indirection buffer with pointers related to rows of input pixels.
  const void** indirect_input;
  // Offset, in bytes, to be added to pointers in indirection buffer.
  size_t input_offset;
  // Stride, in bytes, between images of consecutive batches in the input.
  size_t input_batch_stride;
  // Packed pairs of (x, y) linear interpolation coefficients.
  const void* packed_weights;
  // Pointer to the output tensor.
  void* output;
  // Stride, in bytes, between images of consecutive batches in the output.
  size_t output_batch_stride;
  // Stride, in bytes, between consecutive channels of an output image.
  size_t output_channel_stride;
  // Pointer to BILINEAR micro-kernel function.
  xnn_ibilinear_chw_ukernel_fn ukernel;
};

XNN_PRIVATE void xnn_compute_resize_bilinear_indirection(
    const struct resize_bilinear_nhwc_indirection_init_context* context,
    size_t output_y_start, size_t output_y_tile);
XNN_PRIVATE void xnn_compute_resize_bilinear(
    const struct resize_bilinear_context* context, size_t batch_index,
    size_t pixel_start, size_t pixel_range);
XNN_PRIVATE void xnn_compute_resize_bilinear_chw(
    const struct resize_bilinear_chw_context* context, size_t batch_index,
    size_t pixel_start, size_t pixel_range);

struct elementwise_binary_context {
  const void* a;
  size_t a_stride[XNN_MAX_TENSOR_DIMS - 1];
  const void* b;
  size_t b_stride[XNN_MAX_TENSOR_DIMS - 1];
  void* y;
  size_t y_stride[XNN_MAX_TENSOR_DIMS - 1];
  size_t elements;
  union xnn_binary_uparams params;
  xnn_vbinary_ukernel_fn ukernel;
  bool flip_a_b;
};

XNN_PRIVATE void xnn_compute_elementwise_binary_1d_tile(
    const struct elementwise_binary_context* context, size_t offset,
    size_t tile);
XNN_PRIVATE void xnn_compute_elementwise_binary_1d(
    const struct elementwise_binary_context* context, size_t offset,
    size_t count);
XNN_PRIVATE void xnn_compute_elementwise_binary_2d(
    const struct elementwise_binary_context* context, size_t i, size_t offset,
    size_t count);
XNN_PRIVATE void xnn_compute_elementwise_binary_3d(
    const struct elementwise_binary_context* context, size_t i, size_t offset_j,
    size_t offset_k, size_t count_j, size_t count_k);
XNN_PRIVATE void xnn_compute_elementwise_binary_4d(
    const struct elementwise_binary_context* context, size_t i, size_t j,
    size_t offset_k, size_t offset_l, size_t count_k, size_t count_l);
XNN_PRIVATE void xnn_compute_elementwise_binary_5d(
    const struct elementwise_binary_context* context, size_t i, size_t j,
    size_t k, size_t l, size_t m);

struct lut_strided_context {
  size_t n;
  const void* x;
  size_t x_stride;
  const void* t;
  void* y;
  size_t y_stride;
  xnn_x8_lut_ukernel_fn ukernel;
};

XNN_PRIVATE void xnn_compute_lut_strided(
    const struct lut_strided_context* context, size_t batch_offset,
    size_t batch_range);

struct lut_contiguous_context {
  const void* x;
  size_t x_stride;
  const void* t;
  void* y;
  size_t y_stride;
  xnn_x8_lut_ukernel_fn ukernel;
};

XNN_PRIVATE void xnn_compute_lut_contiguous(
    const struct lut_contiguous_context* context, size_t offset, size_t size);

struct univector_strided_context {
  size_t n;
  const void* x;
  size_t x_stride;
  void* y;
  size_t y_stride;
  xnn_vunary_ukernel_fn ukernel;
  union xnn_unary_uparams params;
};

XNN_PRIVATE void xnn_compute_univector_strided(
    const struct univector_strided_context* context, size_t batch_index,
    size_t batch_range);

struct univector_contiguous_context {
  const void* x;
  void* y;
  uint16_t log2_xsize;
  uint16_t log2_ysize;
  xnn_vunary_ukernel_fn ukernel;
  union xnn_unary_uparams params;
};

XNN_PRIVATE void xnn_compute_univector_contiguous(
    const struct univector_contiguous_context* context, size_t offset,
    size_t size);

struct reduce_context {
  const void* input;
  void* output;
  void* workspace;
  uint32_t identity_value;
  const void* zero;
  size_t input_shape[XNN_MAX_TENSOR_DIMS];
  size_t input_stride[XNN_MAX_TENSOR_DIMS];
  size_t output_stride[XNN_MAX_TENSOR_DIMS];
  size_t channels;
  size_t accumulation_element_size;
  size_t output_element_size;
  union {
    xnn_reduce_ukernel_fn contiguous_reduce;
    xnn_reduce_discontiguous_ukernel_fn discontiguous_reduce;
  } ukernel;
  xnn_vunary_ukernel_fn cvt_ukernel;
  xnn_fill_ukernel_fn fill_ukernel;
  struct xnn_reduce_params params;
  union xnn_unary_uparams cvt_params;
};

// Compute contiguous reduction over the 1st, 3rd and 5th dimensions of the
// input tensor.
XNN_PRIVATE void xnn_compute_contiguous_reduce(
    const struct reduce_context* context, size_t output_idx0,
    size_t output_idx1, size_t output_idx2, size_t output1_block_size,
    size_t output2_block_size);

// Compute discontiguous reduction over the 0st, 2rd and 4th dimensions of the
// input tensor.
XNN_PRIVATE void xnn_compute_discontiguous_reduce(
    const struct reduce_context* context, size_t output_idx0,
    size_t output_idx1, size_t output_idx2, size_t output1_block_size,
    size_t output2_block_size);

struct vmulcaddc_context {
  size_t n;
  const void* x;
  size_t x_stride;
  const void* w;
  void* y;
  size_t y_stride;
  xnn_vmulcaddc_ukernel_fn ukernel;
  union {
    struct xnn_f16_minmax_params f16;
    struct xnn_f32_minmax_params f32;
  } params;
};

XNN_PRIVATE void xnn_compute_vmulcaddc(const struct vmulcaddc_context* context,
                                       size_t batch_start, size_t batch_size);

struct pad_context {
  const void* input;
  size_t input_stride[XNN_MAX_TENSOR_DIMS - 1];
  void* output;
  size_t output_stride[XNN_MAX_TENSOR_DIMS - 1];
  size_t pre_paddings[XNN_MAX_TENSOR_DIMS];
  size_t post_paddings[1];
  size_t input_size[XNN_MAX_TENSOR_DIMS];
  size_t output_size[1];
  uint32_t padding_value;
  xnn_pad_ukernel_fn pad_ukernel;
  xnn_fill_ukernel_fn fill_ukernel;
};

XNN_PRIVATE void xnn_compute_pad_5d(const struct pad_context* context, size_t i,
                                    size_t j, size_t k, size_t l, size_t m);

struct slice_context {
  const void* input;
  size_t input_stride[XNN_MAX_TENSOR_DIMS - 1];
  void* output;
  size_t output_stride[XNN_MAX_TENSOR_DIMS - 1];
  size_t offsets[XNN_MAX_TENSOR_DIMS];
  size_t contiguous_size;
  xnn_vunary_ukernel_fn ukernel;
  size_t num_normalized_dims;
};

XNN_PRIVATE void xnn_compute_slice_1d(const struct slice_context* context,
                                      size_t i);
XNN_PRIVATE void xnn_compute_slice_2d(const struct slice_context* context,
                                      size_t i, size_t j);
XNN_PRIVATE void xnn_compute_slice_3d(const struct slice_context* context,
                                      size_t i, size_t j, size_t k);
XNN_PRIVATE void xnn_compute_slice_4d(const struct slice_context* context,
                                      size_t i, size_t j, size_t k, size_t l);
XNN_PRIVATE void xnn_compute_slice_5d(const struct slice_context* context,
                                      size_t i, size_t j, size_t k, size_t l,
                                      size_t m);

struct f16_qd8_convert_context {
  size_t n;
  const void* x;
  size_t x_stride;
  int8_t* y;
  size_t y_stride;
  size_t batch_size;
  struct xnn_qd8_quantization_params* quantization_params;
  xnn_reduce_ukernel_fn rminmax_ukernel;
  xnn_vunary_ukernel_fn convert_ukernel;
  xnn_init_unary_uparams_fn init_params;
  union {
    struct xnn_f16_default_params f16_default;
  } params;
};

struct f32_qd8_convert_context {
  size_t n;
  const float* x;
  size_t x_stride;
  int8_t* y;
  size_t y_stride;
  size_t batch_size;
  struct xnn_qd8_quantization_params* quantization_params;
  xnn_reduce_ukernel_fn rminmax_ukernel;
  xnn_vunary_ukernel_fn convert_ukernel;
  xnn_init_unary_uparams_fn init_params;
  union {
    struct xnn_f32_default_params f32_default;
  } params;
};

XNN_PRIVATE void xnn_compute_f16_qd8_convert(
    const struct f16_qd8_convert_context* context, size_t batch_offset,
    size_t batch_range);

XNN_PRIVATE void xnn_compute_f16_qdu8_convert(
    const struct f16_qd8_convert_context* context, size_t batch_offset,
    size_t batch_range);

XNN_PRIVATE void xnn_compute_f32_qd8_convert(
    const struct f32_qd8_convert_context* context, size_t batch_offset,
    size_t batch_range);

XNN_PRIVATE void xnn_compute_f32_qdu8_convert(
    const struct f32_qd8_convert_context* context, size_t batch_offset,
    size_t batch_range);

XNN_PRIVATE void xnn_compute_pad_qd8_params(
    const struct f32_qd8_convert_context* context, size_t batch_index);

struct pack_lh_context {
  size_t m;
  size_t k;
  size_t mr;
  size_t kr;
  size_t sr;
  const void* XNN_RESTRICT lhs;
  size_t lhs_stride;
  size_t gi_stride;
  size_t gp_stride;
  void* XNN_RESTRICT lhs_packed;
  xnn_pack_lh_ukernel_fn pack_lh_ukernel;
  xnn_pack_lh_offset_fn packed_offset_fn;
  size_t workspace_offset;
};

XNN_PRIVATE void xnn_compute_pack_lh(const struct pack_lh_context* context,
                                     size_t group_idx, size_t m_idx_start,
                                     size_t tile);

struct f32_qp8_convert_context {
  size_t m;
  size_t k;
  size_t mr;
  size_t kr;
  size_t sr;
  size_t group_stride;
  const float* XNN_RESTRICT lhs;
  size_t lhs_stride;
  int8_t* XNN_RESTRICT lhs_packed;
  xnn_x8_packq_f32qp8_ukernel_fn packq_ukernel;
};

XNN_PRIVATE void xnn_compute_f32_qp8_convert(
    const struct f32_qp8_convert_context* context, size_t group_idx,
    size_t m_idx_start, size_t m_tile);

struct u8_softmax_context {
  size_t n;
  const uint8_t* x;
  size_t x_stride;
  const uint32_t* t;
  uint8_t* y;
  size_t y_stride;
  xnn_u8_rmax_ukernel_fn rmax_ukernel;
  xnn_u8_lut32norm_ukernel_fn lut_norm_ukernel;
};

XNN_PRIVATE void xnn_compute_u8_softmax(
    const struct u8_softmax_context* context, size_t batch_index);

typedef void (*xnn_compute_reciprocal_fn)(const void* input, void* output);

struct floating_point_softmax_context {
  size_t n;
  const void* x;
  size_t x_stride;
  void* y;
  size_t y_stride;
  xnn_rmax_ukernel_fn rmax_ukernel;
  xnn_raddstoreexpminusmax_ukernel_fn raddstoreexpminusmax_ukernel;
  xnn_compute_reciprocal_fn compute_reciprocal;
  xnn_vbinary_ukernel_fn vmulc_ukernel;
  union {
    struct xnn_f16_minmax_params f16;
    struct xnn_f32_minmax_params f32;
  } minmax_params;
  union {
    struct xnn_f16_default_params f16;
    struct xnn_f32_default_params f32;
  } expminus_params;
  union {
    struct xnn_f16_default_params f16;
    struct xnn_f32_default_params f32;
  } rmax_params;
  union {
    xnn_float16 f16;
    float f32;
  } rmax_init;
};

XNN_PRIVATE void xnn_compute_floating_point_softmax(
    const struct floating_point_softmax_context* context, size_t batch_index);

struct rope_context {
  size_t scaled_channels;
  size_t batch_stride;
  size_t head_stride;
  size_t sequence_stride;
  const void* input;
  const void* weights;
  void* output;
  xnn_vbinary_ukernel_fn vcmul;
  union {
    struct xnn_f32_default_params f32;
  } params;
};

XNN_PRIVATE void xnn_compute_rope(const struct rope_context* context,
                                  size_t batch_index, size_t head_index,
                                  size_t sequence_index);

#endif  // THIRD_PARTY_XNNPACK_SRC_XNNPACK_COMPUTE_H_
