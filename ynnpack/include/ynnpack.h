// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_INCLUDE_YNNPACK_H_
#define XNNPACK_YNNPACK_INCLUDE_YNNPACK_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// A value ID that is never valid.
#define YNN_INVALID_VALUE_ID UINT32_MAX

// The most dimensions that can appear in a value.
#define YNN_MAX_TENSOR_RANK 8

// This flag indicates that YNNPACK should attempt to produce numerically
// consistent results from a specific build of YNNPACK. This causes YNNPACK to
// avoid using faster codepaths that are numerically inconsistent with any
// other codepath that could be used in the same compiled YNNPACK library.
#define YNN_FLAG_CONSISTENT_ARITHMETIC (1 << 1)

#ifdef __GNUC__
#define YNN_DEPRECATED __attribute__((deprecated))
#else
#define YNN_DEPRECATED
#endif

enum ynn_status {
  ynn_status_success = 0,
  ynn_status_error,
  ynn_status_invalid_parameter,
  ynn_status_unsupported_parameter,
  ynn_status_deprecated,
};

typedef struct ynn_subgraph* ynn_subgraph_t;

// This type is an alias for `slinky::thread_pool`. A `slinky::thread_pool`
// instance may be casted to `ynn_threadpool` and passed to YNNPACK APIs.
typedef struct ynn_threadpool* ynn_threadpool_t;

// Create a new subgraph, with `external_value_ids` reserved ids for external
// values.
// Supported flags: `YNN_FLAG_CONSISTENT_ARITHMETIC`
enum ynn_status ynn_create_subgraph(uint32_t external_value_ids, uint32_t flags,
                                    ynn_subgraph_t* subgraph_out);

// Delete a subgraph previously created with `ynn_create_subgraph`.
void ynn_delete_subgraph(ynn_subgraph_t subgraph);

// Apply subgraph rewrites and other optimizations to the subgraph.
ynn_status ynn_optimize_subgraph(ynn_subgraph_t subgraph,
                                 ynn_threadpool_t threadpool, uint32_t flags);

// Describes a type for a value.
enum ynn_type {
  ynn_type_invalid = -1,

  ynn_type_int2,
  ynn_type_int4,
  ynn_type_uint4,
  ynn_type_int8,
  ynn_type_uint8,
  ynn_type_int32,
  ynn_type_fp16,
  ynn_type_bf16,
  ynn_type_fp32,

  // For internal use only.
  ynn_type_opaque,
};

#define YNN_VALUE_FLAG_EXTERNAL_INPUT (1 << 0)
#define YNN_VALUE_FLAG_EXTERNAL_OUTPUT (1 << 1)
#define YNN_VALUE_FLAG_COPY_DATA (1 << 2)

// Define a new tensor in a subgraph.
//
// If the value is an external input (`YNN_VALUE_FLAG_EXTERNAL_INPUT`), `dims`
// is an optional parameter. If `dims` is NULL, or `dims[d]` is 0, the shape is
// dynamic in dimension `d`, and must be set with `ynn_set_external_value_shape`
// prior to calling `ynn_invoke_runtime`. If `dims` is non-NULL or `dims[d]` is
// not 0, the shape is static in dimension `d`, and cannot be changed.
//
// If the value is an external output ('YNN_VALUE_FLAG_EXTERNAL_OUTPUT'), the
// shape will be retrievable via `ynn_get_external_value_shape` after calling
// `xnn_reshape_runtime`.
//
// If `data` is non-NULL, the value is static, and `dims` is a required
// parameter. The shape is static and cannot be changed.
//
// In all other cases, the `rank` and `dims` parameters are ignored and inferred
// by the subgraph nodes that produce this value.
//
// The ID of the new tensor will be stored in `id_out`.
//
// `data` may be non-null, indicating the tensor has a constant value. The
// caller must maintain the lifetime of this data as long as the subgraph
// exists, unless the `YNN_VALUE_FLAG_COPY_DATA` flag is used, indicating that
// this function will make a copy of the data, releasing the caller of the
// obligation to maintain it.
//
// Quantization parameters are specified via other tensor IDs. Conceptually,
// these tensors must have the same shape as this tensor. Some common examples
// include:
// - Per-tensor quantization: `zero_point_id` and `scale_id` both refer to
//   broadcasted scalar values.
// - Per-channel quantization: `zero_point_id` refers to a broadcasted scalar,
//   `scale_id` refers to a tensor with a single non-broadcasted dimension.
// - Blockwise quantization: `zero_point_id` refers to a broadcasted scalar,
//   `scale_id` refers to a tensor that is broadcasted and reshaped.
enum ynn_status ynn_define_tensor_value(ynn_subgraph_t subgraph,
                                        enum ynn_type type, size_t rank,
                                        const size_t* dims, const void* data,
                                        uint32_t zero_point_id,
                                        uint32_t scale_id, uint32_t flags,
                                        uint32_t* id_out);

#define YNN_NODE_FLAG_KEEP_DIMS (1 << 0)
#define YNN_NODE_FLAG_SLICE_DIMS (1 << 0)
#define YNN_NODE_FLAG_RESHAPE_1D (1 << 0)

enum ynn_unary_operator {
  ynn_unary_invalid = 0,

  ynn_unary_abs,
  ynn_unary_ceil,
  ynn_unary_convert,
  ynn_unary_cosine,
  ynn_unary_cube_root,
  ynn_unary_erf,
  ynn_unary_exp,
  ynn_unary_expm1,
  ynn_unary_floor,
  ynn_unary_hardswish,
  ynn_unary_log,
  ynn_unary_log1p,
  ynn_unary_negate,
  ynn_unary_reciprocal_square_root,
  ynn_unary_round,
  ynn_unary_sigmoid,
  ynn_unary_sign,
  ynn_unary_sine,
  ynn_unary_square,
  ynn_unary_square_root,
  ynn_unary_tanh,
};

// Defines a unary operation of a single input to a single output.
enum ynn_status ynn_define_unary(ynn_subgraph_t subgraph,
                                 enum ynn_unary_operator op,
                                 uint32_t input_a_id, uint32_t* output_id,
                                 uint32_t flags);

enum ynn_binary_operator {
  ynn_binary_invalid = 0,

  ynn_binary_add,
  ynn_binary_copysign,
  ynn_binary_divide,
  ynn_binary_leaky_relu,  // computes a < 0 ? a * b : a
  ynn_binary_max,
  ynn_binary_min,
  ynn_binary_multiply,
  ynn_binary_pow,
  ynn_binary_squared_difference,
  ynn_binary_subtract,
};

// Defines a binary operation of two inputs to a single output. The two inputs
// are permitted to have a differing number of dimensions. The input with fewer
// dimensions will have leading broadcast dimensions inserted to match the rank
// of the other input. Dimensions that exist in both inputs must have the same
// extent.
enum ynn_status ynn_define_binary(ynn_subgraph_t subgraph,
                                  enum ynn_binary_operator op,
                                  uint32_t input_a_id, uint32_t input_b_id,
                                  uint32_t* output_id, uint32_t flags);

// Changes the shape of `input_id` to have the shape `new_dims`, by broadcasting
// extent 1 dimensions. If `new_dims[d]` is zero, dimension `d` is passed
// through unchanged. If the rank of `input_id` is less than `rank`,
// leading broadcasting dimensions are inserted.
enum ynn_status ynn_define_static_broadcast(ynn_subgraph_t subgraph,
                                            size_t rank, const size_t* new_dims,
                                            uint32_t input_id,
                                            uint32_t* output_id,
                                            uint32_t flags);

// Changes the shape of `input_id` to have a similar shape as `template_id`, by
// replacing extent 1 dimensions with broadcasts. The operation is limited to
// the set of dimensions in `axes`. If the rank of `input_id` is less than
// `num_axes`, leading broadcasting dimensions are inserted.
enum ynn_status ynn_define_broadcast_like(ynn_subgraph_t subgraph,
                                          size_t num_axes, const int32_t* axes,
                                          uint32_t input_id,
                                          uint32_t template_id,
                                          uint32_t* output_id, uint32_t flags);

// Replaces `axes` dimensions with broadcast dimensions. `axes` dimensions of
// the input must have extent 1 or already be broadcasted.
enum ynn_status ynn_define_broadcast(ynn_subgraph_t subgraph, size_t num_axes,
                                     const int32_t* axes, uint32_t input_id,
                                     uint32_t* output_id, uint32_t flags);

// Inserts new dimensions of extent 1 at the positions identified by `new_axes`.
enum ynn_status ynn_define_static_expand_dims(
    ynn_subgraph_t subgraph, size_t num_new_axes, const int32_t* new_axes,
    uint32_t input_id, uint32_t* output_id, uint32_t flags);

// Reinterprets the memory of `input_id` to have the shape `new_dims`. The new
// shape must have the same number of elements in it as the shape of `input_id`.
// `new_dims` may have exactly one zero in it, indicating that dimension should
// be computed such that the input and output have the same total size.
enum ynn_status ynn_define_static_reshape(ynn_subgraph_t subgraph, size_t rank,
                                          const size_t* new_dims,
                                          uint32_t input_id,
                                          uint32_t* output_id, uint32_t flags);

// Fuses `axes_count` dimensions starting at `axis` into one dimension. This is
// equivalent to `ynn_define_static_reshape`, except `new_dims` are dynamically
// determined from `input_id`.
enum ynn_status ynn_define_fuse_dim(ynn_subgraph_t subgraph, int32_t axis,
                                    size_t axes_count, uint32_t input_id,
                                    uint32_t* output_id, uint32_t flags);

// Splits `axis` into new dimensions identified by `splits`. `splits` may
// contain exactly one zero in it, indicating that dimension should be computed
// such that the input and output have the same total size.
enum ynn_status ynn_define_split_dim(ynn_subgraph_t subgraph, int32_t axis,
                                     size_t num_splits, const size_t* splits,
                                     uint32_t input_id, uint32_t* output_id,
                                     uint32_t flags);

// Fuses `axes_count` pairs of dimensions starting with the dimension identified
// by `axes`.
enum ynn_status ynn_define_fuse_dims(ynn_subgraph_t subgraph, size_t num_axes,
                                     const int32_t* axes, uint32_t input_id,
                                     uint32_t* output_id, uint32_t flags);

// Concatenates `input_ids` along the `axis` dimension.
enum ynn_status ynn_define_concatenate(ynn_subgraph_t subgraph, int32_t axis,
                                       size_t num_inputs,
                                       const uint32_t* input_ids,
                                       uint32_t* output_id, uint32_t flags);

// Stacks `input_ids` into an output with a new `axis` dimension.
enum ynn_status ynn_define_stack(ynn_subgraph_t subgraph, int32_t axis,
                                 size_t num_inputs, const uint32_t* input_ids,
                                 uint32_t* output_id, uint32_t flags);

// Copies `input_id` to `output_id`.
// TODO: I don't think we need this.
enum ynn_status ynn_define_copy(ynn_subgraph_t subgraph, uint32_t input_id,
                                uint32_t* output_id, uint32_t flags);

// Splits `input_id` into `output_ids` evenly in the `axis` dimension.
enum ynn_status ynn_define_even_split(ynn_subgraph_t subgraph, int32_t axis,
                                      uint32_t input_id, size_t num_outputs,
                                      uint32_t* output_ids, uint32_t flags);

// Extracts the range of indices `[begin, end)` with stride `strides` in the
// `axes` dimensions. If the `YNN_NODE_FLAG_SLICE_DIMS` flag is set, this
// operation slices `axes` at `begins`, and then removes those axes from the
// result.
enum ynn_status ynn_define_static_slice(
    ynn_subgraph_t subgraph, size_t num_axes, const int32_t* axes,
    const int64_t* begins, const int64_t* ends, const int64_t* strides,
    uint32_t input_id, uint32_t* output_id, uint32_t flags);

// Copy the input to the output, using a permutation to select the dimensions of
// the input.
enum ynn_status ynn_define_static_transpose(
    ynn_subgraph_t subgraph, size_t rank, const int32_t* permutation,
    uint32_t input_id, uint32_t* output_id, uint32_t flags);

// Copies from `input_id` (when not in the padded area) or `padding_id` (when in
// the padded area) to `output_id`. The padded area is defined by the
// `pre_paddings` and `post_paddings` extents in each dimension. The extents may
// be negative, indicating that the input is cropped instead of padded in that
// dimension.
enum ynn_status ynn_define_static_pad(ynn_subgraph_t subgraph, size_t num_axes,
                                      const int32_t* axes,
                                      const int64_t* pre_paddings,
                                      const int64_t* post_paddings,
                                      uint32_t input_id, uint32_t padding_id,
                                      uint32_t* output_id, uint32_t flags);

// Copies potentially overlapping windows of `input_id` into new "stencil
// dimensions". An example of one stencil dimension is:
//
//   output(dx, x) = input(x * stencil_stride + dx * stencil_dilation)
//
// If `padding_id` is `YNN_INVALID_VALUE_ID`, `dx` will have extent
// `stencil_axes`, and `x` will have the maximum extent such that
// `x * stencil_stride + dx * stencil_dilation` does not exceed the max of the
// original dimension in `input_id`.
//
// If `padding_id` is a valid value, padding is added such that the output
// (before dividing by the stride) has the same extent as the input.
//
// `stencil_axes` identifies which dimensions should have this transformation
// applied. The stencil dimension is placed as specified by `new_axes`.
//
// For example, to produce a 3x3 stencil with the following layout:
//
//    output(n, y, x, dy, dx, c) = input(n, y + dy, x + dx, c)
//
// Use:
// - num_stencils = 2
// - stencil_axes = [1, 2]
// - new_axes = [3, 4]
// - stencil_dims = [3, 3]
// - stencil_strides = [1, 1]
// - stencil_dilations = [1, 1]
enum ynn_status ynn_define_stencil_copy(
    ynn_subgraph_t subgraph, size_t num_stencils, const int32_t* stencil_axes,
    const int32_t* new_axes, const size_t* stencil_dims,
    const size_t* stencil_strides, const size_t* stencil_dilations,
    uint32_t input_id, uint32_t padding_id, uint32_t* output_id,
    uint32_t flags);

// Performs the operation:
//
//   output(batch_dims..., i, j) = c(batch_dims..., i, j)
//   output(batch_dims..., i, j) +=
//     a(batch_dims..., i, k_dims...) * b(batch_dims..., k_dims..., j)
//
// If num_k_dims = 1, this is a matrix multiply.
//
// If `output_id` is `YNN_INVALID_VALUE_ID`, the output type will be:
// - ynn_type_int32 if both `input_a_id` and `input_b_id` are integer values,
// - ynn_type_fp32 otherwise.
//
// If `input_b_id` is `YNN_INVALID_VALUE_ID`, `b` is defined to be the
// "identity" value for the reduction operator:
// - sum: 0
// - product: 1
// - min: (max value of type of `input_a_id`)
// - max: (min value of type of `input_a_id`)
enum ynn_status ynn_define_dot(ynn_subgraph_t subgraph, size_t num_k_dims,
                               uint32_t input_a_id, uint32_t input_b_id,
                               uint32_t input_c_id, uint32_t* output_id,
                               uint32_t flags);

enum ynn_reduce_operator {
  ynn_reduce_invalid = 0,

  ynn_reduce_max,
  ynn_reduce_min,
  ynn_reduce_min_max,
  ynn_reduce_sum,
  ynn_reduce_sum_squared,
};

// Performs the operation:
//
//   output(...) = b(...)
//   output(...) = op(output(...), a(...))
//
// `XNN_NODE_FLAG_KEEP_DIMS` indicates that a reduction should keep the reduced
// dimensions in the result (with extent 1).
//
// If `output_id` is `YNN_INVALID_VALUE_ID` and `op` is `ynn_reduce_sum` or the
// output type will be:
// - ynn_type_int32 if the input is an integer type
// - ynn_type_fp32 if the input is a floating point type
// If `output_id` is `YNN_INVALID_VALUE_ID` and `op` is `ynn_reduce_min` or
// `ynn_reduce_max`, the output type will be the same as the input type.
//
// If `input_b_id` is `YNN_INVALID_VALUE_ID`, `b` is defined to be the
// "identity" value for the reduction operator:
// - 0 if `op` is `ynn_reduce_sum`,
// - The min or max value of the type for `ynn_reduce_max` or `ynn_reduce_min`.
//
// If `op` produces multiple outputs, the outputs are stored in a new dimension
// 0 of the result.
enum ynn_status ynn_define_reduce(ynn_subgraph_t subgraph,
                                  enum ynn_reduce_operator op, size_t num_axes,
                                  const int32_t* axes, uint32_t input_a_id,
                                  uint32_t input_b_id, uint32_t* output_id,
                                  uint32_t flags);

// Get `axes` dimensions of the shape of `value_id` and store it in `output_id`.
// If the `YNN_NODE_FLAG_RESHAPE_1D` flag is set, the result will be the product
// of the selected axes.
enum ynn_status ynn_define_get_tensor_shape(ynn_subgraph_t subgraph,
                                            size_t num_axes,
                                            const int32_t* axes, ynn_type type,
                                            size_t rank, uint32_t value_id,
                                            uint32_t* output_id,
                                            uint32_t flags);

typedef struct ynn_runtime* ynn_runtime_t;

// An interface for allowing YNNPACK to access a provider of parallelism.
struct ynn_scheduler {
  // Returns how many tasks can be executed in parallel.
  int (*num_threads)(void* context);

  // Schedules a task to run, may return immediately before the task is
  // complete.
  void (*schedule)(void* context, void* task_context,
                   void (*task)(void* task_context));
};

typedef const struct ynn_scheduler* ynn_scheduler_t;

// Create a threadpool that uses `scheduler` to start work on other threads.
// `scheduler` is not copied, since it is stateless we expect it to be stored
// globally.
enum ynn_status ynn_create_threadpool(ynn_scheduler_t scheduler,
                                      void* scheduler_context, uint32_t flags,
                                      ynn_threadpool_t* threadpool_out);

void ynn_delete_threadpool(ynn_threadpool_t threadpool);

#define YNN_RUNTIME_FLAG_NO_SCHEDULE (1 << 0)

enum ynn_status ynn_create_runtime(ynn_subgraph_t subgraph,
                                   ynn_threadpool_t threadpool, uint32_t flags,
                                   ynn_runtime_t* runtime_out);

enum ynn_status ynn_update_runtime_with_threadpool(ynn_runtime_t runtime,
                                                   ynn_threadpool_t threadpool);

enum ynn_status ynn_set_external_value_shape(ynn_runtime_t runtime,
                                             uint32_t external_id, size_t rank,
                                             const size_t* dims);

enum ynn_status ynn_get_external_value_shape(ynn_runtime_t runtime,
                                             uint32_t external_id, size_t* rank,
                                             size_t* dims);

enum ynn_status ynn_reshape_runtime(ynn_runtime_t runtime);

enum ynn_status ynn_set_external_value_data(ynn_runtime_t runtime,
                                            uint32_t external_id, void* data);

enum ynn_status ynn_invoke_runtime(ynn_runtime_t runtime);

enum ynn_runtime_property {
  // The maximum number of tasks the runtime expects to run concurrently.
  // `result` should be an `int32_t`
  ynn_runtime_property_concurrency = 0,
};

// Query the runtime for the value of a property identified by `property`.
// `result` points to a property-specific type.`result_size` should indicate how
// much memory is available to write to `result`, and the value will be updated
// to indicate how much memory was actually written.
enum ynn_status ynn_query_runtime(ynn_runtime_t runtime,
                                  enum ynn_runtime_property property,
                                  void* result, size_t* result_size);

void ynn_delete_runtime(ynn_runtime_t runtime);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_YNNPACK_INCLUDE_YNNPACK_H_
