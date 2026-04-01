// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// PoC: Stack buffer overflows in XNNPACK/YNNPACK public API functions.
//
// Bug class: User-controlled count parameters used as loop bounds to write
// into fixed-size stack arrays without bounds checking.
//
// xnn_define_static_expand_dims: int32_t ynn_axes[XNN_MAX_TENSOR_DIMS=6]
//   written with num_new_axes as loop bound. No bounds check.
//
// xnn_define_static_reduce: int64_t signed_reduction_axes[XNN_MAX_TENSOR_DIMS=6]
//   written with num_reduction_axes as loop bound. No bounds check.

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "ynnpack/include/ynnpack.h"

namespace {

// Bug 1: xnn_define_static_expand_dims stack buffer overflow.
// ynnpack/xnnpack/subgraph.cc:774-776:
//   int32_t ynn_axes[XNN_MAX_TENSOR_DIMS];  // size 6
//   for (size_t i = 0; i < num_new_axes; ++i) {
//     ynn_axes[i] = new_axes[i];  // OOB when num_new_axes > 6
//   }
TEST(OobWrite, StaticExpandDimsStackOverflow) {
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_create_subgraph(2, 0, &subgraph), xnn_status_success);

  size_t dims[] = {4, 4};
  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 2, dims,
                                     nullptr, 0, XNN_VALUE_FLAG_EXTERNAL_INPUT,
                                     &input_id),
            xnn_status_success);
  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 2, dims,
                                     nullptr, 1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                                     &output_id),
            xnn_status_success);

  // num_new_axes=10 > XNN_MAX_TENSOR_DIMS=6.
  // The loop writes 10 int32_t elements into a 6-element stack array.
  // ASAN detects: stack-buffer-overflow WRITE of size 4.
  size_t new_axes[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  xnn_status status = xnn_define_static_expand_dims(subgraph, 10, new_axes,
                                                     input_id, output_id, 0);
  // Should return error, not overflow.
  EXPECT_NE(status, xnn_status_success);

  xnn_delete_subgraph(subgraph);
}

// Bug 2: xnn_define_static_reduce stack buffer overflow.
// ynnpack/xnnpack/subgraph.cc:805-807:
//   int64_t signed_reduction_axes[XNN_MAX_TENSOR_DIMS];  // size 6
//   for (int i = 0; i < num_reduction_axes; i++) {
//     signed_reduction_axes[i] = reduction_axes[i];  // OOB when > 6
//   }
TEST(OobWrite, StaticReduceStackOverflow) {
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_create_subgraph(2, 0, &subgraph), xnn_status_success);

  size_t dims[] = {2, 3, 4, 5};
  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 4, dims,
                                     nullptr, 0, XNN_VALUE_FLAG_EXTERNAL_INPUT,
                                     &input_id),
            xnn_status_success);
  uint32_t output_id = XNN_INVALID_VALUE_ID;
  size_t out_dims[] = {1};
  ASSERT_EQ(xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 1, out_dims,
                                     nullptr, 1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                                     &output_id),
            xnn_status_success);

  // num_reduction_axes=10 > XNN_MAX_TENSOR_DIMS=6.
  // The loop writes 10 int64_t elements into a 6-element stack array.
  // ASAN detects: stack-buffer-overflow WRITE of size 8.
  size_t reduction_axes[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  xnn_status status = xnn_define_static_reduce(subgraph, xnn_reduce_sum, 10,
                                                reduction_axes, input_id,
                                                output_id, 0);
  EXPECT_NE(status, xnn_status_success);

  xnn_delete_subgraph(subgraph);
}

// Bug 3: Missing axis validation in ynn_define_broadcast.
// broadcast.cc:38-41: axes_set[axis_to_slinky_dim(rank, axes[i])] = true
// No validate_axis call. Out-of-range axes produce invalid bitset indices.
TEST(OobWrite, BroadcastMissingAxisValidation) {
  ynn_subgraph_t sg = nullptr;
  ynn_create_subgraph(2, 0, &sg);
  size_t dims[] = {4, 4};
  uint32_t input_id = YNN_INVALID_VALUE_ID;
  ynn_define_tensor(sg, ynn_type_fp32, 2, dims, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id);
  uint32_t output_id = YNN_INVALID_VALUE_ID;
  ynn_define_tensor(sg, ynn_type_fp32, 2, dims, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id);

  // axis=100 on rank-2 tensor. axis_to_slinky_dim(2, 100) = -99.
  // This is UB: writes to bitset at index SIZE_MAX-98.
  int32_t axes[] = {100};
  ynn_status status =
      ynn_define_broadcast(sg, 1, axes, input_id, &output_id, 0);
  EXPECT_NE(status, ynn_status_success);
  ynn_delete_subgraph(sg);
}

// Bug 4: Off-by-one in ynn_define_fuse_dims.
// copy.cc:560: op.axes[axis_to_slinky_dim(rank, axes[i]+1)] = true
// axes[i]=rank-1 passes validate_axis, but axes[i]+1 goes to
// axis_to_slinky_dim which returns -1 → bitset[SIZE_MAX].
TEST(OobWrite, FuseDimsOffByOneOverflow) {
  ynn_subgraph_t sg = nullptr;
  ynn_create_subgraph(2, 0, &sg);
  size_t dims[] = {2, 3, 4, 5};
  uint32_t input_id = YNN_INVALID_VALUE_ID;
  ynn_define_tensor(sg, ynn_type_fp32, 4, dims, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id);
  uint32_t output_id = YNN_INVALID_VALUE_ID;
  size_t out_dims[] = {2, 3, 20};
  ynn_define_tensor(sg, ynn_type_fp32, 3, out_dims, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id);

  // axes[0]=3 (last valid axis for rank-4).
  // axis_to_slinky_dim(4, 4) = -1 → bitset[SIZE_MAX] = UB/OOB write.
  int32_t axes[] = {3};
  ynn_status status =
      ynn_define_fuse_dims(sg, 1, axes, input_id, &output_id, 0);
  EXPECT_NE(status, ynn_status_success);
  ynn_delete_subgraph(sg);
}

// Bug 5: Missing axis validation in ynn_define_reduce.
// reduce.cc:456-462: k_dims[axis_to_slinky_dim(rank, axes[i])] = true
// No validate_axis call.
TEST(OobWrite, ReduceMissingAxisValidation) {
  ynn_subgraph_t sg = nullptr;
  ynn_create_subgraph(2, 0, &sg);
  size_t dims[] = {2, 3, 4, 5};
  uint32_t input_id = YNN_INVALID_VALUE_ID;
  ynn_define_tensor(sg, ynn_type_fp32, 4, dims, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id);
  uint32_t output_id = YNN_INVALID_VALUE_ID;
  size_t out_dims[] = {2};
  ynn_define_tensor(sg, ynn_type_fp32, 1, out_dims, nullptr,
                    YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id);

  int32_t axes[] = {100};
  ynn_status status =
      ynn_define_reduce(sg, ynn_reduce_sum, 1, axes, input_id,
                        YNN_INVALID_VALUE_ID, &output_id, 0);
  EXPECT_NE(status, ynn_status_success);
  ynn_delete_subgraph(sg);
}

}  // namespace
