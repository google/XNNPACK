// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/allocation-type.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/subgraph.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/runtime-flags.h"

namespace {

void DefineGraphWithPersistentTensors(xnn_subgraph_t* subgraph,
                                      std::array<size_t, 4> dims) {
  // (input) -> abs ---(persistent)--> copy ---(persistent)--> hard swish ->
  // (output)
  xnn_create_subgraph(/*external_value_ids=*/0, /*flags=*/0, subgraph);
  uint32_t input_id = XNN_INVALID_VALUE_ID;
  xnn_define_tensor_value(*subgraph, xnn_datatype_fp32, dims.size(),
                          dims.data(), nullptr, XNN_INVALID_VALUE_ID,
                          XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id);
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t intermediate_id = XNN_INVALID_VALUE_ID;
  xnn_define_tensor_value(*subgraph, xnn_datatype_fp32, dims.size(),
                          dims.data(), nullptr, XNN_INVALID_VALUE_ID,
                          XNN_VALUE_FLAG_PERSISTENT, &intermediate_id);
  ASSERT_NE(intermediate_id, XNN_INVALID_VALUE_ID);

  uint32_t intermediate_id2 = XNN_INVALID_VALUE_ID;
  xnn_define_tensor_value(*subgraph, xnn_datatype_fp32, dims.size(),
                          dims.data(), nullptr, XNN_INVALID_VALUE_ID,
                          XNN_VALUE_FLAG_PERSISTENT, &intermediate_id2);
  ASSERT_NE(intermediate_id, XNN_INVALID_VALUE_ID);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  xnn_define_tensor_value(*subgraph, xnn_datatype_fp32, dims.size(),
                          dims.data(), nullptr, XNN_INVALID_VALUE_ID,
                          XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id);
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(xnn_status_success,
            xnn_define_unary(*subgraph, xnn_unary_abs, /*params=*/nullptr,
                             input_id, intermediate_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success, xnn_define_copy(*subgraph, intermediate_id,
                                                intermediate_id2, /*flags=*/0));
  ASSERT_EQ(xnn_status_success,
            xnn_define_unary(*subgraph, xnn_unary_hardswish, /*params=*/nullptr,
                             intermediate_id2, output_id, /*flags=*/0));
}

}  // namespace

TEST(WORKSPACE, persistent_tensors_allocated_at_start_of_workspace) {
  // Persistent tensors allocated at the start.
  xnn_initialize(/*allocator=*/nullptr);
  xnn_workspace_t workspace = nullptr;
  xnn_create_workspace(&workspace);
  const std::unique_ptr<xnn_workspace, decltype(&xnn_release_workspace)>
      auto_workspace(workspace, xnn_release_workspace);

  const std::array<size_t, 4> dims = {2, 20, 20, 3};
  xnnpack::Buffer<float> dummy_data(2 * 20 * 20 * 3, 0.0f,
                                    xnnpack::XnnExtraBytes);
  const std::array<xnn_external_value, 2> external_values = {
      xnn_external_value{0, dummy_data.data()},
      xnn_external_value{3, dummy_data.data()},
  };

  xnn_subgraph_t subgraph = nullptr;
  DefineGraphWithPersistentTensors(&subgraph, dims);
  const std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)>
      auto_subgraph(subgraph, xnn_delete_subgraph);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v4(subgraph, nullptr, workspace, nullptr,
                                  xnn_test_runtime_flags(), &runtime));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, 2, external_values.data()));
  const std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)>
      auto_runtime(runtime, xnn_delete_runtime);

  const size_t old_workspace_size = workspace->size;
  ASSERT_GE(old_workspace_size, 0);
  const void* old_runtime_workspace = runtime->workspace->data;
  ASSERT_NE(old_runtime_workspace, nullptr);

  size_t persistent_size = 0;
  for (size_t i = 0; i < runtime->num_values; i++) {
    const xnn_value* value = &runtime->values[i];
    if (value->allocation_type == xnn_allocation_type_persistent) {
      ASSERT_EQ((uintptr_t)value->data,
                (uintptr_t)workspace->data + persistent_size);
      persistent_size += round_up_po2(value->size, XNN_EXTRA_BYTES);
    }
  }

  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));
}

TEST(WORKSPACE, persistent_tensors_updated_correct_when_workspace_grows) {
  // Persistent tensors allocated at the start.
  xnn_initialize(/*allocator=*/nullptr);
  xnn_workspace_t workspace = nullptr;
  xnn_create_workspace(&workspace);
  const std::unique_ptr<xnn_workspace, decltype(&xnn_release_workspace)>
      auto_workspace(workspace, xnn_release_workspace);

  const std::array<size_t, 4> dims1 = {2, 20, 20, 3};
  xnnpack::Buffer<float> dummy_data(2 * 20 * 20 * 3, 0.0f,
                                    xnnpack::XnnExtraBytes);
  const std::array<xnn_external_value, 2> external_values = {
      xnn_external_value{0, dummy_data.data()},
      xnn_external_value{3, dummy_data.data()},
  };
  xnnpack::Buffer<float> dummy_data2(2 * 20 * 20 * 3 * 16, 0.0f,
                                     xnnpack::XnnExtraBytes);
  const std::array<xnn_external_value, 2> external_values2 = {
      xnn_external_value{0, dummy_data2.data()},
      xnn_external_value{3, dummy_data2.data()},
  };

  xnn_subgraph_t subgraph1 = nullptr;
  DefineGraphWithPersistentTensors(&subgraph1, dims1);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph1(
      subgraph1, xnn_delete_subgraph);

  xnn_runtime_t runtime1 = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v4(subgraph1, nullptr, workspace, nullptr,
                                  xnn_test_runtime_flags(), &runtime1));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime1, 2, external_values.data()));
  const std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)>
      auto_runtime(runtime1, xnn_delete_runtime);

  const size_t old_workspace_size = workspace->size;
  ASSERT_GE(old_workspace_size, 0);
  const void* old_runtime_workspace = runtime1->workspace->data;
  ASSERT_NE(old_runtime_workspace, nullptr);

  std::array<size_t, 4> dims2 = dims1;
  // Create the same graph but with larger tensors, this will require a larger
  // workspace.
  std::transform(dims2.begin(), dims2.end(), dims2.begin(),
                 [](size_t i) { return i * 2; });
  xnn_subgraph_t subgraph2 = nullptr;
  DefineGraphWithPersistentTensors(&subgraph2, dims2);
  const std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)>
      auto_subgraph2(subgraph2, xnn_delete_subgraph);
  xnn_runtime_t runtime2 = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v4(subgraph2, nullptr, workspace, nullptr,
                                  xnn_test_runtime_flags(), &runtime2));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime2, 2, external_values2.data()));
  const std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)>
      auto_runtime2(runtime2, xnn_delete_runtime);

  // Check that the workspace grew.
  ASSERT_GT(workspace->size, old_workspace_size);
  ASSERT_NE(runtime2->workspace->data, old_runtime_workspace);
  ASSERT_EQ(runtime1->workspace->data, runtime2->workspace->data);
  ASSERT_EQ(runtime1->workspace->size, runtime2->workspace->size);

  size_t persistent_size = 0;
  for (size_t i = 0; i < runtime1->num_values; i++) {
    const xnn_value* value = &runtime1->values[i];
    if (value->allocation_type == xnn_allocation_type_persistent) {
      ASSERT_EQ((uintptr_t)value->data,
                (uintptr_t)workspace->data + persistent_size);
      persistent_size += round_up_po2(value->size, XNN_EXTRA_BYTES);
    }
  }

  persistent_size = 0;
  for (size_t i = 0; i < runtime2->num_values; i++) {
    const xnn_value* value = &runtime2->values[i];
    if (value->allocation_type == xnn_allocation_type_persistent) {
      ASSERT_EQ((uintptr_t)value->data,
                (uintptr_t)workspace->data + persistent_size);
      persistent_size += round_up_po2(value->size, XNN_EXTRA_BYTES);
    }
  }

  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime1));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime2));
}

TEST(WORKSPACE, persistent_tensors_values_copied_when_workspace_grows) {
  xnn_initialize(/*allocator=*/nullptr);
  xnn_workspace_t workspace = nullptr;
  xnn_create_workspace(&workspace);
  const std::unique_ptr<xnn_workspace, decltype(&xnn_release_workspace)>
      auto_workspace(workspace, xnn_release_workspace);

  const std::array<size_t, 4> small_dims = {2, 2, 2, 3};
  const std::array<size_t, 4> large_dims = {22, 2, 2, 3};

  xnn_subgraph_t subgraph1 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/0,
                                                    /*flags=*/0, &subgraph1));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph1(
      subgraph1, xnn_delete_subgraph);

  {
    // Define subgraph1: external input -> [copy] -> persistent tensor
    uint32_t input_id = XNN_INVALID_VALUE_ID;
    uint32_t persistent_id = XNN_INVALID_VALUE_ID;
    xnn_define_tensor_value(subgraph1, xnn_datatype_fp32, small_dims.size(),
                            small_dims.data(), nullptr, XNN_INVALID_VALUE_ID,
                            XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id);
    ASSERT_NE(XNN_INVALID_VALUE_ID, input_id);
    xnn_define_tensor_value(subgraph1, xnn_datatype_fp32, small_dims.size(),
                            small_dims.data(), nullptr, XNN_INVALID_VALUE_ID,
                            XNN_VALUE_FLAG_PERSISTENT, &persistent_id);
    ASSERT_NE(XNN_INVALID_VALUE_ID, persistent_id);
    ASSERT_EQ(xnn_status_success,
              xnn_define_copy(subgraph1, input_id, persistent_id, /*flags=*/0));
  }

  xnn_subgraph_t subgraph2 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/0,
                                                    /*flags=*/0, &subgraph2));
  const std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)>
      auto_subgraph2(subgraph2, xnn_delete_subgraph);
  {
    // Define subgraph2:
    // persistent tensor (same as subgraph 1) [copy] output
    // persistent tensor (bigger to force workspace growth) [copy] output2
    uint32_t persistent_id = XNN_INVALID_VALUE_ID;
    uint32_t persistent_id2 = XNN_INVALID_VALUE_ID;
    uint32_t out_id = XNN_INVALID_VALUE_ID;
    uint32_t out_id2 = XNN_INVALID_VALUE_ID;
    xnn_define_tensor_value(subgraph2, xnn_datatype_fp32, small_dims.size(),
                            small_dims.data(), nullptr, XNN_INVALID_VALUE_ID,
                            XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id);
    ASSERT_NE(XNN_INVALID_VALUE_ID, out_id);
    xnn_define_tensor_value(subgraph2, xnn_datatype_fp32, large_dims.size(),
                            large_dims.data(), nullptr, XNN_INVALID_VALUE_ID,
                            XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id2);
    ASSERT_NE(XNN_INVALID_VALUE_ID, out_id2);
    xnn_define_tensor_value(subgraph2, xnn_datatype_fp32, small_dims.size(),
                            small_dims.data(), nullptr, XNN_INVALID_VALUE_ID,
                            XNN_VALUE_FLAG_PERSISTENT, &persistent_id);
    ASSERT_NE(XNN_INVALID_VALUE_ID, persistent_id);
    xnn_define_tensor_value(subgraph2, xnn_datatype_fp32, large_dims.size(),
                            large_dims.data(), nullptr, XNN_INVALID_VALUE_ID,
                            XNN_VALUE_FLAG_PERSISTENT, &persistent_id2);
    ASSERT_NE(XNN_INVALID_VALUE_ID, persistent_id2);
    ASSERT_EQ(xnn_status_success,
              xnn_define_copy(subgraph2, persistent_id, out_id, /*flags=*/0));
    ASSERT_EQ(xnn_status_success,
              xnn_define_copy(subgraph2, persistent_id2, out_id2, /*flags=*/0));
  }

  xnn_runtime_t runtime1 = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v4(subgraph1, nullptr, workspace, nullptr,
                                  xnn_test_runtime_flags(), &runtime1));
  const std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)>
      auto_runtime(runtime1, xnn_delete_runtime);
  xnnpack::Buffer<float> expected(2 * 2 * 2 * 3, 3.14f, xnnpack::XnnExtraBytes);
  const std::array<xnn_external_value, 1> external_values = {
      xnn_external_value{0, expected.data()},
  };
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime1, external_values.size(),
                              external_values.data()));

  // Create the same graph but with larger tensors, this will require a larger
  // workspace.
  xnn_runtime_t runtime2 = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v4(subgraph2, nullptr, workspace, nullptr,
                                  xnn_test_runtime_flags(), &runtime2));
  const std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)>
      auto_runtime2(runtime2, xnn_delete_runtime);

  const size_t old_workspace_size = workspace->size;

  ASSERT_GE(old_workspace_size, 0);
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime1));

  // Setup second runtime, this should grow.
  xnnpack::Buffer<float> actual(2 * 2 * 2 * 3, xnnpack::XnnExtraBytes);
  xnnpack::Buffer<float> dummy(22 * 2 * 2 * 3, xnnpack::XnnExtraBytes);
  const std::array<xnn_external_value, 2> external_values2 = {
      xnn_external_value{0, actual.data()},
      xnn_external_value{1, dummy.data()},
  };
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime2, external_values2.size(),
                              external_values2.data()));

  // Check that the workspace grew.
  ASSERT_GT(workspace->size, old_workspace_size);

  // And check that the persistent values are unchanged.
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime2));
  for (size_t i = 0; i < 2 * 2 * 2 * 3; i++) {
    EXPECT_EQ(expected[i], actual[i]);
  }
}
