// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/allocation-type.h"
#include "xnnpack/math.h"
#include "xnnpack/subgraph.h"
#include "replicable_random_device.h"

namespace {
void DefineGraphWithoutInternalTensors(xnn_subgraph_t* subgraph, std::array<size_t, 4> dims)
{
  xnn_create_subgraph(/*external_value_ids=*/0, /*flags=*/0, subgraph);
  uint32_t input_id = XNN_INVALID_VALUE_ID;
  xnn_define_tensor_value(
    *subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, XNN_INVALID_VALUE_ID,
    XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id);
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  xnn_define_tensor_value(
    *subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, XNN_INVALID_VALUE_ID,
    XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id);
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_abs(*subgraph, input_id, output_id, /*flags=*/0));
}

// Helper function to create a subgraph with 1 input, 1 output, and 1 intermediate tensor.
// input -> (abs) -> intermediate -> (hard swish) -> output
// The size of the tensors are all the same, specified by `dims`.
void DefineGraph(xnn_subgraph_t* subgraph, std::array<size_t, 4> dims)
{
  xnn_create_subgraph(/*external_value_ids=*/0, /*flags=*/0, subgraph);
  uint32_t input_id = XNN_INVALID_VALUE_ID;
  xnn_define_tensor_value(
    *subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, XNN_INVALID_VALUE_ID,
    XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id);
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t intermediate_id = XNN_INVALID_VALUE_ID;
  xnn_define_tensor_value(
    *subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0,
    &intermediate_id);
  ASSERT_NE(intermediate_id, XNN_INVALID_VALUE_ID);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  xnn_define_tensor_value(
    *subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, XNN_INVALID_VALUE_ID,
    XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id);
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_abs(*subgraph, input_id, intermediate_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success, xnn_define_hardswish(*subgraph, intermediate_id, output_id, /*flags=*/0));
}

void DefineGraphWithStaticData(xnn_subgraph_t* subgraph, std::array<size_t, 4> dims, const std::vector<float>* static_value)
{
  xnn_create_subgraph(/*external_value_ids=*/0, /*flags=*/0, subgraph);
  uint32_t input_id = XNN_INVALID_VALUE_ID;
  xnn_define_tensor_value(
    *subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, XNN_INVALID_VALUE_ID,
    XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id);
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t static_value_id = XNN_INVALID_VALUE_ID;
  xnn_define_tensor_value(
    *subgraph, xnn_datatype_fp32, dims.size(), dims.data(), static_value->data(), XNN_INVALID_VALUE_ID, /*flags=*/0,
    &static_value_id);
  ASSERT_NE(static_value_id, XNN_INVALID_VALUE_ID);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  xnn_define_tensor_value(
    *subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, XNN_INVALID_VALUE_ID,
    XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id);
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(xnn_status_success,
            xnn_define_add2(*subgraph, -std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::infinity(), input_id,
                            static_value_id, output_id, /*flags=*/0));
}

void DefineGraphWithPersistentTensors(xnn_subgraph_t* subgraph, std::array<size_t, 4> dims)
{
  // (input) -> abs ---(persistent)--> copy ---(persistent)--> hard swish -> (output)
  xnn_create_subgraph(/*external_value_ids=*/0, /*flags=*/0, subgraph);
  uint32_t input_id = XNN_INVALID_VALUE_ID;
  xnn_define_tensor_value(
    *subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, XNN_INVALID_VALUE_ID,
    XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id);
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t intermediate_id = XNN_INVALID_VALUE_ID;
  xnn_define_tensor_value(
    *subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, XNN_INVALID_VALUE_ID, XNN_VALUE_FLAG_PERSISTENT,
    &intermediate_id);
  ASSERT_NE(intermediate_id, XNN_INVALID_VALUE_ID);

  uint32_t intermediate_id2 = XNN_INVALID_VALUE_ID;
  xnn_define_tensor_value(
    *subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, XNN_INVALID_VALUE_ID, XNN_VALUE_FLAG_PERSISTENT,
    &intermediate_id2);
  ASSERT_NE(intermediate_id, XNN_INVALID_VALUE_ID);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  xnn_define_tensor_value(
    *subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, XNN_INVALID_VALUE_ID,
    XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id);
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_abs(*subgraph, input_id, intermediate_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success, xnn_define_copy(*subgraph, intermediate_id, intermediate_id2, /*flags=*/0));
  ASSERT_EQ(xnn_status_success, xnn_define_hardswish(*subgraph, intermediate_id2, output_id, /*flags=*/0));
}

testing::AssertionResult ValueInWorkspace(xnn_value* value, xnn_workspace_t workspace) {
  if ((value->data >= workspace->data) &&
         ((uintptr_t) value->data + value->size) <= ((uintptr_t) workspace->data + workspace->size)) {
    return testing::AssertionSuccess();
  } else {
    return testing::AssertionFailure()
        << "value at " << value->data << " of size " << value->size
        << " is outside of workspace at " << workspace->data << " of size " << workspace->size;
  }
}

testing::AssertionResult Contains(std::vector<xnn_runtime_t> workspace_users, xnn_runtime_t runtime) {
  if (std::find(workspace_users.begin(), workspace_users.end(), runtime) != workspace_users.end()) {
    return testing::AssertionSuccess();
  } else {
    return testing::AssertionFailure() << "runtime " << runtime << " not found in list of workspace users";
  }
}

std::vector<xnn_runtime_t> workspace_user_to_list(xnn_workspace_t workspace)
{
  std::vector<xnn_runtime_t> users;
  for (xnn_runtime_t rt = workspace->first_user; rt != nullptr; rt = rt->next_workspace_user) {
    users.push_back(rt);
  }
  return users;
}
}  // namespace

TEST(WORKSPACE, static_data_not_moved_does_not_segv)
{
  std::array<size_t, 4> dims = {2, 20, 20, 3};
  size_t num_elements = dims[0] * dims[1] * dims[2] * dims[3];

  xnn_initialize(/*allocator=*/nullptr);
  xnn_workspace_t workspace = nullptr;
  xnn_create_workspace(&workspace);
  std::unique_ptr<xnn_workspace, decltype(&xnn_release_workspace)> auto_workspace(workspace, xnn_release_workspace);

  // Create a graph that with static data.
  xnn_subgraph_t subgraph1 = nullptr;
  std::vector<float> static_data = std::vector<float>(num_elements, 1.0f);
  DefineGraphWithStaticData(&subgraph1, dims, &static_data);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph1(subgraph1, xnn_delete_subgraph);
  xnn_runtime_t runtime1 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph1, nullptr, workspace, nullptr, 0, &runtime1));
  const std::array<xnn_external_value, 2> external_values1 = {
    xnn_external_value{0, static_data.data()},
    xnn_external_value{2, static_data.data()},
  };
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime1, 2, external_values1.data()));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime1(runtime1, xnn_delete_runtime);

  // The workspace remains at size 0, without any memory allocated, since we don't have any internal tensors.
  size_t old_workspace_size = workspace->size;
  ASSERT_EQ(old_workspace_size, 0);
  void* old_runtime_workspace = runtime1->workspace->data;
  ASSERT_EQ(old_runtime_workspace, nullptr);

  // Then create a graph that has internal tensors, we will need to resize the workspace.
  xnn_subgraph_t subgraph2 = nullptr;
  DefineGraph(&subgraph2, dims);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph2(subgraph2, xnn_delete_subgraph);
  xnn_runtime_t runtime2 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph2, nullptr, workspace, nullptr, 0, &runtime2));
  const std::array<xnn_external_value, 2> external_values2 = {
    xnn_external_value{0, static_data.data()},
    xnn_external_value{2, static_data.data()},
  };
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime2, 2, external_values2.data()));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime2(runtime2, xnn_delete_runtime);

  // Check that the workspace grew.
  ASSERT_GE(workspace->size, num_elements * sizeof(float));
  ASSERT_NE(runtime2->workspace->data, nullptr);

  // Try to access all the values and ensure that we don't segfault.
  for (size_t i = 0; i < runtime1->num_values; i++) {
    xnn_value* value = &runtime1->values[i];
    if (value->allocation_type == xnn_allocation_type_external) {
      continue;
    }
    ASSERT_GT(value->size, 0);
    char access = *((char *)value->data);
    (void) access;
  }

  for (size_t i = 0; i < runtime2->num_values; i++) {
    xnn_value* value = &runtime2->values[i];
    if (value->allocation_type == xnn_allocation_type_external) {
      continue;
    }
    ASSERT_GT(value->size, 0);
    char access = *((char *)value->data);
    (void) access;
  }

  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime1));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime2));
}

TEST(WORKSPACE, workspace_no_growth)
{
  xnn_initialize(/*allocator=*/nullptr);
  xnn_workspace_t workspace = nullptr;
  xnn_create_workspace(&workspace);
  std::unique_ptr<xnn_workspace, decltype(&xnn_release_workspace)> auto_workspace(workspace, xnn_release_workspace);

  std::array<size_t, 4> dims = {2, 20, 20, 3};
  std::vector<float> dummy_data(2 * 20 * 20 * 3);
  const std::array<xnn_external_value, 2> external_values = {
    xnn_external_value{0, dummy_data.data()},
    xnn_external_value{2, dummy_data.data()},
  };

  xnn_subgraph_t subgraph1 = nullptr;
  DefineGraph(&subgraph1, dims);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph1(subgraph1, xnn_delete_subgraph);

  xnn_runtime_t runtime1 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph1, nullptr, workspace, nullptr, 0, &runtime1));
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime1, 2, external_values.data()));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime1(runtime1, xnn_delete_runtime);

  size_t old_workspace_size = workspace->size;
  ASSERT_GE(old_workspace_size, 0);
  void* old_runtime_workspace = runtime1->workspace->data;
  ASSERT_NE(old_runtime_workspace, nullptr);

  // Create the same graph again with a different runtime that shares the workspace.
  xnn_subgraph_t subgraph2 = nullptr;
  DefineGraph(&subgraph2, dims);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph2(subgraph2, xnn_delete_subgraph);

  xnn_runtime_t runtime2 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph2, nullptr, workspace, nullptr, 0, &runtime2));
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime2, 2, external_values.data()));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime2(runtime2, xnn_delete_runtime);

  // Check that the workspace did not grow.
  ASSERT_EQ(workspace->size, old_workspace_size);
  // Check that runtime 2 uses the same workspace.
  ASSERT_EQ(runtime2->workspace->data, old_runtime_workspace);

  ASSERT_EQ(runtime1->num_values, runtime2->num_values);
  for (size_t i = 0; i < runtime1->num_values; i++) {
    xnn_value* value1 = &runtime1->values[i];
    if (value1->allocation_type != xnn_allocation_type_workspace) {
      continue;
    }
    ASSERT_TRUE(ValueInWorkspace(value1, runtime1->workspace));
    xnn_value* value2 = &runtime2->values[i];
    ASSERT_TRUE(ValueInWorkspace(value2, runtime2->workspace));
  }

  std::vector<xnn_runtime_t> workspace_users = workspace_user_to_list(workspace);
  ASSERT_EQ(workspace_users.size(), 2);
  ASSERT_TRUE(Contains(workspace_users, runtime1));
  ASSERT_TRUE(Contains(workspace_users, runtime2));
  ASSERT_EQ(workspace->ref_count, 3);

  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime1));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime2));
}

TEST(WORKSPACE, workspace_grow)
{
  xnn_initialize(/*allocator=*/nullptr);
  xnn_workspace_t workspace = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_workspace(&workspace));
  std::unique_ptr<xnn_workspace, decltype(&xnn_release_workspace)> auto_workspace(workspace, xnn_release_workspace);

  std::array<size_t, 4> dims1 = {2, 20, 20, 3};
  std::vector<float> dummy_data(2 * 20 * 20 * 3 + XNN_EXTRA_BYTES / sizeof(float));
  const std::array<xnn_external_value, 2> external_values = {
    xnn_external_value{0, dummy_data.data()},
    xnn_external_value{2, dummy_data.data()},
  };
  std::vector<float> dummy_data2(2 * 20 * 20 * 3 * 16 + XNN_EXTRA_BYTES / sizeof(float));
  const std::array<xnn_external_value, 2> external_values2 = {
    xnn_external_value{0, dummy_data2.data()},
    xnn_external_value{2, dummy_data2.data()},
  };

  xnn_subgraph_t subgraph1 = nullptr;
  DefineGraph(&subgraph1, dims1);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph1(subgraph1, xnn_delete_subgraph);

  xnn_runtime_t runtime1 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph1, nullptr, workspace, nullptr, 0, &runtime1));

  // No workspace allocated yet, it should be only allocated on setup.
  ASSERT_EQ(workspace->size, 0);
  ASSERT_EQ(runtime1->workspace->data, nullptr);

  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime1, 2, external_values.data()));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime1(runtime1, xnn_delete_runtime);

  size_t old_workspace_size = workspace->size;
  ASSERT_GE(old_workspace_size, 0);
  void* old_runtime_workspace = runtime1->workspace->data;
  ASSERT_NE(old_runtime_workspace, nullptr);

  std::array<size_t, 4> dims2 = dims1;
  // Create the same graph but with larger tensors, this will require a larger workspace.
  std::transform(dims2.begin(), dims2.end(), dims2.begin(), [](size_t i) { return i * 2; });
  xnn_subgraph_t subgraph2 = nullptr;
  DefineGraph(&subgraph2, dims2);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph2(subgraph2, xnn_delete_subgraph);

  xnn_runtime_t runtime2 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph2, nullptr, workspace, nullptr, 0, &runtime2));
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime2, 2, external_values2.data()));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime2(runtime2, xnn_delete_runtime);

  // Check that the workspace grew.
  ASSERT_GT(workspace->size, old_workspace_size);
  // We free first, then allocate memory, so whether the workspace data changes depends on the system. Asserting that
  // the data pointers are different will result in a flaky test.
  // Check that runtime1's workspace has been updated as well.
  ASSERT_EQ(runtime1->workspace->data, runtime2->workspace->data);
  ASSERT_EQ(runtime1->workspace->size, runtime2->workspace->size);

  // Check that both runtime's value pointers are within range.
  for (size_t i = 0; i < runtime1->num_values; i++) {
    xnn_value* value = &runtime1->values[i];
    if (value->allocation_type != xnn_allocation_type_workspace) {
      continue;
    }
    ASSERT_TRUE(ValueInWorkspace(value, runtime1->workspace));
  }
  for (size_t i = 0; i < runtime2->num_values; i++) {
    xnn_value* value = &runtime2->values[i];
    if (value->allocation_type != xnn_allocation_type_workspace) {
      continue;
    }
    ASSERT_TRUE(ValueInWorkspace(value, runtime2->workspace));
  }

  std::vector<xnn_runtime_t> workspace_users = workspace_user_to_list(workspace);
  ASSERT_EQ(workspace_users.size(), 2);
  ASSERT_TRUE(Contains(workspace_users, runtime1));
  ASSERT_TRUE(Contains(workspace_users, runtime2));
  ASSERT_EQ(workspace->ref_count, 3);

  xnn_invoke_runtime(runtime1);
  xnn_invoke_runtime(runtime2);
}

TEST(WORKSPACE, workspace_runtime_delete_head_runtime_first)
{
  xnn_initialize(/*allocator=*/nullptr);
  xnn_workspace_t workspace = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_workspace(&workspace));
  std::unique_ptr<xnn_workspace, decltype(&xnn_release_workspace)> auto_workspace(workspace, xnn_release_workspace);

  const std::array<size_t, 4> dims = {2, 20, 20, 3};
  std::vector<float> dummy_data(2 * 20 * 20 * 3 + XNN_EXTRA_BYTES / sizeof(float));
  const std::array<xnn_external_value, 2> external_values = {
    xnn_external_value{0, dummy_data.data()},
    xnn_external_value{2, dummy_data.data()},
  };

  xnn_subgraph_t subgraph1 = nullptr;
  DefineGraph(&subgraph1, dims);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph1(subgraph1, xnn_delete_subgraph);

  xnn_runtime_t runtime1 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph1, nullptr, workspace, nullptr, 0, &runtime1));
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime1, 2, external_values.data()));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime1(runtime1, xnn_delete_runtime);

  xnn_subgraph_t subgraph2 = nullptr;
  DefineGraph(&subgraph2, dims);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph2(subgraph2, xnn_delete_subgraph);

  xnn_runtime_t runtime2 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph2, nullptr, workspace, nullptr, 0, &runtime2));
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime2, 2, external_values.data()));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime2(runtime2, xnn_delete_runtime);

  ASSERT_EQ(workspace->first_user, runtime2);
  ASSERT_EQ(runtime2->next_workspace_user, runtime1);
  ASSERT_EQ(runtime1->next_workspace_user, nullptr);

  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime1));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime2));

  ASSERT_EQ(workspace->ref_count, 3);
  xnn_delete_runtime(auto_runtime2.release());
  ASSERT_EQ(workspace->first_user, runtime1);
  ASSERT_EQ(runtime1->next_workspace_user, nullptr);
  ASSERT_EQ(workspace->ref_count, 2);

  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime1));

  xnn_delete_runtime(auto_runtime1.release());
  ASSERT_EQ(workspace->first_user, nullptr);
  ASSERT_EQ(workspace->ref_count, 1);
}

TEST(WORKSPACE, workspace_runtime_delete_tail_runtime_first)
{
  xnn_initialize(/*allocator=*/nullptr);
  xnn_workspace_t workspace = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_workspace(&workspace));
  std::unique_ptr<xnn_workspace, decltype(&xnn_release_workspace)> auto_workspace(workspace, xnn_release_workspace);

  std::array<size_t, 4> dims = {2, 20, 20, 3};
  std::vector<float> dummy_data(2 * 20 * 20 * 3 + XNN_EXTRA_BYTES / sizeof(float));
  const std::array<xnn_external_value, 2> external_values = {
    xnn_external_value{0, dummy_data.data()},
    xnn_external_value{2, dummy_data.data()},
  };

  xnn_subgraph_t subgraph1 = nullptr;
  DefineGraph(&subgraph1, dims);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph1(subgraph1, xnn_delete_subgraph);

  xnn_runtime_t runtime1 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph1, nullptr, workspace, nullptr, 0, &runtime1));
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime1, 2, external_values.data()));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime1(runtime1, xnn_delete_runtime);

  xnn_subgraph_t subgraph2 = nullptr;
  DefineGraph(&subgraph2, dims);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph2(subgraph2, xnn_delete_subgraph);

  xnn_runtime_t runtime2 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph2, nullptr, workspace, nullptr, 0, &runtime2));
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime2, 2, external_values.data()));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime2(runtime2, xnn_delete_runtime);

  ASSERT_EQ(workspace->first_user, runtime2);
  ASSERT_EQ(runtime2->next_workspace_user, runtime1);
  ASSERT_EQ(runtime1->next_workspace_user, nullptr);

  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime1));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime2));

  ASSERT_EQ(workspace->ref_count, 3);
  xnn_delete_runtime(auto_runtime1.release());

  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime2));

  ASSERT_EQ(workspace->first_user, runtime2);
  ASSERT_EQ(runtime2->next_workspace_user, nullptr);
  ASSERT_EQ(workspace->ref_count, 2);

  xnn_delete_runtime(auto_runtime2.release());
  ASSERT_EQ(workspace->first_user, nullptr);
  ASSERT_EQ(workspace->ref_count, 1);
}

TEST(WORKSPACE, workspace_runtime_delete_middle_runtime_first)
{
  xnn_initialize(/*allocator=*/nullptr);
  xnn_workspace_t workspace = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_workspace(&workspace));
  std::unique_ptr<xnn_workspace, decltype(&xnn_release_workspace)> auto_workspace(workspace, xnn_release_workspace);

  std::array<size_t, 4> dims = {2, 20, 20, 3};
  std::vector<float> dummy_data(2 * 20 * 20 * 3 + XNN_EXTRA_BYTES / sizeof(float));
  const std::array<xnn_external_value, 2> external_values = {
    xnn_external_value{0, dummy_data.data()},
    xnn_external_value{2, dummy_data.data()},
  };

  xnn_subgraph_t subgraph1 = nullptr;
  DefineGraph(&subgraph1, dims);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph1(subgraph1, xnn_delete_subgraph);

  xnn_runtime_t runtime1 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph1, nullptr, workspace, nullptr, 0, &runtime1));
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime1, 2, external_values.data()));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime1(runtime1, xnn_delete_runtime);

  xnn_subgraph_t subgraph2 = nullptr;
  DefineGraph(&subgraph2, dims);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph2(subgraph2, xnn_delete_subgraph);

  xnn_runtime_t runtime2 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph2, nullptr, workspace, nullptr, 0, &runtime2));
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime2, 2, external_values.data()));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime2(runtime2, xnn_delete_runtime);

  xnn_subgraph_t subgraph3 = nullptr;
  DefineGraph(&subgraph3, dims);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph3(subgraph3, xnn_delete_subgraph);

  xnn_runtime_t runtime3 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph3, nullptr, workspace, nullptr, 0, &runtime3));
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime3, 2, external_values.data()));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime3(runtime3, xnn_delete_runtime);

  ASSERT_EQ(workspace->first_user, runtime3);
  ASSERT_EQ(runtime3->next_workspace_user, runtime2);
  ASSERT_EQ(runtime2->next_workspace_user, runtime1);
  ASSERT_EQ(runtime1->next_workspace_user, nullptr);

  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime1));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime2));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime3));

  ASSERT_EQ(workspace->ref_count, 4);
  xnn_delete_runtime(auto_runtime2.release());

  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime1));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime3));

  ASSERT_EQ(workspace->first_user, runtime3);
  ASSERT_EQ(runtime3->next_workspace_user, runtime1);
  ASSERT_EQ(runtime1->next_workspace_user, nullptr);
  ASSERT_EQ(workspace->ref_count, 3);

  xnn_delete_runtime(auto_runtime3.release());
  ASSERT_EQ(workspace->first_user, runtime1);
  ASSERT_EQ(runtime1->next_workspace_user, nullptr);
  ASSERT_EQ(workspace->ref_count, 2);

  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime1));

  xnn_delete_runtime(auto_runtime1.release());
  ASSERT_EQ(workspace->first_user, nullptr);
  ASSERT_EQ(workspace->ref_count, 1);
}

TEST(WORKSPACE, zero_sized_workspace_for_graph_without_internal_tensors)
{
  xnn_initialize(/*allocator=*/nullptr);
  xnn_workspace_t workspace = nullptr;
  xnn_create_workspace(&workspace);
  std::unique_ptr<xnn_workspace, decltype(&xnn_release_workspace)> auto_workspace(workspace, xnn_release_workspace);

  std::array<size_t, 4> dims = {2, 20, 20, 3};
  std::vector<float> dummy_data(2 * 20 * 20 * 3 + XNN_EXTRA_BYTES / sizeof(float));
  const std::array<xnn_external_value, 2> external_values = {
    xnn_external_value{0, dummy_data.data()},
    xnn_external_value{1, dummy_data.data()},
  };

  xnn_subgraph_t subgraph = nullptr;
  DefineGraphWithoutInternalTensors(&subgraph, dims);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph, nullptr, workspace, nullptr, 0, &runtime));
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, 2, external_values.data()));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);

  ASSERT_EQ(0, workspace->size);
  ASSERT_EQ(nullptr, workspace->data);
  ASSERT_EQ(std::vector<xnn_runtime_t>({runtime}), workspace_user_to_list(workspace));
  ASSERT_EQ(workspace->ref_count, 2);

  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));
}

TEST(WORKSPACE, persistent_tensors_allocated_at_start_of_workspace)
{
  // Persistent tensors allocated at the start.
  xnn_initialize(/*allocator=*/nullptr);
  xnn_workspace_t workspace = nullptr;
  xnn_create_workspace(&workspace);
  const std::unique_ptr<xnn_workspace, decltype(&xnn_release_workspace)> auto_workspace(
    workspace, xnn_release_workspace);

  const std::array<size_t, 4> dims = {2, 20, 20, 3};
  std::vector<float> dummy_data(2 * 20 * 20 * 3 + XNN_EXTRA_BYTES / sizeof(float));
  const std::array<xnn_external_value, 2> external_values = {
    xnn_external_value{0, dummy_data.data()},
    xnn_external_value{3, dummy_data.data()},
  };

  xnn_subgraph_t subgraph = nullptr;
  DefineGraphWithPersistentTensors(&subgraph, dims);
  const std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph, nullptr, workspace, nullptr, 0, &runtime));
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, 2, external_values.data()));
  const std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);

  const size_t old_workspace_size = workspace->size;
  ASSERT_GE(old_workspace_size, 0);
  const void* old_runtime_workspace = runtime->workspace->data;
  ASSERT_NE(old_runtime_workspace, nullptr);

  size_t persistent_size = 0;
  for (size_t i = 0; i < runtime->num_values; i++) {
    const xnn_value* value = &runtime->values[i];
    if (value->allocation_type == xnn_allocation_type_persistent) {
      ASSERT_EQ((uintptr_t) value->data, (uintptr_t) workspace->data + persistent_size);
      persistent_size += round_up_po2(value->size, XNN_EXTRA_BYTES);
    }
  }

  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));
}

TEST(WORKSPACE, persistent_tensors_updated_correct_when_workspace_grows)
{
  // Persistent tensors allocated at the start.
  xnn_initialize(/*allocator=*/nullptr);
  xnn_workspace_t workspace = nullptr;
  xnn_create_workspace(&workspace);
  const std::unique_ptr<xnn_workspace, decltype(&xnn_release_workspace)> auto_workspace(
    workspace, xnn_release_workspace);

  const std::array<size_t, 4> dims1 = {2, 20, 20, 3};
  std::vector<float> dummy_data(2 * 20 * 20 * 3 + XNN_EXTRA_BYTES / sizeof(float));
  const std::array<xnn_external_value, 2> external_values = {
    xnn_external_value{0, dummy_data.data()},
    xnn_external_value{3, dummy_data.data()},
  };
  std::vector<float> dummy_data2(2 * 20 * 20 * 3 * 16 + XNN_EXTRA_BYTES / sizeof(float));
  const std::array<xnn_external_value, 2> external_values2 = {
    xnn_external_value{0, dummy_data2.data()},
    xnn_external_value{3, dummy_data2.data()},
  };

  xnn_subgraph_t subgraph1 = nullptr;
  DefineGraphWithPersistentTensors(&subgraph1, dims1);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph1(subgraph1, xnn_delete_subgraph);

  xnn_runtime_t runtime1 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph1, nullptr, workspace, nullptr, 0, &runtime1));
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime1, 2, external_values.data()));
  const std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime1, xnn_delete_runtime);

  const size_t old_workspace_size = workspace->size;
  ASSERT_GE(old_workspace_size, 0);
  const void* old_runtime_workspace = runtime1->workspace->data;
  ASSERT_NE(old_runtime_workspace, nullptr);

  std::array<size_t, 4> dims2 = dims1;
  // Create the same graph but with larger tensors, this will require a larger workspace.
  std::transform(dims2.begin(), dims2.end(), dims2.begin(), [](size_t i) { return i * 2; });
  xnn_subgraph_t subgraph2 = nullptr;
  DefineGraphWithPersistentTensors(&subgraph2, dims2);
  const std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph2(subgraph2, xnn_delete_subgraph);
  xnn_runtime_t runtime2 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph2, nullptr, workspace, nullptr, 0, &runtime2));
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime2, 2, external_values2.data()));
  const std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime2(runtime2, xnn_delete_runtime);

  // Check that the workspace grew.
  ASSERT_GT(workspace->size, old_workspace_size);
  ASSERT_NE(runtime2->workspace->data, old_runtime_workspace);
  ASSERT_EQ(runtime1->workspace->data, runtime2->workspace->data);
  ASSERT_EQ(runtime1->workspace->size, runtime2->workspace->size);

  size_t persistent_size = 0;
  for (size_t i = 0; i < runtime1->num_values; i++) {
    const xnn_value* value = &runtime1->values[i];
    if (value->allocation_type == xnn_allocation_type_persistent) {
      ASSERT_EQ((uintptr_t) value->data, (uintptr_t) workspace->data + persistent_size);
      persistent_size += round_up_po2(value->size, XNN_EXTRA_BYTES);
    }
  }

  persistent_size = 0;
  for (size_t i = 0; i < runtime2->num_values; i++) {
    const xnn_value* value = &runtime2->values[i];
    if (value->allocation_type == xnn_allocation_type_persistent) {
      ASSERT_EQ((uintptr_t) value->data, (uintptr_t) workspace->data + persistent_size);
      persistent_size += round_up_po2(value->size, XNN_EXTRA_BYTES);
    }
  }

  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime1));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime2));
}

TEST(WORKSPACE, persistent_tensors_values_copied_when_workspace_grows)
{
  xnn_initialize(/*allocator=*/nullptr);
  xnn_workspace_t workspace = nullptr;
  xnn_create_workspace(&workspace);
  const std::unique_ptr<xnn_workspace, decltype(&xnn_release_workspace)> auto_workspace(
    workspace, xnn_release_workspace);

  const std::array<size_t, 4> small_dims = {2, 2, 2, 3};
  const std::array<size_t, 4> large_dims = {22, 2, 2, 3};

  xnn_subgraph_t subgraph1 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/0, /*flags=*/0, &subgraph1));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph1(subgraph1, xnn_delete_subgraph);

  {
    // Define subgraph1: external input -> [copy] -> persistent tensor
    uint32_t input_id = XNN_INVALID_VALUE_ID;
    uint32_t persistent_id = XNN_INVALID_VALUE_ID;
    xnn_define_tensor_value(
        subgraph1, xnn_datatype_fp32, small_dims.size(), small_dims.data(), nullptr, XNN_INVALID_VALUE_ID,
        XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id);
    ASSERT_NE(XNN_INVALID_VALUE_ID, input_id);
    xnn_define_tensor_value(
        subgraph1, xnn_datatype_fp32, small_dims.size(), small_dims.data(), nullptr, XNN_INVALID_VALUE_ID,
        XNN_VALUE_FLAG_PERSISTENT, &persistent_id);
    ASSERT_NE(XNN_INVALID_VALUE_ID, persistent_id);
    ASSERT_EQ(xnn_status_success, xnn_define_copy(subgraph1, input_id, persistent_id, /*flags=*/0));
  }

  xnn_subgraph_t subgraph2 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/0, /*flags=*/0, &subgraph2));
  const std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph2(subgraph2, xnn_delete_subgraph);
  {
    // Define subgraph2:
    // persistent tensor (same as subgraph 1) [copy] output
    // persistent tensor (bigger to force workspace growth) [copy] output2
    uint32_t persistent_id = XNN_INVALID_VALUE_ID;
    uint32_t persistent_id2 = XNN_INVALID_VALUE_ID;
    uint32_t out_id = XNN_INVALID_VALUE_ID;
    uint32_t out_id2 = XNN_INVALID_VALUE_ID;
    xnn_define_tensor_value(
        subgraph2, xnn_datatype_fp32, small_dims.size(), small_dims.data(), nullptr, XNN_INVALID_VALUE_ID,
        XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id);
    ASSERT_NE(XNN_INVALID_VALUE_ID, out_id);
    xnn_define_tensor_value(
        subgraph2, xnn_datatype_fp32, large_dims.size(), large_dims.data(), nullptr, XNN_INVALID_VALUE_ID,
        XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id2);
    ASSERT_NE(XNN_INVALID_VALUE_ID, out_id2);
    xnn_define_tensor_value(
        subgraph2, xnn_datatype_fp32, small_dims.size(), small_dims.data(), nullptr, XNN_INVALID_VALUE_ID,
        XNN_VALUE_FLAG_PERSISTENT, &persistent_id);
    ASSERT_NE(XNN_INVALID_VALUE_ID, persistent_id);
    xnn_define_tensor_value(
        subgraph2, xnn_datatype_fp32, large_dims.size(), large_dims.data(), nullptr, XNN_INVALID_VALUE_ID,
        XNN_VALUE_FLAG_PERSISTENT, &persistent_id2);
    ASSERT_NE(XNN_INVALID_VALUE_ID, persistent_id2);
    ASSERT_EQ(xnn_status_success, xnn_define_copy(subgraph2, persistent_id, out_id, /*flags=*/0));
    ASSERT_EQ(xnn_status_success, xnn_define_copy(subgraph2, persistent_id2, out_id2, /*flags=*/0));
  }

  xnn_runtime_t runtime1 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph1, nullptr, workspace, nullptr, 0, &runtime1));
  const std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime1, xnn_delete_runtime);
  std::vector<float> expected(2 * 2 * 2 * 3 + XNN_EXTRA_BYTES / sizeof(float), 3.14f);
  const std::array<xnn_external_value, 1> external_values = {
    xnn_external_value{0, expected.data()},
  };
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime1, external_values.size(), external_values.data()));

  // Create the same graph but with larger tensors, this will require a larger workspace.
  xnn_runtime_t runtime2 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph2, nullptr, workspace, nullptr, 0, &runtime2));
  const std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime2(runtime2, xnn_delete_runtime);

  const size_t old_workspace_size = workspace->size;

  ASSERT_GE(old_workspace_size, 0);
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime1));

  // Setup second runtime, this should grow.
  std::vector<float> actual(2 * 2 * 2 * 3 + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> dummy(22 * 2 * 2 * 3 + XNN_EXTRA_BYTES / sizeof(float));
  const std::array<xnn_external_value, 2> external_values2 = {
    xnn_external_value{0, actual.data()},
    xnn_external_value{1, dummy.data()},
  };
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime2, external_values2.size(), external_values2.data()));

  // Check that the workspace grew.
  ASSERT_GT(workspace->size, old_workspace_size);

  // And check that the persistent values are unchanged.
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime2));
  for (size_t i = 0; i < 2 * 2 * 2 * 3; i++) {
    EXPECT_EQ(expected[i], actual[i]);
  }
}

TEST(WORKSPACE, internally_allocated_dynamic_quantization_parameters)
{
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  uint32_t input_id = XNN_INVALID_NODE_ID;
  xnnpack::ReplicableRandomDevice rng;
  auto scalerng = std::bind(std::uniform_real_distribution<float>(0.5f, 2.f), std::ref(rng));
  const size_t batch_size = 3;
  const size_t input_channels = 4;
  const size_t output_channels = 5;
  std::vector<size_t> input_dims{batch_size, input_channels};
  std::vector<size_t> kernel_dims{output_channels, input_channels};
  std::vector<size_t> bias_dims{output_channels};
  std::vector<float> input(batch_size * input_channels + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> subgraph_output(batch_size * output_channels);
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));
  std::vector<xnn_dynamic_quantization_params> quantization_params(3 + XNN_EXTRA_QUANTIZATION_PARAMS);
  std::vector<float> kernel_scale(output_channels);
  std::vector<float> bias(output_channels);
  std::vector<int8_t> kernel(input_channels * output_channels);
  std::vector<size_t> output_dims{batch_size, output_channels};
  std::generate(kernel_scale.begin(), kernel_scale.end(), std::ref(scalerng));

  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = std::numeric_limits<float>::infinity();

  // Call subgraph API.
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(), nullptr, /*external_id=*/0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t dq_quantized_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_dynamically_quantized_tensor_value(
                          subgraph, xnn_datatype_qdint8, input_dims.size(), /*num_nonbatch_dims=*/1, input_dims.data(),
                          XNN_INVALID_VALUE_ID, /*flags=*/0, &dq_quantized_id));
  ASSERT_NE(dq_quantized_id, XNN_INVALID_NODE_ID);
  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_channelwise_quantized_tensor_value(
                          subgraph, xnn_datatype_qcint8, kernel_scale.data(), kernel_dims.size(), /*channel_dim=*/0,
                          kernel_dims.data(), kernel.data(), /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, bias_dims.size(), bias_dims.data(), bias.data(),
                          /*external_id=*/2, /*flags=*/0, &bias_id));
  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, /*external_id=*/3,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_define_convert(subgraph, input_id, dq_quantized_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success, xnn_define_fully_connected(subgraph, output_min, output_max, dq_quantized_id, kernel_id, bias_id, output_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));

  size_t dq_tensors = 0;
  for (size_t i = 0; i < runtime->num_values; i++) {
    const xnn_value* value = &runtime->values[i];
    if (value->datatype == xnn_datatype_qdint8) {
      ++dq_tensors;
      ASSERT_NE(value->quantization.dynamic_params, nullptr);
    }
  }
  ASSERT_EQ(dq_tensors, 1);
}
