// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

#include <xnnpack.h>
#include <xnnpack/math.h>
#include <xnnpack/subgraph.h>

#include <gtest/gtest.h>

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

testing::AssertionResult BlobInWorkspace(xnn_blob* blob, xnn_workspace_t workspace) {
  if ((blob->data >= workspace->data) &&
         ((uintptr_t) blob->data + blob->size) <= ((uintptr_t) workspace->data + workspace->size)) {
    return testing::AssertionSuccess();
  } else {
    return testing::AssertionFailure()
        << "blob at " << blob->data << " of size " << blob->size
        << "is outside of workspace at " << workspace->data << " of size " << workspace->size;
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
  for (xnn_runtime_t rt = workspace->first_user; rt != NULL; rt = rt->next_workspace_user) {
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
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime2(runtime2, xnn_delete_runtime);

  // Check that the workspace grew.
  ASSERT_GE(workspace->size, num_elements * sizeof(float));
  ASSERT_NE(runtime2->workspace->data, nullptr);

  // Try to access all the blobs and ensure that we don't segfault.
  for (size_t i = 0; i < runtime1->num_blobs; i++) {
    xnn_blob* blob = &runtime1->blobs[i];
    if (blob->allocation_type == xnn_allocation_type_external) {
      continue;
    }
    ASSERT_GT(blob->size, 0);
    char access = *((char *)blob->data);
    (void) access;
  }

  for (size_t i = 0; i < runtime2->num_blobs; i++) {
    xnn_blob* blob = &runtime2->blobs[i];
    if (blob->allocation_type == xnn_allocation_type_external) {
      continue;
    }
    ASSERT_GT(blob->size, 0);
    char access = *((char *)blob->data);
    (void) access;
  }
}

TEST(WORKSPACE, workspace_no_growth)
{
  xnn_initialize(/*allocator=*/nullptr);
  xnn_workspace_t workspace = nullptr;
  xnn_create_workspace(&workspace);
  std::unique_ptr<xnn_workspace, decltype(&xnn_release_workspace)> auto_workspace(workspace, xnn_release_workspace);

  std::array<size_t, 4> dims = {2, 20, 20, 3};

  xnn_subgraph_t subgraph1 = nullptr;
  DefineGraph(&subgraph1, dims);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph1(subgraph1, xnn_delete_subgraph);

  xnn_runtime_t runtime1 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph1, nullptr, workspace, nullptr, 0, &runtime1));
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
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime2(runtime2, xnn_delete_runtime);

  // Check that the workspace did not grow.
  ASSERT_EQ(workspace->size, old_workspace_size);
  // Check that runtime 2 uses the same workspace.
  ASSERT_EQ(runtime2->workspace->data, old_runtime_workspace);

  ASSERT_EQ(runtime1->num_blobs, runtime2->num_blobs);
  for (size_t i = 0; i < runtime1->num_blobs; i++) {
    xnn_blob* blob1 = &runtime1->blobs[i];
    if (blob1->allocation_type != xnn_allocation_type_workspace) {
      continue;
    }
    ASSERT_TRUE(BlobInWorkspace(blob1, runtime1->workspace));
    xnn_blob* blob2 = &runtime2->blobs[i];
    ASSERT_TRUE(BlobInWorkspace(blob2, runtime2->workspace));
  }

  std::vector<xnn_runtime_t> workspace_users = workspace_user_to_list(workspace);
  ASSERT_EQ(workspace_users.size(), 2);
  ASSERT_TRUE(Contains(workspace_users, runtime1));
  ASSERT_TRUE(Contains(workspace_users, runtime2));
  ASSERT_EQ(workspace->ref_count, 3);
}

TEST(WORKSPACE, workspace_grow)
{
  xnn_initialize(/*allocator=*/nullptr);
  xnn_workspace_t workspace = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_workspace(&workspace));
  std::unique_ptr<xnn_workspace, decltype(&xnn_release_workspace)> auto_workspace(workspace, xnn_release_workspace);

  std::array<size_t, 4> dims1 = {2, 20, 20, 3};

  xnn_subgraph_t subgraph1 = nullptr;
  DefineGraph(&subgraph1, dims1);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph1(subgraph1, xnn_delete_subgraph);

  xnn_runtime_t runtime1 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph1, nullptr, workspace, nullptr, 0, &runtime1));
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
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime2(runtime2, xnn_delete_runtime);

  // Check that the workspace grew.
  ASSERT_GT(workspace->size, old_workspace_size);
  // Check that the workspace is different.
  ASSERT_NE(runtime2->workspace->data, old_runtime_workspace);
  // Check that runtime1's workspace has been updated as well.
  ASSERT_EQ(runtime1->workspace->data, runtime2->workspace->data);
  ASSERT_EQ(runtime1->workspace->size, runtime2->workspace->size);

  // Check that both runtime's blob pointers are within range.
  for (size_t i = 0; i < runtime1->num_blobs; i++) {
    xnn_blob* blob = &runtime1->blobs[i];
    if (blob->allocation_type != xnn_allocation_type_workspace) {
      continue;
    }
    ASSERT_TRUE(BlobInWorkspace(blob, runtime1->workspace));
  }
  for (size_t i = 0; i < runtime2->num_blobs; i++) {
    xnn_blob* blob = &runtime2->blobs[i];
    if (blob->allocation_type != xnn_allocation_type_workspace) {
      continue;
    }
    ASSERT_TRUE(BlobInWorkspace(blob, runtime2->workspace));
  }

  std::vector<xnn_runtime_t> workspace_users = workspace_user_to_list(workspace);
  ASSERT_EQ(workspace_users.size(), 2);
  ASSERT_TRUE(Contains(workspace_users, runtime1));
  ASSERT_TRUE(Contains(workspace_users, runtime2));
  ASSERT_EQ(workspace->ref_count, 3);
}

TEST(WORKSPACE, workspace_runtime_delete_head_runtime_first)
{
  xnn_initialize(/*allocator=*/nullptr);
  xnn_workspace_t workspace = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_workspace(&workspace));
  std::unique_ptr<xnn_workspace, decltype(&xnn_release_workspace)> auto_workspace(workspace, xnn_release_workspace);

  const std::array<size_t, 4> dims = {2, 20, 20, 3};

  xnn_subgraph_t subgraph1 = nullptr;
  DefineGraph(&subgraph1, dims);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph1(subgraph1, xnn_delete_subgraph);

  xnn_runtime_t runtime1 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph1, nullptr, workspace, nullptr, 0, &runtime1));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime1(runtime1, xnn_delete_runtime);

  xnn_subgraph_t subgraph2 = nullptr;
  DefineGraph(&subgraph2, dims);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph2(subgraph2, xnn_delete_subgraph);

  xnn_runtime_t runtime2 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph2, nullptr, workspace, nullptr, 0, &runtime2));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime2(runtime2, xnn_delete_runtime);

  ASSERT_EQ(workspace->first_user, runtime2);
  ASSERT_EQ(runtime2->next_workspace_user, runtime1);
  ASSERT_EQ(runtime1->next_workspace_user, nullptr);

  ASSERT_EQ(workspace->ref_count, 3);
  xnn_delete_runtime(auto_runtime2.release());
  ASSERT_EQ(workspace->first_user, runtime1);
  ASSERT_EQ(runtime1->next_workspace_user, nullptr);
  ASSERT_EQ(workspace->ref_count, 2);

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

  xnn_subgraph_t subgraph1 = nullptr;
  DefineGraph(&subgraph1, dims);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph1(subgraph1, xnn_delete_subgraph);

  xnn_runtime_t runtime1 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph1, nullptr, workspace, nullptr, 0, &runtime1));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime1(runtime1, xnn_delete_runtime);

  xnn_subgraph_t subgraph2 = nullptr;
  DefineGraph(&subgraph2, dims);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph2(subgraph2, xnn_delete_subgraph);

  xnn_runtime_t runtime2 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph2, nullptr, workspace, nullptr, 0, &runtime2));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime2(runtime2, xnn_delete_runtime);

  ASSERT_EQ(workspace->first_user, runtime2);
  ASSERT_EQ(runtime2->next_workspace_user, runtime1);
  ASSERT_EQ(runtime1->next_workspace_user, nullptr);

  ASSERT_EQ(workspace->ref_count, 3);
  xnn_delete_runtime(auto_runtime1.release());

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

  xnn_subgraph_t subgraph1 = nullptr;
  DefineGraph(&subgraph1, dims);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph1(subgraph1, xnn_delete_subgraph);

  xnn_runtime_t runtime1 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph1, nullptr, workspace, nullptr, 0, &runtime1));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime1(runtime1, xnn_delete_runtime);

  xnn_subgraph_t subgraph2 = nullptr;
  DefineGraph(&subgraph2, dims);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph2(subgraph2, xnn_delete_subgraph);

  xnn_runtime_t runtime2 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph2, nullptr, workspace, nullptr, 0, &runtime2));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime2(runtime2, xnn_delete_runtime);

  xnn_subgraph_t subgraph3 = nullptr;
  DefineGraph(&subgraph3, dims);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph3(subgraph3, xnn_delete_subgraph);

  xnn_runtime_t runtime3 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph3, nullptr, workspace, nullptr, 0, &runtime3));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime3(runtime3, xnn_delete_runtime);

  ASSERT_EQ(workspace->first_user, runtime3);
  ASSERT_EQ(runtime3->next_workspace_user, runtime2);
  ASSERT_EQ(runtime2->next_workspace_user, runtime1);
  ASSERT_EQ(runtime1->next_workspace_user, nullptr);

  ASSERT_EQ(workspace->ref_count, 4);
  xnn_delete_runtime(auto_runtime2.release());

  ASSERT_EQ(workspace->first_user, runtime3);
  ASSERT_EQ(runtime3->next_workspace_user, runtime1);
  ASSERT_EQ(runtime1->next_workspace_user, nullptr);
  ASSERT_EQ(workspace->ref_count, 3);

  xnn_delete_runtime(auto_runtime3.release());
  ASSERT_EQ(workspace->first_user, runtime1);
  ASSERT_EQ(runtime1->next_workspace_user, nullptr);
  ASSERT_EQ(workspace->ref_count, 2);

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

  xnn_subgraph_t subgraph = nullptr;
  DefineGraphWithoutInternalTensors(&subgraph, dims);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph, nullptr, workspace, nullptr, 0, &runtime));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);

  ASSERT_EQ(0, workspace->size);
  ASSERT_EQ(nullptr, workspace->data);
  ASSERT_EQ(std::vector<xnn_runtime_t>({runtime}), workspace_user_to_list(workspace));
  ASSERT_EQ(workspace->ref_count, 2);
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

  xnn_subgraph_t subgraph = nullptr;
  DefineGraphWithPersistentTensors(&subgraph, dims);
  const std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph, nullptr, workspace, nullptr, 0, &runtime));
  const std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);

  const size_t old_workspace_size = workspace->size;
  ASSERT_GE(old_workspace_size, 0);
  const void* old_runtime_workspace = runtime->workspace->data;
  ASSERT_NE(old_runtime_workspace, nullptr);

  size_t persistent_size = 0;
  for (size_t i = 0; i < runtime->num_blobs; i++) {
    const xnn_blob* blob = &runtime->blobs[i];
    if (blob->allocation_type == xnn_allocation_type_persistent) {
      ASSERT_EQ((uintptr_t) blob->data, (uintptr_t) workspace->data + persistent_size);
      persistent_size += round_up_po2(blob->size, XNN_EXTRA_BYTES);
    }
  }
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

  xnn_subgraph_t subgraph1 = nullptr;
  DefineGraphWithPersistentTensors(&subgraph1, dims1);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph1(subgraph1, xnn_delete_subgraph);

  xnn_runtime_t runtime1 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph1, nullptr, workspace, nullptr, 0, &runtime1));
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
  const std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime2(runtime2, xnn_delete_runtime);

  // Check that the workspace grew.
  ASSERT_GT(workspace->size, old_workspace_size);
  ASSERT_NE(runtime2->workspace->data, old_runtime_workspace);
  ASSERT_EQ(runtime1->workspace->data, runtime2->workspace->data);
  ASSERT_EQ(runtime1->workspace->size, runtime2->workspace->size);

  size_t persistent_size = 0;
  for (size_t i = 0; i < runtime1->num_blobs; i++) {
    const xnn_blob* blob = &runtime1->blobs[i];
    if (blob->allocation_type == xnn_allocation_type_persistent) {
      ASSERT_EQ((uintptr_t) blob->data, (uintptr_t) workspace->data + persistent_size);
      persistent_size += round_up_po2(blob->size, XNN_EXTRA_BYTES);
    }
  }

  persistent_size = 0;
  for (size_t i = 0; i < runtime2->num_blobs; i++) {
    const xnn_blob* blob = &runtime2->blobs[i];
    if (blob->allocation_type == xnn_allocation_type_persistent) {
      ASSERT_EQ((uintptr_t) blob->data, (uintptr_t) workspace->data + persistent_size);
      persistent_size += round_up_po2(blob->size, XNN_EXTRA_BYTES);
    }
  }
}
