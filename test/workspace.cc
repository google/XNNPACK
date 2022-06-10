// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include <xnnpack.h>
#include <xnnpack/subgraph.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {
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

using testing::PrintToString;

MATCHER_P(
  IsInWorkspace,
  workspace,
  std::string(negation ? "is not" : "is") + " in workspace [" + PrintToString(workspace->data) + ", " +
    PrintToString((void*) ((uintptr_t) workspace->data + workspace->size)) + "] of size " +
    PrintToString(workspace->size))
{
  *result_listener << "blob data: " << arg->data << " size: " << arg->size << " ";
  return (arg->data >= workspace->data) &&
         ((uintptr_t) arg->data + arg->size) <= ((uintptr_t) workspace->data + workspace->size);
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

TEST(WORKSPACE, workspace_no_growth)
{
  xnn_initialize(/*allocator=*/nullptr);
  xnn_workspace_t workspace = nullptr;
  xnn_create_workspace(&workspace);
  std::unique_ptr<xnn_workspace, decltype(&xnn_delete_workspace)> auto_workspace(workspace, xnn_delete_workspace);

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
    if (blob1->external) {
      continue;
    }
    ASSERT_THAT(blob1, IsInWorkspace(runtime1->workspace));
    xnn_blob* blob2 = &runtime2->blobs[i];
    ASSERT_THAT(blob2, IsInWorkspace(runtime2->workspace));
  }

  std::vector<xnn_runtime_t> workspace_users = workspace_user_to_list(workspace);
  ASSERT_EQ(workspace_users.size(), 2);
  ASSERT_THAT(workspace_users, ::testing::Contains(runtime1));
  ASSERT_THAT(workspace_users, ::testing::Contains(runtime2));
}

TEST(WORKSPACE, workspace_grow)
{
  xnn_initialize(/*allocator=*/nullptr);
  xnn_workspace_t workspace = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_workspace(&workspace));
  std::unique_ptr<xnn_workspace, decltype(&xnn_delete_workspace)> auto_workspace(workspace, xnn_delete_workspace);

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
  std::transform(dims2.begin(), dims2.end(), dims2.begin(), [](auto i) { return i * 2; });
  xnn_subgraph_t subgraph2 = nullptr;
  DefineGraph(&subgraph2, dims2);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph2(subgraph2, xnn_delete_subgraph);

  xnn_runtime_t runtime2 = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v4(subgraph2, nullptr, workspace, nullptr, 0, &runtime2));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime2(runtime2, xnn_delete_runtime);

  // Check that the workspace grew.
  ASSERT_GE(workspace->size, old_workspace_size);
  // Check that runtime 2 uses the same workspace.
  ASSERT_NE(runtime2->workspace->data, old_runtime_workspace);
  // Check that runtime1's workspace has been updated as well.
  ASSERT_EQ(runtime1->workspace->data, runtime2->workspace->data);
  ASSERT_EQ(runtime1->workspace->size, runtime2->workspace->size);

  // Check that both runtime's blob pointers are within range.
  for (size_t i = 0; i < runtime1->num_blobs; i++) {
    xnn_blob* blob = &runtime1->blobs[i];
    if (blob->external) {
      continue;
    }
    ASSERT_THAT(blob, IsInWorkspace(runtime1->workspace));
  }
  for (size_t i = 0; i < runtime2->num_blobs; i++) {
    xnn_blob* blob = &runtime2->blobs[i];
    if (blob->external) {
      continue;
    }
    ASSERT_THAT(blob, IsInWorkspace(runtime2->workspace));
  }

  std::vector<xnn_runtime_t> workspace_users = workspace_user_to_list(workspace);
  ASSERT_EQ(workspace_users.size(), 2);
  ASSERT_THAT(workspace_users, ::testing::Contains(runtime1));
  ASSERT_THAT(workspace_users, ::testing::Contains(runtime2));
}

TEST(WORKSPACE, workspace_runtime_delete_head_runtime_first)
{
  xnn_initialize(/*allocator=*/nullptr);
  xnn_workspace_t workspace = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_workspace(&workspace));
  std::unique_ptr<xnn_workspace, decltype(&xnn_delete_workspace)> auto_workspace(workspace, xnn_delete_workspace);

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

  xnn_delete_runtime(auto_runtime2.release());
  ASSERT_EQ(workspace->first_user, runtime1);
  ASSERT_EQ(runtime1->next_workspace_user, nullptr);

  xnn_delete_runtime(auto_runtime1.release());
}

TEST(WORKSPACE, workspace_runtime_delete_tail_runtime_first)
{
  xnn_initialize(/*allocator=*/nullptr);
  xnn_workspace_t workspace = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_workspace(&workspace));
  std::unique_ptr<xnn_workspace, decltype(&xnn_delete_workspace)> auto_workspace(workspace, xnn_delete_workspace);

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

  xnn_delete_runtime(auto_runtime1.release());

  ASSERT_EQ(workspace->first_user, runtime2);
  ASSERT_EQ(runtime2->next_workspace_user, nullptr);

  xnn_delete_runtime(auto_runtime2.release());
}

TEST(WORKSPACE, workspace_runtime_delete_middle_runtime_first)
{
  xnn_initialize(/*allocator=*/nullptr);
  xnn_workspace_t workspace = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_workspace(&workspace));
  std::unique_ptr<xnn_workspace, decltype(&xnn_delete_workspace)> auto_workspace(workspace, xnn_delete_workspace);

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

  xnn_delete_runtime(auto_runtime2.release());

  ASSERT_EQ(workspace->first_user, runtime3);
  ASSERT_EQ(runtime3->next_workspace_user, runtime1);
  ASSERT_EQ(runtime1->next_workspace_user, nullptr);

  xnn_delete_runtime(auto_runtime3.release());
  ASSERT_EQ(workspace->first_user, runtime1);
  ASSERT_EQ(runtime1->next_workspace_user, nullptr);

  xnn_delete_runtime(auto_runtime1.release());
  ASSERT_EQ(workspace->first_user, nullptr);
}
