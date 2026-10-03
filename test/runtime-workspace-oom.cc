// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Regression test for xnn_create_runtime_v4() reporting success when the
// internal workspace allocation fails.
//
// When the caller passes workspace == NULL, xnn_create_runtime_v4() allocates the
// workspace itself. On failure it jumps to the `error` label, which returns
// whatever `status` happens to hold. By that point `status` was last set by the
// final `node->create()` call, which succeeded, so `status` is
// xnn_status_success. The caller therefore receives xnn_status_success together
// with an untouched *runtime_out == NULL, and the natural
// `if (status == xnn_status_success) xnn_reshape_runtime(runtime);` dereferences
// NULL.
//
// The test runs in its own binary because xnn_initialize() installs the allocator
// process-wide and only the first call wins (init.c, __sync_bool_compare_and_swap),
// so the failing allocator cannot be shared with other tests. The allocator
// delegates to malloc unless armed, and it is armed only for the workspace-sized
// allocation, so the surrounding setup still gets real memory.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <gtest/gtest.h>

#include "include/xnnpack.h"
#include "src/xnnpack/subgraph.h"

namespace {

// Size of the allocation that xnn_create_runtime_v4() makes when the caller does
// not supply a workspace.
const size_t kWorkspaceBytes = sizeof(struct xnn_workspace);

bool g_fail_workspace_alloc = false;

void* TestAllocate(void* /*context*/, size_t size) {
  if (g_fail_workspace_alloc && size == kWorkspaceBytes) {
    g_fail_workspace_alloc = false;
    return nullptr;
  }
  return std::malloc(size);
}

void* TestReallocate(void* /*context*/, void* pointer, size_t size) {
  if (g_fail_workspace_alloc && size == kWorkspaceBytes) {
    g_fail_workspace_alloc = false;
    return nullptr;
  }
  return std::realloc(pointer, size);
}

void TestDeallocate(void* /*context*/, void* pointer) { std::free(pointer); }

void* TestAlignedAllocate(void* /*context*/, size_t alignment, size_t size) {
  if (g_fail_workspace_alloc && size == kWorkspaceBytes) {
    g_fail_workspace_alloc = false;
    return nullptr;
  }
  void* pointer = nullptr;
  if (posix_memalign(&pointer, alignment, size) != 0) {
    return nullptr;
  }
  return pointer;
}

void TestAlignedDeallocate(void* /*context*/, void* pointer) { std::free(pointer); }

// A one-node subgraph: enough to make xnn_create_runtime_v4() do real work, so
// that reaching the workspace allocation means everything before it succeeded.
xnn_subgraph_t CreateTrivialSubgraph() {
  xnn_subgraph_t subgraph = nullptr;
  if (xnn_create_subgraph(2, 0, &subgraph) != xnn_status_success) {
    return nullptr;
  }
  const size_t dims[1] = {8};
  uint32_t input_id;
  uint32_t output_id;
  if (xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 1, dims, nullptr, 0,
                              XNN_VALUE_FLAG_EXTERNAL_INPUT,
                              &input_id) != xnn_status_success) {
    xnn_delete_subgraph(subgraph);
    return nullptr;
  }
  if (xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 1, dims, nullptr, 1,
                              XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                              &output_id) != xnn_status_success) {
    xnn_delete_subgraph(subgraph);
    return nullptr;
  }
  if (xnn_define_copy(subgraph, input_id, output_id, 0) != xnn_status_success) {
    xnn_delete_subgraph(subgraph);
    return nullptr;
  }
  return subgraph;
}

}  // namespace

// Must run before any other XNNPACK use in this binary: xnn_initialize() only
// honours the allocator passed to its first call.
static bool InitializeWithTestAllocator() {
  struct xnn_allocator allocator = {
      /*context=*/nullptr,
      /*allocate=*/TestAllocate,
      /*reallocate=*/TestReallocate,
      /*deallocate=*/TestDeallocate,
      /*aligned_allocate=*/TestAlignedAllocate,
      /*aligned_deallocate=*/TestAlignedDeallocate,
  };
  return xnn_initialize(&allocator) == xnn_status_success;
}

TEST(RuntimeWorkspaceAllocationFailureTest, ReportsOutOfMemory) {
  ASSERT_TRUE(InitializeWithTestAllocator());

  xnn_subgraph_t subgraph = CreateTrivialSubgraph();
  ASSERT_NE(subgraph, nullptr);

  // Without injection the runtime must be created normally, so that reaching the
  // workspace allocation below means everything before it already succeeded.
  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_create_runtime_v4(subgraph, nullptr, nullptr, nullptr, 0, &runtime),
            xnn_status_success);
  ASSERT_NE(runtime, nullptr);
  xnn_delete_runtime(runtime);

  g_fail_workspace_alloc = true;
  runtime = reinterpret_cast<xnn_runtime_t>(-1);
  const xnn_status status =
      xnn_create_runtime_v4(subgraph, nullptr, nullptr, nullptr, 0, &runtime);

  // The allocation failed, so the caller must be told so. Reporting success here
  // hands back a NULL runtime and crashes the caller's next call.
  EXPECT_EQ(status, xnn_status_out_of_memory);
  EXPECT_EQ(runtime, nullptr);

  xnn_delete_subgraph(subgraph);
}
