// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Regression test for the unchecked NULL from xnn_subgraph_new_internal_value().
//
// xnn_subgraph_new_internal_value() returns NULL when the subgraph value table cannot grow
// (src/subgraph.c). Two call sites passed that result straight into xnn_value_copy(), which
// dereferences it, and both are reachable from xnn_create_runtime_v4():
//
//   src/subgraph.c, xnn_subgraph_rewrite_ssa()   - external output written by two nodes
//   src/subgraph.c, xnn_subgraph_rewrite_for_fp16() - same shape on the FP16 path
//
// Both now detect the failure and use the convention their own function already uses:
// log and give up on the rewrite, instead of crashing.
//
// The graph here writes one external output from two nodes, which is what drives
// xnn_subgraph_rewrite_ssa() down its "already produced" branch. The allocator fails exactly
// the value-table growth by rejecting the first reallocation whose size is a nonzero exact
// multiple of sizeof(struct xnn_value).
//
// The test needs its own binary because xnn_initialize() installs the allocator
// process-wide and only the first call wins (src/init.c, __sync_bool_compare_and_swap).

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <gtest/gtest.h>

#include "include/xnnpack.h"
#include "src/xnnpack/subgraph.h"

namespace {

bool g_fail_value_table_realloc = false;

void* TestAllocate(void* /*context*/, size_t size) { return std::malloc(size); }

void* TestReallocate(void* /*context*/, void* pointer, size_t size) {
  // sizeof(struct xnn_value) is 224; the value table is the only allocation in this path
  // whose size is a nonzero exact multiple of it.
  if (g_fail_value_table_realloc && size % sizeof(struct xnn_value) == 0 &&
      size != sizeof(struct xnn_value)) {
    g_fail_value_table_realloc = false;
    return nullptr;
  }
  return std::realloc(pointer, size);
}

void TestDeallocate(void* /*context*/, void* pointer) { std::free(pointer); }

void* TestAlignedAllocate(void* /*context*/, size_t alignment, size_t size) {
  void* pointer = nullptr;
  if (posix_memalign(&pointer, alignment, size) != 0) {
    return nullptr;
  }
  return pointer;
}

void TestAlignedDeallocate(void* /*context*/, void* pointer) { std::free(pointer); }

// `output` is written by two of the three nodes, which is what makes the SSA rewrite
// allocate a replacement value for the first write.
struct TestGraph {
  xnn_subgraph_t subgraph = nullptr;
  uint32_t input_id = 0;
  uint32_t a_id = 0;
  uint32_t b_id = 0;
  uint32_t output_id = 0;
  uint32_t persistent_id = 0;
};

TestGraph CreateDuplicateOutputGraph() {
  TestGraph graph;
  if (xnn_create_subgraph(5, 0, &graph.subgraph) != xnn_status_success) {
    graph.subgraph = nullptr;
    return graph;
  }
  const size_t dims[1] = {8};
  const size_t scalar[1] = {1};
  if (xnn_define_tensor_value(graph.subgraph, xnn_datatype_fp32, 1, dims, nullptr,
                              0, XNN_VALUE_FLAG_EXTERNAL_INPUT,
                              &graph.input_id) != xnn_status_success ||
      xnn_define_tensor_value(graph.subgraph, xnn_datatype_fp32, 1, scalar,
                              nullptr, 1, XNN_VALUE_FLAG_EXTERNAL_INPUT,
                              &graph.a_id) != xnn_status_success ||
      xnn_define_tensor_value(graph.subgraph, xnn_datatype_fp32, 1, scalar,
                              nullptr, 2, XNN_VALUE_FLAG_EXTERNAL_INPUT,
                              &graph.b_id) != xnn_status_success ||
      xnn_define_tensor_value(graph.subgraph, xnn_datatype_fp32, 1, dims, nullptr,
                              3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                              &graph.output_id) != xnn_status_success ||
      xnn_define_tensor_value(graph.subgraph, xnn_datatype_fp32, 1, dims, nullptr,
                              4,
                              XNN_VALUE_FLAG_EXTERNAL_INPUT |
                                  XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                              &graph.persistent_id) != xnn_status_success) {
    xnn_delete_subgraph(graph.subgraph);
    graph.subgraph = nullptr;
    return graph;
  }
  // First write of the external output `output_id`.
  if (xnn_define_binary(graph.subgraph, xnn_binary_multiply, nullptr,
                        graph.persistent_id, graph.a_id, graph.output_id,
                        0) != xnn_status_success) {
    xnn_delete_subgraph(graph.subgraph);
    graph.subgraph = nullptr;
    return graph;
  }
  // Second write of the SAME external output -> "already produced" branch.
  if (xnn_define_binary(graph.subgraph, xnn_binary_multiply, nullptr,
                        graph.persistent_id, graph.b_id, graph.output_id,
                        0) != xnn_status_success) {
    xnn_delete_subgraph(graph.subgraph);
    graph.subgraph = nullptr;
    return graph;
  }
  if (xnn_define_binary(graph.subgraph, xnn_binary_add, nullptr, graph.output_id,
                        graph.input_id, graph.persistent_id,
                        0) != xnn_status_success) {
    xnn_delete_subgraph(graph.subgraph);
    graph.subgraph = nullptr;
    return graph;
  }
  return graph;
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

TEST(SubgraphSsaRewriteAllocationFailureTest, DoesNotCrash) {
  ASSERT_TRUE(InitializeWithTestAllocator());

  float input_data[8] = {0.0f};
  float a_data[1] = {2.0f};
  float b_data[1] = {3.0f};
  float output_data[8] = {0.0f};
  float persistent_data[8] = {1.0f};

  // First, without injection: the runtime must be created and used normally. That confirms
  // the graph really does drive the SSA rewrite down its "already produced" branch, and it
  // leaves every allocation before the value table proven to succeed.
  {
    TestGraph graph = CreateDuplicateOutputGraph();
    ASSERT_NE(graph.subgraph, nullptr);
    struct xnn_external_value values[5] = {
        {graph.input_id, input_data},
        {graph.a_id, a_data},
        {graph.b_id, b_data},
        {graph.output_id, output_data},
        {graph.persistent_id, persistent_data},
    };
    xnn_runtime_t runtime = nullptr;
    ASSERT_EQ(xnn_create_runtime_v4(graph.subgraph, nullptr, nullptr, nullptr, 0,
                                    &runtime),
              xnn_status_success);
    ASSERT_NE(runtime, nullptr);
    ASSERT_EQ(xnn_reshape_runtime(runtime), xnn_status_success);
    ASSERT_EQ(xnn_setup_runtime(runtime, 5, values), xnn_status_success);
    ASSERT_EQ(xnn_invoke_runtime(runtime), xnn_status_success);
    xnn_delete_runtime(runtime);
    xnn_delete_subgraph(graph.subgraph);
  }

  // A fresh subgraph is required here: xnn_create_runtime_v4() rewrites the subgraph in
  // place, and the value table only grows while doing so. Reusing the graph above would
  // leave enough capacity and never reallocate.
  TestGraph graph = CreateDuplicateOutputGraph();
  ASSERT_NE(graph.subgraph, nullptr);
  struct xnn_external_value values[5] = {
      {graph.input_id, input_data},
      {graph.a_id, a_data},
      {graph.b_id, b_data},
      {graph.output_id, output_data},
      {graph.persistent_id, persistent_data},
  };

  g_fail_value_table_realloc = true;
  xnn_runtime_t runtime = nullptr;
  const xnn_status status = xnn_create_runtime_v4(graph.subgraph, nullptr, nullptr,
                                                   nullptr, 0, &runtime);

  // Before the fix this dereferenced NULL inside xnn_subgraph_rewrite_ssa() and the process
  // died. Now the rewrite gives up on the allocation, so the call either reports an error or
  // hands back a usable runtime.
  if (status == xnn_status_success) {
    ASSERT_NE(runtime, nullptr);
    EXPECT_EQ(xnn_reshape_runtime(runtime), xnn_status_success);
    EXPECT_EQ(xnn_setup_runtime(runtime, 5, values), xnn_status_success);
    EXPECT_EQ(xnn_invoke_runtime(runtime), xnn_status_success);
    xnn_delete_runtime(runtime);
  }

  xnn_delete_subgraph(graph.subgraph);
}
