// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Regression test for a double free on the xnn_create_runtime_v4() error path.
//
// xnn_create_runtime_v4() copies the subgraph values into runtime->values, sharing
// the data pointers. Buffers that rewrites allocated for static values are flagged
// XNN_VALUE_FLAG_NEEDS_CLEANUP, and xnn_delete_runtime() frees every buffer with that
// flag. Ownership was only transferred from the subgraph (by clearing
// subgraph->values[i].data) AFTER every node had been created -- so when node
// creation failed and the error path called xnn_delete_runtime(), the runtime freed
// buffers the subgraph still pointed at, and xnn_delete_subgraph() freed them again.
//
// optimize_common_subgraphs_broadcast() is one way to get such a buffer: eliding a
// static broadcast allocates a static zero tensor flagged NEEDS_CLEANUP. This test
// builds that graph, fails the operator allocation so node creation fails, and then
// checks the ownership invariant directly, which is deterministic and needs no ASan.
//
// The test needs its own binary because xnn_initialize() installs the allocator
// process-wide and only the first call wins (src/init.c, __sync_bool_compare_and_swap).

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <gtest/gtest.h>

#include "include/xnnpack.h"
#include "src/xnnpack/internal.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/subgraph.h"

namespace {

// The failing allocation is the operator descriptor that xnn_create_runtime_v4()'s
// node loop asks for.
const size_t kOperatorBytes = sizeof(struct xnn_operator);

bool g_fail_operator_alloc = false;

void* TestAllocate(void* /*context*/, size_t size) { return std::malloc(size); }

void* TestReallocate(void* /*context*/, void* pointer, size_t size) {
  if (g_fail_operator_alloc && size == kOperatorBytes) {
    g_fail_operator_alloc = false;
    return nullptr;
  }
  return std::realloc(pointer, size);
}

void TestDeallocate(void* /*context*/, void* pointer) { std::free(pointer); }

void* TestAlignedAllocate(void* /*context*/, size_t alignment, size_t size) {
  if (g_fail_operator_alloc && size == kOperatorBytes) {
    g_fail_operator_alloc = false;
    return nullptr;
  }
  void* pointer = nullptr;
  if (posix_memalign(&pointer, alignment, size) != 0) {
    return nullptr;
  }
  return pointer;
}

void TestAlignedDeallocate(void* /*context*/, void* pointer) { std::free(pointer); }

// A broadcast whose shape has no zero dimensions is a candidate for elision, which is
// what makes optimize_common_subgraphs_broadcast() allocate a static zero tensor.
xnn_subgraph_t CreateStaticBroadcastSubgraph() {
  xnn_subgraph_t subgraph = nullptr;
  if (xnn_create_subgraph(2, 0, &subgraph) != xnn_status_success) {
    return nullptr;
  }
  const size_t dims[3] = {2, 3, 4};
  const size_t broadcast_shape[3] = {2, 3, 4};
  uint32_t input_id;
  uint32_t output_id;
  if (xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 3, dims, nullptr, 0,
                              XNN_VALUE_FLAG_EXTERNAL_INPUT,
                              &input_id) != xnn_status_success ||
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 3, dims, nullptr, 1,
                              XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                              &output_id) != xnn_status_success) {
    xnn_delete_subgraph(subgraph);
    return nullptr;
  }
  if (xnn_define_static_broadcast(subgraph, 3, broadcast_shape, input_id, output_id, 0) !=
      xnn_status_success) {
    xnn_delete_subgraph(subgraph);
    return nullptr;
  }
  return subgraph;
}

// Counts the static values that xnn_delete_runtime() would free on the subgraph's
// behalf but which the subgraph still claims to own.
size_t CountStaleOwnedCleanupBuffers(xnn_subgraph_t subgraph) {
  size_t stale = 0;
  for (uint32_t i = 0; i < subgraph->num_values; i++) {
    const struct xnn_value* value = &subgraph->values[i];
    if (value->data == nullptr) {
      continue;
    }
    if (value->allocation_type != xnn_allocation_type_static) {
      continue;
    }
    if ((value->flags & XNN_VALUE_FLAG_NEEDS_CLEANUP) ||
        value->fp16_rewrite.fp16_compatible) {
      stale++;
    }
  }
  return stale;
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

TEST(RuntimeErrorPathOwnershipTest, FailedCreateDoesNotLeaveSubgraphOwningFreedBuffers) {
  ASSERT_TRUE(InitializeWithTestAllocator());

  // Baseline: without injection the graph must optimize into a binary-add with a
  // static zero tensor, so that the rewrite under test really produced a
  // NEEDS_CLEANUP buffer. Without that, the rest of the test would pass vacuously.
  {
    xnn_subgraph_t subgraph = CreateStaticBroadcastSubgraph();
    ASSERT_NE(subgraph, nullptr);
    xnn_runtime_t runtime = nullptr;
    ASSERT_EQ(xnn_create_runtime_v4(subgraph, nullptr, nullptr, nullptr, 0, &runtime),
              xnn_status_success);
    ASSERT_NE(runtime, nullptr);
    xnn_delete_runtime(runtime);
    // Ownership moved to the runtime, so the subgraph keeps nothing.
    EXPECT_EQ(CountStaleOwnedCleanupBuffers(subgraph), 0u);
    xnn_delete_subgraph(subgraph);
  }

  // Fresh subgraph: xnn_create_runtime_v4() rewrites the subgraph in place, so the
  // optimization only runs once per subgraph.
  xnn_subgraph_t subgraph = CreateStaticBroadcastSubgraph();
  ASSERT_NE(subgraph, nullptr);

  g_fail_operator_alloc = true;
  xnn_runtime_t runtime = nullptr;
  const xnn_status status =
      xnn_create_runtime_v4(subgraph, nullptr, nullptr, nullptr, 0, &runtime);
  ASSERT_EQ(status, xnn_status_out_of_memory);

  // The error path already called xnn_delete_runtime(), which freed every
  // NEEDS_CLEANUP buffer in the runtime's copy of the values. Any such buffer the
  // subgraph still points at is already freed, and xnn_delete_subgraph() below will
  // free it a second time.
  EXPECT_EQ(CountStaleOwnedCleanupBuffers(subgraph), 0u)
      << "subgraph still owns " << CountStaleOwnedCleanupBuffers(subgraph)
      << " buffer(s) that xnn_delete_runtime() already freed";

  xnn_delete_subgraph(subgraph);
}
