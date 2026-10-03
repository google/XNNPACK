// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Regression test for the ignored status of xnn_allocate_extra_params().
//
// xnn_allocate_extra_params() returns xnn_status_out_of_memory when it cannot
// allocate op->extra_params, but every call site discarded the status. The very
// next thing each caller does is write through op->extra_params, so an allocation
// failure became a NULL dereference instead of a reported error. In
// init_binary_elementwise_nd() the write is:
//
//   memcpy(op->extra_params, &uparams2, sizeof(uparams2));
//
// The test runs in its own binary because xnn_initialize() installs the allocator
// process-wide and only the first call wins (src/init.c, __sync_bool_compare_and_swap),
// so an allocator that fails on demand cannot be shared with other tests. Outside the
// armed window the allocator delegates to malloc, so the surrounding setup gets real
// memory.

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <gtest/gtest.h>

#include "include/xnnpack.h"
#include "src/xnnpack/operator.h"

namespace {

// Size of the allocation that xnn_allocate_extra_params() makes for
// /*num_extra_params=*/1.
const size_t kExtraParamsBytes = sizeof(union xnn_params);

bool g_fail_extra_params_alloc = false;

void* TestAllocate(void* /*context*/, size_t size) {
  if (g_fail_extra_params_alloc && size == kExtraParamsBytes) {
    g_fail_extra_params_alloc = false;
    return nullptr;
  }
  return std::malloc(size);
}

void* TestReallocate(void* /*context*/, void* pointer, size_t size) {
  if (g_fail_extra_params_alloc && size == kExtraParamsBytes) {
    g_fail_extra_params_alloc = false;
    return nullptr;
  }
  return std::realloc(pointer, size);
}

void TestDeallocate(void* /*context*/, void* pointer) { std::free(pointer); }

void* TestAlignedAllocate(void* /*context*/, size_t alignment, size_t size) {
  if (g_fail_extra_params_alloc && size == kExtraParamsBytes) {
    g_fail_extra_params_alloc = false;
    return nullptr;
  }
  void* pointer = nullptr;
  if (posix_memalign(&pointer, alignment, size) != 0) {
    return nullptr;
  }
  return pointer;
}

void TestAlignedDeallocate(void* /*context*/, void* pointer) { std::free(pointer); }

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

TEST(BinaryElementwiseExtraParamsAllocationFailureTest, ReportsOutOfMemory) {
  ASSERT_TRUE(InitializeWithTestAllocator());

  // Without injection the operator must be created normally, so that reaching the
  // failing allocation below means every earlier one already succeeded.
  xnn_operator_t op = nullptr;
  ASSERT_EQ(xnn_create_binary_elementwise_nd(xnn_binary_multiply,
                                             xnn_datatype_fp32,
                                             /*a_quantization=*/nullptr,
                                             /*b_quantization=*/nullptr,
                                             /*output_quantization=*/nullptr,
                                             /*flags=*/0,
                                             &op),
            xnn_status_success);
  ASSERT_NE(op, nullptr);
  ASSERT_EQ(xnn_delete_operator(op), xnn_status_success);

  g_fail_extra_params_alloc = true;
  op = nullptr;
  const xnn_status status = xnn_create_binary_elementwise_nd(
      xnn_binary_multiply, xnn_datatype_fp32, /*a_quantization=*/nullptr,
      /*b_quantization=*/nullptr, /*output_quantization=*/nullptr, /*flags=*/0, &op);

  // The allocation failed, so the caller must be told so. Ignoring this status
  // left op->extra_params NULL and the following memcpy() dereferenced it.
  EXPECT_EQ(status, xnn_status_out_of_memory);
}
