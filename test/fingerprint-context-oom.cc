// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Regression test for create_fingerprint_context() reporting an allocation failure as
// "no fingerprint yet".
//
// create_fingerprint_context() allocates the cache context lazily and, when that
// allocation failed, left `status` at xnn_status_uninitialized. Callers use that value to
// mean "no fingerprint exists yet, go compute one":
//
//   struct fingerprint_context f_context = create_fingerprint_context(fingerprint_id);
//   if (f_context.status != xnn_status_uninitialized) {
//     return f_context.status;          // already exists, or a real error
//   }
//   // ... compute the fingerprint, calling cache.reserve_space(cache.context, bytes)
//
// So an allocation failure made the caller compute a fingerprint against a NULL cache
// context, and fingerprint_cache_reserve_space() dereferenced it. The status is now
// xnn_status_out_of_memory, which callers already propagate.
//
// The test needs its own binary because xnn_initialize() installs the allocator
// process-wide and only the first call wins (src/init.c, __sync_bool_compare_and_swap).

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <gtest/gtest.h>

#include "include/experimental.h"
#include "include/xnnpack.h"
#include "src/operators/fingerprint_cache.h"
#include "src/operators/fingerprint_id.h"

namespace {

// Mirrors the private `struct fingerprint_cache_context` in
// src/operators/fingerprint_cache.c, which is not exposed in a header. If the two ever
// diverge, the size selector below misses and this test fails loudly rather than
// passing vacuously.
struct FingerprintCacheContextMirror {
  void* buffer;
  size_t bytes;
  uint32_t hash;
};
const size_t kFingerprintContextBytes = sizeof(FingerprintCacheContextMirror);

bool g_fail_context_alloc = false;

void* TestAllocate(void* /*context*/, size_t size) {
  if (g_fail_context_alloc && size == kFingerprintContextBytes) {
    g_fail_context_alloc = false;
    return nullptr;
  }
  return std::malloc(size);
}

void* TestReallocate(void* /*context*/, void* pointer, size_t size) {
  if (g_fail_context_alloc && size == kFingerprintContextBytes) {
    g_fail_context_alloc = false;
    return nullptr;
  }
  return std::realloc(pointer, size);
}

void TestDeallocate(void* /*context*/, void* pointer) { std::free(pointer); }

void* TestAlignedAllocate(void* /*context*/, size_t alignment, size_t size) {
  if (g_fail_context_alloc && size == kFingerprintContextBytes) {
    g_fail_context_alloc = false;
    return nullptr;
  }
  void* pointer = nullptr;
  if (posix_memalign(&pointer, alignment, size) != 0) {
    return nullptr;
  }
  return pointer;
}

void TestAlignedDeallocate(void* /*context*/, void* pointer) { std::free(pointer); }

// A fingerprint id that no operator registers, so the fingerprint is genuinely absent and
// create_fingerprint_context() takes the allocating branch.
const enum xnn_fingerprint_id kAbsentFingerprintId =
    xnn_fingerprint_id_test_f16_f32_qc8w_nr2;

}  // namespace

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

TEST(FingerprintContextAllocationFailureTest, ReportsOutOfMemory) {
  ASSERT_TRUE(InitializeWithTestAllocator());

  // The injected attempt runs first, because finalizing a context registers the
  // fingerprint, after which create_fingerprint_context() no longer allocates.
  g_fail_context_alloc = true;
  struct fingerprint_context context =
      create_fingerprint_context(kAbsentFingerprintId);

  // xnn_status_uninitialized here would tell the caller to go compute a fingerprint
  // against context.cache.context == NULL, which crashes in
  // fingerprint_cache_reserve_space().
  EXPECT_EQ(context.status, xnn_status_out_of_memory);
  EXPECT_EQ(context.cache.context, nullptr);

  // With no injection the very same call must still take the allocating branch and
  // succeed. This proves the size selector above targets exactly this allocation, so
  // the assertion is not passing for an unrelated reason.
  struct fingerprint_context baseline =
      create_fingerprint_context(kAbsentFingerprintId);
  EXPECT_EQ(baseline.status, xnn_status_uninitialized);
  EXPECT_NE(baseline.cache.context, nullptr);

  finalize_fingerprint_context(&context);
  finalize_fingerprint_context(&baseline);
}
