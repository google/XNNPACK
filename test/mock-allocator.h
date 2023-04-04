// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <xnnpack/allocator.h>
#include <gmock/gmock.h>

namespace xnnpack {

class MockAllocator : public xnn_allocator {
 public:
  MockAllocator() {
    // Setup calls to perform actuall memory alloc/realloc/free by delegating to
    // xnn_default_allocator.
    ON_CALL(*this, allocate).WillByDefault([](void* context, size_t size) {
      return xnn_default_allocator.allocate(context, size);
    });

    ON_CALL(*this, reallocate)
        .WillByDefault([](void* context, void* pointer, size_t size) {
          return xnn_default_allocator.reallocate(context, pointer, size);
        });

    ON_CALL(*this, deallocate)
        .WillByDefault([](void* context, void* pointer) {
          return xnn_default_allocator.deallocate(context, pointer);
        });

    ON_CALL(*this, aligned_allocate)
        .WillByDefault([](void* context, size_t alignment, size_t size) {
          return xnn_default_allocator.aligned_allocate(context, alignment,
                                                        size);
        });

    ON_CALL(*this, aligned_deallocate)
        .WillByDefault([](void* context, void* pointer) {
          return xnn_default_allocator.aligned_deallocate(context, pointer);
        });
  }

  MOCK_METHOD(void*, allocate, (void* context, size_t size));
  MOCK_METHOD(void*, reallocate, (void* context, void* pointer, size_t size));
  MOCK_METHOD(void, deallocate, (void* context, void* pointer));
  MOCK_METHOD(void*, aligned_allocate,
              (void* context, size_t alignment, size_t size));
  MOCK_METHOD(void, aligned_deallocate, (void* context, void* pointer));
};

static MockAllocator* mock_allocator_;

const struct xnn_allocator mock_allocator_wrapper_ = {
    /*context=*/nullptr,
    /*allocate=*/[](void* context, size_t size) -> void* {
      return mock_allocator_->allocate(context, size);
    },
    /*reallocate=*/[](void* context, void* pointer, size_t size) -> void* {
      return mock_allocator_->reallocate(context, pointer, size);
    },
    /*deallocate=*/[](void* context, void* pointer) -> void {
      return mock_allocator_->deallocate(context, pointer);
    },
    /*aligned_allocate=*/[](void* context, size_t alignment,
                           size_t size) -> void* {
      return mock_allocator_->aligned_allocate(context, alignment, size);
    },
    /*aligned_deallocate=*/[](void* context, void* pointer) -> void {
      return xnn_default_allocator.aligned_deallocate(context, pointer);
    },
};

/// Replaces the memory allocator with the given mock.
/// The allocator must be restored as soon as the lifetime of the mock ends.
inline void SetUpMockAllocator(MockAllocator* mock_allocator) {
  mock_allocator_ = mock_allocator;
  memcpy(&xnn_params.allocator, &mock_allocator_wrapper_,
         sizeof(struct xnn_allocator));
}

/// Restores the default XNNPACK memory allocator.
inline void RestoreDefaultAllocator(MockAllocator* mock_allocator) {
  memcpy(&xnn_params.allocator, &xnn_default_allocator,
         sizeof(struct xnn_allocator));
}

}  // namespace xnnpack
