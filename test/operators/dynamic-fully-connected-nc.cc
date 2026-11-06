// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/params.h"
#include "test/operators/dynamic-fully-connected-operator-tester.h"

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, unit_batch) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(23)
      .output_channels(19)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, unit_batch_with_qmin) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(23)
      .output_channels(19)
      .qmin(128)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, unit_batch_with_qmax) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(23)
      .output_channels(19)
      .qmax(128)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, unit_batch_with_input_stride) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(23)
      .input_stride(28)
      .output_channels(19)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, unit_batch_with_output_stride) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(23)
      .output_channels(19)
      .output_stride(29)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, unit_batch_transpose_weights) {
  DynamicFullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(23)
      .output_channels(19)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, unit_batch_without_bias) {
  DynamicFullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(23)
      .output_channels(19)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, small_batch) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(23)
      .output_channels(19)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, small_batch_with_qmin) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(23)
      .output_channels(19)
      .qmin(128)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, small_batch_with_qmax) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(23)
      .output_channels(19)
      .qmax(128)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, small_batch_with_input_stride) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(23)
      .input_stride(28)
      .output_channels(19)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, small_batch_with_output_stride) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(23)
      .output_channels(19)
      .output_stride(29)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, small_batch_transpose_weights) {
  DynamicFullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(12)
      .input_channels(23)
      .output_channels(19)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, small_batch_without_bias) {
  DynamicFullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(12)
      .input_channels(23)
      .output_channels(19)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, unit_batch) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(23)
      .output_channels(19)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, unit_batch_with_qmin) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(23)
      .output_channels(19)
      .qmin(128)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, unit_batch_with_qmax) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(23)
      .output_channels(19)
      .qmax(128)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, unit_batch_with_input_stride) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(23)
      .input_stride(28)
      .output_channels(19)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, unit_batch_with_output_stride) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(23)
      .output_channels(19)
      .output_stride(29)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, unit_batch_transpose_weights) {
  DynamicFullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(23)
      .output_channels(9)
      .iterations(1)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, unit_batch_without_bias) {
  DynamicFullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(23)
      .output_channels(19)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, small_batch) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(23)
      .output_channels(19)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, small_batch_with_qmin) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(23)
      .output_channels(19)
      .qmin(128)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, small_batch_with_qmax) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(23)
      .output_channels(19)
      .qmax(128)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, small_batch_with_input_stride) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(23)
      .input_stride(28)
      .output_channels(19)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, small_batch_with_output_stride) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(23)
      .output_channels(19)
      .output_stride(29)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, small_batch_transpose_weights) {
  DynamicFullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(12)
      .input_channels(23)
      .output_channels(19)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, small_batch_without_bias) {
  DynamicFullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(12)
      .input_channels(23)
      .output_channels(19)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, overflow_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_operator_t op = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_dynamic_fully_connected_nc_f32(
                -std::numeric_limits<float>::infinity(),
                +std::numeric_limits<float>::infinity(),
                /*flags=*/0, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  size_t workspace_size = 0;
  ASSERT_EQ(xnn_status_out_of_memory,
            xnn_reshape_dynamic_fully_connected_nc_f32(
                op, /*batch_size=*/1, /*input_channels=*/10,
                /*output_channels=*/10, /*input_stride=*/SIZE_MAX / 2,
                /*output_stride=*/10, &workspace_size,
                /*threadpool=*/nullptr));
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, overflow_batch_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_operator_t op = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_dynamic_fully_connected_nc_f32(
                -std::numeric_limits<float>::infinity(),
                +std::numeric_limits<float>::infinity(),
                /*flags=*/0, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  size_t workspace_size = 0;
  ASSERT_EQ(xnn_status_out_of_memory,
            xnn_reshape_dynamic_fully_connected_nc_f32(
                op, /*batch_size=*/SIZE_MAX / 50, /*input_channels=*/10,
                /*output_channels=*/10, /*input_stride=*/100,
                /*output_stride=*/100, &workspace_size,
                /*threadpool=*/nullptr));
}


namespace {

// Fails the Nth allocation made while it is installed, counting the plain and
// the aligned hook together, and passes every other allocation through. It
// swaps xnn_params.allocator directly, the way CountingAllocatorGuard does in
// convolution-nhwc.cc, so it does not depend on xnn_initialize() being called
// first.
class FailingAllocatorGuard {
 public:
  explicit FailingAllocatorGuard(size_t fail_at)
      : saved_allocator_(xnn_params.allocator), fail_at_(fail_at) {
    xnn_params.allocator.context = this;
    xnn_params.allocator.allocate = Allocate;
    xnn_params.allocator.reallocate = Reallocate;
    xnn_params.allocator.aligned_allocate = AlignedAllocate;
  }

  ~FailingAllocatorGuard() { xnn_params.allocator = saved_allocator_; }

  FailingAllocatorGuard(const FailingAllocatorGuard&) = delete;
  FailingAllocatorGuard& operator=(const FailingAllocatorGuard&) = delete;

  // How many allocations were attempted, so a caller can tell whether the
  // injection index was ever reached.
  size_t attempts() const { return attempts_; }

 private:
  bool ShouldFail() { return ++attempts_ == fail_at_; }

  static void* Allocate(void* context, size_t size) {
    auto* self = static_cast<FailingAllocatorGuard*>(context);
    if (self->ShouldFail()) {
      return nullptr;
    }
    return self->saved_allocator_.allocate(
        self->saved_allocator_.context, size);
  }

  static void* Reallocate(void* context, void* pointer, size_t size) {
    auto* self = static_cast<FailingAllocatorGuard*>(context);
    if (self->ShouldFail()) {
      return nullptr;
    }
    return self->saved_allocator_.reallocate(self->saved_allocator_.context,
                                             pointer, size);
  }

  static void* AlignedAllocate(void* context, size_t alignment, size_t size) {
    auto* self = static_cast<FailingAllocatorGuard*>(context);
    if (self->ShouldFail()) {
      return nullptr;
    }
    return self->saved_allocator_.aligned_allocate(
        self->saved_allocator_.context, alignment, size);
  }

  const struct xnn_allocator saved_allocator_;
  const size_t fail_at_;
  size_t attempts_ = 0;
};

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr float kPosInf = std::numeric_limits<float>::infinity();

}  // namespace

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, out_of_memory_during_create_is_reported) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Warm the fingerprint cache with one uninjected create, so the sweep below
  // only ever fails an allocation that create_dynamic_fully_connected_nc()
  // makes itself.
  {
    xnn_operator_t warmup_op = nullptr;
    ASSERT_EQ(
        xnn_status_success,
        xnn_create_dynamic_fully_connected_nc_f32(
            kNegInf, kPosInf, /*flags=*/0, &warmup_op));
    ASSERT_NE(warmup_op, nullptr);
    ASSERT_EQ(xnn_status_success, xnn_delete_operator(warmup_op));
  }

  // The operator descriptor, the compute array, extra_params, the GEMM ukernel
  // table and the GEMM context. Sweep past the end of that so each index is
  // exercised.
  for (size_t fail_at = 1; fail_at <= 8; fail_at++) {
    FailingAllocatorGuard allocator_guard(fail_at);

    xnn_operator_t op = nullptr;
    const xnn_status status =
        xnn_create_dynamic_fully_connected_nc_f32(kNegInf, kPosInf,
                                                  /*flags=*/0, &op);

    if (allocator_guard.attempts() < fail_at) {
      // This create never reached the failing index, so nothing was injected.
      continue;
    }

    if (status == xnn_status_success) {
      // Reporting success obliges the caller to receive an operator.
      EXPECT_NE(op, nullptr) << "fail_at=" << fail_at;
      if (op != nullptr) {
        EXPECT_EQ(xnn_status_success, xnn_delete_operator(op))
            << "fail_at=" << fail_at;
      }
    } else {
      // Reporting failure must not hand back an operator.
      EXPECT_EQ(op, nullptr) << "fail_at=" << fail_at;
    }
  }
}
