#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/internal.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/params.h"
#include "src/xnnpack/subgraph.h"
#include "test/subgraph/runtime-tester.h"

TEST(RUNTIME, reshape_runtime) {
  xnnpack::RuntimeTester tester(4);
  uint32_t input0_id = 0;
  uint32_t input1_id = 1;
  uint32_t input2_id = 2;
  uint32_t output_id = 3;
  uint32_t add1_out, add2_out;
  size_t dim0 = 3;
  size_t new_dim0 = 400;
  size_t dummy_internal_dim = 1;

  // Set up input and output tensors.
  tester.AddInputTensorF32({dim0}, input0_id)
      .AddInputTensorF32({dim0}, input1_id)
      .AddInputTensorF32({dim0}, input2_id)
      .AddOutputTensorF32({dim0}, output_id)
      .AddInternalDynamicTensorF32({dummy_internal_dim}, &add1_out)
      .AddInternalDynamicTensorF32({dummy_internal_dim}, &add2_out);

  // Add ops. Note that we do this in two steps to avoid problems with the
  // `cmake-windows-x86` (using Visual C) build which doesn't propagate the
  // values for `add1_out` and `add2_out` properly.
  tester.AddAddition(input0_id, input1_id, add1_out)
      .AddAddition(input0_id, input2_id, add2_out)
      .AddMultiply(add1_out, add2_out, output_id);

  xnnpack::Buffer<float> expected(dim0);
  const float* input0_data = tester.GetExternalTensorDataF32(input0_id);
  const float* input1_data = tester.GetExternalTensorDataF32(input1_id);
  const float* input2_data = tester.GetExternalTensorDataF32(input2_id);
  for (size_t i = 0; i < dim0; ++i) {
    expected[i] =
        (input0_data[i] + input1_data[i]) * (input0_data[i] + input2_data[i]);
  }
  auto output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(expected, output);

  tester.ReshapeInput({new_dim0}, input0_id);
  tester.ReshapeInput({new_dim0}, input1_id);
  tester.ReshapeInput({new_dim0}, input2_id);

  tester.ReshapeRuntime();
  tester.SetupRuntimeV2();

  output = tester.RepeatRun<float>();
  expected = xnnpack::Buffer<float>(new_dim0);
  input0_data = tester.GetExternalTensorDataF32(input0_id);
  input1_data = tester.GetExternalTensorDataF32(input1_id);
  input2_data = tester.GetExternalTensorDataF32(input2_id);
  for (size_t i = 0; i < new_dim0; ++i) {
    expected[i] =
        (input0_data[i] + input1_data[i]) * (input0_data[i] + input2_data[i]);
  }
  ASSERT_EQ(expected, output);
}

TEST(RUNTIME, null_runtime) {
  size_t required_size = 0;
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_get_runtime_profiling_info(nullptr,
                                           xnn_profile_info_num_operators, 0,
                                           nullptr, &required_size));
  EXPECT_EQ(xnn_status_invalid_parameter, xnn_invoke_runtime(nullptr));
}

namespace {

class FailingAllocatorGuard {
 public:
  explicit FailingAllocatorGuard(size_t fail_bytes)
      : saved_allocator_(xnn_params.allocator), fail_bytes_(fail_bytes) {
    xnn_params.allocator.context = this;
    xnn_params.allocator.reallocate = FailingReallocate;
    xnn_params.allocator.aligned_allocate = FailingAlignedAllocate;
  }
  ~FailingAllocatorGuard() { xnn_params.allocator = saved_allocator_; }

  FailingAllocatorGuard(const FailingAllocatorGuard&) = delete;
  FailingAllocatorGuard& operator=(const FailingAllocatorGuard&) = delete;

 private:
  static void* FailingReallocate(void* context, void* pointer, size_t size) {
    auto* self = static_cast<FailingAllocatorGuard*>(context);
    if (size == self->fail_bytes_) {
      return nullptr;
    }
    return self->saved_allocator_.reallocate(
        self->saved_allocator_.context, pointer, size);
  }

  static void* FailingAlignedAllocate(
      void* context, size_t alignment, size_t size) {
    auto* self = static_cast<FailingAllocatorGuard*>(context);
    if (size == self->fail_bytes_) {
      return nullptr;
    }
    return self->saved_allocator_.aligned_allocate(
        self->saved_allocator_.context, alignment, size);
  }

  const struct xnn_allocator saved_allocator_;
  const size_t fail_bytes_;
};

xnn_subgraph_t CreateStaticBroadcastSubgraph() {
  xnn_subgraph_t subgraph = nullptr;
  if (xnn_create_subgraph(2, 0, &subgraph) != xnn_status_success) {
    return nullptr;
  }
  const size_t dims[3] = {2, 3, 4};
  const size_t broadcast_shape[3] = {2, 3, 4};
  uint32_t input_id;
  uint32_t output_id;
  if (xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 3, dims, nullptr, 0,
          XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id) != xnn_status_success ||
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 3, dims, nullptr, 1,
          XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id) != xnn_status_success) {
    xnn_delete_subgraph(subgraph);
    return nullptr;
  }
  if (xnn_define_static_broadcast(
          subgraph, 3, broadcast_shape, input_id, output_id, 0) !=
      xnn_status_success) {
    xnn_delete_subgraph(subgraph);
    return nullptr;
  }
  return subgraph;
}

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

TEST(RUNTIME, failed_create_does_not_leave_subgraph_owning_freed_buffers) {
  xnn_subgraph_t subgraph = CreateStaticBroadcastSubgraph();
  ASSERT_NE(subgraph, nullptr);

  xnn_runtime_t runtime = nullptr;
  {
    FailingAllocatorGuard guard(sizeof(struct xnn_operator));
    const xnn_status status = xnn_create_runtime_v4(
        subgraph, nullptr, nullptr, nullptr, 0, &runtime);
    ASSERT_EQ(status, xnn_status_out_of_memory);
  }

  EXPECT_EQ(CountStaleOwnedCleanupBuffers(subgraph), 0u);
  xnn_delete_subgraph(subgraph);
}

