// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <functional>

#include <gtest/gtest.h>
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"  // IWYU pragma: keep
#include "src/xnnpack/hardware-config.h"  // IWYU pragma: keep
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/reduce.h"  // IWYU pragma: keep
#include "test/replicable_random_device.h"

struct Kernel;

enum class OpType {
  Min,
  Max,
  MinMax,
};

class Tester {
 public:
  explicit Tester(OpType op_type) : op_type_(op_type) {}

  OpType op_type() const { return op_type_; }

  Tester& batch_size(size_t size) {
    batch_size_ = size;
    return *this;
  }
  size_t batch_size() const { return batch_size_; }

  // Type deduction helper.
  template <typename T, typename Params>
  using UKernelFn = void (*)(size_t, const T*, T*, const Params*);

  template <typename T, typename Params>
  void Test(UKernelFn<T, Params> ukernel) const {
    xnnpack::ReplicableRandomDevice rng;
    xnnpack::Buffer<T> input(batch_size(), xnnpack::XnnExtraBytes);
    xnnpack::DatatypeGenerator<T> input_gen;
    std::generate_n(input.data(), input.size(),
                    [&]() { return input_gen(rng); });

    T output[2] = {input_gen(rng), input_gen(rng)};
    T expected[2]{output[0], output[1]};

    ukernel(batch_size() * sizeof(T), input.data(), output, nullptr);

    for (size_t i = 0; i < batch_size(); ++i) {
      switch (op_type()) {
        case OpType::Min:
          expected[0] = std::min(expected[0], input[i]);
          break;
        case OpType::Max:
          expected[0] = std::max(expected[0], input[i]);
          break;
        case OpType::MinMax:
          expected[0] = std::min(expected[0], input[i]);
          expected[1] = std::max(expected[1], input[i]);
          break;
      }
    }

    switch (op_type()) {
      case OpType::Min:
        ASSERT_EQ(expected[0], output[0]);
        break;
      case OpType::Max:
        ASSERT_EQ(expected[0], output[0]);
        break;
      case OpType::MinMax:
        ASSERT_EQ(expected[0], output[0]);
        ASSERT_EQ(expected[1], output[1]);
        break;
    }
  }

  void Test(const Kernel& kernel) const;

 private:
  OpType op_type_;
  size_t batch_size_;
};

struct Kernel {
  explicit Kernel(xnn_f32_reduce_ukernel_fn fn) {
    dispatch = [=](const Tester& tester) { tester.Test(fn); };
  }
  explicit Kernel(xnn_f16_reduce_ukernel_fn fn) {
    dispatch = [=](const Tester& tester) { tester.Test(fn); };
  }
  explicit Kernel(xnn_s8_reduce_ukernel_fn fn) {
    dispatch = [=](const Tester& tester) { tester.Test(fn); };
  }
  explicit Kernel(xnn_u8_reduce_ukernel_fn fn) {
    dispatch = [=](const Tester& tester) { tester.Test(fn); };
  }
  std::function<void(const Tester&)> dispatch;
};

void Tester::Test(const Kernel& kernel) const { kernel.dispatch(*this); }

struct KernelInfo {
  const char* name;
  uint64_t arch_flags;
  Kernel kernel;
  OpType op_type;
  size_t batch_tile;
  bool vector_tile;
  size_t elem_size;
};

KernelInfo kernels[] = {
#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype_in, \
                    datatype_out)                                              \
  {#ukernel,   arch_flags,  Kernel{ukernel},    OpType::Max,                   \
   batch_tile, vector_tile, sizeof(datatype_in)},
#include "src/f16-rminmax/f16-rmax.h"
#include "src/f32-rminmax/f32-rmax.h"
#include "src/s8-rminmax/s8-rmax.h"
#include "src/u8-rminmax/u8-rmax.h"
#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype_in, \
                    datatype_out)                                              \
  {#ukernel,   arch_flags,  Kernel{ukernel},    OpType::Min,                   \
   batch_tile, vector_tile, sizeof(datatype_in)},
#include "src/f16-rminmax/f16-rmin.h"
#include "src/f32-rminmax/f32-rmin.h"
#include "src/s8-rminmax/s8-rmin.h"
#include "src/u8-rminmax/u8-rmin.h"
#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype_in, \
                    datatype_out)                                              \
  {#ukernel,   arch_flags,  Kernel{ukernel},    OpType::MinMax,                \
   batch_tile, vector_tile, sizeof(datatype_in)},
#include "src/f16-rminmax/f16-rminmax.h"
#include "src/f32-rminmax/f32-rminmax.h"
#include "src/s8-rminmax/s8-rminmax.h"
#include "src/u8-rminmax/u8-rminmax.h"
#undef XNN_UKERNEL
};

class Test : public testing::TestWithParam<KernelInfo> {};

INSTANTIATE_TEST_SUITE_P(
    rminmax, Test, testing::ValuesIn(kernels),
    [](const testing::TestParamInfo<Test::ParamType>& info) {
      return info.param.name;
    });

TEST_P(Test, batch_eq) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  const size_t batch_tile = param.batch_tile * get_batch_scale(param.elem_size);
  Tester(param.op_type)
      .batch_size(batch_tile * get_batch_scale(param.elem_size))
      .Test(param.kernel);
}

TEST_P(Test, batch_div) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  const size_t batch_tile = param.batch_tile * get_batch_scale(param.elem_size);
  for (size_t batch_size = batch_tile; batch_size < batch_tile * 5;
       batch_size += batch_tile) {
    Tester(param.op_type).batch_size(batch_tile).Test(param.kernel);
  }
}

TEST_P(Test, batch_lt) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  const size_t batch_tile = param.batch_tile * get_batch_scale(param.elem_size);
  for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
    Tester(param.op_type).batch_size(batch_size).Test(param.kernel);
  }
}

TEST_P(Test, batch_gt) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  const size_t batch_tile = param.batch_tile * get_batch_scale(param.elem_size);
  for (size_t batch_size = batch_tile + 1; batch_size < batch_tile * 2;
       batch_size++) {
    Tester(param.op_type).batch_size(batch_size).Test(param.kernel);
  }
}
