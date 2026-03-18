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
#include "src/xnnpack/microparams-init.h"  // IWYU pragma: keep
#include "src/xnnpack/reduce.h"  // IWYU pragma: keep
#include "test/replicable_random_device.h"

struct Kernel;

class Tester {
 public:
  Tester& batch_size(size_t size) {
    batch_size_ = size;
    return *this;
  }
  size_t batch_size() const { return batch_size_; }

  Tester& scale(float scale) {
    scale_ = scale;
    return *this;
  }
  float scale() const { return scale_; }

  // Type deduction helper.
  template <typename Input, typename Output, typename Params>
  using UKernelFn = void (*)(size_t, const Input*, Output*, const Params*);

  template <typename Input, typename Output, typename Params,
            typename InitParams>
  void Test(UKernelFn<Input, Output, Params> ukernel,
            InitParams init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    xnnpack::Buffer<Input> input(batch_size(), xnnpack::XnnExtraBytes);

    const float max_abs_value = 10.0f;
    xnnpack::DatatypeGenerator<Input> input_gen(-max_abs_value, max_abs_value);
    std::generate_n(input.data(), input.size(),
                    [&]() { return input_gen(rng); });

    xnnpack::DatatypeGenerator<Output> output_gen(-max_abs_value,
                                                  max_abs_value);
    Output output = output_gen(rng);

    float expected = 0.0f;
    for (size_t i = 0; i < batch_size(); ++i) {
      expected += input[i];
    }

    // Note accumulation with output happens after scale.
    const float scale = init_params ? this->scale() : 1.0f;
    expected *= scale;
    expected += output;

    Params params;
    if (init_params) {
      init_params(&params, scale);
    }

    ukernel(batch_size() * sizeof(Input), input.data(), &output, &params);

    const float tolerance = batch_size() * max_abs_value * scale * 2.0f *
                            xnnpack::NumericLimits<Output>::epsilon();
    ASSERT_NEAR(expected, output, tolerance);
  }

  void Test(const Kernel& kernel) const;

 private:
  size_t batch_size_;
  float scale_ = 1.0f;
};

struct Kernel {
  explicit Kernel(xnn_f32_rsum_ukernel_fn fn,
                  xnn_init_f32_scale_params_fn init_params) {
    dispatch = [=](const Tester& tester) { tester.Test(fn, init_params); };
  }
  explicit Kernel(xnn_f16_rsum_ukernel_fn fn,
                  xnn_init_f16_scale_params_fn init_params) {
    dispatch = [=](const Tester& tester) { tester.Test(fn, init_params); };
  }
  explicit Kernel(xnn_f16_f32acc_rsum_ukernel_fn fn,
                  xnn_init_f16_f32acc_scale_params_fn init_params) {
    dispatch = [=](const Tester& tester) { tester.Test(fn, init_params); };
  }
  explicit Kernel(xnn_qs8_rsum_ukernel_fn fn,
                  xnn_init_qs8_rsum_params_fn init_params) {
    dispatch = [=](const Tester& tester) { tester.Test(fn, init_params); };
  }
  explicit Kernel(xnn_qu8_rsum_ukernel_fn fn,
                  xnn_init_qs8_rsum_params_fn init_params) {
    dispatch = [=](const Tester& tester) { tester.Test(fn, init_params); };
  }
  std::function<void(const Tester&)> dispatch;
};

void Tester::Test(const Kernel& kernel) const { kernel.dispatch(*this); }

struct KernelInfo {
  const char* name;
  uint64_t arch_flags;
  Kernel kernel;
  size_t batch_tile;
  bool vector_tile;
  size_t elem_size;
};

KernelInfo kernels[] = {
#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype_in, \
                    datatype_out, params_type, init_params)                    \
  {#ukernel,   arch_flags,  Kernel{ukernel, init_params},                      \
   batch_tile, vector_tile, sizeof(datatype_in)},
#include "src/f16-f32acc-rsum/f16-f32acc-rsum.inc"
#include "src/f16-rsum/f16-rsum.inc"
#include "src/f32-rsum/f32-rsum.inc"
#include "src/qs8-rsum/qs8-rsum.inc"
#include "src/qu8-rsum/qu8-rsum.inc"
#undef XNN_UKERNEL
};

class Test : public testing::TestWithParam<KernelInfo> {};

INSTANTIATE_TEST_SUITE_P(
    rsum, Test, testing::ValuesIn(kernels),
    [](const testing::TestParamInfo<Test::ParamType>& info) {
      return info.param.name;
    });

TEST_P(Test, batch_eq) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  const size_t batch_tile = param.batch_tile * get_batch_scale(param.elem_size);
  Tester()
      .batch_size(batch_tile * get_batch_scale(param.elem_size))
      .Test(param.kernel);
}

TEST_P(Test, batch_div) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  const size_t batch_tile = param.batch_tile * get_batch_scale(param.elem_size);
  for (size_t batch_size = batch_tile; batch_size < batch_tile * 5;
       batch_size += batch_tile) {
    Tester().batch_size(batch_tile).Test(param.kernel);
  }
}

TEST_P(Test, batch_lt) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  const size_t batch_tile = param.batch_tile * get_batch_scale(param.elem_size);
  for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
    Tester().batch_size(batch_size).Test(param.kernel);
  }
}

TEST_P(Test, batch_gt) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  const size_t batch_tile = param.batch_tile * get_batch_scale(param.elem_size);
  for (size_t batch_size = batch_tile + 1; batch_size < batch_tile * 2;
       batch_size++) {
    Tester().batch_size(batch_size).Test(param.kernel);
  }
}

TEST_P(Test, scale) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
    Tester().batch_size(2).scale(scale).Test(param.kernel);
  }
}

TEST_P(Test, overflow_accumulator) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester().batch_size(128).Test(param.kernel);
}
