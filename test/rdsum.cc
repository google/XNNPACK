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

struct LoopInfo {
  size_t begin = 0;
  size_t end = 0;
  size_t step = 1;

  LoopInfo() = default;
  LoopInfo(size_t once) : begin(once), end(once + 1), step(1) {}
  LoopInfo(size_t begin, size_t end, size_t step = 1)
      : begin(begin), end(end), step(step) {}
};

class Tester {
 public:
  Tester& channels(LoopInfo value) {
    channels_ = value;
    return *this;
  }
  LoopInfo channels() const { return channels_; }

  Tester& rows(LoopInfo value) {
    rows_ = value;
    return *this;
  }
  LoopInfo rows() const { return rows_; }

  Tester& input_stride(size_t value) {
    input_stride_ = value;
    return *this;
  }
  size_t input_stride() const { return input_stride_; }

  Tester& scale(float scale) {
    scale_ = scale;
    return *this;
  }
  float scale() const { return scale_; }

  // Type deduction helper.
  template <typename Input, typename Output, typename Params>
  using UKernelFn = void (*)(size_t, size_t, const Input*, size_t, const Input*,
                             Output*, const Params*);

  template <typename Input, typename Output, typename Params,
            typename InitParams>
  void Test(UKernelFn<Input, Output, Params> ukernel,
            InitParams init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    for (size_t channels = this->channels().begin;
         channels < this->channels().end; channels += this->channels().step) {
      const size_t input_stride =
          input_stride_ == -1 ? channels : input_stride_;
      xnnpack::Buffer<Input> zero(channels, 0, xnnpack::XnnExtraBytes);
      for (size_t rows = this->rows().begin; rows < this->rows().end;
           rows += this->rows().step) {
        xnnpack::Buffer<Input> input((rows - 1) * input_stride + channels,
                                     xnnpack::XnnExtraBytes);
        xnnpack::Buffer<Output> output(channels);

        const float max_abs_value = 10.0f;
        xnnpack::DatatypeGenerator<Input> input_gen(-max_abs_value,
                                                    max_abs_value);
        std::generate_n(input.data(), input.size(),
                        [&]() { return input_gen(rng); });
        xnnpack::DatatypeGenerator<Output> output_gen(-max_abs_value,
                                                      max_abs_value);
        std::generate_n(output.data(), output.size(),
                        [&]() { return output_gen(rng); });

        xnnpack::Buffer<Output> expected(channels, static_cast<Output>(0));
        for (size_t r = 0; r < rows; ++r) {
          const Input* input_row = input.data() + r * input_stride;
          for (size_t c = 0; c < channels; ++c) {
            expected[c] += input_row[c];
          }
        }

        // Note accumulation with output happens after scale.
        const float scale = init_params ? this->scale() : 1.0f;
        for (size_t c = 0; c < channels; ++c) {
          expected[c] *= scale;
          expected[c] += output[c];
        }

        Params params;
        if (init_params) {
          init_params(&params, scale);
        }

        ukernel(rows, channels, input.data(), input_stride * sizeof(Input),
                zero.data(), output.data(), &params);

        const float tolerance = channels * max_abs_value * scale * 2.0f *
                                xnnpack::NumericLimits<Output>::epsilon();
        for (size_t c = 0; c < channels; ++c) {
          ASSERT_NEAR(expected[c], output[c], tolerance);
        }
      }
    }
  }

  void Test(const Kernel& kernel) const;

 private:
  LoopInfo channels_;
  LoopInfo rows_;
  size_t input_stride_ = -1;
  float scale_ = 1.0f;
};

struct Kernel {
  explicit Kernel(xnn_f32_rdsum_ukernel_fn fn,
                  xnn_init_f32_scale_params_fn init_params) {
    dispatch = [=](const Tester& tester) { tester.Test(fn, init_params); };
  }
  explicit Kernel(xnn_f16_f32acc_rdsum_ukernel_fn fn,
                  xnn_init_f16_f32acc_scale_params_fn init_params) {
    dispatch = [=](const Tester& tester) { tester.Test(fn, init_params); };
  }
  explicit Kernel(xnn_qs8_rdsum_ukernel_fn fn,
                  xnn_init_qs8_rsum_params_fn init_params) {
    dispatch = [=](const Tester& tester) { tester.Test(fn, init_params); };
  }
  explicit Kernel(xnn_qu8_rdsum_ukernel_fn fn,
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
  size_t row_tile;
  size_t channel_tile;
  bool vector_tile;
  size_t elem_size;
};

KernelInfo kernels[] = {
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, row_tile, channel_tile, \
                                vector_tile, datatype_in, datatype_out,      \
                                params_type, init_params)                    \
  {#ukernel,     arch_flags,  Kernel{ukernel, init_params}, row_tile,        \
   channel_tile, vector_tile, sizeof(datatype_in)},
#include "src/f16-f32acc-rdsum/f16-f32acc-rdsum.h"
#include "src/f32-rdsum/f32-rdsum.h"
#include "src/qs8-rdsum/qs8-rdsum-minmax-fp32.h"
#include "src/qu8-rdsum/qu8-rdsum.h"
#undef XNN_UKERNEL_WITH_PARAMS
};

class Test : public testing::TestWithParam<KernelInfo> {};

INSTANTIATE_TEST_SUITE_P(
    rdsum, Test, testing::ValuesIn(kernels),
    [](const testing::TestParamInfo<Test::ParamType>& info) {
      return info.param.name;
    });

TEST_P(Test, channels_eq_2pass_fulltile) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester()
      .channels(param.channel_tile)
      .rows(param.row_tile * 2)
      .Test(param.kernel);
}

TEST_P(Test, channels_eq_2pass_fulltile_with_input_stride) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester()
      .channels(param.channel_tile)
      .rows(param.row_tile * 2)
      .input_stride(param.channel_tile + 5)
      .Test(param.kernel);
}

TEST_P(Test, channels_eq_2pass_subtile) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester()
      .channels(param.channel_tile)
      .rows({1, param.row_tile * 2})
      .Test(param.kernel);
}

TEST_P(Test, channels_eq_2pass_subtile_with_input_stride) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester()
      .channels(param.channel_tile)
      .rows({1, param.row_tile * 2})
      .input_stride(param.channel_tile + 5)
      .Test(param.kernel);
}

TEST_P(Test, channels_eq_multipass_fulltile) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester()
      .channels(param.channel_tile)
      .rows({param.row_tile, param.row_tile * 4, param.row_tile})
      .Test(param.kernel);
}

TEST_P(Test, channels_eq_multipass_fulltile_with_input_stride) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester()
      .channels(param.channel_tile)
      .rows({param.row_tile, param.row_tile * 4, param.row_tile})
      .input_stride(param.channel_tile + 5)
      .Test(param.kernel);
}

TEST_P(Test, channels_div_2pass_fulltile) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester()
      .channels(param.channel_tile * 2)
      .rows(param.row_tile * 2)
      .Test(param.kernel);
}

TEST_P(Test, channels_div_2pass_subtile) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester()
      .channels(param.channel_tile * 2)
      .rows({1, param.row_tile * 2})
      .Test(param.kernel);
}

TEST_P(Test, channels_div_multipass_fulltile) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester()
      .channels(
          {param.channel_tile * 2, param.channel_tile * 8, param.channel_tile})
      .rows({param.row_tile, param.row_tile * 4, param.row_tile})
      .Test(param.kernel);
}

TEST_P(Test, channels_div_multipass_fulltile_with_input_stride) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester()
      .channels(
          {param.channel_tile * 2, param.channel_tile * 8, param.channel_tile})
      .rows({param.row_tile, param.row_tile * 4, param.row_tile})
      .input_stride(param.channel_tile * 8 + 5)
      .Test(param.kernel);
}

TEST_P(Test, channels_lt_2pass_fulltile) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester().channels(param.row_tile * 2).rows(param.row_tile).Test(param.kernel);
}

TEST_P(Test, channels_lt_2pass_subtile) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester()
      .channels({1, param.channel_tile})
      .rows({1, param.row_tile * 2})
      .Test(param.kernel);
}

TEST_P(Test, channels_lt_multipass_fulltile) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester()
      .channels({1, param.channel_tile})
      .rows({param.row_tile, param.row_tile * 4, param.row_tile})
      .Test(param.kernel);
}

TEST_P(Test, channels_lt_multipass_fulltile_with_input_stride) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester()
      .channels({1, param.channel_tile})
      .rows({param.row_tile, param.row_tile * 4, param.row_tile})
      .input_stride(param.channel_tile + 5)
      .Test(param.kernel);
}

TEST_P(Test, channels_gt_2pass_fulltile) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester()
      .rows(param.row_tile * 2)
      .channels({param.channel_tile + 1, param.channel_tile * 2})
      .Test(param.kernel);
}

TEST_P(Test, channels_gt_2pass_subtile) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester()
      .channels({param.channel_tile + 1, param.channel_tile * 2})
      .rows({1, param.row_tile * 2})
      .Test(param.kernel);
}

TEST_P(Test, channels_gt_multipass_fulltile) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester()
      .channels({param.channel_tile + 1, param.channel_tile * 2})
      .rows({param.row_tile, param.row_tile * 4, param.row_tile})
      .Test(param.kernel);
}

TEST_P(Test, channels_gt_multipass_fulltile_with_input_stride) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester()
      .channels({param.channel_tile + 1, param.channel_tile * 2})
      .rows({param.row_tile, param.row_tile * 4, param.row_tile})
      .input_stride(param.channel_tile * 2 + 5)
      .Test(param.kernel);
}

TEST_P(Test, overflow_accumulator) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester()
      .channels({param.channel_tile + 1, param.channel_tile * 2})
      .rows(512)
      .Test(param.kernel);
}