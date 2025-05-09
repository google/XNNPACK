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

struct LoopInfo {
  size_t begin = 0;
  size_t end = 0;
  size_t step = 1;

  LoopInfo() = default;
  LoopInfo(size_t once) : begin(once), end(once + 1), step(1) {}
  LoopInfo(size_t begin, size_t end, size_t step = 1)
      : begin(begin), end(end), step(step) {}
};

enum class OpType {
  Min,
  Max,
};

class Tester {
 public:
  explicit Tester(OpType op_type) : op_type_(op_type) {}

  OpType op_type() const { return op_type_; }

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

  // Type deduction helper.
  template <typename T, typename Params>
  using UKernelFn = void (*)(size_t, size_t, const T*, size_t, const T*, T*,
                             const Params*);

  template <typename T, typename Params>
  void Test(UKernelFn<T, Params> ukernel) const {
    xnnpack::ReplicableRandomDevice rng;
    for (size_t channels = this->channels().begin;
         channels < this->channels().end; channels += this->channels().step) {
      const size_t input_stride =
          input_stride_ == -1 ? channels : input_stride_;
      xnnpack::Buffer<T> zero(channels, 0, xnnpack::XnnExtraBytes);
      for (size_t rows = this->rows().begin; rows < this->rows().end;
           rows += this->rows().step) {
        xnnpack::Buffer<T> input((rows - 1) * input_stride + channels,
                                 xnnpack::XnnExtraBytes);
        xnnpack::Buffer<T> output(channels);

        xnnpack::DatatypeGenerator<T> input_gen;
        std::generate_n(input.data(), input.size(),
                        [&]() { return input_gen(rng); });
        std::generate_n(output.data(), output.size(),
                        [&]() { return input_gen(rng); });

        xnnpack::Buffer<T> expected(channels);
        std::copy_n(output.data(), channels, expected.data());

        ukernel(rows, channels, input.data(), input_stride * sizeof(T),
                zero.data(), output.data(), nullptr);

        for (size_t r = 0; r < rows; ++r) {
          const T* input_row = input.data() + r * input_stride;
          for (size_t c = 0; c < channels; ++c) {
            switch (op_type()) {
              case OpType::Min:
                expected[c] = std::min(expected[c], input_row[c]);
                break;
              case OpType::Max:
                expected[c] = std::max(expected[c], input_row[c]);
                break;
            }
          }
        }

        for (size_t c = 0; c < channels; ++c) {
          switch (op_type()) {
            case OpType::Min:
              ASSERT_EQ(expected[c], output[c]);
              break;
            case OpType::Max:
              ASSERT_EQ(expected[c], output[c]);
              break;
          }
        }
      }
    }
  }

  void Test(const Kernel& kernel) const;

 private:
  OpType op_type_;
  LoopInfo channels_;
  LoopInfo rows_;
  size_t input_stride_ = -1;
};

struct Kernel {
  explicit Kernel(xnn_f32_rdminmax_ukernel_fn fn) {
    dispatch = [=](const Tester& tester) { tester.Test(fn); };
  }
  explicit Kernel(xnn_f16_rdminmax_ukernel_fn fn) {
    dispatch = [=](const Tester& tester) { tester.Test(fn); };
  }
  explicit Kernel(xnn_s8_rdminmax_ukernel_fn fn) {
    dispatch = [=](const Tester& tester) { tester.Test(fn); };
  }
  explicit Kernel(xnn_u8_rdminmax_ukernel_fn fn) {
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
  size_t row_tile;
  size_t channel_tile;
  bool vector_tile;
  size_t elem_size;
};

KernelInfo kernels[] = {
#define XNN_UKERNEL(arch_flags, ukernel, row_tile, channel_tile, vector_tile, \
                    datatype_in, datatype_out)                                \
  {#ukernel, arch_flags,   Kernel{ukernel}, OpType::Max,                      \
   row_tile, channel_tile, vector_tile,     sizeof(datatype_in)},
#include "src/f16-rdminmax/f16-rdmax.h"
#include "src/f32-rdminmax/f32-rdmax.h"
#include "src/s8-rdminmax/s8-rdmax.h"
#include "src/u8-rdminmax/u8-rdmax.h"
#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, ukernel, row_tile, channel_tile, vector_tile, \
                    datatype_in, datatype_out)                                \
  {#ukernel, arch_flags,   Kernel{ukernel}, OpType::Min,                      \
   row_tile, channel_tile, vector_tile,     sizeof(datatype_in)},
#include "src/f16-rdminmax/f16-rdmin.h"
#include "src/f32-rdminmax/f32-rdmin.h"
#include "src/s8-rdminmax/s8-rdmin.h"
#include "src/u8-rdminmax/u8-rdmin.h"
#undef XNN_UKERNEL
};

class Test : public testing::TestWithParam<KernelInfo> {};

INSTANTIATE_TEST_SUITE_P(
    rminmax, Test, testing::ValuesIn(kernels),
    [](const testing::TestParamInfo<Test::ParamType>& info) {
      return info.param.name;
    });

TEST_P(Test, channels_eq_2pass_fulltile) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester(param.op_type)
      .channels(param.channel_tile)
      .rows(param.row_tile * 2)
      .Test(param.kernel);
}

TEST_P(Test, channels_eq_2pass_fulltile_with_input_stride) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester(param.op_type)
      .channels(param.channel_tile)
      .rows(param.row_tile * 2)
      .input_stride(param.channel_tile + 5)
      .Test(param.kernel);
}

TEST_P(Test, channels_eq_2pass_subtile) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester(param.op_type)
      .channels(param.channel_tile)
      .rows({1, param.row_tile * 2})
      .Test(param.kernel);
}

TEST_P(Test, channels_eq_2pass_subtile_with_input_stride) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester(param.op_type)
      .channels(param.channel_tile)
      .rows({1, param.row_tile * 2})
      .input_stride(param.channel_tile + 5)
      .Test(param.kernel);
}

TEST_P(Test, channels_eq_multipass_fulltile) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester(param.op_type)
      .channels(param.channel_tile)
      .rows({param.row_tile, param.row_tile * 4, param.row_tile})
      .Test(param.kernel);
}

TEST_P(Test, channels_eq_multipass_fulltile_with_input_stride) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester(param.op_type)
      .channels(param.channel_tile)
      .rows({param.row_tile, param.row_tile * 4, param.row_tile})
      .input_stride(param.channel_tile + 5)
      .Test(param.kernel);
}

TEST_P(Test, channels_div_2pass_fulltile) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester(param.op_type)
      .channels(param.channel_tile * 2)
      .rows(param.row_tile * 2)
      .Test(param.kernel);
}

TEST_P(Test, channels_div_2pass_subtile) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester(param.op_type)
      .channels(param.channel_tile * 2)
      .rows({1, param.row_tile * 2})
      .Test(param.kernel);
}

TEST_P(Test, channels_div_multipass_fulltile) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester(param.op_type)
      .channels(
          {param.channel_tile * 2, param.channel_tile * 8, param.channel_tile})
      .rows({param.row_tile, param.row_tile * 4, param.row_tile})
      .Test(param.kernel);
}

TEST_P(Test, channels_div_multipass_fulltile_with_input_stride) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester(param.op_type)
      .channels(
          {param.channel_tile * 2, param.channel_tile * 8, param.channel_tile})
      .rows({param.row_tile, param.row_tile * 4, param.row_tile})
      .input_stride(param.channel_tile * 8 + 5)
      .Test(param.kernel);
}

TEST_P(Test, channels_lt_2pass_fulltile) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester(param.op_type)
      .channels(param.row_tile * 2)
      .rows(param.row_tile)
      .Test(param.kernel);
}

TEST_P(Test, channels_lt_2pass_subtile) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester(param.op_type)
      .channels({1, param.channel_tile})
      .rows({1, param.row_tile * 2})
      .Test(param.kernel);
}

TEST_P(Test, channels_lt_multipass_fulltile) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester(param.op_type)
      .channels({1, param.channel_tile})
      .rows({param.row_tile, param.row_tile * 4, param.row_tile})
      .Test(param.kernel);
}

TEST_P(Test, channels_lt_multipass_fulltile_with_input_stride) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester(param.op_type)
      .channels({1, param.channel_tile})
      .rows({param.row_tile, param.row_tile * 4, param.row_tile})
      .input_stride(param.channel_tile + 5)
      .Test(param.kernel);
}

TEST_P(Test, channels_gt_2pass_fulltile) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester(param.op_type)
      .rows(param.row_tile * 2)
      .channels({param.channel_tile + 1, param.channel_tile * 2})
      .Test(param.kernel);
}

TEST_P(Test, channels_gt_2pass_subtile) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester(param.op_type)
      .channels({param.channel_tile + 1, param.channel_tile * 2})
      .rows({1, param.row_tile * 2})
      .Test(param.kernel);
}

TEST_P(Test, channels_gt_multipass_fulltile) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester(param.op_type)
      .channels({param.channel_tile + 1, param.channel_tile * 2})
      .rows({param.row_tile, param.row_tile * 4, param.row_tile})
      .Test(param.kernel);
}

TEST_P(Test, channels_gt_multipass_fulltile_with_input_stride) {
  const KernelInfo& param = GetParam();
  TEST_REQUIRES_ARCH_FLAGS(param.arch_flags);
  Tester(param.op_type)
      .channels({param.channel_tile + 1, param.channel_tile * 2})
      .rows({param.row_tile, param.row_tile * 4, param.row_tile})
      .input_stride(param.channel_tile * 2 + 5)
      .Test(param.kernel);
}
