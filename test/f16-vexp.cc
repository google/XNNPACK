// clang-format off
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: f16-vexp
//   Generator: tools/generate-vunary-test.py


#include <array>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"
#include "test/next_prime.h"
#include "test/unary-ops.h"
#include "test/vunary-microkernel-tester.h"

using TestInfo = Exp;

#define XNN_QUANTIZED(T) xnnpack::quantized<T>
#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype, params_type, init_params)                   \
  TEST(ukernel, batch_eq) { TestBatchEq<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }  \
  TEST(ukernel, batch_div) { TestBatchDiv<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }\
  TEST(ukernel, batch_lt) { TestBatchLT<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }  \
  TEST(ukernel, batch_gt) { TestBatchGT<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }  \
  TEST(ukernel, inplace) { TestInPlace<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }   \
TEST(ukernel, propagates_nan) {                                                                                         \
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                                 \
  const size_t n = 3 * batch_tile * get_batch_scale<datatype>() - 1;                                                    \
  const std::array<float, 4> values = {NAN, 0.0f, -NAN, 1.0f};                                                          \
  xnnpack::Buffer<datatype> inputs(n, xnnpack::XnnExtraBytes);                                                          \
  xnnpack::Buffer<datatype> outputs(n);                                                                                 \
  for (size_t rotation = 0; rotation < values.size(); rotation++) {                                                     \
    for (size_t i = 0; i < n; i++) {                                                                                    \
      inputs[i] = values[(i + rotation) % values.size()];                                                               \
    }                                                                                                                   \
    ukernel(n * sizeof(datatype), inputs.data(), outputs.data(), nullptr);                                              \
    for (size_t i = 0; i < n; i++) {                                                                                    \
      EXPECT_EQ(std::fpclassify(static_cast<float>(outputs[i])), (i + rotation) % 2 == 0 ? FP_NAN : FP_NORMAL)          \
          << "at " << i << ", rotation " << rotation;                                                                   \
    }                                                                                                                   \
  }                                                                                                                     \
}
#include "src/f16-vexp/f16-vexp.inc"
#undef XNN_UKERNEL
#undef XNN_QUANTIZED
