// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: f32-vtanh
//   Generator: tools/generate-vunary-test.py


#include <array>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"
#include "next_prime.h"
#include "vunary-microkernel-tester.h"

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype, params_type, init_params)\
                                                                                                                 \
XNN_TEST_UNARY_BATCH_EQ(ukernel, arch_flags, batch_tile, datatype, ukernel, init_params);                        \
XNN_TEST_UNARY_BATCH_DIV(ukernel, arch_flags, batch_tile, datatype, ukernel, init_params);                       \
XNN_TEST_UNARY_BATCH_LT(ukernel, arch_flags, batch_tile, datatype, ukernel, init_params);                        \
XNN_TEST_UNARY_BATCH_GT(ukernel, arch_flags, batch_tile, datatype, ukernel, init_params);                        \
                                                                                                                 \
XNN_TEST_UNARY_INPLACE(ukernel, arch_flags, batch_tile, datatype, ukernel, init_params);                         \
TEST(ukernel, special_values) {                                                                                  \
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                          \
  constexpr size_t num_elements = 7;                                                                             \
  constexpr size_t buffered_size =                                                                               \
      num_elements + XNN_EXTRA_BYTES / sizeof(float);                                                            \
  std::array<float, buffered_size> inputs =                                                                      \
      {0.0f, -0.0f, 10.0f, -10.0f, INFINITY, -INFINITY, NAN};                                                    \
  std::array<float, num_elements> expected =                                                                     \
      {0.0f, -0.0f, 1.0f, -1.0f, 1.0f, -1.0f, NAN};                                                              \
  std::array<float, buffered_size> outputs;                                                                      \
  union xnn_f32_tanh_params params;                                                                              \
  if (init_params) {                                                                                             \
    init_params(&params);                                                                                        \
  }                                                                                                              \
  ukernel(                                                                                                       \
      num_elements * sizeof(float), inputs.data(), outputs.data(), &params);                                     \
  for (int i = 0; i < num_elements; i++) {                                                                       \
    if (std::isfinite(expected[i])) {                                                                            \
      EXPECT_NEAR(                                                                                               \
          expected[i], outputs[i],                                                                               \
          3 * std::abs(expected[i]) * std::numeric_limits<float>::epsilon())                                     \
          << "for input " << inputs[i];                                                                          \
    } else {                                                                                                     \
      EXPECT_EQ(std::fpclassify(expected[i]), std::fpclassify(outputs[i]))                                       \
          << "for input " << inputs[i] << " and output " << outputs[i]                                           \
          << " (FP_INFINITE=" << FP_INFINITE << ", FP_NAN=" << FP_NAN                                            \
          << ", FP_NORMAL=" << FP_NORMAL << ", FP_SUBNORMAL=" << FP_SUBNORMAL                                    \
          << ", FP_ZERO=" << FP_ZERO << ")";                                                                     \
    }                                                                                                            \
  }                                                                                                              \
}
#include "src/f32-vtanh/f32-vtanh.h"
#undef XNN_UKERNEL_WITH_PARAMS
