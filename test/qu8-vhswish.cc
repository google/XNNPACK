// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: qu8-vhswish
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
#include "vhswish-microkernel-tester.h"

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype, params_type, init_params)\
                                                                                                                 \
XNN_TEST_UNARY_BATCH_EQ(ukernel, arch_flags, batch_tile, datatype, ukernel, init_params);                        \
XNN_TEST_UNARY_BATCH_DIV(ukernel, arch_flags, batch_tile, datatype, ukernel, init_params);                       \
XNN_TEST_UNARY_BATCH_LT(ukernel, arch_flags, batch_tile, datatype, ukernel, init_params);                        \
XNN_TEST_UNARY_BATCH_GT(ukernel, arch_flags, batch_tile, datatype, ukernel, init_params);                        \
                                                                                                                 \
XNN_TEST_UNARY_INPLACE(ukernel, arch_flags, batch_tile, datatype, ukernel, init_params);                         \
TEST(ukernel, input_scale) {                                                                                     \
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                          \
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {                                               \
    for (float input_scale : {4.0f, 16.0f, 64.0f}) {                                                             \
      VHSwishMicrokernelTester()                                                                                 \
        .batch_size(batch_size)                                                                                  \
        .input_scale(input_scale)                                                                                \
        .input_zero_point(150)                                                                                   \
        .output_zero_point(100)                                                                                  \
        .Test(ukernel, init_params);                                                                             \
      }                                                                                                          \
  }                                                                                                              \
}                                                                                                                \
                                                                                                                 \
TEST(ukernel, output_scale) {                                                                                    \
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                          \
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {                                               \
    for (float output_scale : {4.0f, 16.0f, 64.0f}) {                                                            \
      VHSwishMicrokernelTester()                                                                                 \
        .batch_size(batch_size)                                                                                  \
        .output_scale(output_scale)                                                                              \
        .input_zero_point(150)                                                                                   \
        .output_zero_point(100)                                                                                  \
        .Test(ukernel, init_params);                                                                             \
      }                                                                                                          \
  }                                                                                                              \
}                                                                                                                \
                                                                                                                 \
TEST(ukernel, input_zero_point) {                                                                                \
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                          \
  for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {                             \
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {                                             \
      VHSwishMicrokernelTester()                                                                                 \
        .batch_size(batch_size)                                                                                  \
        .input_zero_point(input_zero_point)                                                                      \
        .output_zero_point(100)                                                                                  \
        .Test(ukernel, init_params);                                                                             \
    }                                                                                                            \
  }                                                                                                              \
}                                                                                                                \
                                                                                                                 \
TEST(ukernel, output_zero_point) {                                                                               \
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                          \
  for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {                          \
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {                                             \
      VHSwishMicrokernelTester()                                                                                 \
        .batch_size(batch_size)                                                                                  \
        .input_zero_point(150)                                                                                   \
        .output_zero_point(output_zero_point)                                                                    \
        .Test(ukernel, init_params);                                                                             \
    }                                                                                                            \
  }                                                                                                              \
}
#include "qu8-vhswish/qu8-vhswish.h"
#undef XNN_UKERNEL_WITH_PARAMS
