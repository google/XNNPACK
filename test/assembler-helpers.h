// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <iomanip>
#include <ios>

// clang-format off
#define EXPECT_INSTR(expected, actual)                                                                        \
  EXPECT_EQ(expected, actual) << "expected = 0x" << std::hex << std::setw(8) << std::setfill('0') << expected \
                              << std::endl << "  actual = 0x" << actual;
// clang-format on

#define CHECK_ENCODING(expected, call)   \
  a.reset();                             \
  call;                                  \
  EXPECT_EQ(Error::kNoError, a.error()); \
  EXPECT_INSTR(expected, *reinterpret_cast<const uint32_t*>(a.start()))

#define EXPECT_ERROR(expected, call) \
  a.reset();                         \
  call;                              \
  EXPECT_EQ(expected, a.error());

namespace xnnpack {

// Arguments are: input (r0|x0), output (r1|x1), params (r2|x2).
typedef void (*JitF32HardswishFn)(float*, float*, void*);

// Reference implementation of hardswish
constexpr float hardswish(float x) {
  return x * std::min(std::max(0.0f, (x + 3.0f)), 6.0f) / 6.0f;
}

}  // namespace xnnpack
