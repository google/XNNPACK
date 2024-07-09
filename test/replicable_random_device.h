// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef __XNNPACK_TEST_REPLICABLE_RANDOM_NUMBER_GENERATOR_H_
#define __XNNPACK_TEST_REPLICABLE_RANDOM_NUMBER_GENERATOR_H_

#include <cstddef>
#include <random>
#include <string>

#include <gtest/gtest.h>
#include "xnnpack.h"

namespace xnnpack {

using BaseRandomDevice = std::mt19937;

// Wraps `std::mt19937` such that on construction, the random generator is
// seeded with the underlying `UnitTest`'s random seed, which can be coerced on
// the command line.
//
// This class additionally registers a `ScopedTrace` that prints the random seed
// used on test failure, so that the (randomized) test failure can be easily
// reproduced.
class ReplicableRandomDevice {
 public:
  typedef BaseRandomDevice::result_type result_type;
  ReplicableRandomDevice()
      : random_seed_(testing::UnitTest::GetInstance()->random_seed()),
        random_generator_(random_seed_),
        scoped_trace_(__FILE__, __LINE__,
                      "To replicate this failure, re-run the test with "
                      "`--gunit_random_seed=" +
                          std::to_string(random_seed_) + "`.") {}

  // Wrapped methods from `std::mt19937`.
  result_type operator()() { return random_generator_(); }
  void discard(size_t count) { random_generator_.discard(count); }
  static constexpr result_type min() { return BaseRandomDevice::min(); }
  static constexpr result_type max() { return BaseRandomDevice::max(); }

 private:
  const int random_seed_;
  BaseRandomDevice random_generator_;
  testing::ScopedTrace scoped_trace_;
};

}  // namespace xnnpack

#endif  // __XNNPACK_TEST_REPLICABLE_RANDOM_NUMBER_GENERATOR_H_
