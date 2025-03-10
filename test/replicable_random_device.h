// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef __XNNPACK_TEST_REPLICABLE_RANDOM_NUMBER_GENERATOR_H_
#define __XNNPACK_TEST_REPLICABLE_RANDOM_NUMBER_GENERATOR_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include <gtest/gtest.h>

namespace xnnpack {

namespace internal {

// Since we spend a lot of time generating random numbers (especially on
// emulated devices), and we don't really care about long-range correlations, we
// use a fast and simple default random number generator.
class Xoshiro128Plus {
 public:
  using result_type = uint64_t;

  explicit Xoshiro128Plus(uint64_t s1) : state_{s1, 0} {}

  uint64_t operator()() {
    uint64_t s1 = state_[0];
    uint64_t s0 = state_[1];
    uint64_t result = s0 + s1;
    state_[0] = s0;
    s1 ^= s1 << 23;
    state_[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return result;
  }

  void discard(size_t count) {
    while (count--) {
      (*this)();
    }
  }
  static constexpr uint64_t min() { return 0; }
  static constexpr uint64_t max() { return ~0; }

 private:
  static uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
  }
  uint64_t state_[2];
};

}  // namespace internal

// Wraps a random generator such that on construction, the random generator is
// seeded with the underlying `UnitTest`'s random seed, which can be coerced on
// the command line.
//
// This class additionally registers a `ScopedTrace` that prints the random seed
// used on test failure, so that the (randomized) test failure can be easily
// reproduced.
class ReplicableRandomDevice {
 public:
  using BaseRandomDevice = internal::Xoshiro128Plus;
  using result_type = BaseRandomDevice::result_type;

  ReplicableRandomDevice()
      : random_seed_(testing::UnitTest::GetInstance()->random_seed()),
        random_generator_(random_seed_),
        scoped_trace_(__FILE__, __LINE__,
                      "To replicate this failure, re-run the test with "
                      "`--gunit_random_seed=" +
                          std::to_string(random_seed_) + "`.") {}

  // Wrapped methods from `BaseRandomDevice`.
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
