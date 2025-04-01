// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef __XNNPACK_TEST_REPLICABLE_RANDOM_NUMBER_GENERATOR_H_
#define __XNNPACK_TEST_REPLICABLE_RANDOM_NUMBER_GENERATOR_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
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

// ReplicableRandomDevice is used in randomized tests, which also often want to
// run for an amount of time (instead of a fixed number of iterations). This
// small helper helps with that.
// Usage example:
// for (auto _ : FuzzTest(std::chrono::seconds(1))) {...}
class FuzzTest {
 public:
  class FuzzIterator {
   public:
    explicit FuzzIterator(FuzzTest* parent) : parent_(parent) {}

    void operator++() { parent_->iters_++; }

    bool operator!=(const FuzzIterator& other) const {
      return !parent_->Done();
    }

    int operator*() const { return 0; }

   private:
    FuzzTest* parent_;
  };

  template <typename Duration>
  explicit FuzzTest(Duration duration, int min_iters = 1,
                    int max_iters = std::numeric_limits<int>::max())
      : expire_at_(clock::now() + duration),
        min_iters_(min_iters),
        max_iters_(max_iters) {}

  auto begin() { return FuzzIterator(this); }
  auto end() { return FuzzIterator(this); }

  bool Done() const {
    if (iters_ >= max_iters_) {
      return true;
    } else {
      return iters_ >= min_iters_ && clock::now() >= expire_at_;
    }
  }

 private:
  using clock = std::chrono::steady_clock;
  clock::time_point expire_at_;
  int min_iters_;
  int max_iters_;
  int iters_ = 0;
};

}  // namespace xnnpack

#endif  // __XNNPACK_TEST_REPLICABLE_RANDOM_NUMBER_GENERATOR_H_
