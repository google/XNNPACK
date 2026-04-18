// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/dot/schedule.h"

#include <cassert>
#include <cstddef>
#include <ostream>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/span.h"

using testing::ElementsAre;

namespace ynn {

struct dot_call {
  size_t m;
  size_t n;
  size_t k;
  const void* a;
  const void* b;
  const void* init_c;
  const void* c;

  bool operator==(const dot_call& other) const {
    return m == other.m && n == other.n && k == other.k && a == other.a &&
           b == other.b && init_c == other.init_c && c == other.c;
  }
};

std::ostream& operator<<(std::ostream& os, const dot_call& call) {
  return os << "dot_call(" << call.m << ", " << call.n << ", " << call.k << ", "
            << call.a << ", " << call.b << ", " << call.init_c << ", " << call.c
            << ")";
}

constexpr size_t m = 12;
constexpr size_t n = 15;
constexpr size_t k = 8;
constexpr size_t block_m = 4;
constexpr size_t block_n = 3;
constexpr size_t block_k = 2;
constexpr size_t a_stride_m = 10;
constexpr size_t a_stride_k = 1;
constexpr size_t b_stride_k = 12;
constexpr size_t b_stride_n = 1;
constexpr size_t init_c_stride_m = 7;
constexpr size_t c_stride_m = 13;
constexpr size_t c_stride_n = 1;

const char* a = reinterpret_cast<const char*>(0xa000);
const char* b = reinterpret_cast<const char*>(0xb000);
const char* init_c = reinterpret_cast<const char*>(0x1c000);
char* c = reinterpret_cast<char*>(0xc000);

const void* a_at(size_t m, size_t k) {
  return a + m * a_stride_m + k * a_stride_k;
};
const void* b_at(size_t k, size_t n) {
  return b + k * b_stride_k + n * b_stride_n;
};
const void* init_c_at(size_t m, size_t n) {
  return init_c + m * init_c_stride_m + n * c_stride_n;
};
const void* c_at(size_t m, size_t n) {
  return c + m * c_stride_m + n * c_stride_n;
};

dot_call dot_call_at(size_t m, size_t n, size_t k, size_t i, size_t j,
                     size_t k_at) {
  return dot_call{
      m,
      n,
      k,
      a_at(i, k_at),
      b_at(k_at, j),
      k_at == 0 ? init_c_at(i, j) : c_at(i, j),
      c_at(i, j),
  };
};

auto make_record_calls(std::vector<dot_call>& calls) {
  return [&](size_t m, size_t n, span<const size_t> k, const void* a,
             size_t a_stride_m, span<const size_t> a_k_strides, const void* b,
             span<const size_t> b_k_strides, size_t init_c_stride_m,
             const void* init_c,
             const void* c) { calls.push_back({m, n, k[0], a, b, init_c, c}); };
}

TEST(run_dot, loop_m) {
  const dot_loop loops[] = {{dot_loop::m, 1}};
  const size_t ks[] = {k};
  const size_t a_k_strides[] = {a_stride_k};
  const size_t b_k_strides[] = {b_stride_k};

  std::vector<dot_call> calls;
  run_dot(loops, m, n, ks, block_m, block_n, block_k, a_stride_m, a_k_strides,
          a, b_k_strides, b_stride_n, b, init_c_stride_m, init_c, c_stride_m,
          c_stride_n, c, make_record_calls(calls));
  ASSERT_THAT(calls,
              ElementsAre(dot_call_at(block_m, n, k, 0 * block_m, 0, 0),
                          dot_call_at(block_m, n, k, 1 * block_m, 0, 0),
                          dot_call_at(block_m, n, k, 2 * block_m, 0, 0)));
}

TEST(run_dot, loop_n) {
  const dot_loop loops[] = {{dot_loop::n, 1}};
  const size_t ks[] = {k};
  const size_t a_k_strides[] = {a_stride_k};
  const size_t b_k_strides[] = {b_stride_k};

  std::vector<dot_call> calls;
  run_dot(loops, m, n, ks, block_m, block_n, block_k, a_stride_m, a_k_strides,
          a, b_k_strides, b_stride_n, b, init_c_stride_m, init_c, c_stride_m,
          c_stride_n, c, make_record_calls(calls));

  ASSERT_THAT(calls,
              ElementsAre(dot_call_at(m, block_n, k, 0, 0 * block_n, 0),
                          dot_call_at(m, block_n, k, 0, 1 * block_n, 0),
                          dot_call_at(m, block_n, k, 0, 2 * block_n, 0),
                          dot_call_at(m, block_n, k, 0, 3 * block_n, 0),
                          dot_call_at(m, block_n, k, 0, 4 * block_n, 0)));
}

TEST(run_dot, loop_n_tail) {
  const dot_loop loops[] = {{dot_loop::n, 2}};
  const size_t ks[] = {k};
  const size_t a_k_strides[] = {a_stride_k};
  const size_t b_k_strides[] = {b_stride_k};

  std::vector<dot_call> calls;
  run_dot(loops, m, n, ks, block_m, block_n, block_k, a_stride_m, a_k_strides,
          a, b_k_strides, b_stride_n, b, init_c_stride_m, init_c, c_stride_m,
          c_stride_n, c, make_record_calls(calls));

  ASSERT_THAT(
      calls,
      ElementsAre(dot_call_at(m, 2 * block_n, k, 0, 0 * block_n, 0),
                  dot_call_at(m, 2 * block_n, k, 0, 2 * block_n, 0),
                  dot_call_at(m, n - 4 * block_n, k, 0, 4 * block_n, 0)));
}

TEST(run_dot, loop_k) {
  const dot_loop loops[] = {{dot_loop::k, 1}};
  const size_t ks[] = {k};
  const size_t a_k_strides[] = {a_stride_k};
  const size_t b_k_strides[] = {b_stride_k};

  std::vector<dot_call> calls;
  run_dot(loops, m, n, ks, block_m, block_n, block_k, a_stride_m, a_k_strides,
          a, b_k_strides, b_stride_n, b, init_c_stride_m, init_c, c_stride_m,
          c_stride_n, c, make_record_calls(calls));

  ASSERT_THAT(calls,
              ElementsAre(dot_call_at(m, n, block_k, 0, 0, 0 * block_k),
                          dot_call_at(m, n, block_k, 0, 0, 1 * block_k),
                          dot_call_at(m, n, block_k, 0, 0, 2 * block_k),
                          dot_call_at(m, n, block_k, 0, 0, 3 * block_k)));
}

// -- Targeted tests for schedule_dot itself --

bool operator==(const dot_loop& a, const dot_loop& b) {
  return a.dim == b.dim && a.blocks == b.blocks;
}

std::ostream& operator<<(std::ostream& os, const dot_loop& l) {
  const char* d = l.dim == dot_loop::m   ? "m"
                  : l.dim == dot_loop::n ? "n"
                  : l.dim == dot_loop::k ? "k"
                                         : "?";
  return os << d << "x" << l.blocks;
}

// A cache budget much larger than the working set yields no blocking — the
// default {m, 1} safety loop is emitted so run_dot always has at least one
// loop to walk.
TEST(schedule_dot, no_blocking_when_everything_fits) {
  dot_loop storage[3];
  const size_t cache_sizes[] = {8 * 1024 * 1024};  // 8 MiB
  const size_t ks[] = {64};
  auto loops = schedule_dot(cache_sizes, /*m=*/16, /*n=*/64, ks,
                            /*block_m=*/16, /*block_n=*/64, /*block_k=*/1,
                            /*a_elem_size=*/4, /*b_elem_size=*/4, storage);
  EXPECT_THAT(loops, ElementsAre(dot_loop{dot_loop::m, 1}));
}

// Large shape vs a 128 KiB cache: kc_max = 15/16 * 128 KiB /
// (n * b_elem * block_k) = 120 KiB / (2048 * 4 * 1) = 15 block_k units.
// The 15/16 factor is the safety headroom applied in schedule.cc.
TEST(schedule_dot, k_loop_sized_from_current_n) {
  dot_loop storage[3];
  const size_t cache_sizes[] = {128 * 1024};
  const size_t ks[] = {2048};
  auto loops = schedule_dot(cache_sizes, /*m=*/2048, /*n=*/2048, ks,
                            /*block_m=*/16, /*block_n=*/64, /*block_k=*/1,
                            /*a_elem_size=*/4, /*b_elem_size=*/4, storage);
  EXPECT_THAT(loops, ElementsAre(dot_loop{dot_loop::k, 15},
                                 dot_loop{dot_loop::m, 1},
                                 dot_loop{dot_loop::n, 1}));
}

// Even-split: when k slightly overflows the natural kc, we split into two
// near-equal iterations rather than one cache-max iter plus a small tail.
// With a 16 MiB cache, n = 4096, and the 15/16 safety headroom, kc_max =
// 15 MiB / (4096 * 4) = 960. For k = 1200, niter = 2, blocks = 600.
TEST(schedule_dot, k_loop_splits_evenly_when_k_slightly_over_kc_max) {
  dot_loop storage[3];
  const size_t cache_sizes[] = {16ULL * 1024 * 1024};
  const size_t ks[] = {1200};
  auto loops = schedule_dot(cache_sizes, /*m=*/4096, /*n=*/4096, ks,
                            /*block_m=*/16, /*block_n=*/64, /*block_k=*/1,
                            /*a_elem_size=*/4, /*b_elem_size=*/4, storage);
  EXPECT_THAT(loops, ElementsAre(dot_loop{dot_loop::k, 600},
                                 dot_loop{dot_loop::m, 1},
                                 dot_loop{dot_loop::n, 1}));
}

// Even-split at ~1.5x: k = 1536 is ~1.6 * kc_max (960). Old policy would
// have run one cache-max iter plus a small tail; the new even-split gives
// two near-equal iters of 768, each comfortably inside cache.
TEST(schedule_dot, k_loop_splits_evenly_at_1p5x_boundary) {
  dot_loop storage[3];
  const size_t cache_sizes[] = {16ULL * 1024 * 1024};
  const size_t ks[] = {1024 + 1024 / 2};  // k1 = 1536
  auto loops = schedule_dot(cache_sizes, /*m=*/4096, /*n=*/4096, ks,
                            /*block_m=*/16, /*block_n=*/64, /*block_k=*/1,
                            /*a_elem_size=*/4, /*b_elem_size=*/4, storage);
  EXPECT_THAT(loops, ElementsAre(dot_loop{dot_loop::k, 768},
                                 dot_loop{dot_loop::m, 1},
                                 dot_loop{dot_loop::n, 1}));
}

// Larger overflow: k = 1600 gives niter = 2, blocks = ceil(1600/2) = 800.
TEST(schedule_dot, k_loop_splits_evenly_into_two_when_below_2x_kc_max) {
  dot_loop storage[3];
  const size_t cache_sizes[] = {16ULL * 1024 * 1024};
  const size_t ks[] = {1600};
  auto loops = schedule_dot(cache_sizes, /*m=*/4096, /*n=*/4096, ks,
                            /*block_m=*/16, /*block_n=*/64, /*block_k=*/1,
                            /*a_elem_size=*/4, /*b_elem_size=*/4, storage);
  EXPECT_THAT(loops, ElementsAre(dot_loop{dot_loop::k, 800},
                                 dot_loop{dot_loop::m, 1},
                                 dot_loop{dot_loop::n, 1}));
}

// Many iterations: k = 2880 = 3 * kc_max (960 after the 15/16 headroom).
// The resulting blocks equals kc_max exactly when k is a clean multiple.
TEST(schedule_dot, k_loop_uses_kc_max_when_k_is_multiple_of_kc_max) {
  dot_loop storage[3];
  const size_t cache_sizes[] = {16ULL * 1024 * 1024};
  const size_t ks[] = {2880};
  auto loops = schedule_dot(cache_sizes, /*m=*/4096, /*n=*/4096, ks,
                            /*block_m=*/16, /*block_n=*/64, /*block_k=*/1,
                            /*a_elem_size=*/4, /*b_elem_size=*/4, storage);
  EXPECT_THAT(loops, ElementsAre(dot_loop{dot_loop::k, 960},
                                 dot_loop{dot_loop::m, 1},
                                 dot_loop{dot_loop::n, 1}));
}

// Boundary: k = kc_max exactly -> fits in one iteration, no k-loop emitted.
TEST(schedule_dot, k_loop_skipped_when_k_equals_kc_max) {
  dot_loop storage[3];
  const size_t cache_sizes[] = {16ULL * 1024 * 1024};
  const size_t ks[] = {960};  // kc_max = 960 after the 15/16 headroom
  auto loops = schedule_dot(cache_sizes, /*m=*/4096, /*n=*/4096, ks,
                            /*block_m=*/16, /*block_n=*/64, /*block_k=*/1,
                            /*a_elem_size=*/4, /*b_elem_size=*/4, storage);
  EXPECT_THAT(loops, ElementsAre(dot_loop{dot_loop::m, 1},
                                 dot_loop{dot_loop::n, 1}));
}

}  // namespace ynn
