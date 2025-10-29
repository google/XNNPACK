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
constexpr size_t b_stride_block_n = 1;
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
  return b + k * b_stride_k + n * b_stride_block_n / block_n;
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
  return [&](size_t m, size_t n, size_t k, const void* a, const void* b,
             size_t init_c_stride_m, const void* init_c,
             const void* c) { calls.push_back({m, n, k, a, b, init_c, c}); };
}

TEST(run_dot, loop_m) {
  const dot_loop loops[] = {{dot_loop::m, 1}};

  std::vector<dot_call> calls;
  run_dot(loops, m, n, k, block_m, block_n, block_k, a_stride_m, a_stride_k, a,
          b_stride_k, b_stride_block_n, b, init_c_stride_m, init_c, c_stride_m,
          c_stride_n, c, make_record_calls(calls));
  ASSERT_THAT(calls,
              ElementsAre(dot_call_at(block_m, n, k, 0 * block_m, 0, 0),
                          dot_call_at(block_m, n, k, 1 * block_m, 0, 0),
                          dot_call_at(block_m, n, k, 2 * block_m, 0, 0)));
}

TEST(run_dot, loop_n) {
  const dot_loop loops[] = {{dot_loop::n, 1}};

  std::vector<dot_call> calls;
  run_dot(loops, m, n, k, block_m, block_n, block_k, a_stride_m, a_stride_k, a,
          b_stride_k, b_stride_block_n, b, init_c_stride_m, init_c, c_stride_m,
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

  std::vector<dot_call> calls;
  run_dot(loops, m, n, k, block_m, block_n, block_k, a_stride_m, a_stride_k, a,
          b_stride_k, b_stride_block_n, b, init_c_stride_m, init_c, c_stride_m,
          c_stride_n, c, make_record_calls(calls));

  ASSERT_THAT(calls,
              ElementsAre(dot_call_at(m, 2 * block_n, k, 0, 0 * block_n, 0),
                          dot_call_at(m, 2 * block_n, k, 0, 2 * block_n, 0),
                          dot_call_at(m, block_n, k, 0, 4 * block_n, 0)));
}

TEST(run_dot, loop_k) {
  const dot_loop loops[] = {{dot_loop::k, 1}};

  std::vector<dot_call> calls;
  run_dot(loops, m, n, k, block_m, block_n, block_k, a_stride_m, a_stride_k, a,
          b_stride_k, b_stride_block_n, b, init_c_stride_m, init_c, c_stride_m,
          c_stride_n, c, make_record_calls(calls));

  ASSERT_THAT(calls,
              ElementsAre(dot_call_at(m, n, block_k, 0, 0, 0 * block_k),
                          dot_call_at(m, n, block_k, 0, 0, 1 * block_k),
                          dot_call_at(m, n, block_k, 0, 0, 2 * block_k),
                          dot_call_at(m, n, block_k, 0, 0, 3 * block_k)));
}

}  // namespace ynn
