// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/dot/schedule.h"

#include <cassert>
#include <cstddef>
#include <ostream>
#include <utility>
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
             const void* init_c, const void* c,
             dot_kernel_state* state = nullptr) {
    calls.push_back({m, n, k[0], a, b, init_c, c});
  };
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
  EXPECT_THAT(calls,
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

  EXPECT_THAT(calls,
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

  EXPECT_THAT(
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

  EXPECT_THAT(calls,
              ElementsAre(dot_call_at(m, n, block_k, 0, 0, 0 * block_k),
                          dot_call_at(m, n, block_k, 0, 0, 1 * block_k),
                          dot_call_at(m, n, block_k, 0, 0, 2 * block_k),
                          dot_call_at(m, n, block_k, 0, 0, 3 * block_k)));
}

// Convert loops to (dim, blocks) pairs for easy comparison.
std::vector<std::pair<int, size_t>> to_pairs(span<dot_loop> loops) {
  std::vector<std::pair<int, size_t>> result;
  for (const dot_loop& loop : loops) {
    result.push_back({loop.dim, loop.blocks});
  }
  return result;
}

size_t scheduled_k_chunk(size_t l2_cache_size, size_t m, span<const size_t> ks,
                         size_t b_elem_size) {
  dot_loop storage[3];
  auto loops = schedule_dot(l2_cache_size, m, /*n=*/16, ks, /*block_m=*/16,
                            /*block_n=*/16, /*block_k=*/1,
                            /*a_elem_size=*/b_elem_size, b_elem_size,
                            /*pack_a=*/false, /*pack_b=*/true, storage);
  return loops[0].dim == dot_loop::k ? loops[0].blocks : ks[0];
}

TEST(schedule_dot, k_chunk_balanced) {
  const size_t l2_cache_size = 128 * 1024;
  const size_t m = 64;
  const size_t granularity = 64;
  // With fp32 B and nominal block_n = 32, we can fit l2_cache_size / (32 * 4)
  // values of k in a chunk.
  const size_t max_k = l2_cache_size / (32 * sizeof(float));
  EXPECT_EQ(max_k % granularity, 0);
  const size_t fits[] = {max_k};
  EXPECT_EQ(scheduled_k_chunk(l2_cache_size, m, fits, 4), max_k);
  // If k fits, the result is k, even if it is not a multiple of the
  // granularity.
  const size_t small[] = {100};
  EXPECT_EQ(scheduled_k_chunk(l2_cache_size, m, small, 4), 100);
  const size_t almost_fits[] = {max_k - 1};
  EXPECT_EQ(scheduled_k_chunk(l2_cache_size, m, almost_fits, 4), max_k - 1);
  // Chunks of k should be balanced (in units of the granularity), rather than
  // leaving a small chunk at the end.
  const size_t just_over[] = {max_k + 1};
  EXPECT_EQ(scheduled_k_chunk(l2_cache_size, m, just_over, 4),
            max_k / 2 + granularity);
  const size_t three_chunks[] = {3 * max_k - 3};
  EXPECT_EQ(scheduled_k_chunk(l2_cache_size, m, three_chunks, 4), max_k);
  const size_t four_chunks[] = {3944};
  EXPECT_EQ(scheduled_k_chunk(l2_cache_size, m, four_chunks, 4), 1024);
  // The other k dimensions reduce the chunk of k1.
  const size_t k2[] = {max_k, 4};
  EXPECT_EQ(scheduled_k_chunk(l2_cache_size, m, k2, 4), max_k / 4);
  // The chunk is at least the granularity.
  const size_t huge_k2[] = {max_k, 1024};
  EXPECT_EQ(scheduled_k_chunk(l2_cache_size, m, huge_k2, 4), granularity);
  // Smaller elements of B allow larger chunks of k.
  const size_t bf16[] = {8192};
  EXPECT_EQ(scheduled_k_chunk(l2_cache_size, m, bf16, 2), 4096);
  const size_t int8[] = {8192 + 1};
  EXPECT_EQ(scheduled_k_chunk(l2_cache_size, m, int8, 1), 4096 + 64);
}

TEST(schedule_dot, k_chunk_small_m) {
  const size_t l2_cache_size = 128 * 1024;
  const size_t small_m = 16;
  const size_t large_m = 64;
  // When m is small (8 < m && m <= 32), we use smaller chunks of k for B of at
  // least 4 bytes (nominal block_n = 64 instead of 32).
  const size_t max_k = l2_cache_size / (64 * sizeof(float));
  const size_t fits[] = {max_k};
  EXPECT_EQ(scheduled_k_chunk(l2_cache_size, small_m, fits, 4), max_k);
  const size_t fp32[] = {2048};
  EXPECT_EQ(scheduled_k_chunk(l2_cache_size, small_m, fp32, 4), 512);
  EXPECT_EQ(scheduled_k_chunk(l2_cache_size, large_m, fp32, 4), 1024);
  const size_t fp32_balanced[] = {1152};
  EXPECT_EQ(scheduled_k_chunk(l2_cache_size, small_m, fp32_balanced, 4), 384);
  const size_t fp64[] = {1024};
  EXPECT_EQ(scheduled_k_chunk(l2_cache_size, small_m, fp64, 8), 256);
  // Smaller elements of B are not affected.
  const size_t bf16[] = {8192};
  EXPECT_EQ(scheduled_k_chunk(l2_cache_size, small_m, bf16, 2), 4096);
  const size_t int8[] = {16384};
  EXPECT_EQ(scheduled_k_chunk(l2_cache_size, small_m, int8, 1), 8192);

  EXPECT_EQ(scheduled_k_chunk(l2_cache_size, /*m=*/1, fp32, 4), 1024);
  EXPECT_EQ(scheduled_k_chunk(l2_cache_size, /*m=*/8, fp32, 4), 1024);
  EXPECT_EQ(scheduled_k_chunk(l2_cache_size, /*m=*/9, fp32, 4), 512);
  EXPECT_EQ(scheduled_k_chunk(l2_cache_size, /*m=*/32, fp32, 4), 512);
  EXPECT_EQ(scheduled_k_chunk(l2_cache_size, /*m=*/33, fp32, 4), 1024);
}

TEST(schedule_dot, k_loop_consistent) {
  const size_t l2_cache_size = 128 * 1024;
  dot_loop storage[3];

  // For numeric consistency, the chunks of k must be the same for all
  // combinations of pack_a, pack_b, and block_k (which depends on the kernel),
  // regardless of n and block_n.
  for (size_t b_elem_size : {2, 4}) {
    for (size_t m_extent : {1, 16, 64}) {
      for (size_t k1 : {4096, 4095, 2050, 128, 32}) {
        const size_t ks[] = {k1, 3};
        const size_t expected_k_chunk =
            scheduled_k_chunk(l2_cache_size, m_extent, ks, b_elem_size);
        const bool expect_k_loop = k1 > 1000;
        EXPECT_EQ(expected_k_chunk < k1, expect_k_loop);
        for (size_t block_k : {1, 2, 4, 8, 16, 32, 64}) {
          for (size_t n_extent : {32, 1024}) {
            for (bool pack_a : {false, true}) {
              for (bool pack_b : {false, true}) {
                // When B is not packed, it is not blocked in n.
                const size_t block_n_extent = pack_b ? 16 : n_extent;
                auto loops = schedule_dot(
                    l2_cache_size, m_extent, n_extent, ks,
                    /*block_m=*/16, block_n_extent, block_k,
                    /*a_elem_size=*/2, b_elem_size, pack_a, pack_b, storage);
                EXPECT_FALSE(loops.empty());
                if (expect_k_loop) {
                  // k doesn't fit in cache, we should have a k loop, and it
                  // should be the outermost loop.
                  EXPECT_EQ(loops[0].dim, dot_loop::k);
                  EXPECT_EQ(loops[0].blocks * block_k, expected_k_chunk);
                } else {
                  // k fits in cache, there should be no k loop.
                  for (const dot_loop& loop : loops) {
                    EXPECT_NE(loop.dim, dot_loop::k);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(schedule_dot, pack_a_pack_b) {
  const size_t l2_cache_size = 128 * 1024;
  const size_t ks[] = {128};
  dot_loop storage[3];

  // When pack_a is true, m is outside n, so each packed panel of A is reused
  // for all of n.
  auto loops = schedule_dot(l2_cache_size, /*m=*/64, /*n=*/32, ks,
                            /*block_m=*/16, /*block_n=*/16, /*block_k=*/32,
                            /*a_elem_size=*/2, /*b_elem_size=*/2,
                            /*pack_a=*/true, /*pack_b=*/true, storage);
  EXPECT_EQ(to_pairs(loops), (std::vector<std::pair<int, size_t>>{
                                 {dot_loop::m, 1}, {dot_loop::n, 1}}));

  // Even if tiles of A are smaller than tiles of B.
  loops = schedule_dot(l2_cache_size, /*m=*/32, /*n=*/64, ks,
                       /*block_m=*/16, /*block_n=*/16, /*block_k=*/32,
                       /*a_elem_size=*/2, /*b_elem_size=*/2, /*pack_a=*/true,
                       /*pack_b=*/true, storage);
  EXPECT_EQ(to_pairs(loops), (std::vector<std::pair<int, size_t>>{
                                 {dot_loop::m, 1}, {dot_loop::n, 1}}));
}

TEST(schedule_dot, pack_b) {
  const size_t l2_cache_size = 128 * 1024;
  const size_t ks[] = {128};
  dot_loop storage[3];

  // When m * a_elem_size >= n * b_elem_size and pack_a is false, m is outer and
  // n is inner.
  auto m_outer_loops = schedule_dot(
      l2_cache_size, /*m=*/64, /*n=*/32, ks, /*block_m=*/16, /*block_n=*/16,
      /*block_k=*/32, /*a_elem_size=*/2, /*b_elem_size=*/2, /*pack_a=*/false,
      /*pack_b=*/true, storage);
  EXPECT_EQ(to_pairs(m_outer_loops), (std::vector<std::pair<int, size_t>>{
                                         {dot_loop::m, 1}, {dot_loop::n, 1}}));

  // When m * a_elem_size < n * b_elem_size and pack_a is false, n is outer and
  // m is inner.
  auto n_outer_loops = schedule_dot(
      l2_cache_size, /*m=*/32, /*n=*/64, ks, /*block_m=*/16, /*block_n=*/16,
      /*block_k=*/32, /*a_elem_size=*/2, /*b_elem_size=*/2, /*pack_a=*/false,
      /*pack_b=*/true, storage);
  EXPECT_EQ(to_pairs(n_outer_loops), (std::vector<std::pair<int, size_t>>{
                                         {dot_loop::n, 1}, {dot_loop::m, 1}}));
}

TEST(schedule_dot, unpacked_b) {
  const size_t l2_cache_size = 128 * 1024;
  const size_t ks[] = {8192};
  dot_loop storage[3];

  // When B is not packed, it is not blocked in n (block_n == n), so there is no
  // n loop, regardless of whether A is packed. k doesn't fit in cache, so it is
  // split into 2 balanced chunks.
  for (bool pack_a : {false, true}) {
    auto loops =
        schedule_dot(l2_cache_size, /*m=*/64, /*n=*/1024, ks, /*block_m=*/16,
                     /*block_n=*/1024, /*block_k=*/32, /*a_elem_size=*/2,
                     /*b_elem_size=*/2, pack_a, /*pack_b=*/false, storage);
    EXPECT_EQ(to_pairs(loops), (std::vector<std::pair<int, size_t>>{
                                   {dot_loop::k, 128}, {dot_loop::m, 1}}));
  }
}

}  // namespace ynn
