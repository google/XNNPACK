// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/dot/schedule.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <ostream>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/arithmetic.h"
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

std::ostream& operator<<(std::ostream& os, const dot_loop& loop) {
  return os << "dot_loop(" << loop.dim << ", " << loop.step << ")";
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
  const dot_loop loops[] = {{dot_loop::m, block_m}};
  const size_t ks[] = {k};
  const size_t a_k_strides[] = {a_stride_k};
  const size_t b_k_strides[] = {b_stride_k};

  std::vector<dot_call> calls;
  run_dot(loops, m, n, ks, a_stride_m, a_k_strides, a, b_k_strides, b_stride_n,
          b, init_c_stride_m, init_c, c_stride_m, c_stride_n, c,
          make_record_calls(calls));
  ASSERT_THAT(calls,
              ElementsAre(dot_call_at(block_m, n, k, 0 * block_m, 0, 0),
                          dot_call_at(block_m, n, k, 1 * block_m, 0, 0),
                          dot_call_at(block_m, n, k, 2 * block_m, 0, 0)));
}

TEST(run_dot, loop_n) {
  const dot_loop loops[] = {{dot_loop::n, block_n}};
  const size_t ks[] = {k};
  const size_t a_k_strides[] = {a_stride_k};
  const size_t b_k_strides[] = {b_stride_k};

  std::vector<dot_call> calls;
  run_dot(loops, m, n, ks, a_stride_m, a_k_strides, a, b_k_strides, b_stride_n,
          b, init_c_stride_m, init_c, c_stride_m, c_stride_n, c,
          make_record_calls(calls));

  ASSERT_THAT(calls,
              ElementsAre(dot_call_at(m, block_n, k, 0, 0 * block_n, 0),
                          dot_call_at(m, block_n, k, 0, 1 * block_n, 0),
                          dot_call_at(m, block_n, k, 0, 2 * block_n, 0),
                          dot_call_at(m, block_n, k, 0, 3 * block_n, 0),
                          dot_call_at(m, block_n, k, 0, 4 * block_n, 0)));
}

TEST(run_dot, loop_n_tail) {
  const dot_loop loops[] = {{dot_loop::n, 2 * block_n}};
  const size_t ks[] = {k};
  const size_t a_k_strides[] = {a_stride_k};
  const size_t b_k_strides[] = {b_stride_k};

  std::vector<dot_call> calls;
  run_dot(loops, m, n, ks, a_stride_m, a_k_strides, a, b_k_strides, b_stride_n,
          b, init_c_stride_m, init_c, c_stride_m, c_stride_n, c,
          make_record_calls(calls));

  ASSERT_THAT(
      calls,
      ElementsAre(dot_call_at(m, 2 * block_n, k, 0, 0 * block_n, 0),
                  dot_call_at(m, 2 * block_n, k, 0, 2 * block_n, 0),
                  dot_call_at(m, n - 4 * block_n, k, 0, 4 * block_n, 0)));
}

TEST(run_dot, loop_k) {
  const dot_loop loops[] = {{dot_loop::k, block_k}};
  const size_t ks[] = {k};
  const size_t a_k_strides[] = {a_stride_k};
  const size_t b_k_strides[] = {b_stride_k};

  std::vector<dot_call> calls;
  run_dot(loops, m, n, ks, a_stride_m, a_k_strides, a, b_k_strides, b_stride_n,
          b, init_c_stride_m, init_c, c_stride_m, c_stride_n, c,
          make_record_calls(calls));

  ASSERT_THAT(calls,
              ElementsAre(dot_call_at(m, n, block_k, 0, 0, 0 * block_k),
                          dot_call_at(m, n, block_k, 0, 0, 1 * block_k),
                          dot_call_at(m, n, block_k, 0, 0, 2 * block_k),
                          dot_call_at(m, n, block_k, 0, 0, 3 * block_k)));
}

TEST(run_dot, loop_k2_k3) {
  const dot_loop loops[] = {{dot_loop::k3, 2}, {dot_loop::k2, 3}};
  const size_t ks[] = {k, 5, 3};
  const size_t a_k_strides[] = {a_stride_k, 100, 1000};
  const size_t b_k_strides[] = {b_stride_k, 200, 2000};

  struct k_nd_call {
    size_t k1, k2, k3;
    const void* a;
    const void* b;
    const void* init_c;
    bool operator==(const k_nd_call& o) const {
      return k1 == o.k1 && k2 == o.k2 && k3 == o.k3 && a == o.a && b == o.b &&
             init_c == o.init_c;
    }
  };
  std::vector<k_nd_call> calls;
  run_dot(loops, m, n, ks, a_stride_m, a_k_strides, a, b_k_strides, b_stride_n,
          b, init_c_stride_m, init_c, c_stride_m, c_stride_n, c,
          [&](size_t m_arg, size_t n_arg, span<const size_t> k_arg,
              const void* a_arg, size_t, span<const size_t>, const void* b_arg,
              span<const size_t>, size_t, const void* init_c_arg,
              const void* c_arg, dot_kernel_state* = nullptr) {
            EXPECT_EQ(m_arg, m);
            EXPECT_EQ(n_arg, n);
            EXPECT_EQ(c_arg, c);
            calls.push_back(
                {k_arg[0], k_arg[1], k_arg[2], a_arg, b_arg, init_c_arg});
          });

  ASSERT_THAT(calls,
              ElementsAre(k_nd_call{k, 3, 2, a, b, init_c},
                          k_nd_call{k, 2, 2, a + 3 * 100, b + 3 * 200, c},
                          k_nd_call{k, 3, 1, a + 2 * 1000, b + 2 * 2000, c},
                          k_nd_call{k, 2, 1, a + 2 * 1000 + 3 * 100,
                                    b + 2 * 2000 + 3 * 200, c}));
}

TEST(schedule_dot, k_loop_independent_of_cpu_params) {
  const size_t ks[] = {2000, 3, 1};
  dot_loop storage_a[8];
  dot_loop storage_b[8];
  const size_t caches_a[] = {32 * 1024, 256 * 1024};
  const size_t caches_b[] = {64 * 1024, 1024 * 1024};

  auto loops_a = schedule_dot(caches_a, /*m=*/128, /*n=*/16, ks, /*block_m=*/4,
                              /*block_n=*/16, /*a_elem_size=*/4,
                              /*b_elem_size=*/4, storage_a);
  auto loops_b = schedule_dot(caches_b, /*m=*/64, /*n=*/256, ks, /*block_m=*/16,
                              /*block_n=*/64, /*a_elem_size=*/4,
                              /*b_elem_size=*/4, storage_b);

  ASSERT_FALSE(loops_a.empty());
  ASSERT_FALSE(loops_b.empty());
  EXPECT_EQ(loops_a[0].dim, dot_loop::k);
  EXPECT_EQ(loops_b[0].dim, dot_loop::k);
  EXPECT_EQ(loops_a[0].step, loops_b[0].step);
  EXPECT_EQ(loops_a[0].step % 64, 0);
}

TEST(schedule_dot, k_step_bytes_bound) {
  const size_t caches[] = {16 * 1024, 256 * 1024, 4096 * 1024};
  struct Case {
    std::vector<size_t> ks;
    size_t a_elem_size;
    size_t b_elem_size;
  };
  const Case cases[] = {
      {{131, 15, 1}, 2, 2}, {{180, 6, 1}, 2, 2},    {{4, 4096, 1}, 2, 2},
      {{4, 64, 64}, 4, 4},  {{2000, 64, 64}, 4, 2}, {{2000, 17, 13}, 8, 8},
  };
  for (const Case& tc : cases) {
    dot_loop storage[8];
    auto loops = schedule_dot(caches, /*m=*/128, /*n=*/128, tc.ks,
                              /*block_m=*/32, /*block_n=*/32, tc.a_elem_size,
                              tc.b_elem_size, storage);
    ASSERT_LE(loops.size(), std::size(storage));
    size_t eff_k[3] = {tc.ks[0], tc.ks[1], tc.ks[2]};
    for (const dot_loop& loop : loops) {
      if (loop.dim == dot_loop::k1) {
        EXPECT_EQ(loop.step % 64, 0);
        eff_k[0] = std::min(eff_k[0], loop.step);
      } else if (loop.dim == dot_loop::k2) {
        eff_k[1] = std::min(eff_k[1], loop.step);
      } else if (loop.dim == dot_loop::k3) {
        eff_k[2] = std::min(eff_k[2], loop.step);
      }
    }
    const size_t elem_size = std::max(tc.a_elem_size, tc.b_elem_size);
    EXPECT_LE(eff_k[0] * eff_k[1] * eff_k[2] * elem_size, dot_k_step_bytes);
  }
}

TEST(schedule_dot, pack_a_loop_structure_and_panel_bound) {
  const size_t caches[] = {16 * 1024, 512 * 1024, 4096 * 1024};
  const size_t ks[] = {2048, 3, 1};
  dot_loop storage[std::size(caches) + 6];

  // Choose `m < n` so `schedule_dot` without `pack_a` would tile `m` and put
  // `{n, block_n}` outside `{m, block_m}`; with `pack_a = true`, it must tile
  // `n` for cache and end with `{m, block_m}, {pack_a, 0}, {n, block_n}`.
  constexpr size_t blk_m = 32;
  constexpr size_t blk_n = 64;
  auto loops = schedule_dot(caches, /*m=*/64, /*n=*/2048, ks, blk_m, blk_n,
                            /*a_elem_size=*/2, /*b_elem_size=*/2, storage,
                            /*pack_a=*/true);
  const dot_loop expected_loops[] = {
      {dot_loop::k1, 320},   {dot_loop::n, 3 * blk_n}, {dot_loop::m, blk_m},
      {dot_loop::pack_a, 0}, {dot_loop::n, blk_n},
  };
  EXPECT_EQ(loops, span<dot_loop>(expected_loops));

  const size_t panel_bytes = packed_dot_panel_bytes(
      loops, dot_loop::pack_a, /*m=*/64, /*n=*/2048, ks, /*elem_size_bits=*/16,
      /*tile_k=*/4, /*tile_p=*/1);
  EXPECT_GT(panel_bytes, 0);
  EXPECT_LE(panel_bytes, 64 * 1024);

  // When `m <= block_m`, B is only streamed once, so `n` should only have the
  // innermost `block_n` loop after `pack_a`.
  auto single_m_loops =
      schedule_dot(caches, /*m=*/blk_m, /*n=*/2048, ks, blk_m, blk_n,
                   /*a_elem_size=*/2, /*b_elem_size=*/2, storage,
                   /*pack_a=*/true);
  const dot_loop expected_single_m_loops[] = {
      {dot_loop::k1, 320},
      {dot_loop::pack_a, 0},
      {dot_loop::n, blk_n},
  };
  EXPECT_EQ(single_m_loops, span<dot_loop>(expected_single_m_loops));
}

TEST(run_dot, pack_a_reuses_panel_across_n) {
  constexpr size_t m_ext = 5;
  constexpr size_t n_ext = 7;
  constexpr size_t k1_ext = 5;
  constexpr size_t k2_ext = 2;
  constexpr size_t tile_k = 4;
  constexpr size_t blk_m = 3;
  constexpr size_t blk_n = 3;

  std::vector<int> a_buf(k2_ext * m_ext * k1_ext);
  for (size_t i = 0; i < a_buf.size(); ++i) {
    a_buf[i] = static_cast<int>(i + 1);
  }
  const size_t ks[] = {k1_ext, k2_ext, 1};
  const size_t a_row_stride = k1_ext * sizeof(int);
  const size_t a_k_strides[] = {sizeof(int), m_ext * a_row_stride,
                                k2_ext * m_ext * a_row_stride};
  const size_t b_k_strides[] = {b_stride_k, 0, 0};

  const dot_loop loops[] = {
      {dot_loop::m, blk_m},
      {dot_loop::pack_a, 0},
      {dot_loop::n, blk_n},
  };
  const size_t ks_padded[] = {align_up(k1_ext, tile_k), k2_ext, 1};
  const size_t panel_bytes = packed_dot_panel_bytes(
      loops, dot_loop::pack_a, m_ext, n_ext, ks_padded, sizeof(int) * 8, tile_k,
      /*tile_p=*/1);
  std::vector<int> panel(panel_bytes / sizeof(int), -1);
  dot_packer packer_a(sizeof(int), a_row_stride, sizeof(int) * 8, tile_k,
                      /*tile_p=*/1, panel.data());

  int calls_count = 0;
  run_dot(
      loops, m_ext, n_ext, ks, a_row_stride, a_k_strides, a_buf.data(),
      b_k_strides, b_stride_n, b, init_c_stride_m, init_c, c_stride_m,
      c_stride_n, c,
      [&](size_t m_blk, size_t n_blk, span<const size_t> k_blk,
          const void* a_ptr, size_t packed_stride_m,
          span<const size_t> packed_k_strides, const void*, span<const size_t>,
          size_t, const void*, const void*, dot_kernel_state* = nullptr) {
        const size_t k1_padded = align_up(k1_ext, tile_k);
        EXPECT_EQ(k_blk[0], k1_padded);
        EXPECT_EQ(a_ptr, panel.data());
        EXPECT_EQ(packed_stride_m, k1_padded * sizeof(int));
        EXPECT_EQ(packed_k_strides[0], sizeof(int));
        EXPECT_EQ(packed_k_strides[1], m_blk * packed_stride_m);
        const int* p_vals = static_cast<const int*>(a_ptr);
        const size_t m_offset = (calls_count / 3) * blk_m;
        for (size_t i2 = 0; i2 < k2_ext; ++i2) {
          for (size_t r = 0; r < m_blk; ++r) {
            for (size_t c_k = 0; c_k < k1_padded; ++c_k) {
              const int expected = c_k < k1_ext
                                       ? a_buf[i2 * (m_ext * k1_ext) +
                                               (m_offset + r) * k1_ext + c_k]
                                       : 0;
              EXPECT_EQ(p_vals[i2 * (m_blk * k1_padded) + r * k1_padded + c_k],
                        expected);
            }
          }
        }
        ++calls_count;
      },
      /*state=*/nullptr, &packer_a);
  EXPECT_EQ(calls_count, 6);
}

TEST(should_pack_a, packs_profitable_aligned_and_unaligned_k1) {
  const size_t caches_32k_l1[] = {16 * 1024, 512 * 1024, 4096 * 1024};
  const size_t ks_multi_step[] = {2048, 1, 1};
  const size_t a_k_strides_bf16[] = {2, 0, 0};
  const size_t a_k_strides_f32[] = {4, 0, 0};

  // 4 KiB-aliasing row stride where `panel_m >= l1_ways_4k` and `panel_n >= 4 *
  // kernel_block_n` (e.g. 8x32 BF16 kernel on 32 KiB 8-way L1D).
  EXPECT_TRUE(should_pack_a(caches_32k_l1, /*m=*/64, /*n=*/128, ks_multi_step,
                            /*block_m=*/8, /*block_n=*/64,
                            /*kernel_block_n=*/32, /*tile_m=*/1, /*tile_k=*/2,
                            /*a_elem_size=*/2, /*b_elem_size=*/2,
                            /*a_stride_m=*/2048 * 2, a_k_strides_bf16));

  // High N-reuse (`panel_n >= 12 * kernel_block_n`) on strided single-step A.
  const size_t ks_single_step[] = {256, 1, 1};
  EXPECT_TRUE(should_pack_a(caches_32k_l1, /*m=*/64, /*n=*/256, ks_single_step,
                            /*block_m=*/5, /*block_n=*/64,
                            /*kernel_block_n=*/16, /*tile_m=*/1, /*tile_k=*/1,
                            /*a_elem_size=*/4, /*b_elem_size=*/4,
                            /*a_stride_m=*/600 * 4, a_k_strides_f32));

  // High N-reuse on multi-dimensional reduction where `k2` is strided across
  // full `m` rather than packed per `panel_m`.
  const size_t ks_2d[] = {256, 2, 1};
  const size_t a_k_strides_2d[] = {4, 64 * 256 * 4, 0};
  EXPECT_TRUE(should_pack_a(caches_32k_l1, /*m=*/64, /*n=*/256, ks_2d,
                            /*block_m=*/4, /*block_n=*/64,
                            /*kernel_block_n=*/16, /*tile_m=*/1, /*tile_k=*/1,
                            /*a_elem_size=*/4, /*b_elem_size=*/4,
                            /*a_stride_m=*/256 * 4, a_k_strides_2d));

  // Unaligned `k1` folds body + tail into one pass even on 2D tile kernels.
  const size_t ks_unaligned[] = {1023, 1, 1};
  EXPECT_TRUE(should_pack_a(caches_32k_l1, /*m=*/64, /*n=*/64, ks_unaligned,
                            /*block_m=*/16, /*block_n=*/64,
                            /*kernel_block_n=*/32, /*tile_m=*/16, /*tile_k=*/2,
                            /*a_elem_size=*/2, /*b_elem_size=*/2,
                            /*a_stride_m=*/1023 * 2, a_k_strides_bf16));

  // Non-contiguous `k1` (`a_k_strides[0] != a_elem_size`) gathers once.
  const size_t a_k_strides_strided_k1[] = {16, 0, 0};
  EXPECT_TRUE(should_pack_a(caches_32k_l1, /*m=*/64, /*n=*/64, ks_single_step,
                            /*block_m=*/5, /*block_n=*/64,
                            /*kernel_block_n=*/16, /*tile_m=*/1, /*tile_k=*/1,
                            /*a_elem_size=*/4, /*b_elem_size=*/4,
                            /*a_stride_m=*/256 * 16, a_k_strides_strided_k1));
}

TEST(should_pack_a, skips_unprofitable_or_oversized_cases) {
  const size_t caches_32k_l1[] = {16 * 1024, 512 * 1024, 4096 * 1024};
  const size_t ks_single_step[] = {512, 1, 1};
  const size_t a_k_strides_f32[] = {4, 0, 0};
  const size_t a_k_strides_bf16[] = {2, 0, 0};

  // Already contiguous within a single k-step.
  EXPECT_FALSE(should_pack_a(caches_32k_l1, /*m=*/64, /*n=*/512, ks_single_step,
                             /*block_m=*/5, /*block_n=*/64,
                             /*kernel_block_n=*/16, /*tile_m=*/1, /*tile_k=*/1,
                             /*a_elem_size=*/4, /*b_elem_size=*/4,
                             /*a_stride_m=*/512 * 4, a_k_strides_f32));

  // Already contiguous 2D panel (`a_k_strides[1] == panel_m * row_bytes`).
  const size_t ks_2d[] = {256, 2, 1};
  const size_t a_k_strides_contig_2d[] = {4, 4 * 256 * 4, 0};
  EXPECT_FALSE(should_pack_a(caches_32k_l1, /*m=*/64, /*n=*/512, ks_2d,
                             /*block_m=*/4, /*block_n=*/64,
                             /*kernel_block_n=*/16, /*tile_m=*/1, /*tile_k=*/1,
                             /*a_elem_size=*/4, /*b_elem_size=*/4,
                             /*a_stride_m=*/256 * 4, a_k_strides_contig_2d));

  const size_t ks_multi_step[] = {2048, 1, 1};

  // Hardware 2D tile kernels (`tile_m > 1`, e.g. AMX) load strided A directly
  // when `k1` is aligned.
  const size_t caches_48k_l1[] = {24 * 1024, 1024 * 1024, 16384 * 1024};
  EXPECT_FALSE(should_pack_a(caches_48k_l1, /*m=*/1024, /*n=*/512,
                             ks_multi_step, /*block_m=*/32, /*block_n=*/64,
                             /*kernel_block_n=*/32, /*tile_m=*/16, /*tile_k=*/2,
                             /*a_elem_size=*/2, /*b_elem_size=*/2,
                             /*a_stride_m=*/2048 * 2, a_k_strides_bf16));

  // Low N-reuse (`n = 64`, 4x reuse) with non-4KiB-aliasing row stride.
  const size_t ks_non_aliasing[] = {1536, 1, 1};
  EXPECT_FALSE(should_pack_a(caches_32k_l1, /*m=*/195, /*n=*/64,
                             ks_non_aliasing, /*block_m=*/6, /*block_n=*/64,
                             /*kernel_block_n=*/16, /*tile_m=*/1, /*tile_k=*/1,
                             /*a_elem_size=*/4, /*b_elem_size=*/4,
                             /*a_stride_m=*/1536 * 4, a_k_strides_f32));

  // Low N-reuse with 4 KiB-multiple stride when `panel_m < l1_ways_4k` (fits in
  // L1D ways without set-conflict self-eviction, even for large `m`).
  EXPECT_FALSE(should_pack_a(caches_32k_l1, /*m=*/2048, /*n=*/64, ks_multi_step,
                             /*block_m=*/6, /*block_n=*/64,
                             /*kernel_block_n=*/16, /*tile_m=*/1, /*tile_k=*/1,
                             /*a_elem_size=*/4, /*b_elem_size=*/4,
                             /*a_stride_m=*/2048 * 4, a_k_strides_f32));

  // 4 KiB-aliasing row stride with only 2x N-reuse (`panel_n < 4 *
  // kernel_block_n`).
  EXPECT_FALSE(should_pack_a(caches_32k_l1, /*m=*/64, /*n=*/64, ks_multi_step,
                             /*block_m=*/8, /*block_n=*/64,
                             /*kernel_block_n=*/32, /*tile_m=*/1, /*tile_k=*/2,
                             /*a_elem_size=*/2, /*b_elem_size=*/2,
                             /*a_stride_m=*/2048 * 2, a_k_strides_bf16));

  // 64 KiB L1D (e.g. ARM, `l1_budget = 32 KiB`, `l1_ways_4k = 16`) with low
  // N-reuse (`panel_m = 5 < 16`).
  const size_t caches_64k_l1[] = {32 * 1024, 512 * 1024, 16384 * 1024};
  EXPECT_FALSE(should_pack_a(caches_64k_l1, /*m=*/2048, /*n=*/64, ks_multi_step,
                             /*block_m=*/5, /*block_n=*/64,
                             /*kernel_block_n=*/16, /*tile_m=*/1, /*tile_k=*/1,
                             /*a_elem_size=*/4, /*b_elem_size=*/4,
                             /*a_stride_m=*/2048 * 4, a_k_strides_f32));

  // Insufficient N reuse (e.g. AVX-512 5x64 kernel on N = 64 -> 1 read).
  EXPECT_FALSE(should_pack_a(caches_32k_l1, /*m=*/1024, /*n=*/64, ks_multi_step,
                             /*block_m=*/5, /*block_n=*/64,
                             /*kernel_block_n=*/64, /*tile_m=*/1, /*tile_k=*/1,
                             /*a_elem_size=*/4, /*b_elem_size=*/4,
                             /*a_stride_m=*/2048 * 4, a_k_strides_f32));

  // Small panel_m (< 4 rows).
  EXPECT_FALSE(should_pack_a(caches_32k_l1, /*m=*/3, /*n=*/512, ks_multi_step,
                             /*block_m=*/5, /*block_n=*/64,
                             /*kernel_block_n=*/16, /*tile_m=*/1, /*tile_k=*/1,
                             /*a_elem_size=*/4, /*b_elem_size=*/4,
                             /*a_stride_m=*/2048 * 4, a_k_strides_f32));

  // Broadcast A along m (`a_stride_m == 0`).
  EXPECT_FALSE(should_pack_a(caches_32k_l1, /*m=*/64, /*n=*/512, ks_multi_step,
                             /*block_m=*/5, /*block_n=*/64,
                             /*kernel_block_n=*/16, /*tile_m=*/1, /*tile_k=*/1,
                             /*a_elem_size=*/4, /*b_elem_size=*/4,
                             /*a_stride_m=*/0, a_k_strides_f32));

  // Exceeds `dot_max_panel_bytes` / `2 * l1_budget` stack limit (`block_m = 64`
  // -> 128 KiB).
  EXPECT_FALSE(should_pack_a(caches_32k_l1, /*m=*/64, /*n=*/512, ks_multi_step,
                             /*block_m=*/64, /*block_n=*/64,
                             /*kernel_block_n=*/16, /*tile_m=*/1, /*tile_k=*/1,
                             /*a_elem_size=*/4, /*b_elem_size=*/4,
                             /*a_stride_m=*/2048 * 4, a_k_strides_f32));

  // Forcing inner-`n` loop would spill active B panel past L2 when unpacked
  // schedule would keep B outer (`n * b_elem_size > m * a_elem_size`).
  const size_t small_l2_caches[] = {16 * 1024, 256 * 1024};
  EXPECT_FALSE(should_pack_a(small_l2_caches, /*m=*/64, /*n=*/512,
                             ks_multi_step, /*block_m=*/8, /*block_n=*/256,
                             /*kernel_block_n=*/16, /*tile_m=*/1, /*tile_k=*/1,
                             /*a_elem_size=*/4, /*b_elem_size=*/4,
                             /*a_stride_m=*/2048 * 4, a_k_strides_f32));
}

}  // namespace ynn
