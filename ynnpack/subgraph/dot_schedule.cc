// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/dot_schedule.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/type.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "slinky/builder/simplify.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/evaluate.h"
#include "slinky/runtime/expr.h"

using slinky::index_t;

namespace ynn {

namespace {

// Radices, maximum multipliers, and step sizes for packing split factors into a
// single 32-bit integer in choose_split_factors:
//   splits = (mult_k * radix_m + mult_m) * radix_n + mult_n
//
// Bit allocation (31 bits total, fitting in positive 32-bit signed int):
// - mult_n: 11 bits (bits 0-10, [0, 2047]), multiplied by block_n (step size =
//   block_n). Max representable split_n: 2047 * block_n (e.g. 131,008 for
//   block_n = 64).
// - mult_m: 11 bits (bits 11-21, [0, 2047]), multiplied by step_m (step size =
//   16). Max representable split_m: 2047 * 16 = 32,752.
// - mult_k: 9 bits (bits 22-30, [0, 511]), multiplied by step_k (step size =
//   1024). Max representable split_k: 511 * 1024 = 523,264.
//
// Maximum packed value: (511 * 2048 + 2047) * 2048 + 2047 = 2,147,483,647
// (INT32_MAX).
constexpr index_t radix_n = 2048;  // 2^11
constexpr index_t radix_m = 2048;  // 2^11
constexpr index_t radix_k = 512;   // 2^9

constexpr index_t max_multiplier_n = radix_n - 1;
constexpr index_t max_multiplier_m = radix_m - 1;
constexpr index_t max_multiplier_k = radix_k - 1;

constexpr index_t step_m = 16;
constexpr index_t step_k = 1024;

// The estimated duration of a task below which it is not worth running in
// parallel with other tasks: waking up threads and synchronizing with them
// costs ~5-20us per parallel loop on x86 and ARM servers.
constexpr float min_task_seconds = 16e-6f;

// Conservative estimates of the bandwidths (in bytes per second) at which one
// core streams the operands of a task from L3 or DRAM, and writes its outputs
// (to cache, which is faster). The kernel cost models are fit to L1-resident
// benchmarks, and don't account for these, which dominate small tasks: adding
// them makes the estimated duration of tasks track their measured duration
// within ~2x on x86 and ARM servers, including AMX kernels, whose cost models
// alone underestimate it by ~10x.
constexpr float task_bytes_per_second = 8e9f;
constexpr float task_output_bytes_per_second = 16e9f;

}  // namespace

const std::array<size_t, 3>& get_dot_cache_budgets() {
  static const std::array<size_t, 3> budgets = []() {
    const std::array<size_t, 3> sizes = get_cache_sizes();
    const size_t l3_size = sizes[2] > sizes[1] ? sizes[2] : 0;
    return std::array<size_t, 3>{sizes[0] / 2, sizes[1] / 4, l3_size / 4};
  }();
  return budgets;
}

size_t get_dot_l1_l2_budget() {
  const std::array<size_t, 3> cache_sizes = get_cache_sizes();
  return std::min(cache_sizes[0] * 4, cache_sizes[1] / 4);
}

namespace {

// Trade-offs for splitting k:
// - Benefit: For very large values of k, splitting the reduction dimension
//   keeps the active working set of inputs A and B within the L1 cache,
//   preventing input cache lines from being evicted before they can be reused
//   across the m and n tile loops.
// - Cost: Splitting k across multiple serial iterations forces the
//   intermediate accumulator buffer C (m_tile x n_tile) to be written to
//   memory and reloaded for each k chunk, adding memory bandwidth and loop
//   overhead.
// - Data-type dependence: For narrow data types (e.g. INT8 with 1-byte input
//   and 4-byte INT32 accumulator, or BF16 with 2-byte input and 4-byte FP32
//   accumulator), the accumulator is 2-4x larger than the inputs, making
//   accumulator spilling much more expensive. Moreover, at k = 8192
//   (threshold for FP32), narrow inputs take only 8-16 KB (already fitting in
//   L1).
//
// Sizing rationale:
// - When `m <= step_m`, there is only a single `m` tile, so B is not reused
//   across `m` tiles and `schedule_dot` already tiles `k` within each task.
//   Avoid splitting `k` so we don't force serial threadpool dispatches, spill
//   the accumulator, or prevent loop fusion of consumers into `dot`.
// - Otherwise, we set `k_threshold` so that k-splitting only triggers when a
//   single input slice along k reaches `proxy_cache_for_k`.
// - We use half of `proxy_cache_for_k` to set the split size so that both A
//   and B fit in L1.
// - A proxy L1 cache size is used to maintain a consistent splitting behavior
//   across different machines.
index_t choose_split_k(index_t m, index_t k, size_t a_elem_size) {
  assert(a_elem_size > 0);
  constexpr index_t proxy_cache_for_k = 32 * 1024;
  const index_t k_threshold = proxy_cache_for_k / a_elem_size;
  const index_t k_split_size = k_threshold / 2;
  index_t split_k = (m > step_m && k >= k_threshold) ? k_split_size : k;
  split_k = align_up<index_t>(split_k, step_k);
  return std::min<index_t>(split_k, max_multiplier_k * step_k);
}

std::tuple<index_t, index_t, index_t> choose_split_factors(
    index_t m, index_t n, index_t k, index_t block_n, size_t a_elem_size,
    size_t b_elem_bits, size_t c_elem_size, float seconds_per_mac,
    bool a_is_produced) {
  assert(block_n > 0);

  index_t split_k = choose_split_k(m, k, a_elem_size);
  const index_t task_k = std::min(k, split_k);

  const double a_bytes_total = static_cast<double>(m) * task_k * a_elem_size;
  const double b_bytes_total =
      (static_cast<double>(n) * task_k * b_elem_bits) / 8.0;
  const float operand_seconds = static_cast<float>(
      (a_bytes_total + b_bytes_total) / task_bytes_per_second);
  const float dot_seconds =
      static_cast<float>(m) * n * task_k * seconds_per_mac + operand_seconds;

  // If A is produced in this pipeline (neither an external input nor a
  // constant), each task computes its rows of A and reads them back alongside
  // B and C. Cap this working set at twice the L1/L2 budget when A's O(m*k)
  // traffic is a meaningful share of total dot time, and relax the cap
  // proportionally once GEMM compute and B traffic dominate A.
  uint64_t max_produced_task_bytes = 0;
  if (a_is_produced) {
    max_produced_task_bytes = 2 * get_dot_l1_l2_budget();
    const double a_seconds = a_bytes_total / task_bytes_per_second;
    const double l2_reload_seconds =
        a_seconds * (b_bytes_total / get_cache_sizes()[1]);
    const double total_dot_seconds = dot_seconds + l2_reload_seconds;
    constexpr double min_dot_to_a_ratio = 12.0;
    if (a_seconds > 0.0 && total_dot_seconds > min_dot_to_a_ratio * a_seconds) {
      const double scale = total_dot_seconds / (min_dot_to_a_ratio * a_seconds);
      constexpr double max_task_bytes = 1024.0 * 1024.0 * 1024.0;
      max_produced_task_bytes = static_cast<uint64_t>(
          std::min<double>(max_task_bytes, max_produced_task_bytes * scale));
    }
  }

  // If k gets big, we're going to tile k anyways. It could be faster to
  // parallelize more finely, but it will waste CPU cycles due to more memory
  // traffic out of the cache.
  const index_t effective_k = std::min<index_t>(k, step_k);

  // Considerations for task size:
  // - We want tasks to be square-ish, to maximize the number of times we can
  // use data we load from either side.
  // - Tasks shouldn't be too small, to avoid parallelism overhead.
  // - Tasks shouldn't be too large, so we get enough parallelism.
  const index_t max_tiles_1d = 32;
  const index_t min_tile = std::max<index_t>(std::min(m, n) / max_tiles_1d, 64);
  const index_t min_area_2d =
      std::min<index_t>(m, min_tile) * std::min<index_t>(n, min_tile);
  const index_t min_area_1d = 256;
  const index_t min_area = std::max<index_t>(min_area_2d, min_area_1d);
  const index_t max_area = 256 * 256;
  // The maximum cost of a tile, according to the cost function (m + n) * k.
  const index_t max_cost = 1024 * 64;

  // A parameter indicating the target split_m/split_n ratio.
  // TODO(b/438841352): Figure out why we want tall skinny tiles, at least on
  // AMD Rome.
  const index_t aspect_ratio = 4;

  // Make the tile bigger, doubling the dimension that keeps it closest to
  // `aspect_ratio`. Returns false if the tile cannot grow further.
  auto grow = [&](bool limit_produced_a, index_t& split_m, index_t& split_n) {
    bool can_grow_m = split_m < m;
    if (can_grow_m && limit_produced_a) {
      const index_t next_m = split_m * 2;
      const uint64_t a_bytes =
          static_cast<uint64_t>(next_m) * task_k * a_elem_size;
      const uint64_t b_bytes =
          (static_cast<uint64_t>(split_n) * task_k * b_elem_bits) / 8;
      const uint64_t c_bytes =
          static_cast<uint64_t>(next_m) * split_n * c_elem_size;
      can_grow_m = a_bytes + b_bytes + c_bytes <= max_produced_task_bytes;
    }
    const bool can_grow_n = split_n < n;
    if ((aspect_ratio * split_n <= split_m || !can_grow_m) && can_grow_n) {
      split_n *= 2;
    } else if ((split_m <= aspect_ratio * split_n || !can_grow_n) &&
               can_grow_m) {
      split_m *= 2;
    } else {
      return false;
    }
    return true;
  };

  index_t split_n = std::min<index_t>(n, block_n);
  index_t split_m = std::min<index_t>(m, step_m);
  while (true) {
    if (split_n * split_m >= min_area) {
      // We've reached the minimum tile size, should we stop?
      if ((split_m + split_n) * effective_k >= max_cost ||
          split_m * split_n >= max_area) {
        // We've reached the maximum task size, we should stop.
        break;
      }
    }
    // We want to make the tile bigger, figure out which dimension to grow.
    if (!grow(a_is_produced, split_m, split_n)) break;
  }

  // Make the tasks bigger if they are too small to be worth running in
  // parallel.
  while (true) {
    index_t next_m = split_m;
    index_t next_n = split_n;
    if (!grow(/*limit_produced_a=*/false, next_m, next_n)) break;
    const index_t tasks =
        ceil_div(m, std::min(m, split_m)) * ceil_div(n, std::min(n, split_n));
    const float area =
        static_cast<float>(std::min(m, split_m)) * std::min(n, split_n);
    // The estimated duration of a task: its arithmetic (according to the cost
    // model of the kernel), plus streaming its share of the operands.
    const float task_duration_s =
        area * task_k * seconds_per_mac + operand_seconds / tasks;
    const bool output_bound =
        area * c_elem_size / task_output_bytes_per_second > task_duration_s;

    // Stop growing if tasks are large enough or further merging hurts
    // performance:
    // - Duration: The task duration already reaches `min_task_seconds`.
    // - Full-height tasks (`split_m >= m`): Merging along n reuses no more of
    //   B and reduces streaming threads; stop if the entire dot takes >= 4
    //   minimum tasks.
    // - Output-bound tasks: If writing outputs takes longer than compute,
    //   merging rows (`next_m > split_m`) reduces threads writing outputs.
    const bool large_enough =
        task_duration_s >= min_task_seconds ||
        (split_m >= m && dot_seconds >= 4 * min_task_seconds) ||
        (output_bound && next_m > split_m);
    if (large_enough) break;
    split_m = next_m;
    split_n = next_n;
  }

  // Balance the tasks: use the smallest splits (multiples of the steps) that
  // give the same number of tasks, so the last task in each dimension is not
  // much smaller than the others (e.g. split 133 rows as 80 + 53 rather than
  // 128 + 5). Note `block_n` is not necessarily a power of 2.
  auto balance = [](index_t extent, index_t split, index_t step) {
    if (extent <= split) return split;
    const index_t splits = ceil_div(extent, split);
    return std::min(split, ceil_div(ceil_div(extent, splits), step) * step);
  };
  split_m = balance(m, split_m, step_m);
  split_n = balance(n, split_n, block_n);

  // Ensure the split factors do not exceed the maximum representable numbers
  // in the 32-bit packing scheme (11 bits for n, 11 bits for m, 9 bits for k).
  split_n = std::min<index_t>(split_n, max_multiplier_n * block_n);
  split_m = std::min<index_t>(split_m, max_multiplier_m * step_m);
  split_k = std::min<index_t>(split_k, max_multiplier_k * step_k);
  return {split_n, split_m, split_k};
}

std::tuple<slinky::expr, slinky::expr, slinky::expr> choose_split_factors(
    ynn_runtime& runtime, slinky::expr m, slinky::expr n, slinky::expr k,
    slinky::expr block_n, size_t a_elem_size, size_t b_elem_bits,
    size_t c_elem_size, float seconds_per_mac, bool a_is_produced) {
  auto impl = [=](const slinky::call* op,
                  slinky::eval_context& ctx) -> index_t {
    index_t m = evaluate(op->args[0], ctx);
    index_t n = evaluate(op->args[1], ctx);
    index_t k = evaluate(op->args[2], ctx);
    index_t block_n = evaluate(op->args[3], ctx);
    auto [split_n, split_m, split_k] =
        choose_split_factors(m, n, k, block_n, a_elem_size, b_elem_bits,
                             c_elem_size, seconds_per_mac, a_is_produced);
    index_t mult_n = std::max<index_t>(1, split_n / block_n);
    index_t mult_m = std::max<index_t>(1, split_m / step_m);
    index_t mult_k = std::max<index_t>(1, split_k / step_k);
    return (mult_k * radix_m + mult_m) * radix_n + mult_n;
  };
  slinky::expr splits = slinky::call::make(impl, {m, n, k, block_n});

  // Extract the splits from the single 32-bit index_t result.
  splits = runtime.globals.get(splits, "dot_splits");
  slinky::expr mult_n = splits % radix_n;
  slinky::expr splits_m_k = splits / radix_n;
  slinky::expr mult_m = splits_m_k % radix_m;
  slinky::expr mult_k = splits_m_k / radix_m;
  slinky::expr split_n = slinky::min(n, mult_n * block_n);
  slinky::expr split_m = slinky::min(m, mult_m * step_m);
  slinky::expr split_k = slinky::min(k, mult_k * step_k);
  split_n = runtime.globals.get(split_n, "split_n");
  split_m = runtime.globals.get(split_m, "split_m");
  split_k = runtime.globals.get(split_k, "split_k");

  // The choose_split_factors is an opaque call behind global variables, but its
  // result has provable structure: each split is min(extent, mult * step)
  // with a multiplier in [1, max_multiplier]. Register those intervals for
  // the globals so scheduler proofs and simplifications can reason about the
  // steps without evaluating the call.
  runtime.globals.learn_bounds(
      split_m,
      {slinky::min(m, step_m), slinky::min(m, max_multiplier_m * step_m)});
  runtime.globals.learn_bounds(
      split_n,
      {slinky::min(n, block_n), slinky::min(n, max_multiplier_n * block_n)});
  runtime.globals.learn_bounds(
      split_k,
      {slinky::min(k, step_k), slinky::min(k, max_multiplier_k * step_k)});

  return {split_n, split_m, split_k};
}

}  // namespace

scheduling_info make_dot_schedule(
    ynn_runtime& runtime, const ynn_runtime_value& input_a,
    const ynn_runtime_value& input_b, const ynn_runtime_value& output,
    const std::vector<slinky::var>& all_dims,
    const std::vector<slinky::expr>& all_extents,
    const std::vector<slinky::var>& output_dims, size_t num_k_dims,
    bool packed_b, bool transposed_a, int block_n_unpacked,
    float seconds_per_mac) {
  const int b_k1_dim = packed_b ? 2 : 1;
  const int b_k2_dim = packed_b ? 3 : 1;
  const int num_a_k_dims = num_k_dims + (transposed_a ? 2 : 0);

  slinky::expr block_n = packed_b ? input_b.extent(1) : block_n_unpacked;
  slinky::expr n = output.extent(0);
  slinky::expr m = output.extent(1);

  // Compute k from b because it is more likely to be constant.
  slinky::expr k = input_b.extent(b_k1_dim);
  if (packed_b) {
    k *= input_b.extent(0);
  }
  for (size_t d = 1; d < num_k_dims; ++d) {
    k *= input_b.extent(b_k2_dim + d);
  }

  auto [split_n, split_m, split_k] = choose_split_factors(
      runtime, m, n, k, block_n, type_size_bytes(input_a.type),
      type_size_bits(input_b.type), type_size_bytes(output.type),
      seconds_per_mac, input_a.is_internal());

  const int rank = output.rank();
  const bool is_split_k = slinky::prove_true(split_k < k);
  std::vector<int> loop_order;
  // We need to choose whether m or n is an outermost loop. n outermost allows
  // fusing producers of B, while m outermost allows fusing producers of A.
  // We should prefer fusing the larger of A or B if both are candidates for
  // fusion.
  // TODO: b/555814090 - More sophisticated automatic scheduling would help.
  const bool n_outer = rank >= 2 && !input_b.is_static() &&
                       (!input_a.is_internal() || !slinky::prove_true(n < m));
  if (is_split_k) {
    if (n_outer) {
      loop_order.push_back(num_k_dims + 1);  // m (innermost)
      loop_order.push_back(num_k_dims);      // n
    } else {
      loop_order.push_back(num_k_dims);  // n (innermost)
      if (rank >= 2) {
        loop_order.push_back(num_k_dims + 1);  // m
      }
    }
    for (size_t i = 0; i < num_k_dims; ++i) {
      loop_order.push_back(i);  // k (outermost reduction loop)
    }
    for (size_t i = 2; i < rank; ++i) {
      loop_order.push_back(num_k_dims + i);  // batch dims (outermost)
    }
  } else if (n_outer) {
    loop_order.resize(num_k_dims + 2);
    for (size_t i = 0; i < loop_order.size(); ++i) {
      loop_order[i] = i;
    }
    // Loop over n first so we don't redundantly compute the packing for
    // each split of m.
    std::swap(loop_order[num_k_dims], loop_order[num_k_dims + 1]);
  }

  // Provide splits only for the reduction dims and n/m. The batch dims come
  // after these in all_dims and are intentionally left out: make_schedule
  // auto-computes a cache-aware tile for them.
  std::vector<slinky::expr> splits;

  // If output is rank >= 2, we want to split n, m, and k. Otherwise, we only
  // split n and k (e.g. fully-connected layers).
  if (is_split_k) {
    splits.push_back(split_k);
  } else {
    splits.push_back({});
  }
  for (size_t i = 1; i < num_k_dims; ++i) {
    // Do not create loops for the remaining k dims.
    splits.push_back({});
  }
  splits.push_back(split_n);
  if (output.rank() >= 2) {
    splits.push_back(split_m);
  }

  auto sched = runtime.make_schedule(
      all_dims, all_extents, output.buffer->elem_size(), splits, loop_order);

  // We want to use exactly these loop splits for two innermost dot loops.
  for (size_t dim_idx = 0; dim_idx < std::min<size_t>(output_dims.size(), 2);
       ++dim_idx) {
    slinky::var sym = output_dims[dim_idx];
    for (size_t i = 0; i < sched.loop_splits.size(); ++i) {
      if (sched.loop_splits[i].var == sym) {
        sched.loop_splits[i].step_is_required = true;
        break;
      }
    }
  }

  // The real bounds of the packed input's blocks_n dimension are block
  // indices (j / block_n), which breaks the scheduler's source region
  // inference. Declare a virtual 1-to-1 mapping with `j` instead, so the
  // pack (and anything feeding it) can be fused with loops derived from j.
  sched.input_scheduler_bounds.resize(2);
  if (packed_b) {
    sched.input_scheduler_bounds[1].resize(4);
    sched.input_scheduler_bounds[1][3] = slinky::point(output_dims[0]);
  }
  if (transposed_a) {
    sched.input_scheduler_bounds[0].resize(num_a_k_dims + 1);
    if (output_dims.size() >= 2) {
      sched.input_scheduler_bounds[0][num_a_k_dims] =
          slinky::point(output_dims[1]);
    }
  }

  return sched;
}

}  // namespace ynn
