// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "ynnpack/kernels/dot/dot.h"
#include "ynnpack/kernels/dot/pack.h"
#include "ynnpack/kernels/grouped_dot/grouped_dot.h"
#include <benchmark/benchmark.h>

namespace ynn {
namespace {

// Helper to align value.
size_t align_up(size_t v, size_t alignment) {
  return (v + alignment - 1) / alignment * alignment;
}

void BM_GroupedDot(benchmark::State& state) {
  const size_t E = state.range(0);
  const size_t N = state.range(1);
  const size_t K = state.range(2);  // top-k
  const size_t D_in = state.range(3);
  const size_t D_out = state.range(4);
  const bool uniform_routing = state.range(5);

  const size_t NK = N * K;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // 1. Generate routing.
  std::vector<int32_t> expert_indices(NK);
  if (uniform_routing) {
    std::uniform_int_distribution<int32_t> exp_dist(0, E - 1);
    for (size_t i = 0; i < NK; ++i) {
      expert_indices[i] = exp_dist(rng);
    }
  } else {
    // Unbalanced: all tokens go to expert 0.
    std::fill(expert_indices.begin(), expert_indices.end(), 0);
  }

  // Compute counts and offsets.
  std::vector<int32_t> expert_counts(E, 0);
  for (size_t i = 0; i < NK; ++i) {
    expert_counts[expert_indices[i]]++;
  }
  std::vector<int32_t> offsets(E + 1, 0);
  for (size_t e = 0; e < E; ++e) {
    offsets[e + 1] = offsets[e] + expert_counts[e];
  }

  // 2. Generate inputs.
  std::vector<float> input_a(NK * D_in);
  for (auto& v : input_a) v = dist(rng);

  std::vector<float> input_b(E * D_in * D_out);
  for (auto& v : input_b) v = dist(rng);

  std::vector<float> output(NK * D_out, 0.0f);

  // 3. Get dot kernel.
  dot_type type{ynn_type_fp32, ynn_type_fp32, ynn_type_fp32};
  dot_kernel kernel = get_dot_kernel(type);
  if (kernel.kernel == nullptr) {
    state.SkipWithMessage("No dot kernel found");
    return;
  }

  size_t tile_k = kernel.tile_k;
  size_t tile_n = kernel.tile_n;

  // 4. Pack weights.
  size_t num_k_blocks = (D_in + tile_k - 1) / tile_k;
  size_t aligned_n = align_up(D_out, tile_n);
  size_t packed_b_stride = aligned_n * tile_k * sizeof(float);
  size_t expert_packed_size = num_k_blocks * packed_b_stride;

  std::vector<char> packed_b(E * expert_packed_size, 0);
  packer p(/*transpose=*/false, /*elem_size_bits=*/32, tile_k, aligned_n);

  for (size_t e = 0; e < E; ++e) {
    const float* B_e = input_b.data() + e * D_in * D_out;
    char* packed_B_e = packed_b.data() + e * expert_packed_size;
    p.pack(D_in, D_out, D_out * sizeof(float), B_e, packed_b_stride, 0,
           packed_B_e);
  }

  // 5. Benchmark.
  for (auto _ : state) {
    grouped_dot(E, expert_counts.data(), offsets.data(), input_a.data(),
                packed_b.data(), output.data(), D_in, D_out, packed_b_stride,
                kernel);
  }

  // Log GFLOPS.
  // ops per run = NK * D_in * D_out * 2 (multiply-adds)
  const double ops = static_cast<double>(NK * D_in * D_out * 2);
  state.counters["FLOP"] =
      benchmark::Counter(state.iterations() * ops, benchmark::Counter::kIsRate);
}

void BenchmarkArgs(benchmark::internal::Benchmark* b) {
  // Args: E, N, K, D_in, D_out, uniform_routing

  // Gemma MoE shapes
  // Decode (N=1)
  b->Args({32, 1, 4, 512, 448, 1});  // Uniform
  b->Args({32, 1, 4, 512, 448, 0});  // Unbalanced

  // Prefill (N=256)
  b->Args({32, 256, 4, 512, 448, 1});  // Uniform
  b->Args({32, 256, 4, 512, 448, 0});  // Unbalanced
}

BENCHMARK(BM_GroupedDot)->Apply(BenchmarkArgs)->UseRealTime();

}  // namespace
}  // namespace ynn
