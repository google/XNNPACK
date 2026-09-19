// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Benchmarks `packer::pack` for the B-packing shapes the AMX bf16 dot kernels
// use.
//
// `kernels/transpose:bench` is not a substitute for this: it calls the
// interleave kernel once with `m = factor` and a very large `n`, which is the
// opposite of how `packer::pack` uses it (many calls, each with `n` equal to a
// block width of 48 or 96). The per-call overhead this is meant to measure is
// invisible at that shape.
//
// Both the fused and the per-row variants are compiled into this one binary and
// selected at runtime, so the comparison is not confounded by code layout
// differing between two separately linked binaries.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/kernels/dot/pack.h"
#include <benchmark/benchmark.h>

namespace ynn {
namespace {

constexpr size_t tile_m = 2;
constexpr size_t elem_size_bits = 16;

void bench_pack_b(benchmark::State& state, bool allow_fused) {
  const size_t k = state.range(0);
  const size_t n = state.range(1);
  const size_t tile_n = state.range(2);

  const size_t elem_size = elem_size_bits / 8;
  const size_t input_stride = n * elem_size;
  const size_t output_stride = tile_n * tile_m * elem_size;
  const size_t output_block_stride = ceil_div(k, tile_m) * output_stride;
  const size_t output_bytes = ceil_div(n, tile_n) * output_block_stride;

  std::vector<uint16_t> input(k * n);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<uint16_t>(i);
  }
  std::vector<uint16_t> output(output_bytes / sizeof(uint16_t));

  packer p(/*transpose=*/false, elem_size_bits, tile_m, tile_n, allow_fused);
  if (allow_fused) {
    packer ref(/*transpose=*/false, elem_size_bits, tile_m, tile_n,
               /*allow_fused=*/false);
    std::vector<uint16_t> expected(output.size(), 0xCDCD);
    std::vector<uint16_t> actual(output.size(), 0xCDCD);
    ref.pack(k, n, input_stride, input.data(), output_stride,
             output_block_stride, expected.data());
    p.pack(k, n, input_stride, input.data(), output_stride, output_block_stride,
           actual.data());
    if (actual != expected) {
      state.SkipWithError("Fused pack output does not match reference!");
      return;
    }
  }

  for (auto s : state) {
    p.pack(k, n, input_stride, input.data(), output_stride, output_block_stride,
           output.data());
    benchmark::DoNotOptimize(output.data());
  }

  state.SetBytesProcessed(state.iterations() * output_bytes);
  // Number of calls the per-row contract would make, for reference.
  state.counters["calls"] = ceil_div(n, tile_n) * ceil_div(k, tile_m);
}

// Shapes are chosen to cover three distinct regimes:
//   - (1024, 96): L2-resident single panel.
//   - (240, 250): Unaligned N requiring tail padding.
//   - (1024, 2304) & (1152, 4096): Large L3/DRAM-bound shapes where 2D
//     cache-blocked strip-mining (chunk_n=4, chunk_m=32) prevents DTLB and
//     cache line thrashing across adjacent panels.
void add_args(benchmark::Benchmark* b) {
  const std::pair<int, int> shapes[] = {
      {1024, 96},
      {240, 250},
      {1024, 2304},
      {1152, 4096},
  };
  for (int tile_n : {32, 48, 64, 96}) {
    for (auto [k, n] : shapes) {
      b->Args({k, n, tile_n});
    }
  }
}

void bench_pack_dispatch(benchmark::State& state, bool hoisted) {
  const size_t k = state.range(0);
  const size_t n = state.range(1);
  const size_t tile_n = state.range(2);

  const size_t input_stride = n * (elem_size_bits / 8);
  const size_t output_stride = tile_n * tile_m * (elem_size_bits / 8);
  const size_t output_block_stride = ceil_div(k, tile_m) * output_stride;
  const size_t output_bytes = ceil_div(n, tile_n) * output_block_stride;

  std::vector<uint16_t> input(k * n, 1);
  std::vector<uint16_t> output(output_bytes / sizeof(uint16_t));

  const packer p_hoisted(/*transpose=*/false, elem_size_bits, tile_m,
                         /*tile_n=*/1);

  for (auto s : state) {
    if (hoisted) {
      const packer p = p_hoisted.with_tile_n(tile_n);
      p.pack(k, n, input_stride, input.data(), output_stride,
             output_block_stride, output.data());
    } else {
      const packer p(/*transpose=*/false, elem_size_bits, tile_m, tile_n);
      p.pack(k, n, input_stride, input.data(), output_stride,
             output_block_stride, output.data());
    }
    benchmark::DoNotOptimize(output.data());
  }

  state.SetBytesProcessed(state.iterations() * output_bytes);
}

void add_dispatch_args(benchmark::internal::Benchmark* b) {
  b->Args({16, 32, 32});
  b->Args({16, 48, 48});
  b->Args({32, 64, 64});
  b->Args({1024, 2304, 64});
}

BENCHMARK_CAPTURE(bench_pack_b, per_row, false)->Apply(add_args);
BENCHMARK_CAPTURE(bench_pack_b, fused, true)->Apply(add_args);
BENCHMARK_CAPTURE(bench_pack_dispatch, per_tile_lookup, false)
    ->Apply(add_dispatch_args);
BENCHMARK_CAPTURE(bench_pack_dispatch, hoisted, true)->Apply(add_dispatch_args);

}  // namespace
}  // namespace ynn
