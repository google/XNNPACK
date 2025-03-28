// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef THIRD_PARTY_XNNPACK_BENCH_GEMM_BENCHMARK_H_
#define THIRD_PARTY_XNNPACK_BENCH_GEMM_BENCHMARK_H_

#include <cstddef>

#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/pack.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h"
#endif  // XNN_ENABLE_KLEIDIAI

#include "bench/gemm.h"  // IWYU pragma: export
#include "bench/utils.h"
#include <benchmark/benchmark.h>

void GEMMBenchmark(benchmark::State& state, xnn_qs8_gemm_minmax_ukernel_fn gemm,
                   xnn_init_qs8_conv_minmax_params_fn init_params,
                   xnn_pack_qs8_gemm_fn pack, size_t mr, size_t nr, size_t kr,
                   size_t sr, benchmark::utils::IsaCheckFunction isa_check);

void GEMMBenchmark(benchmark::State& state,
                   xnn_qs8_qc8w_gemm_minmax_ukernel_fn gemm,
                   xnn_init_qs8_qc8w_conv_minmax_params_fn init_params,
                   xnn_pack_qs8_gemm_fn pack, size_t mr, size_t nr, size_t kr,
                   size_t sr, benchmark::utils::IsaCheckFunction isa_check);

void GEMMBenchmark(benchmark::State& state,
                   xnn_qd8_f16_qc8w_gemm_ukernel_fn gemm,
                   xnn_init_f16_minmax_params_fn init_params,
                   xnn_pack_qs8_gemm_fn pack, size_t mr, size_t nr, size_t kr,
                   size_t sr, benchmark::utils::IsaCheckFunction isa_check);

void GEMMBenchmark(benchmark::State& state,
                   xnn_qd8_f32_qc8w_gemm_ukernel_fn gemm,
                   xnn_init_f32_minmax_params_fn init_params,
                   xnn_pack_qs8_gemm_fn pack, size_t mr, size_t nr, size_t kr,
                   size_t sr, benchmark::utils::IsaCheckFunction isa_check);

void GEMMBenchmark(benchmark::State& state,
                   xnn_qd8_f16_qb4w_gemm_ukernel_fn gemm,
                   xnn_init_f16_qb4w_minmax_params_fn init_params,
                   xnn_pack_qs8_qb4w_gemm_fn pack, size_t mr, size_t nr,
                   size_t kr, size_t sr,
                   benchmark::utils::IsaCheckFunction isa_check);

void GEMMBenchmark(benchmark::State& state,
                   xnn_qd8_f16_qc4w_gemm_ukernel_fn gemm,
                   xnn_init_f16_qc4w_minmax_params_fn init_params,
                   xnn_pack_qs8_qc4w_gemm_fn pack, size_t mr, size_t nr,
                   size_t kr, size_t sr,
                   benchmark::utils::IsaCheckFunction isa_check);

void GEMMBenchmark(benchmark::State& state,
                   xnn_qd8_f32_qb4w_gemm_ukernel_fn gemm,
                   xnn_init_f32_qb4w_minmax_params_fn init_params,
                   xnn_pack_qs8_qb4w_gemm_fn pack, size_t mr, size_t nr,
                   size_t kr, size_t sr,
                   benchmark::utils::IsaCheckFunction isa_check);

void GEMMBenchmark(benchmark::State& state,
                   xnn_qd8_f32_qc4w_gemm_ukernel_fn gemm,
                   xnn_init_f32_qc4w_minmax_params_fn init_params,
                   xnn_pack_qs8_qc4w_gemm_fn pack, size_t mr, size_t nr,
                   size_t kr, size_t sr,
                   benchmark::utils::IsaCheckFunction isa_check);

void GEMMBenchmark(benchmark::State& state,
                   xnn_qp8_f32_qc4w_gemm_minmax_ukernel_fn gemm,
                   xnn_init_f32_minmax_params_fn init_minmax_params,
                   xnn_pack_weights_and_biases_fn pack_weights,
                   xnn_packed_stride_weights_and_biases_fn packed_stride,
                   size_t mr, size_t nr, size_t kr, size_t sr, size_t mr_packed,
                   benchmark::utils::IsaCheckFunction isa_check);

void GEMMBenchmark(benchmark::State& state,
                   xnn_qp8_f32_qb4w_gemm_minmax_ukernel_fn gemm,
                   xnn_init_f32_qb4w_minmax_params_fn init_params,
                   xnn_pack_weights_and_biases_fn pack_weights,
                   xnn_packed_stride_weights_and_biases_fn packed_stride,
                   size_t mr, size_t nr, size_t kr, size_t sr, size_t mr_packed,
                   benchmark::utils::IsaCheckFunction isa_check);

void GEMMBenchmark(benchmark::State& state, xnn_qu8_gemm_minmax_ukernel_fn gemm,
                   xnn_init_qu8_conv_minmax_params_fn init_params,
                   xnn_pack_qu8_gemm_fn pack, size_t mr, size_t nr, size_t kr,
                   size_t sr,
                   benchmark::utils::IsaCheckFunction isa_check = nullptr);

void GEMMBenchmark(benchmark::State& state, xnn_f32_gemm_minmax_ukernel_fn gemm,
                   xnn_init_f32_minmax_params_fn init_params,
                   xnn_pack_f32_gemm_fn pack, size_t mr, size_t nr, size_t kr,
                   size_t sr,
                   benchmark::utils::IsaCheckFunction isa_check = nullptr);

void GEMMBenchmark(benchmark::State& state, xnn_f32_gemm_minmax_ukernel_fn gemm,
                   xnn_init_f32_minmax_params_fn init_params, size_t mr,
                   size_t nr, size_t kr, size_t sr,
                   benchmark::utils::IsaCheckFunction isa_check = nullptr);

void GEMMBenchmark(benchmark::State& state, xnn_f16_gemm_minmax_ukernel_fn gemm,
                   xnn_init_f16_minmax_params_fn init_params,
                   xnn_pack_f16_gemm_fn pack, size_t mr, size_t nr, size_t kr,
                   size_t sr,
                   benchmark::utils::IsaCheckFunction isa_check = nullptr);

#if XNN_ENABLE_KLEIDIAI
void GEMMBenchmark(benchmark::State& state,
                   xnn_pf32_gemm_minmax_ukernel_fn gemm,
                   xnn_init_f32_minmax_params_fn init_minmax_params,
                   xnn_pack_weights_and_biases_fn pack_weights,
                   xnn_packed_stride_weights_and_biases_fn packed_stride,
                   size_t mr, size_t nr, size_t kr, size_t sr, size_t mr_packed,
                   benchmark::utils::IsaCheckFunction isa_check);
#endif  // XNN_ENABLE_KLEIDIAI

#endif  // THIRD_PARTY_XNNPACK_BENCH_GEMM_BENCHMARK_H_
