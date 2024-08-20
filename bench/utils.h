// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>

#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/memory.h"

#include <benchmark/benchmark.h>

namespace benchmark {
namespace utils {

uint32_t WipeCache();
uint32_t PrefetchToL1(const void* ptr, size_t size);

// Disable support for denormalized numbers in floating-point units.
void DisableDenormals();

// Return clock rate, in Hz, for the currently used logical processor.
uint64_t GetCurrentCpuFrequency();

// Return maximum (across all cores/clusters/sockets) last level cache size.
// Can overestimate, but not underestimate LLC size.
size_t GetMaxCacheSize();

// Set number of elements for a reduction microkernel such that:
// - It is divisible by 2, 3, 4, 5, 6.
// - It is divisible by AVX512 width.
// - Total memory footprint does not exceed the characteristic cache size for
//   the architecture.
template<class InType>
void ReductionParameters(benchmark::internal::Benchmark* benchmark) {
  benchmark->ArgName("N");

  size_t characteristic_l1 = 32 * 1024;
  size_t characteristic_l2 = 256 * 1024;
#if XNN_ARCH_ARM
  characteristic_l1 = 16 * 1024;
  characteristic_l2 = 128 * 1024;
#endif  // XNN_ARCH_ARM

  const size_t elementwise_size = sizeof(InType);
  benchmark->Arg(characteristic_l1 / elementwise_size / 960 * 960);
  benchmark->Arg(characteristic_l2 / elementwise_size / 960 * 960);
}

// Set number of elements for a unary elementwise microkernel such that:
// - It is divisible by 2, 3, 4, 5, 6.
// - It is divisible by AVX512 width.
// - Total memory footprint does not exceed the characteristic cache size for
//   the architecture.
template<class InType, class OutType>
void UnaryElementwiseParameters(benchmark::internal::Benchmark* benchmark) {
  benchmark->ArgName("N");

  size_t characteristic_l1 = 32 * 1024;
  size_t characteristic_l2 = 256 * 1024;
#if XNN_ARCH_ARM
  characteristic_l1 = 16 * 1024;
  characteristic_l2 = 128 * 1024;
#endif  // XNN_ARCH_ARM

  const size_t elementwise_size = sizeof(InType) + sizeof(OutType);
  benchmark->Arg(characteristic_l1 / elementwise_size / 960 * 960);
  benchmark->Arg(characteristic_l2 / elementwise_size / 960 * 960);
}

// Set number of elements for a binary elementwise microkernel such that:
// - It is divisible by 2, 3, 4, 5, 6.
// - It is divisible by AVX512 width.
// - Total memory footprint does not exceed the characteristic cache size for
//   the architecture.
template<class InType, class OutType>
void BinaryElementwiseParameters(benchmark::internal::Benchmark* benchmark) {
  benchmark->ArgName("N");

  size_t characteristic_l1 = 32 * 1024;
  size_t characteristic_l2 = 256 * 1024;
#if XNN_ARCH_ARM
  characteristic_l1 = 16 * 1024;
  characteristic_l2 = 128 * 1024;
#endif  // XNN_ARCH_ARM

  const size_t elementwise_size = 2 * sizeof(InType) + sizeof(OutType);
  benchmark->Arg(characteristic_l1 / elementwise_size / 960 * 960);
  benchmark->Arg(characteristic_l2 / elementwise_size / 960 * 960);
}

// Set multi-threading parameters appropriate for the processor.
void MultiThreadingParameters(benchmark::internal::Benchmark* benchmark);

typedef bool (*IsaCheckFunction)(benchmark::State& state);

// Check if either ARM VFPv2 or VFPv3 extension is supported.
// If VFP is unsupported, report error in benchmark state, and return false.
bool CheckVFP(benchmark::State& state);

// Check if ARMv6 extensions are supported.
// If ARMv6 extensions are unsupported, report error in benchmark state, and return false.
bool CheckARMV6(benchmark::State& state);

// Check if ARM FP16-ARITH extension is supported.
// If FP16-ARITH is unsupported, report error in benchmark state, and return false.
bool CheckFP16ARITH(benchmark::State& state);

// Check if ARM NEON extension is supported.
// If NEON is unsupported, report error in benchmark state, and return false.
bool CheckNEON(benchmark::State& state);

// Check if ARM NEON-FP16 extension is supported.
// If NEON-FP16 is unsupported, report error in benchmark state, and return false.
bool CheckNEONFP16(benchmark::State& state);

// Check if ARM NEON-FMA extension is supported.
// If NEON-FMA is unsupported, report error in benchmark state, and return false.
bool CheckNEONFMA(benchmark::State& state);

// Check if ARMv8 NEON instructions are supported.
// If ARMv8 NEON is unsupported, report error in benchmark state, and return false.
bool CheckNEONV8(benchmark::State& state);

// Check if ARM NEON-FP16-ARITH extension is supported.
// If NEON-FP16-ARITH is unsupported, report error in benchmark state, and return false.
bool CheckNEONFP16ARITH(benchmark::State& state);

// Check if ARM NEON-BF16 extension is supported.
// If NEON-BF16 is unsupported, report error in benchmark state, and return false.
bool CheckNEONBF16(benchmark::State& state);

// Check if ARM DOT extension is supported.
// If DOT is unsupported, report error in benchmark state, and return false.
bool CheckNEONDOT(benchmark::State& state);

// Check if ARM I8MM extension is supported.
// If I8MM is unsupported, report error in benchmark state, and return false.
bool CheckNEONI8MM(benchmark::State& state);

// Check if RISC-V V (vector) extension is supported.
// If V is unsupported, report error in benchmark state, and return false.
bool CheckRVV(benchmark::State& state);

// Check if RISC-V V (vector) FP16-ARITH extension is supported.
// If RVV-FP16-ARITH is unsupported, report error in benchmark state, and return false.
bool CheckRVVFP16ARITH(benchmark::State& state);

// Check if x86 SSSE3 extension is supported.
// If SSSE3 is unsupported, report error in benchmark state, and return false.
bool CheckSSSE3(benchmark::State& state);

// Check if x86 SSE4.1 extension is supported.
// If SSE4.1 is unsupported, report error in benchmark state, and return false.
bool CheckSSE41(benchmark::State& state);

// Check if x86 AVX extension is supported.
// If AVX is unsupported, report error in benchmark state, and return false.
bool CheckAVX(benchmark::State& state);

// Check if x86 F16C extension is supported.
// If F16C is unsupported, report error in benchmark state, and return false.
bool CheckF16C(benchmark::State& state);

// Check if x86 FMA3 extension is supported.
// If FMA3 is unsupported, report error in benchmark state, and return false.
bool CheckFMA3(benchmark::State& state);

// Check if x86 AVX2 extension is supported.
// If AVX2 is unsupported, report error in benchmark state, and return false.
bool CheckAVX2(benchmark::State& state);

// Check if x86 AVX512F extension is supported.
// If AVX512F is unsupported, report error in benchmark state, and return false.
bool CheckAVX512F(benchmark::State& state);

// Check if x86 SKX-level AVX512 extensions (AVX512F, AVX512CD, AVX512BW, AVX512DQ, and AVX512VL) are supported.
// If SKX-level AVX512 extensions are unsupported, report error in benchmark state, and return false.
bool CheckAVX512SKX(benchmark::State& state);

// Check if x86 VBMI + SKX-level AVX512 extensions (AVX512F, AVX512CD, AVX512BW, AVX512DQ, and AVX512VL) are supported.
// If VBMI or SKX-level AVX512 extensions are unsupported, report error in benchmark state, and return false.
bool CheckAVX512VBMI(benchmark::State& state);

// Check if x86 VNNI + SKX-level AVX512 extensions (AVX512F, AVX512CD, AVX512BW, AVX512DQ, and AVX512VL) are supported.
// If VNNI or SKX-level AVX512 extensions are unsupported, report error in benchmark state, and return false.
bool CheckAVX512VNNI(benchmark::State& state);

// Check if x86 VNNI + GFNI + SKX-level AVX512 extensions (AVX512F, AVX512CD, AVX512BW, AVX512DQ, AVX512VL, and GFNI) are supported.
// If VNNI or GFNI or SKX-level AVX512 extensions are unsupported, report error in benchmark state, and return false.
bool CheckAVX512VNNIGFNI(benchmark::State& state);

// Check if x86 VNNI + GFNI + SKX-level + AMX AVX512 extensions (AAVX512F, AVX512CD, AVX512BW, AVX512DQ, AVX512VL, GFNI and AMX) are supported.
// If AVX512 or AMX are unsupported, report error in benchmark state, and return false.
bool CheckAVX512AMX(benchmark::State& state);

// Check if x86 VNNI + GFNI + SKX-level + FP16 AVX512 extensions (AAVX512F, AVX512CD, AVX512BW, AVX512DQ, AVX512VL, GFNI and FP16) are supported.
// If AVX512 or FP16 are unsupported, report error in benchmark state, and return false.
bool CheckAVX512FP16(benchmark::State& state);

// Check if x86 AVX-VNNI extension is supported.
// If AVX-VNNI extension is unsupported, report error in benchmark state, and return false.
bool CheckAVXVNNI(benchmark::State& state);

// Check if x86 AVX256SKX extension is supported.
// If AVX256SKX extension is unsupported, report error in benchmark state, and return false.
bool CheckAVX256SKX(benchmark::State& state);

// Check if x86 AVXVNNI + AVX10 or AVX512 is supported
// If VNNI or SKX-level AVX256 extensions are unsupported, report error in benchmark state, and return false.
bool CheckAVX256VNNI(benchmark::State& state);

// Check if x86 VNNI + GFNI + AVX10 or AVX512 is supported
bool CheckAVX256VNNIGFNI(benchmark::State& state);

// Check if Hexagon HVX extension is supported.
// If HVX is unsupported, report error in benchmark state, and return false.
bool CheckHVX(benchmark::State& state);

// Check if PSHUFB instruction is available in WAsm Relaxed SIMD as Relaxed Swizzle.
// If WAsm PSHUFB is unsupported, report error in benchmark state, and return false.
bool CheckWAsmPSHUFB(benchmark::State& state);

// Check if SDOT instruction is available in WAsm Relaxed SIMD as Relaxed Integer Dot Product with Accumulation.
// If WAsm SDOT is unsupported, report error in benchmark state, and return false.
bool CheckWAsmSDOT(benchmark::State& state);

// Check if USDOT instruction is available in WAsm Relaxed SIMD as Relaxed Integer Dot Product with Accumulation.
// If WAsm USDOT is unsupported, report error in benchmark state, and return false.
bool CheckWAsmUSDOT(benchmark::State& state);

// Check if BLENDVPS instruction is available in WAsm Relaxed SIMD as Relaxed Lane Select.
// If WAsm BLENDVPS is unsupported, report error in benchmark state, and return false.
bool CheckWAsmBLENDVPS(benchmark::State& state);

template <class T>
inline T DivideRoundUp(T x, T q) {
  return x / q + T(x % q != 0);
}

template <class T>
inline T RoundUp(T x, T q) {
  return q * DivideRoundUp(x, q);
}

template <class T>
inline T Doz(T a, T b) {
  return a >= b ? a - b : T(0);
}


}  // namespace utils
}  // namespace benchmark
