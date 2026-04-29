// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/arch.h"

#include <cstddef>
#include <cstdint>

#include "ynnpack/base/base.h"

#ifdef YNN_ENABLE_CPUINFO
#include "ynnpack/base/log.h"
#include <cpuinfo.h>
#endif

namespace ynn {

#if defined(YNN_ARCH_X86_64) && defined(__linux__) && !defined(CHROMIUM)
#include <sys/syscall.h>

#define XFEATURE_XTILEDATA 18
#define ARCH_REQ_XCOMP_PERM 0x1023

ssize_t ynn_syscall(size_t rax, size_t rdi, size_t rsi, size_t rdx) {
  __asm("syscall"
        : "+a"(rax)
        : "D"(rdi), "S"(rsi), "d"(rdx)
        : "rcx", "r11", "memory");
  return rax;
}

bool can_use_amx_tile() {
  return ynn_syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA,
                     0) == 0;
}

#else
bool can_use_amx_tile() { return false; }
#endif

uint64_t get_supported_arch_flags() {
  static uint64_t flags = []() -> uint64_t {
    uint64_t result = 0;
#ifdef YNN_ENABLE_CPUINFO
    if (!cpuinfo_initialize()) {
      YNN_LOG_WARNING() << "Failed to initialize cpuinfo";
      return 0;
    }

#ifdef YNN_ARCH_X86
    result |= arch_flag::sse2;
    if (cpuinfo_has_x86_ssse3()) result |= arch_flag::ssse3;
    if (cpuinfo_has_x86_sse4_1()) result |= arch_flag::sse41;
    if (cpuinfo_has_x86_avx()) result |= arch_flag::avx;
    if (cpuinfo_has_x86_f16c()) result |= arch_flag::f16c;
    if (cpuinfo_has_x86_avx2()) result |= arch_flag::avx2;
    if (cpuinfo_has_x86_fma3()) result |= arch_flag::fma3;
    if (cpuinfo_has_x86_avx512f()) result |= arch_flag::avx512f;
    if (cpuinfo_has_x86_avx512bw()) result |= arch_flag::avx512bw;
    if (cpuinfo_has_x86_avx512vl()) result |= arch_flag::avx512vl;
    if (cpuinfo_has_x86_avx512dq()) result |= arch_flag::avx512dq;
    if (cpuinfo_has_x86_avx512bf16()) result |= arch_flag::avx512bf16;
    if (cpuinfo_has_x86_avx512fp16()) result |= arch_flag::avx512fp16;
    if (cpuinfo_has_x86_avx512vnni()) result |= arch_flag::avx512vnni;
    if (cpuinfo_has_x86_amx_tile() && can_use_amx_tile()) {
      if (cpuinfo_has_x86_amx_bf16()) result |= arch_flag::amxbf16;
      if (cpuinfo_has_x86_amx_fp16()) result |= arch_flag::amxfp16;
      if (cpuinfo_has_x86_amx_int8()) result |= arch_flag::amxint8;
    }
#endif  // YNN_ARCH_X86
#ifdef YNN_ARCH_ARM
    if (cpuinfo_has_arm_neon()) result |= arch_flag::neon;
    if (cpuinfo_has_arm_neon_fma()) result |= arch_flag::neonfma;
    if (cpuinfo_has_arm_neon_dot()) result |= arch_flag::neondot;
    if (cpuinfo_has_arm_neon_fp16()) result |= arch_flag::neonfp16;
    if (cpuinfo_has_arm_neon_fp16_arith()) result |= arch_flag::neonfp16arith;
    if (cpuinfo_has_arm_neon_bf16()) result |= arch_flag::neonbf16;
    if (cpuinfo_has_arm_i8mm()) result |= arch_flag::neoni8mm;
#if !YNN_COMPILER_HAS_FEATURE(memory_sanitizer)
    // msan (understandably) does not support SVE/SME (b/494230133).
    if (cpuinfo_has_arm_sme()) result |= arch_flag::sme;
    if (cpuinfo_has_arm_sme2()) result |= arch_flag::sme2;
    if (cpuinfo_has_arm_sve()) result |= arch_flag::sve;
#endif  //  YNN_COMPILER_HAS_FEATURE(memory_sanitizer)
#endif  // YNN_ARCH_ARM
#endif  // YNN_ENABLE_CPUINFO
#ifdef YNN_ARCH_HEXAGON
    result |= arch_flag::hvx;
#endif  // YNN_ARCH_HEXAGON
#ifdef YNN_ARCH_WASM
    result |= arch_flag::simd128;
#endif  // YNN_ARCH_WASM
    return result;
  }();
  return flags;
}

size_t get_l2_cache_size() {
  static const size_t size = []() -> size_t {
    // Conservative default when cpuinfo isn't available: 1 MiB. This is
    // within a small factor of what typical Cortex-A7xx / Neoverse cores
    // have per core, and large enough that kc stays usefully big for
    // typical GEMM shapes (for N <= 4096, f32, it keeps kc >= 64).
    constexpr size_t kFallback = 1 * 1024 * 1024;
#ifdef YNN_ENABLE_CPUINFO
    if (!cpuinfo_initialize()) {
      return kFallback;
    }
    const uint32_t count = cpuinfo_get_l2_caches_count();
    if (count == 0) {
      return kFallback;
    }
    // Pick the L2 with the largest per-thread share. On asymmetric systems
    // (Apple M-series P+E, Arm big.LITTLE) this deterministically selects
    // the performance cluster, which is where latency-critical GEMM work
    // lands. On homogeneous systems every L2 yields the same answer. Also
    // track the total size of the selected L2 for the second branch below.
    size_t best_per_thread = 0;
    size_t best_total = 0;
    for (uint32_t i = 0; i < count; ++i) {
      const struct cpuinfo_cache* l2 = cpuinfo_get_l2_cache(i);
      if (l2 == nullptr || l2->size == 0) continue;
      const uint32_t sharers =
          l2->processor_count > 0 ? l2->processor_count : 1;
      const size_t per_thread = static_cast<size_t>(l2->size) / sharers;
      if (per_thread > best_per_thread) {
        best_per_thread = per_thread;
        best_total = static_cast<size_t>(l2->size);
      }
    }
    if (best_per_thread == 0) return kFallback;
    // Two bounds, take the larger:
    //
    //   per_thread * 2: assumes all sharers run GEMM concurrently; the 2x
    //     absorbs graceful spillover into outer SLC/L3 on Apple M-series
    //     and Neoverse cores.
    //
    //   total * 3/4: on a physically-shared L2 cluster (M-series P-cluster,
    //     big.LITTLE P-cluster) a single-threaded GEMM gets near-full L2 —
    //     the per-thread model under-counts. The 1/4 headroom covers A/C
    //     tiles, TLB, and prefetch interference near capacity.
    //
    // On per-core L2 systems (sharers=1) the first always dominates, so
    // behavior there is unchanged. The second fires only on shared clusters,
    // which is exactly the topology the per-thread model under-counts.
    //
    // Both are tuned for single-threaded latency. Under MT on clustered L2,
    // threads each using 3/4 of total oversubscribe the cache — the
    // long-term fix is plumbing active-thread count from pthreadpool.
    const size_t per_thread_budget = best_per_thread * 2;
    const size_t shared_cluster_budget = best_total - best_total / 4;
    return per_thread_budget > shared_cluster_budget ? per_thread_budget
                                                     : shared_cluster_budget;
#else
    return kFallback;
#endif
  }();
  return size;
}

}  // namespace ynn
