// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/arch.h"

#include <cstddef>
#include <cstdint>

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
    if (cpuinfo_has_arm_sme()) result |= arch_flag::sme;
    if (cpuinfo_has_arm_sme2()) result |= arch_flag::sme2;
    if (cpuinfo_has_arm_sve()) result |= arch_flag::sve;
#endif  // YNN_ARCH_ARM
#endif  // YNN_ENABLE_CPUINFO
#ifdef YNN_ARCH_HEXAGON
    result |= arch_flag::hvx;
#endif  // YNN_ARCH_HEXAGON
    return result;
  }();
  return flags;
}

}  // namespace ynn
