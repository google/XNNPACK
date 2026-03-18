// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_ARCH_H_
#define XNNPACK_YNNPACK_BASE_ARCH_H_

#include <cstdint>

namespace ynn {

// This should be an enum class, but bitfields are messy.
namespace arch_flag {

enum {
  none = 0,
#ifdef YNN_ARCH_X86
  sse2 = 1 << 0,
  ssse3 = 1 << 1,
  sse41 = 1 << 2,
  avx = 1 << 3,
  f16c = 1 << 4,
  avx2 = 1 << 5,
  fma3 = 1 << 6,
  avx512f = 1 << 7,
  avx512bw = 1 << 8,
  avx512vl = 1 << 9,
  avx512dq = 1 << 10,
  avx512bf16 = 1 << 11,
  avx512fp16 = 1 << 12,
  avx512vnni = 1 << 13,
  amxbf16 = 1 << 14,
  amxfp16 = 1 << 15,
  amxint8 = 1 << 16,

  avx2_fma3 = avx2 | fma3,
  avx512 = avx512f | avx512bw | avx512vl | avx512dq,
#endif  // YNN_ARCH_X86
#ifdef YNN_ARCH_ARM
  neon = 1 << 0,
  neonfma = 1 << 1,
  neondot = 1 << 2,
  neonfp16 = 1 << 3,
  neonfp16arith = 1 << 4,
  neonbf16 = 1 << 5,
  neoni8mm = 1 << 6,
  sme = 1 << 7,
  sme2 = 1 << 8,
  sve = 1 << 9,
#endif
#ifdef YNN_ARCH_HEXAGON
  hvx = 1 << 0,
#endif  // YNN_ARCH_HEXAGON
};

}  // namespace arch_flag

uint64_t get_supported_arch_flags();

inline bool is_arch_supported(
    uint64_t arch_flags,
    uint64_t supported_arch_flags = get_supported_arch_flags()) {
  return (arch_flags & supported_arch_flags) == arch_flags;
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_ARCH_H_
