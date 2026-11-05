// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// This file contains stubs of the AArch64 SME ABI support routines, defined in
// the AAPCS64.
// See:
// https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst#81sme-support-routines.
//
// These routines are missing on some platforms, so we define weak fallbacks to
// avoid linker errors.
//
// They must be no-ops rather than XNN_UNREACHABLE: when compiling without +sve,
// clang calls __arm_get_current_vg in the prologue of every
// __arm_locally_streaming function, and __arm_tpidr2_save is called whenever
// one __arm_new("za") microkernel calls another out of line (the caller arms
// TPIDR2_EL0, and the callee's prologue then branches to it). Doing nothing is
// safe here because non-streaming SVE state is unused and callers do not read
// their own ZA state after such calls; trapping instead crashes at runtime.
//
// Per AAPCS64 §8.1, these routines must preserve almost all registers:
// __arm_get_current_vg preserves x1–x15 (and vector/predicate registers), while
// __arm_sme_state preserves x2–x15 and __arm_tpidr2_save/restore/za_disable
// preserve x0–x13. Callers (e.g. __arm_locally_streaming functions) keep
// incoming arguments in x9 across the call.
//
// They must be declared naked with inline assembly to avoid compiler prologues,
// spills, or sanitizer instrumentation, and no_profile_instrument_function to
// prevent coverage profiling (gcov / -fprofile-arcs) from injecting counter
// increments that clobber x8/x9.

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"

#if XNN_ARCH_ARM64 && (XNN_ENABLE_ARM_SME || XNN_ENABLE_ARM_SME2) && \
    XNN_ENABLE_ARM_SME_ACLE

typedef struct sme_state {
  int64_t x0;
  int64_t x1;
} sme_state_t;

// On Windows/COFF (clang-cl / MSVC), __declspec(selectany) is only valid on
// data symbols; use linker /alternatename directives to provide weak function
// fallbacks instead.
#if defined(_MSC_VER)
#define XNN_SME_STUB_DECL(ret, name)                                     \
  __pragma(comment(linker, "/alternatename:" #name "=__xnn_" #name)) ret \
      __xnn_##name
#else
#define XNN_SME_STUB_DECL(ret, name) __attribute__((weak)) ret name
#endif

#if XNN_COMPILER_CLANG
#define XNN_STUB_ATTRIBUTES \
  __attribute__((naked, no_profile_instrument_function))

XNN_STUB_ATTRIBUTES XNN_SME_STUB_DECL(uint64_t, __arm_get_current_vg)(void) {
  __asm__("mov x0, xzr\n\tret");
}

XNN_STUB_ATTRIBUTES XNN_SME_STUB_DECL(sme_state_t, __arm_sme_state)(void) {
  __asm__("mov x0, xzr\n\tmov x1, xzr\n\tret");
}

XNN_STUB_ATTRIBUTES XNN_SME_STUB_DECL(void, __arm_tpidr2_restore)(void* blk) {
  __asm__("ret");
}

XNN_STUB_ATTRIBUTES XNN_SME_STUB_DECL(void, __arm_tpidr2_save)(void) {
  __asm__("ret");
}

XNN_STUB_ATTRIBUTES XNN_SME_STUB_DECL(void, __arm_za_disable)(void) {
  __asm__("ret");
}

#undef XNN_STUB_ATTRIBUTES
#else
// GCC and MSVC cl.exe do not support __attribute__((naked)) on AArch64.
#if XNN_COMPILER_GCC
#define XNN_STUB_ATTRIBUTES __attribute__((no_profile_instrument_function))
#else
#define XNN_STUB_ATTRIBUTES
#endif

XNN_STUB_ATTRIBUTES XNN_SME_STUB_DECL(uint64_t, __arm_get_current_vg)(void) {
  return 0;
}

XNN_STUB_ATTRIBUTES XNN_SME_STUB_DECL(sme_state_t, __arm_sme_state)(void) {
  sme_state_t s = {0, 0};
  return s;
}

XNN_STUB_ATTRIBUTES XNN_SME_STUB_DECL(void, __arm_tpidr2_restore)(void* blk) {
  (void) blk;
}

XNN_STUB_ATTRIBUTES XNN_SME_STUB_DECL(void, __arm_tpidr2_save)(void) {}

XNN_STUB_ATTRIBUTES XNN_SME_STUB_DECL(void, __arm_za_disable)(void) {}

#undef XNN_STUB_ATTRIBUTES
#endif  // XNN_COMPILER_CLANG

#if XNN_COMPILER_CLANG || XNN_COMPILER_GCC
#define XNN_SC_STUB_ATTRIBUTES \
  __attribute__((target("general-regs-only"), no_profile_instrument_function))
#else
#define XNN_SC_STUB_ATTRIBUTES
#endif

XNN_SC_STUB_ATTRIBUTES XNN_SME_STUB_DECL(void*, __arm_sc_memcpy)(
    void* dest, const void* src, size_t n) {
  volatile uint8_t* d = (volatile uint8_t*)dest;
  const volatile uint8_t* s = (const volatile uint8_t*)src;
  for (size_t i = 0; i < n; ++i) {
    d[i] = s[i];
  }
  return dest;
}

XNN_SC_STUB_ATTRIBUTES XNN_SME_STUB_DECL(void*, __arm_sc_memmove)(
    void* dest, const void* src, size_t n) {
  volatile uint8_t* d = (volatile uint8_t*)dest;
  const volatile uint8_t* s = (const volatile uint8_t*)src;
  if (d < s) {
    for (size_t i = 0; i < n; ++i) {
      d[i] = s[i];
    }
  } else if (d > s) {
    for (size_t i = n; i > 0; --i) {
      d[i - 1] = s[i - 1];
    }
  }
  return dest;
}

XNN_SC_STUB_ATTRIBUTES XNN_SME_STUB_DECL(void*, __arm_sc_memset)(void* dest,
                                                                 int c,
                                                                 size_t n) {
  volatile uint8_t* d = (volatile uint8_t*)dest;
  const uint8_t v = (uint8_t)c;
  for (size_t i = 0; i < n; ++i) {
    d[i] = v;
  }
  return dest;
}

#undef XNN_SC_STUB_ATTRIBUTES
#undef XNN_SME_STUB_DECL

#endif  // XNN_ARCH_ARM64 && (XNN_ENABLE_ARM_SME || XNN_ENABLE_ARM_SME2) &&
        // XNN_ENABLE_ARM_SME_ACLE
