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

#include <stdint.h>

#include "src/xnnpack/common.h"

#if XNN_ARCH_ARM64 && (XNN_ENABLE_ARM_SME || XNN_ENABLE_ARM_SME2) && \
    XNN_ENABLE_ARM_SME_ACLE && !defined(_WIN32)

typedef struct sme_state {
  int64_t x0;
  int64_t x1;
} sme_state_t;

#define XNN_STUB_ATTRIBUTES \
  __attribute__((naked, no_profile_instrument_function))

XNN_WEAK_SYMBOL XNN_STUB_ATTRIBUTES uint64_t __arm_get_current_vg(void) {
  __asm__("mov x0, xzr\n\tret");
}

XNN_WEAK_SYMBOL XNN_STUB_ATTRIBUTES sme_state_t __arm_sme_state(void) {
  __asm__("mov x0, xzr\n\tmov x1, xzr\n\tret");
}

XNN_WEAK_SYMBOL XNN_STUB_ATTRIBUTES void __arm_tpidr2_restore(void* blk) {
  __asm__("ret");
}

XNN_WEAK_SYMBOL XNN_STUB_ATTRIBUTES void __arm_tpidr2_save(void) {
  __asm__("ret");
}

XNN_WEAK_SYMBOL XNN_STUB_ATTRIBUTES void __arm_za_disable(void) {
  __asm__("ret");
}

#undef XNN_STUB_ATTRIBUTES

#endif  // XNN_ARCH_ARM64 && (XNN_ENABLE_ARM_SME || XNN_ENABLE_ARM_SME2) &&
        // XNN_ENABLE_ARM_SME_ACLE && !defined(_WIN32)
