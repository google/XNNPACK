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
// They must be no-ops rather than XNN_UNREACHABLE: __arm_tpidr2_save is really
// called whenever one __arm_new("za") microkernel calls another out of line
// (the caller arms TPIDR2_EL0, and the callee's prologue then branches to it),
// which happens whenever a large packing helper is too big for clang to inline
// into its dispatcher. Doing nothing is safe here because such a caller never
// reads its own ZA state after the call; trapping instead crashes at runtime.

#include <stdint.h>

#include "src/xnnpack/common.h"

typedef struct sme_state {
  int64_t x0;
  int64_t x1;
} sme_state_t;

XNN_WEAK_SYMBOL sme_state_t __arm_sme_state(void) { return (sme_state_t){0, 0}; }
XNN_WEAK_SYMBOL void __arm_tpidr2_restore(void* blk) {}
XNN_WEAK_SYMBOL void __arm_tpidr2_save(void) {}
XNN_WEAK_SYMBOL void __arm_za_disable(void) {}
