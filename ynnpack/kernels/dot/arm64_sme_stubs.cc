// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// This file contains stubs of the AArch64 SME ABI support routines, defined in
// the AAPCS64.
// See:
// https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst#81sme-support-routines.
//
// We don't expect these to actually be called because we don't call SME
// functions from other SME functions (that we don't expect to inline).
// These functions are missing on some platforms, so to avoid linker problems,
// we define a fake implementation (that will crash if they actually get
// called).

#include <cstdint>

#include "ynnpack/base/base.h"

typedef struct sme_state {
  int64_t x0;
  int64_t x1;
} sme_state_t;

extern "C" {

sme_state_t __arm_sme_state() { YNN_UNREACHABLE; }
void __arm_tpidr2_restore(void* blk) { YNN_UNREACHABLE; }
void __arm_tpidr2_save(void) { YNN_UNREACHABLE; }
void __arm_za_disable(void) { YNN_UNREACHABLE; }

}  // extern "C"
