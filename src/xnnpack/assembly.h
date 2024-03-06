// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifdef __wasm__
  .macro BEGIN_FUNCTION name
    .text
    .section    .text.\name,"",@
    .hidden     \name
    .globl      \name
    .type       \name,@function
    \name:
  .endm

  .macro END_FUNCTION name
    end_function
  .endm
#elif defined(__ELF__)
  .macro BEGIN_FUNCTION name
    .text
    .p2align 4
    .global \name
    .internal \name
    .type \name, %function
    \name:
  .endm

  .macro END_FUNCTION name
    .size \name, .-\name
  .endm
#elif defined(__MACH__)
  .macro BEGIN_FUNCTION name
    .text
    .p2align 4
    .global _\name
    .private_extern _\name
    _\name:
  .endm

  .macro END_FUNCTION name
  .endm
#elif defined(_WIN64)
  .macro BEGIN_FUNCTION name
    .text
    .p2align 4
    .global \name
    \name:
  .endm

  .macro END_FUNCTION name
  .endm
#elif defined(_WIN32)
  .macro BEGIN_FUNCTION name
    .text
    .p2align 4
    .global _\name
    _\name:
  .endm

  .macro END_FUNCTION name
  .endm
#endif
