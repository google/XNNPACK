// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifdef __ELF__
  .macro BEGIN_FUNCTION name
    .text
    .p2align 4
    .global \name
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
#endif
