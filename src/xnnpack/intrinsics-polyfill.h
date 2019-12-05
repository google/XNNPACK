// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once


#ifdef __AVX512F__
#include <immintrin.h>

// gcc before 7.1 lacks _cvtu32_mask16
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && (__GNUC__ < 7)
static inline __mmask16 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
_cvtu32_mask16(unsigned int mask) {
  return (__mmask16) mask;
}
#endif

#endif  // __AVX512F__
