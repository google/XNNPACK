// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#ifdef _MSC_VER
  #include <intrin.h>
#endif

#include <xnnpack/common.h>


XNN_INLINE static void xnn_prefetch_to_l1(const void* address) {
  #if defined(__GNUC__)
    __builtin_prefetch(address);
  #elif defined(_MSC_VER)
    #if defined(_M_ARM) || defined(_M_ARM64) || defined(_M_ARM64EC)
      __prefetch((void*) address);
    #elif defined(_M_X64)
      _mm_prefetch(address, _MM_HINT_T0);
    #elif defined(_M_IX86)
      #if _M_IX86_FP >= 1
        // Targeting SSE+
        _mm_prefetch(address, _MM_HINT_T0);
      #else
        _m_prefetch((void*) address);
      #endif
    #else
      #error "Architecture-specific implementation of xnn_prefetch_to_l1 required"
    #endif
  #else
    #error "Compiler-specific implementation of xnn_prefetch_to_l1 required"
  #endif
}
