// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#ifdef _MSC_VER
  #include <intrin.h>  
#endif

#ifdef __hexagon__
  #include <hexagon_protos.h>

  // Use fetchl2 at least several hundred cycles prior to using the data.
  XNN_INLINE static void xnn_prefetch_to_l2(const void *address,
                                        uint32_t stride,
                                        uint32_t width,
                                        uint32_t height,
                                        uint32_t iter)
  {
    uint64_t info = HEXAGON_V64_CREATE_H(iter, stride, width, height);
    Q6_l2fetch_AP(address, info);
  }
#endif

#include "xnnpack/common.h"


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
    #elif defined(__hexagon__)
      Q6_dcfetch_A(address);
    #else
      #error "Architecture-specific implementation of xnn_prefetch_to_l1 required"
    #endif
  #else
    #error "Compiler-specific implementation of xnn_prefetch_to_l1 required"
  #endif
}
