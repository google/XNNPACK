// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once


#if defined(__GNUC__)
  #if defined(__clang__) || (__GNUC__ > 4 || __GNUC__ == 4 && __GNUC_MINOR__ >= 5)
    #define XNN_UNREACHABLE do { __builtin_unreachable(); } while (0)
  #else
    #define XNN_UNREACHABLE do { __builtin_trap(); } while (0)
  #endif
#elif defined(_MSC_VER)
  #define XNN_UNREACHABLE __assume(0)
#else
  #define XNN_UNREACHABLE do { } while (0)
#endif

#define XNN_ALIGN(alignment) __attribute__((__aligned__(alignment)))

#define XNN_COUNT_OF(array) (sizeof(array) / sizeof(0[array]))

#if defined(__GNUC__)
  #define XNN_LIKELY(condition) (__builtin_expect(!!(condition), 1))
  #define XNN_UNLIKELY(condition) (__builtin_expect(!!(condition), 0))
#else
  #define XNN_LIKELY(condition) (!!(condition))
  #define XNN_UNLIKELY(condition) (!!(condition))
#endif

// TODO - __builtin_expect_with_probability for GCC 9+
#if defined(__clang__) && (__has_builtin(__builtin_unpredictable))
  #define XNN_UNPREDICTABLE(condition) (__builtin_unpredictable(!!(condition)))
#else
  #define XNN_UNPREDICTABLE(condition) (!!(condition))
#endif

#if defined(__GNUC__)
  #define XNN_INLINE inline __attribute__((__always_inline__))
#else
  #define XNN_INLINE inline
#endif

#ifndef XNN_INTERNAL
  #if defined(__ELF__)
    #define XNN_INTERNAL __attribute__((__visibility__("internal")))
  #elif defined(__MACH__)
    #define XNN_INTERNAL __attribute__((__visibility__("hidden")))
  #else
    #define XNN_INTERNAL
  #endif
#endif

#ifndef XNN_PRIVATE
  #if defined(__ELF__)
    #define XNN_PRIVATE __attribute__((__visibility__("hidden")))
  #elif defined(__MACH__)
    #define XNN_PRIVATE __attribute__((__visibility__("hidden")))
  #else
    #define XNN_PRIVATE
  #endif
#endif
