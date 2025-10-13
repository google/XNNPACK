// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_BASE_H_
#define XNNPACK_YNNPACK_BASE_BASE_H_

#if defined(__has_builtin)
#define YNN_COMPILER_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#define YNN_COMPILER_HAS_BUILTIN(builtin) 0
#endif

#if defined(__has_feature)
#define YNN_COMPILER_HAS_FEATURE(builtin) __has_feature(builtin)
#else
#define YNN_COMPILER_HAS_FEATURE(builtin) 0
#endif

#if !defined(__has_attribute)
#define YNN_COMPILER_HAS_ATTRIBUTE(x) 0
#else
#define YNN_COMPILER_HAS_ATTRIBUTE(x) __has_attribute(x)
#endif

#if defined(__GNUC__)
#define YNN_ALWAYS_INLINE inline __attribute__((__always_inline__))
#elif defined(_MSC_VER)
#define YNN_ALWAYS_INLINE __forceinline
#else
#define YNN_ALWAYS_INLINE inline
#endif

#if YNN_COMPILER_HAS_ATTRIBUTE(unused)
#define YNN_UNUSED __attribute__((unused))
#else
#define YNN_UNUSED
#endif

#if defined(__GNUC__)
#if defined(__clang__) || (__GNUC__ > 4 || __GNUC__ == 4 && __GNUC_MINOR__ >= 5)
#define YNN_UNREACHABLE      \
  do {                       \
    __builtin_unreachable(); \
  } while (0)
#else
#define YNN_UNREACHABLE \
  do {                  \
    __builtin_trap();   \
  } while (0)
#endif
#elif defined(_MSC_VER)
#define YNN_UNREACHABLE __assume(0)
#else
#define YNN_UNREACHABLE \
  do {                  \
  } while (0)
#endif

#if defined(__GNUC__)
#define YNN_ALIGN(alignment) __attribute__((__aligned__(alignment)))
#elif defined(_MSC_VER)
#define YNN_ALIGN(alignment) __declspec(align(alignment))
#else
#error "Platform-specific implementation of YNN_ALIGN required"
#endif

// TODO: std::hardware_destructive_interference_size would be better, but it's
// not defined on ARM?
#define YNN_CACHE_LINE_SIZE 64

#define YNN_ALLOCA(T, N) reinterpret_cast<T*>(__builtin_alloca((N) * sizeof(T)))

#endif  // XNNPACK_YNNPACK_BASE_BASE_H_
