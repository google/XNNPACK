// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__APPLE__)
  #include <TargetConditionals.h>
#endif


// Define architecture identification macros

#if defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__) || defined(_M_IX86)
  #define XNN_ARCH_X86 1
#else
  #define XNN_ARCH_X86 0
#endif

#if defined(__x86_64__) || defined(__x86_64) || defined(_M_X64) && !defined(_M_ARM64EC)
  #define XNN_ARCH_X86_64 1
#else
  #define XNN_ARCH_X86_64 0
#endif

#if defined(__arm__) || defined(_M_ARM)
  #define XNN_ARCH_ARM 1
#else
  #define XNN_ARCH_ARM 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
  #define XNN_ARCH_ARM64 1
#else
  #define XNN_ARCH_ARM64 0
#endif

#if defined(__PPC64__) || defined(__ppc64__) || defined(__powerpc64__) || defined(_ARCH_PPC64)
  #define XNN_ARCH_PPC64 1
#else
  #define XNN_ARCH_PPC64 0
#endif

#if defined(__riscv) || defined(__riscv__)
  #define XNN_ARCH_RISCV 1
#else
  #define XNN_ARCH_RISCV 0
#endif

#if defined(__hexagon__)
  #define XNN_ARCH_HEXAGON 1
#else
  #define XNN_ARCH_HEXAGON 0
#endif

#if defined(__wasm__)
  #if defined(__wasm_relaxed_simd__)
    #define XNN_ARCH_WASM 0
    #define XNN_ARCH_WASMSIMD 0
    #define XNN_ARCH_WASMRELAXEDSIMD 1
  #elif defined(__wasm_simd128__)
    #define XNN_ARCH_WASM 0
    #define XNN_ARCH_WASMSIMD 1
    #define XNN_ARCH_WASMRELAXEDSIMD 0
  #else
    #define XNN_ARCH_WASM 1
    #define XNN_ARCH_WASMSIMD 0
    #define XNN_ARCH_WASMRELAXEDSIMD 0
  #endif
#else
  #define XNN_ARCH_WASM 0
  #define XNN_ARCH_WASMSIMD 0
  #define XNN_ARCH_WASMRELAXEDSIMD 0
#endif

// Define platform identification macros

#if defined(__ANDROID__)
  #define XNN_PLATFORM_ANDROID 1
#else
  #define XNN_PLATFORM_ANDROID 0
#endif

#if defined(__linux__)
  #define XNN_PLATFORM_LINUX 1
#else
  #define XNN_PLATFORM_LINUX 0
#endif

#if defined(__APPLE__) && TARGET_OS_IPHONE
  // iOS on iPhone / iPad Touch, iPad OS, watchOS, or tvOS
  #define XNN_PLATFORM_IOS 1
#else
  #define XNN_PLATFORM_IOS 0
#endif

#if defined(__APPLE__) && TARGET_OS_MAC
  #define XNN_PLATFORM_MAC 1
#else
  #define XNN_PLATFORM_MAC 0
#endif

// Mobile build x86 versions for debugging
#if XNN_PLATFORM_ANDROID || XNN_PLATFORM_IOS
  #define XNN_PLATFORM_MOBILE 1
#else
  #define XNN_PLATFORM_MOBILE 0
#endif

#if defined(__EMSCRIPTEN__) || defined(__wasm__)
  #define XNN_PLATFORM_WEB 1
#else
  #define XNN_PLATFORM_WEB 0
#endif

#if defined(_WIN32)
  #define XNN_PLATFORM_WINDOWS 1
#else
  #define XNN_PLATFORM_WINDOWS 0
#endif

#if defined(__Fuchsia__)
  #define XNN_PLATFORM_FUCHSIA 1
#else
  #define XNN_PLATFORM_FUCHSIA 0
#endif

#if defined(__hexagon__) && !defined(__linux__)
  #define XNN_PLATFORM_QURT 1
#else
  #define XNN_PLATFORM_QURT 0
#endif

#if XNN_PLATFORM_WINDOWS
  #define XNN_HAS_MMAP 0
#else
  #define XNN_HAS_MMAP 1
#endif

#if XNN_PLATFORM_WINDOWS
  #define XNN_HAS_PTHREADS 0
#else
  #define XNN_HAS_PTHREADS 1
#endif

// Define compile identification macros

#if defined(__clang__)
  #define XNN_COMPILER_CLANG 1
#elif defined(__INTEL_COMPILER)
  #define XNN_COMPILER_ICC 1
#elif defined(_MSC_VER)
  #define XNN_COMPILER_MSVC 1
#elif defined(__GNUC__)
  #define XNN_COMPILER_GCC 1
#endif

#ifndef XNN_COMPILER_CLANG
  #define XNN_COMPILER_CLANG 0
#endif

#ifndef XNN_COMPILER_GCC
  #define XNN_COMPILER_GCC 0
#endif

#ifndef XNN_COMPILER_MSVC
  #define XNN_COMPILER_MSVC 0
#endif

#ifndef XNN_COMPILER_ICC
  #define XNN_COMPILER_ICC 0
#endif

#ifndef XNN_MAX_UARCH_TYPES
  #if (XNN_ARCH_ARM || XNN_ARCH_ARM64) && !XNN_PLATFORM_IOS
    #define XNN_MAX_UARCH_TYPES 3
  #else
    #define XNN_MAX_UARCH_TYPES 1
  #endif
#endif

#define XNN_UARCH_DEFAULT 0

#if defined(__has_builtin)
  #define XNN_COMPILER_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
  #define XNN_COMPILER_HAS_BUILTIN(builtin) 0
#endif

#if defined(__has_feature)
  #define XNN_COMPILER_HAS_FEATURE(builtin) __has_feature(builtin)
#else
  #define XNN_COMPILER_HAS_FEATURE(builtin) 0
#endif

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

#if defined(__GNUC__)
  #define XNN_ALIGN(alignment) __attribute__((__aligned__(alignment)))
#elif defined(_MSC_VER)
  #define XNN_ALIGN(alignment) __declspec(align(alignment))
#else
  #error "Platform-specific implementation of XNN_ALIGN required"
#endif

#if defined(__GNUC__)
  #define XNN_UNALIGNED __attribute__((__aligned__(1)))
#elif defined(_MSC_VER)
  #if defined(_M_IX86)
    #define XNN_UNALIGNED
  #else
    #define XNN_UNALIGNED __unaligned
  #endif
#else
  #error "Platform-specific implementation of XNN_UNALIGNED required"
#endif

#define XNN_COUNT_OF(array) (sizeof(array) / sizeof(0[array]))

#if defined(__cplusplus) || XNN_COMPILER_MSVC || XNN_COMPILER_CLANG
  // static as array indices in function parameter declaration is a C99 feature, not supported in C++.
  // MSVC does not support this feature, even in C mode.
  // Clang generates suboptimal code, see https://github.com/llvm/llvm-project/issues/59120
  #define XNN_MIN_ELEMENTS(count) count
#else
  #define XNN_MIN_ELEMENTS(count) static count
#endif

#if defined(__cplusplus) || XNN_COMPILER_MSVC
  #define XNN_RESTRICT
#else
  #define XNN_RESTRICT restrict
#endif

#if defined(__GNUC__)
  #define XNN_LIKELY(condition) (__builtin_expect(!!(condition), 1))
  #define XNN_UNLIKELY(condition) (__builtin_expect(!!(condition), 0))
#else
  #define XNN_LIKELY(condition) (!!(condition))
  #define XNN_UNLIKELY(condition) (!!(condition))
#endif

#if XNN_COMPILER_HAS_BUILTIN(__builtin_unpredictable)
  #define XNN_UNPREDICTABLE(condition) (__builtin_unpredictable(!!(condition)))
#elif defined(__GNUC__) && (__GNUC__ >= 9) && !defined(__INTEL_COMPILER)
  #define XNN_UNPREDICTABLE(condition) (__builtin_expect_with_probability(!!(condition), 0, 0.5))
#else
  #define XNN_UNPREDICTABLE(condition) (!!(condition))
#endif

#if XNN_COMPILER_HAS_FEATURE(thread_sanitizer)
  #define XNN_DISABLE_TSAN __attribute__((__no_sanitize__("thread")))
#else
  #define XNN_DISABLE_TSAN
#endif

#if XNN_COMPILER_HAS_FEATURE(memory_sanitizer)
  #define XNN_DISABLE_MSAN __attribute__((__no_sanitize__("memory")))
#else
  #define XNN_DISABLE_MSAN
#endif

#if XNN_COMPILER_HAS_FEATURE(hwaddress_sanitizer)
  #define XNN_DISABLE_HWASAN __attribute__((__no_sanitize__("hwaddress")))
#else
  #define XNN_DISABLE_HWASAN
#endif

#define XNN_OOB_READS XNN_DISABLE_TSAN XNN_DISABLE_MSAN XNN_DISABLE_HWASAN

#if defined(__GNUC__)
  #define XNN_FALLTHROUGH __attribute__((fallthrough));
#else
  #define XNN_FALLTHROUGH /* fall through */
#endif

#if defined(__GNUC__)
  #define XNN_INTRINSIC inline __attribute__((__always_inline__, __artificial__))
#elif defined(_MSC_VER)
  #define XNN_INTRINSIC __forceinline
#else
  #define XNN_INTRINSIC inline
#endif

#if defined(__GNUC__)
  #define XNN_INLINE inline __attribute__((__always_inline__))
#elif defined(_MSC_VER)
  #define XNN_INLINE __forceinline
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

#ifndef XNN_WEAK_SYMBOL
  #if defined(_WIN32)
    #define XNN_WEAK_SYMBOL __declspec(selectany)
  #elif XNN_COMPILER_CLANG || XNN_COMPILER_GCC || XNN_COMPILER_ICC
    #define XNN_WEAK_SYMBOL __attribute__((weak))
  #else
    #define XNN_WEAK_SYMBOL
  #endif
#endif

#if defined(__clang__)
  #define XNN_PRAGMA_CLANG(pragma) _Pragma(pragma)
#else
  #define XNN_PRAGMA_CLANG(pragma)
#endif

#if XNN_ARCH_WASM
  #define XNN_ALLOCATION_ALIGNMENT 4
#elif XNN_ARCH_X86 || XNN_ARCH_X86_64
  #if XNN_PLATFORM_MOBILE
    #define XNN_ALLOCATION_ALIGNMENT 32
  #else
    #define XNN_ALLOCATION_ALIGNMENT 64
  #endif
#elif XNN_ARCH_HEXAGON
  #define XNN_ALLOCATION_ALIGNMENT 128
#else
  #define XNN_ALLOCATION_ALIGNMENT 16
#endif

// Number of extra elements to allocate for DWCONV accumulators/buffers.
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  // For AVX512.
  #define XNN_MAX_SIMD_SIZE 64
#elif XNN_ARCH_HEXAGON
  #define XNN_MAX_SIMD_SIZE 128
#else
  // XNN_ARCH_ARM, XNN_ARCH_ARM64, XNN_ARCH_WASM, XNN_ARCH_WASMSIMD, XNN_ARCH_WASMRELAXEDSIMD, XNN_ARCH_RISVC.
  // Wasm/Scalar gavgpool microkernels can over-read by 4 buffers.
  #define XNN_MAX_SIMD_SIZE 16
#endif

// Use constant here to avoid dependency on xnnpack.h
#if XNN_MAX_SIMD_SIZE >= 16
  #define XNN_MULTIPASS_EXTRA_BYTES XNN_MAX_SIMD_SIZE
#else
  #define XNN_MULTIPASS_EXTRA_BYTES 16
#endif

#if XNN_ARCH_ARM || XNN_ARCH_X86 || XNN_ARCH_X86_64 || XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  // These architectures are slow to broadcast, the compiler tries to move them
  // into loops, and when it runs out of registers, it will redundantly perform
  // the broadcast. Marking them as memory read by assembly forces the compiler
  // to maintain the value in memory.
  #if defined(__GNUC__)
    #define XNN_FORCE_REALIZATION(x) __asm volatile(""::"m"(x));
  #else
    #define XNN_FORCE_REALIZATION(x)
  #endif
#else
  #define XNN_FORCE_REALIZATION(x)
#endif

#define XNN_LOG2_SIZEOF_INT8_T   0  // log2(sizeof(int8_t))
#define XNN_LOG2_SIZEOF_UINT8_T  0  // log2(sizeof(uint8_t))
#define XNN_LOG2_SIZEOF_INT16_T  1  // log2(sizeof(int16_t))
#define XNN_LOG2_SIZEOF_UINT16_T 1  // log2(sizeof(uint16_t))
#define XNN_LOG2_SIZEOF_HALF     1  // log2(sizeof(half))
#define XNN_LOG2_SIZEOF_FLOAT    2  // log2(sizeof(float))
#define XNN_LOG2_SIZEOF_INT32_T  2  // log2(sizeof(int32_t))
#define XNN_LOG2_SIZEOF_UINT32_T 2  // log2(sizeof(uint32_t))
