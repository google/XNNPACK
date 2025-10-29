// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_BUILD_CONFIG_H_
#define XNNPACK_YNNPACK_BASE_BUILD_CONFIG_H_

#define YNN_ALLOCATION_ALIGNMENT 64

#if defined(__x86_64__) || defined(__x86_64) || \
    defined(_M_X64) && !defined(_M_ARM64EC)
#define YNN_ARCH_X86_64
#endif

#if defined(__i386__) || defined(__i486__) || defined(__i586__) || \
    defined(__i686__) || defined(_M_IX86)
#define YNN_ARCH_X86_32
#endif

#if defined(YNN_ARCH_X86_64) || defined(YNN_ARCH_X86_32)
#define YNN_ARCH_X86
#endif

#if defined(__arm__) || defined(_M_ARM)
#define YNN_ARCH_ARM32
#endif

#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
#define YNN_ARCH_ARM64
#endif

#if defined(YNN_ARCH_ARM32) || defined(YNN_ARCH_ARM64)
#define YNN_ARCH_ARM
#endif

// We want to use _Float16 if the compiler supports it fully, but it's
// tricky to do this detection; there are compiler versions that define the
// type in broken ways. We're only going to bother using it if the support is
// known to be at least a robust f16<->f32 conversion, which generally means a
// recent version of Clang or GCC, x86 or ARM or RISC-V architectures, and
// (in some cases) the right architecture flags specified on the command line.

#ifndef YNN_ARCH_FLOAT16

// Some non-GCC compilers define __GNUC__, but we only want to detect the Real
// Thing
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && \
    !defined(__INTEL_LLVM_COMPILER)
#define YNN_GNUC_ACTUAL __GNUC__
#else
#define YNN_GNUC_ACTUAL 0
#endif

#if (defined(__i386__) || defined(__x86_64__)) && defined(__SSE2__) && \
    defined(__FLT16_MAX__) && defined(__F16C__) &&                     \
    ((__clang_major__ >= 15 && !defined(_MSC_VER)) || (YNN_GNUC_ACTUAL >= 12))
#define YNN_ARCH_FLOAT16 1
#endif

#if (defined(YNN_ARCH_ARM) && !defined(_MSC_VER)) && \
    defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
#define YNN_ARCH_FLOAT16 1
#endif

#if defined(__riscv) && defined(__riscv_zvfh) && __clang__ >= 1600
#define YNN_ARCH_FLOAT16 1
#endif

#ifndef YNN_ARCH_FLOAT16
#define YNN_ARCH_FLOAT16 0
#endif

#endif  // YNN_ARCH_FLOAT16

#endif  // XNNPACK_YNNPACK_BASE_BUILD_CONFIG_H_
