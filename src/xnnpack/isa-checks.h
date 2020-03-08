// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cpuinfo.h>

#include <xnnpack/common.h>


#if XNN_ARCH_WASMSIMD
  #define TEST_REQUIRES_PSIMD
#else
  #define TEST_REQUIRES_PSIMD \
    do { \
      if (!cpuinfo_initialize() || !(cpuinfo_has_arm_neon() || cpuinfo_has_x86_sse2())) { \
        GTEST_SKIP(); \
      } \
    } while (0)
#endif

#define TEST_REQUIRES_X86_SSE \
  do { \
    if (!cpuinfo_initialize() || !cpuinfo_has_x86_sse()) { \
      GTEST_SKIP(); \
    } \
  } while (0)

#define TEST_REQUIRES_X86_SSE2 \
  do { \
    if (!cpuinfo_initialize() || !cpuinfo_has_x86_sse2()) { \
      GTEST_SKIP(); \
    } \
  } while (0)

#define TEST_REQUIRES_X86_SSE41 \
  do { \
    if (!cpuinfo_initialize() || !cpuinfo_has_x86_sse4_1()) { \
      GTEST_SKIP(); \
    } \
  } while (0)

#define TEST_REQUIRES_X86_AVX \
  do { \
    if (!cpuinfo_initialize() || !cpuinfo_has_x86_avx()) { \
      GTEST_SKIP(); \
    } \
  } while (0)

#define TEST_REQUIRES_X86_FMA3 \
  do { \
    if (!cpuinfo_initialize() || !cpuinfo_has_x86_fma3()) { \
      GTEST_SKIP(); \
    } \
  } while (0)

#define TEST_REQUIRES_X86_AVX2 \
  do { \
    if (!cpuinfo_initialize() || !cpuinfo_has_x86_avx2()) { \
      GTEST_SKIP(); \
    } \
  } while (0)

#define TEST_REQUIRES_X86_AVX512F \
  do { \
    if (!cpuinfo_initialize() || !cpuinfo_has_x86_avx512f()) { \
      GTEST_SKIP(); \
    } \
  } while (0)

#define TEST_REQUIRES_ARM_NEON \
  do { \
    if (!cpuinfo_initialize() || !cpuinfo_has_arm_neon()) { \
      GTEST_SKIP(); \
    } \
  } while (0)

#define TEST_REQUIRES_ARM_NEON_FMA \
  do { \
    if (!cpuinfo_initialize() || !cpuinfo_has_arm_neon_fma()) { \
      GTEST_SKIP(); \
    } \
  } while (0)

#define TEST_REQUIRES_ARM_NEON_FP16_ARITH \
  do { \
    if (!cpuinfo_initialize() || !cpuinfo_has_arm_neon_fp16_arith()) { \
      GTEST_SKIP(); \
    } \
  } while (0)
