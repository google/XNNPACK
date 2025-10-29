# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//ynnpack:build_defs.bzl", "ynn_cc_library")

_COMPATIBLE_WITH = []

cc_library(
    name = "generic",
    hdrs = [
        "generic.h",
        "min_max_accumulator.h",
        "sum_accumulator.h",
    ],
    compatible_with = _COMPATIBLE_WITH,
    deps = [
        "//ynnpack/base",
        "//ynnpack/base/simd",  # buildcleaner: keep
    ],
)

ynn_cc_library(
    name = "reduce",
    srcs = [
        "generic.h",
        "min_max_accumulator.h",
        "reduce.cc",
        "sum_accumulator.h",
    ],
    hdrs = [
        "max.inc",
        "min.inc",
        "min_max.inc",
        "reduce.h",
        "sum.inc",
        "sum_squared.inc",
    ],
    compatible_with = _COMPATIBLE_WITH,
    copts = [
        "-fno-unroll-loops",
        "-O2",
    ],
    per_arch_srcs = {
        "arm_neonfp16arith": [
            "arm_neonfp16arith.cc",
            "arm_neon_xf16.h",
        ],
        "arm_neonbf16": [
            "arm_neonbf16.cc",
            "arm_neon_xf16.h",
        ],
        "arm_neondot": ["arm_neondot.cc"],
        "arm_neon": ["arm_neon.cc"],
        "x86_ssse3": ["x86_ssse3.cc"],
        "x86_sse2": ["x86_sse2.cc"],
        "x86_sse41": ["x86_sse41.cc"],
        "x86_avx512bf16": ["x86_avx512bf16.cc"],
        "x86_avx512bw": ["x86_avx512bw.cc"],
        "x86_avx512f": ["x86_avx512f.cc"],
        "x86_f16c": ["x86_f16c.cc"],
        "x86_avx512fp16": ["x86_avx512fp16.cc"],
        "x86_avx2": ["x86_avx2.cc"],
    },
    visibility = ["//ynnpack:__subpackages__"],
    deps = [
        ":generic",
        "//ynnpack:ynnpack_h",
        "//ynnpack/base",
        "//ynnpack/base:log",
        "//ynnpack/base/simd",  # buildcleaner: keep
    ],
)

cc_test(
    name = "test",
    srcs = ["test.cc"],
    deps = [
        ":reduce",
        "//ynnpack/base",
        "//ynnpack/base/test:buffer",
        "//ynnpack/base/test:fuzz_test",
        "//ynnpack/base/test:random",
        "//ynnpack/base/test:tensor",
        "//ynnpack/base/test:util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "bench",
    srcs = ["bench.cc"],
    args = ["--benchmark_min_time=1x"],
    deps = [
        ":reduce",
        "//ynnpack/base",
        "//ynnpack/base/test:buffer",
        "//ynnpack/base/test:tensor",
        "@com_google_benchmark//:benchmark",
    ],
)
