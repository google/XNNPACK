# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//ynnpack:build_defs.bzl", "ynn_cc_library")

_COMPATIBLE_WITH = []

ynn_cc_library(
    name = "transpose",
    srcs = [
        "interleave.cc",
        "transpose.cc",
    ],
    hdrs = [
        "generic.h",
        "interleave.h",
        "interleave.inc",
        "transpose.h",
        "transpose.inc",
    ],
    compatible_with = _COMPATIBLE_WITH,
    copts = ["-O2"],
    per_arch_srcs = {
        "arm_neon": [
            "arm_neon.h",
            "arm_neon.cc",
        ],
        "x86_sse2": [
            "x86_sse2.h",
            "x86_sse2.cc",
        ],
        "x86_avx": ["x86_avx.cc"],
        "x86_avx2": [
            "x86_avx2.h",
            "x86_avx2.cc",
        ],
    },
    visibility = ["//ynnpack:__subpackages__"],
    deps = [
        "//ynnpack/base",
        "//ynnpack/base:log",
    ],
)

cc_test(
    name = "test",
    srcs = ["test.cc"],
    deps = [
        ":transpose",
        "//ynnpack/base",
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
        ":transpose",
        "//ynnpack/base",
        "//ynnpack/base/test:tensor",
        "@com_google_benchmark//:benchmark_main",
    ],
)
