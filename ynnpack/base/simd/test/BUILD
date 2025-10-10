# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//ynnpack:build_defs.bzl", "ynn_cc_library")

package(default_visibility = ["//ynnpack:__subpackages__"])

ynn_cc_library(
    name = "tests",
    testonly = True,
    per_arch_srcs = {
        # This test defines different tests depending on which architecture is being compiled.
        "x86_sse2": [
            "test.cc",
            "convert_only_test.cc",
        ],
        "x86_sse41": ["test.cc"],
        "x86_avx": ["test.cc"],
        "x86_avx2": [
            "test.cc",
            "convert_only_test.cc",
        ],
        "x86_f16c": ["convert_only_test.cc"],
        "x86_avx512bf16": ["convert_only_test.cc"],
        "x86_avx512fp16": ["convert_only_test.cc"],
        "x86_avx512f": ["test.cc"],
        "x86_avx512bw": ["test.cc"],
        "arm_neon": [
            "convert_only_test.cc",
            "test.cc",
        ],
    },
    deps = [
        "//ynnpack/base",  # buildcleaner:keep
        "//ynnpack/base/simd",  # buildcleaner:keep
        "@com_google_googletest//:gtest",
    ],
    alwayslink = True,
)

cc_test(
    name = "test",
    tags = ["simd_wrapper_test"],
    deps = [
        ":tests",
        "@com_google_googletest//:gtest_main",
    ],
)
