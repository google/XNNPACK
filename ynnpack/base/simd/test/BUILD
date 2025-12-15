# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//ynnpack:build_defs.bzl", "ynn_binary_linkopts", "ynn_binary_malloc", "ynn_cc_library", "ynn_test_deps")

package(default_visibility = ["//ynnpack:__subpackages__"])

ynn_cc_library(
    name = "tests",
    testonly = True,
    hdrs = ["generic.h"],
    per_arch_srcs = {
        # This test defines different tests depending on which architecture is being compiled.
        "arm_neon": ["arm_neon.cc"],
        "arm_neonfp16": ["arm_neonfp16.cc"],
        "arm_neonfma": ["arm_neonfma.cc"],
        "x86_sse2": ["x86_sse2.cc"],
        "x86_sse41": ["x86_sse41.cc"],
        "x86_avx": ["x86_avx.cc"],
        "x86_avx2": ["x86_avx2.cc"],
        "x86_avx512f": ["x86_avx512f.cc"],
        "x86_avx512bw": ["x86_avx512bw.cc"],
        "x86_f16c": ["x86_f16c.cc"],
        "x86_fma3": ["x86_fma3.cc"],
    },
    deps = [
        "//ynnpack/base",  # buildcleaner:keep
        "//ynnpack/base/simd",  # buildcleaner:keep
        "//ynnpack/base/test:fuzz_test",
        "//ynnpack/base/test:random",
        "@com_google_googletest//:gtest",
    ],
    alwayslink = True,
)

cc_test(
    name = "test",
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    tags = ["simd_wrapper_test"],
    deps = [
        ":tests",
    ] + ynn_test_deps(),
)
