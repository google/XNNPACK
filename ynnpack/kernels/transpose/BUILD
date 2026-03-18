# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//ynnpack:build_defs.bzl", "ynn_benchmark_deps", "ynn_binary_linkopts", "ynn_binary_malloc", "ynn_cc_library", "ynn_kernel_copts", "ynn_test_deps")

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
    copts = ynn_kernel_copts(),
    per_arch_srcs = {
        "arm_neon": ["arm_neon.cc"],
        "arm64_sve": ["arm64_sve.cc"],
        "x86_sse2": ["x86_sse2.cc"],
        "x86_avx": ["x86_avx.cc"],
        "x86_avx2": ["x86_avx2.cc"],
        "x86_avx512": ["x86_avx512.cc"],
        "hexagon_hvx": ["hexagon_hvx.cc"],
    },
    visibility = ["//ynnpack:__subpackages__"],
    deps = [
        "//ynnpack/base",
        "//ynnpack/base:log",
        "//ynnpack/base/simd",  # buildcleaner: keep
    ],
)

cc_test(
    name = "test",
    srcs = [
        "switch_element_size.h",
        "test.cc",
    ],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":transpose",
        "//ynnpack/base",
        "//ynnpack/base/test:tensor",
        "//ynnpack/base/test:util",
    ] + ynn_test_deps(),
)

cc_test(
    name = "bench",
    srcs = [
        "bench.cc",
        "switch_element_size.h",
    ],
    args = ["--benchmark_min_time=1x"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":transpose",
        "//ynnpack/base",
        "//ynnpack/base/test:tensor",
    ] + ynn_benchmark_deps(),
)
