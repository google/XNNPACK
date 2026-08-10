# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_test.bzl", "cc_test")
load(
    "//ynnpack:build_defs.bzl",
    "ynn_benchmark_deps",
    "ynn_binary_linkopts",
    "ynn_binary_malloc",
    "ynn_cc_library",
    "ynn_kernel_copts",
    "ynn_test_deps",
)

_COMPATIBLE_WITH = []

ynn_cc_library(
    name = "grouped_dot",
    srcs = ["grouped_dot.cc"],
    hdrs = ["grouped_dot.h"],
    compatible_with = _COMPATIBLE_WITH,
    copts = ynn_kernel_copts(),
    visibility = ["//ynnpack:__subpackages__"],
    deps = [
        "//ynnpack/base",
        "//ynnpack/kernels/dot",
    ],
)

cc_test(
    name = "test",
    srcs = ["test.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":grouped_dot",
        "//ynnpack/base",
        "//ynnpack/kernels/dot",
        "//ynnpack/kernels/dot:pack",
    ] + ynn_test_deps(),
)

cc_test(
    name = "bench",
    srcs = ["bench.cc"],
    args = ["--benchmark_min_time=1x"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":grouped_dot",
        "//ynnpack/base",
        "//ynnpack/kernels/dot",
        "//ynnpack/kernels/dot:pack",
    ] + ynn_benchmark_deps(),
)
