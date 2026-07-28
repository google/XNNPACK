# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//ynnpack:build_defs.bzl", "ynn_benchmark_deps", "ynn_cc_library", "ynn_test_deps")

_COMPATIBLE_WITH = []

ynn_cc_library(
    name = "grouped_dot",
    srcs = ["grouped_dot.cc"],
    hdrs = ["grouped_dot.h"],
    compatible_with = _COMPATIBLE_WITH,
    visibility = ["//ynnpack:__subpackages__"],
    deps = [
        "//ynnpack/base",
        "//ynnpack/kernels/dot",
    ],
)

cc_test(
    name = "grouped_dot_test",
    srcs = ["test.cc"],
    compatible_with = _COMPATIBLE_WITH,
    deps = [
        ":grouped_dot",
        "//ynnpack/kernels/dot",
        "//ynnpack/kernels/dot:pack",
    ] + ynn_test_deps(),
)

cc_binary(
    name = "bench",
    srcs = ["bench.cc"],
    compatible_with = _COMPATIBLE_WITH,
    deps = [
        ":grouped_dot",
        "//ynnpack/kernels/dot",
        "//ynnpack/kernels/dot:pack",
    ] + ynn_benchmark_deps(),
)
