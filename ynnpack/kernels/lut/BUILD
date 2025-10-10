# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

_COMPATIBLE_WITH = []

cc_library(
    name = "lut",
    srcs = ["lut.cc"],
    hdrs = [
        "kernels.inc",
        "lut.h",
    ],
    compatible_with = _COMPATIBLE_WITH,
    visibility = ["//ynnpack:__subpackages__"],
    deps = [
        "//ynnpack:ynnpack_h",
        "//ynnpack/base",
    ],
)

cc_test(
    name = "test",
    srcs = ["test.cc"],
    deps = [
        ":lut",
        "//ynnpack/base",
        "//ynnpack/base/test:buffer",
        "//ynnpack/base/test:util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "bench",
    srcs = ["bench.cc"],
    args = ["--benchmark_min_time=1x"],
    deps = [
        ":lut",
        "//ynnpack/base",
        "//ynnpack/base/test:buffer",
        "//ynnpack/base/test:random",
        "@com_google_benchmark//:benchmark_main",
    ],
)
