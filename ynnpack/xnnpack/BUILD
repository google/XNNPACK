# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

_COMPATIBLE_WITH = []

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    compatible_with = _COMPATIBLE_WITH,
    deps = [
        "//:xnnpack_h",
        "//ynnpack",
        "//ynnpack/base",
        "//ynnpack/subgraph",
    ],
)

cc_library(
    name = "dynamic_quantization",
    srcs = ["dynamic_quantization.cc"],
    hdrs = ["dynamic_quantization.h"],
    compatible_with = _COMPATIBLE_WITH,
    deps = [
        ":utils",
        "//ynnpack",
        "//ynnpack/base",
        "//ynnpack/base:log",
        "//ynnpack/kernels/dot",
        "//ynnpack/subgraph",
        "@slinky//slinky/builder",
        "@slinky//slinky/runtime",
    ],
)

# This library is a drop-in replacement for XNNPACK, using ynnpack as the implementation.
cc_library(
    name = "xnnpack",
    srcs = [
        "deprecated.cc",
        "operator.cc",
        "runtime.cc",
        "subgraph.cc",
        "tensor.cc",
        "weights_cache.cc",
        "workspace.cc",
        "xnnpack.h",
    ],
    compatible_with = _COMPATIBLE_WITH,
    defines = ["XNNPACK_USING_YNNPACK=1"],
    visibility = ["//visibility:public"],
    deps = [
        ":dynamic_quantization",
        ":utils",
        "//:xnnpack_h",
        "//ynnpack",
        "//ynnpack/base",
        "//ynnpack/base:log",
        "//ynnpack/subgraph",
        "@pthreadpool",
        "@slinky//slinky/base:thread_pool",
    ],
)

cc_test(
    name = "dynamic_quantization_test",
    srcs = ["dynamic_quantization_test.cc"],
    deps = [
        ":dynamic_quantization",
        "//ynnpack:ynnpack_h",
        "//ynnpack/base",
        "//ynnpack/base/test:fuzz_test",
        "//ynnpack/base/test:random",
        "//ynnpack/base/test:tensor",
        "//ynnpack/base/test:util",
        "//ynnpack/subgraph/test:subgraph_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

config_setting(
    name = "xnnpack_use_ynnpack_explicit_true",
    define_values = {"xnnpack_use_ynnpack": "true"},
)

alias(
    name = "use_ynnpack",
    actual = ":xnnpack_use_ynnpack_explicit_true",
)
