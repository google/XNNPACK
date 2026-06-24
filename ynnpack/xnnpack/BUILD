# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_library.bzl", "cc_library")

_COMPATIBLE_WITH = []

# This library is a drop-in replacement for XNNPACK, using ynnpack as the implementation.
cc_library(
    name = "xnnpack",
    srcs = [
        "deprecated.cc",
        "dynamic_quantization.cc",
        "dynamic_quantization.h",
        "experimental.cc",
        "operator.cc",
        "runtime.cc",
        "subgraph.cc",
        "tensor.cc",
        "utils.cc",
        "utils.h",
        "weights_cache.cc",
        "workspace.cc",
        "xnnpack.h",
    ],
    compatible_with = _COMPATIBLE_WITH,
    defines = ["XNNPACK_USING_YNNPACK=1"],
    visibility = ["//visibility:public"],
    deps = [
        "//:xnnpack_h",
        "//ynnpack",
        "//ynnpack/base",
        "//ynnpack/base:log",
        "//ynnpack/composites",
        "//ynnpack/kernels/dot",
        "//ynnpack/subgraph",
        "@pthreadpool",
        "@slinky//slinky/base:thread_pool",
        "@slinky//slinky/base:thread_pool_impl",
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
