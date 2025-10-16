# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_library.bzl", "cc_library")

_COMPATIBLE_WITH = []

cc_library(
    name = "slinky_thread_pool",
    srcs = ["slinky_thread_pool.cc"],
    hdrs = ["slinky_thread_pool.h"],
    compatible_with = _COMPATIBLE_WITH,
    visibility = ["//ynnpack:__subpackages__"],
    deps = [
        "//ynnpack:ynnpack_h",
        "@slinky//slinky/base",
        "@slinky//slinky/base:thread_pool",
        "@slinky//slinky/base:thread_pool_impl",
    ],
)

cc_library(
    name = "threadpool",
    srcs = ["threadpool.cc"],
    compatible_with = _COMPATIBLE_WITH,
    visibility = ["//ynnpack:__subpackages__"],
    deps = [
        ":slinky_thread_pool",
        "//ynnpack:ynnpack_h",
        "//ynnpack/base:log",
        "@slinky//slinky/base:thread_pool",
    ],
)

cc_library(
    name = "subgraph",
    srcs = [
        "broadcast.cc",
        "broadcast_like.cc",
        "concatenate.cc",
        "copy.cc",
        "dot.cc",
        "elementwise.cc",
        "elementwise.h",
        "even_split.cc",
        "fusion.cc",
        "get_tensor_shape.cc",
        "reduce.cc",
        "runtime.cc",
        "slinky.cc",
        "stack.cc",
        "static_pad.cc",
        "static_slice.cc",
        "static_transpose.cc",
        "stencil_copy.cc",
        "subgraph.cc",
        "tensor.cc",
        "utils.cc",
        "utils.h",
    ],
    hdrs = [
        "runtime.h",
        "slinky.h",
        "subgraph.h",
        "tensor.h",
    ],
    compatible_with = _COMPATIBLE_WITH,
    visibility = ["//ynnpack:__subpackages__"],
    deps = [
        ":threadpool",
        "//ynnpack:ynnpack_h",
        "//ynnpack/base",
        "//ynnpack/base:log",
        "//ynnpack/kernels/binary",
        "//ynnpack/kernels/dot",
        "//ynnpack/kernels/dot:schedule",
        "//ynnpack/kernels/reduce",
        "//ynnpack/kernels/ternary",
        "//ynnpack/kernels/transpose",
        "//ynnpack/kernels/unary",
        "@slinky//slinky/base",
        "@slinky//slinky/base:thread_pool",
        "@slinky//slinky/builder",
        "@slinky//slinky/runtime",
    ],
)
