# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//ynnpack:build_defs.bzl", "define_build_option")

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

define_build_option(
    name = "ynn_enable_perfetto",
    default_all = [],
)

cc_library(
    name = "perfetto",
    srcs = ["perfetto.cc"],
    hdrs = ["perfetto.h"],
    compatible_with = _COMPATIBLE_WITH,
    defines = ["YNN_ENABLE_PERFETTO"],
    tags = ["manual"],
    deps = [
        # The capability to capture perfetto traces is disabled by default. To enable and use it:add:
        #
        # git_repository(name = "perfetto", branch = "main", remote = "https://github.com/google/perfetto")
        #
        # To MODULE.bazel, add `--define ynn_enable_perfetto=true` to the bazel command line, and
        # uncomment this dependency.
        # "@perfetto//:libperfetto_client_experimental"
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
        "threadpool.cc",
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
        ":slinky_thread_pool",
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
    ] + select({
        ":ynn_enable_perfetto": [":perfetto"],
        "//conditions:default": [],
    }),
)
