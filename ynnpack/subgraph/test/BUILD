# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//ynnpack:build_defs.bzl", "ynn_binary_linkopts", "ynn_binary_malloc", "ynn_test_deps")

cc_library(
    name = "scheduler",
    testonly = True,
    srcs = ["scheduler.cc"],
    hdrs = ["scheduler.h"],
    visibility = ["//ynnpack/subgraph:__subpackages__"],
    deps = [
        "//ynnpack:ynnpack_h",
        "@slinky//slinky/base:thread_pool_impl",
    ],
)

cc_library(
    name = "subgraph_builder",
    testonly = True,
    srcs = ["subgraph_builder.cc"],
    hdrs = ["subgraph_builder.h"],
    visibility = ["//ynnpack:__subpackages__"],
    deps = [
        ":scheduler",
        "//ynnpack",
        "//ynnpack/base",
        "//ynnpack/base/test:tensor",
    ],
)

cc_test(
    name = "slinky_thread_pool",
    srcs = ["slinky_thread_pool.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":scheduler",
        "//ynnpack/subgraph:slinky_thread_pool",
    ] + ynn_test_deps(),
)

cc_test(
    name = "fold_constants",
    srcs = ["fold_constants.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":subgraph_builder",
        "//ynnpack:ynnpack_h",
        "//ynnpack/base",
        "//ynnpack/subgraph",
    ] + ynn_test_deps(),
)

cc_test(
    name = "invalidate_dead_values",
    srcs = ["invalidate_dead_values.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":subgraph_builder",
        "//ynnpack:ynnpack_h",
        "//ynnpack/base",
        "//ynnpack/subgraph",
    ] + ynn_test_deps(),
)

TEST_DEPS = [
    ":subgraph_builder",
    ":scheduler",
    "//ynnpack",
    "//ynnpack/base",
    "//ynnpack/base/test:buffer",
    "//ynnpack/base/test:fuzz_test",
    "//ynnpack/base/test:random",
    "//ynnpack/base/test:tensor",
    "//ynnpack/base/test:util",
] + ynn_test_deps()

cc_test(
    name = "binary",
    srcs = ["binary.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = TEST_DEPS + [
        "//ynnpack/kernels/binary:reference",
    ],
)

cc_test(
    name = "unary",
    srcs = ["unary.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = TEST_DEPS + [
        "//ynnpack/kernels/unary:reference",
    ],
)

cc_test(
    name = "dot",
    srcs = ["dot.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    shard_count = 18,
    deps = TEST_DEPS,
)

cc_test(
    name = "fusion",
    srcs = select({
        # This test assumes we can pattern match and use ternary kernels, which are only guaranteed
        # to exist on x86, and the logic being tested is architecture independent.
        "//ynnpack:x86": ["fusion.cc"],
        "//conditions:default": [],
    }),
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = TEST_DEPS + [
        "//ynnpack/kernels/ternary",
        "//ynnpack/subgraph",
    ],
)

cc_test(
    name = "fusion_lut",
    srcs = ["fusion_lut.cc"],
    deps = TEST_DEPS + [
        "//ynnpack/subgraph",
    ],
)

cc_test(
    name = "subgraphs",
    srcs = ["subgraphs.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = TEST_DEPS + ["//ynnpack/subgraph"],
)

[cc_test(
    name = name,
    srcs = [name + ".cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = TEST_DEPS,
) for name in [
    "copy",
    "static_broadcast",
    "static_expand_dims",
    "static_reshape",
    "split_dim",
    "fuse_dim",
    "fuse_dims",
    "concatenate",
    "errors",
    "get_tensor_shape",
    "lut",
    "reduce",
    "even_split",
    "runtime",
    "stack",
    "static_transpose",
    "static_slice",
    "static_pad",
    "stencil_copy",
]]

cc_test(
    name = "utils",
    srcs = ["utils.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = TEST_DEPS + ["//ynnpack/subgraph"],
)
