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

cc_library(
    name = "matchers",
    testonly = True,
    hdrs = ["matchers.h"],
    deps = [
        "//ynnpack:ynnpack_h",
        "//ynnpack/subgraph",
        "@com_google_googletest//:gtest",
        "@slinky//slinky/base",
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
        ":matchers",
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
        ":matchers",
        ":subgraph_builder",
        "//ynnpack:ynnpack_h",
        "//ynnpack/base",
        "//ynnpack/subgraph",
    ] + ynn_test_deps(),
)

TEST_DEPS = [
    ":matchers",
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
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
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

cc_test(
    name = "eliminate_common_subgraphs",
    srcs = ["eliminate_common_subgraphs.cc"],
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
    "concatenate",
    "copy",
    "broadcast_like",
    "errors",
    "even_split",
    "fuse_dim",
    "fuse_dims",
    "get_tensor_shape",
    "lut",
    "reduce",
    "runtime",
    "split_dim",
    "stack",
    "static_broadcast",
    "static_expand_dims",
    "static_pad",
    "static_reshape",
    "static_slice",
    "static_transpose",
    "stencil_copy",
]]

cc_test(
    name = "reduce_window",
    srcs = ["reduce_window.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = TEST_DEPS + ["//ynnpack/subgraph"],
)

cc_test(
    name = "utils",
    srcs = ["utils.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = TEST_DEPS + ["//ynnpack/subgraph"],
)

cc_test(
    name = "dot_bench",
    srcs = ["dot_bench.cc"],
    args = ["--benchmark_min_time=1x"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":scheduler",
        "//ynnpack",
        "//ynnpack/base",
        "@com_google_benchmark//:benchmark",
    ],
)
