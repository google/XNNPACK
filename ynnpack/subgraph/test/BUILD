# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

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
    deps = [
        ":scheduler",
        "//ynnpack/subgraph:slinky_thread_pool",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "fold_constants",
    srcs = ["fold_constants.cc"],
    deps = [
        ":subgraph_builder",
        "//ynnpack:ynnpack_h",
        "//ynnpack/base",
        "//ynnpack/subgraph",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "invalidate_dead_values",
    srcs = ["invalidate_dead_values.cc"],
    deps = [
        ":subgraph_builder",
        "//ynnpack:ynnpack_h",
        "//ynnpack/base",
        "//ynnpack/subgraph",
        "@com_google_googletest//:gtest_main",
    ],
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
    "@com_google_googletest//:gtest_main",
]

cc_test(
    name = "binary",
    srcs = ["binary.cc"],
    deps = TEST_DEPS + [
        "//ynnpack/kernels/binary:reference",
    ],
)

cc_test(
    name = "unary",
    srcs = ["unary.cc"],
    deps = TEST_DEPS + [
        "//ynnpack/kernels/unary:reference",
    ],
)

cc_test(
    name = "broadcast_like",
    srcs = ["broadcast_like.cc"],
    deps = TEST_DEPS,
)

cc_test(
    name = "copy",
    srcs = ["copy.cc"],
    deps = TEST_DEPS,
)

cc_test(
    name = "static_broadcast",
    srcs = ["static_broadcast.cc"],
    deps = TEST_DEPS,
)

cc_test(
    name = "static_expand_dims",
    srcs = ["static_expand_dims.cc"],
    deps = TEST_DEPS,
)

cc_test(
    name = "static_reshape",
    srcs = ["static_reshape.cc"],
    deps = TEST_DEPS,
)

cc_test(
    name = "split_dim",
    srcs = ["split_dim.cc"],
    deps = TEST_DEPS,
)

cc_test(
    name = "fuse_dim",
    srcs = ["fuse_dim.cc"],
    deps = TEST_DEPS,
)

cc_test(
    name = "fuse_dims",
    srcs = ["fuse_dims.cc"],
    deps = TEST_DEPS,
)

cc_test(
    name = "concatenate",
    srcs = ["concatenate.cc"],
    deps = TEST_DEPS,
)

cc_test(
    name = "dot",
    srcs = ["dot.cc"],
    shard_count = 12,
    deps = TEST_DEPS,
)

cc_test(
    name = "errors",
    srcs = ["errors.cc"],
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
    deps = TEST_DEPS + ["//ynnpack/subgraph"],
)

cc_test(
    name = "get_tensor_shape",
    srcs = ["get_tensor_shape.cc"],
    deps = TEST_DEPS,
)

cc_test(
    name = "reduce",
    srcs = ["reduce.cc"],
    deps = TEST_DEPS,
)

cc_test(
    name = "even_split",
    srcs = ["even_split.cc"],
    deps = TEST_DEPS,
)

cc_test(
    name = "stack",
    srcs = ["stack.cc"],
    deps = TEST_DEPS,
)

cc_test(
    name = "static_transpose",
    srcs = ["static_transpose.cc"],
    deps = TEST_DEPS,
)

cc_test(
    name = "static_slice",
    srcs = ["static_slice.cc"],
    deps = TEST_DEPS,
)

cc_test(
    name = "static_pad",
    srcs = ["static_pad.cc"],
    deps = TEST_DEPS,
)

cc_test(
    name = "stencil_copy",
    srcs = ["stencil_copy.cc"],
    deps = TEST_DEPS,
)
