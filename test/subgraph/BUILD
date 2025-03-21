# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load(
    "//:build_defs.bzl",
    "xnnpack_binary",
    "xnnpack_cxx_library",
    "xnnpack_test_deps_for_library",
    "xnnpack_unit_test",
)

SUBGRAPH_TEST_DEPS = [
    ":runtime_flags",
    ":subgraph_tester",
    "//:XNNPACK",
    "//:buffer",
    "//:common",
    "//:datatype",
    "//:math",
    "//:node_type",
    "//:operators",
    "//:operator_utils",
    "//:reference_ukernels",
    "//:requantization",
    "//:subgraph",
    "//test:replicable_random_device",
]

############################## Testing utilities ###############################

xnnpack_cxx_library(
    name = "runtime_flags",
    testonly = True,
    srcs = ["runtime-flags.cc"],
    hdrs = ["runtime-flags.h"],
    deps = xnnpack_test_deps_for_library(),
)

########################## Size tests for the library #########################

xnnpack_binary(
    name = "subgraph_size_test",
    srcs = ["subgraph-size.c"],
    deps = ["//:XNNPACK"],
)

########################### Unit tests for subgraph ###########################

xnnpack_cxx_library(
    name = "subgraph_tester",
    testonly = True,
    hdrs = [
        "subgraph-tester.h",
    ],
    deps = xnnpack_test_deps_for_library() + [
        ":runtime_flags",
        "//:buffer",
        "//:datatype",
        "//:subgraph_h",
        "//:xnnpack_h",
        "//test:replicable_random_device",
    ],
)

xnnpack_unit_test(
    name = "unary_test",
    srcs = ["unary.cc"],
    deps = SUBGRAPH_TEST_DEPS + ["//test:unary_ops"],
)

[xnnpack_unit_test(
    name = "%s_test" % operator,
    timeout = timeout,
    srcs = [
        "%s.cc" % operator.replace("_", "-"),
    ],
    shard_count = shard_count,
    deps = SUBGRAPH_TEST_DEPS,
) for operator, shard_count, timeout in [
    ("copy", 1, "short"),
    ("softmax", 1, "moderate"),
    ("space_to_depth_2d", 1, "short"),
    ("depth_to_space_2d", 1, "short"),
    ("static_constant_pad", 1, "moderate"),
    ("static_expand_dims", 1, "short"),
    ("static_reshape", 1, "short"),
    ("static_slice", 5, "short"),
    ("static_transpose", 5, "short"),
]]

xnnpack_unit_test(
    name = "workspace_test",
    srcs = ["workspace.cc"],
    deps = SUBGRAPH_TEST_DEPS + ["//:allocation_type"],
)

xnnpack_unit_test(
    name = "binary_test",
    timeout = "moderate",
    srcs = ["binary.cc"],
    deps = SUBGRAPH_TEST_DEPS,
)

xnnpack_unit_test(
    name = "argmax_pooling_2d_test",
    srcs = ["argmax-pooling-2d.cc"],
    deps = SUBGRAPH_TEST_DEPS,
)

xnnpack_unit_test(
    name = "average_pooling_2d_test",
    srcs = ["average-pooling-2d.cc"],
    deps = SUBGRAPH_TEST_DEPS,
)

xnnpack_unit_test(
    name = "average_pooling_2d_reshape_test",
    srcs = ["average-pooling-2d-reshape.cc"],
    deps = [
        ":runtime_flags",
        "//:XNNPACK",
        "//:node_type",
        "//:subgraph",
    ],
)

xnnpack_unit_test(
    name = "batch_matrix_multiply_test",
    srcs = ["batch-matrix-multiply.cc"],
    deps = SUBGRAPH_TEST_DEPS,
)

xnnpack_unit_test(
    name = "concatenate_test",
    srcs = ["concatenate.cc"],
    deps = SUBGRAPH_TEST_DEPS,
)

xnnpack_unit_test(
    name = "convolution_2d_test",
    srcs = [
        "convolution-2d.cc",
    ],
    deps = SUBGRAPH_TEST_DEPS + [
        "//test/operators:convolution_test_helpers",
    ],
)

xnnpack_unit_test(
    name = "deconvolution_2d_test",
    timeout = "moderate",
    srcs = ["deconvolution-2d.cc"],
    shard_count = 5,
    deps = SUBGRAPH_TEST_DEPS + [
        "//test/operators:convolution_test_helpers",
    ],
)

xnnpack_unit_test(
    name = "depthwise_convolution_2d_test",
    srcs = [
        "depthwise-convolution-2d.cc",
    ],
    deps = SUBGRAPH_TEST_DEPS + [
        "//test/operators:convolution_test_helpers",
    ],
)

xnnpack_unit_test(
    name = "even_split_test",
    srcs = ["even-split.cc"],
    deps = SUBGRAPH_TEST_DEPS,
)

xnnpack_unit_test(
    name = "fully_connected_test",
    timeout = "moderate",
    srcs = ["fully-connected.cc"],
    deps = SUBGRAPH_TEST_DEPS,
)

xnnpack_unit_test(
    name = "max_pooling_2d_test",
    srcs = ["max-pooling-2d.cc"],
    deps = SUBGRAPH_TEST_DEPS,
)

xnnpack_unit_test(
    name = "rope_test",
    timeout = "moderate",
    srcs = ["rope.cc"],
    deps = SUBGRAPH_TEST_DEPS,
)

xnnpack_unit_test(
    name = "scaled_dot_product_attention_test",
    srcs = ["scaled-dot-product-attention.cc"],
    deps = SUBGRAPH_TEST_DEPS + [
        "//:aligned_allocator",
    ],
)

xnnpack_unit_test(
    name = "static_reduce_test",
    srcs = ["static-reduce.cc"],
    deps = SUBGRAPH_TEST_DEPS,
)

xnnpack_unit_test(
    name = "static_resize_bilinear_2d_test",
    srcs = ["static-resize-bilinear-2d.cc"],
    deps = SUBGRAPH_TEST_DEPS,
)

xnnpack_unit_test(
    name = "unpooling_2d_test",
    srcs = ["unpooling-2d.cc"],
    deps = SUBGRAPH_TEST_DEPS,
)

xnnpack_unit_test(
    name = "fusion_test",
    srcs = [
        "fusion.cc",
        "runtime-tester.h",
        "subgraph-tester.h",
    ],
    deps = [
        ":runtime_flags",
        "//:XNNPACK",
        "//:buffer",
        "//:datatype",
        "//:node_type",
        "//:subgraph",
        "//test:replicable_random_device",
    ],
)

############################### Misc unit tests ###############################

xnnpack_unit_test(
    name = "runtime_test",
    srcs = [
        "runtime.cc",
        "runtime-tester.h",
        "subgraph-tester.h",
    ],
    deps = [
        ":runtime_flags",
        "//:XNNPACK",
        "//:buffer",
        "//:datatype",
        "//:subgraph",
        "//test:replicable_random_device",
    ],
)

xnnpack_unit_test(
    name = "subgraph_test",
    srcs = [
        "runtime-tester.h",
        "subgraph.cc",
        "subgraph-tester.h",
    ],
    deps = [
        ":runtime_flags",
        "//:XNNPACK",
        "//:buffer",
        "//:datatype",
        "//:subgraph",
        "//test:replicable_random_device",
    ],
)

xnnpack_unit_test(
    name = "memory_planner_test",
    srcs = [
        "memory-planner.cc",
        "runtime-tester.h",
        "subgraph-tester.h",
    ],
    deps = [
        ":runtime_flags",
        "//:XNNPACK",
        "//:buffer",
        "//:datatype",
        "//:node_type",
        "//:subgraph",
        "//test:replicable_random_device",
    ],
)

xnnpack_unit_test(
    name = "subgraph_nchw_test",
    srcs = ["subgraph-nchw.cc"],
    deps = [
        ":subgraph_tester",
        "//:XNNPACK",
        "//:node_type",
        "//:subgraph",
        "//test:replicable_random_device",
    ],
)

xnnpack_unit_test(
    name = "subgraph_fp16_test",
    srcs = [
        "mock-allocator.h",
        "runtime-tester.h",
        "subgraph-fp16.cc",
        "subgraph-tester.h",
    ],
    deps = SUBGRAPH_TEST_DEPS + [
        "//:allocation_type",
        "//:allocator",
        "//:params",
    ],
)
