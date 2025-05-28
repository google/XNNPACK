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

xnnpack_cxx_library(
    name = "stencil",
    hdrs = ["stencil.h"],
    deps = [
        "//:buffer",
        "//:math",
    ],
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
    srcs = ["subgraph-tester.cc"],
    hdrs = ["subgraph-tester.h"],
    deps = xnnpack_test_deps_for_library() + [
        ":runtime_flags",
        "//:XNNPACK",
        "//:buffer",
        "//:datatype",
        "//:math",
        "//:subgraph_h",
        "//:xnnpack_h",
        "//test:replicable_random_device",
        "@pthreadpool",
    ],
)

xnnpack_cxx_library(
    name = "runtime_tester",
    testonly = True,
    hdrs = [
        "runtime-tester.h",
    ],
    deps = xnnpack_test_deps_for_library() + [
        ":runtime_flags",
        ":subgraph_tester",
        "//:subgraph_h",
        "//:xnnpack_h",
    ],
)

xnnpack_unit_test(
    name = "unary_test",
    srcs = ["unary.cc"],
    deps = SUBGRAPH_TEST_DEPS + ["//test:unary_ops"],
)

[xnnpack_unit_test(
    name = "%s_test" % operator,
    srcs = [
        "%s.cc" % operator.replace("_", "-"),
    ],
    deps = SUBGRAPH_TEST_DEPS,
) for operator in [
    "copy",
    "softmax",
    "space_to_depth_2d",
    "depth_to_space_2d",
    "static_constant_pad",
    "static_expand_dims",
    "static_reshape",
    "static_slice",
]]

xnnpack_unit_test(
    name = "workspace_test",
    srcs = ["workspace.cc"],
    deps = SUBGRAPH_TEST_DEPS + ["//:allocation_type"],
)

xnnpack_unit_test(
    name = "persistent_test",
    srcs = ["persistent.cc"],
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
    timeout = "moderate",
    srcs = ["argmax-pooling-2d.cc"],
    deps = SUBGRAPH_TEST_DEPS + [":stencil"],
)

xnnpack_unit_test(
    name = "average_pooling_2d_test",
    timeout = "moderate",
    srcs = ["average-pooling-2d.cc"],
    deps = SUBGRAPH_TEST_DEPS + [":stencil"],
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
    srcs = ["convolution-2d.cc"],
    deps = SUBGRAPH_TEST_DEPS + [":stencil"],
)

xnnpack_unit_test(
    name = "deconvolution_2d_test",
    srcs = ["deconvolution-2d.cc"],
    deps = SUBGRAPH_TEST_DEPS + [":stencil"],
)

xnnpack_unit_test(
    name = "depthwise_convolution_2d_test",
    srcs = ["depthwise-convolution-2d.cc"],
    deps = SUBGRAPH_TEST_DEPS + [":stencil"],
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
    timeout = "moderate",
    srcs = ["max-pooling-2d.cc"],
    deps = SUBGRAPH_TEST_DEPS + [":stencil"],
)

xnnpack_unit_test(
    name = "rope_test",
    timeout = "moderate",
    srcs = ["rope.cc"],
    deps = SUBGRAPH_TEST_DEPS,
)

xnnpack_unit_test(
    name = "static_reduce_test",
    srcs = ["static-reduce.cc"],
    deps = SUBGRAPH_TEST_DEPS,
)

xnnpack_unit_test(
    name = "static_transpose_test",
    timeout = "moderate",
    srcs = ["static-transpose.cc"],
    shard_count = 5,
    deps = SUBGRAPH_TEST_DEPS,
)

xnnpack_unit_test(
    name = "split_fuse_test",
    srcs = ["split-fuse.cc"],
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
    deps = SUBGRAPH_TEST_DEPS + [":stencil"],
)

xnnpack_unit_test(
    name = "fusion_test",
    srcs = [
        "fusion.cc",
    ],
    deps = [
        ":runtime_tester",
        ":subgraph_tester",
        "//:buffer",
        "//:node_type",
        "//:subgraph_h",
        "//:xnnpack_h",
    ],
)

############################### Misc unit tests ###############################

xnnpack_unit_test(
    name = "runtime_test",
    srcs = [
        "runtime.cc",
    ],
    deps = [
        ":runtime_tester",
    ],
)

xnnpack_unit_test(
    name = "subgraph_test",
    srcs = [
        "subgraph.cc",
    ],
    deps = [
        ":runtime_tester",
        ":subgraph_tester",
        "//:subgraph_h",
    ],
)

xnnpack_unit_test(
    name = "memory_planner_test",
    srcs = [
        "memory-planner.cc",
    ],
    deps = [
        ":runtime_flags",
        ":runtime_tester",
        ":subgraph_tester",
        "//:node_type",
        "//:subgraph",
        "//:subgraph_h",
        "//:xnnpack_h",
    ],
)

xnnpack_unit_test(
    name = "subgraph_nchw_test",
    srcs = ["subgraph-nchw.cc"],
    deps = [
        ":subgraph_tester",
        "//:node_type",
        "//:subgraph_h",
    ],
)

xnnpack_unit_test(
    name = "subgraph_fp16_test",
    srcs = [
        "mock-allocator.h",
        "subgraph-fp16.cc",
    ],
    deps = [
        ":runtime_flags",
        ":runtime_tester",
        ":subgraph_tester",
        "//:allocation_type",
        "//:allocator",
        "//:buffer",
        "//:math",
        "//:node_type",
        "//:params",
        "//:subgraph_h",
        "//:xnnpack_h",
        "//test:replicable_random_device",
    ],
)

xnnpack_unit_test(
    name = "stencil_test",
    srcs = ["stencil.cc"],
    deps = [
        ":stencil",
        "//:buffer",
        "//:operator_utils",
        "//test:replicable_random_device",
    ],
)
