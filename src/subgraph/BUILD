# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load(
    "//:build_defs.bzl",
    "xnnpack_cc_library",
)

package(default_visibility = ["//:__subpackages__"])

SUBGRAPH_COMMON_DEPS = [
    ":node_type",
    ":subgraph_h",
    ":subgraph_types",
    "//:common",
    "//:xnnpack_h",
    "//:logging",
    "//src/operators",
    "//src/operators:operator_type",
    "@pthreadpool",
]

xnnpack_cc_library(
    name = "subgraph_types",
    hdrs = [
        "//:src/xnnpack/subgraph_types.h",
    ],
    deps = [
        ":node_type",
        "//:cache",
        "//:common",
        "//:math",
        "//:xnnpack_h",
        "//src/configs:config_types",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "subgraph_h",
    hdrs = [
        "//:src/xnnpack/subgraph.h",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":node_type",
        ":subgraph_types",
        "//:cache",
        "//:common",
        "//:math",
        "//:xnnpack_h",
        "//src/configs:config_types",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "node_type",
    hdrs = [
        "//:src/xnnpack/node-type.h",
    ],
    textual_hdrs = [
        "//:src/xnnpack/node-type-defs.h",
    ],
    deps = [
        "//:common",
    ],
)

xnnpack_cc_library(
    name = "tensor",
    srcs = ["tensor.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        "//:allocation_type",
        "//:datatype",
        "//:math",
        "//:microkernel_hdrs",
        "//:params",
    ],
)

xnnpack_cc_library(
    name = "depthwise_convolution_2d",
    srcs = ["depthwise-convolution-2d.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":subgraph_validation",
        "//:internal",
        "//:requantization",
    ],
)

xnnpack_cc_library(
    name = "softmax",
    srcs = ["softmax.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":reshape_helpers",
        ":subgraph_validation",
    ],
)

xnnpack_cc_library(
    name = "static_transpose",
    srcs = ["static-transpose.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":subgraph_validation",
        "//:datatype",
    ],
)

xnnpack_cc_library(
    name = "static_resize_bilinear_2d",
    srcs = ["static-resize-bilinear-2d.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":subgraph_validation",
        "//:math",
    ],
)

xnnpack_cc_library(
    name = "static_constant_pad",
    srcs = ["static-constant-pad.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":subgraph_validation",
        "//:datatype",
        "//:fp16",
        "//:math",
        "//:requantization",
    ],
)

xnnpack_cc_library(
    name = "rope",
    srcs = ["rope.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":subgraph_validation",
    ],
)

xnnpack_cc_library(
    name = "fully_connected_sparse",
    srcs = ["fully-connected-sparse.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":subgraph_validation",
        "//:internal",
    ],
)

xnnpack_cc_library(
    name = "unary",
    srcs = ["unary.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":reshape_helpers",
        ":subgraph_validation",
        "//:internal",
        "//src/operators:operator_utils",
    ],
)

xnnpack_cc_library(
    name = "pack_lh",
    srcs = ["pack-lh.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":subgraph_validation",
        "//:internal",
    ],
)

xnnpack_cc_library(
    name = "depth_to_space_2d",
    srcs = ["depth-to-space-2d.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":subgraph_validation",
        "//:datatype",
    ],
)

xnnpack_cc_library(
    name = "subgraph_validation",
    srcs = ["validation.c"],
    hdrs = ["//:src/xnnpack/subgraph-validation.h"],
    deps = [
        ":node_type",
        ":subgraph_h",
        "//:logging",
        "//:params",
        "//:xnnpack_h",
    ],
)

xnnpack_cc_library(
    name = "convolution_2d",
    srcs = ["convolution-2d.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":subgraph_validation",
        "//:internal",
        "//:requantization",
        "//src/configs:config_types",
        "//src/configs:microkernel_configs",
    ],
)

xnnpack_cc_library(
    name = "unpooling_2d",
    srcs = ["unpooling-2d.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":subgraph_validation",
    ],
)

xnnpack_cc_library(
    name = "binary",
    srcs = ["binary.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":reshape_helpers",
        ":subgraph_validation",
        "//src/operators:operator_utils",
    ],
)

xnnpack_cc_library(
    name = "space_to_depth_2d",
    srcs = ["space-to-depth-2d.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":subgraph_validation",
        "//:datatype",
    ],
)

xnnpack_cc_library(
    name = "concatenate",
    srcs = ["concatenate.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":subgraph_validation",
        "//:datatype",
    ],
)

xnnpack_cc_library(
    name = "subgraph_utils",
    srcs = ["subgraph-utils.c"],
    hdrs = ["subgraph-utils.h"],
    deps = [
        ":node_type",
        ":subgraph_h",
        "//:logging",
        "//:xnnpack_h",
    ],
)

xnnpack_cc_library(
    name = "batch_matrix_multiply",
    srcs = ["batch-matrix-multiply.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":subgraph_validation",
        "//:allocation_type",
        "//:internal",
        "//:math",
        "//src/configs:config_types",
        "//src/configs:microkernel_configs",
    ],
)

xnnpack_cc_library(
    name = "deconvolution_2d",
    srcs = ["deconvolution-2d.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":subgraph_validation",
        "//:internal",
        "//:requantization",
    ],
)

xnnpack_cc_library(
    name = "argmax_pooling_2d",
    srcs = ["argmax-pooling-2d.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":subgraph_validation",
    ],
)

xnnpack_cc_library(
    name = "even_split",
    srcs = ["even-split.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":subgraph_validation",
        "//:allocation_type",
        "//:datatype",
    ],
)

xnnpack_cc_library(
    name = "copy",
    srcs = ["copy.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":reshape_helpers",
        ":subgraph_validation",
        "//:datatype",
    ],
)

xnnpack_cc_library(
    name = "reshape_helpers",
    srcs = ["reshape-helpers.c"],
    hdrs = ["//:src/xnnpack/reshape-helpers.h"],
    deps = [
        ":subgraph_h",
        "//:logging",
        "//:math",
        "//:xnnpack_h",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "max_pooling_2d",
    srcs = ["max-pooling-2d.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":subgraph_validation",
        "//:requantization",
    ],
)

xnnpack_cc_library(
    name = "average_pooling_2d",
    srcs = ["average-pooling-2d.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":subgraph_validation",
    ],
)

xnnpack_cc_library(
    name = "deprecated",
    srcs = ["deprecated.c"],
    deps = [
        ":binary",  # buildcleaner: keep
        ":concatenate",  # buildcleaner: keep
        ":subgraph_h",
        ":unary",  # buildcleaner: keep
        "//:xnnpack_h",
    ],
)

xnnpack_cc_library(
    name = "static_slice",
    srcs = ["static-slice.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":subgraph_validation",
        "//:datatype",
        "//:math",
    ],
)

xnnpack_cc_library(
    name = "static_reduce",
    srcs = ["static-reduce.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":subgraph_validation",
    ],
)

xnnpack_cc_library(
    name = "fully_connected",
    srcs = ["fully-connected.c"],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":convolution_2d",  # buildcleaner: keep
        ":subgraph_validation",
        "//:allocation_type",
        "//:internal",
        "//:requantization",
        "//src/configs:config_types",
        "//src/configs:microkernel_configs",
    ],
)

xnnpack_cc_library(
    name = "subgraph",
    srcs = ["subgraph.c"],
    hdrs = [
        "//:src/xnnpack/subgraph.h",
    ],
    deps = SUBGRAPH_COMMON_DEPS + [
        ":argmax_pooling_2d",  # buildcleaner: keep
        ":average_pooling_2d",  # buildcleaner: keep
        ":batch_matrix_multiply",  # buildcleaner: keep
        ":binary",  # buildcleaner: keep
        ":concatenate",  # buildcleaner: keep
        ":convolution_2d",  # buildcleaner: keep
        ":copy",  # buildcleaner: keep
        ":deconvolution_2d",  # buildcleaner: keep
        ":deprecated",  # buildcleaner: keep
        ":depth_to_space_2d",  # buildcleaner: keep
        ":depthwise_convolution_2d",  # buildcleaner: keep
        ":even_split",  # buildcleaner: keep
        ":fully_connected",  # buildcleaner: keep
        ":fully_connected_sparse",  # buildcleaner: keep
        ":max_pooling_2d",  # buildcleaner: keep
        ":pack_lh",  # buildcleaner: keep
        ":rope",  # buildcleaner: keep
        ":softmax",  # buildcleaner: keep
        ":space_to_depth_2d",  # buildcleaner: keep
        ":static_constant_pad",  # buildcleaner: keep
        ":static_reduce",  # buildcleaner: keep
        ":static_resize_bilinear_2d",  # buildcleaner: keep
        ":static_slice",  # buildcleaner: keep
        ":static_transpose",  # buildcleaner: keep
        ":subgraph_utils",
        ":subgraph_validation",
        ":tensor",  # buildcleaner: keep
        ":unary",  # buildcleaner: keep
        ":unpooling_2d",  # buildcleaner: keep
        "//:allocation_type",
        "//:allocator",
        "//:cache",
        "//:datatype",
        "//:fp16",
        "//:internal",
        "//:math",
        "//:params",
        "//src/configs:config_types",
        "//src/configs:hardware_config",
        "//src/configs:microkernel_configs",
        "//src/operators:operator_utils",
    ],
)
