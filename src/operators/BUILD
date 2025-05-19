# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load(
    "//:build_defs.bzl",
    "xnnpack_cc_library",
    "xnnpack_min_size_copts",
)

package(default_visibility = ["//:__subpackages__"])

OPERATOR_COMMON_DEPS = [
    ":operator_delete",
    ":operator_h",
    ":operator_run",
    ":operator_type",
    ":operator_utils",
    "//src/configs:config_types",
    "//src/configs:microkernel_configs",
]

xnnpack_cc_library(
    name = "operator_h",
    hdrs = [
        "//:src/xnnpack/operator.h",
    ],
    deps = [
        ":compute_h",
        ":operator_type",
        "//:common",
        "//:xnnpack_h",
        "//src/configs:config_types",
    ],
)

xnnpack_cc_library(
    name = "compute_h",
    hdrs = [
        "//:src/xnnpack/compute.h",
    ],
    deps = [
        ":operator_type",
        "//:common",
        "//:xnnpack_h",
    ],
)

xnnpack_cc_library(
    name = "operator_type",
    hdrs = [
        "//:src/xnnpack/operator-type.h",
    ],
    textual_hdrs = [
        "//:src/xnnpack/operator-type-defs.inc",
    ],
    deps = [
        "//:common",
    ],
)

xnnpack_cc_library(
    name = "operator_utils",
    srcs = ["operator-utils.c"],
    hdrs = ["//:src/xnnpack/operator-utils.h"],
    deps = [
        ":operator_h",
        ":operator_type",
        "//:allocator",
        "//:common",
        "//:logging",
        "//:math",
        "//:microfnptr_h",
        "//:node_type",
        "//:params",
        "//:xnnpack_h",
    ],
)

xnnpack_cc_library(
    name = "average_pooling_nhwc",
    srcs = ["average-pooling-nhwc.c"],
    deps = OPERATOR_COMMON_DEPS + [
        "//:allocator",
        "//:common",
        "//:indirection",
        "//:logging",
        "//:math",
        "//:microkernel_type",
        "//:params",
        "//:xnnpack_h",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "dynamic_fully_connected_nc",
    srcs = ["dynamic-fully-connected-nc.c"],
    deps = OPERATOR_COMMON_DEPS + [
        "//:allocator",
        "//:common",
        "//:logging",
        "//:math",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
        "//:microkernel_type",
        "//:microkernel_utils",
        "//:microparams_h",
        "//:params",
        "//:xnnpack_h",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "rope_nthc",
    srcs = ["rope-nthc.c"],
    deps = OPERATOR_COMMON_DEPS + [
        "//:allocator",
        "//:common",
        "//:logging",
        "//:params",
        "//:xnnpack_h",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "deconvolution_nhwc",
    srcs = ["deconvolution-nhwc.c"],
    deps = OPERATOR_COMMON_DEPS + [
        "//:allocator",
        "//:cache",
        "//:common",
        "//:indirection",
        "//:logging",
        "//:math",
        "//:microfnptr_h",
        "//:microkernel_type",
        "//:microkernel_utils",
        "//:microparams_h",
        "//:microparams_init",
        "//:packing",
        "//:params",
        "//:xnnpack_h",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "unpooling_nhwc",
    srcs = ["unpooling-nhwc.c"],
    deps = OPERATOR_COMMON_DEPS + [
        "//:allocator",
        "//:common",
        "//:indirection",
        "//:logging",
        "//:math",
        "//:params",
        "//:xnnpack_h",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "resize_bilinear_nhwc",
    srcs = ["resize-bilinear-nhwc.c"],
    deps = OPERATOR_COMMON_DEPS + [
        "//:allocator",
        "//:common",
        "//:indirection",
        "//:logging",
        "//:math",
        "//:microfnptr_h",
        "//:params",
        "//:xnnpack_h",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "convolution_nhwc",
    srcs = ["convolution-nhwc.c"],
    deps = OPERATOR_COMMON_DEPS + [
        "//:allocator",
        "//:cache",
        "//:common",
        "//:indirection",
        "//:logging",
        "//:math",
        "//:microfnptr_h",
        "//:microkernel_type",
        "//:microkernel_utils",
        "//:microparams_h",
        "//:microparams_init",
        "//:packing",
        "//:params",
        "//:xnnpack_h",
        "//src/configs:hardware_config",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "pack_lh",
    srcs = ["pack-lh.c"],
    deps = OPERATOR_COMMON_DEPS + [
        "//:allocator",
        "//:logging",
        "//:math",
        "//:microfnptr_h",
        "//:params",
        "//:xnnpack_h",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "constant_pad_nd",
    srcs = ["constant-pad-nd.c"],
    deps = OPERATOR_COMMON_DEPS + [
        "//:allocator",
        "//:common",
        "//:logging",
        "//:params",
        "//:xnnpack_h",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "unary_elementwise_nc",
    srcs = ["unary-elementwise-nc.c"],
    deps = OPERATOR_COMMON_DEPS + [
        "//:allocator",
        "//:common",
        "//:datatype",
        "//:internal",
        "//:logging",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
        "//:microparams_h",
        "//:params",
        "//:reference_ukernels",
        "//:xnnpack_h",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "argmax_pooling_nhwc",
    srcs = ["argmax-pooling-nhwc.c"],
    deps = OPERATOR_COMMON_DEPS + [
        "//:allocator",
        "//:common",
        "//:indirection",
        "//:logging",
        "//:math",
        "//:params",
        "//:xnnpack_h",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "transpose_nd",
    srcs = ["transpose-nd.c"],
    deps = OPERATOR_COMMON_DEPS + [
        "//:allocator",
        "//:common",
        "//:logging",
        "//:math",
        "//:microkernel_type",
        "//:normalization",
        "//:params",
        "//:xnnpack_h",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "max_pooling_nhwc",
    srcs = ["max-pooling-nhwc.c"],
    deps = OPERATOR_COMMON_DEPS + [
        "//:allocator",
        "//:common",
        "//:indirection",
        "//:logging",
        "//:math",
        "//:microparams_h",
        "//:params",
        "//:xnnpack_h",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "binary_elementwise_nd",
    srcs = ["binary-elementwise-nd.c"],
    deps = OPERATOR_COMMON_DEPS + [
        "//:allocator",
        "//:datatype",
        "//:logging",
        "//:math",
        "//:microparams_h",
        "//:params",
        "//:reference_ukernels",
        "//:xnnpack_h",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "reduce_nd",
    srcs = ["reduce-nd.c"],
    deps = OPERATOR_COMMON_DEPS + [
        "//:allocator",
        "//:common",
        "//:datatype",
        "//:logging",
        "//:microkernel_type",
        "//:microparams_h",
        "//:normalization",
        "//:params",
        "//:reference_ukernels",
        "//:xnnpack_h",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "resize_bilinear_nchw",
    srcs = ["resize-bilinear-nchw.c"],
    deps = OPERATOR_COMMON_DEPS + [
        "//:allocator",
        "//:common",
        "//:logging",
        "//:math",
        "//:params",
        "//:xnnpack_h",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "batch_matrix_multiply_nc",
    srcs = ["batch-matrix-multiply-nc.c"],
    deps = OPERATOR_COMMON_DEPS + [
        "//:allocator",
        "//:cache",
        "//:common",
        "//:logging",
        "//:math",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
        "//:microkernel_type",
        "//:microkernel_utils",
        "//:microparams_h",
        "//:microparams_init",
        "//:packing",
        "//:params",
        "//:xnnpack_h",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "softmax_nc",
    srcs = ["softmax-nc.c"],
    deps = OPERATOR_COMMON_DEPS + [
        "//:allocator",
        "//:common",
        "//:logging",
        "//:math",
        "//:microfnptr_h",
        "//:microparams_h",
        "//:params",
        "//:xnnpack_h",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "fully_connected_nc",
    srcs = ["fully-connected-nc.c"],
    deps = OPERATOR_COMMON_DEPS + [
        "//:allocator",
        "//:cache",
        "//:common",
        "//:logging",
        "//:math",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
        "//:microkernel_type",
        "//:microkernel_utils",
        "//:microparams_h",
        "//:microparams_init",
        "//:packing",
        "//:params",
        "//:xnnpack_h",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "slice_nd",
    srcs = ["slice-nd.c"],
    deps = OPERATOR_COMMON_DEPS + [
        "//:allocator",
        "//:common",
        "//:logging",
        "//:normalization",
        "//:params",
        "//:xnnpack_h",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "convolution_nchw",
    srcs = ["convolution-nchw.c"],
    deps = OPERATOR_COMMON_DEPS + [
        "//:allocator",
        "//:cache",
        "//:common",
        "//:logging",
        "//:math",
        "//:microfnptr_h",
        "//:microkernel_type",
        "//:packing",
        "//:params",
        "//:xnnpack_h",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "operator_run",
    srcs = ["operator-run.c"],
    hdrs = [
        "//:src/xnnpack/compute.h",
    ],
    deps = [
        ":operator_h",
        ":operator_type",
        ":operator_utils",
        "//:common",
        "//:indirection",
        "//:logging",
        "//:math",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
        "//:microkernel_type",
        "//:microparams_h",
        "//:microparams_init",
        "//:quantization",
        "//:xnnpack_h",
        "//src/configs:config_types",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "operator_delete",
    srcs = ["operator-delete.c"],
    deps = [
        ":operator_h",
        ":operator_utils",
        "//:allocator",
        "//:logging",
        "//:params",
        "//:xnnpack_h",
    ],
)

xnnpack_cc_library(
    name = "operators",
    hdrs = [
        "//:src/xnnpack/operator.h",
    ],
    copts = select({
        "//:debug_build": [],
        "//conditions:default": xnnpack_min_size_copts(),
    }) + select({
        "//:xnn_enable_hmp_explicit_false": ["-DXNN_MAX_UARCH_TYPES=1"],
        "//conditions:default": [],
    }),
    deps = [
        ":argmax_pooling_nhwc",
        ":average_pooling_nhwc",
        ":batch_matrix_multiply_nc",
        ":binary_elementwise_nd",
        ":constant_pad_nd",
        ":convolution_nchw",
        ":convolution_nhwc",
        ":deconvolution_nhwc",
        ":dynamic_fully_connected_nc",
        ":fully_connected_nc",
        ":max_pooling_nhwc",
        ":pack_lh",
        ":reduce_nd",
        ":resize_bilinear_nchw",
        ":resize_bilinear_nhwc",
        ":rope_nthc",
        ":slice_nd",
        ":softmax_nc",
        ":transpose_nd",
        ":unary_elementwise_nc",
        ":unpooling_nhwc",
        "//:common",
        "//:microfnptr_h",
        "//:microkernel_type",
        "//:microparams_h",
        "//:xnnpack_h",
        "//src/configs:hardware_config",
        "@pthreadpool",
    ],
)
