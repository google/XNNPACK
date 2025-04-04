# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load(
    "//:build_defs.bzl",
    "xnnpack_cxx_library",
    "xnnpack_unit_test",
)

OPERATOR_TEST_DEPS = [
    "@pthreadpool",
    "//:XNNPACK",
    "//:buffer",
    "//:aligned_allocator",
    "//:allocator",
    "//:cache",
    "//:common",
    "//:datatype",
    "//:fp16",
    "//:internal",
    "//:math",
    "//:microkernel_configs",
    "//:microparams",
    "//:normalization",
    "//:params",
    "//:quantization",
    "//test:replicable_random_device",
]

xnnpack_cxx_library(
    name = "convolution_test_helpers",
    testonly = True,
    srcs = [
        "convolution-test-helpers.cc",
    ],
    hdrs = [
        "convolution-test-helpers.h",
    ],
    # TODO: This shouldn't be required outside of here.
    visibility = ["//:__subpackages__"],
    deps = [
        "//:buffer",
        "//:microparams",
        "//:quantization",
    ],
)

xnnpack_cxx_library(
    name = "operator_test_utils",
    testonly = True,
    hdrs = ["operator-test-utils.h"],
    # TODO: This shouldn't be required outside of here.
    visibility = ["//:__subpackages__"],
    deps = ["//test:replicable_random_device"],
)

xnnpack_unit_test(
    name = "operator_utils_test",
    srcs = ["operator-utils.cc"],
    deps = [
        "//:microkernel_configs",
        "//:operator_utils",
    ],
)

xnnpack_unit_test(
    name = "unary_elementwise_nc_test",
    timeout = "long",
    srcs = ["unary-elementwise-nc.cc"],
    deps = OPERATOR_TEST_DEPS + [
        "//:operator_h",
        "//:operator_utils",
        "//test:unary_ops",
    ],
)

xnnpack_unit_test(
    name = "binary_elementwise_nd_test",
    timeout = "long",
    srcs = ["binary-elementwise-nd.cc"],
    shard_count = 10,
    deps = OPERATOR_TEST_DEPS + [
        ":operator_test_utils",
        "//:operator_utils",
        "//:reference_ukernels",
    ],
)

xnnpack_unit_test(
    name = "argmax_pooling_nhwc_test",
    srcs = [
        "argmax-pooling-nhwc.cc",
        "argmax-pooling-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "average_pooling_nhwc_test",
    srcs = [
        "average-pooling-nhwc.cc",
        "average-pooling-operator-tester.h",
    ],
    shard_count = 5,
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "batch_matrix_multiply_nc_test",
    timeout = "moderate",
    srcs = [
        "batch-matrix-multiply-nc.cc",
        "batch-matrix-multiply-operator-tester.h",
    ],
    shard_count = 5,
    deps = OPERATOR_TEST_DEPS + ["//:microkernels_h"],
)

xnnpack_unit_test(
    name = "constant_pad_nd_test",
    srcs = [
        "constant-pad-nd.cc",
        "constant-pad-operator-tester.h",
    ],
    shard_count = 5,
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "constant_pad_nd_eager_test",
    srcs = [
        "constant-pad-nd-eager.cc",
        "constant-pad-operator-tester.h",
    ],
    shard_count = 5,
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "convert_nc_test",
    srcs = [
        "convert-nc.cc",
    ],
    deps = OPERATOR_TEST_DEPS + ["//:microkernels_h"],
)

xnnpack_unit_test(
    name = "convolution_nhwc_test",
    timeout = "moderate",
    srcs = [
        "convolution-nhwc.cc",
        "convolution-operator-tester.h",
    ],
    shard_count = 10,
    deps = OPERATOR_TEST_DEPS + [
        ":convolution_test_helpers",
    ],
)

xnnpack_unit_test(
    name = "convolution_nchw_test",
    timeout = "moderate",
    srcs = [
        "convolution-nchw.cc",
        "convolution-operator-tester.h",
    ],
    shard_count = 10,
    deps = OPERATOR_TEST_DEPS + [
        ":convolution_test_helpers",
    ],
)

xnnpack_unit_test(
    name = "copy_nc_test",
    srcs = [
        "copy-nc.cc",
        "copy-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "copy_nc_eager_test",
    srcs = [
        "copy-nc-eager.cc",
        "copy-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "deconvolution_nhwc_test",
    timeout = "moderate",
    srcs = [
        "deconvolution-nhwc.cc",
        "deconvolution-nhwc-qd8-f32-qc8w.cc",
        "deconvolution-operator-tester.h",
    ],
    shard_count = 10,
    deps = OPERATOR_TEST_DEPS + [
        ":convolution_test_helpers",
    ],
)

xnnpack_unit_test(
    name = "depth_to_space_nchw2nhwc_test",
    srcs = [
        "depth-to-space-nchw2nhwc.cc",
        "depth-to-space-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "depth_to_space_nhwc_test",
    srcs = [
        "depth-to-space-nhwc.cc",
        "depth-to-space-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "dynamic_fully_connected_nc_test",
    srcs = [
        "dynamic-fully-connected-nc.cc",
        "dynamic-fully-connected-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "fully_connected_nc_test",
    srcs = [
        "fully-connected-nc.cc",
        "fully-connected-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS + [
        "//:microkernels_h",
    ],
)

xnnpack_unit_test(
    name = "max_pooling_nhwc_test",
    timeout = "moderate",
    srcs = [
        "max-pooling-nhwc.cc",
        "max-pooling-operator-tester.h",
    ],
    shard_count = 10,
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "reduce_nd_test",
    timeout = "moderate",
    srcs = [
        "reduce-nd.cc",
    ],
    deps = OPERATOR_TEST_DEPS + [
        "//:requantization",
    ],
)

xnnpack_unit_test(
    name = "slice_normalization_test",
    srcs = [
        "slice-normalization.cc",
        "slice-normalization-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "reduce_normalization_test",
    srcs = [
        "reduce-normalization.cc",
        "reduce-normalization-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "transpose_normalization_test",
    srcs = [
        "transpose-normalization.cc",
        "transpose-normalization-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "resize_bilinear_nhwc_test",
    srcs = [
        "resize-bilinear-nhwc.cc",
        "resize-bilinear-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "resize_bilinear_nchw_test",
    srcs = [
        "resize-bilinear-nchw.cc",
        "resize-bilinear-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "rope_nthc_test",
    srcs = [
        "rope-nthc.cc",
        "rope-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "scaled_dot_product_attention_nhtc_test",
    timeout = "moderate",
    srcs = [
        "scaled-dot-product-attention-nhtc.cc",
        "scaled-dot-product-attention-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "slice_nd_test",
    timeout = "moderate",
    srcs = [
        "slice-nd.cc",
        "slice-operator-tester.h",
    ],
    shard_count = 5,
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "slice_nd_eager_test",
    timeout = "moderate",
    srcs = [
        "slice-nd-eager.cc",
        "slice-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "softmax_nc_test",
    srcs = [
        "softmax-nc.cc",
        "softmax-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "space_to_depth_nhwc_test",
    srcs = [
        "space-to-depth-nhwc.cc",
        "space-to-depth-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "transpose_nd_test",
    srcs = [
        "transpose-nd.cc",
        "transpose-operator-tester.h",
    ],
    shard_count = 10,
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "transpose_nd_eager_test",
    srcs = [
        "transpose-nd-eager.cc",
        "transpose-operator-tester.h",
    ],
    shard_count = 5,
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "unpooling_nhwc_test",
    srcs = [
        "unpooling-nhwc.cc",
        "unpooling-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)
