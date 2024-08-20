# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load(
    "//:build_defs.bzl",
    "xnnpack_cc_library",
    "xnnpack_std_cxxopts",
    "xnnpack_visibility",
)

xnnpack_cc_library(
    name = "fp16_mobilenet_v1",
    srcs = ["fp16-mobilenet-v1.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "@FP16",
        "//:XNNPACK",
        "//:cache",
        "//:common",
        "//:models_h",
    ],
)

xnnpack_cc_library(
    name = "fp32_mobilenet_v1",
    srcs = ["fp32-mobilenet-v1.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK",
        "//:cache",
        "//:common",
        "//:models_h",
    ],
)

xnnpack_cc_library(
    name = "fp16_sparse_mobilenet_v1",
    srcs = ["fp16-sparse-mobilenet-v1.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "@FP16",
        "//:XNNPACK",
        "//:cache",
        "//:common",
        "//:models_h",
    ],
)

xnnpack_cc_library(
    name = "fp32_sparse_mobilenet_v1",
    srcs = ["fp32-sparse-mobilenet-v1.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK",
        "//:cache",
        "//:common",
        "//:models_h",
    ],
)

xnnpack_cc_library(
    name = "qs8_qc8w_mobilenet_v1",
    srcs = ["qs8-qc8w-mobilenet-v1.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK",
        "//:cache",
        "//:common",
        "//:models_h",
    ],
)

xnnpack_cc_library(
    name = "qs8_qc8w_mobilenet_v2",
    srcs = ["qs8-qc8w-mobilenet-v2.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK",
        "//:cache",
        "//:common",
        "//:models_h",
    ],
)

xnnpack_cc_library(
    name = "qs8_mobilenet_v1",
    srcs = ["qs8-mobilenet-v1.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK",
        "//:cache",
        "//:common",
        "//:models_h",
    ],
)

xnnpack_cc_library(
    name = "qs8_mobilenet_v2",
    srcs = ["qs8-mobilenet-v2.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK",
        "//:cache",
        "//:common",
        "//:models_h",
    ],
)

xnnpack_cc_library(
    name = "qu8_mobilenet_v1",
    srcs = ["qu8-mobilenet-v1.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK",
        "//:cache",
        "//:common",
        "//:models_h",
    ],
)

xnnpack_cc_library(
    name = "qu8_mobilenet_v2",
    srcs = ["qu8-mobilenet-v2.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK",
        "//:cache",
        "//:common",
        "//:models_h",
    ],
)

xnnpack_cc_library(
    name = "qu8_mobilenet_v3_large",
    srcs = ["qu8-mobilenet-v3-large.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK",
        "//:cache",
        "//:common",
        "//:models_h",
    ],
)

xnnpack_cc_library(
    name = "qu8_mobilenet_v3_small",
    srcs = ["qu8-mobilenet-v3-small.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK",
        "//:cache",
        "//:common",
        "//:models_h",
    ],
)

xnnpack_cc_library(
    name = "fp16_mobilenet_v2",
    srcs = ["fp16-mobilenet-v2.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "@FP16",
        "//:XNNPACK",
        "//:cache",
        "//:common",
        "//:models_h",
    ],
)

xnnpack_cc_library(
    name = "fp32_mobilenet_v2",
    srcs = ["fp32-mobilenet-v2.cc"],
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK",
        "//:cache",
        "//:common",
        "//:models_h",
    ],
)

xnnpack_cc_library(
    name = "fp16_sparse_mobilenet_v2",
    srcs = ["fp16-sparse-mobilenet-v2.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "@FP16",
        "//:XNNPACK",
        "//:cache",
        "//:common",
        "//:models_h",
    ],
)

xnnpack_cc_library(
    name = "fp32_sparse_mobilenet_v2",
    srcs = ["fp32-sparse-mobilenet-v2.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK",
        "//:cache",
        "//:common",
        "//:models_h",
    ],
)

xnnpack_cc_library(
    name = "fp16_mobilenet_v3_large",
    srcs = ["fp16-mobilenet-v3-large.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "@FP16",
        "//:XNNPACK",
        "//:cache",
        "//:common",
        "//:models_h",
    ],
)

xnnpack_cc_library(
    name = "fp32_mobilenet_v3_large",
    srcs = ["fp32-mobilenet-v3-large.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK",
        "//:cache",
        "//:common",
        "//:models_h",
    ],
)

xnnpack_cc_library(
    name = "fp16_sparse_mobilenet_v3_large",
    srcs = ["fp16-sparse-mobilenet-v3-large.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "@FP16",
        "//:XNNPACK",
        "//:cache",
        "//:common",
        "//:models_h",
    ],
)

xnnpack_cc_library(
    name = "fp32_sparse_mobilenet_v3_large",
    srcs = ["fp32-sparse-mobilenet-v3-large.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK",
        "//:cache",
        "//:common",
        "//:models_h",
    ],
)

xnnpack_cc_library(
    name = "fp16_mobilenet_v3_small",
    srcs = ["fp16-mobilenet-v3-small.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "@FP16",
        "//:XNNPACK",
        "//:cache",
        "//:common",
        "//:models_h",
    ],
)

xnnpack_cc_library(
    name = "fp32_mobilenet_v3_small",
    srcs = ["fp32-mobilenet-v3-small.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK",
        "//:cache",
        "//:common",
        "//:models_h",
    ],
)

xnnpack_cc_library(
    name = "fp16_sparse_mobilenet_v3_small",
    srcs = ["fp16-sparse-mobilenet-v3-small.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "@FP16",
        "//:XNNPACK",
        "//:cache",
        "//:common",
        "//:models_h",
    ],
)

xnnpack_cc_library(
    name = "fp32_sparse_mobilenet_v3_small",
    srcs = ["fp32-sparse-mobilenet-v3-small.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK",
        "//:cache",
        "//:common",
        "//:models_h",
    ],
)
