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

[xnnpack_cc_library(
    name = "fp16_mobilenet_v1%s" % mode,
    srcs = ["fp16-mobilenet-v1.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "@FP16",
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "fp32_mobilenet_v1%s" % mode,
    srcs = ["fp32-mobilenet-v1.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "fp32_mobilenet_v1_jit%s" % mode,
    srcs = ["fp32-mobilenet-v1-jit.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "fp16_sparse_mobilenet_v1%s" % mode,
    srcs = ["fp16-sparse-mobilenet-v1.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "fp32_sparse_mobilenet_v1%s" % mode,
    srcs = ["fp32-sparse-mobilenet-v1.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "qs8_qc8w_mobilenet_v1%s" % mode,
    srcs = ["qs8-qc8w-mobilenet-v1.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "qs8_qc8w_mobilenet_v2%s" % mode,
    srcs = ["qs8-qc8w-mobilenet-v2.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "qs8_mobilenet_v1%s" % mode,
    srcs = ["qs8-mobilenet-v1.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "qs8_mobilenet_v2%s" % mode,
    srcs = ["qs8-mobilenet-v2.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "qu8_mobilenet_v1%s" % mode,
    srcs = ["qu8-mobilenet-v1.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "qu8_mobilenet_v2%s" % mode,
    srcs = ["qu8-mobilenet-v2.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "qu8_mobilenet_v3_large%s" % mode,
    srcs = ["qu8-mobilenet-v3-large.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "qu8_mobilenet_v3_small%s" % mode,
    srcs = ["qu8-mobilenet-v3-small.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "fp16_mobilenet_v2%s" % mode,
    srcs = ["fp16-mobilenet-v2.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "@FP16",
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "fp32_mobilenet_v2%s" % mode,
    srcs = ["fp32-mobilenet-v2.cc"],
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "fp32_mobilenet_v2_jit%s" % mode,
    srcs = ["fp32-mobilenet-v2-jit.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "fp16_sparse_mobilenet_v2%s" % mode,
    srcs = ["fp16-sparse-mobilenet-v2.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "fp32_sparse_mobilenet_v2%s" % mode,
    srcs = ["fp32-sparse-mobilenet-v2.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "fp16_mobilenet_v3_large%s" % mode,
    srcs = ["fp16-mobilenet-v3-large.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "@FP16",
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "fp32_mobilenet_v3_large%s" % mode,
    srcs = ["fp32-mobilenet-v3-large.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "fp32_mobilenet_v3_large_jit%s" % mode,
    srcs = ["fp32-mobilenet-v3-large-jit.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "fp16_sparse_mobilenet_v3_large%s" % mode,
    srcs = ["fp16-sparse-mobilenet-v3-large.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "fp32_sparse_mobilenet_v3_large%s" % mode,
    srcs = ["fp32-sparse-mobilenet-v3-large.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "fp16_mobilenet_v3_small%s" % mode,
    srcs = ["fp16-mobilenet-v3-small.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "@FP16",
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "fp32_mobilenet_v3_small%s" % mode,
    srcs = ["fp32-mobilenet-v3-small.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "fp32_mobilenet_v3_small_fused%s" % mode,
    srcs = ["fp32-mobilenet-v3-small-fused.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "//:post_operation",
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "fp32_mobilenet_v3_small_jit%s" % mode,
    srcs = ["fp32-mobilenet-v3-small-jit.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "fp16_sparse_mobilenet_v3_small%s" % mode,
    srcs = ["fp16-sparse-mobilenet-v3-small.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]

[xnnpack_cc_library(
    name = "fp32_sparse_mobilenet_v3_small%s" % mode,
    srcs = ["fp32-sparse-mobilenet-v3-small.cc"],
    copts = xnnpack_std_cxxopts(),
    visibility = xnnpack_visibility(),
    deps = [
        "//:XNNPACK%s" % mode,
        "//:aligned_allocator",
        "//:cache",
        "//:models_h%s" % mode,
        "@pthreadpool",
    ],
) for mode in [
    "",
    "_test_mode",
]]
