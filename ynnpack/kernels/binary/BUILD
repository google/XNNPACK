# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_python//python:py_binary.bzl", "py_binary")
load("//ynnpack:build_defs.bzl", "ynn_benchmark_deps", "ynn_binary_linkopts", "ynn_binary_malloc", "ynn_cc_library", "ynn_generate_src_hdr", "ynn_kernel_copts", "ynn_test_deps")

py_binary(
    name = "generator",
    srcs = [
        "generator.py",
        "kernels.py",
    ],
    tags = ["manual"],
    deps = ["//ynnpack/kernels/elementwise:generator"],
)

_COMPATIBLE_WITH = []

ynn_generate_src_hdr(
    name = "x86_sse2",
    compatible_with = _COMPATIBLE_WITH,
    generator = ":generator",
    generator_args = [
        "x86_sse2",
        "add_fp32,8x1",
        "subtract_fp32,8x1",
        "multiply_fp32,8x1",
        "multiply_int32_fp32,8x1",
        "divide_fp32,8x1",
        "copysign_fp32,8x1",
        "max_fp32,8x1",
        "min_fp32,8x1",
    ],
    output_hdr = "x86_sse2.inc",
    output_src = "x86_sse2.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx",
    compatible_with = _COMPATIBLE_WITH,
    generator = ":generator",
    generator_args = [
        "x86_avx",
        "add_fp32,16x1",
        "subtract_fp32,16x1",
        "multiply_fp32,16x1",
        "divide_fp32,16x1",
        "copysign_fp32,16x1",
        "max_fp32,16x1",
        "min_fp32,16x1",
    ],
    output_hdr = "x86_avx.inc",
    output_src = "x86_avx.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx2",
    compatible_with = _COMPATIBLE_WITH,
    generator = ":generator",
    generator_args = [
        "x86_avx2",
        "multiply_int32_fp32,16x1",
        "subtract_fp32_bf16,16x1",
    ],
    output_hdr = "x86_avx2.inc",
    output_src = "x86_avx2.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx512f",
    compatible_with = _COMPATIBLE_WITH,
    generator = ":generator",
    generator_args = [
        "x86_avx512f",
        "add_fp32,32x1",
        "subtract_fp32,32x1",
        "multiply_fp32,32x1",
        "multiply_int32_fp32,32x1",
        "divide_fp32,32x1",
        # Requires avc512dq in the current implementation.
        # "copysign_fp32,32x1",
        "max_fp32,32x1",
        "min_fp32,32x1",
    ],
    output_hdr = "x86_avx512f.inc",
    output_src = "x86_avx512f.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx512bf16",
    compatible_with = _COMPATIBLE_WITH,
    generator = ":generator",
    generator_args = [
        "x86_avx512bf16",
        "subtract_fp32_bf16,32x1",
    ],
    output_hdr = "x86_avx512bf16.inc",
    output_src = "x86_avx512bf16.cc",
)

ynn_generate_src_hdr(
    name = "arm_neon",
    compatible_with = _COMPATIBLE_WITH,
    generator = ":generator",
    generator_args = [
        "arm_neon",
        "add_fp32,8x1",
        "subtract_fp32,8x1",
        "multiply_fp32,8x1",
        "multiply_int32_fp32,8x1",
        "divide_fp32,8x1",
        "copysign_fp32,8x1",
        "max_fp32,8x1",
        "min_fp32,8x1",
    ],
    output_hdr = "arm_neon.inc",
    output_src = "arm_neon.cc",
)

ynn_cc_library(
    name = "binary",
    srcs = ["binary.cc"],
    hdrs = [
        "arm_neon.inc",
        "binary.h",
        "kernels.inc",
        "x86_avx.inc",
        "x86_avx2.inc",
        "x86_avx512bf16.inc",
        "x86_avx512f.inc",
        "x86_sse2.inc",
    ],
    compatible_with = _COMPATIBLE_WITH,
    copts = ynn_kernel_copts(),
    per_arch_srcs = {
        "arm_neon": ["arm_neon.cc"],
        "x86_sse2": ["x86_sse2.cc"],
        "x86_avx": ["x86_avx.cc"],
        "x86_avx512f": ["x86_avx512f.cc"],
        "x86_avx2": ["x86_avx2.cc"],
        "x86_avx512bf16": ["x86_avx512bf16.cc"],
    },
    visibility = ["//ynnpack:__subpackages__"],
    deps = [
        "//ynnpack:ynnpack_h",
        "//ynnpack/base",
        "//ynnpack/base:log",
    ],
)

cc_library(
    name = "reference",
    testonly = True,
    srcs = ["reference.cc"],
    hdrs = ["reference.h"],
    visibility = ["//ynnpack:__subpackages__"],
    deps = [
        "//ynnpack:ynnpack_h",
        "//ynnpack/base",
        "//ynnpack/base/test:tensor",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "test",
    srcs = ["test.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":binary",
        ":reference",
        "//ynnpack:ynnpack_h",
        "//ynnpack/base",
        "//ynnpack/base/test:fuzz_test",
        "//ynnpack/base/test:random",
        "//ynnpack/base/test:tensor",
        "//ynnpack/base/test:util",
    ] + ynn_test_deps(),
)

cc_test(
    name = "bench",
    srcs = ["bench.cc"],
    args = ["--benchmark_min_time=1x"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":binary",
        "//ynnpack/base",
        "//ynnpack/base/test:tensor",
    ] + ynn_benchmark_deps(),
)
