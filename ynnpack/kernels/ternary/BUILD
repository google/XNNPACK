# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_python//python:py_binary.bzl", "py_binary")
load("//ynnpack:build_defs.bzl", "ynn_benchmark_deps", "ynn_binary_linkopts", "ynn_binary_malloc", "ynn_cc_library", "ynn_generate_src_hdr", "ynn_kernel_copts", "ynn_test_deps")

_COMPATIBLE_WITH = []

py_binary(
    name = "generator",
    srcs = [
        "convert.py",
        "generator.py",
        "kernels.py",
    ],
    tags = ["manual"],
    deps = ["//ynnpack/kernels/elementwise:generator"],
)

ynn_generate_src_hdr(
    name = "x86_sse2",
    compatible_with = _COMPATIBLE_WITH,
    generator = ":generator",
    generator_args = [
        "x86_sse2",
        "quantize_fp32_to_int8,16x1",
        "multiply_fp32_fp32_fp32,8x1",
        "multiply_int32_fp32_fp32,8x1",
        "multiply_add_fp32_fp32_fp32,8x1",
        "clamp_fp32_fp32_fp32,8x1",
    ],
    output_hdr = "x86_sse2.inc",
    output_src = "x86_sse2.cc",
)

ynn_generate_src_hdr(
    name = "x86_sse41",
    compatible_with = _COMPATIBLE_WITH,
    generator = ":generator",
    generator_args = [
        "x86_sse41",
        "subtract_multiply_int32_int32_int32,8x1",
        "quantize_fp32_to_uint8,16x1",
    ],
    output_hdr = "x86_sse41.inc",
    output_src = "x86_sse41.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx",
    compatible_with = _COMPATIBLE_WITH,
    generator = ":generator",
    generator_args = [
        "x86_avx",
        "multiply_fp32_fp32_fp32,16x1",
        "multiply_add_fp32_fp32_fp32,16x1",
        "clamp_fp32_fp32_fp32,16x1",
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
        "quantize_fp32_to_int8,32x1",
        "quantize_fp32_to_uint8,32x1",
        "multiply_int32_fp32_fp32,16x1",
        "subtract_multiply_int32_int32_int32,16x1",
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
        "multiply_fp32_fp32_fp32,32x1",
        "multiply_int32_fp32_fp32,32x1",
        "subtract_multiply_int32_int32_int32,32x1",
        "multiply_add_fp32_fp32_fp32,32x1",
        "clamp_fp32_fp32_fp32,32x1",
    ],
    output_hdr = "x86_avx512f.inc",
    output_src = "x86_avx512f.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx512bw",
    compatible_with = _COMPATIBLE_WITH,
    generator = ":generator",
    generator_args = [
        "x86_avx512bw",
        "quantize_fp32_to_int8,64x1",
        "quantize_fp32_to_uint8,64x1",
    ],
    output_hdr = "x86_avx512bw.inc",
    output_src = "x86_avx512bw.cc",
)

ynn_generate_src_hdr(
    name = "arm_neon",
    compatible_with = _COMPATIBLE_WITH,
    generator = ":generator",
    generator_args = [
        "arm_neon",
        "quantize_fp32_to_int8,16x1",
        "multiply_fp32_fp32_fp32,8x1",
        "multiply_int32_fp32_fp32,8x1",
        "subtract_multiply_int32_int32_int32,8x1",
        # TODO(vksnk): add neon_fma library for this function.
        "multiply_add_fp32_fp32_fp32,8x1",
        "clamp_fp32_fp32_fp32,8x1",
    ],
    output_hdr = "arm_neon.inc",
    output_src = "arm_neon.cc",
)

ynn_cc_library(
    name = "ternary",
    srcs = ["ternary.cc"],
    hdrs = [
        "arm_neon.inc",
        "kernels.inc",
        "ternary.h",
        "x86_avx.inc",
        "x86_avx2.inc",
        "x86_avx512bw.inc",
        "x86_avx512f.inc",
        "x86_sse2.inc",
        "x86_sse41.inc",
    ],
    compatible_with = _COMPATIBLE_WITH,
    copts = ynn_kernel_copts(),
    per_arch_srcs = {
        "arm_neon": ["arm_neon.cc"],
        "x86_avx": ["x86_avx.cc"],
        "x86_avx512": [
            "x86_avx512bw.cc",
            "x86_avx512f.cc",
        ],
        "x86_avx2": ["x86_avx2.cc"],
        "x86_sse2": ["x86_sse2.cc"],
        "x86_sse41": ["x86_sse41.cc"],
    },
    visibility = ["//ynnpack:__subpackages__"],
    deps = [
        "//ynnpack:ynnpack_h",
        "//ynnpack/base",
    ],
)

cc_library(
    name = "reference",
    testonly = True,
    srcs = ["reference.cc"],
    hdrs = ["reference.h"],
    visibility = ["//visibility:private"],
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
        ":reference",
        ":ternary",
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
        ":ternary",
        "//ynnpack/base",
        "//ynnpack/base/test:tensor",
    ] + ynn_benchmark_deps(),
)
