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
        "exp.py",
        "generator.py",
        "kernels.py",
        "sigmoid.py",
        "sine_cosine.py",
        "tanh.py",
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
        "abs_fp32,8x1",
        "erf_fp32,8x1",
        "exp_fp32,8x1",
        "negate_fp32,8x1",
        "reciprocal_square_root_fp32,8x1",
        "square_root_fp32,8x1",
        "sigmoid_fp32,8x1",
        "tanh_fp32,8x1",
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
        "ceil_fp32,8x1",
        "cosine_fp32,8x1",
        "floor_fp32,8x1",
        "round_fp32,8x1",
        "sine_fp32,8x1",
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
        "abs_fp32,16x1",
        "ceil_fp32,16x1",
        "cosine_fp32,16x1",
        "erf_fp32,16x1",
        "floor_fp32,16x1",
        "negate_fp32,16x1",
        "reciprocal_square_root_fp32,16x1",
        "round_fp32,16x1",
        "sine_fp32,16x1",
        "square_root_fp32,16x1",
        "tanh_fp32,16x1",
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
        "exp_fp32,16x1",
        "sigmoid_fp32,16x1",
    ],
    output_hdr = "x86_avx2.inc",
    output_src = "x86_avx2.cc",
)

ynn_generate_src_hdr(
    name = "x86_fma3",
    compatible_with = _COMPATIBLE_WITH,
    generator = ":generator",
    generator_args = [
        "x86_fma3",
        "cosine_fp32,16x1",
        "erf_fp32,16x1",
        "reciprocal_square_root_fp32,16x1",
        "sine_fp32,16x1",
    ],
    output_hdr = "x86_fma3.inc",
    output_src = "x86_fma3.cc",
)

ynn_generate_src_hdr(
    name = "x86_f16c",
    compatible_with = _COMPATIBLE_WITH,
    generator = ":generator",
    generator_args = [
        "x86_f16c",
        "convert_fp16_to_fp32,16x1",
        "convert_fp32_to_fp16,16x1",
    ],
    output_hdr = "x86_f16c.inc",
    output_src = "x86_f16c.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx512f",
    compatible_with = _COMPATIBLE_WITH,
    generator = ":generator",
    generator_args = [
        "x86_avx512f",
        # Requires avc512dq in the current implementation.
        # "abs_fp32,32x1",
        "ceil_fp32,32x1",
        "cosine_fp32,32x1",
        "erf_fp32,32x1",
        "exp_fp32,32x1",
        "floor_fp32,32x1",
        "negate_fp32,32x1",
        "reciprocal_square_root_fp32,32x1",
        "round_fp32,32x1",
        "sine_fp32,32x1",
        "square_root_fp32,32x1",
        # Comparison masks are very different in pre-AVX512 vs AVX512,
        # so we need to have a more general approach to support both.
        # "sigmoid_fp32,32x1",
        "tanh_fp32,32x1",
    ],
    output_hdr = "x86_avx512f.inc",
    output_src = "x86_avx512f.cc",
)

ynn_generate_src_hdr(
    name = "arm_neon",
    compatible_with = _COMPATIBLE_WITH,
    generator = ":generator",
    generator_args = [
        "arm_neon",
        "abs_fp32,8x1",
        "ceil_fp32,8x1",
        "cosine_fp32,8x1",
        "erf_fp32,8x1",
        "exp_fp32,8x1",
        "floor_fp32,8x1",
        "negate_fp32,8x1",
        "reciprocal_square_root_fp32,8x1",
        "round_fp32,8x1",
        "sine_fp32,8x1",
        "square_root_fp32,8x1",
        "tanh_fp32,8x1",
    ],
    output_hdr = "arm_neon.inc",
    output_src = "arm_neon.cc",
)

ynn_generate_src_hdr(
    name = "arm_neonfp16",
    compatible_with = _COMPATIBLE_WITH,
    generator = ":generator",
    generator_args = [
        "arm_neonfp16",
        "convert_fp16_to_fp32,8x1",
        "convert_fp32_to_fp16,8x1",
    ],
    output_hdr = "arm_neonfp16.inc",
    output_src = "arm_neonfp16.cc",
)

ynn_cc_library(
    name = "unary",
    srcs = ["unary.cc"],
    hdrs = [
        "arm_neon.inc",
        "arm_neonfp16.inc",
        "kernels.inc",
        "unary.h",
        "x86_avx.inc",
        "x86_avx2.inc",
        "x86_avx512f.inc",
        "x86_fma3.inc",
        "x86_sse2.inc",
        "x86_sse41.inc",
    ],
    compatible_with = _COMPATIBLE_WITH,
    copts = ynn_kernel_copts(),
    per_arch_srcs = {
        "arm_neon": ["arm_neon.cc"],
        "arm_neonfp16": ["arm_neonfp16.cc"],
        "x86_avx": ["x86_avx.cc"],
        "x86_avx2": ["x86_avx2.cc"],
        "x86_avx512": ["x86_avx512f.cc"],
        "x86_f16c": ["x86_f16c"],
        "x86_fma3": ["x86_fma3.cc"],
        "x86_sse2": ["x86_sse2.cc"],
        "x86_sse41": ["x86_sse41.cc"],
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
        ":reference",
        ":unary",
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
        ":unary",
        "//ynnpack/base",
        "//ynnpack/base/test:tensor",
    ] + ynn_benchmark_deps(),
)
