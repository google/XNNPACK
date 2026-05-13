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
    tags = [
        "manual",
        "notap",
    ],
    deps = ["//ynnpack/kernels/elementwise:generator"],
)

_COMPATIBLE_WITH = []

ARCHS = [
    "x86_sse2",
    "x86_avx",
    "x86_avx2",
    "x86_avx512",
    "x86_avx512bf16",
    "arm64_neon",
    "arm_neon",
    "arm_neonbf16",
    "wasm_simd128",
]

[ynn_generate_src_hdr(
    name = arch,
    compatible_with = _COMPATIBLE_WITH,
    generator = ":generator",
    generator_args = [arch],
    output_hdr = arch + ".inc",
    output_src = arch + ".cc",
) for arch in ARCHS]

ynn_cc_library(
    name = "binary",
    srcs = ["binary.cc"],
    hdrs = [
        "binary.h",
        "kernels.inc",
    ] + [arch + ".inc" for arch in ARCHS],
    compatible_with = _COMPATIBLE_WITH,
    copts = ynn_kernel_copts(),
    per_arch_srcs = {arch: [arch + ".cc"] for arch in ARCHS},
    visibility = ["//ynnpack:__subpackages__"],
    deps = [
        "//ynnpack:ynnpack_h",
        "//ynnpack/base",
        "//ynnpack/base:log",
        "//ynnpack/base/simd",
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
        "//ynnpack/base/test:util",
        "@com_google_googletest//:gtest",  # no g3
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
