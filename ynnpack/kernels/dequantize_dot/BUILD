# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//ynnpack:build_defs.bzl", "ynn_benchmark_deps", "ynn_binary_linkopts", "ynn_binary_malloc", "ynn_cc_library", "ynn_kernel_copts", "ynn_test_deps")

_COMPATIBLE_WITH = []

ynn_cc_library(
    name = "dequantize_dot",
    srcs = [
        "dequantize_dot.cc",
    ],
    hdrs = [
        "dequantize_dot.h",
        "generic.h",
        "kernels.inc",
    ],
    compatible_with = _COMPATIBLE_WITH,
    copts = ynn_kernel_copts(),
    per_arch_srcs = {
        "arm_neon": ["arm_neon.cc"],
        "arm_neonfp16": ["arm_neonfp16.cc"],
        "arm_neonbf16": ["arm_neonbf16.cc"],
        "wasm_simd128": ["wasm_simd128.cc"],
        "x86_sse41": ["x86_sse41.cc"],
        "x86_avx": ["x86_avx.cc"],
        "x86_avx2": ["x86_avx2.cc"],
        "x86_avx512": ["x86_avx512.cc"],
        "x86_avx512bf16": ["x86_avx512bf16.cc"],
        "x86_f16c": ["x86_f16c.cc"],
    },
    visibility = ["//ynnpack:__subpackages__"],
    deps = [
        "//ynnpack:ynnpack_h",
        "//ynnpack/base",
        "//ynnpack/base:log",
        "//ynnpack/base/simd",  # buildcleaner: keep
    ],
)

cc_test(
    name = "test",
    srcs = ["test.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":dequantize_dot",
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
        ":dequantize_dot",
        "//ynnpack/base",
        "//ynnpack/base/test:tensor",
    ] + ynn_benchmark_deps(),
)
