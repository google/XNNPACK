# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//ynnpack:build_defs.bzl", "ynn_benchmark_deps", "ynn_binary_linkopts", "ynn_binary_malloc", "ynn_cc_library")

package(default_visibility = ["//ynnpack:__subpackages__"])

ynn_cc_library(
    name = "bench_lib",
    per_arch_srcs = {
        "arm_neon": ["arm_neon.cc"],
        "x86_sse2": ["x86_sse2.cc"],
        "x86_avx": ["x86_avx.cc"],
        "x86_avx512f": ["x86_avx512f.cc"],
        "x86_avx512bw": ["x86_avx512bw.cc"],
        "x86_fma3": ["x86_fma3.cc"],
    },
    textual_hdrs = ["generic.h"],
    deps = [
        "//ynnpack/base",  # buildcleaner:keep
        "//ynnpack/base/simd",  # buildcleaner:keep
        "@com_google_benchmark//:benchmark",
    ],
)

cc_test(
    name = "bench",
    args = ["--benchmark_min_time=1x"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [":bench_lib"] + ynn_benchmark_deps(),
)
