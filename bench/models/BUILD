# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load(
    "//:build_defs.bzl",
    "xnnpack_benchmark",
    "xnnpack_cxx_library",
    "xnnpack_slow_benchmark_tags",
)

xnnpack_cxx_library(
    name = "models",
    srcs = [
        "fp32-attention.cc",
        "fp32-mobilenet-v1.cc",
        "fp32-mobilenet-v2.cc",
        "fp32-mobilenet-v3-large.cc",
        "fp32-mobilenet-v3-small.cc",
        "qs8-mobilenet-v2.cc",
    ],
    hdrs = [
        "models.h",
    ],
    deps = [
        "//:XNNPACK",
    ],
)

xnnpack_benchmark(
    name = "benchmark",
    srcs = ["benchmark.cc"],
    tags = xnnpack_slow_benchmark_tags(),
    deps = [
        ":models",
        "//:allocator",
        "//:subgraph",
        "//:xnnpack_h",
        "//bench:bench_utils",
        "@pthreadpool",
    ],
)
