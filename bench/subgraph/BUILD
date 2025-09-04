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
    name = "benchmark",
    srcs = [
        "benchmark.cc",
    ],
    hdrs = [
        "benchmark.h",
    ],
    deps = [
        "//:XNNPACK",
        "//:subgraph_h",
        "//bench:bench_utils",
        "@com_google_benchmark//:benchmark",
        "@pthreadpool",
    ],
)

xnnpack_cxx_library(
    name = "models",
    testonly = 1,
    srcs = [
        "fp32-mobilenet-v1.cc",
        "fp32-mobilenet-v2.cc",
        "fp32-mobilenet-v3-large.cc",
        "fp32-mobilenet-v3-small.cc",
        "fp32-transformer.cc",
        "qd8-transformer.cc",
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
    name = "attention",
    srcs = ["attention.cc"],
    tags = xnnpack_slow_benchmark_tags(),
    deps = [
        ":benchmark",
        "//:xnnpack_h",
        "//bench:bench_utils",
    ],
)

xnnpack_benchmark(
    name = "depthwise_separable",
    srcs = ["depthwise-separable.cc"],
    deps = [
        ":benchmark",
        "//:xnnpack_h",
    ],
)

xnnpack_benchmark(
    name = "elementwise",
    srcs = ["elementwise.cc"],
    deps = [
        ":benchmark",
        "//:xnnpack_h",
    ],
)

xnnpack_benchmark(
    name = "l2_norm",
    srcs = ["l2-norm.cc"],
    deps = [
        ":benchmark",
        "//:xnnpack_h",
    ],
)

xnnpack_benchmark(
    name = "layer-norm",
    srcs = ["layer-norm.cc"],
    deps = [
        ":benchmark",
        "//:xnnpack_h",
    ],
)

xnnpack_benchmark(
    name = "mobilenet",
    srcs = ["mobilenet.cc"],
    deps = [
        ":benchmark",
        ":models",
        "//:xnnpack_h",
    ],
)

xnnpack_benchmark(
    name = "softmax",
    srcs = ["softmax.cc"],
    deps = [
        ":benchmark",
        "//:xnnpack_h",
    ],
)

xnnpack_benchmark(
    name = "transformer",
    srcs = ["transformer.cc"],
    tags = xnnpack_slow_benchmark_tags(),
    deps = [
        ":benchmark",
        ":models",
        "//:xnnpack_h",
        "//bench:bench_utils",
    ],
)
