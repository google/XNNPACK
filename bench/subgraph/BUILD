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

BENCHMARKS_SLOW = {
    "attention": True,
    "depthwise_separable": False,
    "elementwise": False,
    "l2_norm": False,
    "layer_norm": False,
    "mobilenet": False,
    "softmax": False,
    "transformer": True,
}

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

[[
    xnnpack_cxx_library(
        name = name + "_lib",
        testonly = 1,
        srcs = [name.replace("_", "-") + ".cc"],
        deps = [
            ":benchmark",
            ":models",
            "//:xnnpack_h",
            "//bench:bench_utils",
            "@com_google_benchmark//:benchmark",
        ],
        alwayslink = True,
    ),
    xnnpack_benchmark(
        name = name,
        srcs = [],
        tags = xnnpack_slow_benchmark_tags() if is_slow else None,
        deps = [
            ":" + name + "_lib",
            ":models",
            "//:xnnpack_h",
            "//bench:bench_utils",
        ],
    ),
] for name, is_slow in BENCHMARKS_SLOW.items()]

xnnpack_benchmark(
    name = "all_benchmarks",
    srcs = [],
    tags = xnnpack_slow_benchmark_tags(),
    deps = [":" + name + "_lib" for name in BENCHMARKS_SLOW] + [
        ":benchmark",
        ":models",
        "//:xnnpack_h",
        "//bench:bench_utils",
    ],
)
