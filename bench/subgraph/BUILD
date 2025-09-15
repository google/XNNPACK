# Copyright 2023-2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load(
    "//:build_defs.bzl",
    "xnnpack_benchmark",
    "xnnpack_cxx_library",
    "xnnpack_slow_benchmark_tags",
)

BENCHMARKS_SLOW_LARGE = {
    "attention": (True, False),
    "binary": (False, True),
    "depthwise_separable": (False, False),
    "convolution": (True, False),
    "elementwise": (False, False),
    "fully_connected": (True, True),
    "l2_norm": (False, False),
    "layer_norm": (False, False),
    "mobilenet": (False, False),
    "softmax": (False, False),
    "transformer": (True, False),
    "unary": (False, True),
}

# Enables usage of WASM SIMD AVX-256 revectorization kernels.
config_setting(
    name = "xnn_benchmark_use_xnn_threadpool",
    define_values = {"xnn_benchmark_use_xnn_threadpool": "true"},
)

# Disables usage of WASM SIMD AVX-256 revectorization kernels.
xnnpack_cxx_library(
    name = "simple_scheduler",
    testonly = 1,
    hdrs = ["simple_scheduler.h"],
    deps = ["//:xnnpack_h"],
)

xnnpack_cxx_library(
    name = "benchmark",
    testonly = True,
    srcs = [
        "benchmark.cc",
    ],
    hdrs = [
        "benchmark.h",
    ],
    local_defines = select({
        ":xnn_benchmark_use_xnn_threadpool": ["XNN_BENCHMARK_USE_THREADPOOL=1"],
        "//conditions:default": [],
    }),
    deps = [
        ":simple_scheduler",
        "//:XNNPACK",
        "//:datatype",
        "//:math",
        "//:subgraph_h",
        "//bench:bench_utils",
        "//test:replicable_random_device",
        "@com_google_benchmark//:benchmark",
        "@pthreadpool",
    ],
    alwayslink = True,
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
        "//test:replicable_random_device",
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
            "//:datatype",
            "//:math",
            "//:operator_utils",
            "//:xnnpack_h",
            "//bench:bench_utils",
            "//test:replicable_random_device",
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
] for name, (is_slow, is_large) in BENCHMARKS_SLOW_LARGE.items()]

xnnpack_benchmark(
    name = "all_benchmarks",
    srcs = [],
    tags = xnnpack_slow_benchmark_tags(),
    deps = [
        ":" + name + "_lib"
        for name, (is_slow, is_large) in BENCHMARKS_SLOW_LARGE.items()
        if not is_large
    ] + [
        ":benchmark",
        ":models",
        "//:xnnpack_h",
        "//bench:bench_utils",
    ],
)
