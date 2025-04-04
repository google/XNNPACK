# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load(
    "//:build_defs.bzl",
    "xnnpack_benchmark",
    "xnnpack_optional_tflite_copts",
    "xnnpack_optional_tflite_deps",
    "xnnpack_slow_benchmark_tags",
)

OPERATOR_BENCHMARK_DEPS = [
    "//bench:bench_utils",
    "//:XNNPACK",
    "//:aligned_allocator",
    "//:buffer",
    "//:cache",
    "//:common",
    "//:datatype",
    "//:math",
]

xnnpack_benchmark(
    name = "unary_bench",
    srcs = ["unary.cc"],
    copts = xnnpack_optional_tflite_copts(),
    tags = ["nowin32"],
    deps = OPERATOR_BENCHMARK_DEPS + xnnpack_optional_tflite_deps(),
)

xnnpack_benchmark(
    name = "binary_bench",
    srcs = ["binary.cc"],
    copts = xnnpack_optional_tflite_copts(),
    tags = ["nowin32"],
    deps = OPERATOR_BENCHMARK_DEPS + xnnpack_optional_tflite_deps(),
)

xnnpack_benchmark(
    name = "average_pooling_bench",
    srcs = ["average-pooling.cc"],
    copts = xnnpack_optional_tflite_copts(),
    tags = ["nowin32"],
    deps = OPERATOR_BENCHMARK_DEPS + xnnpack_optional_tflite_deps(),
)

xnnpack_benchmark(
    name = "batch_matrix_multiply_bench",
    srcs = ["batch-matrix-multiply.cc"],
    copts = xnnpack_optional_tflite_copts(),
    tags = xnnpack_slow_benchmark_tags() + ["nowin32"],
    deps = OPERATOR_BENCHMARK_DEPS + xnnpack_optional_tflite_deps() + [
        "//test:next_prime",
        "@cpuinfo",
        "@pthreadpool",
    ],
)

xnnpack_benchmark(
    name = "convolution_bench",
    srcs = ["convolution.cc"],
    copts = xnnpack_optional_tflite_copts(),
    tags = xnnpack_slow_benchmark_tags() + ["nowin32"],
    deps = OPERATOR_BENCHMARK_DEPS + xnnpack_optional_tflite_deps() + ["@pthreadpool"],
)

xnnpack_benchmark(
    name = "deconvolution_bench",
    srcs = ["deconvolution.cc"],
    copts = xnnpack_optional_tflite_copts(),
    tags = xnnpack_slow_benchmark_tags() + ["nowin32"],
    deps = OPERATOR_BENCHMARK_DEPS + xnnpack_optional_tflite_deps() + ["@pthreadpool"],
)

xnnpack_benchmark(
    name = "fully_connected_bench",
    srcs = ["fully-connected.cc"],
    copts = xnnpack_optional_tflite_copts(),
    tags = ["nowin32"],
    deps = OPERATOR_BENCHMARK_DEPS + xnnpack_optional_tflite_deps(),
)

xnnpack_benchmark(
    name = "max_pooling_bench",
    srcs = ["max-pooling.cc"],
    deps = OPERATOR_BENCHMARK_DEPS,
)

xnnpack_benchmark(
    name = "prelu_bench",
    srcs = ["prelu.cc"],
    copts = xnnpack_optional_tflite_copts(),
    tags = ["nowin32"],
    deps = OPERATOR_BENCHMARK_DEPS + xnnpack_optional_tflite_deps(),
)

xnnpack_benchmark(
    name = "resize_bilinear_nhwc_bench",
    srcs = ["resize-bilinear-nhwc.cc"],
    deps = OPERATOR_BENCHMARK_DEPS,
)

xnnpack_benchmark(
    name = "scaled_dot_product_attention_bench",
    srcs = ["scaled-dot-product-attention.cc"],
    copts = xnnpack_optional_tflite_copts(),
    tags = xnnpack_slow_benchmark_tags() + ["nowin32"],
    deps = OPERATOR_BENCHMARK_DEPS + xnnpack_optional_tflite_deps(),
)

xnnpack_benchmark(
    name = "softmax_bench",
    srcs = [
        "softmax.cc",
    ],
    copts = xnnpack_optional_tflite_copts(),
    tags = ["nowin32"],
    deps = OPERATOR_BENCHMARK_DEPS + xnnpack_optional_tflite_deps(),
)
