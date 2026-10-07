# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load(
    "//ynnpack:build_defs.bzl",
    "ynn_benchmark_deps",
    "ynn_binary_linkopts",
    "ynn_binary_malloc",
)

package(default_visibility = ["//ynnpack:__subpackages__"])

_COMPATIBLE_WITH = []

cc_library(
    name = "cost_model",
    hdrs = glob(["*.h"]),
    compatible_with = _COMPATIBLE_WITH,
)

# This tool generates training data for a dot cost model for a particular microarchitecture. Run it
# on the target CPU core with `bench --benchmark_min_time=0.1s --benchmark_format=csv > <cpu>.csv`.
# Then, run `fit_cost_models.py` to generate a header describing the cost model, for use in YNNPACK.
cc_test(
    name = "bench",
    srcs = ["bench.cc"],
    args = [
        "--benchmark_min_time=1x",
        "--benchmark_format=csv",
    ],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":cost_model",
        "//ynnpack/base",
        "//ynnpack/base/test:buffer",
        "//ynnpack/base/test:tensor",
        "//ynnpack/kernels/dot",
    ] + ynn_benchmark_deps(),
)
