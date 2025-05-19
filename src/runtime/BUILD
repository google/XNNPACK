# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load(
    "//:build_defs.bzl",
    "xnnpack_cc_library",
    "xnnpack_slinky_deps",
)

package(default_visibility = ["//:__subpackages__"])

xnnpack_cc_library(
    name = "memory_planner",
    srcs = ["memory-planner.c"],
    hdrs = [
        "//:src/xnnpack/memory-planner.h",
    ],
    deps = [
        "//:allocator",
        "//:common",
        "//:xnnpack_h",
        "//src/subgraph",
    ],
)

xnnpack_cc_library(
    name = "weights_cache",
    srcs = ["weights_cache.c"],
    deps = [
        "//:allocator",
        "//:cache",
        "//:common",
        "//:logging",
        "//:memory",
        "//:params",
        "//:xnnpack_h",
    ],
)

xnnpack_cc_library(
    name = "runtime",
    srcs = ["runtime.c"],
    deps = [
        ":memory_planner",
        ":weights_cache",
        "//:allocation_type",
        "//:allocator",
        "//:cache",
        "//:common",
        "//:internal",
        "//:logging",
        "//:memory",
        "//:microkernel_type",
        "//:params",
        "//:xnnpack_h",
        "//src/operators",
        "//src/operators:operator_utils",
        "//src/subgraph",
        "//src/subgraph:node_type",
        "//src/subgraph:subgraph_types",
        "@pthreadpool",
    ] + xnnpack_slinky_deps(),
)
