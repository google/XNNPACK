# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load(
    "//:build_defs.bzl",
    "xnnpack_cxx_library",
    "xnnpack_unit_test",
)

package(default_visibility = ["//:__subpackages__"])

xnnpack_cxx_library(
    name = "subgraph_matcher",
    testonly = True,
    srcs = ["subgraph_matcher.cc"],
    hdrs = ["subgraph_matcher.h"],
    tags = ["no_ynnpack"],
    deps = [
        "//:cache",
        "//:logging",
        "//:node_type",
        "//:operator_utils",
        "//:subgraph",
        "//:xnnpack_h",
        "@com_google_googletest//:gtest",  # no g3
    ],
)

xnnpack_unit_test(
    name = "subgraph_matcher_test",
    srcs = ["subgraph_matcher_test.cc"],
    tags = ["no_ynnpack"],
    deps = [
        ":subgraph_matcher",
        "//litert/tensor",
        "//litert/tensor:arithmetic",
        "//litert/tensor:datatypes",
        "//litert/tensor/backends/xnnpack:conversion",
        "//litert/tensor/utils:matchers_no_g3",
    ],
)
