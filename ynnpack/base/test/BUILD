# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//ynnpack:__subpackages__"])

cc_library(
    name = "buffer",
    testonly = True,
    hdrs = ["buffer.h"],
)

cc_library(
    name = "tensor",
    testonly = True,
    hdrs = ["tensor.h"],
    deps = [
        ":buffer",
        "//ynnpack/base",
    ],
)

cc_library(
    name = "random",
    testonly = True,
    hdrs = ["random.h"],
    deps = [
        "//ynnpack:ynnpack_h",
        "//ynnpack/base",
    ],
)

cc_library(
    name = "util",
    testonly = True,
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//ynnpack/base",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "fuzz_test",
    testonly = True,
    hdrs = ["fuzz_test.h"],
    deps = [
        "//ynnpack/base",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "random_test",
    srcs = ["random_test.cc"],
    deps = [
        ":fuzz_test",
        ":random",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "tensor_test",
    srcs = ["tensor_test.cc"],
    deps = [
        ":fuzz_test",
        ":tensor",
        "@com_google_googletest//:gtest_main",
    ],
)
