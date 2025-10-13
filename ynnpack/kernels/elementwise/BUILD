# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

py_library(
    name = "generator",
    srcs = [
        "arm.py",
        "common_rules.py",
        "compiler.py",
        "generator.py",
        "x86.py",
    ],
    visibility = [
        "//ynnpack/kernels/binary:__pkg__",
        "//ynnpack/kernels/ternary:__pkg__",
        "//ynnpack/kernels/unary:__pkg__",
    ],
)

py_test(
    name = "compiler_test",
    srcs = ["compiler_test.py"],
    tags = [
        "xnnpack_tap_main_only",
    ],
    deps = [":generator"],
)
