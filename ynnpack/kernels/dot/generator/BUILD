# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")

package(default_visibility = ["//ynnpack/kernels/dot:__pkg__"])

py_library(
    name = "dot_base",
    srcs = ["dot_base.py"],
)

py_library(
    name = "arm",
    srcs = ["arm.py"],
    deps = [":dot_base"],
)

[py_binary(
    name = name,
    srcs = [name + ".py"],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":arm",
        ":dot_base",
    ],
) for name in [
    "arm_bf16_bf16_fp32",
    "arm_bf16_bf16_fp32_k2",
    "arm_bf16_bf16_fp32_k4",
    "arm_fp32",
    "arm_fp64",
    "arm_int8_int8_int32",
]]

py_library(
    name = "x86",
    srcs = ["x86.py"],
    deps = [":dot_base"],
)

[py_binary(
    name = name,
    srcs = [name + ".py"],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":dot_base",
        ":x86",
    ],
) for name in [
    "x86_bf16_bf16_fp32",
    "x86_bf16_bf16_fp32_k1",
    "x86_fp16_fp16_fp32",
    "x86_fp32",
    "x86_fp32_k2",
    "x86_fp32_k4",
    "x86_fp64",
    "x86_int8_int8_int32",
    "x86_int8_int8_int32_k1",
    "x86_int8_int8_int32_k16",
    "x86_uint8_int8_int32",
    "x86_uint8_int8_int32_k16",
]]
