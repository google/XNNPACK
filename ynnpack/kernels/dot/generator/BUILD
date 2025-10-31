# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_python//python:py_binary.bzl", "py_binary")

package(default_visibility = ["//ynnpack/kernels/dot:__pkg__"])

py_binary(
    name = "generator",
    srcs = [
        "arm.py",
        "arm_bf16_bf16_fp32.py",
        "arm_fp32.py",
        "arm_int8_int8_int32.py",
        "dot_base.py",
        "generator.py",
        "x86.py",
        "x86_bf16_bf16_fp32.py",
        "x86_fp32.py",
        "x86_fp32_k2.py",
        "x86_fp32_k4.py",
        "x86_int8_int8_int32.py",
        "x86_int8_int8_int32_k16.py",
        "x86_uint8_int8_int32.py",
        "x86_uint8_int8_int32_k16.py",
    ],
    main = "generator.py",
)
