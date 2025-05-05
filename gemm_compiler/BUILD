# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")

py_binary(
    name = "generate_gemm_microkernels_main",
    srcs = [
        "generate_gemm_microkernels_main.py",
    ],
    main = "generate_gemm_microkernels_main.py",
    tags = [
        "notap",
    ],
    deps = [
        ":generate_gemm_microkernels",
    ],
)

py_library(
    name = "generate_gemm_microkernels",
    srcs = [
        "generate.py",
        "generate_bf16_f32_gemm_microkernels.py",
        "generate_f32_gemm_microkernels.py",
        "generate_qd8_f32_qc4w_gemm_microkernels.py",
        "generate_qd8_f32_qc8w_gemm_microkernels.py",
        "generate_qs8_qc4w_gemm_microkernels.py",
        "generate_qs8_qc8w_gemm_microkernels.py",
    ],
    deps = [
        ":arm_isa_templates",
        ":base_architecture",
        ":x64_isa_templates",
    ],
)

py_library(
    name = "x64_isa_templates",
    srcs = [
        "avx512bf16_template.py",
        "avx512f_template.py",
        "avx512vnni_template.py",
        "fma3_template.py",
    ],
    deps = [":x64_arch_template"],
)

py_library(
    name = "x64_arch_template",
    srcs = [
        "x64_template.py",
    ],
    deps = [
        ":base_architecture",
    ],
)

py_library(
    name = "arm_isa_templates",
    srcs = [
        "neondot_template.py",
        "neonfma_template.py",
        "neonmlal_aarch32_template.py",
    ],
    deps = [":arm_arch_template"],
)

py_library(
    name = "arm_arch_template",
    srcs = [
        "aarch32_template.py",
        "aarch64_template.py",
        "arm_template.py",
    ],
    deps = [
        ":base_architecture",
    ],
)

py_library(
    name = "base_architecture",
    srcs = [
        "base_architecture.py",
    ],
)
