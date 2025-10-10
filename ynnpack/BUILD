# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//ynnpack:build_defs.bzl", "define_build_option")

package(default_visibility = ["//ynnpack:__subpackages__"])

_COMPATIBLE_WITH = []

cc_library(
    name = "ynnpack_h",
    hdrs = ["include/ynnpack.h"],
    compatible_with = _COMPATIBLE_WITH,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ynnpack",
    hdrs = ["include/ynnpack.h"],
    compatible_with = _COMPATIBLE_WITH,
    visibility = ["//visibility:public"],
    deps = [
        ":ynnpack_h",
        "//ynnpack/subgraph",  # buildcleaner: keep
    ],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:private"],
)

define_build_option(
    name = "ynn_enable_arm_neon",
    default_conditions = ["//ynnpack/base/build_config:arm"],
)

define_build_option(
    name = "ynn_enable_arm_neondot",
    default_conditions = ["//ynnpack/base/build_config:arm"],
)

define_build_option(
    name = "ynn_enable_arm64_neoni8mm",
    default_conditions = ["//ynnpack/base/build_config:arm64"],
)

define_build_option(
    name = "ynn_enable_arm64_neon",
    default_conditions = ["//ynnpack/base/build_config:arm64"],
)

define_build_option(
    name = "ynn_enable_arm64_sme",
    default_conditions = ["//ynnpack/base/build_config:arm64"],
)

define_build_option(
    name = "ynn_enable_arm64_sme2",
    default_conditions = [":ynn_enable_arm64_sme"],
)

define_build_option(
    name = "ynn_enable_x86_sse2",
    default_conditions = ["//ynnpack/base/build_config:x86"],
)

define_build_option(
    name = "ynn_enable_x86_ssse3",
    default_conditions = ["//ynnpack/base/build_config:x86"],
)

define_build_option(
    name = "ynn_enable_x86_sse41",
    default_conditions = ["//ynnpack/base/build_config:x86"],
)

define_build_option(
    name = "ynn_enable_x86_avx",
    default_conditions = ["//ynnpack/base/build_config:x86"],
)

define_build_option(
    name = "ynn_enable_x86_f16c",
    default_conditions = ["//ynnpack/base/build_config:x86"],
)

define_build_option(
    name = "ynn_enable_x86_avx2",
    default_conditions = ["//ynnpack/base/build_config:x86"],
)

define_build_option(
    name = "ynn_enable_x86_fma3",
    default_conditions = ["//ynnpack/base/build_config:x86"],
)

define_build_option(
    name = "ynn_enable_x86_avx2_fma3",
    default_conditions = ["//ynnpack/base/build_config:x86"],
)

define_build_option(
    name = "ynn_enable_x86_avx512",
    default_conditions = ["//ynnpack/base/build_config:x86"],
)

define_build_option(
    name = "ynn_enable_x86_avx512f",
    default_conditions = [":ynn_enable_x86_avx512"],
)

define_build_option(
    name = "ynn_enable_x86_avx512bw",
    default_conditions = [":ynn_enable_x86_avx512"],
)

define_build_option(
    name = "ynn_enable_x86_avx512bf16",
    default_conditions = [":ynn_enable_x86_avx512"],
)

define_build_option(
    name = "ynn_enable_x86_avx512fp16",
    default_conditions = [":ynn_enable_x86_avx512"],
)

define_build_option(
    name = "ynn_enable_x86_avx512vnni",
    default_conditions = [":ynn_enable_x86_avx512"],
)

define_build_option(
    name = "ynn_enable_x86_amx",
    default_conditions = ["//ynnpack/base/build_config:x86"],
)

define_build_option(
    name = "ynn_enable_x86_amxbf16",
    default_conditions = [":ynn_enable_x86_amx"],
)

define_build_option(
    name = "ynn_enable_x86_amxfp16",
    default_conditions = [":ynn_enable_x86_amx"],
)

define_build_option(
    name = "ynn_enable_x86_amxint8",
    default_conditions = [":ynn_enable_x86_amx"],
)
