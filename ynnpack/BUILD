# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@bazel_skylib//lib:selects.bzl", "selects")
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

selects.config_setting_group(
    name = "arm64",
    match_any = [
        "@platforms//cpu:aarch64",
        "@platforms//cpu:arm64e",
        "@platforms//cpu:arm64_32",
    ],
)

selects.config_setting_group(
    name = "arm32",
    match_any = [
        "@platforms//cpu:armv7",
    ],
)

selects.config_setting_group(
    name = "arm",
    match_any = [
        ":arm32",
        ":arm64",
    ],
)

selects.config_setting_group(
    name = "x86",
    match_any = [
        "@platforms//cpu:x86_32",
        "@platforms//cpu:x86_64",
    ],
)

# TODO(dsharlet): This doesn't actually do anything yet, it's just a placeholder for a proper
# toolchain configuration.
config_setting(
    name = "hexagon-clang",
    define_values = {"ynn_hexagon_clang": "true"},
)

selects.config_setting_group(
    name = "hexagon",
    match_any = [
        ":hexagon-clang",
    ],
)

selects.config_setting_group(
    name = "apple_clang",
    match_any = [
        "@platforms//os:ios",
        "@platforms//os:macos",
    ],
)

selects.config_setting_group(
    name = "not_apple",
    match_any = [
        "@platforms//os:linux",
        "@platforms//os:android",
        "@platforms//os:windows",
    ],
)

selects.config_setting_group(
    name = "msvc",
    match_any = [
        "@rules_cc//cc/compiler:clang-cl",
        "@rules_cc//cc/compiler:msvc-cl",
    ],
)

define_build_option(
    name = "ynn_enable_cpuinfo",
    default_any = [
        ":arm",
        ":x86",
    ],
)

define_build_option(
    name = "ynn_enable_arm_neon",
    default_all = [":arm"],
)

define_build_option(
    name = "ynn_enable_arm_neondot",
    default_all = [":ynn_enable_arm_neon"],
)

define_build_option(
    name = "ynn_enable_arm_neonfma",
    default_all = [":ynn_enable_arm_neon"],
)

define_build_option(
    name = "ynn_enable_arm_neonfp16",
    default_all = [":ynn_enable_arm_neon"],
)

define_build_option(
    name = "ynn_enable_arm_neonfp16arith",
    default_all = [":ynn_enable_arm_neon"],
)

define_build_option(
    name = "ynn_enable_arm_neonbf16",
    default_all = [":ynn_enable_arm_neon"],
)

define_build_option(
    name = "ynn_enable_arm64",
    default_all = [":arm64"],
)

define_build_option(
    name = "ynn_enable_arm64_neoni8mm",
    default_all = [":ynn_enable_arm64"],
)

define_build_option(
    name = "ynn_enable_arm64_neon",
    default_all = [":ynn_enable_arm64"],
)

define_build_option(
    name = "ynn_enable_arm64_sme",
    default_all = [":ynn_enable_arm64"],
)

define_build_option(
    name = "ynn_enable_arm64_sme2",
    default_all = [":ynn_enable_arm64_sme"],
)

define_build_option(
    name = "ynn_enable_arm64_sve",
    default_all = [
        ":ynn_enable_arm64",
        # Apple's compilers appear to generate SVE code in *any* function, regardless of whether
        # the function actually uses it, which causes us to execute SVE code on non-apple targets.
        ":not_apple",
    ],
)

define_build_option(
    name = "ynn_enable_x86_sse",
    default_all = [":x86"],
)

define_build_option(
    name = "ynn_enable_x86_sse2",
    default_all = [":ynn_enable_x86_sse"],
)

define_build_option(
    name = "ynn_enable_x86_ssse3",
    default_all = [":ynn_enable_x86_sse"],
)

define_build_option(
    name = "ynn_enable_x86_sse41",
    default_all = [":ynn_enable_x86_sse"],
)

define_build_option(
    name = "ynn_enable_x86_avx",
    default_all = [":ynn_enable_x86_sse"],
)

define_build_option(
    name = "ynn_enable_x86_f16c",
    default_all = [":ynn_enable_x86_avx"],
)

define_build_option(
    name = "ynn_enable_x86_f16c_fma3",
    default_all = [
        ":ynn_enable_x86_f16c",
        ":ynn_enable_x86_fma3",
    ],
)

define_build_option(
    name = "ynn_enable_x86_avx2",
    default_all = [":ynn_enable_x86_avx"],
)

define_build_option(
    name = "ynn_enable_x86_fma3",
    default_all = [":ynn_enable_x86_avx"],
)

define_build_option(
    name = "ynn_enable_x86_avx2_fma3",
    default_all = [
        ":ynn_enable_x86_avx2",
        ":ynn_enable_x86_fma3",
    ],
)

define_build_option(
    name = "ynn_enable_x86_avx512",
    default_all = [":ynn_enable_x86_avx"],
)

define_build_option(
    name = "ynn_enable_x86_avx512bf16",
    default_all = [
        ":ynn_enable_x86_avx512",
        # This hits internal compiler errors on MSVC on windows. "not_windows" is hard to express in
        # the build system, and I think linux is the only place we need this to work for now.
        "@platforms//os:linux",
    ],
)

define_build_option(
    name = "ynn_enable_x86_avx512fp16",
    default_all = [":ynn_enable_x86_avx512"],
)

define_build_option(
    name = "ynn_enable_x86_avx512vnni",
    default_all = [":ynn_enable_x86_avx512"],
)

define_build_option(
    name = "ynn_enable_x86_amx",
    default_all = [":x86"],
)

define_build_option(
    name = "ynn_enable_x86_amxbf16",
    default_all = [":ynn_enable_x86_amx"],
)

define_build_option(
    name = "ynn_enable_x86_amxfp16",
    default_all = [":ynn_enable_x86_amx"],
)

define_build_option(
    name = "ynn_enable_x86_amxint8",
    default_all = [":ynn_enable_x86_amx"],
)

define_build_option(
    name = "ynn_enable_hvx",
    default_all = [":hexagon"],
)
