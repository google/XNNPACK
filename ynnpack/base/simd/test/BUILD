# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//ynnpack:build_defs.bzl", "ynn_arch_copts", "ynn_arch_flag", "ynn_binary_linkopts", "ynn_binary_malloc", "ynn_if_arch", "ynn_test_deps")

package(default_visibility = ["//ynnpack:__subpackages__"])

[cc_test(
    name = arch,
    srcs = ynn_if_arch(
        arch,
        [
            "generic.h",
            "main.cc",
            arch + ".cc",
        ],
    ),
    copts = ynn_arch_copts(arch),
    features = [
        # We can't use copts with header modules...?
        "-use_header_modules",
    ],
    linkopts = ynn_binary_linkopts(),
    local_defines = ynn_if_arch(
        arch,
        ["YNN_ARCH=" + ynn_arch_flag(arch)],
        ["YNN_UNSUPPORTED_ARCH"],
    ),
    malloc = ynn_binary_malloc(),
    deps = [
        "//ynnpack/base",  # buildcleaner:keep
        "//ynnpack/base/simd",  # buildcleaner:keep
        "//ynnpack/base/test:fuzz_test",
        "//ynnpack/base/test:random",
        "@com_google_googletest//:gtest",
    ] + ynn_test_deps(),
) for arch in [
    "arm_neon",
    "arm_neonfp16",
    "arm_neonfma",
    "x86_sse2",
    "x86_sse41",
    "x86_avx",
    "x86_avx2",
    "x86_avx512",
    "x86_f16c",
    "x86_fma3",
]]

cc_test(
    name = "multi_vec",
    srcs = ynn_if_arch(
        "x86_sse2",
        [
            "generic.h",
            "main.cc",
            "multi_vec.cc",
        ],
    ),
    copts = ynn_arch_copts("x86_sse2"),
    features = [
        # We can't use copts with header modules...?
        "-use_header_modules",
    ],
    linkopts = ynn_binary_linkopts(),
    local_defines = ynn_if_arch(
        "x86_sse2",
        ["YNN_ARCH=" + ynn_arch_flag("x86_sse2")],
        ["YNN_UNSUPPORTED_ARCH"],
    ),
    malloc = ynn_binary_malloc(),
    deps = [
        "//ynnpack/base",  # buildcleaner:keep
        "//ynnpack/base/simd",  # buildcleaner:keep
        "//ynnpack/base/test:fuzz_test",
        "//ynnpack/base/test:random",
        "@com_google_googletest//:gtest",
    ] + ynn_test_deps(),
)
