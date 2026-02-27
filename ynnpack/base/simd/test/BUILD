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
    ),
    malloc = ynn_binary_malloc(),
    deps = [
        "//ynnpack/base",  # buildcleaner:keep
        "//ynnpack/base/simd",  # buildcleaner:keep
        "//ynnpack/base/test:fuzz_test",
        "//ynnpack/base/test:random",
        "@slinky//slinky/base",
    ] + ynn_test_deps(),
) for arch in [
    "arm_neon",
    "arm_neonfp16",
    "arm_neonfma",
    "arm64_sve",
    "x86_sse2",
    "x86_sse41",
    "x86_avx",
    "x86_avx2",
    "x86_avx512",
    "x86_f16c",
    "x86_fma3",
    "hexagon_hvx",
]]

cc_test(
    name = "multi_vec",
    srcs = [
        "generic.h",
        "multi_vec.cc",
    ],
    features = [
        # We can't use copts with header modules...?
        "-use_header_modules",
    ],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        "//ynnpack/base",  # buildcleaner:keep
        "//ynnpack/base/simd",  # buildcleaner:keep
        "//ynnpack/base/test:fuzz_test",
        "//ynnpack/base/test:random",
        "@slinky//slinky/base",
    ] + ynn_test_deps(),
)
