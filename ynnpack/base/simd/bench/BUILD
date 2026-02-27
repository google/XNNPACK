# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("//ynnpack:build_defs.bzl", "ynn_arch_copts", "ynn_arch_flag", "ynn_benchmark_deps", "ynn_binary_linkopts", "ynn_binary_malloc", "ynn_if_arch")

package(default_visibility = ["//ynnpack:__subpackages__"])

# These benchmarks are difficult to test, because we need to compile the benchmark code with the
# required architecture flags, but that means we can't check if the architecture is supported
# before running code that requires that architecture. Most of the time we can get away with
# this, but for some targets, such as ARM SVE, every function has code that requires SVE in its
# prologue.
#
# We could solve this by declaring the benchmark functions in a separate file, compiled with the
# appropriate flags, and then call thoes functions from a different file, which checks the
# architecture flags first. This would be comparable to how most kernels work. This would need a
# lot of extra bookkeeping and boilerplate or hacks though.
#
# To avoid all this, we just make these `cc_binary` targets. They are not continuously tested,
# avoiding the problem described above. They aren't a lot of code, so if they do break, it hopefully
# would not be difficult to get them working again when needed manually.
[cc_binary(
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
        "//ynnpack/base",
        "//ynnpack/base/simd",
        "@slinky//slinky/base",
    ] + ynn_benchmark_deps(),
) for arch in [
    "arm_neon",
    "arm64_sve",
    "hexagon_hvx",
    "x86_sse2",
    "x86_avx",
    "x86_avx512",
    "x86_fma3",
]]
