# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//ynnpack:__subpackages__"])

_COMPATIBLE_WITH = []

# These headers provide SIMD wrappers shared among multiple targets.
cc_library(
    name = "internal",
    compatible_with = _COMPATIBLE_WITH,
    textual_hdrs = [
    ],
    visibility = ["//visibility:private"],
    deps = [
        "//ynnpack/base",
    ],
)

cc_library(
    name = "simd",
    hdrs = ["vec.h"],
    compatible_with = _COMPATIBLE_WITH,
    # These headers need to be textual because we can't compile them with the appropriate copts.
    textual_hdrs = [
        # These headers should not be directly included.
        "arm_neon_base.h",
        "arm_neon_partial_load_store.h",
        "arm64_sve_partial_load_store.h",
        "x86_avx2_base.h",
        "x86_avx_base.h",
        "x86_sse2_base.h",
        "x86_sse41_base.h",
        "x86_avx_partial_load_store.h",
        "x86_sse2_partial_load_store.h",
        "generic.inc",
        # For the most part, only one of these headers should be included. Multiple of these headers
        # may define the same operation and type, using a different implementation, depending on the
        # target. For example, f32x16 is provided by both avx512 as a single 512-bit vector, and
        # avx as two 256-bit vectors, and the `concat` operation of two f32x8 values is implemented
        # appropriately for that type.
        "arm_neon.h",
        "arm64_sve.h",
        "arm_neonfma.h",
        "arm_neonfp16.h",
        "hexagon_hvx.h",
        "scalar.h",
        "x86_avx.h",
        "x86_avx2.h",
        "x86_avx512.h",
        "x86_f16c.h",
        "x86_fma3.h",
        "x86_sse2.h",
        "x86_sse41.h",
    ],
    deps = ["//ynnpack/base"],
)
