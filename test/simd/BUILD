# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load(
    "//:build_defs.bzl",
    "xnnpack_unit_test",
)
load(
    "//:build_params.bzl",
    # "xnnpack_select_if",
    "xnnpack_simd_copts_for_arch",
    "xnnpack_simd_f16_archs",
    "xnnpack_simd_f32_archs",
    "xnnpack_simd_s16_archs",
    "xnnpack_simd_s32_archs",
    "xnnpack_simd_s8_archs",
    "xnnpack_simd_u8_archs",
)

SIMD_TEST_DEPS = [
    "//test:replicable_random_device",
    "//:buffer",
    "//:common",
    "//:isa_checks",
    "//:math",
    "//:microkernels_h",
]

[xnnpack_unit_test(
    name = "f32_simd_" + arch + "_test",
    srcs = [
        "f32-simd-" + arch + ".cc",
    ],
    copts = xnnpack_simd_copts_for_arch(arch),
    tags = ["simd_wrapper_test"],
    deps = SIMD_TEST_DEPS,
) for arch in xnnpack_simd_f32_archs()]

[xnnpack_unit_test(
    name = "f16_simd_" + arch + "_test",
    srcs = [
        "f16-simd-" + arch + ".cc",
    ],
    copts = xnnpack_simd_copts_for_arch(arch),
    features = ["-use_header_modules"],
    tags = ["simd_wrapper_test"],
    deps = SIMD_TEST_DEPS,
) for arch in xnnpack_simd_f16_archs()]

[xnnpack_unit_test(
    name = "s16_simd_" + arch + "_test",
    srcs = [
        "s16-simd-" + arch + ".cc",
    ],
    copts = xnnpack_simd_copts_for_arch(arch),
    tags = ["simd_wrapper_test"],
    deps = SIMD_TEST_DEPS,
) for arch in xnnpack_simd_s16_archs()]

[xnnpack_unit_test(
    name = "s32_simd_" + arch + "_test",
    srcs = [
        "s32-simd-" + arch + ".cc",
    ],
    copts = xnnpack_simd_copts_for_arch(arch),
    tags = ["simd_wrapper_test"],
    deps = SIMD_TEST_DEPS,
) for arch in xnnpack_simd_s32_archs()]

[xnnpack_unit_test(
    name = "s8_simd_" + arch + "_test",
    srcs = [
        "s8-simd-" + arch + ".cc",
    ],
    copts = xnnpack_simd_copts_for_arch(arch),
    tags = ["simd_wrapper_test"],
    deps = SIMD_TEST_DEPS,
) for arch in xnnpack_simd_s8_archs()]

[xnnpack_unit_test(
    name = "u8_simd_" + arch + "_test",
    srcs = [
        "u8-simd-" + arch + ".cc",
    ],
    copts = xnnpack_simd_copts_for_arch(arch),
    tags = ["simd_wrapper_test"],
    deps = SIMD_TEST_DEPS,
) for arch in xnnpack_simd_u8_archs()]
