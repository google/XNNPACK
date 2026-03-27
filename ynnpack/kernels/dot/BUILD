# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("//ynnpack:build_defs.bzl", "ynn_binary_linkopts", "ynn_binary_malloc", "ynn_cc_library", "ynn_generate_srcs", "ynn_kernel_copts", "ynn_test_deps")

_COMPATIBLE_WITH = []

ynn_generate_srcs(
    name = "arm_bf16_bf16_fp32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator:arm_bf16_bf16_fp32",
    output_srcs = [
        "arm64_neon_bf16_bf16_fp32.cc",
        "arm64_neon_bf16_bf16_fp32.inc",
    ],
)

ynn_generate_srcs(
    name = "arm_bf16_bf16_fp32_k2",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator:arm_bf16_bf16_fp32_k2",
    output_srcs = [
        "arm_neonbf16_bf16_bf16_fp32_k2.cc",
        "arm_neonbf16_bf16_bf16_fp32_k2.inc",
    ],
)

ynn_generate_srcs(
    name = "arm_bf16_bf16_fp32_k4",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator:arm_bf16_bf16_fp32_k4",
    output_srcs = [
        "arm_neonbf16_bf16_bf16_fp32_k4.cc",
        "arm_neonbf16_bf16_bf16_fp32_k4.inc",
    ],
)

ynn_generate_srcs(
    name = "arm_fp32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator:arm_fp32",
    output_srcs = [
        "arm64_neon_fp32.cc",
        "arm64_neon_fp32.inc",
    ],
)

ynn_generate_srcs(
    name = "arm_fp64",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator:arm_fp64",
    output_srcs = [
        "arm64_neon_fp64.cc",
        "arm64_neon_fp64.inc",
    ],
)

ynn_generate_srcs(
    name = "arm_int8_int8_int32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator:arm_int8_int8_int32",
    output_srcs = [
        "arm_neon_int8_int8_int32.cc",
        "arm_neon_int8_int8_int32.inc",
        "arm_neondot_int8_int8_int32.cc",
        "arm_neondot_int8_int8_int32.inc",
        "arm64_neoni8mm_int8_int8_int32.cc",
        "arm64_neoni8mm_int8_int8_int32.inc",
    ],
)

ynn_generate_srcs(
    name = "x86_bf16_bf16_fp32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator:x86_bf16_bf16_fp32",
    output_srcs = [
        "x86_avx2_fma3_bf16_bf16_fp32.cc",
        "x86_avx2_fma3_bf16_bf16_fp32.inc",
        "x86_avx512_bf16_bf16_fp32.cc",
        "x86_avx512_bf16_bf16_fp32.inc",
        "x86_avx512bf16_bf16_bf16_fp32.cc",
        "x86_avx512bf16_bf16_bf16_fp32.inc",
    ],
)

ynn_generate_srcs(
    name = "x86_bf16_bf16_fp32_k1",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator:x86_bf16_bf16_fp32_k1",
    output_srcs = [
        "x86_avx2_fma3_bf16_bf16_fp32_k1.cc",
        "x86_avx2_fma3_bf16_bf16_fp32_k1.inc",
        "x86_avx512_bf16_bf16_fp32_k1.cc",
        "x86_avx512_bf16_bf16_fp32_k1.inc",
    ],
)

ynn_generate_srcs(
    name = "x86_fp16_fp16_fp32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator:x86_fp16_fp16_fp32",
    output_srcs = [
        "x86_f16c_fp16_fp16_fp32.cc",
        "x86_f16c_fp16_fp16_fp32.inc",
        "x86_f16c_fma3_fp16_fp16_fp32.cc",
        "x86_f16c_fma3_fp16_fp16_fp32.inc",
        "x86_avx512_fp16_fp16_fp32.cc",
        "x86_avx512_fp16_fp16_fp32.inc",
    ],
)

ynn_generate_srcs(
    name = "x86_fp32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator:x86_fp32",
    output_srcs = [
        "x86_sse2_fp32.cc",
        "x86_sse2_fp32.inc",
        "x86_avx_fp32.cc",
        "x86_avx_fp32.inc",
        "x86_fma3_fp32.cc",
        "x86_fma3_fp32.inc",
        "x86_avx512_fp32.cc",
        "x86_avx512_fp32.inc",
    ],
)

ynn_generate_srcs(
    name = "x86_fp32_k2",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator:x86_fp32_k2",
    output_srcs = [
        "x86_avx2_fp32_k2.cc",
        "x86_avx2_fp32_k2.inc",
        "x86_avx2_fma3_fp32_k2.cc",
        "x86_avx2_fma3_fp32_k2.inc",
        "x86_avx512_fp32_k2.cc",
        "x86_avx512_fp32_k2.inc",
    ],
)

ynn_generate_srcs(
    name = "x86_fp32_k4",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator:x86_fp32_k4",
    output_srcs = [
        "x86_avx512_fp32_k4.cc",
        "x86_avx512_fp32_k4.inc",
    ],
)

ynn_generate_srcs(
    name = "x86_fp64",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator:x86_fp64",
    output_srcs = [
        "x86_avx_fp64.cc",
        "x86_avx_fp64.inc",
        "x86_fma3_fp64.cc",
        "x86_fma3_fp64.inc",
        "x86_avx512_fp64.cc",
        "x86_avx512_fp64.inc",
    ],
)

ynn_generate_srcs(
    name = "x86_int8_int8_int32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator:x86_int8_int8_int32",
    output_srcs = [
        "x86_avx2_int8_int8_int32.cc",
        "x86_avx2_int8_int8_int32.inc",
        "x86_avx512_int8_int8_int32.cc",
        "x86_avx512_int8_int8_int32.inc",
    ],
)

ynn_generate_srcs(
    name = "x86_int8_int8_int32_k1",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator:x86_int8_int8_int32_k1",
    output_srcs = [
        "x86_avx2_int8_int8_int32_k1.cc",
        "x86_avx2_int8_int8_int32_k1.inc",
        "x86_avx512_int8_int8_int32_k1.cc",
        "x86_avx512_int8_int8_int32_k1.inc",
    ],
)

ynn_generate_srcs(
    name = "x86_int8_int8_int32_k16",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator:x86_int8_int8_int32_k16",
    output_srcs = [
        "x86_avx512_int8_int8_int32_k16.cc",
        "x86_avx512_int8_int8_int32_k16.inc",
    ],
)

ynn_generate_srcs(
    name = "x86_uint8_int8_int32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator:x86_uint8_int8_int32",
    output_srcs = [
        "x86_avx512vnni_uint8_int8_int32.cc",
        "x86_avx512vnni_uint8_int8_int32.inc",
    ],
)

ynn_generate_srcs(
    name = "x86_uint8_int8_int32_k16",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator:x86_uint8_int8_int32_k16",
    output_srcs = [
        "x86_avx512vnni_uint8_int8_int32_k16.cc",
        "x86_avx512vnni_uint8_int8_int32_k16.inc",
    ],
)

cc_library(
    name = "schedule",
    srcs = ["schedule.cc"],
    hdrs = ["schedule.h"],
    compatible_with = _COMPATIBLE_WITH,
    visibility = ["//ynnpack:__subpackages__"],
    deps = ["//ynnpack/base"],
)

cc_library(
    name = "pack",
    srcs = ["pack.cc"],
    hdrs = ["pack.h"],
    compatible_with = _COMPATIBLE_WITH,
    visibility = ["//ynnpack:__subpackages__"],
    deps = [
        "//ynnpack/base",
        "//ynnpack/kernels/transpose",
    ],
)

cc_library(
    name = "arm64_sme_stubs",
    srcs = ["arm64_sme_stubs.cc"],
    compatible_with = _COMPATIBLE_WITH,
    deps = ["//ynnpack/base"],
)

ynn_cc_library(
    name = "dot",
    srcs = ["dot.cc"],
    hdrs = [
        "arm64_neon_bf16_bf16_fp32.inc",
        "arm64_neon_fp32.inc",
        "arm64_neon_fp64.inc",
        "arm64_neoni8mm_int8_int8_int32.inc",
        "arm64_sme.h",
        "arm_neon_int8_int8_int32.inc",
        "arm_neonbf16_bf16_bf16_fp32_k2.inc",
        "arm_neonbf16_bf16_bf16_fp32_k4.inc",
        "arm_neondot_int8_int8_int32.inc",
        "dot.h",
        "kernels.inc",
        "x86_avx2_fma3_bf16_bf16_fp32.inc",
        "x86_avx2_fma3_bf16_bf16_fp32_k1.inc",
        "x86_avx2_fma3_fp32_k2.inc",
        "x86_avx2_fp32_k2.inc",
        "x86_avx2_int8_int8_int32.inc",
        "x86_avx2_int8_int8_int32_k1.inc",
        "x86_avx512_bf16_bf16_fp32.inc",
        "x86_avx512_bf16_bf16_fp32_k1.inc",
        "x86_avx512_fp16_fp16_fp32.inc",
        "x86_avx512_fp32.inc",
        "x86_avx512_fp32_k2.inc",
        "x86_avx512_fp32_k4.inc",
        "x86_avx512_fp64.inc",
        "x86_avx512_int8_int8_int32.inc",
        "x86_avx512_int8_int8_int32_k1.inc",
        "x86_avx512_int8_int8_int32_k16.inc",
        "x86_avx512bf16_bf16_bf16_fp32.inc",
        "x86_avx512vnni_uint8_int8_int32.inc",
        "x86_avx512vnni_uint8_int8_int32_k16.inc",
        "x86_avx_fp32.inc",
        "x86_avx_fp64.inc",
        "x86_f16c_fma3_fp16_fp16_fp32.inc",
        "x86_f16c_fp16_fp16_fp32.inc",
        "x86_fma3_fp32.inc",
        "x86_fma3_fp64.inc",
        "x86_sse2_fp32.inc",
    ],
    compatible_with = _COMPATIBLE_WITH,
    copts = ynn_kernel_copts(unroll_loops = False),
    per_arch_srcs = {
        "arm64_neon": [
            "arm64_neon_fp32.cc",
            "arm64_neon_fp64.cc",
            "arm64_neon_bf16_bf16_fp32.cc",
        ],
        "arm64_neoni8mm": ["arm64_neoni8mm_int8_int8_int32.cc"],
        "arm64_sme": [
            "arm64_sme.cc",
            "arm64_sme_internal.h",
        ],
        "arm64_sme2": [
            "arm64_sme2.cc",
            "arm64_sme_internal.h",
        ],
        "arm_neon": ["arm_neon_int8_int8_int32.cc"],
        "arm_neonbf16": [
            "arm_neonbf16_bf16_bf16_fp32_k2.cc",
            "arm_neonbf16_bf16_bf16_fp32_k4.cc",
        ],
        "arm_neondot": ["arm_neondot_int8_int8_int32.cc"],
        "x86_sse2": ["x86_sse2_fp32.cc"],
        "x86_avx": [
            "x86_avx_fp32.cc",
            "x86_avx_fp64.cc",
        ],
        "x86_f16c": ["x86_f16c_fp16_fp16_fp32.cc"],
        "x86_f16c_fma3": ["x86_f16c_fma3_fp16_fp16_fp32.cc"],
        "x86_fma3": [
            "x86_fma3_fp32.cc",
            "x86_fma3_fp64.cc",
        ],
        "x86_avx2": [
            "x86_avx2_int8_int8_int32.cc",
            "x86_avx2_int8_int8_int32_k1.cc",
            "x86_avx2_fp32_k2.cc",
        ],
        "x86_avx2_fma3": [
            "x86_avx2_fma3_bf16_bf16_fp32.cc",
            "x86_avx2_fma3_bf16_bf16_fp32_k1.cc",
            "x86_avx2_fma3_fp32_k2.cc",
        ],
        "x86_avx512": [
            "x86_avx512_bf16_bf16_fp32.cc",
            "x86_avx512_bf16_bf16_fp32_k1.cc",
            "x86_avx512_fp16_fp16_fp32.cc",
            "x86_avx512_fp32.cc",
            "x86_avx512_fp64.cc",
            "x86_avx512_fp32_k2.cc",
            "x86_avx512_fp32_k4.cc",
            "x86_avx512_int8_int8_int32.cc",
            "x86_avx512_int8_int8_int32_k1.cc",
            "x86_avx512_int8_int8_int32_k16.cc",
        ],
        "x86_avx512bf16": ["x86_avx512bf16_bf16_bf16_fp32.cc"],
        "x86_avx512vnni": [
            "x86_avx512vnni_uint8_int8_int32.cc",
            "x86_avx512vnni_uint8_int8_int32_k16.cc",
        ],
        "x86_amxbf16": [
            "x86_amxbf16.cc",
            "x86_amx.h",
        ],
        "x86_amxfp16": [
            "x86_amxfp16.cc",
            "x86_amx.h",
        ],
        "x86_amxint8": [
            "x86_amxint8.cc",
            "x86_amx.h",
        ],
    },
    visibility = ["//ynnpack:__subpackages__"],
    deps = [
        "//ynnpack:ynnpack_h",
        "//ynnpack/base",
        "//ynnpack/base:log",
    ] + select({
        "//ynnpack:arm64": [":arm64_sme_stubs"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "pack_test_tensor",
    testonly = True,
    hdrs = ["pack_test_tensor.h"],
    deps = [
        ":pack",
        "//ynnpack/base",
        "//ynnpack/base/test:buffer",
        "//ynnpack/base/test:tensor",
    ],
)

cc_test(
    name = "test",
    srcs = ["test.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":dot",
        ":pack",
        ":pack_test_tensor",
        "//ynnpack:ynnpack_h",
        "//ynnpack/base",
        "//ynnpack/base/test:buffer",
        "//ynnpack/base/test:fuzz_test",
        "//ynnpack/base/test:random",
        "//ynnpack/base/test:tensor",
        "//ynnpack/base/test:util",
    ] + ynn_test_deps(),
)

cc_test(
    name = "consistent_arithmetic_test",
    srcs = ["consistent_arithmetic_test.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":dot",
        ":pack_test_tensor",
        "//ynnpack/base",
        "//ynnpack/base/test:buffer",
        "//ynnpack/base/test:fuzz_test",
        "//ynnpack/base/test:random",
        "//ynnpack/base/test:tensor",
        "//ynnpack/base/test:util",
    ] + ynn_test_deps(),
)

cc_test(
    name = "schedule_test",
    srcs = ["schedule_test.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":schedule",
    ] + ynn_test_deps(),
)

cc_test(
    name = "pack_test",
    srcs = ["pack_test.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":pack",
        "//ynnpack/base",
    ] + ynn_test_deps(),
)

cc_test(
    name = "get_dot_kernel_test",
    srcs = ["get_dot_kernel_test.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":dot",
        "//ynnpack:ynnpack_h",
        "//ynnpack/base",
    ] + ynn_test_deps(),
)

cc_test(
    name = "bench",
    srcs = ["bench.cc"],
    args = ["--benchmark_min_time=1x"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":dot",
        "//ynnpack/base",
        "//ynnpack/base/test:buffer",
        "//ynnpack/base/test:tensor",
        "@com_google_benchmark//:benchmark",
    ],
)

cc_binary(
    name = "schedule_bench",
    testonly = True,
    srcs = ["schedule_bench.cc"],
    linkopts = ynn_binary_linkopts(),
    malloc = ynn_binary_malloc(),
    deps = [
        ":dot",
        ":schedule",
        "//ynnpack/base",
        "//ynnpack/base/test:buffer",
        "//ynnpack/base/test:tensor",
        "//ynnpack/base/test:util",
    ],
)

sh_test(
    name = "schedule_bench_test",
    srcs = ["schedule_bench_test.sh"],
    args = ["$(rootpath :schedule_bench)"],
    data = [":schedule_bench"],
    # This doesn't work with emulators that use --run_under to execute the test.
    tags = ["noemulator"],
)
