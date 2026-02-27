# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//ynnpack:build_defs.bzl", "ynn_binary_linkopts", "ynn_binary_malloc", "ynn_cc_library", "ynn_generate_src_hdr", "ynn_kernel_copts", "ynn_test_deps")

_COMPATIBLE_WITH = []

ynn_generate_src_hdr(
    name = "x86_sse2_fp32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "x86_sse2_fp32",
        "dot,1x16x1",
        "dot,2x16x1",
        "dot,3x16x1",
        "dot,2x8x1",
        "dot,3x8x1",
        "dot,4x8x1",
        "dot,4x4x1",
        "dot,6x4x1",
        "dot,8x4x1",
    ],
    output_hdr = "x86_sse2_fp32.inc",
    output_src = "x86_sse2_fp32.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx_fp32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "x86_avx_fp32",
        "dot,1x32x1",
        "dot,2x32x1",
        "dot,2x16x1",
        "dot,3x16x1",
        "dot,4x16x1",
        "dot,4x8x1",
        "dot,6x8x1",
        "dot,8x8x1",
    ],
    output_hdr = "x86_avx_fp32.inc",
    output_src = "x86_avx_fp32.cc",
)

ynn_generate_src_hdr(
    name = "x86_fma3_fp32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "x86_fma3_fp32",
        "dot,1x32x1",
        "dot,2x32x1",
        "dot,1x16x1",  # This is needed to avoid using sse2 for n <= 16, to avoid mul+add numerics.
        "dot,2x16x1",
        "dot,3x16x1",
        "dot,4x16x1",
        "dot,5x16x1",
        "dot,6x16x1",
        "dot,8x8x1",
        # There doesn't seem to be anything wrong with this kernel, but for some shapes on AMD Rome,
        # it is super slow, e.g. 128x8x16384 is >20x slower than 8x8. It also doesn't seem like it
        # should be that much better than 8x8 when it is working well.
        # "dot,10x8x1",
    ],
    output_hdr = "x86_fma3_fp32.inc",
    output_src = "x86_fma3_fp32.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx2_fp32_k2",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "x86_avx2_fp32_k2",
        "dot,1x16x2",
        "dot,2x16x2",
        "dot,3x8x2",
        "dot,4x8x2",
        "dot,5x8x2",
        "dot,6x8x2",
        "dot,4x4x2",
        "dot,5x4x2",
        "dot,6x4x2",
        "dot,8x4x2",
    ],
    output_hdr = "x86_avx2_fp32_k2.inc",
    output_src = "x86_avx2_fp32_k2.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx2_fma3_fp32_k2",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "x86_avx2_fma3_fp32_k2",
        "dot,1x16x2",
        "dot,2x16x2",
        "dot,3x8x2",
        "dot,4x8x2",
        "dot,5x8x2",
        "dot,6x8x2",
        "dot,4x4x2",
        "dot,5x4x2",
        "dot,6x4x2",
        "dot,8x4x2",
    ],
    output_hdr = "x86_avx2_fma3_fp32_k2.inc",
    output_src = "x86_avx2_fma3_fp32_k2.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx512_fp32_k2",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "x86_avx512_fp32_k2",
        "dot,1x32x2",
        "dot,2x32x2",
        "dot,3x32x2",
        "dot,4x32x2",
        "dot,5x32x2",
        # The kernels which are commented out should be good, but for some
        # reason they don't perform well. They don't seem to spill, so keeping them
        # until we understand why are they slower.
        # "dot,6x32x2",
        "dot,4x16x2",
        # "dot,5x16x2",
        # "dot,6x16x2",
        # "dot,8x16x2",
        # "dot,10x16x2",
        "dot,4x8x2",
        "dot,8x8x2",
        # "dot,12x8x2",
        # "dot,16x8x2",
    ],
    output_hdr = "x86_avx512_fp32_k2.inc",
    output_src = "x86_avx512_fp32_k2.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx2_fma3_bf16_bf16_fp32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "x86_avx2_fma3_bf16_bf16_fp32",
        "dot,1x32x2",
        "dot,2x32x2",
        "dot,2x16x2",
        "dot,3x16x2",
        "dot,4x16x2",
        "dot,5x16x2",
        "dot,8x8x2",
        "dot,10x8x2",
        "dot,12x8x2",
    ],
    output_hdr = "x86_avx2_fma3_bf16_bf16_fp32.inc",
    output_src = "x86_avx2_fma3_bf16_bf16_fp32.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx2_fma3_bf16_bf16_fp32_k1",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "x86_avx2_fma3_bf16_bf16_fp32_k1",
        "dot,1x32x1",
        "dot,2x32x1",
        "dot,2x16x1",
        "dot,3x16x1",
        "dot,4x16x1",
        "dot,5x16x1",
        "dot,8x8x1",
        "dot,10x8x1",
    ],
    output_hdr = "x86_avx2_fma3_bf16_bf16_fp32_k1.inc",
    output_src = "x86_avx2_fma3_bf16_bf16_fp32_k1.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx512_bf16_bf16_fp32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "x86_avx512_bf16_bf16_fp32",
        "dot,1x64x2",
        "dot,2x64x2",
        "dot,3x64x2",
        "dot,4x64x2",
        "dot,5x64x2",
        "dot,2x32x2",
        "dot,3x32x2",
        "dot,4x32x2",
        "dot,5x32x2",
        "dot,6x32x2",
        "dot,8x32x2",
        "dot,10x32x2",
        "dot,12x32x2",
        "dot,16x16x2",
    ],
    output_hdr = "x86_avx512_bf16_bf16_fp32.inc",
    output_src = "x86_avx512_bf16_bf16_fp32.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx512_bf16_bf16_fp32_k1",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "x86_avx512_bf16_bf16_fp32_k1",
        "dot,1x64x1",
        "dot,2x64x1",
        "dot,3x64x1",
        "dot,4x64x1",
        "dot,5x64x1",
        "dot,2x32x1",
        "dot,3x32x1",
        "dot,4x32x1",
        "dot,5x32x1",
        "dot,6x32x1",
        "dot,8x32x1",
        "dot,10x32x1",
        "dot,12x32x1",
        "dot,16x16x1",
    ],
    output_hdr = "x86_avx512_bf16_bf16_fp32_k1.inc",
    output_src = "x86_avx512_bf16_bf16_fp32_k1.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx512_fp32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "x86_avx512_fp32",
        "dot,1x64x1",
        "dot,2x64x1",
        "dot,3x64x1",
        "dot,4x64x1",
        "dot,5x64x1",
        "dot,2x32x1",
        "dot,3x32x1",
        "dot,4x32x1",
        "dot,5x32x1",
        # The kernels which are commented out should be good, but for some
        # reason they don't perform well. They don't seem to spill, so keeping them
        # until we understand why are they slower.
        # "dot,6x32x1",
        # "dot,8x32x1",
        # "dot,10x32x1",
        # "dot,12x32x1",
        "dot,5x16x1",
        # "dot,16x16x1",
    ],
    output_hdr = "x86_avx512_fp32.inc",
    output_src = "x86_avx512_fp32.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx512_fp32_k4",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "x86_avx512_fp32_k4",
        "dot,1x32x4",
        "dot,2x32x4",
        "dot,3x16x4",
        "dot,4x16x4",
        "dot,5x16x4",
        "dot,2x8x4",
        "dot,4x8x4",
        "dot,6x8x4",
        "dot,4x4x4",
        "dot,8x4x4",
    ],
    output_hdr = "x86_avx512_fp32_k4.inc",
    output_src = "x86_avx512_fp32_k4.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx512bf16_bf16_bf16_fp32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "x86_avx512bf16_bf16_bf16_fp32",
        "dot,1x64x2",
        "dot,2x64x2",
        "dot,3x64x2",
        "dot,4x64x2",
        "dot,5x64x2",
        "dot,2x32x2",
        "dot,3x32x2",
        "dot,4x32x2",
        "dot,5x32x2",
        "dot,6x32x2",
        "dot,8x32x2",
        "dot,10x32x2",
        "dot,12x32x2",
        "dot,16x16x2",  # This kernel is required by the AMX to handle odd tails.
    ],
    output_hdr = "x86_avx512bf16_bf16_bf16_fp32.inc",
    output_src = "x86_avx512bf16_bf16_bf16_fp32.cc",
)

ynn_generate_src_hdr(
    name = "x86_f16c_fp16_fp16_fp32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "x86_f16c_fp16_fp16_fp32",
        "dot,1x32x1",
        "dot,2x32x1",
        "dot,2x16x1",
        "dot,3x16x1",
        "dot,4x16x1",
        "dot,4x8x1",
        "dot,6x8x1",
        "dot,8x8x1",
    ],
    output_hdr = "x86_f16c_fp16_fp16_fp32.inc",
    output_src = "x86_f16c_fp16_fp16_fp32.cc",
)

ynn_generate_src_hdr(
    name = "x86_f16c_fma3_fp16_fp16_fp32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "x86_f16c_fma3_fp16_fp16_fp32",
        "dot,1x32x1",
        "dot,2x32x1",
        "dot,1x16x1",
        "dot,2x16x1",
        "dot,4x16x1",
        "dot,2x8x1",
        "dot,4x8x1",
        "dot,8x8x1",
    ],
    output_hdr = "x86_f16c_fma3_fp16_fp16_fp32.inc",
    output_src = "x86_f16c_fma3_fp16_fp16_fp32.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx512_fp16_fp16_fp32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "x86_avx512_fp16_fp16_fp32",
        "dot,1x64x1",
        "dot,2x64x1",
        "dot,3x64x1",
        "dot,4x64x1",
        "dot,5x64x1",
        "dot,2x32x1",
        "dot,3x32x1",
        "dot,4x32x1",
        "dot,5x32x1",
        # The kernels which are commented out should be good, but for some
        # reason they don't perform well. They don't seem to spill, so keeping them
        # until we understand why are they slower.
        # "dot,6x32x1",
        # "dot,8x32x1",
        # "dot,10x32x1",
        # "dot,12x32x1",
        "dot,5x16x1",
        # "dot,16x16x1",
    ],
    output_hdr = "x86_avx512_fp16_fp16_fp32.inc",
    output_src = "x86_avx512_fp16_fp16_fp32.cc",
)

ynn_generate_src_hdr(
    name = "arm64_neon_fp32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "arm64_neon_fp32",
        "dot,1x32x4",
        "dot,2x32x4",
        "dot,3x32x4",
        "dot,2x16x4",
        "dot,3x16x4",
        "dot,4x16x4",
        "dot,5x16x4",
        "dot,4x8x4",
        "dot,6x8x4",
        "dot,8x8x4",
        "dot,8x4x4",
    ],
    output_hdr = "arm64_neon_fp32.inc",
    output_src = "arm64_neon_fp32.cc",
)

ynn_generate_src_hdr(
    name = "arm64_neon_bf16_bf16_fp32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "arm64_neon_bf16_bf16_fp32",
        "dot,1x32x4",
        "dot,2x32x4",
        "dot,3x32x4",
        "dot,2x16x4",
        "dot,3x16x4",
        "dot,4x16x4",
        "dot,5x16x4",
        "dot,4x8x4",
        "dot,6x8x4",
        "dot,8x8x4",
        "dot,8x4x4",
    ],
    output_hdr = "arm64_neon_bf16_bf16_fp32.inc",
    output_src = "arm64_neon_bf16_bf16_fp32.cc",
)

ynn_generate_src_hdr(
    name = "arm_neonbf16_bf16_bf16_fp32_k2",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "arm_neonbf16_bf16_bf16_fp32_k2",
        # TODO: https://github.com/llvm/llvm-project/issues/170883, the tile sizes could probably be
        # improved after indexed bfdot bugs are fixed.
        "dot,1x32x4",
        "dot,2x32x4",
        "dot,2x16x4",
        "dot,3x16x4",
        "dot,4x16x4",
        "dot,4x8x4",
        "dot,6x8x4",
        "dot,8x4x4",
    ],
    output_hdr = "arm_neonbf16_bf16_bf16_fp32_k2.inc",
    output_src = "arm_neonbf16_bf16_bf16_fp32_k2.cc",
)

ynn_generate_src_hdr(
    name = "arm_neon_int8_int8_int32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "arm_neon_int8_int8_int32",
        "dot,1x16x8",
        "dot,2x16x8",
        "dot,3x16x8",
        "dot,4x16x8",
        "dot,2x8x8",
        "dot,3x8x8",
        "dot,4x8x8",
        "dot,5x8x8",
        "dot,6x8x8",
        "dot,8x8x8",
        "dot,8x4x8",
    ],
    output_hdr = "arm_neon_int8_int8_int32.inc",
    output_src = "arm_neon_int8_int8_int32.cc",
)

# TODO: We can do an arm64-only version of this that unrolls k by 16.
ynn_generate_src_hdr(
    name = "arm_neondot_int8_int8_int32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "arm_neondot_int8_int8_int32",
        "dot,1x32x8",
        "dot,2x32x8",
        "dot,3x32x8",
        "dot,2x16x8",
        "dot,3x16x8",
        "dot,4x16x8",
        "dot,5x16x8",
        "dot,8x8x8",
        "dot,10x8x8",
        "dot,8x4x8",
    ],
    output_hdr = "arm_neondot_int8_int8_int32.inc",
    output_src = "arm_neondot_int8_int8_int32.cc",
)

ynn_generate_src_hdr(
    name = "arm64_neoni8mm_int8_int8_int32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "arm_neoni8mm_int8_int8_int32",
        "dot,2x32x8",
        "dot,4x16x8",
        "dot,6x16x8",
        "dot,6x8x8",
        "dot,8x8x8",
        "dot,10x8x8",
        "dot,16x4x8",
    ],
    output_hdr = "arm64_neoni8mm_int8_int8_int32.inc",
    output_src = "arm64_neoni8mm_int8_int8_int32.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx2_int8_int8_int32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "x86_avx2_int8_int8_int32",
        "dot,1x32x4",
        "dot,2x16x4",
        "dot,3x8x4",
        "dot,4x8x4",
        "dot,6x8x4",
    ],
    output_hdr = "x86_avx2_int8_int8_int32.inc",
    output_src = "x86_avx2_int8_int8_int32.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx2_int8_int8_int32_k1",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "x86_avx2_int8_int8_int32_k1",
        "dot,1x32x1",
        "dot,2x32x1",
        "dot,3x16x1",
        "dot,4x16x1",
        "dot,5x16x1",
        "dot,8x8x1",
    ],
    output_hdr = "x86_avx2_int8_int8_int32_k1.inc",
    output_src = "x86_avx2_int8_int8_int32_k1.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx512_int8_int8_int32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "x86_avx512_int8_int8_int32",
        "dot,1x64x4",
        "dot,2x64x4",
        "dot,3x32x4",
        "dot,4x32x4",
        "dot,5x32x4",
        "dot,6x32x4",
        "dot,8x16x4",
        "dot,12x16x4",
    ],
    output_hdr = "x86_avx512_int8_int8_int32.inc",
    output_src = "x86_avx512_int8_int8_int32.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx512_int8_int8_int32_k1",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "x86_avx512_int8_int8_int32_k1",
        "dot,1x64x1",
        "dot,2x64x1",
        "dot,3x64x1",
        "dot,4x32x1",
        "dot,5x32x1",
        "dot,6x32x1",
        "dot,8x32x1",
        "dot,16x16x1",
    ],
    output_hdr = "x86_avx512_int8_int8_int32_k1.inc",
    output_src = "x86_avx512_int8_int8_int32_k1.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx512_int8_int8_int32_k16",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "x86_avx512_int8_int8_int32_k16",
        "dot,2x16x16",
        "dot,2x8x16",
        "dot,4x8x16",
        "dot,6x8x16",
        "dot,2x4x16",
        "dot,4x4x16",
        "dot,6x4x16",
        "dot,8x4x16",
    ],
    output_hdr = "x86_avx512_int8_int8_int32_k16.inc",
    output_src = "x86_avx512_int8_int8_int32_k16.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx512vnni_uint8_int8_int32",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "x86_avx512vnni_uint8_int8_int32",
        "dot,1x64x4",
        "dot,2x64x4",
        "dot,3x64x4",
        "dot,4x64x4",
        "dot,5x64x4",
        "dot,1x32x4",
        "dot,2x32x4",
        "dot,3x32x4",
        "dot,4x32x4",
        "dot,5x32x4",
        "dot,6x32x4",
        "dot,8x32x4",
        "dot,10x32x4",
        "dot,16x16x4",
    ],
    output_hdr = "x86_avx512vnni_uint8_int8_int32.inc",
    output_src = "x86_avx512vnni_uint8_int8_int32.cc",
)

ynn_generate_src_hdr(
    name = "x86_avx512vnni_uint8_int8_int32_k16",
    compatible_with = _COMPATIBLE_WITH,
    generator = "//ynnpack/kernels/dot/generator",
    generator_args = [
        "x86_avx512vnni_uint8_int8_int32_k16",
        "dot,4x8x16",
        "dot,8x8x16",
        "dot,4x4x16",
        "dot,8x4x16",
    ],
    output_hdr = "x86_avx512vnni_uint8_int8_int32_k16.inc",
    output_src = "x86_avx512vnni_uint8_int8_int32_k16.cc",
)

cc_library(
    name = "schedule",
    srcs = ["schedule.cc"],
    hdrs = ["schedule.h"],
    compatible_with = _COMPATIBLE_WITH,
    visibility = ["//ynnpack:__subpackages__"],
    deps = [
        "//ynnpack/base",
        "@slinky//slinky/base",
    ],
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
        "arm64_neoni8mm_int8_int8_int32.inc",
        "arm64_sme.h",
        "arm_neon_int8_int8_int32.inc",
        "arm_neonbf16_bf16_bf16_fp32_k2.inc",
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
        "x86_avx512_int8_int8_int32.inc",
        "x86_avx512_int8_int8_int32_k1.inc",
        "x86_avx512_int8_int8_int32_k16.inc",
        "x86_avx512bf16_bf16_bf16_fp32.inc",
        "x86_avx512vnni_uint8_int8_int32.inc",
        "x86_avx512vnni_uint8_int8_int32_k16.inc",
        "x86_avx_fp32.inc",
        "x86_f16c_fma3_fp16_fp16_fp32.inc",
        "x86_f16c_fp16_fp16_fp32.inc",
        "x86_fma3_fp32.inc",
        "x86_sse2_fp32.inc",
    ],
    compatible_with = _COMPATIBLE_WITH,
    copts = ynn_kernel_copts(unroll_loops = False),
    per_arch_srcs = {
        "arm64_neon": [
            "arm64_neon_fp32.cc",
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
        "arm_neonbf16": ["arm_neonbf16_bf16_bf16_fp32_k2.cc"],
        "arm_neondot": ["arm_neondot_int8_int8_int32.cc"],
        "x86_sse2": ["x86_sse2_fp32.cc"],
        "x86_avx": ["x86_avx_fp32.cc"],
        "x86_f16c": ["x86_f16c_fp16_fp16_fp32.cc"],
        "x86_f16c_fma3": ["x86_f16c_fma3_fp16_fp16_fp32.cc"],
        "x86_fma3": ["x86_fma3_fp32.cc"],
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
