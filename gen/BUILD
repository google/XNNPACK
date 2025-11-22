# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description:
#   XNNPACK - optimized floating-point neural network operators library

load("//tools/build_defs/testing:bzl_library.bzl", "bzl_library")

filegroup(
    name = "bzl_microkernel_lists",
    srcs = glob([
        "**/*.bzl",
    ]),
    visibility = ["//:__subpackages__"],
)

bzl_library(
    name = "microkernels_bzl",
    srcs = ["microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
    deps = [
        ":aarch32_microkernels_bzl",
        ":aarch64_microkernels_bzl",
        ":amd64_microkernels_bzl",
        ":armsimd32_microkernels_bzl",
        ":avx256skx_microkernels_bzl",
        ":avx256vnni_microkernels_bzl",
        ":avx256vnnigfni_microkernels_bzl",
        ":avx2_microkernels_bzl",
        ":avx512amx_microkernels_bzl",
        ":avx512f_microkernels_bzl",
        ":avx512fp16_microkernels_bzl",
        ":avx512skx_microkernels_bzl",
        ":avx512vbmi_microkernels_bzl",
        ":avx512vnni_microkernels_bzl",
        ":avx512vnnigfni_microkernels_bzl",
        ":avx_microkernels_bzl",
        ":avxvnni_microkernels_bzl",
        ":avxvnniint8_microkernels_bzl",
        ":f16c_microkernels_bzl",
        ":fma3_microkernels_bzl",
        ":fp16arith_microkernels_bzl",
        ":hexagon_microkernels_bzl",
        ":hvx_microkernels_bzl",
        ":neon_aarch64_microkernels_bzl",
        ":neon_microkernels_bzl",
        ":neonbf16_microkernels_bzl",
        ":neondot_aarch64_microkernels_bzl",
        ":neondot_microkernels_bzl",
        ":neondotfp16arith_microkernels_bzl",
        ":neonfma_aarch64_microkernels_bzl",
        ":neonfma_microkernels_bzl",
        ":neonfp16_microkernels_bzl",
        ":neonfp16arith_aarch64_microkernels_bzl",
        ":neonfp16arith_microkernels_bzl",
        ":neoni8mm_microkernels_bzl",
        ":neonsme2_microkernels_bzl",
        ":neonsme_microkernels_bzl",
        ":neonv8_microkernels_bzl",
        ":rvv_microkernels_bzl",
        ":rvvfp16arith_microkernels_bzl",
        ":scalar_microkernels_bzl",
        ":sse2_microkernels_bzl",
        ":sse2fma_microkernels_bzl",
        ":sse41_microkernels_bzl",
        ":sse_microkernels_bzl",
        ":ssse3_microkernels_bzl",
        ":wasm32_microkernels_bzl",
        ":wasmrelaxedsimd32_microkernels_bzl",
        ":wasmrelaxedsimd_microkernels_bzl",
        ":wasmsimd32_microkernels_bzl",
        ":wasmsimd_microkernels_bzl",
    ],
)

bzl_library(
    name = "avx512vnni_microkernels_bzl",
    srcs = ["avx512vnni_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "neonv8_microkernels_bzl",
    srcs = ["neonv8_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "neonbf16_microkernels_bzl",
    srcs = ["neonbf16_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "neonsme_microkernels_bzl",
    srcs = ["neonsme_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "rvvfp16arith_microkernels_bzl",
    srcs = ["rvvfp16arith_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "rvv_microkernels_bzl",
    srcs = ["rvv_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "avx512vnnigfni_microkernels_bzl",
    srcs = ["avx512vnnigfni_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "aarch64_microkernels_bzl",
    srcs = ["aarch64_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "avxvnni_microkernels_bzl",
    srcs = ["avxvnni_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "scalar_microkernels_bzl",
    srcs = ["scalar_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "amd64_microkernels_bzl",
    srcs = ["amd64_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "sse2_microkernels_bzl",
    srcs = ["sse2_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "sse41_microkernels_bzl",
    srcs = ["sse41_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "fp16arith_microkernels_bzl",
    srcs = ["fp16arith_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "neoni8mm_microkernels_bzl",
    srcs = ["neoni8mm_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "hvx_microkernels_bzl",
    srcs = ["hvx_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "fma3_microkernels_bzl",
    srcs = ["fma3_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "wasmrelaxedsimd_microkernels_bzl",
    srcs = ["wasmrelaxedsimd_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "avx512f_microkernels_bzl",
    srcs = ["avx512f_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "avxvnniint8_microkernels_bzl",
    srcs = ["avxvnniint8_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "avx512fp16_microkernels_bzl",
    srcs = ["avx512fp16_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "neonfma_aarch64_microkernels_bzl",
    srcs = ["neonfma_aarch64_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "wasmrelaxedsimd32_microkernels_bzl",
    srcs = ["wasmrelaxedsimd32_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "wasmsimd32_microkernels_bzl",
    srcs = ["wasmsimd32_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "avx_microkernels_bzl",
    srcs = ["avx_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "neonsme2_microkernels_bzl",
    srcs = ["neonsme2_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "avx512amx_microkernels_bzl",
    srcs = ["avx512amx_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "ssse3_microkernels_bzl",
    srcs = ["ssse3_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "avx256vnni_microkernels_bzl",
    srcs = ["avx256vnni_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "avx512vbmi_microkernels_bzl",
    srcs = ["avx512vbmi_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "avx512skx_microkernels_bzl",
    srcs = ["avx512skx_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "aarch32_microkernels_bzl",
    srcs = ["aarch32_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "neonfp16arith_aarch64_microkernels_bzl",
    srcs = ["neonfp16arith_aarch64_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "f16c_microkernels_bzl",
    srcs = ["f16c_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "neonfp16_microkernels_bzl",
    srcs = ["neonfp16_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "wasm32_microkernels_bzl",
    srcs = ["wasm32_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "avx2_microkernels_bzl",
    srcs = ["avx2_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "neon_microkernels_bzl",
    srcs = ["neon_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "neon_aarch64_microkernels_bzl",
    srcs = ["neon_aarch64_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "neondot_aarch64_microkernels_bzl",
    srcs = ["neondot_aarch64_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "neondotfp16arith_microkernels_bzl",
    srcs = ["neondotfp16arith_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "wasmsimd_microkernels_bzl",
    srcs = ["wasmsimd_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "armsimd32_microkernels_bzl",
    srcs = ["armsimd32_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "hexagon_microkernels_bzl",
    srcs = ["hexagon_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "avx256skx_microkernels_bzl",
    srcs = ["avx256skx_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "sse2fma_microkernels_bzl",
    srcs = ["sse2fma_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "sse_microkernels_bzl",
    srcs = ["sse_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "neonfma_microkernels_bzl",
    srcs = ["neonfma_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "neondot_microkernels_bzl",
    srcs = ["neondot_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "avx256vnnigfni_microkernels_bzl",
    srcs = ["avx256vnnigfni_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "neonfp16arith_microkernels_bzl",
    srcs = ["neonfp16arith_microkernels.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)
