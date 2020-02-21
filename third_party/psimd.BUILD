# Description:
#   Portable 128-bit SIMD intrinsics

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "psimd",
    hdrs = glob(["include/psimd.h"]),
    includes = ["include"],
    defines = select({
        ":psimd_enable_wasm_qfma_explicit_true": ["PSIMD_ENABLE_WASM_QFMA=1"],
        ":psimd_enable_wasm_qfma_explicit_false": ["PSIMD_ENABLE_WASM_QFMA=0"],
        "//conditions:default": ["PSIMD_ENABLE_WASM_QFMA=0"],
    }),
    strip_include_prefix = "include",
)

# Enables usage of QFMA WAsm SIMD instructions.
config_setting(
    name = "psimd_enable_wasm_qfma_explicit_true",
    define_values = {"psimd_enable_wasm_qfma": "true"},
)

# Disables usage of QFMA WAsm SIMD instructions.
config_setting(
    name = "psimd_enable_wasm_qfma_explicit_false",
    define_values = {"psimd_enable_wasm_qfma": "false"},
)
