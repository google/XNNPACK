"""Build definitions and rules for XNNPACK."""

load("//:emscripten.bzl", "xnnpack_emscripten_benchmark_linkopts", "xnnpack_emscripten_deps", "xnnpack_emscripten_minimal_linkopts", "xnnpack_emscripten_test_linkopts")

def xnnpack_visibility():
    """Visibility of :XNNPACK target.

    All other targets have private visibility, and can not have external
    dependencies.
    """
    return ["//visibility:public"]

def xnnpack_min_size_copts():
    """Compiler flags for size-optimized builds."""
    return ["-Os"]

def xnnpack_gcc_std_copts():
    """GCC-like compiler flags to specify language standard for C sources."""
    return ["-std=c99"]

def xnnpack_msvc_std_copts():
    """MSVC compiler flags to specify language standard for C sources."""
    return ["/Drestrict="]

def xnnpack_std_cxxopts():
    """Compiler flags to specify language standard for C++ sources."""
    return ["-std=gnu++14"]

def xnnpack_test_deps_for_library():
    """Depencies needed for a library to use gunit."""
    return ["@com_google_googletest//:gtest_main"]

def xnnpack_optional_ruy_copts():
    """Compiler flags to optionally enable Ruy benchmarks."""
    return []

def xnnpack_optional_gemmlowp_copts():
    """Compiler flags to optionally enable Gemmlowp benchmarks."""
    return []

def xnnpack_optional_tflite_copts():
    """Compiler flags to optionally enable TensorFlow Lite benchmarks."""
    return []

def xnnpack_optional_dnnl_copts():
    """Compiler flags to optionally enable Intel DNNL benchmarks."""
    return []

def xnnpack_optional_ruy_deps():
    """Optional Ruy dependencies."""
    return []

def xnnpack_optional_gemmlowp_deps():
    """Optional Gemmlowp dependencies."""
    return []

def xnnpack_optional_tflite_deps():
    """Optional TensorFlow Lite dependencies."""
    return []

def xnnpack_optional_dnnl_deps():
    """Optional Intel DNNL dependencies."""
    return []

def xnnpack_slinky_srcs():
    return []

def xnnpack_slinky_deps():
    return []

def xnnpack_slinky_defines():
    return []

def xnnpack_if_kleidiai_enabled(enabled = [], not_enabled = []):
    return select({
        "//:kleidiai_enabled": enabled,
        "//conditions:default": not_enabled,
    })

def xnnpack_kleidiai_defines():
    return xnnpack_if_kleidiai_enabled(
        enabled = ["XNN_ENABLE_KLEIDIAI=1"],
        not_enabled = ["XNN_ENABLE_KLEIDIAI=0"],
    )

_XNNPACK_ARCH_COPT_MAPPING = {
    "avx": select({
        "//build_config:x86": ["-mavx"],
        "//conditions:default": [],
    }),
    "avx2": select({
        "//build_config:x86": ["-mavx2"],
        "//conditions:default": [],
    }),
    "avx512bw": select({
        "//build_config:x86": ["-mavx512bw"],
        "//conditions:default": [],
    }),
    "avx512f": select({
        "//build_config:x86": ["-mavx512f"],
        "//conditions:default": [],
    }),
    "fma3": select({
        "//build_config:x86": ["-mfma"],
        "//conditions:default": [],
    }),
    "hvx": select({
        "//build_config:hexagon": ["-mhvx-ieee-fp"],
        "//conditions:default": [],
    }),
    "neon": select({
        "//build_config:aarch32": [
            "-marm",
            "-march=armv7-a",
            "-mfpu=neon",
        ],
        "//conditions:default": [],
    }),
    "scalar": [],
    "sse2": select({
        "//build_config:x86": ["-msse2"],
        "//conditions:default": [],
    }),
    "sse41": select({
        "//build_config:x86": ["-msse4.1"],
        "//conditions:default": [],
    }),
    "wasmsimd": [],
}

def xnnpack_simd_archs():
    return _XNNPACK_ARCH_COPT_MAPPING.keys()

def xnnpack_simd_f32_archs():
    return ["avx", "avx2", "avx512f", "fma3", "hvx", "neon", "scalar", "sse2", "wasmsimd"]

def xnnpack_simd_f16_archs():
    return ["scalar"]

def xnnpack_simd_s16_archs():
    return ["avx2", "avx512bw", "neon", "scalar", "sse41", "wasmsimd"]

def xnnpack_simd_s32_archs():
    return ["avx2", "avx512f", "neon", "scalar", "sse41", "wasmsimd"]

def xnnpack_simd_s8_archs():
    return ["scalar"]

def xnnpack_simd_copts_for_arch(arch):
    return _XNNPACK_ARCH_COPT_MAPPING[arch]

def xnnpack_cc_library(
        name,
        srcs = [],
        x86_srcs = [],
        aarch32_srcs = [],
        aarch64_srcs = [],
        hexagon_srcs = [],
        riscv_srcs = [],
        wasm_srcs = [],
        wasmsimd_srcs = [],
        wasmrelaxedsimd_srcs = [],
        linkopts = [],
        copts = [],
        gcc_copts = [],
        msvc_copts = [],
        mingw_copts = [],
        msys_copts = [],
        gcc_x86_copts = [],
        msvc_x86_32_copts = [],
        msvc_x86_64_copts = [],
        aarch32_copts = [],
        aarch64_copts = [],
        hexagon_copts = [],
        riscv_copts = [],
        wasm_copts = [],
        wasmsimd_copts = [],
        wasmrelaxedsimd_copts = [],
        optimized_copts = ["-O2"],
        hdrs = [],
        defines = [],
        includes = [],
        deps = [],
        visibility = [":__subpackages__"],
        testonly = False):
    """C/C++/assembly library with architecture-specific configuration.

    Define a static library with architecture- and instruction-specific
    source files and/or compiler flags.

    Args:
      name: The name of the library target to define.
      srcs: The list of architecture-independent source files.
      x86_srcs: The list of x86-specific source files.
      aarch32_srcs: The list of AArch32-specific source files.
      aarch64_srcs: The list of AArch64-specific source files.
      hexagon_srcs: The list of Hexagon-specific source files.
      riscv_srcs: The list of RISC-V-specific source files.
      wasm_srcs: The list of WebAssembly 1.0-specific source files.
      wasmsimd_srcs: The list of WebAssembly SIMD-specific source files.
      wasmrelaxedsimd_srcs: The list of WebAssembly Relaxed SIMD-specific
                            source files.
      copts: The list of compiler flags to use in all builds. -I flags for
             include/ and src/ directories of XNNPACK are always prepended
             before these user-specified flags.
      gcc_copts: The list of compiler flags to use with GCC-like compilers.
      msvc_copts: The list of compiler flags to use with MSVC compiler.
      mingw_copts: The list of compiler flags to use with MinGW GCC compilers.
      msys_copts: The list of compiler flags to use with MSYS (Cygwin) GCC
                  compilers.
      gcc_x86_copts: The list of GCC-like compiler flags to use in x86 (32-bit
                     and 64-bit) builds.
      msvc_x86_32_copts: The list of MSVC compiler flags to use in x86 (32-bit)
                         builds.
      msvc_x86_64_copts: The list of MSVC compiler flags to use in x86 (64-bit)
                         builds.
      aarch32_copts: The list of compiler flags to use in AArch32 builds.
      aarch64_copts: The list of compiler flags to use in AArch64 builds.
      hexagon_copts: The list of compiler flags to use in hexagon builds.
      riscv_copts: The list of compiler flags to use in RISC-V builds.
      wasm_copts: The list of compiler flags to use in WebAssembly 1.0 builds.
      wasmsimd_copts: The list of compiler flags to use in WebAssembly SIMD
                      builds.
      wasmrelaxedsimd_copts: The list of compiler flags to use in WebAssembly
                             Relaxed SIMD builds.
      optimized_copts: The list of compiler flags to use in optimized builds.
                       Defaults to -O2.
      hdrs: The list of header files published by this library to be textually
            included by sources in dependent rules.
      defines: List of predefines macros to be added to the compile line.
      includes: List of include dirs to be added to the compile line.
      deps: The list of other libraries to be linked.
      visibility: The list of packages that can depend on this target.
    """
    native.cc_library(
        name = name,
        srcs = srcs + select({
            "//build_config:aarch32": aarch32_srcs,
            "//build_config:aarch64": aarch64_srcs,
            "//build_config:riscv": riscv_srcs,
            "//build_config:x86": x86_srcs,
            "//build_config:emscripten_wasm": wasm_srcs,
            "//build_config:emscripten_wasmsimd": wasmsimd_srcs,
            "//build_config:emscripten_wasmrelaxedsimd": wasmrelaxedsimd_srcs,
            "//conditions:default": [],
        }),
        copts = [
            "-Iinclude",
            "-Isrc",
        ] + copts + select({
            "//build_config:linux_k8": gcc_x86_copts,
            "//build_config:linux_arm": aarch32_copts,
            "//build_config:linux_armeabi": aarch32_copts,
            "//build_config:linux_armhf": aarch32_copts,
            "//build_config:linux_armv7a": aarch32_copts,
            "//build_config:linux_arm64": aarch64_copts,
            "//build_config:macos_x86_64": gcc_x86_copts,
            "//build_config:macos_x86_64_legacy": gcc_x86_copts,
            "//build_config:macos_arm64": aarch64_copts,
            "//build_config:windows_x86_64_clang": ["/clang:" + opt for opt in gcc_x86_copts],
            "//build_config:windows_x86_64_mingw": mingw_copts + gcc_x86_copts,
            "//build_config:windows_x86_64_msys": msys_copts + gcc_x86_copts,
            "//build_config:windows_x86_64": msvc_x86_64_copts,
            "//build_config:android_armv7": aarch32_copts,
            "//build_config:android_arm64": aarch64_copts,
            "//build_config:android_x86": gcc_x86_copts,
            "//build_config:android_x86_64": gcc_x86_copts,
            "//build_config:ios_arm64": aarch64_copts,
            "//build_config:ios_arm64e": aarch64_copts,
            "//build_config:ios_sim_arm64": aarch64_copts,
            "//build_config:ios_x86_64": gcc_x86_copts,
            "//build_config:watchos_arm64_32": aarch64_copts,
            "//build_config:watchos_x86_64": gcc_x86_copts,
            "//build_config:tvos_arm64": aarch64_copts,
            "//build_config:tvos_x86_64": gcc_x86_copts,
            "//build_config:emscripten_wasm": wasm_copts,
            "//build_config:emscripten_wasmsimd": wasmsimd_copts,
            "//build_config:emscripten_wasmrelaxedsimd": wasmrelaxedsimd_copts,
            "//conditions:default": [],
        }) + select({
            "//build_config:windows_x86_64_clang": ["/clang:" + opt for opt in gcc_copts],
            "//build_config:windows_x86_64_mingw": gcc_copts,
            "//build_config:windows_x86_64_msys": gcc_copts,
            "//build_config:windows_x86_64": msvc_copts,
            "//conditions:default": gcc_copts,
        }) + select({
            "//:optimized_build": optimized_copts,
            "//conditions:default": [],
        }),
        defines = defines,
        deps = deps,
        includes = ["include", "src"] + includes,
        linkstatic = True,
        linkopts = select({
            "//build_config:linux_k8": ["-lpthread"],
            "//build_config:linux_arm": ["-lpthread"],
            "//build_config:linux_armeabi": ["-lpthread"],
            "//build_config:linux_armhf": ["-lpthread"],
            "//build_config:linux_armv7a": ["-lpthread"],
            "//build_config:linux_arm64": ["-lpthread"],
            "//build_config:android": ["-lm"],
            "//conditions:default": [],
        }),
        textual_hdrs = hdrs,
        visibility = visibility,
        testonly = testonly,
    )

def xnnpack_microkernel_cc_library(
        name,
        srcs_prod = [],
        srcs_test = [],
        x86_srcs_prod = [],
        x86_srcs_test = [],
        aarch32_srcs_prod = [],
        aarch32_srcs_test = [],
        aarch64_srcs_prod = [],
        aarch64_srcs_test = [],
        hexagon_srcs_prod = [],
        hexagon_srcs_test = [],
        riscv_srcs_prod = [],
        riscv_srcs_test = [],
        wasm_srcs_prod = [],
        wasm_srcs_test = [],
        wasmsimd_srcs_prod = [],
        wasmsimd_srcs_test = [],
        wasmrelaxedsimd_srcs_prod = [],
        wasmrelaxedsimd_srcs_test = [],
        linkopts = [],
        copts = [],
        gcc_copts = [],
        msvc_copts = [],
        mingw_copts = [],  # buildifier: disable=unused-variable
        msys_copts = [],  # buildifier: disable=unused-variable
        gcc_x86_copts = [],
        msvc_x86_32_copts = [],
        msvc_x86_64_copts = [],
        aarch32_copts = [],
        aarch64_copts = [],
        hexagon_copts = [],
        riscv_copts = [],
        wasm_copts = [],
        wasmsimd_copts = [],
        wasmrelaxedsimd_copts = [],
        optimized_copts = ["-O2"],
        hdrs = [],
        defines = [],
        includes = [],
        deps = [],
        visibility = [":__subpackages__"],
        testonly = False):
    """C/C++/assembly library with architecture-specific configuration.

    A wrapper for xnnpack_cc_library called twice, once with prod and once with test.
    """
    xnnpack_cc_library(
        name = name + "_prod_microkernels",
        srcs = srcs_prod,
        x86_srcs = x86_srcs_prod,
        aarch32_srcs = aarch32_srcs_prod,
        aarch64_srcs = aarch64_srcs_prod,
        hexagon_srcs = hexagon_srcs_prod,
        riscv_srcs = riscv_srcs_prod,
        wasm_srcs = wasm_srcs_prod,
        wasmsimd_srcs = wasmsimd_srcs_prod,
        wasmrelaxedsimd_srcs = wasmrelaxedsimd_srcs_prod,
        linkopts = linkopts,
        copts = copts,
        gcc_copts = gcc_copts,
        msvc_copts = msvc_copts,
        mingw_copts = mingw_copts,  # buildifier: disable=unused-variable
        msys_copts = msys_copts,  # buildifier: disable=unused-variable
        gcc_x86_copts = gcc_x86_copts,
        msvc_x86_32_copts = msvc_x86_32_copts,
        msvc_x86_64_copts = msvc_x86_64_copts,
        aarch32_copts = aarch32_copts,
        aarch64_copts = aarch64_copts,
        hexagon_copts = hexagon_copts,
        riscv_copts = riscv_copts,
        wasm_copts = wasm_copts,
        wasmsimd_copts = wasmsimd_copts,
        wasmrelaxedsimd_copts = wasmrelaxedsimd_copts,
        optimized_copts = optimized_copts,
        hdrs = hdrs,
        defines = defines,
        includes = includes,
        deps = deps,
        visibility = visibility,
        testonly = testonly,
    )
    xnnpack_cc_library(
        name = name + "_test_microkernels",
        srcs = srcs_test,
        x86_srcs = x86_srcs_test,
        aarch32_srcs = aarch32_srcs_test,
        aarch64_srcs = aarch64_srcs_test,
        hexagon_srcs = hexagon_srcs_test,
        riscv_srcs = riscv_srcs_test,
        wasm_srcs = wasm_srcs_test,
        wasmsimd_srcs = wasmsimd_srcs_test,
        wasmrelaxedsimd_srcs = wasmrelaxedsimd_srcs_test,
        linkopts = linkopts,
        copts = copts,
        gcc_copts = gcc_copts,
        msvc_copts = msvc_copts,
        mingw_copts = mingw_copts,  # buildifier: disable=unused-variable
        msys_copts = msys_copts,  # buildifier: disable=unused-variable
        gcc_x86_copts = gcc_x86_copts,
        msvc_x86_32_copts = msvc_x86_32_copts,
        msvc_x86_64_copts = msvc_x86_64_copts,
        aarch32_copts = aarch32_copts,
        aarch64_copts = aarch64_copts,
        hexagon_copts = hexagon_copts,
        riscv_copts = riscv_copts,
        wasm_copts = wasm_copts,
        wasmsimd_copts = wasmsimd_copts,
        wasmrelaxedsimd_copts = wasmrelaxedsimd_copts,
        optimized_copts = optimized_copts,
        hdrs = hdrs,
        defines = defines,
        includes = includes,
        deps = deps,
        visibility = visibility,
        testonly = testonly,
    )

def xnnpack_aggregate_library(
        name,
        generic_deps = [],
        x86_deps = [],
        aarch32_deps = [],
        aarch64_deps = [],
        hexagon_deps = [],
        riscv_deps = [],
        wasm_deps = [],
        wasmsimd_deps = [],
        wasmrelaxedsimd_deps = [],
        defines = [],
        compatible_with = None):
    """Static library that aggregates architecture-specific dependencies.

    Args:
      name: The name of the library target to define.
      generic_deps: The list of libraries to link on all architectures.
      x86_deps: The list of libraries to link in x86 and x86-64 builds.
      aarch32_deps: The list of libraries to link in AArch32 builds.
      aarch64_deps: The list of libraries to link in AArch64 builds.
      hexagon_deps: The list of libraries to link in Hexagon builds.
      riscv_deps: The list of libraries to link in RISC-V builds.
      wasm_deps: The list of libraries to link in WebAssembly 1.0 builds.
      wasmsimd_deps: The list of libraries to link in WebAssembly SIMD builds.
      wasmrelaxedsimd_deps: The list of libraries to link in WebAssembly
                            Relaxed SIMD builds.
      defines: List of predefines macros to be added to the compile line.
      compatible_with: The list of additional environments this rule can be built for.
    """

    native.cc_library(
        name = name,
        linkstatic = True,
        deps = generic_deps + select({
            "//build_config:aarch32": aarch32_deps,
            "//build_config:aarch64": aarch64_deps,
            "//build_config:x86": x86_deps,
            "//build_config:emscripten_wasm": wasm_deps,
            "//build_config:emscripten_wasmsimd": wasmsimd_deps,
            "//build_config:emscripten_wasmrelaxedsimd": wasmrelaxedsimd_deps,
            "//build_config:riscv": riscv_deps,
            "//conditions:default": [],
        }),
        defines = defines,
        compatible_with = compatible_with,
        visibility = ["//:__subpackages__"],
    )

def xnnpack_unit_test(name, srcs, copts = [], mingw_copts = [], msys_copts = [], deps = [], tags = [], linkopts = [], defines = [], automatic = True, timeout = "short", shard_count = 1):
    """Unit test binary based on Google Test.

    Args:
      name: The name of the test target to define.
      srcs: The list of source and header files.
      copts: The list of additional compiler flags for the target. -I flags
             for include/ and src/ directories of XNNPACK are always prepended
             before these user-specified flags.
      mingw_copts: The list of compiler flags to use with MinGW GCC compilers.
      msys_copts: The list of compiler flags to use with MSYS (Cygwin) GCC compilers.
      deps: The list of additional libraries to be linked. Google Test library
            (with main() function) is always added as a dependency and does not
            need to be explicitly specified.
      linkopts: The list of linking options
      defines: List of predefines macros to be added to the compile line.
      tags: List of arbitrary text tags.
      automatic: Whether to create the test or testable binary.
      timeout: How long the test is expected to run before returning.
      shard_count: Specifies the number of parallel shards to use to run the test.
    """

    if automatic:
        native.cc_test(
            name = name,
            srcs = srcs,
            copts = xnnpack_std_cxxopts() + [
                "-Iinclude",
                "-Isrc",
            ] + select({
                "//build_config:windows_x86_64_mingw": mingw_copts,
                "//build_config:windows_x86_64_msys": msys_copts,
                "//conditions:default": [],
            }) + select({
                "//build_config:windows_x86_64_clang": ["/clang:-Wno-unused-function"],
                "//build_config:windows_x86_64_mingw": ["-Wno-unused-function"],
                "//build_config:windows_x86_64_msys": ["-Wno-unused-function"],
                "//build_config:windows_x86_64": [],
                "//conditions:default": ["-Wno-unused-function"],
            }) + copts,
            linkopts = select({
                "//build_config:emscripten": xnnpack_emscripten_test_linkopts(),
                "//conditions:default": [],
            }) + linkopts,
            linkstatic = True,
            defines = defines,
            deps = [
                "@com_google_googletest//:gtest_main",
            ] + deps + select({
                "//build_config:emscripten": xnnpack_emscripten_deps(),
                "//conditions:default": [],
            }),
            tags = tags,
            timeout = timeout,
            shard_count = shard_count,
        )
    else:
        native.cc_binary(
            name = name,
            srcs = srcs,
            copts = xnnpack_std_cxxopts() + [
                "-Iinclude",
                "-Isrc",
            ] + select({
                "//build_config:windows_x86_64_mingw": mingw_copts,
                "//build_config:windows_x86_64_msys": msys_copts,
                "//conditions:default": [],
            }) + select({
                "//build_config:windows_x86_64_clang": ["/clang:-Wno-unused-function"],
                "//build_config:windows_x86_64_mingw": ["-Wno-unused-function"],
                "//build_config:windows_x86_64_msys": ["-Wno-unused-function"],
                "//build_config:windows_x86_64": [],
                "//conditions:default": ["-Wno-unused-function"],
            }) + copts,
            linkopts = select({
                "//build_config:emscripten": xnnpack_emscripten_test_linkopts(),
                "//conditions:default": [],
            }),
            linkstatic = True,
            defines = defines,
            deps = [
                "@com_google_googletest//:gtest_main",
            ] + deps + select({
                "//build_config:emscripten": xnnpack_emscripten_deps(),
                "//conditions:default": [],
            }),
            testonly = True,
            tags = tags,
        )

def xnnpack_binary(name, srcs, copts = [], deps = [], linkopts = []):
    """Minimal binary

    Args:
      name: The name of the binary target to define.
      srcs: The list of source and header files.
      copts: The list of additional compiler flags for the target. -I flags
             for include/ and src/ directories of XNNPACK are always prepended
             before these user-specified flags.
      deps: The list of libraries to be linked.
      linkopts: The list of additional linker options
    """
    native.cc_binary(
        name = name,
        srcs = srcs,
        copts = [
            "-Iinclude",
            "-Isrc",
        ] + copts,
        linkopts = select({
            "//build_config:emscripten": xnnpack_emscripten_minimal_linkopts(),
            "//conditions:default": [],
        }) + linkopts,
        linkstatic = True,
        deps = deps,
    )

def xnnpack_benchmark(name, srcs, copts = [], deps = [], tags = [], defines = []):
    """Microbenchmark binary based on Google Benchmark

    Args:
      name: The name of the binary target to define.
      srcs: The list of source and header files.
      copts: The list of additional compiler flags for the target. -I flags
             for include/ and src/ directories of XNNPACK are always prepended
             before these user-specified flags.
      deps: The list of additional libraries to be linked. Google Benchmark
            library is always added as a dependency and does not need to be
            explicitly specified.
      tags: The list of arbitrary text tags.
      defines: The list of arbitrary defines tags.
    """
    native.cc_binary(
        name = name,
        srcs = srcs,
        copts = xnnpack_std_cxxopts() + [
            "-Iinclude",
            "-Isrc",
        ] + select({
            "//build_config:windows_x86_64_clang": ["/clang:-Wno-unused-function"],
            "//build_config:windows_x86_64_mingw": ["-Wno-unused-function"],
            "//build_config:windows_x86_64_msys": ["-Wno-unused-function"],
            "//build_config:windows_x86_64": [],
            "//conditions:default": ["-Wno-unused-function"],
        }) + copts,
        linkopts = select({
            "//build_config:emscripten": xnnpack_emscripten_benchmark_linkopts(),
            "//build_config:windows_x86_64_mingw": ["-lshlwapi"],
            "//build_config:windows_x86_64_msys": ["-lshlwapi"],
            "//conditions:default": [],
        }),
        linkstatic = True,
        deps = [
            "@com_google_benchmark//:benchmark",
        ] + deps + select({
            "//build_config:emscripten": xnnpack_emscripten_deps(),
            "//conditions:default": [],
        }),
        tags = tags,
        defines = defines,
    )

SrcListInfo = provider("A list of source files.", fields = {"srcs": "sources"})

def _source_list_aspect_impl(_target, ctx):
    srcs = []
    if hasattr(ctx.rule.attr, "srcs"):
        srcs += [s for src in ctx.rule.attr.srcs for s in src.files.to_list()]
    transitive = [dep[SrcListInfo].srcs for dep in ctx.rule.attr.deps]
    return [SrcListInfo(srcs = depset(srcs, transitive = transitive))]

source_list_aspect = aspect(
    implementation = _source_list_aspect_impl,
    attr_aspects = ["deps"],
)

def _transitive_source_list_rule_impl(ctx):
    get_repo_name = lambda x: getattr(x, "repo_name", getattr(x, "workspace_name"))
    files = [p for dep in ctx.attr.deps for p in dep[SrcListInfo].srcs.to_list() if get_repo_name(p.owner) == get_repo_name(ctx.label) and p.owner.package.startswith(ctx.label.package)]
    return [DefaultInfo(files = depset(files))]

xnnpack_transitive_source_list = rule(
    implementation = _transitive_source_list_rule_impl,
    attrs = {
        "deps": attr.label_list(aspects = [source_list_aspect]),
    },
)

