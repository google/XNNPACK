"""Build definitions and rules for XNNPACK."""

load(":emscripten.bzl", "xnnpack_emscripten_benchmark_linkopts", "xnnpack_emscripten_deps", "xnnpack_emscripten_minimal_linkopts", "xnnpack_emscripten_test_linkopts")

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
    return ["-std=gnu++11"]

def xnnpack_optional_ruy_copts():
    """Compiler flags to optionally enable Ruy benchmarks."""
    return []

def xnnpack_optional_gemmlowp_copts():
    """Compiler flags to optionally enable Gemmlowp benchmarks."""
    return []

def xnnpack_optional_tflite_copts():
    """Compiler flags to optionally enable TensorFlow Lite benchmarks."""
    return []

def xnnpack_optional_armcl_copts():
    """Compiler flags to optionally enable ARM ComputeLibrary benchmarks."""
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

def xnnpack_optional_armcl_deps():
    """Optional ARM ComputeLibrary dependencies."""
    return []

def xnnpack_optional_dnnl_deps():
    """Optional Intel DNNL dependencies."""
    return []

def xnnpack_cc_library(
        name,
        srcs = [],
        psimd_srcs = [],
        x86_srcs = [],
        aarch32_srcs = [],
        aarch64_srcs = [],
        asmjs_srcs = [],
        wasm_srcs = [],
        wasmsimd_srcs = [],
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
        asmjs_copts = [],
        wasm_copts = [],
        wasmsimd_copts = [],
        optimized_copts = ["-O2"],
        hdrs = [],
        defines = [],
        includes = [],
        deps = [],
        visibility = []):
    """C/C++/assembly library with architecture-specific configuration.

    Define a static library with architecture- and instruction-specific
    source files and/or compiler flags.

    Args:
      name: The name of the library target to define.
      srcs: The list of architecture-independent source files.
      psimd_srcs: The list of psimd-specific source files.
      x86_srcs: The list of x86-specific source files.
      aarch32_srcs: The list of AArch32-specific source files.
      aarch64_srcs: The list of AArch64-specific source files.
      asmjs_srcs: The list of Asm.js-specific source files.
      wasm_srcs: The list of WebAssembly/MVP-specific source files.
      wasmsimd_srcs: The list of WebAssembly/SIMD-specific source files.
      copts: The list of compiler flags to use in all builds. -I flags for
             include/ and src/ directories of XNNPACK are always prepended
             before these user-specified flags.
      gcc_copts: The list of compiler flags to use with GCC-like compilers.
      msvc_copts: The list of compiler flags to use with MSVC compiler.
      mingw_copts: The list of compiler flags to use with MinGW GCC compilers.
      msys_copts: The list of compiler flags to use with MSYS (Cygwin) GCC compilers.
      gcc_x86_copts: The list of GCC-like compiler flags to use in x86 (32-bit and 64-bit) builds.
      msvc_x86_32_copts: The list of MSVC compiler flags to use in x86 (32-bit) builds.
      msvc_x86_64_copts: The list of MSVC compiler flags to use in x86 (64-bit) builds.
      aarch32_copts: The list of compiler flags to use in AArch32 builds.
      aarch64_copts: The list of compiler flags to use in AArch64 builds.
      asmjs_copts: The list of compiler flags to use in Asm.js builds.
      wasm_copts: The list of compiler flags to use in WebAssembly/MVP builds.
      wasmsimd_copts: The list of compiler flags to use in WebAssembly/SIMD
                      builds.
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
            ":linux_k8": psimd_srcs + x86_srcs,
            ":linux_arm": psimd_srcs + aarch32_srcs,
            ":linux_armhf": psimd_srcs + aarch32_srcs,
            ":linux_aarch64": psimd_srcs + aarch64_srcs,
            ":macos_x86_64": psimd_srcs + x86_srcs,
            ":windows_x86_64_clang": psimd_srcs + x86_srcs,
            ":windows_x86_64_mingw": psimd_srcs + x86_srcs,
            ":windows_x86_64_msys": psimd_srcs + x86_srcs,
            ":windows_x86_64": x86_srcs,
            ":android_armv7": psimd_srcs + aarch32_srcs,
            ":android_arm64": psimd_srcs + aarch64_srcs,
            ":android_x86": psimd_srcs + x86_srcs,
            ":android_x86_64": psimd_srcs + x86_srcs,
            ":ios_armv7": psimd_srcs + aarch32_srcs,
            ":ios_arm64": psimd_srcs + aarch64_srcs,
            ":ios_arm64e": psimd_srcs + aarch64_srcs,
            ":ios_x86": psimd_srcs + x86_srcs,
            ":ios_x86_64": psimd_srcs + x86_srcs,
            ":watchos_armv7k": psimd_srcs + aarch32_srcs,
            ":watchos_arm64_32": psimd_srcs + aarch64_srcs,
            ":watchos_x86": psimd_srcs + x86_srcs,
            ":watchos_x86_64": psimd_srcs + x86_srcs,
            ":tvos_arm64": psimd_srcs + aarch64_srcs,
            ":tvos_x86_64": psimd_srcs + x86_srcs,
            ":emscripten_asmjs": asmjs_srcs,
            ":emscripten_wasm": wasm_srcs,
            ":emscripten_wasmsimd": psimd_srcs + wasmsimd_srcs,
            "//conditions:default": [],
        }),
        copts = [
            "-Iinclude",
            "-Isrc",
        ] + copts + select({
            ":linux_k8": gcc_x86_copts,
            ":linux_arm": aarch32_copts,
            ":linux_armhf": aarch32_copts,
            ":linux_aarch64": aarch64_copts,
            ":macos_x86_64": gcc_x86_copts,
            ":windows_x86_64_clang": ["/clang:" + opt for opt in gcc_x86_copts],
            ":windows_x86_64_mingw": mingw_copts + gcc_x86_copts,
            ":windows_x86_64_msys": msys_copts + gcc_x86_copts,
            ":windows_x86_64": msvc_x86_64_copts,
            ":android_armv7": aarch32_copts,
            ":android_arm64": aarch64_copts,
            ":android_x86": gcc_x86_copts,
            ":android_x86_64": gcc_x86_copts,
            ":ios_armv7": aarch32_copts,
            ":ios_arm64": aarch64_copts,
            ":ios_arm64e": aarch64_copts,
            ":ios_x86": gcc_x86_copts,
            ":ios_x86_64": gcc_x86_copts,
            ":watchos_armv7k": aarch32_copts,
            ":watchos_arm64_32": aarch64_copts,
            ":watchos_x86": gcc_x86_copts,
            ":watchos_x86_64": gcc_x86_copts,
            ":tvos_arm64": aarch64_copts,
            ":tvos_x86_64": gcc_x86_copts,
            ":emscripten_asmjs": asmjs_copts,
            ":emscripten_wasm": wasm_copts,
            ":emscripten_wasmsimd": wasmsimd_copts,
            "//conditions:default": [],
        }) + select({
            ":windows_x86_64_clang": ["/clang:" + opt for opt in gcc_copts],
            ":windows_x86_64_mingw": gcc_copts,
            ":windows_x86_64_msys": gcc_copts,
            ":windows_x86_64": msvc_copts,
            "//conditions:default": gcc_copts,
        }) + select({
            ":optimized_build": optimized_copts,
            "//conditions:default": [],
        }),
        defines = defines,
        deps = deps,
        includes = ["include", "src"] + includes,
        linkstatic = True,
        linkopts = select({
            ":linux_k8": ["-lpthread"],
            ":linux_arm": ["-lpthread"],
            ":linux_armhf": ["-lpthread"],
            ":linux_aarch64": ["-lpthread"],
            ":android": ["-lm"],
            "//conditions:default": [],
        }),
        textual_hdrs = hdrs,
        visibility = visibility,
    )

def xnnpack_aggregate_library(
        name,
        generic_deps = [],
        psimd_deps = [],
        x86_deps = [],
        aarch32_deps = [],
        aarch64_deps = [],
        wasm_deps = [],
        wasmsimd_deps = []):
    """Static library that aggregates architecture-specific dependencies.

    Args:
      name: The name of the library target to define.
      generic_deps: The list of libraries to link on all architectures.
      psimd_deps: The list of libraries to link in psimd-enabled builds.
      x86_deps: The list of libraries to link in x86 and x86-64 builds.
      aarch32_deps: The list of libraries to link in AArch32 builds.
      aarch64_deps: The list of libraries to link in AArch32 builds.
      wasm_deps: The list of libraries to link in WebAssembly (MVP) builds.
      wasmsimd_deps: The list of libraries to link in WebAssembly SIMD builds.
    """

    native.cc_library(
        name = name,
        linkstatic = True,
        deps = generic_deps + select({
            ":linux_k8": psimd_deps + x86_deps,
            ":linux_arm": psimd_deps + aarch32_deps,
            ":linux_armhf": psimd_deps + aarch32_deps,
            ":linux_aarch64": psimd_deps + aarch64_deps,
            ":macos_x86_64": psimd_deps + x86_deps,
            ":windows_x86_64_clang": psimd_deps + x86_deps,
            ":windows_x86_64_mingw": psimd_deps + x86_deps,
            ":windows_x86_64_msys": psimd_deps + x86_deps,
            ":windows_x86_64": x86_deps,
            ":android_armv7": psimd_deps + aarch32_deps,
            ":android_arm64": psimd_deps + aarch64_deps,
            ":android_x86": psimd_deps + x86_deps,
            ":android_x86_64": psimd_deps + x86_deps,
            ":ios_armv7": psimd_deps + aarch32_deps,
            ":ios_arm64": psimd_deps + aarch64_deps,
            ":ios_arm64e": psimd_deps + aarch64_deps,
            ":ios_x86": psimd_deps + x86_deps,
            ":ios_x86_64": psimd_deps + x86_deps,
            ":watchos_armv7k": psimd_deps + aarch32_deps,
            ":watchos_arm64_32": psimd_deps + aarch64_deps,
            ":watchos_x86": psimd_deps + x86_deps,
            ":watchos_x86_64": psimd_deps + x86_deps,
            ":tvos_arm64": psimd_deps + aarch64_deps,
            ":tvos_x86_64": psimd_deps + x86_deps,
            ":emscripten_wasm": wasm_deps,
            ":emscripten_wasmsimd": psimd_deps + wasmsimd_deps,
            ":emscripten_asmjs": [],
        }),
    )

def xnnpack_unit_test(name, srcs, copts = [], mingw_copts = [], msys_copts = [], deps = []):
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
    """

    native.cc_test(
        name = name,
        srcs = srcs,
        copts = xnnpack_std_cxxopts() + [
            "-Iinclude",
            "-Isrc",
        ] + select({
            ":windows_x86_64_mingw": mingw_copts,
            ":windows_x86_64_msys": msys_copts,
            "//conditions:default": [],
        }) + select({
            ":windows_x86_64_clang": ["/clang:-Wno-unused-function"],
            ":windows_x86_64_mingw": ["-Wno-unused-function"],
            ":windows_x86_64_msys": ["-Wno-unused-function"],
            ":windows_x86_64": [],
            "//conditions:default": ["-Wno-unused-function"],
        }) + copts,
        linkopts = select({
            ":emscripten": xnnpack_emscripten_test_linkopts(),
            "//conditions:default": [],
        }),
        linkstatic = True,
        deps = [
            "@com_google_googletest//:gtest_main",
        ] + deps + select({
            ":emscripten": xnnpack_emscripten_deps(),
            "//conditions:default": [],
        }),
    )

def xnnpack_binary(name, srcs, copts = [], deps = []):
    """Minimal binary

    Args:
      name: The name of the binary target to define.
      srcs: The list of source and header files.
      copts: The list of additional compiler flags for the target. -I flags
             for include/ and src/ directories of XNNPACK are always prepended
             before these user-specified flags.
      deps: The list of libraries to be linked.
    """
    native.cc_binary(
        name = name,
        srcs = srcs,
        copts = [
            "-Iinclude",
            "-Isrc",
        ] + copts,
        linkopts = select({
            ":emscripten": xnnpack_emscripten_minimal_linkopts(),
            "//conditions:default": [],
        }),
        linkstatic = True,
        deps = deps,
    )

def xnnpack_benchmark(name, srcs, copts = [], deps = [], tags = []):
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
    """
    native.cc_binary(
        name = name,
        srcs = srcs,
        copts = xnnpack_std_cxxopts() + [
            "-Iinclude",
            "-Isrc",
        ] + select({
            ":windows_x86_64_clang": ["/clang:-Wno-unused-function"],
            ":windows_x86_64_mingw": ["-Wno-unused-function"],
            ":windows_x86_64_msys": ["-Wno-unused-function"],
            ":windows_x86_64": [],
            "//conditions:default": ["-Wno-unused-function"],
        }) + copts,
        linkopts = select({
            ":emscripten": xnnpack_emscripten_benchmark_linkopts(),
            ":windows_x86_64_mingw": ["-lshlwapi"],
            ":windows_x86_64_msys": ["-lshlwapi"],
            "//conditions:default": [],
        }),
        linkstatic = True,
        deps = [
            "@com_google_benchmark//:benchmark",
        ] + deps + select({
            ":emscripten": xnnpack_emscripten_deps(),
            "//conditions:default": [],
        }),
	tags = tags,
    )
