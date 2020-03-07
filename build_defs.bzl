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

def xnnpack_std_copts():
    """Compiler flags to specify language standard for C sources."""
    return ["-std=c99"]

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
        x86_srcs = [],
        aarch32_srcs = [],
        aarch64_srcs = [],
        asmjs_srcs = [],
        wasm_srcs = [],
        wasmsimd_srcs = [],
        copts = [],
        x86_copts = [],
        aarch32_copts = [],
        aarch64_copts = [],
        asmjs_copts = [],
        wasm_copts = [],
        wasmsimd_copts = [],
        optimized_copts = ["-O2"],
        hdrs = [],
        deps = []):
    """C/C++/assembly library with architecture-specific configuration.

    Define a static library with architecture- and instruction-specific
    source files and/or compiler flags.

    Args:
      name: The name of the library target to define.
      srcs: The list of architecture-independent source files.
      x86_srcs: The list of x86-specific source files.
      aarch32_srcs: The list of AArch32-specific source files.
      aarch64_srcs: The list of AArch64-specific source files.
      asmjs_srcs: The list of Asm.js-specific source files.
      wasm_srcs: The list of WebAssembly/MVP-specific source files.
      wasmsimd_srcs: The list of WebAssembly/SIMD-specific source files.
      copts: The list of compiler flags to use in all builds. -I flags for
             include/ and src/ directories of XNNPACK are always prepended
             before these user-specified flags.
      x86_copts: The list of compiler flags to use in x86 builds.
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
      deps: The list of other libraries to be linked.
    """
    native.cc_library(
        name = name,
        srcs = srcs + select({
            ":linux_k8": x86_srcs,
            ":linux_aarch64": aarch64_srcs,
            ":macos_x86_64": x86_srcs,
            ":android_armv7": aarch32_srcs,
            ":android_arm64": aarch64_srcs,
            ":android_x86": x86_srcs,
            ":android_x86_64": x86_srcs,
            ":ios_armv7": aarch32_srcs,
            ":ios_arm64": aarch64_srcs,
            ":ios_arm64e": aarch64_srcs,
            ":ios_x86": x86_srcs,
            ":ios_x86_64": x86_srcs,
            ":watchos_armv7k": aarch32_srcs,
            ":watchos_arm64_32": aarch64_srcs,
            ":watchos_x86": x86_srcs,
            ":watchos_x86_64": x86_srcs,
            ":tvos_arm64": aarch64_srcs,
            ":tvos_x86_64": x86_srcs,
            ":emscripten_asmjs": asmjs_srcs,
            ":emscripten_wasm": wasm_srcs,
            ":emscripten_wasmsimd": wasmsimd_srcs,
            "//conditions:default": [],
        }),
        copts = [
            "-Iinclude",
            "-Isrc",
        ] + copts + select({
            ":linux_k8": x86_copts,
            ":linux_aarch64": aarch64_copts,
            ":macos_x86_64": x86_copts,
            ":android_armv7": aarch32_copts,
            ":android_arm64": aarch64_copts,
            ":android_x86": x86_copts,
            ":android_x86_64": x86_copts,
            ":ios_armv7": aarch32_copts,
            ":ios_arm64": aarch64_copts,
            ":ios_arm64e": aarch64_copts,
            ":ios_x86": x86_copts,
            ":ios_x86_64": x86_copts,
            ":watchos_armv7k": aarch32_copts,
            ":watchos_arm64_32": aarch64_copts,
            ":watchos_x86": x86_copts,
            ":watchos_x86_64": x86_copts,
            ":tvos_arm64": aarch64_copts,
            ":tvos_x86_64": x86_copts,
            ":emscripten_asmjs": asmjs_copts,
            ":emscripten_wasm": wasm_copts,
            ":emscripten_wasmsimd": wasmsimd_copts,
            "//conditions:default": [],
        }) + select({
            ":optimized_build": optimized_copts,
            "//conditions:default": [],
        }),
        includes = ["include", "src"],
        linkstatic = True,
        linkopts = select({
            ":linux_k8": ["-lpthread"],
            ":linux_aarch64": ["-lpthread"],
            ":android": ["-lm"],
            "//conditions:default": [],
        }),
        textual_hdrs = hdrs,
        deps = deps,
    )

def xnnpack_aggregate_library(
        name,
        generic_deps = [],
        x86_deps = [],
        aarch32_deps = [],
        aarch64_deps = [],
        wasm_deps = [],
        wasmsimd_deps = []):
    """Static library that aggregates architecture-specific dependencies.

    Args:
      name: The name of the library target to define.
      generic_deps: The list of libraries to link on all architectures.
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
            ":linux_k8": x86_deps,
            ":linux_aarch64": aarch64_deps,
            ":macos_x86_64": x86_deps,
            ":android_armv7": aarch32_deps,
            ":android_arm64": aarch64_deps,
            ":android_x86": x86_deps,
            ":android_x86_64": x86_deps,
            ":ios_armv7": aarch32_deps,
            ":ios_arm64": aarch64_deps,
            ":ios_arm64e": aarch64_deps,
            ":ios_x86": x86_deps,
            ":ios_x86_64": x86_deps,
            ":watchos_armv7k": aarch32_deps,
            ":watchos_arm64_32": aarch64_deps,
            ":watchos_x86": x86_deps,
            ":watchos_x86_64": x86_deps,
            ":tvos_arm64": aarch64_deps,
            ":tvos_x86_64": x86_deps,
            ":emscripten_wasm": wasm_deps,
            ":emscripten_wasmsimd": wasmsimd_deps,
            ":emscripten_asmjs": [],
        }),
    )

def xnnpack_unit_test(name, srcs, copts = [], deps = []):
    """Unit test binary based on Google Test.

    Args:
      name: The name of the test target to define.
      srcs: The list of source and header files.
      copts: The list of additional compiler flags for the target. -I flags
             for include/ and src/ directories of XNNPACK are always prepended
             before these user-specified flags.
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
        ] + copts,
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

def xnnpack_benchmark(name, srcs, copts = [], deps = []):
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
        ] + copts,
        linkopts = select({
            ":emscripten": xnnpack_emscripten_benchmark_linkopts(),
            "//conditions:default": [],
        }),
        linkstatic = True,
        deps = [
            "@com_google_benchmark//:benchmark",
        ] + deps + select({
            ":emscripten": xnnpack_emscripten_deps(),
            "//conditions:default": [],
        }),
    )
