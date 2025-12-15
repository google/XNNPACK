"""Build definitions for YNNPACK.
"""

load("@bazel_skylib//lib:selects.bzl", "selects")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//:generated_file.bzl", "generated_file")
load("//:register_extension_info.bzl", "register_extension_info")

def define_build_option(name, default_all = [], default_any = []):
    """Defines a build flag `name` that is explicitly true or false, all(default_all), or any(default_any).

    Args:
      name: The name of the build flag to define.
      default_all: A list of conditions that must all be true for the flag to be enabled by default.
      default_any: A list of conditions for the flag to be enabled by default if any are true.
    """
    explicit_true = name + "_explicit_true"
    explicit_false = name + "_explicit_false"
    default = explicit_true
    if default_all or default_any:
        default = name + "_enabled_by_default"
        selects.config_setting_group(
            name = default,
            match_all = default_all,
            match_any = default_any,
        )

    native.config_setting(
        name = explicit_true,
        define_values = {name: "true"},
    )

    native.config_setting(
        name = explicit_false,
        define_values = {name: "false"},
    )

    native.alias(
        name = name,
        actual = select({
            explicit_true: ":" + explicit_true,
            explicit_false: ":" + explicit_true,
            "//conditions:default": ":" + default,
        }),
    )

_YNN_PARAMS_FOR_ARCH = {
    "arm_neon": {
        "cond": "//ynnpack:ynn_enable_arm_neon",
        "arch_copts": select({
            "//ynnpack:arm32": [
                "-marm",
                "-march=armv7-a",
                "-mfpu=neon",
            ],
            "//conditions:default": [],
        }),
    },
    "arm_neondot": {
        "cond": "//ynnpack:ynn_enable_arm_neondot",
        "arch_copts": select({
            "//ynnpack:arm32": [
                "-marm",
                "-march=armv8.2-a+dotprod",
                "-mfpu=neon-fp-armv8",
            ],
            "//ynnpack:arm64": ["-march=armv8.2-a+dotprod"],
            "//conditions:default": [],
        }),
    },
    "arm_neonfma": {
        "cond": "//ynnpack:ynn_enable_arm_neonfma",
        "arch_copts": select({
            "//ynnpack:arm32": [
                "-marm",
                "-march=armv7-a",
                "-mfpu=neon-vfpv4",
            ],
            "//conditions:default": [],
        }),
    },
    "arm_neonfp16": {
        "cond": "//ynnpack:ynn_enable_arm_neonfp16",
        "arch_copts": select({
            "//ynnpack:arm32": [
                "-marm",
                "-march=armv7-a",
                "-mfpu=neon-fp16",
            ],
            "//conditions:default": [],
        }),
    },
    "arm_neonfp16arith": {
        "cond": "//ynnpack:ynn_enable_arm_neonfp16arith",
        "arch_copts": select({
            "//ynnpack:arm32": [
                "-marm",
                "-march=armv8.2-a+fp16",
                "-mfpu=neon-fp-armv8",
            ],
            "//ynnpack:arm64": ["-march=armv8.2-a+fp16"],
            "//conditions:default": [],
        }),
    },
    "arm_neonbf16": {
        "cond": "//ynnpack:ynn_enable_arm_neonbf16",
        "arch_copts": select({
            "//ynnpack:arm32": [
                "-marm",
                "-march=armv8.2-a+bf16",
                "-mfpu=neon-fp-armv8",
            ],
            "//ynnpack:arm64": ["-march=armv8.2-a+bf16"],
            "//conditions:default": [],
        }),
    },
    "arm64_neoni8mm": {
        "cond": "//ynnpack:ynn_enable_arm64_neoni8mm",
        "arch_copts": ["-march=armv8.2-a+i8mm"],
    },
    "arm64": {
        "cond": "//ynnpack:ynn_enable_arm64",
    },
    # TODO(dsharlet): This is the same as above, should we just assume neon exists for arm64?
    "arm64_neon": {
        "cond": "//ynnpack:ynn_enable_arm64_neon",
    },
    "arm64_sme": {
        "cond": "//ynnpack:ynn_enable_arm64_sme",
        "arch_copts": select({
            # Apple's Clang generates code that crashes with -msve (and works without it), while
            # other compilers can't compile this code without it.
            "//ynnpack:apple_clang": ["-march=armv8.2-a+sme"],
            "//conditions:default": ["-march=armv8.2-a+sve+sme"],
        }),
    },
    "arm64_sme2": {
        "cond": "//ynnpack:ynn_enable_arm64_sme2",
        "arch_copts": select({
            "//ynnpack:apple_clang": ["-march=armv8.2-a+sme2"],
            "//conditions:default": ["-march=armv8.2-a+sve+sme2"],
        }),
    },
    "x86_sse2": {
        "cond": "//ynnpack:ynn_enable_x86_sse2",
        "arch_copts": ["-msse2", "-mno-ssse3"],
    },
    "x86_ssse3": {
        "cond": "//ynnpack:ynn_enable_x86_ssse3",
        "arch_copts": ["-mssse3", "-mno-sse4.1"],
    },
    "x86_sse41": {
        "cond": "//ynnpack:ynn_enable_x86_sse41",
        "arch_copts": ["-msse4.1", "-mno-sse4.2"],
    },
    "x86_avx": {
        "cond": "//ynnpack:ynn_enable_x86_avx",
        "arch_copts": ["-mavx", "-mno-avx2", "-mno-f16c", "-mno-fma"],
    },
    "x86_f16c": {
        "cond": "//ynnpack:ynn_enable_x86_f16c",
        "arch_copts": ["-mf16c"],
    },
    "x86_f16c_fma3": {
        "cond": "//ynnpack:ynn_enable_x86_f16c_fma3",
        "arch_copts": ["-mf16c", "-mavx", "-mfma", "-mno-avx2"],
    },
    "x86_avx2": {
        "cond": "//ynnpack:ynn_enable_x86_avx2",
        "arch_copts": ["-mavx2"],
    },
    "x86_fma3": {
        "cond": "//ynnpack:ynn_enable_x86_fma3",
        "arch_copts": ["-mavx", "-mfma", "-mno-avx2"],
    },
    "x86_avx2_fma3": {
        "cond": "//ynnpack:ynn_enable_x86_avx2_fma3",
        "arch_copts": ["-mavx2", "-mfma"],
    },
    "x86_avx512f": {
        "cond": "//ynnpack:ynn_enable_x86_avx512f",
        "arch_copts": ["-mavx512f"],
    },
    "x86_avx512bw": {
        "cond": "//ynnpack:ynn_enable_x86_avx512bw",
        "arch_copts": ["-mavx512bw"],
    },
    "x86_avx512bf16": {
        "cond": "//ynnpack:ynn_enable_x86_avx512bf16",
        "arch_copts": ["-mavx512bf16", "-mavx512dq"],
    },
    "x86_avx512fp16": {
        "cond": "//ynnpack:ynn_enable_x86_avx512fp16",
        "arch_copts": ["-mavx512fp16", "-mavx512vl"],
    },
    "x86_avx512vnni": {
        "cond": "//ynnpack:ynn_enable_x86_avx512vnni",
        "arch_copts": ["-mavx512vnni"],
    },
    "x86_amxbf16": {
        "cond": "//ynnpack:ynn_enable_x86_amxbf16",
        "arch_copts": ["-mamx-tile", "-mamx-bf16"],
    },
    "x86_amxfp16": {
        "cond": "//ynnpack:ynn_enable_x86_amxfp16",
        "arch_copts": ["-mamx-tile", "-mamx-fp16"],
    },
    "x86_amxint8": {
        "cond": "//ynnpack:ynn_enable_x86_amxint8",
        "arch_copts": ["-mamx-tile", "-mamx-int8"],
    },
}

def _map_arch_copts_to_msvc(copts):
    """Maps GNU-style target architecture compiler options to Microsoft Visual Studio compiler options."""

    if any([c.startswith("-mavx512") or c.startswith("-mamx") for c in copts]):
        return ["/arch:AVX512"]
    elif any([c in ["-mavx2"] for c in copts]):
        return ["/arch:AVX2"]
    elif any([c in ["-mavx", "-mfma", "-mf16c"] for c in copts]):
        return ["/arch:AVX"]
    elif any([c in ["-msse2", "-mssse3", "-msse4.1"] for c in copts]):
        return ["/arch:SSE2"]
    else:
        return []

def ynn_kernel_copts(unroll_loops = True):
    return select({
        "//ynnpack:msvc": ["/O2"],
        "//conditions:default": ["-O2"] + ([] if unroll_loops else ["-fno-unroll-loops"]),
    })

def ynn_binary_linkopts():
    return select({
        "//conditions:default": [],
    })

def ynn_binary_malloc():
    return select({
        "//conditions:default": "@bazel_tools//tools/cpp:malloc",
    })

def ynn_test_deps():
    return select({
        "//conditions:default": ["@com_google_googletest//:gtest_main"],
    })

def ynn_benchmark_deps():
    return select({
        "//conditions:default": ["@com_google_benchmark//:benchmark_main"],
    })

def ynn_cc_library(
        name,
        srcs = [],
        per_arch_srcs = {},
        copts = [],
        defines = [],
        local_defines = [],
        deps = [],
        **kwargs):
    """C/C++/assembly library with architecture-specific configuration.

    Define a library with architecture- and instruction-specific source files and/or compiler flags.

    Args:
      name: The name of the library target to define.
      srcs: The list of architecture-independent source files.
      per_arch_srcs: A dictionary of architectures (as found in _YNN_PARAMS_FOR_ARCH) to
          architecture-specific source files.
      copts: A list of compiler options to use for all srcs and per_arch_srcs.
      defines: A list of macros to define for all srcs and per_arch_srcs and dependencies.
      local_defines: A list of macros to define for all srcs and per_arch_srcs.
      deps: A list of dependencies to use for all srcs and per_arch_srcs.
      **kwargs: Other arguments to pass to the cc_library rule.
    """
    deps_plus_arch_deps = deps
    for arch, arch_srcs in per_arch_srcs.items():
        arch_params = _YNN_PARAMS_FOR_ARCH[arch]
        arch_cond = arch_params["cond"]
        arch_copts = arch_params.get("arch_copts", [])
        if type(arch_copts) == "list":
            arch_copts = select({
                "@rules_cc//cc/compiler:clang-cl": ["/clang:" + i for i in arch_copts],
                "@rules_cc//cc/compiler:msvc-cl": _map_arch_copts_to_msvc(arch_copts),
                "//conditions:default": arch_copts,
            })
        else:
            # The arch_params should have handled this.
            pass

        cc_library(
            name = name + "_" + arch,
            srcs = select({
                arch_cond: arch_srcs,
                "//conditions:default": [],
            }),
            defines = defines + ["YNN_ARCH_" + arch.upper()],
            local_defines = local_defines + ["ARCH=" + arch],
            copts = copts + arch_copts,
            hdrs_check = "strict",
            deps = deps,
            features = [
                # We can't use copts with header modules...?
                "-use_header_modules",
            ] + kwargs.get("features", []),
            # Don't build this target unless explicitly requested.
            tags = ["manual"],
            **kwargs
        )

        deps_plus_arch_deps += select({
            arch_cond: [":" + name + "_" + arch],
            "//conditions:default": [],
        })

    cc_library(
        name = name,
        srcs = srcs,
        defines = defines,
        copts = copts,
        local_defines = local_defines,
        hdrs_check = "strict",
        deps = deps_plus_arch_deps,
        **kwargs
    )

def ynn_generate_src_hdr(
        name,
        output_src,
        output_hdr,
        generator,
        generator_args = [],
        **kwargs):
    """Generates a source file from a generator script."""
    native.genrule(
        name = name,
        outs = [output_src, output_hdr],
        cmd = "$(location " + generator + ") " + "$(location " + output_src + ") " + "$(location " + output_hdr + ") " + " ".join(generator_args),
        tools = [generator],
        **kwargs
    )

    generated_file(
        scopes = ["presubmit", "codesearch"],
        wrapped_target = name,
    )

register_extension_info(
    extension = ynn_cc_library,
    label_regex_for_dep = "{extension_name}",
)
