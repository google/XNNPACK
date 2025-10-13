"""Build definitions for YNNPACK.
"""

load("@bazel_skylib//lib:selects.bzl", "selects")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//:generated_file.bzl", "generated_file")
load("//:register_extension_info.bzl", "register_extension_info")

def define_build_option(name, default_conditions):
    """Defines a build flag `name` that is explicitly true or false, or all(default_conditions)."""
    selects.config_setting_group(
        name = name + "_enabled_by_default",
        match_all = default_conditions,
    )

    native.config_setting(
        name = name + "_explicit_true",
        define_values = {name: "true"},
    )

    native.config_setting(
        name = name + "_explicit_false",
        define_values = {name: "false"},
    )

    explicit_true = ":" + name + "_explicit_true"
    explicit_false = ":" + name + "_explicit_false"
    default = ":" + name + "_enabled_by_default"

    native.alias(
        name = name,
        actual = select({
            explicit_true: explicit_true,
            explicit_false: explicit_true,
            "//conditions:default": default,
        }),
    )

def ynn_select_if(cond = None, val_true = [], val_false = []):
    if cond != None:
        return select({
            cond: val_true,
            "//conditions:default": val_false,
        })
    else:
        return val_true

_YNN_PARAMS_FOR_ARCH = {
    "arm_neon": {
        "cond": "//ynnpack:ynn_enable_arm_neon",
        "copts": ynn_select_if(
            "//ynnpack/base/build_config:arm32",
            [
                "-marm",
                "-march=armv7-a",
                "-mfpu=neon",
            ],
        ),
    },
    "arm_neondot": {
        "cond": "//ynnpack:ynn_enable_arm_neondot",
        "copts": select(
            {
                "//ynnpack/base/build_config:arm32": [
                    "-marm",
                    "-march=armv8.2-a+dotprod",
                    "-mfpu=neon-fp-armv8",
                ],
                "//ynnpack/base/build_config:arm64": ["-march=armv8.2-a+dotprod"],
                "//conditions:default": [],
            },
        ),
    },
    "arm64_neoni8mm": {
        "cond": "//ynnpack:ynn_enable_arm64_neoni8mm",
        "copts": ynn_select_if(
            "//ynnpack/base/build_config:arm64",
            ["-march=armv8.2-a+i8mm"],
        ),
    },
    "arm64": {
        "cond": "//ynnpack/base/build_config:arm64",
    },
    # TODO(dsharlet): This is the same as above, should we just assume neon exists for arm64?
    "arm64_neon": {
        "cond": "//ynnpack:ynn_enable_arm64_neon",
    },
    "arm64_sme": {
        "cond": "//ynnpack:ynn_enable_arm64_sme",
        "copts": select({
            # Apple's Clang generates code that crashes with -msve (and works without it), while
            # other compilers can't compile this code without it.
            "//ynnpack/base/build_config:apple_clang_arm64": ["-march=armv8.2-a+sme"],
            "//ynnpack/base/build_config:arm64": ["-march=armv8.2-a+sve+sme"],
            "//conditions:default": [],
        }),
    },
    "arm64_sme2": {
        "cond": "//ynnpack:ynn_enable_arm64_sme2",
        "copts": select({
            "//ynnpack/base/build_config:apple_clang_arm64": ["-march=armv8.2-a+sme2"],
            "//ynnpack/base/build_config:arm64": ["-march=armv8.2-a+sve+sme2"],
            "//conditions:default": [],
        }),
    },
    "x86_sse2": {
        "cond": "//ynnpack:ynn_enable_x86_sse2",
        "copts": ynn_select_if(
            "//ynnpack/base/build_config:x86",
            ["-msse2", "-mno-ssse3"],
        ),
    },
    "x86_ssse3": {
        "cond": "//ynnpack:ynn_enable_x86_ssse3",
        "copts": ynn_select_if(
            "//ynnpack/base/build_config:x86",
            ["-mssse3", "-mno-sse4.1"],
        ),
    },
    "x86_sse41": {
        "cond": "//ynnpack:ynn_enable_x86_sse41",
        "copts": ynn_select_if(
            "//ynnpack/base/build_config:x86",
            ["-msse4.1", "-mno-sse4.2"],
        ),
    },
    "x86_avx": {
        "cond": "//ynnpack:ynn_enable_x86_avx",
        "copts": ynn_select_if(
            "//ynnpack/base/build_config:x86",
            [
                "-mavx",
                "-mno-avx2",
                "-mno-f16c",
                "-mno-fma",
            ],
        ),
    },
    "x86_f16c": {
        "cond": "//ynnpack:ynn_enable_x86_f16c",
        "copts": ynn_select_if(
            "//ynnpack/base/build_config:x86",
            ["-mf16c"],
        ),
    },
    "x86_avx2": {
        "cond": "//ynnpack:ynn_enable_x86_avx2",
        "copts": ynn_select_if(
            "//ynnpack/base/build_config:x86",
            ["-mavx2"],
        ),
    },
    "x86_fma3": {
        "cond": "//ynnpack:ynn_enable_x86_fma3",
        "copts": ynn_select_if(
            "//ynnpack/base/build_config:x86",
            [
                "-mavx",
                "-mfma",
                "-mno-avx2",
            ],
        ),
    },
    "x86_avx2_fma3": {
        "cond": "//ynnpack:ynn_enable_x86_avx2_fma3",
        "copts": ynn_select_if(
            "//ynnpack/base/build_config:x86",
            ["-mavx2", "-mfma"],
        ),
    },
    "x86_avx512f": {
        "cond": "//ynnpack:ynn_enable_x86_avx512f",
        "copts": ynn_select_if(
            "//ynnpack/base/build_config:x86",
            ["-mavx512f"],
        ),
    },
    "x86_avx512bw": {
        "cond": "//ynnpack:ynn_enable_x86_avx512bw",
        "copts": ynn_select_if(
            "//ynnpack/base/build_config:x86",
            ["-mavx512bw"],
        ),
    },
    "x86_avx512bf16": {
        "cond": "//ynnpack:ynn_enable_x86_avx512bf16",
        "copts": ynn_select_if(
            "//ynnpack/base/build_config:x86",
            ["-mavx512bf16", "-mavx512dq"],
        ),
    },
    "x86_avx512fp16": {
        "cond": "//ynnpack:ynn_enable_x86_avx512fp16",
        "copts": ynn_select_if(
            "//ynnpack/base/build_config:x86",
            ["-mavx512fp16", "-mavx512vl"],
        ),
    },
    "x86_avx512vnni": {
        "cond": "//ynnpack:ynn_enable_x86_avx512vnni",
        "copts": ynn_select_if(
            "//ynnpack/base/build_config:x86",
            ["-mavx512vnni"],
        ),
    },
    "x86_amxbf16": {
        "cond": "//ynnpack:ynn_enable_x86_amxbf16",
        "copts": ynn_select_if(
            "//ynnpack/base/build_config:x86",
            ["-mamx-tile", "-mamx-bf16"],
        ),
    },
    "x86_amxfp16": {
        "cond": "//ynnpack:ynn_enable_x86_amxfp16",
        "copts": ynn_select_if(
            "//ynnpack/base/build_config:x86",
            ["-mamx-tile", "-mamx-fp16"],
        ),
    },
    "x86_amxint8": {
        "cond": "//ynnpack:ynn_enable_x86_amxint8",
        "copts": ynn_select_if(
            "//ynnpack/base/build_config:x86",
            ["-mamx-tile", "-mamx-int8"],
        ),
    },
}

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
    for arch, srcs_arch in per_arch_srcs.items():
        arch_params = _YNN_PARAMS_FOR_ARCH[arch]
        cc_library(
            name = name + "_" + arch,
            srcs = ynn_select_if(
                cond = arch_params["cond"],
                val_true = srcs_arch,
            ),
            defines = defines + ["YNN_ARCH_" + arch.upper()],
            local_defines = local_defines + ["ARCH=" + arch],
            copts = copts + arch_params.get("copts", []),
            hdrs_check = "strict",
            deps = deps,
            features = [
                # We can't use copts with header modules...?
                "-use_header_modules",
            ] + kwargs.get("features", []),
            **kwargs
        )

        deps_plus_arch_deps += ynn_select_if(arch_params["cond"], [":" + name + "_" + arch])

    cc_library(
        name = name,
        srcs = srcs,
        defines = defines,
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
