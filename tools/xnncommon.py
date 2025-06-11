#!/usr/bin/env python
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import codecs
import os


def _indent(text):
  return "\n".join(map(lambda t: "  " + t if t else t, text.splitlines()))


def _remove_duplicate_newlines(text):
  filtered_lines = list()
  last_newline = False
  for line in text.splitlines():
    is_newline = len(line.strip()) == 0
    if not is_newline or not last_newline:
      filtered_lines.append(line)
    last_newline = is_newline
  return "\n".join(filtered_lines)


_ARCH_TO_MACRO_MAP = {
    "aarch32": "XNN_ARCH_ARM",
    "aarch64": "XNN_ARCH_ARM64",
    "x86-32": "XNN_ARCH_X86",
    "x86-64": "XNN_ARCH_X86_64",
    "amd64": "XNN_ARCH_X86_64",
    "hexagon": "XNN_ARCH_HEXAGON",
    "riscv": "XNN_ARCH_RISCV",
    "wasm": "XNN_ARCH_WASM",
    "wasmsimd": "XNN_ARCH_WASMSIMD",
    "wasmrelaxedsimd": "XNN_ARCH_WASMRELAXEDSIMD",
    "wasm32": "XNN_ARCH_WASM",
    "wasmsimd32": "XNN_ARCH_WASMSIMD",
    "wasmrelaxedsimd32": "XNN_ARCH_WASMRELAXEDSIMD",
}

# Mapping from ISA extension to macro guarding build-time enabled/disabled
# status for the ISA. Only ISAs that can be enabled/disabled have an entry.
_ISA_TO_MACRO_MAP = {
    "fp16arith": "XNN_ENABLE_ARM_FP16_SCALAR",
    "neonfp16arith": "XNN_ENABLE_ARM_FP16_VECTOR",
    "neonbf16": "XNN_ENABLE_ARM_BF16",
    "neondot": "XNN_ENABLE_ARM_DOTPROD",
    "neondotfp16arith": "XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR",
    "neoni8mm": "XNN_ENABLE_ARM_I8MM",
    "neonsme": "XNN_ENABLE_ARM_SME",
    "neonsme2": "XNN_ENABLE_ARM_SME2",
    "rvv": "XNN_ENABLE_RISCV_VECTOR",
    "rvvfp16arith": "XNN_ENABLE_RISCV_FP16_VECTOR",
    "avxvnni": "XNN_ENABLE_AVXVNNI",
    "avxvnniint8": "XNN_ENABLE_AVXVNNIINT8",
    "avx256skx": "XNN_ENABLE_AVX256SKX",
    "avx256vnni": "XNN_ENABLE_AVX256VNNI",
    "avx256vnnigfni": "XNN_ENABLE_AVX256VNNIGFNI",
    "avx512f": "XNN_ENABLE_AVX512F",
    "avx512skx": "XNN_ENABLE_AVX512SKX",
    "avx512vbmi": "XNN_ENABLE_AVX512VBMI",
    "avx512vnni": "XNN_ENABLE_AVX512VNNI",
    "avx512vnnigfni": "XNN_ENABLE_AVX512VNNIGFNI",
    "avx512amx": "XNN_ENABLE_AVX512AMX",
    "avx512fp16": "XNN_ENABLE_AVX512FP16",
    "avx512bf16": "XNN_ENABLE_AVX512BF16",
    "hvx": "XNN_ENABLE_HVX",
}

_ISA_TO_ARCH_MAP = {
    "armsimd32": ["aarch32"],
    "fp16arith": ["aarch32", "aarch64"],
    "neon": ["aarch32", "aarch64"],
    "neonfp16": ["aarch32", "aarch64"],
    "neonfma": ["aarch32", "aarch64"],
    "neonv8": ["aarch32", "aarch64"],
    "neonfp16arith": ["aarch32", "aarch64"],
    "neonbf16": ["aarch32", "aarch64"],
    "neondot": ["aarch32", "aarch64"],
    "neondotfp16arith": ["aarch32", "aarch64"],
    "neoni8mm": ["aarch64"],
    "neonsme": ["aarch64"],
    "neonsme2": ["aarch64"],
    "sse": ["x86-32", "x86-64"],
    "sse2": ["x86-32", "x86-64"],
    "ssse3": ["x86-32", "x86-64"],
    "sse41": ["x86-32", "x86-64"],
    "avx": ["x86-32", "x86-64"],
    "f16c": ["x86-32", "x86-64"],
    "fma3": ["x86-32", "x86-64"],
    "avx2": ["x86-32", "x86-64"],
    "avx512f": ["x86-32", "x86-64"],
    "avx512skx": ["x86-32", "x86-64"],
    "avx512vbmi": ["x86-32", "x86-64"],
    "avx512vnni": ["x86-32", "x86-64"],
    "avx512vnnigfni": ["x86-32", "x86-64"],
    "avx512amx": ["x86-32", "x86-64"],
    "avx512fp16": ["x86-32", "x86-64"],
    "avx512bf16": ["x86-64"],
    "avxvnni": ["x86-32", "x86-64"],
    "avxvnniint8": ["x86-32", "x86-64"],
    "avx256skx": ["x86-32", "x86-64"],
    "avx256vnni": ["x86-32", "x86-64"],
    "avx256vnnigfni": ["x86-32", "x86-64"],
    "hexagon": ["hexagon"],
    "hvx": ["hexagon"],
    "rvv": ["riscv"],
    "rvvfp16arith": ["riscv"],
    "wasm32": ["wasm", "wasmsimd"],
    "wasm": ["wasm", "wasmsimd", "wasmrelaxedsimd"],
    "wasmsimd": ["wasmsimd", "wasmrelaxedsimd"],
    "wasmrelaxedsimd": ["wasmrelaxedsimd"],
    "wasmpshufb": ["wasmrelaxedsimd"],
    "wasmsdot": ["wasmrelaxedsimd"],
    "wasmusdot": ["wasmrelaxedsimd"],
    "wasmblendvps": ["wasmrelaxedsimd"],
}

_ISA_TO_ARCH_FLAGS_MAP = {
    "armsimd32": "xnn_arch_arm_v6",
    "fp16arith": "xnn_arch_arm_fp16_arith",
    "neon": "xnn_arch_arm_neon",
    "neonfp16": "xnn_arch_arm_neon_fp16",
    "neonfma": "xnn_arch_arm_neon_fma",
    "neonv8": "xnn_arch_arm_neon_v8",
    "neonfp16arith": "xnn_arch_arm_neon_fp16_arith",
    "neonbf16": "xnn_arch_arm_neon_bf16",
    "neondot": "xnn_arch_arm_neon_dot",
    "neondotfp16arith": "xnn_arch_arm_neon_dot | xnn_arch_arm_neon_fp16_arith",
    "neoni8mm": "xnn_arch_arm_neon_i8mm",
    "neonsme": "xnn_arch_arm_sme",
    "neonsme2": "xnn_arch_arm_sme2",
    "ssse3": "xnn_arch_x86_ssse3",
    "sse41": "xnn_arch_x86_sse4_1",
    "avx": "xnn_arch_x86_avx",
    "f16c": "xnn_arch_x86_f16c",
    "avx2": "xnn_arch_x86_avx2",
    "fma3": "xnn_arch_x86_fma3",
    "avx512f": "xnn_arch_x86_avx512f",
    "avx512skx": "xnn_arch_x86_avx512skx",
    "avx512vbmi": "xnn_arch_x86_avx512vbmi",
    "avx512vnni": "xnn_arch_x86_avx512vnni",
    "avx512vnnigfni": "xnn_arch_x86_avx512vnnigfni",
    "avx512amx": "xnn_arch_x86_avx512amx",
    "avx512fp16": "xnn_arch_x86_avx512fp16",
    "avx512bf16": "xnn_arch_x86_avx512bf16",
    "avxvnni": "xnn_arch_x86_avxvnni",
    "avxvnniint8": "xnn_arch_x86_avxvnniint8",
    "avx256skx": "xnn_arch_x86_avx256skx",
    "avx256vnni": "xnn_arch_x86_avx256vnni",
    "avx256vnnigfni": "xnn_arch_x86_avx256vnnigfni",
    "hvx": "xnn_arch_hvx",
    "rvv": "xnn_arch_riscv_vector",
    "rvvfp16arith": "xnn_arch_riscv_vector_fp16_arith",
    "wasmpshufb": "xnn_arch_wasm_pshufb",
    "wasmsdot": "xnn_arch_wasm_sdot",
    "wasmusdot": "xnn_arch_wasm_usdot",
    "wasmblendvps": "xnn_arch_wasm_blendvps",
}


def isa_hierarchy_map():
  return _ISA_HIERARCHY_MAP


def parse_target_name(target_name):
  arch = list()
  isa = None
  assembly = False
  for target_part in target_name.split("_"):
    if target_part in _ARCH_TO_MACRO_MAP:
      if target_part in _ISA_TO_ARCH_MAP:
        arch = _ISA_TO_ARCH_MAP[target_part]
        isa = target_part
      else:
        arch = [target_part]
    elif target_part in _ISA_TO_ARCH_MAP:
      isa = target_part
    elif target_part == "asm":
      assembly = True
  if isa and not arch:
    arch = _ISA_TO_ARCH_MAP[isa]

  return arch, isa, assembly


def generate_isa_check_macro(isa):
  return "TEST_REQUIRES_ARCH_FLAGS(%s)" % _ISA_TO_ARCH_FLAGS_MAP.get(isa, "0")


def get_arch_flags(isa):
  return _ISA_TO_ARCH_FLAGS_MAP.get(isa, "0")


def arch_to_macro(arch, isa):
  return _ARCH_TO_MACRO_MAP[arch]


def postprocess_test_case(test_case, arch, isa, assembly=False):
  test_case = _remove_duplicate_newlines(test_case)
  if arch:
    guard = " || ".join(arch_to_macro(a, isa) for a in arch)
    if isa in _ISA_TO_MACRO_MAP:
      if len(arch) > 1:
        guard = "%s && (%s)" % (_ISA_TO_MACRO_MAP[isa], guard)
      else:
        guard = "%s && %s" % (_ISA_TO_MACRO_MAP[isa], guard)
    if assembly and "||" in guard:
      guard = "(" + guard + ")"
    if assembly:
      guard += " && XNN_ENABLE_ASSEMBLY"
    return (
        "#if %s\n" % guard
        + _indent(test_case)
        + "\n"
        + "#endif  // %s\n" % guard
    )
  else:
    return test_case


_ISA_HIERARCHY = [
    "sse",
    "sse2",
    "ssse3",
    "sse41",
    "avx",
    "avx2",
    "avx512f",
    "avx512vbmi",
    "avxvnni",
    "avxvnniint8",
    "avx256skx",
    "avx256vnni",
    "avx256vnnigfni",
    "avx512skx",
    "avx512vnni",
    "avx512vnnigfni",
    "avx512fp16",
    "avx512bf16",
    "avx512amx",
    "armsimd32",
    "neon",
    "neonv8",
    "neondot",
    "neondotfp16",
    "neoni8mm",
    "wasm",
    "wasmsimd",
    "wasmrelaxedsimd",
    "rvv",
    "rvvfp16",
    "hexagon",
    "hvx",
]

_ISA_HIERARCHY_MAP = {isa: v for v, isa in enumerate(_ISA_HIERARCHY)}


def overwrite_if_changed(filepath, content):
  txt_changed = True
  if os.path.exists(filepath):
    with codecs.open(filepath, "r", encoding="utf-8") as output_file:
      txt_changed = output_file.read() != content
  if txt_changed:
    with codecs.open(filepath, "w", encoding="utf-8") as output_file:
      output_file.write(content)


def make_multiline_macro(x):
  lines = x.strip().split("\n")
  max_len = max([len(i) for i in lines])
  lines = [i.ljust(max_len) + "\\" for i in lines]
  return "\n".join(lines)[:-1].strip() + "\n"
