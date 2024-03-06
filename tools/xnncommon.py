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
  "rvv": "XNN_ENABLE_RISCV_VECTOR",
  "rvvfp16arith": "XNN_ENABLE_RISCV_FP16_VECTOR",
  "avxvnni": "XNN_ENABLE_AVXVNNI",
  "avx512vnnigfni": "XNN_ENABLE_AVX512VNNIGFNI",
  "avx512amx": "XNN_ENABLE_AVX512AMX",
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
  "neoni8mm": ["aarch32", "aarch64"],
  "sse": ["x86-32", "x86-64"],
  "sse2": ["x86-32", "x86-64"],
  "ssse3": ["x86-32", "x86-64"],
  "sse41": ["x86-32", "x86-64"],
  "avx": ["x86-32", "x86-64"],
  "f16c": ["x86-32", "x86-64"],
  "xop": ["x86-32", "x86-64"],
  "fma3": ["x86-32", "x86-64"],
  "avx2": ["x86-32", "x86-64"],
  "avx512f": ["x86-32", "x86-64"],
  "avx512skx": ["x86-32", "x86-64"],
  "avx512vbmi": ["x86-32", "x86-64"],
  "avx512vnni": ["x86-32", "x86-64"],
  "avx512vnnigfni": ["x86-32", "x86-64"],
  "avx512amx": ["x86-32", "x86-64"],
  "avxvnni": ["x86-32", "x86-64"],
  "rvv": ["riscv"],
  "rvvfp16arith": ["riscv"],
  "wasm32": ["wasm", "wasmsimd"],
  "wasm": ["wasm", "wasmsimd", "wasmrelaxedsimd"],
  "wasmsimd": ["wasmsimd", "wasmrelaxedsimd"],
  "wasmrelaxedsimd": ["wasmrelaxedsimd"],
  "wasmpshufb": ["wasmrelaxedsimd"],
  "wasmsdot": ["wasmrelaxedsimd"],
  "wasmblendvps": ["wasmrelaxedsimd"],
}

_ISA_TO_UTILCHECK_MAP = {
  "armsimd32": "CheckARMV6",
  "fp16arith": "CheckFP16ARITH",
  "neon": "CheckNEON",
  "neonfp16": "CheckNEONFP16",
  "neonfma": "CheckNEONFMA",
  "neonv8": "CheckNEONV8",
  "neonfp16arith": "CheckNEONFP16ARITH",
  "neonbf16": "CheckNEONBF16",
  "neondot": "CheckNEONDOT",
  "neondotfp16arith": "CheckNEONDOT",
  "neoni8mm": "CheckNEONI8MM",
  "ssse3": "CheckSSSE3",
  "sse41": "CheckSSE41",
  "avx": "CheckAVX",
  "f16c": "CheckF16C",
  "xop": "CheckXOP",
  "avx2": "CheckAVX2",
  "fma3": "CheckFMA3",
  "avx512f": "CheckAVX512F",
  "avx512skx": "CheckAVX512SKX",
  "avx512vbmi": "CheckAVX512VBMI",
  "avx512vnni": "CheckAVX512VNNI",
  "avx512vnnigfni": "CheckAVX512VNNIGFNI",
  "avx512amx": "CheckAVX512AMX",
  "avxvnni": "CheckAVXVNNI",
  "rvv": "CheckRVV",
  "rvvfp16arith": "CheckRVVFP16ARITH",
  "wasmpshufb": "CheckWAsmPSHUFB",
  "wasmsdot": "CheckWAsmSDOT",
  "wasmblendvps": "CheckWAsmBLENDVPS",
}

_ISA_TO_CHECK_MAP = {
  "armsimd32": "TEST_REQUIRES_ARM_SIMD32",
  "fp16arith": "TEST_REQUIRES_ARM_FP16_ARITH",
  "neon": "TEST_REQUIRES_ARM_NEON",
  "neonfp16": "TEST_REQUIRES_ARM_NEON_FP16",
  "neonfma": "TEST_REQUIRES_ARM_NEON_FMA",
  "neonv8": "TEST_REQUIRES_ARM_NEON_V8",
  "neonfp16arith": "TEST_REQUIRES_ARM_NEON_FP16_ARITH",
  "neonbf16": "TEST_REQUIRES_ARM_NEON_BF16",
  "neondot": "TEST_REQUIRES_ARM_NEON_DOT",
  "neondotfp16arith": "TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH",
  "neoni8mm": "TEST_REQUIRES_ARM_NEON_I8MM",
  "sse": "TEST_REQUIRES_X86_SSE",
  "sse2": "TEST_REQUIRES_X86_SSE2",
  "ssse3": "TEST_REQUIRES_X86_SSSE3",
  "sse41": "TEST_REQUIRES_X86_SSE41",
  "avx": "TEST_REQUIRES_X86_AVX",
  "f16c": "TEST_REQUIRES_X86_F16C",
  "xop": "TEST_REQUIRES_X86_XOP",
  "avx2": "TEST_REQUIRES_X86_AVX2",
  "fma3": "TEST_REQUIRES_X86_FMA3",
  "avx512f": "TEST_REQUIRES_X86_AVX512F",
  "avx512skx": "TEST_REQUIRES_X86_AVX512SKX",
  "avx512vbmi": "TEST_REQUIRES_X86_AVX512VBMI",
  "avx512vnni": "TEST_REQUIRES_X86_AVX512VNNI",
  "avx512vnnigfni": "TEST_REQUIRES_X86_AVX512VNNIGFNI",
  "avx512amx": "TEST_REQUIRES_X86_AVX512AMX",
  "avxvnni": "TEST_REQUIRES_X86_AVXVNNI",
  "rvv": "TEST_REQUIRES_RISCV_VECTOR",
  "rvvfp16arith": "TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH",
  "wasmpshufb": "TEST_REQUIRES_WASM_PSHUFB",
  "wasmsdot": "TEST_REQUIRES_WASM_SDOT",
  "wasmblendvps": "TEST_REQUIRES_WASM_BLENDVPS",
}


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
  return _ISA_TO_CHECK_MAP.get(isa, "")

def generate_isa_utilcheck_macro(isa):
  return _ISA_TO_UTILCHECK_MAP.get(isa, "")

def arch_to_macro(arch, isa):
  return _ARCH_TO_MACRO_MAP[arch]

def postprocess_test_case(test_case, arch, isa, assembly=False, jit=False):
  test_case = _remove_duplicate_newlines(test_case)
  if arch:
    guard = " || ".join(arch_to_macro(a, isa) for a in arch)
    if isa in _ISA_TO_MACRO_MAP:
      if len(arch) > 1:
        guard = "%s && (%s)" % (_ISA_TO_MACRO_MAP[isa], guard)
      else:
        guard = "%s && %s" % (_ISA_TO_MACRO_MAP[isa], guard)
    if (assembly or jit) and "||" in guard:
      guard = '(' + guard + ')'
    if assembly:
      guard += " && XNN_ENABLE_ASSEMBLY"
    if jit:
      guard += " && XNN_PLATFORM_JIT"
    return "#if %s\n" % guard + _indent(test_case) + "\n" + \
      "#endif  // %s\n" % guard
  else:
    return test_case

_ISA_HIERARCHY = [
  "sse",
  "sse2",
  "ssse3",
  "sse41",
  "avx",
  "avx2",
  "xop",
  "avx512f",
  "avx512skx",
  "avx512vbmi",
  "avxvnni",
  "avx512vnni",
  "avx512vnnigfni",
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
