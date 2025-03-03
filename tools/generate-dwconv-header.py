import argparse
import codecs
import codecs
from collections import defaultdict
import io
import math
import os
import os
import re
import sys
import platform

import xnncommon
from xnncommon import _ARCH_TO_MACRO_MAP, _ISA_TO_MACRO_MAP
import yaml

_DATATYPE_TO_CTYPE_MAP = {
     "s8": "int8_t",
     "u8": "uint8_t",
     "qs8": "int8_t",
     "qu8": "uint8_t",
     "s16": "int16_t",
     "u16": "uint16_t",
     "s32": "int32_t",
     "u32": "uint32_t",
     "s64": "int64_t",
     "u64": "uint64_t",
     "bf16": "xnn_bfloat16",
     "f16": "xnn_float16",
     "f16_f32acc": "xnn_float16",
     "f32": "float",
 }
 
yamls = {
 "f32-dwconv2d-chw": "f32-dwconv2d-chw",
#  "f16-dwconv2d-chw": "f16-dwconv2d-chw",
 }
 
HEADER = """\
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, kernel_height, kernel_width, subsampling, padding, height_tile, width_tile, datatype, params_type, init_params) \\
    XNN_UKERNEL(arch_flags, ukernel, kernel_height, kernel_width, subsampling, padding, height_tile, width_tile, datatype, params_type, init_params)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, kernel_height, kernel_width, subsampling, padding, height_tile, width_tile, datatype, params_type, init_params) \\
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, kernel_height, kernel_width, subsampling, padding, height_tile, width_tile, datatype, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif
"""


FOOTER = """
#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
"""

def split_ukernel_name(name):
    match = re.fullmatch(
        r"xnn_(f16|f32)_dwconv2d_chw_ukernel_(\d+)x(\d+)(s2)?p(\d+)__(.+)_(\d+)x(\d+)(_acc\d+)?", 
        name
    )
    if match is None:
        return None
    
    data_type = match.group(1)
    kernel_height, kernel_width = int(match.group(2)), int(match.group(3))
    subsampling = int(match.group(4)[1:]) if match.group(4) else 1
    padding = int(match.group(5))
    height_tile, width_tile = int(match.group(7)), int(match.group(8))
    arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(6))

    return data_type, kernel_height, kernel_width, subsampling, padding, arch, isa, height_tile, width_tile, assembly

isas = {
     "v6": "xnn_arch_arm_v6",
     "armsimd32": "xnn_arch_arm_v6",
     "vfpv2": "xnn_arch_arm_vfpv2",
     "vfpv3": "xnn_arch_arm_vfpv3",
     "neon": "xnn_arch_arm_neon",
     "neonfp16": "xnn_arch_arm_neon_fp16",
     "neonfma": "xnn_arch_arm_neon_fma",
     "neonv8": "xnn_arch_arm_neon_v8",
     "fp16arith": "xnn_arch_arm_fp16_arith",
     "neonfp16arith": "xnn_arch_arm_neon_fp16_arith",
     "neondotfp16arith":"xnn_arch_arm_neon_dot_fp16_arith",
     "neonbf16": "xnn_arch_arm_neon_bf16",
     "neondot": "xnn_arch_arm_neon_dot",
     "neon_i8mm": "xnn_arch_arm_neon_i8mm",
     "neoni8mm": "xnn_arch_arm_neon_i8mm",
     "sse": "0",
     "sse2": "0",
     "ssse3": "xnn_arch_x86_ssse3",
     "sse41": "xnn_arch_x86_sse4_1",
     "avx": "xnn_arch_x86_avx",
     "f16c": "xnn_arch_x86_f16c",
     "fma3": "xnn_arch_x86_fma3",
     "avx2": "xnn_arch_x86_avx2",
     "avx512f": "xnn_arch_x86_avx512f",
     "avx512vbmi": "xnn_arch_x86_avx512vbmi",
     "avx512skx": "xnn_arch_x86_avx512skx",
     "avx512vnni": "xnn_arch_x86_avx512vnni",
     "avx512vnnigfni": "xnn_arch_x86_avx512vnnigfni",
     "avx512amx": "xnn_arch_x86_avx512amx",
     "avx512fp16": "xnn_arch_x86_avx512fp16",
     "avxvnni": "xnn_arch_x86_avxvnni",
     "avxvnniint8": "xnn_arch_x86_avxvnniint8",
     "avx256skx": "xnn_arch_x86_avx256skx",
     "avx256vnni": "xnn_arch_x86_avx256vnni",
     "avx256vnnigfni": "xnn_arch_x86_avx256vnnigfni",
     "rvv": "xnn_arch_riscv_vector",
     "rvvfp16arith": "xnn_arch_riscv_vector_fp16_arith",
     "vlenb": "xnn_arch_riscv_vlenb",
     # xnn_arch_vsx = 1 << 0,
     # xnn_arch_vsx3 = 1 << 1,
     # xnn_arch_mma = 1 << 2,
     "is_x86": "xnn_arch_wasm_is_x86",
     "wasmblendvps": "xnn_arch_wasm_blendvps",
     "pshufb": "xnn_arch_wasm_pshufb",
     "sdot": "xnn_arch_wasm_sdot",
     "usdot": "xnn_arch_wasm_usdot",
     "fma": "xnn_arch_wasm_fma",
     "wasmpshufb": "xnn_arch_wasm_pshufb",
     "wasmsdot": "xnn_arch_wasm_sdot",
     "wasmusdot": "xnn_arch_wasm_usdot",
     "wasmfma": "xnn_arch_wasm_fma",
     "hvx": "xnn_arch_hvx",
     "wasm": "0",
     "wasmsimd": "0",
     "wasm32": "0",
     "wasmrelaxedsimd": "0",
     None: "0",
 }
 
yamls_inverted = defaultdict(list)

for i in yamls.items():
 yamls_inverted[i[1]].append(i[0])

files = []
hdrs = []
for i in yamls_inverted.items():
 for j in i[1]:
  src_path = "/home/mcw/Documents/Google_Project/Internal_XNNPACK/src/" + i[0] + "/" + j + ".h"
  dst = src_path

  hdrs.append(src_path)
  files.append(j)

  output = HEADER
  in_define = ""

  src = "/home/mcw/Documents/Google_Project/Internal_XNNPACK/test/" + j + ".yaml"

  with codecs.open(src, "r", encoding="utf-8") as spec_file:
   spec_yaml = yaml.safe_load(spec_file)
   if not isinstance(spec_yaml, list):
    raise ValueError("expected a list of micro-kernels in the spec")

   for ukernel_spec in spec_yaml:
    name = ukernel_spec["name"]
    init_fn = ukernel_spec.get("init", "NULL")

    data_type, kernel_height, kernel_width, subsampling, padding, arch, isa, height_tile, width_tile, assembly = split_ukernel_name(name)
    ctype = _DATATYPE_TO_CTYPE_MAP[data_type]

    guard = _ISA_TO_MACRO_MAP.get(isa, "")
    isa = isas[isa]
    arch = [_ARCH_TO_MACRO_MAP[i] for i in arch]

    if arch:
     if guard != "":
      define = "#if " + guard + " && (" + " || ".join(arch) + ")\n"
     else:
      define = "#if " + " || ".join(arch) + "\n"
    else:
     if guard != "":
      define = "#if " + guard + "\n"
     else:
      define = ""

    if in_define != define:
     if in_define != "":
      output += "#endif  // " + in_define[4:]
     output += "\n"
     output += define
     in_define = define

    params_type = "struct xnn_f32_minmax_params"

    output += "XNN_UKERNEL_WITH_PARAMS(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n" % (
     isa,
     name,
     kernel_height, 
     kernel_width, 
     subsampling, 
     padding, 
     height_tile, 
     width_tile, 
     ctype, 
     params_type, 
     init_fn
    )

   if in_define != "":
    output += "#endif  // " + in_define[4:] + "\n"

  output += FOOTER

  with codecs.open(dst, "w", encoding="utf-8") as output_file:
   output_file.write(output)


print("MICROKERNEL_DEPS = [")
print(",\n".join(['    "' + i + '"' for i in hdrs]))
print("]")
print(" ".join(files))
