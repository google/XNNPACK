#!/usr/bin/env python
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import io
import itertools
import os
import re
import sys
import tempfile

import xnncommon


TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))

ISA_LIST = frozenset({
    'armsimd32',
    'avx',
    'avx2',
    'avx512f',
    'avx512skx',
    'avx512vbmi',
    'avx512vnni',
    'avx512vnnigfni',
    'avx512amx',
    'avx512fp16',
    'avxvnni',
    'avx256skx',
    'avx256vnni',
    'avx256vnnigfni',
    'f16c',
    'fma3',
    'fp16arith',
    'hexagon',
    'hvx',
    'neon',
    'neonbf16',
    'neondot',
    'neondotfp16arith',
    'neoni8mm',
    'neonfma',
    'neonfp16',
    'neonfp16arith',
    'neonv8',
    'rvv',
    'rvvfp16arith',
    'scalar',
    'sse',
    'sse2',
    'sse41',
    'ssse3',
    'wasm',
    'wasmblendvps',
    'wasmrelaxedsimd',
    'wasmpshufb',
    'wasmsdot',
    'wasmusdot',
    'wasmsimd',
})

ISA_MAP = {
    'wasmblendvps': 'wasmrelaxedsimd',
    'wasmpshufb': 'wasmrelaxedsimd',
    'wasmsdot': 'wasmrelaxedsimd',
    'wasmusdot': 'wasmrelaxedsimd',
}

ARCH_LIST = frozenset({
    'aarch32',
    'aarch64',
    'wasm32',
    'wasmsimd32',
    'wasmrelaxedsimd32',
})

ISA_TO_HEADER_MAP = {
    'armsimd32': 'arm_acle.h',
    'avx': 'immintrin.h',
    'avx2': 'immintrin.h',
    'avx512f': 'immintrin.h',
    'avx512skx': 'immintrin.h',
    'avx512vbmi': 'immintrin.h',
    'avx512vnni': 'immintrin.h',
    'avx512vnnigfni': 'immintrin.h',
    'avx512amx': 'immintrin.h',
    'avx512fp16': 'immintrin.h',
    'avxvnni': 'immintrin.h',
    'avx256skx': 'immintrin.h',
    'avx256vnni': 'immintrin.h',
    'avx256vnnigfni': 'immintrin.h',
    'f16c': 'immintrin.h',
    'fma3': 'immintrin.h',
    'fp16arith': 'arm_fp16.h',
    'neon': 'arm_neon.h',
    'neonbf16': 'arm_neon.h',
    'neondot': 'arm_neon.h',
    'neondotfp16arith': 'arm_neon.h',
    'neoni8mm': 'arm_neon.h',
    'neonfma': 'arm_neon.h',
    'neonfp16': 'arm_neon.h',
    'neonfp16arith': 'arm_neon.h',
    'neonv8': 'arm_neon.h',
    'rvv': 'riscv_vector.h',
    'rvvfp16arith': 'riscv_vector.h',
    'sse': 'immintrin.h',
    'sse2': 'immintrin.h',
    'sse41': 'immintrin.h',
    'ssse3': 'immintrin.h',
    'wasmrelaxedsimd': 'wasm_simd128.h',
    'wasmsimd': 'wasm_simd128.h',
}

MICROKERNEL_NAME_REGEX = re.compile(
    r'\bxnn_(?:[a-z0-9]+(?:_[a-z0-9]+)*)_ukernel(?:_[a-z0-9]+)*__(?:[a-z0-9]+(?:_[a-z0-9]+)*)\b'
)

VERIFICATION_IGNORE_SUBDIRS = {
    os.path.join('src', 'math'),
    os.path.join('src', 'math', 'gen'),
    os.path.join('src', 'qs8-requantization'),
    os.path.join('src', 'qu8-requantization'),
    os.path.join('src', 'xnnpack', 'simd'),
}

UNWANTED_INCLUDES = (
    '<arm_acle.h>',
    '<arm_fp16.h>',
    '<arm_neon.h>',
    '<emmintrin.h>',
    '<immintrin.h>',
    '<nmmintrin.h>',
    '<smmintrin.h>',
    '<tmmintrin.h>',
    '<xmmintrin.h>',
    '<riscv_vector.h>',
    '<wasm_simd128.h>',
)

parser = argparse.ArgumentParser(
    description='Utility for re-generating microkernel lists'
)


def human_sort_key(text):
  return [
      token.zfill(10) if token.isdigit() else token.lower()
      for token in re.split(r'(\d+|\W)', text)
      if token
  ]


def _discard(l, val):
  while val in l:
    l.remove(val)


def make_variable_name(prefix, key, suffix):
  return '_'.join(token for token in [prefix, key.upper(), suffix] if token)


def write_grouped_microkernels_bzl(file, key, microkernels, prefix, suffix):
  if key not in microkernels:
    return []
  variable_name = make_variable_name(prefix, key, suffix)
  file.write(f'\n{variable_name} = [\n')
  for microkernel in sorted(microkernels[key], key=human_sort_key):
    file.write(f'    "{microkernel}",\n')
  file.write(']\n')
  return [variable_name]


def write_grouped_microkernels_cmake(file, key, microkernels, prefix, suffix):
  if key not in microkernels:
    return
  variable_name = make_variable_name(prefix, key, suffix)
  file.write(f'\nSET({variable_name}')
  for microkernel in sorted(microkernels[key], key=human_sort_key):
    file.write(f'\n  {microkernel}')
  file.write(')\n')


def main(args):
  options = parser.parse_args(args)
  root_dir = os.path.normpath(os.path.join(TOOLS_DIR, '..'))
  src_dir = os.path.join(root_dir, 'src')
  configs_dir = os.path.join(src_dir, 'configs')
  ignore_roots = {
      src_dir,
      os.path.join(src_dir, 'configs'),
      os.path.join(src_dir, 'enums'),
      os.path.join(src_dir, 'operators'),
      os.path.join(src_dir, 'subgraph'),
      os.path.join(src_dir, 'tables'),
      os.path.join(src_dir, 'xnnpack'),
  }
  c_microkernels_per_isa = {isa: [] for isa in ISA_LIST if isa not in ISA_MAP}
  c_microkernels_per_isa['neon_aarch64'] = list()
  c_microkernels_per_isa['neondot_aarch64'] = list()
  c_microkernels_per_isa['neonfma_aarch64'] = list()
  c_microkernels_per_isa['neonfp16arith_aarch64'] = list()
  c_microkernels_per_isa['neonbf16_aarch64'] = list()
  c_microkernels_per_isa['neoni8mm_aarch64'] = list()
  temp_c_microkernels_per_isa = {isa: [] for isa in c_microkernels_per_isa}
  asm_microkernels_per_arch = {arch: [] for arch in ARCH_LIST}
  microkernel_name_to_filename = dict()
  microkernel_temp_dir = tempfile.mkdtemp()
  for root, dirs, files in os.walk(src_dir, topdown=False):
    if root in ignore_roots:
      continue

    for name in files:
      if name.endswith('.in'):
        continue
      basename, ext = os.path.splitext(name)
      if ext == '.sollya':
        continue

      subdir = os.path.relpath(root, root_dir)
      filepath = os.path.join(subdir, name)
      temp_filepaths = []

      if ext == '.c':
        arch = None
        for component in basename.split('-'):
          if component in ARCH_LIST:
            arch = component
          elif component in ISA_LIST:
            isa = ISA_MAP.get(component, component)
            key = isa if arch is None else f'{isa}_{arch}'
            c_microkernels_per_isa[key].append(filepath)
            if temp_filepaths:
              temp_c_microkernels_per_isa[key].extend(temp_filepaths)
            break
        else:
          print('Unknown ISA for C microkernel %s' % filepath)
      elif ext == '.S':
        for component in basename.split('-'):
          if component in ARCH_LIST:
            asm_microkernels_per_arch[component].append(filepath)
            break
        else:
          print('Unknown architecture for assembly microkernel %s' % filepath)

  with io.StringIO() as microkernels_bzl:
    microkernels_bzl.write('''\
"""
Microkernel filenames lists.

Auto-generated file. Do not edit!
  Generator: tools/update-microkernels.py
"""

''')
    keys = set(c_microkernels_per_isa.keys()).union(asm_microkernels_per_arch.keys())
    keys = sorted(keys, key=lambda key: key + '_microkernels.bzl')
    exports = ['\n']
    for key in keys:
      arch_microkernels_bzl_filename = key + '_microkernels.bzl'
      with io.StringIO() as arch_microkernels_bzl:
        arch_microkernels_bzl.write(f'''\
"""
Microkernel filenames lists for {key}.

Auto-generated file. Do not edit!
  Generator: tools/update-microkernels.py
"""
''')
        vars = write_grouped_microkernels_bzl(
            arch_microkernels_bzl,
            key,
            c_microkernels_per_isa,
            'ALL',
            'MICROKERNEL_SRCS',
        )
        vars = vars + write_grouped_microkernels_bzl(
            arch_microkernels_bzl,
            key,
            asm_microkernels_per_arch,
            '',
            'ASM_MICROKERNEL_SRCS',
        )
        arch_microkernels_bzl.seek(0)
        xnncommon.overwrite_if_changed(
            os.path.join(root_dir, 'gen', arch_microkernels_bzl_filename),
            arch_microkernels_bzl.read(),
        )
        imports = ', '.join(f'_{var} = "{var}"' for var in vars)
        microkernels_bzl.write(
            f'load("{arch_microkernels_bzl_filename}", {imports})\n'
        )
        for var in vars:
          exports.append(f'{var} = _{var}\n')
    microkernels_bzl.write(''.join(exports))
    microkernels_bzl.seek(0)
    xnncommon.overwrite_if_changed(
        os.path.join(root_dir, 'gen', 'microkernels.bzl'),
        microkernels_bzl.read(),
    )

  with io.StringIO() as microkernels_cmake:
    microkernels_cmake.write("""\
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists.
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py

""")
    keys = sorted(
        set(c_microkernels_per_isa.keys()).union(asm_microkernels_per_arch.keys())
    )
    for key in keys:
      arch_microkernels_cmake_filename = key + '_microkernels.cmake'
      with io.StringIO() as arch_microkernels_cmake:
        arch_microkernels_cmake.write(f"""\
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for {key}
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py

""")
        write_grouped_microkernels_cmake(
            arch_microkernels_cmake,
            key,
            c_microkernels_per_isa,
            'ALL',
            'MICROKERNEL_SRCS',
        )
        write_grouped_microkernels_cmake(
            arch_microkernels_cmake,
            key,
            asm_microkernels_per_arch,
            '',
            'ASM_MICROKERNEL_SRCS',
        )
        arch_microkernels_cmake.seek(0)
        xnncommon.overwrite_if_changed(
            os.path.join(
                root_dir, 'cmake', 'gen', arch_microkernels_cmake_filename
            ),
            arch_microkernels_cmake.read(),
        )
        microkernels_cmake.write(
            f'INCLUDE(cmake/gen/{arch_microkernels_cmake_filename})\n'
        )

    microkernels_cmake.seek(0)
    xnncommon.overwrite_if_changed(
        os.path.join(root_dir, 'cmake', 'gen', 'microkernels.cmake'),
        microkernels_cmake.read(),
    )


if __name__ == '__main__':
  main(sys.argv[1:])
