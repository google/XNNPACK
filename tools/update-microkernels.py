#!/usr/bin/env python
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import re
import sys
import io


TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))

ISA_LIST = frozenset({
  'armsimd32',
  'avx',
  'avx2',
  'avx512f',
  'avx512skx',
  'avx512vbmi',
  'f16c',
  'fma',
  'fma3',
  'fp16arith',
  'hexagon',
  'neon',
  'neonbf16',
  'neondot',
  'neoni8mm',
  'neonfma',
  'neonfp16',
  'neonfp16arith',
  'neonv8',
  'rvv',
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
  'wasmsimd',
  'xop',
})

ISA_MAP = {
  'wasmblendvps': 'wasmrelaxedsimd',
  'wasmpshufb': 'wasmrelaxedsimd',
  'wasmsdot': 'wasmrelaxedsimd',
}

ARCH_LIST = frozenset({
  'aarch32',
  'aarch64',
  'wasm32',
  'wasmsimd32',
  'wasmrelaxedsimd32',
})

parser = argparse.ArgumentParser(
  description='Utility for re-generating microkernel lists')


import re

NUMBERS_REGEX = re.compile('([0-9]+)')

def human_sort_key(text):
  return [int(token) if token.isdigit() else token.lower() for token in NUMBERS_REGEX.split(text)]


def overwrite_if_changed(filepath, mem_file):
  file_changed = True
  if os.path.exists(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
      mem_file.seek(0)
      file_changed = f.read() != mem_file.read()
  if file_changed:
    with open(filepath, 'w', encoding='utf-8') as f:
      mem_file.seek(0)
      f.write(mem_file.read())

def make_variable_name(prefix, key, suffix):
  return '_'.join(token for token in [prefix, key.upper(), suffix] if token)

def write_grouped_microkernels_bzl(file, microkernels, prefix, suffix):
  for key in sorted(microkernels):
    variable_name = make_variable_name(prefix, key, suffix)
    file.write(f'\n{variable_name} = [\n')
    for microkernel in sorted(microkernels[key], key=human_sort_key):
      file.write(f'    "{microkernel}",\n')
    file.write(']\n')

def write_grouped_microkernels_cmake(file, microkernels, prefix, suffix):
  for key in sorted(microkernels):
    variable_name = make_variable_name(prefix, key, suffix)
    file.write(f'\nSET({variable_name}')
    for microkernel in sorted(microkernels[key], key=human_sort_key):
      file.write(f'\n  {microkernel}')
    file.write(')\n')

def main(args):
  options = parser.parse_args(args)
  root_dir = os.path.normpath(os.path.join(TOOLS_DIR, '..'))
  src_dir = os.path.join(root_dir, 'src')
  ignore_roots = {
    src_dir,
    os.path.join(src_dir, 'amalgam', 'gen'),
    os.path.join(src_dir, 'configs'),
    os.path.join(src_dir, 'enums'),
    os.path.join(src_dir, 'jit'),
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
  asm_microkernels_per_arch = {arch: [] for arch in ARCH_LIST}
  jit_microkernels_per_arch = {arch: [] for arch in ARCH_LIST}
  for root, dirs, files in os.walk(src_dir, topdown=False):
    if root in ignore_roots:
      continue
    for name in files:
      if name.endswith('.in'):
        continue
      basename, ext = os.path.splitext(name)
      if ext == '.sollya':
        continue
      filepath = os.path.join(os.path.relpath(root, root_dir), name)

      if ext == '.c':
        arch = None
        for component in basename.split('-'):
          if component in ARCH_LIST:
            arch = component
          elif component in ISA_LIST:
            isa = ISA_MAP.get(component, component)
            key = isa if arch is None else f'{isa}_{arch}'
            c_microkernels_per_isa[key].append(filepath)
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
      elif ext == '.cc':
        for component in basename.split('-'):
          if component in ARCH_LIST:
            jit_microkernels_per_arch[component].append(filepath)
            break
        else:
          print('Unknown architecture for JIT microkernel %s' % filepath)

  with io.StringIO() as microkernels_bzl:
    microkernels_bzl.write('''\
"""
Microkernel filenames lists.

Auto-generated file. Do not edit!
  Generator: tools/update-microkernels.py
"""
''')
    write_grouped_microkernels_bzl(microkernels_bzl, c_microkernels_per_isa, "ALL", "MICROKERNEL_SRCS")
    write_grouped_microkernels_bzl(microkernels_bzl, asm_microkernels_per_arch, "", "ASM_MICROKERNEL_SRCS")
    write_grouped_microkernels_bzl(microkernels_bzl, jit_microkernels_per_arch, "", "JIT_MICROKERNEL_SRCS")
    overwrite_if_changed(os.path.join(root_dir, 'microkernels.bzl'), microkernels_bzl)

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
    write_grouped_microkernels_cmake(microkernels_cmake, c_microkernels_per_isa, "ALL", "MICROKERNEL_SRCS")
    write_grouped_microkernels_cmake(microkernels_cmake, asm_microkernels_per_arch, "", "ASM_MICROKERNEL_SRCS")
    write_grouped_microkernels_cmake(microkernels_cmake, jit_microkernels_per_arch, "", "JIT_MICROKERNEL_SRCS")
    overwrite_if_changed(os.path.join(root_dir, "cmake", 'microkernels.cmake'), microkernels_cmake)

if __name__ == '__main__':
  main(sys.argv[1:])
