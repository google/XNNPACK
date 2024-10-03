#!/usr/bin/env python
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import codecs
import io
import os
import re
import sys

parser = argparse.ArgumentParser(
    description='Utility for re-generating microkernel lists'
)
parser.add_argument(
    '-o',
    '--output',
    required=False,
    help='Directory for generated files.',
)
parser.add_argument(
    '-r',
    '--root_dir',
    required=False,
    help=(
        'Start search for files at this directory instead of the script'
        ' directory.'
    ),
)

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))

_ISA_LIST = frozenset({
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
    'avxvnniint8',
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
    'neonsme',
    'neonsme2',
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

_ISA_MAP = {
    'wasmblendvps': 'wasmrelaxedsimd',
    'wasmpshufb': 'wasmrelaxedsimd',
    'wasmsdot': 'wasmrelaxedsimd',
    'wasmusdot': 'wasmrelaxedsimd',
}

_ARCH_LIST = frozenset({
    'aarch32',
    'aarch64',
    'wasm32',
    'wasmsimd32',
    'wasmrelaxedsimd32',
})

_MICROKERNEL_NAME_REGEX = re.compile(
    r'\bxnn_(?:[a-z0-9]+(?:_[a-z0-9]+)*)_ukernel(?:_[a-z0-9]+)*__(?:[a-z0-9]+(?:_[a-z0-9]+)*)\b'
)

_VERIFICATION_IGNORE_SUBDIRS = {
    os.path.join('src', 'qs8-requantization'),
    os.path.join('src', 'qu8-requantization'),
    os.path.join('src', 'xnnpack', 'simd'),
}


def overwrite_if_changed(filepath, content):
  if isinstance(content, io.IOBase):
    content.seek(0)
    content = content.read()
  txt_changed = True
  if os.path.exists(filepath):
    with codecs.open(filepath, 'r', encoding='utf-8') as output_file:
      txt_changed = output_file.read() != content
  if txt_changed:
    with codecs.open(filepath, 'w', encoding='utf-8') as output_file:
      output_file.write(content)


def human_sort_key(text):
  return [
      token.zfill(10) if token.isdigit() else token.lower()
      for token in re.split(r'(\d+|\W)', text)
      if token
  ]


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
    return []
  variable_name = make_variable_name(prefix, key, suffix)
  file.write(f'\nSET({variable_name}')
  for microkernel in sorted(microkernels[key], key=human_sort_key):
    file.write(f'\n  {microkernel}')
  file.write(')\n')
  return [variable_name]


def main(args):
  options = parser.parse_args(args)
  root_dir = (
      options.root_dir
      if options.root_dir
      else os.path.normpath(os.path.join(TOOLS_DIR, '..'))
  )
  src_dir = os.path.join(root_dir, 'src')
  configs_dir = os.path.join(src_dir, 'configs')
  dst_dir = options.output if options.output else root_dir
  bzl_gen_dir = os.path.join(dst_dir, 'gen')
  if not os.path.exists(bzl_gen_dir):
    os.makedirs(bzl_gen_dir, exist_ok=True)
  cmake_gen_dir = os.path.join(dst_dir, 'cmake', 'gen')
  if not os.path.exists(cmake_gen_dir):
    os.makedirs(cmake_gen_dir, exist_ok=True)
  ignore_roots = {
      src_dir,
      os.path.join(src_dir, 'configs'),
      os.path.join(src_dir, 'enums'),
      os.path.join(src_dir, 'operators'),
      os.path.join(src_dir, 'subgraph'),
      os.path.join(src_dir, 'tables'),
      os.path.join(src_dir, 'xnnpack'),
  }
  c_microkernels_per_isa = {isa: [] for isa in _ISA_LIST if isa not in _ISA_MAP}
  c_microkernels_per_isa['neon_aarch64'] = list()
  c_microkernels_per_isa['neondot_aarch64'] = list()
  c_microkernels_per_isa['neonfma_aarch64'] = list()
  c_microkernels_per_isa['neonfp16arith_aarch64'] = list()
  asm_microkernels_per_arch = {arch: [] for arch in _ARCH_LIST}
  microkernel_name_to_filename = dict()
  for root, _, files in os.walk(src_dir, topdown=False):
    if root in ignore_roots:
      continue

    for name in files:
      if name.endswith('.in'):
        continue
      if name.endswith('.h'):
        continue
      basename, ext = os.path.splitext(name)
      if ext == '.sollya':
        continue

      subdir = os.path.relpath(root, root_dir)
      filepath = os.path.join(subdir, name)

      # Build microkernel name -> microkernel filepath mapping
      with open(os.path.join(root_dir, filepath), 'r', encoding='utf-8') as f:
        content = f.read()
        microkernels = re.findall(_MICROKERNEL_NAME_REGEX, content)
        if not microkernels:
          if subdir in _VERIFICATION_IGNORE_SUBDIRS:
            microkernel_name_to_filename[
                os.path.splitext(os.path.basename(filepath))[0]
            ] = filepath
          else:
            print('No microkernel found in %s' % filepath)
        else:
          microkernels = sorted(set(microkernels))
          if len(microkernels) > 1:
            if ext == '.cc':
              # In generators it is normal to have multiple implementations in
              # the same file. We need to filter out non-generators though.
              microkernels = sorted(
                  filter(
                      lambda fn: fn.startswith('xnn_generate_'), microkernels
                  )
              )
              if (
                  not microkernels
                  and subdir not in _VERIFICATION_IGNORE_SUBDIRS
              ):
                print('No microkernel generator found in %s' % filepath)

              for microkernel in microkernels:
                if microkernel in microkernel_name_to_filename:
                  print(
                      'Duplicate microkernel generator definition: %s and %s'
                      % (microkernel_name_to_filename[microkernel], filepath)
                  )
                else:
                  microkernel_name_to_filename[microkernel] = filepath
            else:
              # Extract the individual function names and their offsets.
              matches = list(re.finditer(_MICROKERNEL_NAME_REGEX, content))

              # Write them to temporary files.
              for k, match in enumerate(matches):
                microkernel = match.group(0)
                if microkernel in microkernel_name_to_filename:
                  print(
                      'Duplicate microkernel definition: %s and %s (%ith'
                      ' function)'
                      % (
                          microkernel_name_to_filename[microkernel],
                          filepath,
                          k,
                      )
                  )
                microkernel_name_to_filename[microkernel] = filepath

          else:
            microkernel = microkernels[0]
            if microkernel in microkernel_name_to_filename:
              print(
                  'Duplicate microkernel definition: %s and %s'
                  % (microkernel_name_to_filename[microkernel], filepath)
              )
            else:
              microkernel_name_to_filename[microkernel] = filepath

      if ext == '.c':
        arch = None
        for component in basename.split('-'):
          if component in _ARCH_LIST:
            arch = component
          elif component in _ISA_LIST:
            isa = _ISA_MAP.get(component, component)
            key = isa if arch is None else f'{isa}_{arch}'
            c_microkernels_per_isa[key].append(filepath)
            break
        else:
          print('Unknown ISA for C microkernel %s' % filepath)
      elif ext == '.S':
        for component in basename.split('-'):
          if component in _ARCH_LIST:
            asm_microkernels_per_arch[component].append(filepath)
            break
        else:
          print('Unknown architecture for assembly microkernel %s' % filepath)

  # Collect filenames of production microkernels as a set
  prod_microkernels = set()
  for configs_filepath in os.listdir(configs_dir):
    with open(
        os.path.join(configs_dir, configs_filepath), 'r', encoding='utf-8'
    ) as config_file:
      content = config_file.read()
      microkernels = re.findall(_MICROKERNEL_NAME_REGEX, content)
      prod_microkernels.update(microkernels)
  prod_microkernels = set(
      map(microkernel_name_to_filename.get, prod_microkernels)
  )

  # Split the microkernels into prod/test.
  prod_c_microkernels_per_isa = dict()
  non_prod_c_microkernels_per_isa = dict()
  for arch, microkernels in c_microkernels_per_isa.items():
    prod_c_microkernels_per_isa[arch] = [
        v for v in microkernels if v in prod_microkernels
    ]
    non_prod_c_microkernels_per_isa[arch] = [
        v for v in microkernels if v not in prod_microkernels
    ]
  prod_asm_microkernels_per_arch = dict()
  non_prod_asm_microkernels_per_arch = dict()
  for arch, microkernels in asm_microkernels_per_arch.items():
    prod_asm_microkernels_per_arch[arch] = [
        v for v in microkernels if v in prod_microkernels
    ]
    non_prod_asm_microkernels_per_arch[arch] = [
        v for v in microkernels if v not in prod_microkernels
    ]

  with io.StringIO() as microkernels_bzl:
    microkernels_bzl.write('''\
"""
Microkernel filenames lists.

Auto-generated file. Do not edit!
  Generator: tools/update-microkernels.py
"""

''')
    prod_c_vars_per_arch = dict()
    non_prod_c_vars_per_arch = dict()
    prod_asm_vars_per_arch = dict()
    non_prod_asm_vars_per_arch = dict()
    keys = set(c_microkernels_per_isa.keys()).union(
        asm_microkernels_per_arch.keys()
    )
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
        prod_c_vars = write_grouped_microkernels_bzl(
            arch_microkernels_bzl,
            key,
            prod_c_microkernels_per_isa,
            'PROD',
            'MICROKERNEL_SRCS',
        )
        if prod_c_vars:
          prod_c_vars_per_arch[key] = prod_c_vars
        non_prod_c_vars = write_grouped_microkernels_bzl(
            arch_microkernels_bzl,
            key,
            non_prod_c_microkernels_per_isa,
            'NON_PROD',
            'MICROKERNEL_SRCS',
        )
        if non_prod_c_vars:
          non_prod_c_vars_per_arch[key] = non_prod_c_vars
        c_vars = prod_c_vars + non_prod_c_vars
        if c_vars:
          all_srcs = make_variable_name('ALL', key, 'MICROKERNEL_SRCS')
          arch_microkernels_bzl.write(f'\n{all_srcs} = {" + ".join(c_vars)}\n')
          c_vars.append(all_srcs)
        prod_asm_vars = write_grouped_microkernels_bzl(
            arch_microkernels_bzl,
            key,
            prod_asm_microkernels_per_arch,
            'PROD',
            'ASM_MICROKERNEL_SRCS',
        )
        if prod_asm_vars:
          prod_asm_vars_per_arch[key] = prod_asm_vars
        non_prod_asm_vars = write_grouped_microkernels_bzl(
            arch_microkernels_bzl,
            key,
            non_prod_asm_microkernels_per_arch,
            'NON_PROD',
            'ASM_MICROKERNEL_SRCS',
        )
        if non_prod_asm_vars:
          non_prod_asm_vars_per_arch[key] = non_prod_asm_vars
        asm_vars = prod_asm_vars + non_prod_asm_vars
        if asm_vars:
          all_srcs = make_variable_name('', key, 'ASM_MICROKERNEL_SRCS')
          arch_microkernels_bzl.write(
              f'\n{all_srcs} = {" + ".join(asm_vars)}\n'
          )
          asm_vars.append(all_srcs)
        all_vars = c_vars + asm_vars
        imports = ', '.join(f'_{var} = "{var}"' for var in sorted(all_vars))
        microkernels_bzl.write(
            f'load("{arch_microkernels_bzl_filename}", {imports})\n'
        )
        for var in all_vars:
          exports.append(f'{var} = _{var}\n')

        overwrite_if_changed(
            os.path.join(bzl_gen_dir, arch_microkernels_bzl_filename),
            arch_microkernels_bzl,
        )

    # Generate dictionaries of microkernel lists per arch.
    microkernels_bzl.write(''.join(sorted(exports)))
    microkernels_bzl.write('\nPROD_C_SRCS_FOR_ARCH = {\n')
    for key, var_list in prod_c_vars_per_arch.items():
      microkernels_bzl.write(f'    "{key}": {" + ".join(var_list)},\n')
    microkernels_bzl.write('}\n')
    microkernels_bzl.write('\nNON_PROD_C_SRCS_FOR_ARCH = {\n')
    for key, var_list in non_prod_c_vars_per_arch.items():
      microkernels_bzl.write(f'    "{key}": {" + ".join(var_list)},\n')
    microkernels_bzl.write('}\n')
    microkernels_bzl.write('\nPROD_ASM_SRCS_FOR_ARCH = {\n')
    for key, var_list in prod_asm_vars_per_arch.items():
      microkernels_bzl.write(f'    "{key}": {" + ".join(var_list)},\n')
    microkernels_bzl.write('}\n')
    microkernels_bzl.write('\nNON_PROD_ASM_SRCS_FOR_ARCH = {\n')
    for key, var_list in non_prod_asm_vars_per_arch.items():
      microkernels_bzl.write(f'    "{key}": {" + ".join(var_list)},\n')
    microkernels_bzl.write('}\n')
    microkernels_bzl.write("""
def prod_c_srcs_for_arch(arch):
    return PROD_C_SRCS_FOR_ARCH.get(arch, [])

def non_prod_c_srcs_for_arch(arch):
    return NON_PROD_C_SRCS_FOR_ARCH.get(arch, [])

def all_c_srcs_for_arch(arch):
    return prod_c_srcs_for_arch(arch) + non_prod_c_srcs_for_arch(arch)

def prod_asm_srcs_for_arch(arch):
    return PROD_ASM_SRCS_FOR_ARCH.get(arch, [])

def non_prod_asm_srcs_for_arch(arch):
    return NON_PROD_ASM_SRCS_FOR_ARCH.get(arch, [])

def all_asm_srcs_for_arch(arch):
    return prod_asm_srcs_for_arch(arch) + non_prod_asm_srcs_for_arch(arch)

def prod_srcs_for_arch(arch):
    return prod_c_srcs_for_arch(arch) + prod_asm_srcs_for_arch(arch)

def non_prod_srcs_for_arch(arch):
    return non_prod_c_srcs_for_arch(arch) + non_prod_asm_srcs_for_arch(arch)

def all_srcs_for_arch(arch):
    return all_c_srcs_for_arch(arch) + all_asm_srcs_for_arch(arch)
""")

    overwrite_if_changed(
        os.path.join(bzl_gen_dir, 'microkernels.bzl'), microkernels_bzl
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
        set(c_microkernels_per_isa.keys()).union(
            asm_microkernels_per_arch.keys()
        )
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
        c_vars = write_grouped_microkernels_cmake(
            arch_microkernels_cmake,
            key,
            prod_c_microkernels_per_isa,
            'PROD',
            'MICROKERNEL_SRCS',
        )
        c_vars += write_grouped_microkernels_cmake(
            arch_microkernels_cmake,
            key,
            non_prod_c_microkernels_per_isa,
            'NON_PROD',
            'MICROKERNEL_SRCS',
        )
        asm_vars = write_grouped_microkernels_cmake(
            arch_microkernels_cmake,
            key,
            prod_asm_microkernels_per_arch,
            'PROD',
            'ASM_MICROKERNEL_SRCS',
        )
        asm_vars += write_grouped_microkernels_cmake(
            arch_microkernels_cmake,
            key,
            non_prod_asm_microkernels_per_arch,
            'NON_PROD',
            'ASM_MICROKERNEL_SRCS',
        )
        if c_vars:
          c_vars = [f'${{{v}}}' for v in c_vars]
          arch_microkernels_cmake.write(
              f'\nSET({make_variable_name("ALL", key, "MICROKERNEL_SRCS")} {" + ".join(c_vars)})\n'
          )
        if asm_vars:
          asm_vars = [f'${{{v}}}' for v in asm_vars]
          arch_microkernels_cmake.write(
              f'\nSET({make_variable_name("ALL", key, "ASM_MICROKERNEL_SRCS")} {" + ".join(asm_vars)})\n'
          )
        microkernels_cmake.write(
            f'INCLUDE(cmake/gen/{arch_microkernels_cmake_filename})\n'
        )
        overwrite_if_changed(
            os.path.join(cmake_gen_dir, arch_microkernels_cmake_filename),
            arch_microkernels_cmake,
        )

    overwrite_if_changed(
        os.path.join(cmake_gen_dir, 'microkernels.cmake'), microkernels_cmake
    )


if __name__ == '__main__':
  main(sys.argv[1:])
