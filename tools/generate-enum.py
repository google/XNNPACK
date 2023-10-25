#!/usr/bin/env python
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import codecs
import math
import os
import re
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xngen
import xnncommon

parser = argparse.ArgumentParser(
    description='Generates enum header and convertion-to-string code.')
parser.add_argument(
    '-s',
    '--spec',
    metavar='FILE',
    required=True,
    help='Specification (YAML) file')
parser.add_argument(
    '--output_src',
    metavar='FILE',
    required=True,
    help='Output C source file')
parser.add_argument(
    '--output_hdr',
    metavar='FILE',
    required=True,
    help='Output C/C++ header file')
parser.add_argument(
    '-e',
    '--enum',
    metavar='NAME',
    required=True,
    help='Name of the enum variable')
parser.add_argument(
    '-d',
    '--debug',
    action='store_true',
    default=False,
    help='Define enum-to-string fuction only when debug logging is enabled')
parser.set_defaults(defines=list())


def generate_source(enum_name, spec_path, output_path, header_path, debug_only):
  with codecs.open(spec_path, 'r', encoding='utf-8') as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError('expected a list of enumeration values in the spec')

    output = f"""\
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {spec_path}
//   Generator: {sys.argv[0]}


#include <assert.h>
#include <stdint.h>

#include <{header_path}>\n\n\n"""

    max_offset = sum(len(entry['string']) + 1 for entry in spec_yaml[:-1])
    if max_offset < 256:
      offset_type = 'uint8_t'
    elif max_offset < 65536:
      offset_type = 'uint16_t'
    else:
      offset_type = 'uint32_t'

    offset_declaration = f'static const {offset_type} offset[{len(spec_yaml)}] = {{\n ';
    string_declaration = 'static const char data[] =\n'
    pos = 0
    for i, spec_entry in enumerate(spec_yaml):
      enum_item_name = spec_entry['name']
      assert enum_item_name.startswith(enum_name + "_")
      enum_item_string = spec_entry['string']

      if i + 1 != len(spec_yaml):
        string_declaration += '  "' + enum_item_string + '\\0"\n'
        offset_declaration += ' ' + str(pos) + ','
      else:
        string_declaration += '  "' + enum_item_string + '";\n'
        offset_declaration += ' ' + str(pos) + '\n};'

      # Wrap offset array on 120 columns
      last_offset_line = offset_declaration[offset_declaration.rfind('\n')+1:]
      if len(last_offset_line) > 120:
        last_offset_start = offset_declaration.rfind(',', 0, -1) + 1
        offset_declaration = offset_declaration[:last_offset_start] + '\n ' + offset_declaration[last_offset_start:]

      pos += len(enum_item_string) + 1

    if debug_only:
      output += '#if XNN_LOG_LEVEL > 0\n'
    output += offset_declaration
    output += '\n\n'
    output += string_declaration

    arg_name = enum_name[len("xnn_"):]
    output += f"""
const char* {enum_name}_to_string(enum {enum_name} {arg_name}) {{
  assert({arg_name} >= {spec_yaml[0]['name']});
  assert({arg_name} <= {spec_yaml[-1]['name']});
  return &data[offset[{arg_name}]];
}}\n"""
    if debug_only:
      output += '#endif  // XNN_LOG_LEVEL > 0\n'

    xnncommon.overwrite_if_changed(output_path, output)

def generate_header(enum_name, spec_path, output_path, debug_only):
  with codecs.open(spec_path, 'r', encoding='utf-8') as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError('expected a list of enumeration values in the spec')

    output = f"""\
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {spec_path}
//   Generator: {sys.argv[0]}

#pragma once

#include <xnnpack/common.h>


#ifdef __cplusplus
extern "C" {{
#endif

enum {enum_name} {{\n"""

    enum_item_name = spec_yaml[0]['name']
    assert enum_item_name.startswith(enum_name + "_")
    output += '  ' + enum_item_name + ' = 0,\n'
    for spec_entry in spec_yaml[1:]:
      enum_item_name = spec_entry['name']
      assert enum_item_name.startswith(enum_name + "_")
      output += '  ' + enum_item_name + ',\n'

    arg_name = enum_name[len("xnn_"):]
    output += '};\n\n'

    if debug_only:
      output += f"""\
#if XNN_LOG_LEVEL <= 0
  XNN_INLINE static const char* {enum_name}_to_string(enum {enum_name} type) {{
    return "<unknown>";
  }}
#else
  XNN_INTERNAL const char* {enum_name}_to_string(enum {enum_name} type);
#endif
"""
    else:
      output += f"""\
XNN_INTERNAL const char* {enum_name}_to_string(enum {enum_name} {arg_name});
"""
    output += """
#ifdef __cplusplus
}  // extern "C"
#endif
"""

    xnncommon.overwrite_if_changed(output_path, output)

def main(args):
  options = parser.parse_args(args)
  generate_header(options.enum, options.spec, options.output_hdr, options.debug)

  assert options.enum.startswith('xnn_')
  header_path = 'xnnpack/' + options.enum[len('xnn_'):].replace('_', '-') + '.h'
  generate_source(options.enum, options.spec, options.output_src, header_path,
                  options.debug)

if __name__ == '__main__':
  main(sys.argv[1:])
