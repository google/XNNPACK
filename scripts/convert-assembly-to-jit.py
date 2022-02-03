#!/usr/bin/env python3
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Converts hand written assembly (.S files) to C++ files using the JIT.

Takes a single argument, an assembly file, and prints converted output to stdout.
"""

import argparse
import datetime
import re
import sys

SPACES = r'\s*'
COMMA = r',' + SPACES
COMMENTS = SPACES + '((//\s+.+)|)$'
WB = r'!'

REG_NO_GROUP = r'r\d+|s\d+|d\d+|q\d+|sp|lr|pc|x\d+|(?:v\d+\.(?:\d+)?(?:d|s|h|b))'
REG = r'(' + REG_NO_GROUP + ')'
IMM_NO_GROUP = r'\d+'
IMM = r'(' + IMM_NO_GROUP + ')'
REG_LANE_NO_GROUP = r'(?:' + REG_NO_GROUP + r')\[' + IMM_NO_GROUP + r'\]'
REG_OR_IMM = r'(' + REG_LANE_NO_GROUP + '|' + REG_NO_GROUP + '|' + IMM_NO_GROUP + ')'

REGLIST_CONSEC = r'\{(\w+)-(\w+)\}' + SPACES
REGLIST_INDIV = r'\{([\w.]+(?:,\s+[\w.]+)*)\}' + SPACES
REGLIST_INDIV_REPLICATE = r'\{(\w+(?:\[\])(,\s*\w+(?:\[\]))*)\}' + SPACES
REGLIST_INDEX = r'\{(' + REG_LANE_NO_GROUP + ')\}' + SPACES

APSR = 'APSR_nzcv'
FPSCR = '(FPSCR)'

MEMOP = r'\[' + SPACES + REG + '\]' + SPACES
MEMOP_MAYBE_WB = r'\[' + SPACES + REG + '\]' + f'({WB})?'
MEMOP_OFFSET = r'\[' + REG + COMMA + '(-?\d+)\]' + SPACES
MEMOP_OFFSET_MAYBE_WB = r'\[' + REG + COMMA + '(-?\d+)\]' + f'({WB})?' + SPACES

B_IMM = r'(\d+)(f|b)'

INSTR = SPACES + r'([A-Z0-9.]+)' + SPACES

# e.g. #ifndef __APPLE__
IFDEF_RE = re.compile(r'\s*#(ifndef|endif|ifdef).*')
# e.g. # Push 96 bytes
COMMENT_RE = re.compile(SPACES + r'((//|#)\s*.+)')
# e.g. 0:
LABEL = re.compile(r'(\w+):')
# e.g. NOP
INSTR_RE = re.compile(INSTR + COMMENTS)
# e.g. VPUSH {d8-d15}
INSTR_REGLIST_CONSEC_RE = re.compile(INSTR + REGLIST_CONSEC + COMMENTS)
# e.g. PUSH {r4, r5}
INSTR_REGLIST_LIST_RE = re.compile(INSTR + REGLIST_INDIV + COMMENTS)
# e.g. BX lr
INSTR_OP_RE = re.compile(INSTR + REG + COMMENTS)
# e.g. BLO 2f
INSTR_B_IMM = re.compile(INSTR + B_IMM + COMMENTS)
# e.g. TBNZ x0, 4, 5f
INSTR_B_REG_IMM_IMM = re.compile(INSTR + REG + COMMA + IMM + COMMA + B_IMM + COMMENTS)
# e.g. .p2align 3
P2ALIGN_RE = re.compile(SPACES + r'\.p2align\s+(\d+)')
# e.g. CMP r0, 2
INSTR_REG_IMM_RE = re.compile(INSTR + REG + COMMA + IMM + COMMENTS)
# e.g. LDR r0, [r12]
INSTR_REG_MEMOP_RE = re.compile(INSTR + REG + COMMA + MEMOP + COMMENTS)
# e.g. LDR q0, [x4], 16
INSTR_REG_MEMOP_IMM_RE = re.compile(INSTR + REG + COMMA + MEMOP + COMMA + IMM + COMMENTS)
# e.g. LDR r0, [sp, 112]
INSTR_REG_MEMOP_OFFSET_RE = re.compile(INSTR + REG + COMMA + MEMOP_OFFSET +
                                       COMMENTS)
# e.g. LDRD r6, r7, [sp]
INSTR_REG_REG_MEMOP_RE = re.compile(INSTR + REG + COMMA + REG + COMMA +
                                           MEMOP + COMMENTS)
# e.g. LDRD r6, r7, [sp, 104], STP d8, d9, [sp, -64]!
INSTR_REG_REG_MEMOP_OFFSET_RE = re.compile(INSTR + REG + COMMA + REG + COMMA +
                                           MEMOP_OFFSET_MAYBE_WB + COMMENTS)
# e.g. LDP q20, q21, [x5], 32
INSTR_REG_REG_MEMOP_IMM_RE = re.compile(INSTR + REG + COMMA + REG + COMMA +
                                           MEMOP + COMMA + IMM + COMMENTS)
# e.g. PLD [r4, 64]
INSTR_MEMOP_OFFSET_RE = re.compile(INSTR + MEMOP_OFFSET + COMMENTS)
# e.g. movlo r12, r3, vdup.32 q0, d14[0]
INSTR_REG_REG_RE = re.compile(INSTR + REG + COMMA + REG_OR_IMM + COMMENTS)
# e.g. SUBS r5, r2, 16 or SUBS r5, r2, r10 or VMLFA.F32 q8, q4, d0[0]
INSTR_REG_REG_REG_RE = re.compile(INSTR + REG + COMMA + REG + COMMA +
                                  REG_OR_IMM + COMMENTS)
# e.g. VEXT.8  q0, q0, q0, 4
INSTR_REG_REG_REG_IMM_RE = re.compile(INSTR + REG + COMMA + REG + COMMA + REG +
                                      COMMA + IMM + COMMENTS)
# e.g. VST1.32 {d16}, [r11], r0
INSTR_REGLIST_INDIV_MEMOP_REG = re.compile(INSTR + REGLIST_INDIV + COMMA +
                                           MEMOP + COMMA + REG + COMMENTS)
# e.g. VST1.32 {d16-d19}, [r11], r0
INSTR_REGLIST_CONSEC_MEMOP_REG = re.compile(INSTR + REGLIST_CONSEC + COMMA +
                                            MEMOP + COMMA + REG + COMMENTS)
# e.g. VLDM r9, {d16-d19}
INSTR_REG_REGLIST_CONSECT = re.compile(INSTR + REG + COMMA + REGLIST_CONSEC +
                                       COMMENTS)
# e.g. VLDM r9!, {d16-d19}
INSTR_REG_REGLIST_CONSECT_WB = re.compile(INSTR + REG + WB + COMMA +
                                          REGLIST_CONSEC + COMMENTS)
# e.g. VLDM r9!, {d16}
INSTR_REG_REGLIST_INDIV_WB = re.compile(INSTR + REG + WB + COMMA +
                                        REGLIST_INDIV + COMMENTS)
# e.g. VLD1.32 {d0}, [r3]{!}
INSTR_REGLIST_INDIV_MEMOP = re.compile(INSTR + REGLIST_INDIV + COMMA +
                                       MEMOP_MAYBE_WB + COMMENTS)
# e.g. LD1 {v16.16b, v17.16b, v18.16b}, [x5], 48
INSTR_REGLIST_INDIV_MEMOP_IMM = re.compile(INSTR + REGLIST_INDIV + COMMA +
                                       MEMOP + COMMA + IMM + COMMENTS)
# e.g. VST1.32 {d24-d25}, [r11]{!}
INSTR_REGLIST_CONSEC_MEMOP = re.compile(INSTR + REGLIST_CONSEC + COMMA +
                                        MEMOP_MAYBE_WB + COMMENTS)
# e.g. VLD1.32 {d0[]}, [r3]!
INSTR_REGLIST_REPLICATE_MEMOP = re.compile(INSTR + REGLIST_INDIV_REPLICATE +
                                           COMMA + MEMOP + r'(!)?' + COMMENTS)
# e.g. VST1.32 {d16[0]}, [r11]{!}
INSTR_REGLIST_INDEX_MEMOP = re.compile(INSTR + REGLIST_INDEX + COMMA +
                                       MEMOP_MAYBE_WB + COMMENTS)
# e.g. VMRS APSR_nzcv, FPSCR
INSTR_REG_FPSCR = re.compile(INSTR + f'({APSR}|{REG_NO_GROUP})' + COMMA +
                             FPSCR + COMMENTS)

# e.g. PRFM PLDL1KEEP, [x5]
INSTR_PLD_MEMOP = re.compile(INSTR + f'(PLDL1KEEP)' + COMMA + MEMOP + COMMENTS)
# e.g. PRFM PLDL1KEEP, [x5, 64]
INSTR_PLD_MEMOP_OFFSET = re.compile(INSTR + f'(PLDL1KEEP)' + COMMA + MEMOP_OFFSET + COMMENTS)

COND = r'([A-Z]+)'
# e.g. CSEL x9, x3, x9, LO
INSTR_REG_REG_REG_COND_RE = re.compile(INSTR + REG + COMMA + REG + COMMA + REG + COMMA + COND + COMMENTS)


def remove_brackets(s):
  return s.replace('[', '').replace(']', '')


def fix_replicate_instruction(s):
  return re.sub(r'_(\d+)', r'r_\1', s, 1)


def fix_instr_name(s):
  return s.lower().replace('.', '_', 2).replace('and', 'and_', 1)


def fix_comments(s):
  return s.replace('#', '//', 1)


def maybe_wb(wb):
  return '++' if wb else ''


def fix_fn_name(name):
  if name.startswith('xnn_'):
    name = name[len('xnn_'):]
  # remove any type of activations from name
  if 'minmax' in name:
    name = name.replace('minmax_', '')
  return f'xnn_generate_{name}'


def fix_regs(regs):
  # Vector registers with datatype need to be method calls.
  # e.g. v2.4s -> v2.v4s(), v2.s -> v2.s()
  def repl(m):
    if m.group(2):
      return f'{m[1]}v{m[2]}{m[3]}()'
    else:
      return f'{m[1]}{m[3]}()'
  return re.sub(r'(\w+\.)(\d+)?(\w+)', repl, regs)


IGNORE_LINES = [r'\s*\.\w+']

AARCH32 = 'aarch32'
AARCH64 = 'aarch64'
GEMM = 'GEMM'
IGEMM = 'IGEMM'

def main(input_file):
  arch = None
  kernel_type = GEMM

  if 'aarch32' in input_file:
    arch = AARCH32
  elif 'aarch64' in input_file:
    arch = AARCH64
  else:
    print('ERROR: unknown architecture')
    sys.exit(1)

  if 'igemm' in input_file:
    kernel_type = IGEMM

  # Whether we are in the copyright section.
  in_copyright = False
  # Whether we are in the microkernel function.
  in_function = False
  # Instructions that make up the microkernel.
  instructions = []
  # Lines of code or comments before the actual function body.
  prologue = []
  # All labels need to be declared first, collect them and output them after
  # function signature.
  labels = []
  # Name of the microkernel function.
  fn_name = ''
  sc = ';'
  # Whether we are in the auto-generated comment.
  in_autogen = False

  with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
      line = line.rstrip()

      # Handle all lines before the microkernel instructions begin.
      if not in_function:
        if 'Auto-generated file' in line:
          in_autogen = True
          continue
        elif 'BEGIN_FUNCTION' in line:
          in_function = True
          fn_name = line.split()[1]
          prologue.append(f'// Converted from: {input_file[20:]}')
          if kernel_type == GEMM:
            prologue.append('void Generator::generate(size_t nc, size_t kc, void* params) {')
          else:
            prologue.append('void Generator::generate(size_t nc, size_t kc, size_t ks, void* params) {')
          continue
        elif 'Copyright ' in line:
          in_autogen = False
          # replace year
          prologue.append(
              re.sub('\d{4}', str(datetime.date.today().year), line,
                     1).rstrip())
          continue
        elif '#include <xnnpack/assembly.h>' in line:
          prologue.append(f'#include <cstddef>')
          prologue.append('')
          prologue.append(f'#include <xnnpack/{arch}-assembler.h>')
          prologue.append('#include <xnnpack/allocator.h>')
          if kernel_type == GEMM:
            prologue.append('#include <xnnpack/gemm.h>')
          else:
            prologue.append('#include <xnnpack/igemm.h>')
          prologue.append('')
          prologue.append('namespace xnnpack {')
          prologue.append(f'namespace {arch} {{')
          prologue.append('namespace {')
          prologue.append('class Generator : public Assembler {')
          prologue.append('  using Assembler::Assembler;')
          prologue.append(' public:')
          if kernel_type == GEMM:
            prologue.append('  void generate(size_t nc, size_t kc, void* params);')
          else:
            prologue.append('  void generate(size_t nc, size_t kc, size_t ks, void* params);')
          prologue.append('};')
          continue
        elif any(re.fullmatch(p, line) for p in IGNORE_LINES):
          continue
        elif in_autogen:
          continue
        else:
          prologue.append(fix_comments(line.rstrip()))
          continue

      # We are now in the microkernel function body.
      # Don't keep the ifdefs.
      m = re.fullmatch(IFDEF_RE, line)
      if m:
        continue
      # But keep other comments.
      m = re.fullmatch(COMMENT_RE, line)
      if m:
        instructions.append(m[1])
        continue

      m = re.fullmatch(LABEL, line)
      if m:
        labels.append(m[1])
        instructions.append(f'bind(l{m[1]}){sc}')
        continue
      m = re.fullmatch(INSTR_RE, line)
      if m:
        instructions.append(f'{fix_instr_name(m[1])}(){sc} {m[2]}')
        continue
      m = re.fullmatch(INSTR_OP_RE, line)
      if m:
        instructions.append(f'{fix_instr_name(m[1])}({m[2]}){sc} {m[3]}')
        continue
      m = re.fullmatch(INSTR_REGLIST_CONSEC_MEMOP_REG, line)
      if m:
        instructions.append(
            f'{fix_instr_name(m[1])}({{{m[2]}-{m[3]}}}, mem[{m[4]}], {m[5]}){sc} {m[6]}'
        )
        continue
      m = re.fullmatch(INSTR_REGLIST_INDIV_MEMOP_REG, line)
      if m:
        instructions.append(
            f'{fix_instr_name(m[1])}({{{fix_regs(m[2])}}}, mem[{m[3]}], {m[4]}){sc} {m[5]}')
        continue
      m = re.fullmatch(INSTR_REGLIST_CONSEC_RE, line)
      if m:
        instructions.append(f'{fix_instr_name(m[1])}({{{m[2]}-{m[3]}}}){sc} {m[4]}')
        continue
      m = re.fullmatch(INSTR_REGLIST_LIST_RE, line)
      if m:
        instructions.append(f'{fix_instr_name(m[1])}({{{m[2]}}}){sc} {m[3]}')
        continue
      m = re.fullmatch(INSTR_MEMOP_OFFSET_RE, line)
      if m:
        instructions.append(f'{fix_instr_name(m[1])}(mem[{m[2]}, {m[3]}]){sc} {m[4]}')
        continue
      m = re.fullmatch(INSTR_REG_MEMOP_RE, line)
      if m:
        instructions.append(f'{fix_instr_name(m[1])}({m[2]}, mem[{m[3]}]){sc} {m[4]}')
        continue
      m = re.fullmatch(INSTR_REG_MEMOP_IMM_RE , line)
      if m:
        instructions.append(f'{fix_instr_name(m[1])}({m[2]}, mem[{m[3]}], {m[4]}){sc} {m[5]}')
        continue
      m = re.fullmatch(INSTR_REG_MEMOP_OFFSET_RE, line)
      if m:
        instructions.append(
            f'{fix_instr_name(m[1])}({m[2]}, mem[{m[3]}, {m[4]}]){sc} {m[5]}')
        continue
      m = re.fullmatch(INSTR_REG_REG_MEMOP_RE, line)
      if m:
        instructions.append(
            f'{fix_instr_name(m[1])}({m[2]}, {m[3]}, mem[{m[4]}]){sc} {m[5]}')
        continue
      m = re.fullmatch(INSTR_REG_REG_MEMOP_OFFSET_RE, line)
      if m:
        if m[6]: # wb
          instructions.append(
              f'{fix_instr_name(m[1])}({m[2]}, {m[3]}, mem[{m[4]}, {m[5]}]++){sc} {m[7]}')
        else: #no wb
          instructions.append(
              f'{fix_instr_name(m[1])}({m[2]}, {m[3]}, mem[{m[4]}, {m[5]}]){sc} {m[7]}')
        continue
      m = re.fullmatch(INSTR_REG_REG_MEMOP_IMM_RE , line)
      if m:
        instructions.append(
            f'{fix_instr_name(m[1])}({m[2]}, {m[3]}, mem[{m[4]}], {m[5]}){sc} {m[6]}')
        continue
      m = re.fullmatch(INSTR_REG_IMM_RE, line)
      if m:
        instructions.append(f'{fix_instr_name(m[1])}({fix_regs(m[2])}, {m[3]}){sc} {m[4]}')
        continue
      m = re.fullmatch(INSTR_REG_REG_REG_RE, line)
      if m:
        instructions.append(
            f'{fix_instr_name(m[1])}({fix_regs(m[2])}, {fix_regs(m[3])}, {fix_regs(m[4])}){sc} {m[5]}')
        continue
      m = re.fullmatch(INSTR_REG_REG_REG_IMM_RE, line)
      if m:
        instructions.append(
            f'{fix_instr_name(m[1])}({m[2]}, {m[3]}, {m[4]}, {m[5]}){sc} {m[6]}')
        continue
      m = re.fullmatch(INSTR_REG_REG_RE, line)
      if m:
        instructions.append(f'{fix_instr_name(m[1])}({fix_regs(m[2])}, {fix_regs(m[3])}){sc} {m[4]}')
        continue
      m = re.fullmatch(INSTR_REG_REGLIST_CONSECT, line)
      if m:
        instructions.append(
            f'{fix_instr_name(m[1])}(mem[{m[2]}], {{{m[3]}-{m[4]}}}){sc} {m[5]}')
        continue
      m = re.fullmatch(INSTR_REG_REGLIST_CONSECT_WB, line)
      if m:
        instructions.append(
            f'{fix_instr_name(m[1])}(mem[{m[2]}]++, {{{m[3]}-{m[4]}}}){sc} {m[5]}')
        continue
      m = re.fullmatch(INSTR_REG_REGLIST_INDIV_WB, line)
      if m:
        instructions.append(
            f'{fix_instr_name(m[1])}(mem[{m[2]}]++, {{{m[3]}}}){sc} {m[4]}')
        continue
      m = re.fullmatch(INSTR_B_IMM, line)
      if m:
        instructions.append(f'{fix_instr_name(m[1])}(l{m[2]}){sc} {m[4]}')
        continue
      m = re.fullmatch(INSTR_B_REG_IMM_IMM , line)
      if m:
        instructions.append(f'{fix_instr_name(m[1])}({m[2]}, {m[3]}, l{m[4]}){sc} {m[6]}')
        continue
      m = re.fullmatch(INSTR_REGLIST_INDIV_MEMOP, line)
      if m:
        instructions.append(
            f'{fix_instr_name(m[1])}({{{fix_regs(m[2])}}}, mem[{m[3]}]{maybe_wb(m[4])}){sc} {m[5]}'
        )
        continue
      m = re.fullmatch(INSTR_REGLIST_INDIV_MEMOP_IMM, line)
      if m:
        instructions.append(
            f'{fix_instr_name(m[1])}({{{fix_regs(m[2])}}}, mem[{m[3]}], {m[4]}){sc} {m[5]}'
        )
        continue
      m = re.fullmatch(INSTR_REGLIST_CONSEC_MEMOP, line)
      if m:
        instructions.append(
            f'{fix_instr_name(m[1])}({{{m[2]}-{m[3]}}}, mem[{m[4]}]{maybe_wb(m[5])}){sc} {m[6]}'
        )
        continue
      m = re.fullmatch(INSTR_REGLIST_REPLICATE_MEMOP, line)
      if m:
        if m[5]:
          instructions.append(
              f'{fix_replicate_instruction(fix_instr_name(m[1]))}({{{remove_brackets(m[2])}}}, mem[{m[4]}]++){sc} {m[6]}'
          )
        else:
          instructions.append(
              f'{fix_replicate_instruction(fix_instr_name(m[1]))}({{{remove_brackets(m[2])}}}, mem[{m[4]}]){sc} {m[6]}'
          )
        continue
      m = re.fullmatch(INSTR_REGLIST_INDEX_MEMOP, line)
      if m:
        instructions.append(
            f'{fix_instr_name(m[1])}({{{m[2]}}}, mem[{m[3]}]{maybe_wb(m[4])}){sc} {m[5]}'
        )
        continue
      m = re.fullmatch(P2ALIGN_RE, line)
      if m:
        instructions.append(f'align({1 << int(m[1])}){sc}')
        continue
      m = re.fullmatch(INSTR_REG_FPSCR, line)
      if m:
        instructions.append(f'{fix_instr_name(m[1])}({m[2]}, {m[3]}){sc} {m[4]}')
        continue
      m = re.fullmatch(INSTR_PLD_MEMOP, line)
      if m:
        instructions.append(f'{fix_instr_name(m[1])}(k{m[2]}, mem[{m[3]}]){sc} {m[4]}')
        continue
      m = re.fullmatch(INSTR_PLD_MEMOP_OFFSET, line)
      if m:
        instructions.append(f'{fix_instr_name(m[1])}(k{m[2]}, mem[{m[3]}, {m[4]}]){sc} {m[5]}')
        continue
      m = re.fullmatch(INSTR_REG_REG_REG_COND_RE, line)
      if m:
        instructions.append(f'{fix_instr_name(m[1])}({m[2]}, {m[3]}, {m[4]}, k{m[5]}){sc} {m[6]}')
        continue

      # Keep empty lines for formatting
      if line.strip() == '':
        instructions.append('')
        continue

      # Assembly directives that we don't are about.
      if line.strip().startswith('.'):
        continue

      if line.startswith('END_FUNCTION'):
        continue

      # All other lines are error.
      print(f'ERROR: {line}', file=sys.stderr)
      sys.exit(1)

  # Actually emit the JIT codegen (to stdout).
  for p in prologue:
    print(p)

  labels_str = ', '.join(f'l{l}' for l in labels)
  print(f'  Label {labels_str};')
  print()

  indent = '  '
  for i in instructions:
    if i.strip().startswith('#'):
      print(indent + fix_comments(i))
    elif i.strip().startswith('//'):
      print(indent + i)
    elif i.strip() == '':
      print()
    else:
      print(indent + (i).rstrip())

  print('}')
  print('}  // namespace')
  print(f'}}  // {arch}')
  print('}  // xnnpack')
  print('')
  if kernel_type == GEMM:
    print(f'xnn_status {fix_fn_name(fn_name)}(xnn_code_buffer* code, size_t nc, size_t kc, const void* params) {{')
  else:
    print(f'xnn_status {fix_fn_name(fn_name)}(xnn_code_buffer* code, size_t nc, size_t kc, size_t ks, const void* params) {{')
  print(f'  using namespace xnnpack::{arch};')
  print('  Generator g(code);')
  if kernel_type == GEMM:
    print('  g.generate(nc, kc, nullptr);')
  else:
    print('  g.generate(nc, kc, ks, nullptr);')
  print('  g.finalize();')
  print('  if (g.error() != xnnpack::Error::kNoError) {')
  print('    return xnn_status_invalid_state;')
  print('  }')
  print('  return xnn_status_success;')
  print('}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Convert assembly to to JIT C++, writes to stdout.')
  parser.add_argument('input_file', help='Input assembly filename')
  args = parser.parse_args()
  main(args.input_file)
