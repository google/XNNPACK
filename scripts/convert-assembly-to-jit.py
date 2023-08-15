#!/usr/bin/env python3
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Converts hand written assembly (.S files) to C++ files using the JIT.

Takes a single argument, an assembly file, and prints converted output to
stdout.
"""

import argparse
import codecs
from collections import defaultdict
import datetime
from enum import Enum
import os
import re
import sys
from typing import List, Tuple, Mapping


class PrfmMode(Enum):
  NoPrfm = 1
  PrfmInFileName = 2
  ForcePrfm = 3

SPACES = r'\s*'
COMMA = r',' + SPACES
COMMENTS = SPACES + '((//\s+.+)|)$'
WB = r'!'

REG_NO_GROUP = r'r\d+|h\d+|s\d+|d\d+|q\d+|sp|lr|pc|w\d+|x\d+|(?:v\d+\.(?:\d+)?(?:d|s|h|b))'
REG = r'(' + REG_NO_GROUP + ')'
IMM_NO_GROUP = r'\d+'
IMM = r'(' + IMM_NO_GROUP + ')'
REG_LANE_NO_GROUP = r'(?:' + REG_NO_GROUP + r')\[' + IMM_NO_GROUP + r'\]'
REG_OR_IMM = r'(' + REG_LANE_NO_GROUP + '|' + REG_NO_GROUP + '|' + IMM_NO_GROUP + ')'
REG_INDEXED = r'(?:' + REG_NO_GROUP + r')\[' + IMM_NO_GROUP + r'\]'

REGLIST_CONSEC = r'\{(\w+)-(\w+)\}' + SPACES
REGLIST_INDIV = r'\{([\w.]+(?:,\s+[\w.]+)*)\}' + SPACES
REGLIST_INDIV_REPLICATE = r'\{(\w+(?:\[\])(,\s*\w+(?:\[\]))*)\}' + SPACES
REGLIST_INDEX = r'\{(' + REG_LANE_NO_GROUP + ')\}' + SPACES
# e.g. {v0.d}[1]
REGLIST_LANE_INDEX = r'\{' + REG + '\}' + r'\[(\d+)\]' + SPACES
# e.g. v0.d[1]
REG_INDEX = REG + r'\[(\d+)\]' + SPACES

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
INSTR_B_REG_IMM_IMM = re.compile(INSTR + REG + COMMA + IMM + COMMA + B_IMM +
                                 COMMENTS)
# e.g. .p2align 3
P2ALIGN_RE = re.compile(SPACES + r'\.p2align\s+(\d+)')
# e.g. CMP r0, 2
INSTR_REG_IMM_RE = re.compile(INSTR + REG + COMMA + IMM + COMMENTS)
# e.g. LDR r0, [r12]
INSTR_REG_MEMOP_RE = re.compile(INSTR + REG + COMMA + MEMOP + COMMENTS)
# e.g. LDR q0, [x4], 16
INSTR_REG_MEMOP_IMM_RE = re.compile(INSTR + REG + COMMA + MEMOP + COMMA + IMM +
                                    COMMENTS)
# e.g. LDR r0, [sp, 112], STR x20, [sp, -80]!
INSTR_REG_MEMOP_OFFSET_RE = re.compile(INSTR + REG + COMMA +
                                       MEMOP_OFFSET_MAYBE_WB + COMMENTS)
# e.g. LDRD r6, r7, [sp]
INSTR_REG_REG_MEMOP_RE = re.compile(INSTR + REG + COMMA + REG + COMMA + MEMOP +
                                    COMMENTS)
# e.g. LDRD r6, r7, [sp, 104], STP d8, d9, [sp, -64]!
INSTR_REG_REG_MEMOP_OFFSET_RE = re.compile(INSTR + REG + COMMA + REG + COMMA +
                                           MEMOP_OFFSET_MAYBE_WB + COMMENTS)
# e.g. LDP q20, q21, [x5], 32
INSTR_REG_REG_MEMOP_IMM_RE = re.compile(INSTR + REG + COMMA + REG + COMMA +
                                        MEMOP + COMMA + IMM + COMMENTS)
# e.g. PLD [r9]
INSTR_MEMOP_RE = re.compile(INSTR + MEMOP + COMMENTS)
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
# e.g. LD1 {v0.d}[0], [x5], 48
INSTR_REGLIST_INDEX_MEMOP_IMM = re.compile(INSTR + REGLIST_LANE_INDEX + COMMA +
                                           MEMOP + COMMA + IMM + COMMENTS)
# e.g. INS v19.d[0], x4
INSTR_REG_INDEX_REG = re.compile(INSTR + REG_INDEX + COMMA + REG + COMMENTS)
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
INSTR_PLD_MEMOP = re.compile(INSTR + f'(PLDL1KEEP|PSTL1KEEP)' + COMMA + MEMOP + COMMENTS)
# e.g. PRFM PLDL1KEEP, [x5, 64]
INSTR_PLD_MEMOP_OFFSET = re.compile(INSTR + f'(PLDL1KEEP)' + COMMA +
                                    MEMOP_OFFSET + COMMENTS)

COND = r'([A-Z]+)'
# e.g. CSEL x9, x3, x9, LO
INSTR_REG_REG_REG_COND_RE = re.compile(INSTR + REG + COMMA + REG + COMMA + REG +
                                       COMMA + COND + COMMENTS)


def remove_brackets(s: str) -> str:
  return s.replace('[', '').replace(']', '')


def fix_replicate_instruction(s: str) -> str:
  return re.sub(r'_(\d+)', r'r_\1', s, 1)


def fix_instr_name(s: str) -> str:
  fixed = s.lower().replace('.', '_', 2)
  if fixed == 'and':
    return 'and_'
  return fixed


def fix_comments(s: str) -> str:
  return s.replace('#', '//', 1)


def maybe_wb(wb: bool) -> str:
  return '++' if wb else ''


def fix_fn_name(name: str) -> str:
  if name.startswith('xnn_'):
    name = name[len('xnn_'):]
  # replace "__asm_<arch>" with "__<arch>"
  name = name.replace("__asm_", "__")
  # remove any type of activations from name
  if 'minmax' in name:
    name = name.replace('minmax_', '')
  return f'xnn_generate_{name}'


def remove_prfm_from_fn_name(name: str) -> str:
  assert ('_prfm' in name)
  return name.replace('_prfm', '')


def fix_regs(regs: str) -> str:
  # Vector registers with datatype need to be method calls.
  # e.g. v2.4s -> v2.v4s(), v2.s -> v2.s()
  def repl(m):
    if m.group(2):
      return f'{m[1]}v{m[2]}{m[3]}()'
    else:
      return f'{m[1]}{m[3]}()'

  return re.sub(r'(\w+\.)(\d+)?(\w+)', repl, regs)


def get_callee_saved() -> List[str]:
  return [
      'd8', 'd9', 'd10', 'd11', 'd12', 'd13', 'd14', 'd15', 'x19', 'x20', 'x21',
      'x22'
  ]


IGNORE_LINES = [r'\s*\.\w+']

AARCH32 = 'aarch32'
AARCH64 = 'aarch64'
GEMM = 'GEMM'
IGEMM = 'IGEMM'

# Hard-coded post operations.
AARCH32_POST_OP = """void Generator::perform_post_operations(
  size_t max_mr,
  size_t num_post_operations,
  const xnn_post_operation* post_operations)
{
  if (num_post_operations == 0) {
    return;
  }
  for (size_t i = 0; i < num_post_operations; i++) {
    switch (post_operations[i].op_type) {
      case xnn_post_operation_type_hardswish: {
        const auto sixth = q0;
        const auto three = q1;
        const auto six = q2;
        const auto zero = q3;
        vld3r_32({sixth.low(), three.low(), six.low()}, mem[PARAMS_REG_PLACEHOLDER]++);
        vmov(zero, 0);
        vmov(three.high(), three.low());
        vmov(six.high(), six.low());
        const QRegister accs[] = {q8, q9, q10, q11, q12, q13, q14, q15};
        const QRegister tmps[] = {q4, q5, q6, q7};
        f32_hardswish(sixth, three, six, zero, &accs[0], XNN_COUNT_OF(accs), &tmps[0], XNN_COUNT_OF(tmps));
        break;
      }
      default:
        XNN_LOG_UNREACHABLE("unsupported post operation: %u", post_operations[i].op_type);
    }
  }
}"""

AARCH32_POST_OP_RELOAD = """void Generator::perform_post_operations(
  size_t max_mr,
  size_t num_post_operations,
  const xnn_post_operation* post_operations)
{
  if (num_post_operations == 0) {
    return;
  }
  ldr(PARAMS_REG_PLACEHOLDER, mem[sp, PARAMS_OFFSET_PLACEHOLDER]);  // params
  for (size_t i = 0; i < num_post_operations; i++) {
    switch (post_operations[i].op_type) {
      case xnn_post_operation_type_hardswish: {
        const auto sixth = q0;
        const auto three = q1;
        const auto six = q2;
        const auto zero = q3;
        vld3r_32({sixth.low(), three.low(), six.low()}, mem[PARAMS_REG_PLACEHOLDER]++);
        vmov(zero, 0);
        vmov(three.high(), three.low());
        vmov(six.high(), six.low());
        const QRegister accs[] = {ACCS_PLACEHOLDER};
        const QRegister tmps[] = {TMPS_PLACEHOLDER};
        f32_hardswish(sixth, three, six, zero, &accs[0], XNN_COUNT_OF(accs), &tmps[0], XNN_COUNT_OF(tmps));
        break;
      }
      default:
        XNN_LOG_UNREACHABLE("unsupported post operation: %u", post_operations[i].op_type);
    }
  }
}"""

AARCH64_POST_OP = """void Generator::perform_post_operations(
  size_t max_mr,
  size_t num_post_operations,
  const xnn_post_operation* post_operations)
{
  if (num_post_operations == 0) {
    return;
  }
  for (size_t i = 0; i < num_post_operations; i++) {
    switch (post_operations[i].op_type) {
      case xnn_post_operation_type_hardswish: {
        // Reuse A pointers (don't use v8-v15 as they are callee saved).
        const auto sixth = v0.v4s();
        const auto three = v1.v4s();
        const auto six = v2.v4s();
        const auto zero = v3.v4s();
        // v4, v5, v6, v7 available for temporaries.
        ld3r({sixth, three, six}, mem[PARAMS_REG_PLACEHOLDER]++);
        movi(zero, 0);
        const VRegister accs[] = {ACCS_PLACEHOLDER
        };
        const VRegister tmps[] = {TMPS_PLACEHOLDER};
        f32_hardswish(sixth, three, six, zero, &accs[0], XNN_COUNT_OF(accs), &tmps[0], XNN_COUNT_OF(tmps));
        break;
      }
      default:
        XNN_LOG_UNREACHABLE("unsupported post operation: %u", post_operations[i].op_type);
    }
  }
}"""

AARCH32_MR1_POST_OP_ACCS = "q8, q9"
AARCH32_MR4_POST_OP_ACCS = "q8, q9, q10, q11, q12, q13, q14, q15"
AARCH32_MR1_POST_OP_TMPS = "q12, q13"
AARCH32_MR4_POST_OP_TMPS = "q4, q5, q6, q7"
AARCH64_MR1_POST_OP_ACCS = """
          v16.v4s(), v17.v4s(),"""
AARCH64_MR4_POST_OP_ACCS = """
          v24.v4s(), v25.v4s(),
          v26.v4s(), v27.v4s(),
          v28.v4s(), v29.v4s(),
          v30.v4s(), v31.v4s(),"""
AARCH64_MR6_POST_OP_ACCS = """
          v20.v4s(), v21.v4s(), v22.v4s(), v23.v4s(),
          v24.v4s(), v25.v4s(), v26.v4s(), v27.v4s(),
          v28.v4s(), v29.v4s(), v30.v4s(), v31.v4s(),"""
AARCH64_MR1_POST_OP_TMPS = """v4.v4s(), v5.v4s()"""
AARCH64_MR6_POST_OP_TMPS = """v4.v4s(), v5.v4s(), v6.v4s(), v7.v4s()"""


def replace_template(template: str, replacements: Mapping[str, str]):
  result = template
  for k, v in replacements.items():
    result = result.replace(k, v)
  return result


def get_post_op_accs(vector_register_usage):
  usage = [reg for sublist in vector_register_usage for reg in sublist];
  assert(len(usage) == 8)
  return f"""
          {usage[0]}.v4s(), {usage[1]}.v4s(),
          {usage[2]}.v4s(), {usage[3]}.v4s(),
          {usage[4]}.v4s(), {usage[5]}.v4s(),
          {usage[6]}.v4s(), {usage[7]}.v4s(),"""


# We use placeholder strings rather than formatted strings due to the many braces in the template (we are generating C++
# after all), and to avoid copious escaping.
def get_post_operation_implementation(arch, mr: int, params_register: str,
                                      params_offset: str, reload_params: bool,
                                      vector_register_usage):
  if arch == AARCH32:
    if reload_params:
      if mr == 1:
        return replace_template(
            AARCH32_POST_OP_RELOAD, {
                'ACCS_PLACEHOLDER': AARCH32_MR1_POST_OP_ACCS,
                'TMPS_PLACEHOLDER': AARCH32_MR1_POST_OP_TMPS,
                'PARAMS_REG_PLACEHOLDER': params_register,
                'PARAMS_OFFSET_PLACEHOLDER': params_offset,
            })
      else:
        return replace_template(
            AARCH32_POST_OP_RELOAD, {
                'ACCS_PLACEHOLDER': AARCH32_MR4_POST_OP_ACCS,
                'TMPS_PLACEHOLDER': AARCH32_MR4_POST_OP_TMPS,
                'PARAMS_REG_PLACEHOLDER': params_register,
                'PARAMS_OFFSET_PLACEHOLDER': params_offset,
            })
    else:
      return replace_template(AARCH32_POST_OP, {
          'PARAMS_REG_PLACEHOLDER': params_register,
      })
  elif arch == AARCH64:
    # MR 1 microkernels have less accumulators.
    # TODO(zhin): we already parsed this form the vector usage, use that instead of hardcoding the registers.
    if mr == 1:
      return replace_template(
          AARCH64_POST_OP, {
              'ACCS_PLACEHOLDER': AARCH64_MR1_POST_OP_ACCS,
              'TMPS_PLACEHOLDER': AARCH64_MR1_POST_OP_TMPS,
              'PARAMS_REG_PLACEHOLDER': params_register
          })
    elif mr == 4:
      return replace_template(
          AARCH64_POST_OP, {
              'ACCS_PLACEHOLDER': get_post_op_accs(vector_register_usage['C']),
              'TMPS_PLACEHOLDER': AARCH64_MR6_POST_OP_TMPS,
              'PARAMS_REG_PLACEHOLDER': params_register,
              'PARAMS_OFFSET_PLACEHOLDER': params_offset,
          })
    elif mr == 6:
      return replace_template(
          AARCH64_POST_OP, {
              'ACCS_PLACEHOLDER': AARCH64_MR6_POST_OP_ACCS,
              'TMPS_PLACEHOLDER': AARCH64_MR6_POST_OP_TMPS,
              'PARAMS_REG_PLACEHOLDER': params_register
          })
    else:
      print(f'unsupported mr {mr} for post operations', file=sys.stderr)
      sys.exit(1)


def parse_prologue(input_file: str, lines: List[str], arch: str, minmax: bool,
                   kernel_type: str, prfm_mode: PrfmMode, mr: int,
                   post_op: bool) -> Tuple[List[str], Mapping[str, int]]:
  prologue = []
  # Whether we are in the auto-generated comment.
  in_autogen = False
  in_a_pointers = False
  # Whether we are in the comment section that lists C (output) registers.
  in_c_pointers = False
  # Whether we are in the comment section that lists register usage.
  in_register_usage = False
  a_pointers = []
  c_pointers = []
  # Mapping from register type (A, B, or C), to the list of list of registers.
  vector_register_usage = defaultdict(list)
  # Mapping from vector registers to their row index (in MR).
  vector_register_map = {}

  for line in lines:
    if 'Auto-generated file' in line:
      in_autogen = True
      continue
    elif line.startswith('.syntax') or 'LINT.' in line:
      continue
    elif 'BEGIN_FUNCTION' in line:
      prologue.append(f'// Converted from: {input_file[20:]}')
      params = 'const jit_gemm_params* jit_gemm_params'
      prefetch = 'bool prefetch, ' if prfm_mode == PrfmMode.PrfmInFileName else ''
      if kernel_type == GEMM:
        prologue.append(
            f'void Generator::generate({prefetch}size_t max_mr, size_t nc_mod_nr, size_t kc, {params})'
        )
        prologue.append('{')
      else:
        prologue.append(
            f'void Generator::generate({prefetch}size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, {params})'
        )
        prologue.append('{')
      continue
    elif 'Copyright ' in line:
      in_autogen = False
      # replace year
      prologue.append(line)
      continue
    elif '#include <xnnpack/assembly.h>' in line:
      prologue.append(f'#include <cassert>')
      prologue.append(f'#include <cstddef>')
      if minmax:
        prologue.append(f'#include <limits>')
      prologue.append('')
      prologue.append('#include <xnnpack.h>')
      prologue.append(f'#include <xnnpack/{arch}-assembler.h>')
      if kernel_type == GEMM:
        prologue.append('#include <xnnpack/gemm.h>')
      else:
        prologue.append('#include <xnnpack/igemm.h>')
      if post_op:
        prologue.append('#include <xnnpack/log.h>')
      prologue.append('#include <xnnpack/memory.h>')
      prologue.append('#include <xnnpack/microparams.h>')
      prologue.append('#include <xnnpack/post-operation.h>')
      prologue.append('')
      prologue.append('namespace xnnpack {')
      prologue.append(f'namespace {arch} {{')
      prologue.append('namespace {')
      prologue.append('class Generator : public MacroAssembler {')
      prologue.append('  using MacroAssembler::MacroAssembler;')
      prologue.append('')
      prologue.append(' public:')
      params = 'float min, float max' if minmax else 'void* params'
      params = 'const jit_gemm_params* jit_gemm_params'
      prefetch = 'bool prefetch, ' if prfm_mode == PrfmMode.PrfmInFileName else ''
      if kernel_type == GEMM:
        prologue.append(
            f'  void generate({prefetch}size_t max_mr, size_t nc_mod_nr, size_t kc, {params});'
        )
      else:
        prologue.append(
            f'  void generate({prefetch}size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, {params});'
        )

      if post_op:
        prologue.append(
            '  void perform_post_operations(size_t max_mr, size_t num_post_operations, const xnn_post_operation* post_operations);'
        )
      prologue.append('};')
      continue
    elif in_a_pointers:
      prologue.append(fix_comments(line.rstrip()))
      if not line.strip():
        in_a_pointers = False
        continue
      # Handle both:
      # - A0 x1
      # - x1 a0
      m = re.search(r'(?:#|//)\W+(?:A\d+\W+)?(\w\d+)', line)
      if not m:
        print(f'ERROR expected to find A pointers: {line}', file=sys.stderr)
        sys.exit(1)
      a_pointers.append(m.group(1))
      continue
    elif 'A pointers' in line:
      prologue.append(fix_comments(line.rstrip()))
      in_a_pointers = True
      continue
    elif in_c_pointers:
      prologue.append(fix_comments(line.rstrip()))
      if not line.strip():
        in_c_pointers = False
        continue
      # Handle both:
      # - C0 x1
      # - x1 c0
      m = re.search(r'(?:#|//)\W+(?:C\d+\W+)?(\w\d+)', line)
      if not m:
        print(f'ERROR expected to find C pointers: {line}', file=sys.stderr)
        sys.exit(1)
      c_pointers.append(m.group(1))
      continue
    elif 'C pointers' in line:
      prologue.append(fix_comments(line.rstrip()))
      in_c_pointers = True
      continue
    elif in_register_usage:
      prologue.append(fix_comments(line.rstrip()))
      if line.strip() == '':
        in_register_usage = False
        continue
      if any(word in line.lower()
             for word in ['unused', 'scratch', 'temp']):
        continue
      # Skip b vector registers.
      if re.search(r'(?:#|//)\W+B', line):
        continue

      # Look for clamps
      m = re.search(r'(?:#|//)\W+[Cc]lamp\W+\(?([vr]\d+)\)?', line)
      if m:
        vector_register_usage['clamp'].append(m.group(1))
        continue

      m = re.search(r'(?:#|//)\W+(A|C)\d?\W+(((?:v|d|q|r|x)\d+(?:\[\d+\])?(?:\W*|-))+)', line)
      if not m:
        print(
            f'ERROR failed to parse vector register usage: {line}',
            file=sys.stderr)
        sys.exit(1)
      param_reg = m.group(1)
      all_regs = m.group(2).split()
      pointer_regs = [reg for reg in all_regs if reg.startswith('r') or reg.startswith('x')]

      vec_regs = [reg for reg in all_regs if reg not in pointer_regs]
      # Support old comments where the pointers are specified separately (len ==
      # 0), and also new comments where we should only have 1 pointer register.
      if len(pointer_regs) > 1:
        print(f'ERROR unexpected pointer registers: {pointer_regs}', file=sys.stderr)
        sys.exit(1)
      if len(pointer_regs) == 1:
        if param_reg.startswith('A'):
          a_pointers.append(pointer_regs[0])
        elif param_reg.startswith('C'):
          c_pointers.append(pointer_regs[0])
        else:
          print(f'ERROR unrecognized param register {param_reg}', file=sys.stderr)
          sys.exit(1)
      vector_register_usage[param_reg].append(vec_regs)
      continue
    elif 'register usage' in line.lower():
      prologue.append(fix_comments(line.rstrip()))
      in_register_usage = True
      continue
    elif any(re.fullmatch(p, line) for p in IGNORE_LINES):
      continue
    elif in_autogen:
      continue
    else:
      prologue.append(fix_comments(line.rstrip()))
      continue

  # check that number of registers matches mr
  if a_pointers and len(a_pointers) != int(mr):
    print(f'len(a_pointers) {len(a_pointers)} != mr {mr}', file=sys.stderr)
    sys.exit(1)
  if c_pointers and len(c_pointers) != int(mr):
    print('len(c_pointers) != mr', file=sys.stderr)
    sys.exit(1)

  for i, v in enumerate(a_pointers):
    vector_register_map[v] = i
  for i, v in enumerate(c_pointers):
    vector_register_map[v] = i

  for reg_alphabet, v in vector_register_usage.items():
    for i, _ in enumerate(v):
      for j, _ in enumerate(v[i]):
        # register maps specify v registers, but we sometimes refer to the q/s/d registers as well.
        regs = v[i][j].split('-')
        for reg in regs:
          vector_register_map[reg] = i
          if '[' in reg:  # handle cases in v0[1]
            continue
          vector_register_map[reg.replace('v', 'q')] = i
          vector_register_map[reg.replace('v', 'd')] = i
          vector_register_map[reg.replace('v', 's')] = i
          vector_register_map[reg.replace('v', 'h')] = i
          vector_register_map[reg.replace('v', 'b')] = i
          # TODO d registers and aarch32 should use q

  return prologue, vector_register_map, vector_register_usage


def emit_prefetch_instruction(instr: str, prfm_mode: PrfmMode,
                              instructions: List[str]) -> None:
  """Emit instructions depending on prfm_mode.

  If prfm_mode is PrfmInFileName, guard instruction behind a prefetch check. instr should be
  the generated prefetch instruction (not the assembly instruction).
  If prfm_mode is ForcePrfm, emit unguarded prefetch.
  """
  if prfm_mode == PrfmMode.PrfmInFileName:
    instructions.append(f'if (prefetch) {{ {instr} }}')
  elif prfm_mode == PrfmMode.ForcePrfm:
    instructions.append(f'{instr}')


def emit_clamp_instruction(instr: str, instructions: List[str]) -> None:
  """Guard fmax/fmin instructions behind a clamp_min/clamp_max check."""
  if 'fmax' in instr or 'vmax' in instr:
    instructions.append(f'if (clamp_min) {{ {instr} }}')
  elif 'fmin' in instr or 'vmin' in instr:
    instructions.append(f'if (clamp_max) {{ {instr} }}')
  else:
    instructions.append(instr)


def emit_instruction(instr: str,
                     instructions: List[str],
                     vector_register_map: Mapping[str, int],
                     vector_register_usage = None,
                     is_a53: bool = False) -> None:
  # If we have a trailing comment indicating the mr, we use it.
  m = re.search(r'// (?:A|C)(\d+)(?:\W+\w+)?$', instr)
  if m:
    if m[1] == '0':
      instructions.append(instr)
    else:
      instructions.append(f'if (max_mr > {m[1]}) {{ {instr} }}')
    return

  # emit a guard for instruction if it is using a particular register
  # mapping is a map from register to the mr it is used, e.g. v0 -> 0 to guard v0 behind max_mr > 0.
  # parse the dest register from this instruction
  # Match instructions like:
  # - add(r6, r8, r6)
  # - vld1_32({d3}, mem[r0]++)
  # - vldm(mem[r0]++, {s6})
  pat = re.compile(r'(\w+)\((?:mem\[)?\{?((?:v|q|d|s|h|b|x|w|r)\d+(?:\.d\(\)\[\d\])?)')
  m = re.search(pat, instr)
  if not m:
    instructions.append(instr)
    return

  instr_name = m.group(1)
  reg = m.group(2)

  # Find instructions like this:
  # - ins(v3.d()[1], x4) (converts this to v3[1], that's what is used in comments
  m = re.search(r'(v\d+)\.d\(\)(\[\d\])', reg)
  if m:
    reg = f'{m[1]}{m[2]}'

  # Find instructions like:
  #   ld1({v0.d()}, 1, mem[x9], 8); // a1
  # and use the destination register as the reg.
  m = re.search(r'ld1\(\{(v\d+)\.[ds]\(\)\},\W+\d+,\W+mem\[(x\d+)\]', instr)
  if m:
    reg = f'{m[2]}'

  if instr_name in ['fmin', 'fmax', 'vmin_f32', 'vmax_f32']:
    max_mr = vector_register_map[reg]
    if (max_mr == 0):
      return emit_clamp_instruction(instr, instructions)
    else:
      return emit_clamp_instruction(
          f'if (max_mr > {max_mr}) {{ {instr.rstrip()} }}', instructions)

  if instr_name == 'ld2r' or instr_name == 'vld1r_32':  # loading min/max
    instructions.append(f'if (clamp_min || clamp_max) {{ {instr} }}')
    return

  if ((instr_name == 'stp' or instr_name == 'ldp') and
      reg in get_callee_saved()) and 'mem[sp' in instr:
    # pushing and popping from stack, no max_mr guard.
    instructions.append(instr)
    return

  # In some AArch64 GEMM microkernels, we use ldp to load 2 A pointers, we need
  # to split this up based on max_mr.
  if (instr_name == 'ldp'):
    m = re.search(r'ldp\((x\d+), (x\d+), (mem\[x\d+\]), (\d+)', instr)
    if m:
      reg1 = m[1]
      reg2 = m[2]
      mem = m[3]
      offset = m[4]
      if all(reg in vector_register_map for reg in [reg1, reg2]):
        max_mr = vector_register_map[reg2]
        instructions.append(f'if (max_mr == {max_mr}) {{ ldr({reg1}, {mem}, {int(offset)//2}); }}');
        instructions.append(f'if (max_mr > {max_mr}) {{ ldp({reg1}, {reg2}, {mem}, {offset}); }}');
        return

  cmp_m = re.search(r'cmp\((?:x|r)0, (\d+)\);', instr)
  if cmp_m:
    instructions.append(f'if (max_mr > {int(cmp_m[1])-1}) {{ {instr} }}')
    return

  if instr_name == 'push' or instr_name == 'pop':
    instructions.append(instr)
    return

  if 'mem[sp' in instr:  # loading from stack is almost always a parameter load.
    instructions.append(instr)
    return

  if '// ks -= MR' in instr or '// a += MR' in instr:
    # Rewrite based on max_mr (since the actual number depends on max_mr).
    instructions.append(
        re.sub(r'(add|subs)\((\w\d+), (\w\d+), (\d+)\);',
               r'\1(\2, \3, max_mr * sizeof(void*));', instr))
    return

  if instr_name == 'ldr':
    # Some microkernels use ldr to load into a temporary register. In that case,
    # the temporary register doesn't show up in the register usage map, so using
    # the dest reg for max_mr guard doesn't work, we need to use the source
    # register instead.
    if reg not in vector_register_map:
      # extract register from the load
      ldr_m = re.search(r'mem\[([rx]\d+)', instr)
      if ldr_m:
        reg = ldr_m[1]

  # Duping of clamp values.
  if instr_name == 'dup':
    # Check if the source register is used to keep clamp values.
    regs = re.findall('(v\d+)[^a-z]', instr)
    clamp_reg = vector_register_usage['clamp']
    if len(regs) > 1 and regs[1] in clamp_reg:
      instructions.append(instr)
      return

  if reg not in vector_register_map:
    instructions.append(instr)
    return

  max_mr = vector_register_map[reg]
  if (max_mr == 0):
    instructions.append(instr)
    return

  instructions.append(f'if (max_mr > {max_mr}) {{ {instr} }}')


def parse_microkernel(
    lines: List[str], prfm_mode: PrfmMode, is_a53: bool,
    vector_register_map: Mapping[str, int], vector_register_usage) -> Tuple[List[str], List[str]]:
  # All labels need to be declared first, collect them and output them after
  # function signature.
  labels = []
  sc = ';'
  instructions = []
  for line in lines:
    # We are now in the microkernel function body.
    # Don't keep the ifdefs.
    m = re.fullmatch(IFDEF_RE, line)
    if m:
      continue
    # But keep other comments.
    m = re.fullmatch(COMMENT_RE, line)
    if m:
      emit_instruction(m[1], instructions, vector_register_map, vector_register_usage)
      continue

    m = re.fullmatch(LABEL, line)
    if m:
      labels.append(m[1])
      instructions.append(f'bind(l{m[1]}){sc}')
      continue
    m = re.fullmatch(INSTR_RE, line)
    if m:
      emit_instruction(f'{fix_instr_name(m[1])}(){sc} {m[2]}', instructions,
                       vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_OP_RE, line)
    if m:
      emit_instruction(f'{fix_instr_name(m[1])}({m[2]}){sc} {m[3]}',
                       instructions, vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_REGLIST_CONSEC_MEMOP_REG, line)
    if m:
      emit_instruction(
          f'{fix_instr_name(m[1])}({{{m[2]}-{m[3]}}}, mem[{m[4]}], {m[5]}){sc} {m[6]}',
          instructions, vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_REGLIST_INDIV_MEMOP_REG, line)
    if m:
      emit_instruction(
          f'{fix_instr_name(m[1])}({{{fix_regs(m[2])}}}, mem[{m[3]}], {m[4]}){sc} {m[5]}',
          instructions, vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_REGLIST_CONSEC_RE, line)
    if m:
      emit_instruction(f'{fix_instr_name(m[1])}({{{m[2]}-{m[3]}}}){sc} {m[4]}',
                       instructions, vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_REGLIST_LIST_RE, line)
    if m:
      emit_instruction(f'{fix_instr_name(m[1])}({{{m[2]}}}){sc} {m[3]}',
                       instructions, vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_MEMOP_OFFSET_RE, line)
    if m:
      if m[1].lower() == 'pld':
        emit_prefetch_instruction(
            f'{fix_instr_name(m[1])}(mem[{m[2]}, {m[3]}]){sc} {m[4]}', prfm_mode,
            instructions)
      else:
        emit_instruction(
            f'{fix_instr_name(m[1])}(mem[{m[2]}, {m[3]}]){sc} {m[4]}',
            instructions, vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_MEMOP_RE, line)
    if m:
      if m[1].lower() == 'pld':
        emit_prefetch_instruction(
            f'{fix_instr_name(m[1])}(mem[{m[2]}]){sc} {m[4]}', prfm_mode,
            instructions)
      else:
        emit_instruction(
            f'{fix_instr_name(m[1])}(mem[{m[2]}]){sc} {m[4]}',
            instructions, vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_REG_MEMOP_RE, line)
    if m:
      emit_instruction(
          f'{fix_instr_name(m[1])}({m[2]}, mem[{m[3]}]){sc} {m[4]}',
          instructions, vector_register_map, is_a53, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_REG_MEMOP_IMM_RE, line)
    if m:
      emit_instruction(
          f'{fix_instr_name(m[1])}({m[2]}, mem[{m[3]}], {m[4]}){sc} {m[5]}',
          instructions, vector_register_map, is_a53, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_REG_MEMOP_OFFSET_RE, line)
    if m:
      if m[5]:  # wb
        emit_instruction(
            f'{fix_instr_name(m[1])}({m[2]}, mem[{m[3]}, {m[4]}]++){sc} {m[6]}',
            instructions, vector_register_map, is_a53, vector_register_usage)
      else:  # no wb
        emit_instruction(
            f'{fix_instr_name(m[1])}({m[2]}, mem[{m[3]}, {m[4]}]){sc} {m[6]}',
            instructions, vector_register_map, is_a53, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_REG_REG_MEMOP_RE, line)
    if m:
      emit_instruction(
          f'{fix_instr_name(m[1])}({m[2]}, {m[3]}, mem[{m[4]}]){sc} {m[5]}',
          instructions, vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_REG_REG_MEMOP_OFFSET_RE, line)
    if m:
      if m[6]:  # wb
        emit_instruction(
            f'{fix_instr_name(m[1])}({m[2]}, {m[3]}, mem[{m[4]}, {m[5]}]++){sc} {m[7]}',
            instructions, vector_register_map, vector_register_usage)
      else:  # no wb
        emit_instruction(
            f'{fix_instr_name(m[1])}({m[2]}, {m[3]}, mem[{m[4]}, {m[5]}]){sc} {m[7]}',
            instructions, vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_REG_REG_MEMOP_IMM_RE, line)
    if m:
      emit_instruction(
          f'{fix_instr_name(m[1])}({m[2]}, {m[3]}, mem[{m[4]}], {m[5]}){sc} {m[6]}',
          instructions, vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_REG_IMM_RE, line)
    if m:
      emit_instruction(
          f'{fix_instr_name(m[1])}({fix_regs(m[2])}, {m[3]}){sc} {m[4]}',
          instructions, vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_REG_REG_REG_RE, line)
    if m:
      emit_instruction(
          f'{fix_instr_name(m[1])}({fix_regs(m[2])}, {fix_regs(m[3])}, {fix_regs(m[4])}){sc} {m[5]}',
          instructions, vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_REG_REG_REG_IMM_RE, line)
    if m:
      emit_instruction(
          f'{fix_instr_name(m[1])}({m[2]}, {m[3]}, {m[4]}, {m[5]}){sc} {m[6]}',
          instructions, vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_REG_REG_RE, line)
    if m:
      emit_instruction(
          f'{fix_instr_name(m[1])}({fix_regs(m[2])}, {fix_regs(m[3])}){sc} {m[4]}',
          instructions, vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_REG_REGLIST_CONSECT, line)
    if m:
      emit_instruction(
          f'{fix_instr_name(m[1])}(mem[{m[2]}], {{{m[3]}-{m[4]}}}){sc} {m[5]}',
          instructions, vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_REG_REGLIST_CONSECT_WB, line)
    if m:
      emit_instruction(
          f'{fix_instr_name(m[1])}(mem[{m[2]}]++, {{{m[3]}-{m[4]}}}){sc} {m[5]}',
          instructions, vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_REG_REGLIST_INDIV_WB, line)
    if m:
      emit_instruction(
          f'{fix_instr_name(m[1])}(mem[{m[2]}]++, {{{m[3]}}}){sc} {m[4]}',
          instructions, vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_B_IMM, line)
    if m:
      emit_instruction(f'{fix_instr_name(m[1])}(l{m[2]}){sc} {m[4]}',
                       instructions, vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_B_REG_IMM_IMM, line)
    if m:
      instructions.append(
          f'{fix_instr_name(m[1])}({m[2]}, {m[3]}, l{m[4]}){sc} {m[6]}')
      continue
    m = re.fullmatch(INSTR_REGLIST_INDIV_MEMOP, line)
    if m:
      emit_instruction(
          f'{fix_instr_name(m[1])}({{{fix_regs(m[2])}}}, mem[{m[3]}]{maybe_wb(m[4])}){sc} {m[5]}',
          instructions, vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_REGLIST_INDIV_MEMOP_IMM, line)
    if m:
      emit_instruction(
          f'{fix_instr_name(m[1])}({{{fix_regs(m[2])}}}, mem[{m[3]}], {m[4]}){sc} {m[5]}',
          instructions, vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_REGLIST_INDEX_MEMOP_IMM, line)
    if m:
      emit_instruction(
          f'{fix_instr_name(m[1])}({{{m[2]}()}}, {m[3]}, mem[{m[4]}], {m[5]}){sc} {m[6]}',
          instructions, vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_REG_INDEX_REG, line)
    if m:
      emit_instruction(
          f'{fix_instr_name(m[1])}({m[2]}()[{m[3]}], {m[4]}){sc} {m[5]}',
          instructions, vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_REGLIST_CONSEC_MEMOP, line)
    if m:
      emit_instruction(
          f'{fix_instr_name(m[1])}({{{m[2]}-{m[3]}}}, mem[{m[4]}]{maybe_wb(m[5])}){sc} {m[6]}',
          instructions, vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_REGLIST_REPLICATE_MEMOP, line)
    if m:
      if m[5]:
        emit_instruction(
            f'{fix_replicate_instruction(fix_instr_name(m[1]))}({{{remove_brackets(m[2])}}}, mem[{m[4]}]++){sc} {m[6]}',
            instructions, vector_register_map, vector_register_usage)
      else:
        emit_instruction(
            f'{fix_replicate_instruction(fix_instr_name(m[1]))}({{{remove_brackets(m[2])}}}, mem[{m[4]}]){sc} {m[6]}',
            instructions, vector_register_map, vector_register_usage)
      continue
    m = re.fullmatch(INSTR_REGLIST_INDEX_MEMOP, line)
    if m:
      emit_instruction(
          f'{fix_instr_name(m[1])}({{{m[2]}}}, mem[{m[3]}]{maybe_wb(m[4])}){sc} {m[5]}',
          instructions, vector_register_map, vector_register_usage)
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
      emit_prefetch_instruction(
          f'{fix_instr_name(m[1])}(k{m[2]}, mem[{m[3]}]){sc} {m[4]}', prfm_mode,
          instructions)
      continue
    m = re.fullmatch(INSTR_PLD_MEMOP_OFFSET, line)
    if m:
      emit_prefetch_instruction(
          f'{fix_instr_name(m[1])}(k{m[2]}, mem[{m[3]}, {m[4]}]){sc} {m[5]}',
          prfm_mode, instructions)
      continue
    m = re.fullmatch(INSTR_REG_REG_REG_COND_RE, line)
    if m:
      emit_instruction(
          f'{fix_instr_name(m[1])}({m[2]}, {m[3]}, {m[4]}, k{m[5]}){sc} {m[6]}',
          instructions, vector_register_map, vector_register_usage)
      continue

    # Keep empty lines for formatting
    if line.strip() == '':
      instructions.append('')
      continue

    # Assembly directives that we don't are about.
    if line.strip().startswith('.'):
      continue

    if line.startswith('END_FUNCTION'):
      break

    # All other lines are error.
    print(f'ERROR: {line}', file=sys.stderr)
    sys.exit(1)

  return instructions, labels


def emit_instructions_with_same_check(check: str, instrs: List[str],
                                      output: List[str]) -> None:
  """A helper method to emit a list of instructions which share the same check.
  """
  if not instrs:
    return
  m = re.search(r'(\W*)if \(([^)]+)\) \{', instrs[0])
  indent = ''
  check = ''
  if m:
    indent = m[1]
    check = m[2]
  output.append(f'{indent}if ({check}) {{')
  for instr in instrs:
    if instr.strip().startswith('//'):  # comment line
      output.append(f'  {instr}')
      continue
    # Each instruction is of the form "if (check) { instr(); }", we only want the instr,
    # so find the opening and closing brace, and only output what's in between.
    start = instr.index('{')
    end = instr.rindex('}')
    output.append(f'  {indent}{instr[start+1:end].strip()}')
  output.append(f'{indent}}}')


def merge_consecutive_checks(instructions: List[str]) -> List[str]:
  """Each instruction has its own check, leading to excessive number of checks, e.g.

    if (clamp) { fmin(v0) }
    if (clamp) { fmin(v1) }
    ...
    if (clamp) { fmin(v10) }

  This walks the instructions stream, checks for consecutive checks for the same
  condition,
  and merge them:

    if (clamp) {
      fmin(v0);
      fmin(v1);
      ...
      fmin(v10);
  }

  This assumes that checks should be on the same line as the instruction, e.g.
  `if (clamp) { ... }`
  """
  previous_check = None
  current_check = None
  # Avoid mutating the input instructions stream, write all output to this new list.
  output = []
  # Holds the list of instructions that have the same check.
  instructions_with_same_check = []
  # cases we consider:
  # 1. Same check
  #     if (checkA) {}
  #     if (checkA) {}
  # 2. Different check
  #     if (checkA) {}
  #     if (checkB) {}
  # 3. Comment line
  #     // comment
  # 4. Everything else
  #     instruction
  for instr in instructions:
    m = re.search(r'if \(([^)]+)\) \{ .+', instr)
    if m:
      current_check = m.group(1)
      if (current_check == previous_check):
        instructions_with_same_check.append(instr)
      else:
        # Special case if the previous line is a comment.
        comment = ''
        if output and output[-1].strip().startswith('//'):
          comment = output.pop()
        elif instructions_with_same_check and instructions_with_same_check[
            -1].strip().startswith('//'):
          comment = instructions_with_same_check.pop()
        emit_instructions_with_same_check(previous_check,
                                          instructions_with_same_check, output)
        previous_check = current_check
        instructions_with_same_check = [instr]
        if comment:
          # Need to wrap comment with this check so that we can merge the checks correctly, this is not valid C code due
          # to the // comment, but we will remove that.
          instructions_with_same_check.insert(
              0, f'if ({current_check}) {{ {comment} }}')
    elif re.fullmatch(r'\W*//.+', instr) and len(
        instructions_with_same_check) != 0:  # purely comment line
      instructions_with_same_check.append(instr)
    else:
      emit_instructions_with_same_check(previous_check,
                                        instructions_with_same_check, output)
      previous_check = None
      output.append(instr)
      instructions_with_same_check = []
  return output


def insert_post_operations(instructions: List[str]):
  index = 0
  # Look for the comment marking where we store full tile, that's where we will
  # perform post operations.
  for i, l in enumerate(instructions):
    if 'Store full ' in l:
      index = i
      break
  assert (instructions[index - 1].strip() == '')
  instructions.insert(
      index - 1,
      'perform_post_operations(max_mr, num_post_operations, post_operations);')
  return instructions


def find_params_offset_and_register(lines: List[str]) -> Tuple[str, str]:
  for line in lines:
    if 'params' in line:
      params_m = re.search(r'sp\W+\+\W+(\d+)', line)
      reg_m = re.search(r'((?:r|x)\d+)', line.split()[-1])
      if params_m and reg_m:
        return params_m[1], reg_m[1]
  return None, None


def convert(input_file: str, post_op: bool, reload_params: bool, debug: bool, force_prfm: bool) -> None:
  output = []
  arch = None
  kernel_type = GEMM
  minmax = False
  base_filename = os.path.basename(input_file)
  if base_filename.startswith('f16-'):
    ctype = 'uint16_t'
  elif base_filename.startswith('f32-'):
    ctype = 'float'
  elif base_filename.startswith('qs8-') or base_filename.startswith('qc8-'):
    ctype = 'int8_t'
  elif base_filename.startswith('qu8-'):
    ctype = 'uint8_t'
  else:
    print('ERROR: unknown ctype')
    sys.exit(1)

  if 'aarch32' in input_file:
    arch = AARCH32
  elif 'aarch64' in input_file:
    arch = AARCH64
  else:
    print('ERROR: unknown architecture')
    sys.exit(1)

  if 'igemm' in input_file:
    kernel_type = IGEMM
  if 'minmax' in input_file:
    minmax = True
  prfm_mode = PrfmMode.NoPrfm
  if 'prfm' in input_file:
    prfm_mode = PrfmMode.PrfmInFileName
    assert(not force_prfm)
  if force_prfm:
    prfm_mode = PrfmMode.ForcePrfm

  mr = 0
  nr = 0
  m = re.search(r'(\d+)x(\d+)', input_file)
  if m:
    mr = int(m[1])
    nr = int(m[2])

  # Instructions that make up the microkernel.
  instructions = []
  # Lines of code or comments before the actual function body.
  prologue = []
  # Name of the microkernel function.
  fn_name = ''

  lines = []
  with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()

  begin_function_index = 0
  for i, line in enumerate(lines):
    if 'BEGIN_FUNCTION' in line:
      begin_function_index = i
      break

  fn_name = lines[begin_function_index].split()[1]

  # Prologue includes the BEING_FUNCTION declaration of function name.
  prologue_lines = lines[:begin_function_index + 1]
  # Microkernel body does not include the BEGIN_FUNCION.
  microkernel_body = lines[begin_function_index + 1:]

  prologue, vector_register_map, vector_register_usage = parse_prologue(input_file, prologue_lines,
                                                 arch, minmax, kernel_type,
                                                 prfm_mode, mr, post_op)
  if debug:
    print(vector_register_map)
    print(vector_register_usage)

  params_offset, params_register = find_params_offset_and_register(
      prologue_lines)
  if not params_register:
    print(fn_name)
    print('Unable to find params register')
    sys.exit(1)

  is_a53 = 'cortex_a53' in fn_name
  instructions, labels = parse_microkernel(microkernel_body, prfm_mode, is_a53,
                                           vector_register_map, vector_register_usage)
  # TODO(zhin): iterate until fixpoint instead.
  instructions = merge_consecutive_checks(instructions)
  instructions = merge_consecutive_checks(instructions)
  if post_op:
    instructions = insert_post_operations(instructions)

  # Actually emit the JIT codegen (to stdout).
  for p in prologue:
    output.append(p)

  labels_str = ', '.join(f'l{l}' for l in labels)
  output.append(f'  assert(max_mr <= {mr});')
  output.append(f'  assert(nc_mod_nr < {nr} || nc_mod_nr == SIZE_MAX);')
  output.append('  assert(kc != 0);')
  output.append(f'  assert(kc % sizeof({ctype}) == 0);')
  if kernel_type == IGEMM:
    output.append('  assert(ks != 0);')
  output.append('')
  output.append(f'  Label {labels_str};')
  output.append(
      '  const size_t num_post_operations = jit_gemm_params->num_post_operations;'
  )
  if not post_op:
    output.append('  (void) num_post_operations;  // Silence unused warning.')
  if post_op:
    output.append(
        '  const xnn_post_operation* post_operations = jit_gemm_params->post_operations;'
    )

  if minmax:
    if ctype == 'float':
      output.append('  const float min = jit_gemm_params->f32_minmax.min;')
      output.append('  const float max = jit_gemm_params->f32_minmax.max;')
      output.append('  const bool clamp_min = min != -std::numeric_limits<float>::infinity();')
      output.append('  const bool clamp_max = max != +std::numeric_limits<float>::infinity();')
    elif ctype == 'uint16_t':
      output.append('  const uint16_t min = jit_gemm_params->f16_minmax.min;')
      output.append('  const uint16_t max = jit_gemm_params->f16_minmax.max;')
      output.append('  const bool clamp_min = min != UINT16_C(0xFC00);  // -Inf.')
      output.append('  const bool clamp_max = max != UINT16_C(0x7C00);  // Inf.')
    else:
      print('ERROR: unknown ctype for min/max params')
      sys.exit(1)

    output.append(
        '  assert(num_post_operations == 0 || (!clamp_min && !clamp_max));')

  indent = '  '
  for i in instructions:
    if i.strip().startswith('#'):
      output.append(indent + fix_comments(i))
    elif i.strip().startswith('//'):
      output.append(indent + i)
    elif i.strip() == '':
      output.append('')
    else:
      output.append(indent + (i).rstrip())
  if arch == AARCH32:
    output.append(indent + 'align(16);')
  else:
    output.append(indent + 'align(16, AlignInstruction::kHlt);')

  output.append('}')
  # print post operations definition
  if post_op:
    output.append('')
    output.append(
        get_post_operation_implementation(arch, mr, params_register,
                                          params_offset, reload_params, vector_register_usage))
    output.append('')
  output.append('}  // namespace')
  output.append(f'}}  // namespace {arch}')
  output.append('}  // namespace xnnpack')
  output.append('')
  if prfm_mode == PrfmMode.PrfmInFileName:
    print_generator_definition(
        output,
        kernel_type,
        remove_prfm_from_fn_name(fn_name),
        arch,
        minmax,
        prefetch='false, ')
    output.append('')
    print_generator_definition(
        output, kernel_type, fn_name, arch, minmax, prefetch='true, ')
  else:
    print_generator_definition(output, kernel_type, fn_name, arch, minmax)

  return output


def print_generator_definition(output,
                               kernel_type,
                               fn_name,
                               arch,
                               minmax,
                               prefetch=''):
  if kernel_type == GEMM:
    output.append(
        f'xnn_status_t {fix_fn_name(fn_name)}(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, const void* params) {{'
    )
  else:
    output.append(
        f'xnn_status_t {fix_fn_name(fn_name)}(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const void* params) {{'
    )
  output.append(f'  using namespace xnnpack::{arch};')
  output.append('  Generator g(code);')
  if minmax:
    output.append('  assert(params != nullptr);')
  if kernel_type == GEMM:
    if minmax:
      output.append(
          f'  g.generate({prefetch}max_mr, nc_mod_nr, kc, static_cast<const jit_gemm_params*>(params));'
      )
    else:
      output.append(f'  g.generate({prefetch}max_mr, nc_mod_nr, kc, nullptr);')
  else:
    if minmax:
      output.append(
          f'  g.generate({prefetch}max_mr, nc_mod_nr, kc, ks, static_cast<const jit_gemm_params*>(params));'
      )
    else:
      output.append(
          f'  g.generate({prefetch}max_mr, nc_mod_nr, kc, ks, nullptr);')
  output.append('  g.finalize();')
  output.append('  if (g.error() != xnnpack::Error::kNoError) {')
  output.append('    return xnn_status_invalid_state;')
  output.append('  }')
  output.append('  return xnn_status_success;')
  output.append('}')


def main(sys_args):
  parser = argparse.ArgumentParser(
      description='Convert assembly to to JIT C++, writes to stdout.')
  parser.add_argument(
      '-i',
      '--input',
      metavar='input_file',
      help='Input assembly filename',
      required=True)
  parser.add_argument(
      '-o',
      '--output',
      metavar='output_file',
      help='Output cc filename',
      required=True)
  parser.add_argument(
      '--post-op',
      help='Should support post operation',
      default=True,
      action=argparse.BooleanOptionalAction)
  parser.add_argument(
      '--reload-params',
      help='Should reload params pointer before post operation',
      default=False,
      action=argparse.BooleanOptionalAction)
  parser.add_argument(
      "--debug",
      help='Output debugging information',
      default=False,
      action=argparse.BooleanOptionalAction)
  parser.add_argument(
      "--force-prfm",
      help='Force PRFM instructions in output',
      default=False,
      action=argparse.BooleanOptionalAction)
  args = parser.parse_args(sys_args)

  output = '\n'.join(convert(args.input, args.post_op, args.reload_params, args.debug, args.force_prfm))
  # Add trailing new line.
  output += '\n'

  output_name = args.output
  txt_changed = True
  if os.path.exists(output_name):
    with codecs.open(output_name, 'r', encoding='utf-8') as output_file:
      ofr = output_file.read()
      txt_changed = ofr != output
  if txt_changed:
    with codecs.open(output_name, 'w', encoding='utf-8') as output_file:
      output_file.write(output)


if __name__ == '__main__':
  main(sys.argv[1:])
