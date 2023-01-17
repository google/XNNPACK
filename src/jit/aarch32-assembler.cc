// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>
#include <cstddef>

#include "xnnpack/aarch32-assembler.h"
#include "xnnpack/assembler.h"
#include "xnnpack/common.h"
#include "xnnpack/math.h"

namespace xnnpack {
namespace aarch32 {
// Max value of imm for vldr/str (takes imm8, but shift right by 2 when encoding).
constexpr int32_t kUint10Max = 1023;
// Max value of imm that fits in ldr/str encoding (takes imm12, with a separate bit for sign).
constexpr int32_t kUint12Max = 4095;

// PC register contains current address of instruction + 8 (2 instructions).
constexpr ptrdiff_t kPCDelta = 8;
// Constants used for checking branch offsets bounds.
constexpr ptrdiff_t kInt24Max = 8388607;
constexpr ptrdiff_t kInt24Min = -8388608;

// Check if a branch offset is valid, it must fit in 24 bits.
bool branch_offset_valid(ptrdiff_t offset) {
  return offset < kInt24Max && offset > kInt24Min;
}

bool invalid_register_list(DRegisterList regs) {
  return regs.length == 0 || regs.length > 16 || regs.start.code + regs.length > 32;
}

bool invalid_register_list(SRegisterList regs) {
  return regs.length == 0 || regs.start.code + regs.length > 32;
}

uint32_t encode(SRegister r, uint32_t single_bit_pos, uint32_t four_bits_pos) {
  return r.d() << single_bit_pos | r.vd() << four_bits_pos;
}

uint32_t encode(DRegister r, uint32_t single_bit_pos, uint32_t four_bits_pos) {
  return r.d() << single_bit_pos | r.vd() << four_bits_pos;
}

uint32_t encode(DRegisterLane r, uint32_t single_bit_pos, uint32_t four_bits_pos) {
  return r.d() << single_bit_pos | r.vd() << four_bits_pos;
}

uint32_t encode(QRegister r, uint32_t single_bit_pos, uint32_t four_bits_pos) {
  return r.d() << single_bit_pos | r.vd() << four_bits_pos;
}

uint32_t encode(SRegisterList regs, uint32_t single_bit_pos, uint32_t four_bits_pos) {
  const SRegister r = regs.start;
  return r.d() << single_bit_pos | r.vd() << four_bits_pos | regs.length;
}

uint32_t encode(DRegisterList regs, uint32_t single_bit_pos, uint32_t four_bits_pos) {
  const DRegister r = regs.start;
  return r.d() << single_bit_pos | r.vd() << four_bits_pos | regs.length * 2;
}

uint32_t encode_mem_puw(MemOperand op) {
  return op.p() << 24 | op.u() << 23 | op.w() << 21 | op.base().code << 16;
}

// Return value of 0 is invalid, indicates error.
uint32_t encode_regs_length_to_type(DRegisterList regs) {
  switch (regs.length) {
    case 1:
      return 0x7;
    case 2:
      return 0xA;
    case 3:
      return 0x6;
    case 4:
      return 0x2;
  }
  return 0;
}

void Assembler::add(CoreRegister rd, CoreRegister rn, CoreRegister rm) {
  emit32(kAL | 0x8 << 20 | rn.code << 16 | rd.code << 12 | rm.code);
}

void Assembler::add(CoreRegister rd, CoreRegister rn, uint8_t imm) {
  // Rotation = 0, since imm is limited to 8 bits and fits in encoding.
  emit32(kAL | 0x28 << 20 | rn.code << 16 | rd.code << 12 | imm);
}

void Assembler::adds(CoreRegister rd, CoreRegister rn, uint8_t imm) {
  // Rotation = 0, since imm is limited to 8 bits and fits in encoding.
  emit32(kAL | 0x29 << 20 | rn.code << 16 | rd.code << 12 | imm);
}

void Assembler::and_(CoreRegister rd, CoreRegister rn, uint8_t imm) {
  // Rotation = 0, since imm is limited to 8 bits and fits in encoding.
  emit32(kAL | 1 << 25 | rn.code << 16 | rd.code << 12 | imm);
}

void Assembler::b(Condition c, Label& l) {
  if (l.bound) {
    // Offset is relative to after this b instruction + kPCDelta.
    const ptrdiff_t offset = l.offset - cursor_ - kPCDelta;
    if (!branch_offset_valid(offset)) {
      error_ = Error::kLabelOffsetOutOfBounds;
      return;
    }

    // No need to shift by 2 since our offset is already in terms of uint32_t.
    emit32(c | 0xA << 24 | ((offset >> kInstructionSizeInBytesLog2) & 0x00FFFFFF));
  } else {
    if (!l.add_use(cursor_)) {
      error_ = Error::kLabelHasTooManyUsers;
      return;
    }
    // Emit 0 offset first, will patch it up when label is bound later.
    emit32(c | 0xA << 24);
  }
}

void Assembler::bind(Label& l) {
  if (error_ != Error::kNoError) {
    return;
  }

  if (l.bound) {
    error_ = Error::kLabelAlreadyBound;
    return;
  }

  l.bound = true;
  l.offset = cursor_;

  // Patch all users.
  for (size_t i = 0; i < l.num_users; i++) {
    byte* user = l.users[i];
    const ptrdiff_t offset = l.offset - user - kPCDelta;
    uint32_t* instr = reinterpret_cast<uint32_t*>(user);

    if (!branch_offset_valid(offset)) {
      error_ = Error::kLabelOffsetOutOfBounds;
      return;
    }

    *instr |= (offset >> kInstructionSizeInBytesLog2) & 0x00FFFFFF;
  }
}

void Assembler::bic(CoreRegister rd, CoreRegister rn, uint8_t imm) {
  emit32(kAL | 0x03C00000 | rn.code << 16 | rd.code << 12 | imm);
}

void Assembler::bx(CoreRegister rm) {
  emit32(kAL | 0x12fff10 | rm.code);
}

void Assembler::cmp(CoreRegister rn, uint8_t imm) {
  emit32(kAL | 0x35 << 20 | rn.code << 16 | imm);
}

void Assembler::cmp(CoreRegister rn, CoreRegister rm) {
  emit32(kAL | 0x01500000 | rn.code << 16 | rm.code);
}

void Assembler::ldr(CoreRegister rt, MemOperand op, int32_t offset) {
  ldr(rt, MemOperand(op.base(), offset, AddressingMode::kPostIndexed));
}

void Assembler::ldr(CoreRegister rt, MemOperand op) {
  const int32_t offset = op.offset();
  if (std::abs(offset) > kUint12Max) {
    error_ = Error::kInvalidOperand;
    return;
  }

  emit32(kAL | 0x41 << 20 | encode_mem_puw(op) | rt.code << 12 | offset);
}

void Assembler::ldrd(CoreRegister rt, CoreRegister rt2, MemOperand op) {
  const int32_t offset = op.offset();
  if ((std::abs(op.offset()) > UINT8_MAX) || (rt.code + 1 != rt2.code)) {
    error_ = Error::kInvalidOperand;
    return;
  }
  const uint32_t offset_top = (offset & 0xF0) << 4;
  const uint32_t offset_bot = (offset & 0xF);

  emit32(kAL | 0x004000D0 | encode_mem_puw(op) | rt.code << 12 | offset_top | offset_bot);
}

void Assembler::mov(CoreRegister rd, CoreRegister rm) {
  mov(kAL, rd, rm);
}

void Assembler::mov(Condition c, CoreRegister Rd, CoreRegister Rm) {
  emit32(c | 0x1A << 20 | Rd.code << 12 | Rm.code);
}

void Assembler::nop() {
  emit32(kAL | 0x0320F000);
}

void Assembler::pld(MemOperand op) {
  emit32(0xF550F000 | op.u() << 23 | op.base().code << 16 | op.offset());
}

void Assembler::pop(CoreRegisterList regs) {
  if (!regs.has_more_than_one_register()) {
    // TODO(zhin): there is a different valid encoding for single register.
    error_ = Error::kInvalidOperand;
    return;
  }

  emit32(kAL | 0x8BD << 16 | regs.list);
}

void Assembler::push(CoreRegisterList regs) {
  if (!regs.has_more_than_one_register()) {
    // TODO(zhin): there is a different valid encoding for single register.
    error_ = Error::kInvalidOperand;
    return;
  }

  emit32(kAL | 0x92D << 16 | regs.list);
}

void Assembler::str(CoreRegister rt, MemOperand op) {
  const int32_t offset = op.offset();
  if (std::abs(offset) > kUint12Max) {
    error_ = Error::kInvalidOperand;
    return;
  }
  emit32(kAL | 1 << 26 | encode_mem_puw(op) | rt.code << 12 | offset);
}

void Assembler::sub(CoreRegister rd, CoreRegister rn, uint8_t imm) {
  emit32(kAL | 0x24 << 20 | rn.code << 16 | rd.code << 12 | imm);
}

void Assembler::sub(CoreRegister rd, CoreRegister rn, CoreRegister rm) {
  emit32(kAL | 0x4 << 20 | rn.code << 16 | rd.code << 12 | rm.code);
}

void Assembler::subs(CoreRegister rd, CoreRegister rn, uint8_t imm) {
  // Rotation = 0, since imm is limited to 8 bits and fits in encoding.
  emit32(kAL | 0x25 << 20 | rn.code << 16 | rd.code << 12 | imm);
}

void Assembler::tst(CoreRegister rn, uint8_t imm) {
  // Rotation = 0, since imm is limited to 8 bits and fits in encoding.
  emit32(kAL | 0x31 << 20 | rn.code << 16 | imm);
}

void Assembler::vabs_f32(QRegister qd, QRegister qm) {
  emit32(0xF3B90740 | encode(qd, 22, 12) | encode(qm, 5, 0));
}

void Assembler::vadd_f32(QRegister qd, QRegister qn, QRegister qm) {
  emit32(0xF2000D40 | encode(qd, 22, 12) | encode(qn, 7, 16) | encode(qm, 5, 0));
}

void Assembler::vcmpe_f32(SRegister sd, SRegister sm) {
  emit32(kAL | 0x0EB40AC0 | encode(sd, 22, 12) | encode(sm, 5, 0));
}

void Assembler::vcvt_f32_s32(QRegister qd, QRegister qm) {
  emit32(0xF3BB0640 | encode(qd, 22, 12) | encode(qm, 5, 0));
}

void Assembler::vcvt_s32_f32(QRegister qd, QRegister qm) {
  emit32(0xF3BB0740 | encode(qd, 22, 12) | encode(qm, 5, 0));
}

void Assembler::vcvtn_s32_f32(QRegister qd, QRegister qm) {
  emit32(0xF3BB0140 | encode(qd, 22, 12) | encode(qm, 5, 0));
}

void Assembler::vdup(DataSize size, QRegister qd, DRegisterLane dm) {
  uint8_t imm4 = 0;
  switch (size) {
    case k8:
      if (dm.lane > 7) {
        error_ = Error::kInvalidLaneIndex;
        return;
      }
      imm4 = 1 | ((dm.lane & 0x7) << 1);
      break;
    case k16:
      if (dm.lane > 3) {
        error_ = Error::kInvalidLaneIndex;
        return;
      }
      imm4 = 2 | ((dm.lane & 0x3) << 2);
      break;
    case k32:
      if (dm.lane > 1) {
        error_ = Error::kInvalidLaneIndex;
        return;
      }
      imm4 = 4 | ((dm.lane & 0x1) << 3);
      break;
  }
  emit32(0xF3B00C40 | imm4 << 16 | encode(qd, 22, 12) | encode(dm, 5, 0));
}

void Assembler::vext_8(QRegister qd, QRegister qn, QRegister qm, uint8_t imm4) {
  if (imm4 > 15) {
    error_ = Error::kInvalidOperand;
    return;
  }
  emit32(0xF2B00040 | encode(qd, 22, 12) | encode(qn, 7, 16) | encode(qm, 5, 0) | imm4 << 8);
}

void Assembler::vld1(DataSize size, DRegisterList regs, MemOperand op) {
  const uint8_t rm = op.mode() == AddressingMode::kPostIndexed ? 0xD : 0xF;
  vld1(size, regs, op, CoreRegister{rm});
}

void Assembler::vld1(DataSize size, DRegisterList regs, MemOperand op, CoreRegister rm) {
  const uint8_t type = encode_regs_length_to_type(regs);
  if (!type) {
    error_ = Error::kInvalidRegisterListLength;
    return;
  }

  emit32(0xF4200000 | encode(regs.start, 22, 12) | op.base().code << 16 | type << 8 | size << 6 | rm.code);
}

void Assembler::vld1_32(DRegisterLane dd, MemOperand op) {
  if (dd.lane > 1) {
    error_ = Error::kInvalidLaneIndex;
    return;
  }
  const uint32_t rm = op.mode() == AddressingMode::kPostIndexed ? 0xD : 0xF;
  emit32(kAL | 0xF4A00800 | dd.lane << 7 | encode(dd, 22, 12) | op.base().code << 16 | rm);
}

void Assembler::vld1r_32(DRegisterList regs, MemOperand op) {
  if ((op.mode() == AddressingMode::kOffset && op.offset() != 0) || regs.length > 2) {
    error_ = Error::kInvalidOperand;
    return;
  }

  const uint32_t rm = op.mode() == AddressingMode::kPostIndexed ? 0xD : 0xF;
  emit32(0xF4A00C80 | encode(regs.start, 22, 12) | op.base().code << 16 | (regs.length - 1) << 5 | rm);
}

void Assembler::vld2r_32(VLoadStoreRegList regs, MemOperand op) {
  if ((op.mode() == AddressingMode::kOffset && op.offset() != 0)) {
    error_ = Error::kInvalidOperand;
    return;
  }
  uint8_t spacing = regs.double_spaced ? 2 : 1;
  if (regs.reg1.code != regs.reg2.code - spacing) {
    error_ = Error::kInvalidOperand;
    return;
  }

  size_t t = spacing - 1;
  const uint32_t rm = op.mode() == AddressingMode::kPostIndexed ? op.base().code : 0xF;
  emit32(0xF4A00D80 | encode(regs.reg1, 22, 12) | op.base().code << 16 | t << 5 | rm);
}


void Assembler::vld3r_32(VLoadStoreRegList regs, MemOperand op) {
  if ((op.mode() == AddressingMode::kOffset && op.offset() != 0)) {
    error_ = Error::kInvalidOperand;
    return;
  }
  uint8_t spacing = regs.double_spaced ? 2 : 1;
  if (regs.reg1.code != regs.reg2.code - spacing || regs.reg2.code != regs.reg3.code - spacing) {
    error_ = Error::kInvalidOperand;
    return;
  }

  size_t t = spacing - 1;
  const uint32_t rm = op.mode() == AddressingMode::kPostIndexed ? op.base().code : 0xF;
  emit32(0xF4A00E80 | encode(regs.reg1, 22, 12) | op.base().code << 16 | t << 5 | rm);
}

void Assembler::vldm(MemOperand rn, SRegisterList regs) {
  if (invalid_register_list(regs)) {
    error_ = Error::kInvalidRegisterListLength;
    return;
  }
  uint32_t w = (rn.mode() == AddressingMode::kOffset ? 0 : 1) << 21;
  emit32(kAL | 0x0C900A00 | w | rn.base().code << 16 | encode(regs, 22, 12));
}

void Assembler::vldm(MemOperand rn, DRegisterList regs) {
  if (invalid_register_list(regs)) {
    error_ = Error::kInvalidRegisterListLength;
    return;
  }
  uint32_t w = (rn.mode() == AddressingMode::kOffset ? 0 : 1) << 21;
  emit32(kAL | 0x0C900B00 | w | rn.base().code << 16 | encode(regs, 22, 12));
}

void Assembler::vldr(SRegister sd, MemOperand op) {
  const uint32_t offset = std::abs(op.offset());
  if (op.mode() != AddressingMode::kOffset || offset > kUint10Max || offset % 4 != 0) {
    error_ = Error::kInvalidOperand;
    return;
  }

  emit32(kAL | 0x0D100A00 | op.u() << 23 | encode(sd, 22, 12) | op.base().code << 16 | offset >> 2);
}

void Assembler::vldr(DRegister dd, MemOperand op) {
  const uint32_t offset = std::abs(op.offset());
  if (op.mode() != AddressingMode::kOffset || offset > kUint10Max || offset % 4 != 0) {
    error_ = Error::kInvalidOperand;
    return;
  }

  emit32(kAL | 0x0D100B00 | op.u() << 23 | encode(dd, 22, 12) | op.base().code << 16 | offset >> 2);
}

void Assembler::vmax_f32(QRegister qd, QRegister qn, QRegister qm) {
  emit32(0xF2000F40 | encode(qd, 22, 12) | encode(qn, 7, 16) | encode(qm, 5, 0));
}

void Assembler::vmax_s8(QRegister qd, QRegister qn, QRegister qm) {
 emit32(0xF2000640 | encode(qd, 22, 12) | encode(qn, 7, 16) | encode(qm, 5, 0));
}

void Assembler::vmin_f32(QRegister qd, QRegister qn, QRegister qm) {
  emit32(0xF2200F40 | encode(qd, 22, 12) | encode(qn, 7, 16) | encode(qm, 5, 0));
}

void Assembler::vmin_s8(QRegister qd, QRegister qn, QRegister qm) {
 emit32(0xF2000650 | encode(qd, 22, 12) | encode(qn, 7, 16) | encode(qm, 5, 0));
}

void Assembler::vmla_f32(SRegister sd, SRegister sn, SRegister sm) {
  emit32(kAL | 0x0E000A00 | encode(sd, 22, 12) | encode (sn, 7, 16) | encode(sm, 5, 0));
}

void Assembler::vmla_f32(QRegister qd, QRegister qn, DRegisterLane dm) {
  if (dm.lane > 1) {
    error_ = Error::kInvalidLaneIndex;
    return;
  }
  emit32(0xF3A00140 | encode(qd, 22, 12) | encode(qn, 7, 16) | dm.lane << 5 | dm.code);
}

void Assembler::vmlal_s16(QRegister qd, DRegister dn, DRegisterLane dm) {
  if (dm.lane > 3) {
    error_ = Error::kInvalidLaneIndex;
    return;
  }
  if (dm.code > 7) {
    error_ = Error::kInvalidOperand;
    return;
  }

  uint8_t lane_top = dm.lane >> 1;
  uint8_t lane_bot = dm.lane & 1;
  emit32(0xF2900240 | encode(qd, 22, 12) | encode(dn, 7, 16) | lane_top << 5 | lane_bot << 3 | dm.code);
}

void Assembler::vmov_i32(QRegister qd, uint8_t imm) {
  if (imm != 0) {
    error_ = Error::kInvalidOperand;
    return;
  }
  vmov(qd, imm);
}

void Assembler::vmov(QRegister qd, uint8_t imm) {
  if (imm != 0) {
    error_ = Error::kInvalidOperand;
    return;
  }

  emit32(0xF2800050 | encode(qd, 22, 12));
}

void Assembler::vmov(SRegister sd, SRegister sm) {
  emit32(kAL | 0x0EB00A40 | encode(sd, 22, 12) | encode(sm, 5, 0));
}

void Assembler::vmov(DRegister dm, CoreRegister rt, CoreRegister rt2) {
  emit32(kAL | 0x0C400B10 | rt2.code << 16 | rt.code << 12 | encode(dm, 5, 0));
}

void Assembler::vmov(DRegister dd, DRegister dm) {
  emit32(0xF2200110 | encode(dd, 22, 12) | encode(dm, 7, 16) | encode(dm, 5, 0));
}

void Assembler::vmov(QRegister qd, QRegister qm) {
  emit32(0xF2200150 | encode(qd, 22, 12) | encode(qm, 7, 16) | encode(qm, 5, 0));
}

void Assembler::vmov_f32(Condition c, SRegister sd, SRegister sm) {
  emit32(c | 0x0EB00A40 | encode(sd, 22, 12) | encode(sm, 5, 0));
}

void Assembler::vmov_f64(DRegister dd, DRegister dm) {
  emit32(kAL | 0x0EB00B40 | encode(dd, 22, 12) | encode(dm, 5, 0));
}

void Assembler::vmovl_s8(QRegister qd, DRegister dm) {
  emit32(0xF2880A10 | encode(qd, 22, 12) | encode(dm, 5, 0));
}

void Assembler::vmrs(CoreRegister rt, SpecialFPRegister spec_reg) {
  emit32(kAL | 0x0EF00A10 | static_cast<uint32_t>(spec_reg) << 16 | rt.code << 12);
}

void Assembler::vmul_f32(QRegister qd, QRegister qn, QRegister qm) {
  emit32(0xF3000D50 | encode(qd, 22, 12) | encode(qn, 7, 16) | encode(qm, 5, 0));
}

void Assembler::vmul_f32(QRegister qd, QRegister qn, DRegisterLane dm) {
  if (dm.lane > 1) {
    error_ = Error::kInvalidLaneIndex;
    return;
  }
  emit32(0xF3A00940 | encode(qd, 22, 12) | encode(qn, 7, 16) | encode(dm, 5, 0) | dm.lane << 5);
}

void Assembler::vneg_f32(QRegister qd, QRegister qm) {
  emit32(0xF3B907C0 | encode(qd, 22, 12) | encode(qm, 5, 0));
}

void Assembler::vpop(DRegisterList regs) {
  if (invalid_register_list(regs)) {
    error_ = Error::kInvalidRegisterListLength;
    return;
  }
  emit32(kAL | encode(regs, 22, 12) | 0xCBD << 16 | 0xB << 8);
}

void Assembler::vpush(DRegisterList regs) {
  if (invalid_register_list(regs)) {
    error_ = Error::kInvalidRegisterListLength;
    return;
  }
  emit32(kAL | encode(regs, 22, 12) | 0xD2D << 16 | 0xB << 8);
}

void Assembler::vpush(SRegisterList regs) {
  if (invalid_register_list(regs)) {
    error_ = Error::kInvalidRegisterListLength;
    return;
  }
  emit32(kAL | encode(regs, 22, 12) | 0xD2D << 16 | 0xA << 8);
}

void Assembler::vqadd_s16(QRegister qd, QRegister qn, QRegister qm) {
  emit32(0xF2100050 | encode(qd, 22, 12) | encode(qn, 7, 16) | encode(qm, 5, 0));
}

void Assembler::vqdmulh_s32(QRegister qd, QRegister qn, DRegisterLane dm) {
  if (dm.code > 15) {
    error_ = Error::kInvalidOperand;
    return;
  }
  if (dm.lane > 1) {
    error_ = Error::kInvalidLaneIndex;
    return;
  }
  emit32(0xF3A00C40 | encode(qd, 22, 12) | encode(qn, 7, 16) | dm.lane << 5 | dm.code);
}

void Assembler::vqmovn_s16(DRegister dd, QRegister qm) {
  emit32(0xF3B20280 | encode(dd, 22, 12) | encode(qm, 5, 0));
}

void Assembler::vqmovn_s32(DRegister dd, QRegister qm) {
  emit32(0xF3B60280 | encode(dd, 22, 12) | encode(qm, 5, 0));
}

void Assembler::vqshl_s32(QRegister qd, QRegister qm, QRegister qn) {
  emit32(0xF2200450 | encode(qd, 22, 12) | encode(qm, 5, 0) | encode(qn, 7, 16));
}

void Assembler::vrshl_s32(QRegister qd, QRegister qm, QRegister qn) {
  emit32(0xF2200540 | encode(qd, 22, 12) | encode(qm, 5, 0) | encode(qn, 7, 16));
}

void Assembler::vsdot_s8(QRegister qd, QRegister qn, DRegisterLane dm) {
  if (dm.lane > 1) {
    error_ = Error::kInvalidLaneIndex;
    return;
  }
  emit32(0xFE200D40 | encode(qd, 22, 12) | encode(qn, 7, 16) | dm.lane << 5 | dm.code);
}

void Assembler::vst1(DataSize size, DRegisterList regs, MemOperand op) {
  const uint8_t type = encode_regs_length_to_type(regs);
  if (!type) {
    error_ = Error::kInvalidRegisterListLength;
    return;
  }

  const uint32_t rm = op.mode() == AddressingMode::kPostIndexed ? 0xD : 0xF;
  emit32(0xF4000000 | encode(regs.start, 22, 12) | op.base().code << 16 | type << 8 | size << 6 | rm);
}

void Assembler::vst1(DataSize size, DRegisterList regs, MemOperand op, CoreRegister rm) {
  if (rm.code == 0b1101 || rm.code == 0b1111) {
    error_ = Error::kInvalidOperand;
    return;
  }

  const uint8_t type = encode_regs_length_to_type(regs);
  if (!type) {
    error_ = Error::kInvalidRegisterListLength;
    return;
  }

  emit32(0xF4000000 | encode(regs.start, 22, 12) | op.base().code << 16 | type << 8 | size << 6 | rm.code);
}

void Assembler::vst1(DataSize size, DRegisterLane dd, MemOperand op) {
  if ((size == k8 && dd.lane > 7) || (size == k16 && dd.lane > 3) || (size == k32 && dd.lane > 1)) {
    error_ = Error::kInvalidLaneIndex;
    return;
  }

  const uint8_t shift = size == k8 ? 5 : size == k16 ? 6 : 7;
  const uint32_t rm = op.mode() == AddressingMode::kPostIndexed ? 0xD : 0xF;
  emit32(0xF4800000 | encode(dd, 22, 12) | op.base().code << 16 | size << 10 | dd.lane << shift | rm);
}

void Assembler::vstm(MemOperand rn, DRegisterList regs) {
  if (invalid_register_list(regs)) {
    error_ = Error::kInvalidRegisterListLength;
    return;
  }
  uint32_t w = (rn.mode() == AddressingMode::kOffset ? 0 : 1) << 21;
  emit32(kAL | 0x0C800B00 | w | rn.base().code << 16 |  encode(regs.start, 22, 12) | regs.length << 1);
}

void Assembler::vstr(SRegister rn, MemOperand op) {
  const uint32_t offset = std::abs(op.offset());
  if (op.mode() != AddressingMode::kOffset || offset > kUint10Max || offset % 4 != 0) {
    error_ = Error::kInvalidOperand;
    return;
  }
  emit32(kAL | 0x0D000A00 | op.u() << 23 | op.base().code << 16 | encode(rn, 22, 12) | offset >> 2);
}

void Assembler::align(uint8_t n) {
  if (!is_po2(n) || (n % kInstructionSizeInBytes != 0)) {
    error_ = Error::kInvalidOperand;
    return;
  }

  uintptr_t cursor = reinterpret_cast<uintptr_t>(cursor_);
  const uintptr_t target = round_up_po2(cursor, n);
  while (cursor < target) {
    nop();
    cursor += kInstructionSizeInBytes;
  }
}

void MacroAssembler::f32_hardswish(QRegister sixth, QRegister three,
                                   QRegister six, QRegister zero,
                                   const QRegister* accs, size_t num_accs,
                                   const QRegister* tmps, size_t num_tmps) {
  if (num_accs < 4) {
    assert(num_tmps >= num_accs);
    for (size_t i = 0; i < num_accs; i++) {
      vmul_f32(tmps[i], accs[i], sixth.low()[0]);
    }
    for (size_t i = 0; i < num_accs; i++) {
      vadd_f32(accs[i], accs[i], three);
    }
    for (size_t i = 0; i < num_accs; i++) {
      vmax_f32(accs[i], accs[i], zero);
    }
    for (size_t i = 0; i < num_accs; i++) {
      vmin_f32(accs[i], accs[i], six);
    }
    for (size_t i = 0; i < num_accs; i++) {
      vmul_f32(accs[i], accs[i], tmps[i]);
    }
    return;
  }

  assert(num_accs >= 4);
  assert(num_accs % 4 == 0);
  assert(num_tmps == 4);

  for (size_t i = 0; i < num_accs; i+= 4) {
    const auto acc0 = accs[i];
    const auto acc1 = accs[i+1];
    const auto acc2 = accs[i+2];
    const auto acc3 = accs[i+3];
    vmul_f32(tmps[0], acc0, sixth.low()[0]);
    vmul_f32(tmps[1], acc1, sixth.low()[0]);
    vmul_f32(tmps[2], acc2, sixth.low()[0]);
    vmul_f32(tmps[3], acc3, sixth.low()[0]);
    vadd_f32(acc0, acc0, three);
    vadd_f32(acc1, acc1, three);
    vadd_f32(acc2, acc2, three);
    vadd_f32(acc3, acc3, three);
    vmax_f32(acc0, acc0, zero);
    vmax_f32(acc1, acc1, zero);
    vmax_f32(acc2, acc2, zero);
    vmax_f32(acc3, acc3, zero);
    vmin_f32(acc0, acc0, six);
    vmin_f32(acc1, acc1, six);
    vmin_f32(acc2, acc2, six);
    vmin_f32(acc3, acc3, six);
    vmul_f32(acc0, acc0, tmps[0]);
    vmul_f32(acc1, acc1, tmps[1]);
    vmul_f32(acc2, acc2, tmps[2]);
    vmul_f32(acc3, acc3, tmps[3]);
  }
}


}  // namespace aarch32
}  // namespace xnnpack
