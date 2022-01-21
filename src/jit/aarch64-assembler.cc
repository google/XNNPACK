// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/common.h>
#include <xnnpack/aarch64-assembler.h>

#include <cmath>

namespace xnnpack {
namespace aarch64 {
// Min and max values for the imm7 for ldp, will be shifted right by 3 when encoding.
constexpr int32_t kImm7Min = -512;
constexpr int32_t kImm7Max = 504;
// Max value for imm12, will be shifted right by 3 when encoding.
constexpr int32_t kImm12Max = 32760;
constexpr uint32_t kUint12Max = 4095;

constexpr ptrdiff_t kInt9Max = 255;
constexpr ptrdiff_t kInt9Min = -256;

// Constants used for checking branch offset bounds.
constexpr ptrdiff_t kConditionalBranchImmMax = 1048572;
constexpr ptrdiff_t kConditionalBranchImmMin = -1048576;

inline uint32_t rn(XRegister xn) { return xn.code << 5; }
inline uint32_t rn(VRegister vn) { return vn.code << 5; }
inline uint32_t q(VRegister vt) { return vt.q << 30; }
inline uint32_t size(VRegister vt) { return vt.size << 10; }

inline bool is_same_shape(VRegister vt1, VRegister vt2) {
  return vt1.size == vt2.size && vt1.q == vt2.q;
}

template <typename Reg, typename... Regs>
inline bool is_same_shape(Reg reg1, Reg reg2, Regs... regs) {
  return is_same_shape(reg1, reg2) && is_same_shape(reg2, regs...);
}

inline bool is_same_shape(VRegisterList vs) {
  switch (vs.length) {
    case 1:
      return true;
    case 2:
      return is_same_shape(vs.vt1, vs.vt2);
    case 3:
      return is_same_shape(vs.vt1, vs.vt2, vs.vt3);
    case 4:
      return is_same_shape(vs.vt1, vs.vt2, vs.vt3, vs.vt4);
    default:
      XNN_UNREACHABLE;
  }
}

inline bool is_same_data_type(VRegister vt1, VRegisterLane vt2) {
  return vt1.size == vt2.size;
}

inline bool is_consecutive(VRegister vt1, VRegister vt2) {
  return (vt1.code + 1) % 32 == vt2.code;
}

template <typename Reg, typename... Regs>
inline bool is_consecutive(Reg reg1, Reg reg2, Regs... regs) {
  return is_consecutive(reg1, reg2) && is_consecutive(reg2, regs...);
}

inline bool is_consecutive(VRegisterList vs) {
  switch (vs.length) {
    case 1:
      return true;
    case 2:
      return is_consecutive(vs.vt1, vs.vt2);
    case 3:
      return is_consecutive(vs.vt1, vs.vt2, vs.vt3);
    case 4:
      return is_consecutive(vs.vt1, vs.vt2, vs.vt3, vs.vt4);
    default:
      XNN_UNREACHABLE;
  }
}

// Check if a branch offset is valid, it must fit in 19 bits.
bool branch_offset_valid(ptrdiff_t offset) {
  return offset < kConditionalBranchImmMax && offset > kConditionalBranchImmMin;
}

inline uint32_t encode_hl(VRegisterLane vl) {
  if (vl.is_s()) {
    return (vl.lane & 1) << 21 | ((vl.lane & 2) >> 1 << 11);
  } else {
    return (vl.lane & 1) << 11;
  }
}

inline bool lane_index_valid(uint8_t q, uint8_t size, uint8_t lane) {
  // The logic here is something like:
  // if (q && size == 0) {
  //   return lane < 16;
  // } else if (q && size == 1) {
  //   return lane < 8;
  // } else if (q && size == 2) {
  //   return lane < 4;
  // } else if (q && size == 3) {
  //   return lane < 2;
  // }
  // then repeat for !q with maximum lane size halved.
  // translated into this formula.
  return lane < ((q + 1) << (3 - size));
}

// Base instructions.

Assembler& Assembler::ldp(XRegister xt1, XRegister xt2, MemOperand xn) {
  if (xn.offset < kImm7Min || xn.offset > kImm7Max || std::abs(xn.offset) % 8 != 0) {
    error_ = Error::kInvalidOperand;
    return *this;
  }

  const uint32_t mode = xn.mode == AddressingMode::kOffset ? 2 : 1;
  const uint32_t offset = (xn.offset >> 3) & 0x7F;

  return emit32(0xA8400000 | mode << 23 | offset << 15 | xt2.code << 10 | rn(xn.base) | xt1.code);
}

Assembler& Assembler::ldp(XRegister xt1, XRegister xt2, MemOperand xn, int32_t imm) {
  if (xn.offset != 0) {
    error_ = Error::kInvalidOperand;
    return *this;
  }
  return ldp(xt1, xt2, {xn.base, imm, AddressingMode::kPostIndex});
}

Assembler& Assembler::ldr(XRegister xt, MemOperand xn) {
  if (xn.mode != AddressingMode::kOffset || xn.offset < 0 || xn.offset > kImm12Max || xn.offset % 8 != 0) {
    error_ = Error::kInvalidOperand;
    return *this;
  }

  return emit32(0xF9400000 | xn.offset >> 3 << 10 | rn(xn.base) | xt.code);
}

Assembler& Assembler::prfm(PrefetchOp prfop, MemOperand xn) {
  if (xn.offset < 0 || xn.offset > kImm12Max) {
    error_ = Error::kInvalidOperand;
    return *this;
  }

  return emit32(0xF9800000 | xn.offset << 10 | rn(xn.base) | static_cast<uint32_t>(prfop));
}

Assembler& Assembler::subs(XRegister xd, XRegister xn, uint16_t imm12) {
  if (imm12 > kUint12Max) {
    error_ = Error::kInvalidOperand;
    return *this;
  }

  return emit32(0xF1000000 | imm12 << 10 | rn(xn) | xd.code);
}

// SIMD instructions.

Assembler& Assembler::fmla(VRegister vd, VRegister vn, VRegisterLane vm) {
  if (!is_same_shape(vd, vn) || !is_same_data_type(vd, vm)) {
    error_ = Error::kInvalidOperand;
    return *this;
  }
  if (!lane_index_valid(vd.q, vm.size, vm.lane)) {
    error_ = Error::kInvalidLaneIndex;
    return *this;
  }

  uint32_t sz = vm.is_s() ? 0 : 1;
  return emit32(0x0F801000 | q(vd) | sz << 22 | encode_hl(vm) | vm.code << 16 | rn(vn) | vd.code);
}

Assembler& Assembler::ld1(VRegisterList vs, MemOperand xn, int32_t imm) {
  VRegister vt = vs.vt1;

  if (!is_same_shape(vs) || !is_consecutive(vs)) {
    error_ = Error::kInvalidOperand;
    return *this;
  }

  // imm must match number of bytes loaded.
  if ((vt.q + 1) * 8 * vs.length != imm) {
    error_ = Error::kInvalidOperand;
    return *this;
  }

  uint8_t opcode = 0;
  switch (vs.length) {
    case 1:
      opcode = 0x7;
      break;
    case 2:
      opcode = 0xA;
      break;
    case 3:
      opcode = 0x6;
      break;
    case 4:
      opcode = 0x2;
      break;
    default:
      XNN_UNREACHABLE;
  }

  return emit32(0x0CDF0000 | q(vt) | opcode << 12 | size(vt) | rn(xn.base) | vt.code);
}

Assembler& Assembler::ld2r(VRegisterList xs, MemOperand xn) {
  if (xs.length != 2 || !is_same_shape(xs.vt1, xs.vt2) || xn.offset != 0) {
    error_ = Error::kInvalidOperand;
    return *this;
  }

  return emit32(0x0D60C000 | q(xs.vt1) | size(xs.vt1) | rn(xn.base) | xs.vt1.code);
}

Assembler& Assembler::ldp(QRegister qt1, QRegister qt2, MemOperand xn, int32_t imm) {
  if (imm < -1024 || imm > 1008 || (imm & 0xF) != 0) {
    error_ = Error::kInvalidOperand;
    return *this;
  }
  const uint32_t offset = (imm >> 4) & 0x7F;

  return emit32(0xACC00000 | offset << 15 | qt2.code << 10 | rn(xn.base) | qt1.code);
}

Assembler& Assembler::ldr(QRegister qt, MemOperand xn, int32_t imm) {
  if (xn.mode != AddressingMode::kOffset || xn.offset != 0 || imm < kInt9Min || imm > kInt9Max) {
    error_ = Error::kInvalidOperand;
    return *this;
  }

  return emit32(0x3CC00400 | (imm & 0x1FF) << 12| rn(xn.base) | qt.code);
}

Assembler& Assembler::movi(VRegister vd, uint8_t imm) {
  if (imm != 0) {
    error_ = Error::kUnimplemented;
    return *this;
  }

  uint32_t cmode = 0;
  switch (vd.size) {
    case 0:
      cmode = 0xE;
      break;
    case 1:
      cmode = 0x8;
      break;
    case 2:
      cmode = 0x0;
      break;
    default:
      error_ = Error::kUnimplemented;
      return *this;
  }

  return emit32(0x0F000400 | q(vd) | cmode << 12 | vd.code);
}

Assembler& Assembler::emit32(uint32_t value) {
  if (error_ != Error::kNoError) {
    return *this;
  }

  if (cursor_ + sizeof(value) > top_) {
    error_ = Error::kOutOfMemory;
    return *this;
  }

  memcpy(cursor_, &value, sizeof(value));
  cursor_ += sizeof(value);
  return *this;
}

Assembler& Assembler::bind(Label& l) {
  if (l.bound) {
    error_ = Error::kLabelAlreadyBound;
    return *this;
  }

  l.bound = true;
  l.offset = cursor_;

  // Patch all users.
  for (size_t i = 0; i < l.num_users; i++) {
    byte* user = l.users[i];
    const ptrdiff_t offset = l.offset - user;

    if (!branch_offset_valid(offset)) {
      error_ = Error::kLabelOffsetOutOfBounds;
      return *this;
    }

    *user = (*user | ((offset >> kInstructionSizeInBytesLog2) & 0x0007FFFF) << 5);
  }
  return *this;
}


Assembler& Assembler::b(Condition c, Label& l) {
  if (l.bound) {
    const ptrdiff_t offset = l.offset - cursor_;
    if (!branch_offset_valid(offset)) {
      error_ = Error::kLabelOffsetOutOfBounds;
      return *this;
    }
    // No need to shift by 2 since our offset is already in terms of uint32_t.
    return emit32(0x54000000 | ((offset >> kInstructionSizeInBytesLog2) & 0x0007FFFF) << 5 | c);
  } else {
    if (!l.add_use(cursor_)) {
      error_ = Error::kLabelHasTooManyUsers;
      return *this;
    }
    // Emit 0 offset first, will patch it up when label is bound later.
    return emit32(0x54000000 | c);
  }
  return *this;
}

}  // namespace aarch64
}  // namespace xnnpack
