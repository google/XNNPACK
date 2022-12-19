// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>

#include <xnnpack/aarch64-assembler.h>
#include <xnnpack/common.h>
#include <xnnpack/math.h>


namespace xnnpack {
namespace aarch64 {
// Min and max values for the imm7 for ldp, will be shifted right by 3 when encoding.
constexpr int32_t kImm7Min = -512;
constexpr int32_t kImm7Max = 504;
constexpr uint32_t kImm7Mask = 0x7F;
// Max value for imm12, will be shifted right by 3 when encoding.
constexpr int32_t kImm12Max = 32760;
constexpr uint32_t kUint12Max = 4095;

constexpr int32_t kInt9Max = 255;
constexpr int32_t kInt9Min = -256;
constexpr uint32_t kImm9Mask = 0x1FF;

// Constants used for checking branch offset bounds.
// Conditional bounds are +/-1MB.
constexpr ptrdiff_t kConditionalBranchImmMax = 1048572;
constexpr ptrdiff_t kConditionalBranchImmMin = -1048576;
// TBZ and TBNZ bounds are +/-32KB.
constexpr ptrdiff_t kTbxzImmMax = 32764;
constexpr ptrdiff_t kTbxzImmMin = -32768;
// Unconditional bounds are +/-128MB.
constexpr ptrdiff_t kUnconditionalBranchImmMax = 134217727;
constexpr ptrdiff_t kUnconditionalBranchImmMin = -134217728;

constexpr uint32_t kConditionalImmMask = 0x0007FFFF;
constexpr uint32_t kTbxzImmMask = 0x3FFF;
constexpr uint32_t kUnconditionalImmMask = 0x03FFFFFF;

template <typename Reg> inline uint32_t rd(Reg rn) { return rn.code; }
template <typename Reg> inline uint32_t rt(Reg rn) { return rn.code; }
template <typename Reg> inline uint32_t rt2(Reg rn) { return rn.code << 10; }
template <typename Reg> inline uint32_t rm(Reg rn) { return rn.code << 16; }
template <typename Reg> inline uint32_t rn(Reg rn) { return rn.code << 5; }
inline uint32_t q(VRegister vt) { return vt.q << 30; }
inline uint32_t size(VRegister vt) { return vt.size << 10; }
inline uint32_t fp_sz(VRegister vn) { return vn.is_s() ? 0 : 1 << 22; }
inline uint32_t postindex(MemOperand op) { return (op.mode == AddressingMode::kPostIndex) ? 0 : 1 << 24; }
inline uint32_t wb(MemOperand op) { return op.mode == AddressingMode::kOffset ? 0 : 1 << 23; }
// Used for ld1/st1 multiple structures.
inline uint32_t l(bool load) { return load ? 1 << 22 : 0; }

inline uint32_t imm9(int32_t imm) {
  assert(!(imm < kInt9Min || imm > kInt9Max));
  return (imm & kImm9Mask) << 12;
}

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
inline bool branch_offset_valid(ptrdiff_t offset, BranchType branch_type) {
  switch (branch_type) {
    case BranchType::kConditional:
      return offset < kConditionalBranchImmMax && offset > kConditionalBranchImmMin;
    case BranchType::kTbxz:
      return offset < kTbxzImmMax && offset > kTbxzImmMin;
    case BranchType::kUnconditional:
      return offset < kUnconditionalBranchImmMax && offset > kUnconditionalBranchImmMin;
    default:
      XNN_UNREACHABLE;
  }
  return false;
}

inline BranchType instruction_branch_type(uint32_t instr) {
  const uint32_t masked = instr & 0xFE000000;
  switch (masked) {
    case 0xB6000000:
    case 0x36000000:
      return BranchType::kTbxz;
    case 0x54000000:
      return BranchType::kConditional;
    case 0x14000000:
    case 0x16000000:
      return BranchType::kUnconditional;
    default:
      XNN_UNREACHABLE;
  }
}

inline uint32_t mask(BranchType branch_type) {
  switch (branch_type) {
    case BranchType::kConditional:
      return kConditionalImmMask;
    case BranchType::kTbxz:
      return kTbxzImmMask;
    case BranchType::kUnconditional:
      return kUnconditionalImmMask;
    default:
      XNN_UNREACHABLE;
  }
}

inline uint8_t shift(BranchType branch_type) {
  switch (branch_type) {
    case BranchType::kConditional:
      return 5;
    case BranchType::kTbxz:
      return 5;
    case BranchType::kUnconditional:
      return 0;
    default:
      XNN_UNREACHABLE;
  }
}

inline uint32_t branch_imm(ptrdiff_t offset, BranchType bt) {
  return ((offset >> kInstructionSizeInBytesLog2) & mask(bt)) << shift(bt);
}

inline uint32_t hl(VRegisterLane vl) {
  if (vl.is_s()) {
    return (vl.lane & 1) << 21 | ((vl.lane & 2) << 10);
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

inline uint8_t load_store_opcode(uint8_t register_length) {
  switch (register_length) {
    case 1:
      return 0x7;
    case 2:
      return 0xA;
    case 3:
      return 0x6;
    case 4:
      return 0x2;
    default:
      XNN_UNREACHABLE;
  }
}

inline bool imm7_offset_valid(int32_t imm, XRegister) {
  return imm >= kImm7Min && imm <= kImm7Max && (imm & 0x7) == 0;
}

inline bool imm7_offset_valid(int32_t imm, DRegister) {
  return imm >= kImm7Min && imm <= kImm7Max && (imm & 0x7) == 0;
}

inline bool imm7_offset_valid(int32_t imm, QRegister) {
  return imm >= (kImm7Min * 2) && imm <= (kImm7Max * 2) && (imm & 0xF) == 0;
}

// Base instructions.

void Assembler::add(XRegister xd, XRegister xn, uint16_t imm12) {
  // The instruction supports larger numbers using the shift by (left shift by 12), but that's unused in kernels.
  if (imm12 > kUint12Max) {
    error_ = Error::kInvalidOperand;
    return;
  }

  emit32(0x91000000 | imm12 << 10 | rn(xn) | rd(xd));
}

void Assembler::add(XRegister xd, XRegister xn, XRegister xm) {
  emit32(0x8B000000 | rd(xd) | rn(xn) | rm(xm));
}

void Assembler::b(Label& l) {
  return branch_to_label(0x14000000, BranchType::kUnconditional, l);
}

void Assembler::cmp(XRegister xn, uint16_t imm12) {
  if (imm12 > kUint12Max) {
    error_ = Error::kInvalidOperand;
    return;
  }
  emit32(0xF100001F | imm12 << 10 | rn(xn));
}

void Assembler::cmp(XRegister xn, XRegister xm) {
  emit32(0xEB00001F | rm(xm) | rn(xn));
}

void Assembler::csel(XRegister xd, XRegister xn, XRegister xm, Condition c) {
  emit32(0x9A800000 | rm(xm) | c << 12 | rn(xn) | rd(xd));
}

void Assembler::hlt() {
  emit32(0xD4400000);
}

void Assembler::ldp(XRegister xt1, XRegister xt2, MemOperand xn) {
  if (!imm7_offset_valid(xn.offset, xt1)) {
    error_ = Error::kInvalidOperand;
    return;
  }

  const uint32_t offset = (xn.offset >> 3) & kImm7Mask;

  emit32(0xA8400000 | postindex(xn) | wb(xn) | offset << 15 | rt2(xt2) | rn(xn.base) | xt1.code);
}

void Assembler::ldp(XRegister xt1, XRegister xt2, MemOperand xn, int32_t imm) {
  if (xn.offset != 0) {
    error_ = Error::kInvalidOperand;
    return;
  }
  return ldp(xt1, xt2, {xn.base, imm, AddressingMode::kPostIndex});
}

void Assembler::ldr(XRegister xt, MemOperand xn) {
  const int32_t imm = xn.offset;
  if (xn.mode != AddressingMode::kOffset || imm < 0 || imm > (kUint12Max << 3) || (imm & 7) != 0) {
    error_ = Error::kInvalidOperand;
    return;
  }

  emit32(0xF9400000 | imm >> 3 << 10 | rn(xn.base) | xt.code);
}

void Assembler::ldr(XRegister xt, MemOperand xn, int32_t imm) {
  if (imm < kInt9Min || imm > kInt9Max) {
    error_ = Error::kInvalidOperand;
    return;
  }

  emit32(0xF8400400 | imm9(imm) | rn(xn.base) | rt(xt));
}

void Assembler::mov(XRegister xd, XRegister xn) {
  emit32(0xAA0003E0 | rm(xn) | rd(xd));
}

void Assembler::nop() {
  emit32(0xD503201F);
}

void Assembler::prfm(PrefetchOp prfop, MemOperand xn) {
  if (xn.offset < 0 || xn.offset > kImm12Max) {
    error_ = Error::kInvalidOperand;
    return;
  }

  emit32(0xF9800000 | xn.offset >> 3 << 10 | rn(xn.base) | prfop);
}

void Assembler::ret() {
  emit32(0xD65F0000 | rn(x30));
}

void Assembler::stp(XRegister xt1, XRegister xt2, MemOperand xn) {
  if (!imm7_offset_valid(xn.offset, xt1)) {
    error_ = Error::kInvalidOperand;
    return;
  }

  const uint32_t offset = (xn.offset >> 3) & kImm7Mask;
  emit32(0xA9000000 | wb(xn) | offset << 15 | rt2(xt2) | rn(xn.base) | rt(xt1));
}

void Assembler::str(XRegister xt1, MemOperand xn) {
  const int32_t offset = xn.offset;
  if (xn.mode == AddressingMode::kPreIndex) {
    if (offset < kInt9Min || offset > kInt9Max) {
      error_ = Error::kInvalidOperand;
      return;
    }
    emit32(0xF8000C00 | imm9(offset) | rn(xn.base) | rt(xt1));
  } else if (xn.mode == AddressingMode::kOffset) {
    if (offset < 0 || offset > kImm12Max || offset % 8 != 0) {
      error_ = Error::kInvalidOperand;
      return;
    }
    emit32(0xF9000000 | offset >> 3 << 10 | rn(xn.base) | rt(xt1));
  } else {
    XNN_UNREACHABLE;
  }
}

void Assembler::sub(XRegister xd, XRegister xn, XRegister xm) {
  emit32(0xCB000000 | rm(xm) | rn(xn) | rd(xd));
}

void Assembler::subs(XRegister xd, XRegister xn, uint16_t imm12) {
  if (imm12 > kUint12Max) {
    error_ = Error::kInvalidOperand;
    return;
  }

  emit32(0xF1000000 | imm12 << 10 | rn(xn) | rd(xd));
}

void Assembler::tbnz(XRegister xd, uint8_t bit, Label& l) {
  return tb_helper(0x37000000, xd, bit, l);
}

void Assembler::tbz(XRegister xd, uint8_t bit, Label& l) {
  return tb_helper(0x36000000, xd, bit, l);
}

void Assembler::tst(XRegister xn, uint8_t imm) {
  // Encoding of immediate is quite complicated, we only support po2-1, which is what assembly microkernel uses.
  uint32_t imm_po2 = imm + 1;
  if (!is_po2(imm_po2)) {
    error_ = Error::kUnimplemented;
    return;
  }

  const uint32_t imm_s = (math_ctz_u32(imm_po2) - 1) << 10;
  emit32(0xF240001F | imm_s | rn(xn));
}

// SIMD instructions.

void Assembler::dup(DRegister dd, VRegisterLane vn) {
  if (vn.size != 3 || vn.lane > 1) {
    error_ = Error::kInvalidOperand;
    return;
  }
  const uint8_t imm5 = 0b1000 | (vn.lane & 1) << 4;
  emit32(0x5E000400 | imm5 << 16 | rn(vn) | rd(dd));
}

void Assembler::fabs(VRegister vd, VRegister vn) {
  if (!is_same_shape(vd, vn)) {
    error_ = Error::kInvalidOperand;
    return;
  }

  emit32(0x0EA0F800 | q(vd) | fp_sz(vn) | rn(vn) | rd(vd));
}

void Assembler::fadd(VRegister vd, VRegister vn, VRegister vm) {
  if (!is_same_shape(vd, vn, vm)) {
    error_ = Error::kInvalidOperand;
    return;
  }

  emit32(0x0E20D400 | q(vd) | fp_sz(vn) | rm(vm) | rn(vn) | rd(vd));
}

void Assembler::fmax(VRegister vd, VRegister vn, VRegister vm) {
  if (!is_same_shape(vd, vn, vm)) {
    error_ = Error::kInvalidOperand;
    return;
  }

  emit32(0x0E20F400 | q(vd) | fp_sz(vn) | rm(vm) | rn(vn) | rd(vd));
}

void Assembler::fmin(VRegister vd, VRegister vn, VRegister vm) {
  if (!is_same_shape(vd, vn, vm)) {
    error_ = Error::kInvalidOperand;
    return;
  }

  emit32(0x0EA0F400 | q(vd) | fp_sz(vn) | rm(vm) | rn(vn) | rd(vd));
}

void Assembler::fmla(VRegister vd, VRegister vn, VRegisterLane vm) {
  if (!is_same_shape(vd, vn) || !is_same_data_type(vd, vm)) {
    error_ = Error::kInvalidOperand;
    return;
  }
  if (!lane_index_valid(vd.q, vm.size, vm.lane)) {
    error_ = Error::kInvalidLaneIndex;
    return;
  }

  emit32(0x0F801000 | q(vd) | fp_sz(vd) | hl(vm) | rm(vm) | rn(vn) | rd(vd));
}

void Assembler::fmul(VRegister vd, VRegister vn, VRegister vm) {
  if (!is_same_shape(vd, vn, vm)) {
    error_ = Error::kInvalidOperand;
    return;
  }

  emit32(0x2E20DC00 | q(vd) | fp_sz(vn) | rm(vm) | rn(vn) | rd(vd));
}

void Assembler::fneg(VRegister vd, VRegister vn) {
  if (!is_same_shape(vd, vn)) {
    error_ = Error::kInvalidOperand;
    return;
  }

  emit32(0x2EA0F800 | q(vd) | fp_sz(vn) | rn(vn) | rd(vd));
}

void Assembler::ins(VRegisterLane vd, XRegister vn) {
  size_t shift = vd.size;
  size_t imm5 = (vd.lane << (shift + 1)) | (1 << shift);

  emit32(0x4E001C00 | imm5 << 16 | rn(vn) | rd(vd));
}

void Assembler::ld1(ScalarVRegisterList vs, size_t lane, MemOperand xn, int32_t imm) {
  emit32(0x4DC08000 | 0b11111 << 16 | (vs.vt1.size & 1) << 10 | rn(xn.base) | rt(vs.vt1));
}

void Assembler::ld1(ScalarVRegister vs, size_t lane, MemOperand xn, int32_t imm) {
  ld1(ScalarVRegisterList{vs}, lane, xn, imm);
}

void Assembler::ld1(VRegisterList vs, MemOperand xn, int32_t imm) {
  ld1_st1_multiple_structures(vs, xn, imm, true);
}

void Assembler::ld1r(VRegisterList xs, MemOperand xn) {
  if (xs.length != 1 || xn.offset != 0) {
    error_ = Error::kInvalidOperand;
    return;
  }

  emit32(0x0D40C000 | q(xs.vt1) | size(xs.vt1) | rn(xn.base) | xs.vt1.code);
}

void Assembler::ld2r(VRegisterList xs, MemOperand xn) {
  if (xs.length != 2 || !is_same_shape(xs.vt1, xs.vt2) || xn.offset != 0 || !is_consecutive(xs)) {
    error_ = Error::kInvalidOperand;
    return;
  }

  emit32(0x0D60C000 | q(xs.vt1) | size(xs.vt1) | rn(xn.base) | xs.vt1.code);
}

void Assembler::ld3r(VRegisterList xs, MemOperand xn) {
  if (xs.length != 3 || !is_same_shape(xs.vt1, xs.vt2, xs.vt3) || xn.offset != 0 || !is_consecutive(xs)) {
    error_ = Error::kInvalidOperand;
    return;
  }

  emit32(0x0D40E000 | q(xs.vt1) | size(xs.vt1) | rn(xn.base) | xs.vt1.code);
}

void Assembler::ldp(DRegister dt1, DRegister dt2, MemOperand xn) {
  if (!imm7_offset_valid(xn.offset, dt1)) {
    error_ = Error::kInvalidOperand;
    return;
  }

  const uint32_t offset = (xn.offset >> 3) & kImm7Mask;
  emit32(0x6C400000 | postindex(xn) | wb(xn) | offset << 15 | rt2(dt2) | rn(xn.base) | rt(dt1));
}

void Assembler::ldp(DRegister dt1, DRegister dt2, MemOperand xn, int32_t imm) {
  return ldp(dt1, dt2, {xn.base, imm, AddressingMode::kPostIndex});
}

void Assembler::ldp(QRegister qt1, QRegister qt2, MemOperand xn, int32_t imm) {
  if (!imm7_offset_valid(imm, qt1)) {
    error_ = Error::kInvalidOperand;
    return;
  }
  const uint32_t offset = (imm >> 4) & kImm7Mask;

  emit32(0xACC00000 | offset << 15 | rt2(qt2) | rn(xn.base) | qt1.code);
}

void Assembler::ldr(DRegister dt, MemOperand xn) {
  if (xn.offset != 0 && (xn.offset < 0 || xn.offset > 32760)) {
    error_ = Error::kInvalidOperand;
    return;
  }
  size_t size = 3;

  emit32(0x3D400000 | size << 30 | 1 << 22 | (xn.offset >> size) << 10 | rn(xn.base) | rt(dt));
}

void Assembler::ldr(DRegister dt, MemOperand xn, int32_t imm) {
  return ldr(/*size=*/3, /*opc=*/1, xn, imm, dt.code);
}

void Assembler::ldr(QRegister qt, MemOperand xn, int32_t imm) {
  return ldr(/*size=*/0, /*opc=*/3, xn, imm, qt.code);
}

void Assembler::ldr(SRegister st, MemOperand xn, int32_t imm) {
  return ldr(/*size=*/2, /*opc=*/1, xn, imm, st.code);
}

void Assembler::mov(VRegister vd, VRegister vn) {
  if (!is_same_shape(vd, vn)) {
    error_ = Error::kInvalidOperand;
    return;
  }
  emit32(0x0EA01C00 | q(vd) | rm(vn) | rn(vn) | rd(vd));
}

void Assembler::movi(VRegister vd, uint8_t imm) {
  if (imm != 0) {
    error_ = Error::kUnimplemented;
    return;
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
      return;
  }

  emit32(0x0F000400 | q(vd) | cmode << 12 | vd.code);
}

void Assembler::st1(VRegisterList vs, MemOperand xn, int32_t imm) {
  ld1_st1_multiple_structures(vs, xn, imm, false);
}

void Assembler::st1(VRegisterList vs, MemOperand xn, XRegister xm) {
  if (!is_same_shape(vs) || !is_consecutive(vs)) {
    error_ = Error::kInvalidOperand;
    return;
  }

  VRegister vt = vs.vt1;

  const uint8_t opcode = load_store_opcode(vs.length);
  emit32(0x0C800000 | q(vt) | rm(xm) | opcode << 12 | size(vt) | rn(xn.base) | rt(vt));
}

void Assembler::stp(DRegister dt1, DRegister dt2, MemOperand xn) {
  if (!imm7_offset_valid(xn.offset, dt1)) {
    error_ = Error::kInvalidOperand;
    return;
  }

  const uint32_t offset = (xn.offset >> 3) & kImm7Mask;
  emit32(0x6D000000 | wb(xn) | offset << 15 | rt2(dt2) | rn(xn.base) | rt(dt1));
}

void Assembler::stp(QRegister qt1, QRegister qt2, MemOperand xn) {
  if (!imm7_offset_valid(xn.offset, qt1)) {
    error_ = Error::kInvalidOperand;
    return;
  }

  const uint32_t offset = (xn.offset >> 4) & kImm7Mask;
  emit32(0xAD000000 | wb(xn) | offset << 15 | rt2(qt2) | rn(xn.base) | rt(qt1));
}

void Assembler::stp(QRegister qt1, QRegister qt2, MemOperand xn, int32_t imm) {
  if (!imm7_offset_valid(imm, qt1)) {
    error_ = Error::kInvalidOperand;
    return;
  }

  const uint32_t offset = (imm >> 4) & kImm7Mask;
  emit32(0xAC800000 | offset << 15 | rt2(qt2) | rn(xn.base) | rt(qt1));
}

void Assembler::str(DRegister dt, MemOperand xn, int32_t imm) {
  return str(/*size=*/3, /*opc=*/0, xn, imm, dt.code);
}

void Assembler::str(QRegister qt, MemOperand xn, int32_t imm) {
  return str(/*size=*/0, /*opc=*/2, xn, imm, qt.code);
}

void Assembler::str(SRegister st, MemOperand xn) {
  const int32_t imm = xn.offset;
  if (imm < 0 || imm > (kUint12Max << 2) || (imm & 0x3) != 0) {
    error_ = Error::kInvalidOperand;
    return;
  }

  emit32(0xBD000000 | imm >> 2 << 10 | rn(xn.base) | rt(st));
}

void Assembler::str(SRegister st, MemOperand xn, int32_t imm) {
  return str(/*size=*/2, /*opc=*/0, xn, imm, st.code);
}

void Assembler::align(uint8_t n, AlignInstruction instr) {
  if (!is_po2(n) || (n % kInstructionSizeInBytes != 0)) {
    error_ = Error::kInvalidOperand;
    return;
  }

  uintptr_t cursor = reinterpret_cast<uintptr_t>(cursor_);
  const uintptr_t target = round_up_po2(cursor, n);
  while (cursor < target) {
    switch (instr) {
      case AlignInstruction::kHlt:
        hlt();
        break;
      case AlignInstruction::kNop:
        nop();
        break;
      default:
        XNN_UNREACHABLE;
    }
    cursor += kInstructionSizeInBytes;
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
    const ptrdiff_t offset = l.offset - user;
    uint32_t* instr = reinterpret_cast<uint32_t*>(user);

    const BranchType bt = instruction_branch_type(*instr);
    if (!branch_offset_valid(offset, bt)) {
      error_ = Error::kLabelOffsetOutOfBounds;
      return;
    }

    *instr |= branch_imm(offset, bt);
  }
}

void Assembler::b(Condition c, Label& l) {
  return branch_to_label(0x54000000 | c, BranchType::kConditional, l);
}

void Assembler::branch_to_label(uint32_t opcode, BranchType bt, Label& l) {
  if (l.bound) {
    const ptrdiff_t offset = l.offset - cursor_;
    if (!branch_offset_valid(offset, bt)) {
      error_ = Error::kLabelOffsetOutOfBounds;
      return;
    }
    emit32(opcode | branch_imm(offset, bt));
  } else {
    if (!l.add_use(cursor_)) {
      error_ = Error::kLabelHasTooManyUsers;
      return;
    }
    emit32(opcode);
  }
}

void Assembler::ld1_st1_multiple_structures(VRegisterList vs, MemOperand xn, int32_t imm, bool load) {
  const VRegister vt = vs.vt1;

  if (!is_same_shape(vs) || !is_consecutive(vs)) {
    error_ = Error::kInvalidOperand;
    return;
  }

  // imm must match number of bytes loaded.
  if ((vt.q + 1) * 8 * vs.length != imm) {
    error_ = Error::kInvalidOperand;
    return;
  }

  const uint8_t opcode = load_store_opcode(vs.length);

  emit32(0x0C800000 | q(vt) | l(load) | rm(sp) | opcode << 12 | size(vt) | rn(xn.base) | rt(vt));
}

void Assembler::ldr(uint32_t size, uint32_t opc, MemOperand xn, int32_t imm, uint8_t rt_code) {
  if (xn.mode != AddressingMode::kOffset || xn.offset != 0 || imm < kInt9Min || imm > kInt9Max) {
    error_ = Error::kInvalidOperand;
    return;
  }

  emit32(0x3C400400 | size << 30 | opc << 22 | imm9(imm) | rn(xn.base) | rt_code);
}

void Assembler::str(uint32_t size, uint32_t opc, MemOperand xn, int32_t imm, uint8_t rt_code) {
  if (imm < kInt9Min || imm > kInt9Max) {
    error_ = Error::kInvalidOperand;
    return;
  }

  emit32(0x3C000400 | size << 30 | opc << 22 | imm9(imm) | rn(xn.base) | rt_code);
}

void Assembler::tb_helper(uint32_t op, XRegister xd, uint8_t bit, Label& l) {
  if (bit > 63) {
    error_ = Error::kInvalidOperand;
    return;
  }

  const uint32_t bit_pos = (bit & 0x20) >> 5 << 31 | (bit & 0x1F) << 19;
  return branch_to_label(op | bit_pos | xd.code, BranchType::kTbxz, l);
}

void MacroAssembler::f32_hardswish(VRegister sixth, VRegister three,
                                   VRegister six, VRegister zero,
                                   const VRegister* accs, size_t num_accs,
                                   const VRegister* tmps, size_t num_tmps) {
  if (num_accs < 4) {
    assert(num_tmps >= num_accs);
    for (size_t i = 0; i < num_accs; i++) {
      fmul(tmps[i], accs[i], sixth);
    }
    for (size_t i = 0; i < num_accs; i++) {
      fadd(accs[i], accs[i], three);
    }
    for (size_t i = 0; i < num_accs; i++) {
      fmax(accs[i], accs[i], zero);
    }
    for (size_t i = 0; i < num_accs; i++) {
      fmin(accs[i], accs[i], six);
    }
    for (size_t i = 0; i < num_accs; i++) {
      fmul(accs[i], accs[i], tmps[i]);
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
    fmul(tmps[0], acc0, sixth);
    fmul(tmps[1], acc1, sixth);
    fmul(tmps[2], acc2, sixth);
    fmul(tmps[3], acc3, sixth);
    fadd(acc0, acc0, three);
    fadd(acc1, acc1, three);
    fadd(acc2, acc2, three);
    fadd(acc3, acc3, three);
    fmax(acc0, acc0, zero);
    fmax(acc1, acc1, zero);
    fmax(acc2, acc2, zero);
    fmax(acc3, acc3, zero);
    fmin(acc0, acc0, six);
    fmin(acc1, acc1, six);
    fmin(acc2, acc2, six);
    fmin(acc3, acc3, six);
    fmul(acc0, acc0, tmps[0]);
    fmul(acc1, acc1, tmps[1]);
    fmul(acc2, acc2, tmps[2]);
    fmul(acc3, acc3, tmps[3]);
  }
}

}  // namespace aarch64
}  // namespace xnnpack
