#include "xnnpack/aarch32-assembler.h"

#include <cmath>

namespace xnnpack {
namespace aarch32 {
static const int DEFAULT_BUFFER_SIZE = 4096;

// PC register contains current address of instruction + 8 (2 instructions).
constexpr ptrdiff_t kPCDelta = 2;
// Constants used for checking branch offsets bounds.
constexpr ptrdiff_t kInt24Max = 8388607;
constexpr ptrdiff_t kInt24Min = -8388608;

// Check if a branch offset is valid, it must fit in 24 bits.
bool branch_offset_valid(ptrdiff_t offset) {
  return offset < kInt24Max && offset > kInt24Min;
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

Assembler::Assembler() {
  buffer_ = new uint32_t[DEFAULT_BUFFER_SIZE];
  cursor_ = buffer_;
  top_ = buffer_ + DEFAULT_BUFFER_SIZE;
  error_ = Error::kNoError;
}

Assembler::~Assembler() {
  delete[] buffer_;
}

Assembler& Assembler::emit32(uint32_t value) {
  if (error_ != Error::kNoError) {
    return *this;
  }

  if (cursor_ == top_) {
    error_ = Error::kOutOfMemory;
    return *this;
  }

  *cursor_++ = value;
  return *this;
}

Assembler& Assembler::add(CoreRegister rd, CoreRegister rn, CoreRegister rm) {
  return emit32(kAL | 0x8 << 20 | rn.code << 16 | rd.code << 12 | rm.code);
}

Assembler& Assembler::add(CoreRegister rd, CoreRegister rn, uint8_t imm) {
  // Rotation = 0, since imm is limited to 8 bits and fits in encoding.
  return emit32(kAL | 0x28 << 20 | rn.code << 16 | rd.code << 12 | imm);
}

Assembler& Assembler::b(Condition c, Label& l) {
  if (l.bound) {
    // Offset is relative to after this b instruction + kPCDelta.
    const ptrdiff_t offset = l.offset - cursor_ - kPCDelta;
    if (!branch_offset_valid(offset)) {
      error_ = Error::kLabelOffsetOutOfBounds;
      return *this;
    }

    // No need to shift by 2 since our offset is already in terms of uint32_t.
    return emit32(c | 0xA << 24 | (offset & 0x00FFFFFF));
  } else {
    if (!l.add_use(cursor_)) {
      error_ = Error::kLabelHasTooManyUsers;
      return *this;
    }
    // Emit 0 offset first, will patch it up when label is bound later.
    return emit32(c | 0xA << 24);
  }
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
    uint32_t* user = l.users[i];
    const ptrdiff_t offset = l.offset - user - kPCDelta;

    if (!branch_offset_valid(offset)) {
      error_ = Error::kLabelOffsetOutOfBounds;
      return *this;
    }

    *user = (*user | (offset & 0x00FFFFFF));
  }
  return *this;
}

Assembler& Assembler::cmp(CoreRegister rn, uint8_t imm) {
  return emit32(kAL | 0x35 << 20 | rn.code << 16 | imm);
}

Assembler& Assembler::ldr(CoreRegister rt, MemOperand op, int32_t offset) {
  return ldr(rt, MemOperand(op.base(), offset, AddressingMode::kPostIndexed));
}

Assembler& Assembler::ldr(CoreRegister rt, MemOperand op) {
  const int32_t offset = op.offset();
  constexpr int32_t max_imm12 = 4095;
  if (std::abs(offset) > max_imm12) {
    error_ = Error::kInvalidOperand;
    return *this;
  }

  return emit32(kAL | 0x41 << 20 | op.p() << 24 | op.u() << 23 | op.w() << 21 | op.base().code << 16 | rt.code << 12 |
                offset);
}

Assembler& Assembler::mov(CoreRegister rd, CoreRegister rm) {
  return mov(kAL, rd, rm);
}

Assembler& Assembler::movlo(CoreRegister rd, CoreRegister rm) {
  return mov(kLO, rd, rm);
}

Assembler& Assembler::movls(CoreRegister rd, CoreRegister rm) {
  return mov(kLS, rd, rm);
}

Assembler& Assembler::mov(Condition c, CoreRegister Rd, CoreRegister Rm) {
  return emit32(c | 0x1A << 20 | Rd.code << 12 | Rm.code);
}

Assembler& Assembler::pld(MemOperand op) {
  return emit32(0xF550F000 | op.u() << 23 | op.base().code << 16 | op.offset());
}

Assembler& Assembler::pop(CoreRegisterList regs) {
  if (!regs.has_more_than_one_register()) {
    // TODO(zhin): there is a different valid encoding for single register.
    error_ = Error::kInvalidOperand;
    return *this;
  }

  return emit32(kAL | 0x8BD << 16 | regs.list);
}

Assembler& Assembler::push(CoreRegisterList regs) {
  if (!regs.has_more_than_one_register()) {
    // TODO(zhin): there is a different valid encoding for single register.
    error_ = Error::kInvalidOperand;
    return *this;
  }

  return emit32(kAL | 0x92D << 16 | regs.list);
}

Assembler& Assembler::sub(CoreRegister rd, CoreRegister rn, CoreRegister rm) {
  return emit32(kAL | 0x4 << 20 | rn.code << 16 | rd.code << 12 | rm.code);
}

Assembler& Assembler::subs(CoreRegister rd, CoreRegister rn, uint8_t imm) {
  // Rotation = 0, since imm is limited to 8 bits and fits in encoding.
  return emit32(kAL | 0x25 << 20 | rn.code << 16 | rd.code << 12 | imm);
}

Assembler& Assembler::tst(CoreRegister rn, uint8_t imm) {
  // Rotation = 0, since imm is limited to 8 bits and fits in encoding.
  return emit32(kAL | 0x31 << 20 | rn.code << 16 | imm);
}

Assembler& Assembler::vld1_32(DRegisterList regs, MemOperand op) {
  if (regs.length != 1) {
    // Unimplemented since only length 1 is used in microkernels.
    error_ = Error::kInvalidOperand;
    return *this;
  }

  const uint32_t rm = op.mode() == AddressingMode::kPostIndexed ? 0xD : 0xF;
  return emit32(0xF4200780 | encode(regs.start, 22, 12) | op.base().code << 16 | rm);
}

Assembler& Assembler::vld1r_32(DRegisterList regs, MemOperand op) {
  if (regs.length != 2) {
    // Unimplemented since only length 2 used in microkernels.
    error_ = Error::kInvalidOperand;
    return *this;
  }

  const uint32_t rm = op.mode() == AddressingMode::kPostIndexed ? 0xD : 0xF;
  return emit32(0xF4A00CA0 | encode(regs.start, 22, 12) | op.base().code << 16 | rm);
}

Assembler& Assembler::vldm(CoreRegister rn, SRegisterList regs, bool wb) {
  return emit32(kAL | 0x0C900A00 | wb << 21 | rn.code << 16 | encode(regs, 22, 12));
}

Assembler& Assembler::vldm(CoreRegister rn, DRegisterList regs, bool wb) {
  return emit32(kAL | 0x0C900B00 | wb << 21 | rn.code << 16 | encode(regs, 22, 12));
}

Assembler& Assembler::vldr(DRegister dd, MemOperand op) {
  // TOOD(zhin): post-increment not used in any microkernels, so not implemented yet.
  if (op.mode() != AddressingMode::kOffset || std::abs(op.offset()) > UINT8_MAX) {
    error_ = Error::kInvalidOperand;
    return *this;
  }

  return emit32(kAL | 0x0D100B00 | op.u() << 23 | encode(dd, 22, 12) | op.base().code << 16 | op.offset() >> 2);
}

Assembler& Assembler::vmax_f32(QRegister qd, QRegister qn, QRegister qm) {
  return emit32(kAL | 0xF2000F40 | encode(qd, 22, 12) | encode(qn, 7, 16) | encode(qm, 5, 0));
}

Assembler& Assembler::vmin_f32(QRegister qd, QRegister qn, QRegister qm) {
  return emit32(kAL | 0xF2200F40 | encode(qd, 22, 12) | encode(qn, 7, 16) | encode(qm, 5, 0));
}

Assembler& Assembler::vmla_f32(QRegister qd, QRegister qn, DRegisterLane dm) {
  if (dm.lane > 1) {
    error_ = Error::kInvalidLaneIndex;
    return *this;
  }
  return emit32(0xF3A00140 | encode(qd, 22, 12) | encode(qn, 7, 16) | dm.lane << 5 | dm.code);
}

Assembler& Assembler::vmov(SRegister sd, SRegister sm) {
  return emit32(kAL | 0x0EB00A40 | encode(sd, 22, 12) | encode(sm, 5, 0));
}

Assembler& Assembler::vmov(DRegister dm, CoreRegister rt, CoreRegister rt2) {
  return emit32(kAL | 0x0C400B10 | rt2.code << 16 | rt.code << 12 | encode(dm, 5, 0));
}

Assembler& Assembler::vmov(DRegister dd, DRegister dm) {
  return emit32(0xF2600110 | encode(dd, 22, 12) | encode(dm, 7, 16) | encode(dm, 5, 0));
}

Assembler& Assembler::vmov(QRegister qd, QRegister qm) {
  return emit32(0xF2200150 | encode(qd, 22, 12) | encode(qm, 7, 16) | encode(qm, 5, 0));
}

Assembler& Assembler::vpop(DRegisterList regs) {
  return emit32(kAL | encode(regs, 22, 12) | 0xCBD << 16 | 0xB << 8);
}

Assembler& Assembler::vpush(SRegisterList regs) {
  return emit32(kAL | encode(regs, 22, 12) | 0xD2D << 16 | 0xA << 8);
}

Assembler& Assembler::vpush(DRegisterList regs) {
  return emit32(kAL | encode(regs, 22, 12) | 0xD2D << 16 | 0xB << 8);
}

Assembler& Assembler::vst1_32(DRegisterList regs, MemOperand op) {
  uint8_t type = encode_regs_length_to_type(regs);
  if (!type) {
    error_ = Error::kInvalidRegisterListLength;
    return *this;
  }

  const uint32_t rm = op.mode() == AddressingMode::kPostIndexed ? 0xD : 0xF;
  return emit32(0xF400'0080 | encode(regs.start, 22, 12) | op.base().code << 16 | type << 8 | rm);
}

Assembler& Assembler::vst1_32(DRegisterList regs, MemOperand op, CoreRegister rm) {
  if (rm.code == 0b1101 || rm.code == 0b1111) {
    error_ = Error::kInvalidOperand;
    return *this;
  }

  uint8_t type = encode_regs_length_to_type(regs);
  if (!type) {
    error_ = Error::kInvalidRegisterListLength;
    return *this;
  }

  return emit32(0xF400'0080 | encode(regs.start, 22, 12) | op.base().code << 16 | type << 8 | rm.code);
}

Assembler& Assembler::vst1_32(DRegisterLane dd, MemOperand op) {
  if (dd.lane > 1) {
    error_ = Error::kInvalidLaneIndex;
    return *this;
  }

  const uint32_t rm = op.mode() == AddressingMode::kPostIndexed ? 0xD : 0xF;
  return emit32(0xF480'0800 | encode(dd, 22, 12) | op.base().code << 16 | dd.lane << 5 | rm);
}

void Assembler::reset() {
  cursor_ = buffer_;
  error_ = Error::kNoError;
}
}  // namespace aarch32
}  // namespace xnnpack
