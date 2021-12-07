#include "xnnpack/aarch32-assembler.h"

#include <cmath>

namespace xnnpack {
namespace aarch32 {
static const int DEFAULT_BUFFER_SIZE = 4096;

Assembler::Assembler() {
  buffer_ = new uint32_t[DEFAULT_BUFFER_SIZE];
  cursor_ = buffer_;
  top_ = buffer_ + DEFAULT_BUFFER_SIZE;
  error_ = Error::kNoError;
}

Assembler::~Assembler() { delete[] buffer_; }

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

Assembler& Assembler::add(CoreRegister Rd, CoreRegister Rn, CoreRegister Rm) {
  return emit32(kAL | 0x8 << 20 | Rn.code << 16 | Rd.code << 12 | Rm.code);
}

Assembler& Assembler::cmp(CoreRegister Rn, uint8_t imm) {
  return emit32(kAL | 0x35 << 20 | Rn.code << 16 | imm);
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

  return emit32(kAL | 0x41 << 20 | op.p() << 24 | op.u() << 23 | op.w() << 21 |
                op.base().code << 16 | rt.code << 12 | offset);
}

Assembler& Assembler::push(CoreRegisterList registers) {
  if (!registers.has_more_than_one_register()) {
    // TODO(zhin): there is a different valid encoding for single register.
    error_ = Error::kInvalidOperand;
  }

  return emit32(kAL | 0x92D << 16 | registers.list);
}

void Assembler::reset() {
  cursor_ = buffer_;
  error_ = Error::kNoError;
}
}  // namespace aarch32
}  // namespace xnnpack
