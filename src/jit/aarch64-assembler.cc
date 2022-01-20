// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/aarch64-assembler.h>

#include <cmath>

namespace xnnpack {
namespace aarch64 {
// Min and max values for the imm7 for ldp, will be shifted right by 3 when encoding.
constexpr int32_t kImm7Min = -512;
constexpr int32_t kImm7Max = 504;

Assembler& Assembler::ldp(XRegister xt1, XRegister xt2, MemOperand xn) {
  if (xn.offset < kImm7Min || xn.offset > kImm7Max || std::abs(xn.offset) % 8 != 0) {
    error_ = Error::kInvalidOperand;
    return *this;
  }

  const uint32_t mode = xn.mode == AddressingMode::kOffset ? 2 : 1;
  const uint32_t offset = (xn.offset >> 3) & 0x7F;

  return emit32(0xA8400000 | mode << 23 | offset << 15 | xt2.code << 10 | xn.base.code << 5 | xt1.code);
}

Assembler& Assembler::ldp(XRegister xt1, XRegister xt2, MemOperand xn, int32_t imm) {
  if (xn.offset != 0) {
    error_ = Error::kInvalidOperand;
    return *this;
  }
  return ldp(xt1, xt2, {xn.base, imm, AddressingMode::kPostIndex});
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

}  // namespace aarch64
}  // namespace xnnpack
