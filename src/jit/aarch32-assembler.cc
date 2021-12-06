#include "xnnpack/aarch32-assembler.h"

#include <string.h>

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

void Assembler::reset() {
  cursor_ = buffer_;
  error_ = Error::kNoError;
}
}  // namespace aarch32
}  // namespace xnnpack
