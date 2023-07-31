// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <cstdint>

#include <xnnpack/assembler.h>
#include <xnnpack/memory.h>

namespace xnnpack {

AssemblerBase::AssemblerBase(xnn_code_buffer* buf) {
  byte* buf_start = reinterpret_cast<byte*>(buf->start);
  buffer_ = buf_start + buf->size;
  cursor_ = buffer_;
  top_ = buf_start + buf->capacity;
  xnn_buffer = buf;
}

uint32_t AssemblerBase::get32(size_t offset) const {
  return get<uint32_t>(offset);
}

void AssemblerBase::emit32(uint32_t value) {
  emit<uint32_t>(value, cursor_);
}

void AssemblerBase::emit32(uint32_t value, size_t* offset) {
  emit<uint32_t>(value, offset);
}

void AssemblerBase::emit8(byte value) {
  emit<byte>(value, cursor_);
}

void AssemblerBase::emit8(byte value, size_t* offset) {
  emit<byte>(value, offset);
}

void* AssemblerBase::finalize() {
  if (error_ != Error::kNoError) {
    return NULL;
  }
  xnn_buffer->size += code_size_in_bytes();
  return reinterpret_cast<void*>(buffer_);
}

void AssemblerBase::reset() {
  xnn_buffer->size -= (cursor_ - buffer_);
  cursor_ = buffer_;
  error_ = Error::kNoError;
}
}  // namespace xnnpack
