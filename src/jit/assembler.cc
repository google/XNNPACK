// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

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

void AssemblerBase::emit32(uint32_t value) {
  if (error_ != Error::kNoError) {
    return;
  }

  if (cursor_ + sizeof(value) > top_) {
    error_ = Error::kOutOfMemory;
    return;
  }

  memcpy(cursor_, &value, sizeof(value));
  cursor_ += sizeof(value);
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
