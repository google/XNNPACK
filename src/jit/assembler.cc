// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembler.h>

#include <xnnpack/allocator.h>

namespace xnnpack {

AssemblerBase::AssemblerBase(xnn_code_buffer* buf) {
  buffer_ = reinterpret_cast<byte*>(buf->code);
  cursor_ = buffer_;
  top_ = buffer_ + buf->capacity;
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
  xnn_buffer->size = code_size_in_bytes();
  if (xnn_finalize_code_memory(xnn_buffer) != xnn_status_success) {
    error_ = Error::kFinalizeCodeMemoryFail;
  }
  return reinterpret_cast<void*>(buffer_);
}

void AssemblerBase::reset() {
  cursor_ = buffer_;
  error_ = Error::kNoError;
}

}  // namespace xnnpack
