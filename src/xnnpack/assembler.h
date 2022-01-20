// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <xnnpack/allocator.h>

#include <cstdint>

namespace xnnpack {

constexpr size_t kInstructionSizeInBytes = 4;

enum class Error {
  kNoError,
  kOutOfMemory,
  kInvalidOperand,
  kLabelAlreadyBound,
  kLabelOffsetOutOfBounds,
  kLabelHasTooManyUsers,
  kInvalidLaneIndex,
  kInvalidRegisterListLength,
  kFinalizeCodeMemoryFail,
};

class AssemblerBase {
 public:
  /* // Takes an xnn_code_buffer with a pointer to allocated memory. */
  explicit AssemblerBase(xnn_code_buffer* buf);

  // Finish assembly of code, this should be the last function called on an
  // instance of Assembler. Returns a pointer to the start of code region.
  void* finalize();
  // Reset the assembler state (no memory is freed).
  void reset();

  // Get a pointer to the start of code buffer.
  const uint32_t* start() const { return buffer_; }
  const uint32_t* offset() const { return cursor_; }
  // Returns the number of bytes of code actually in the buffer.
  size_t code_size_in_bytes() const { return (cursor_ - buffer_) * kInstructionSizeInBytes; }
  const Error error() const { return error_; }

 protected:
  // Pointer to start of code buffer.
  uint32_t* buffer_;
  // Pointer to current place in code buffer.
  uint32_t* cursor_;
  // Pointer to out-of-bounds of code buffer.
  uint32_t* top_;
  // Errors encountered while assembling code.
  Error error_ = Error::kNoError;
  // Holds an xnn_code_buffer, will write code to its code pointer, and unmap
  // unused pages on finalizing.
  xnn_code_buffer* xnn_buffer = nullptr;
};

}  // namespace xnnpack
