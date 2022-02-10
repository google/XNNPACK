// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <xnnpack/allocator.h>

#include <array>
#include <cstdint>

typedef uint8_t byte;

namespace xnnpack {

constexpr size_t kInstructionSizeInBytes = 4;
constexpr size_t kInstructionSizeInBytesLog2 = 2;

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
  kUnimplemented,
};

constexpr size_t max_label_users = 10;
// Label is a target of a branch. You call Assembler::bind to bind a label to an
// actual location in the instruction stream.
//
// ```
// Label l;
// b(kAl, l1); // branch to an unbound label is fine, it will be patched later.
// a.bind(l); // binds label to this location in the instruction stream.
// b(kAl, l1); // branch to an already bound label.
// ```
struct Label {
  // Location of label within Assembler buffer.
  byte* offset = nullptr;
  // A label can only be bound once, binding it again leads to an error.
  bool bound = (offset != nullptr);
  // All users of this label, recorded by their offset in the Assembler buffer.
  std::array<byte*, max_label_users> users{{0}};
  size_t num_users = 0;

  // Records a user (e.g. branch instruction) of this label.
  // Returns true if success, false if number of users exceeds maximum.
  bool add_use(byte* offset) {
    if (num_users >= max_label_users) {
      return false;
    }
    users[num_users++] = offset;
    return true;
  }
};

class AssemblerBase {
 public:
  // Takes an xnn_code_buffer with a pointer to allocated memory.
  explicit AssemblerBase(xnn_code_buffer* buf);

  // Write value into the code buffer and advances cursor_.
  void emit32(uint32_t value);
  // Finish assembly of code, this should be the last function called on an
  // instance of Assembler. Returns a pointer to the start of code region.
  void* finalize();
  // Reset the assembler state (no memory is freed).
  void reset();

  // Get a pointer to the start of code buffer.
  const byte* start() const { return buffer_; }
  const byte* offset() const { return cursor_; }
  template<typename T>
  const T offset() const { return reinterpret_cast<T>(cursor_); }
  // Returns the number of bytes of code actually in the buffer.
  size_t code_size_in_bytes() const { return (cursor_ - buffer_); }
  const Error error() const { return error_; }

 protected:
  // Pointer to start of code buffer.
  byte* buffer_;
  // Pointer to current place in code buffer.
  byte* cursor_;
  // Pointer to out-of-bounds of code buffer.
  byte* top_;
  // Errors encountered while assembling code.
  Error error_ = Error::kNoError;
  // Holds an xnn_code_buffer, will write code to its code pointer, and unmap
  // unused pages on finalizing.
  xnn_code_buffer* xnn_buffer = nullptr;
};

}  // namespace xnnpack
