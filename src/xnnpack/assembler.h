// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <xnnpack/memory.h>

#include <array>
#include <cstdint>
#include <cstring>

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
  kMaxNumberOfFunctionsExceeded,
};

// Biggest user of label is for callee-saved registers check in test mode.
constexpr size_t max_label_users = 16;
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
  size_t offset = SIZE_MAX;
  // A label can only be bound once, binding it again leads to an error.
  bool bound = false;
  // All users of this label, recorded by their offset in the Assembler buffer.
  std::array<size_t, max_label_users> users{{SIZE_MAX}};
  size_t num_users = 0;

  // Records a user (e.g. branch instruction) of this label.
  // Returns true if success, false if number of users exceeds maximum.
  bool add_use(size_t offset) {
    if (num_users >= max_label_users) {
      return false;
    }
    users[num_users++] = offset;
    return true;
  }
};

class AssemblerBase {
 public:
  // Takes an xnn_code_buffer with a pointer to allocated memory. If the buffer
  // already contains content (size != 0), appends to after size (up to capacity).
  explicit AssemblerBase(xnn_code_buffer* buf);

  // Return a copy of the value by offset from the initial buffer size
  uint32_t get32(size_t offset) const;
  // Write value into the code buffer and advances cursor_.
  void emit32(uint32_t value);
  // Writes value by offset from the initial buffer size
  void emit32(uint32_t value, size_t* offset);
  void emit8(byte value);
  // Writes value by offset from the initial buffer size
  void emit8(byte value, size_t* offset);
  // Moves the `size` 8-bit values emitted starting with offset (from the
  // initial buffer size) to `cursor_`.
  void move_emitted(size_t offset, size_t size) {
    std::memmove(cursor_, buffer_ + offset, size);
    cursor_ += size;
  }
  // Finish assembly of code, this should be the last function called on an
  // instance of Assembler. Returns a pointer to the start of code region.
  void* finalize();
  // Reset the assembler state (no memory is freed).
  void reset();

  // Get a pointer to the start of code buffer.
  const byte* start() const { return buffer_; }
  const byte* offset() const { return cursor_; }
  template<typename T>
  T offset() const { return reinterpret_cast<T>(cursor_); }
  // Returns the number of bytes of code actually in the buffer.
  size_t code_size_in_bytes() const { return (cursor_ - buffer_); }
  Error error() const { return error_; }

 protected:
  // Errors encountered while assembling code.
  Error error_ = Error::kNoError;

 private:
  template <typename Value>
  void emit(Value value, size_t* offset) {
    if (error_ != Error::kNoError) {
      return;
    }
    if (sizeof(Value) > (top_ - buffer_) - *offset) {
      error_ = Error::kOutOfMemory;
      return;
    }

    memcpy(buffer_ + *offset, &value, sizeof(Value));
    *offset += sizeof(Value);
  }

  template <typename Value>
  Value get(size_t offset) const {
    return *reinterpret_cast<const Value*>(start() + offset);
  }

  template <typename Value>
  void emit(Value value, byte*& cursor) {
    size_t offset = cursor_ - buffer_;
    emit(value, &offset);
    cursor += sizeof(value);
  }

  byte* buffer_start() const {
    return static_cast<byte*>(xnn_buffer->start);
  }

  // Pointer into code buffer to start writing code.
  byte* buffer_;
  // Pointer to current position in code buffer.
  byte* cursor_;
  // Pointer to out-of-bounds of code buffer.
  byte* top_;
  // Holds an xnn_code_buffer, will write code to its code pointer, and unmap
  // unused pages on finalizing.
  xnn_code_buffer* xnn_buffer = nullptr;
};

}  // namespace xnnpack
