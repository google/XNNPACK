// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembler.h>

namespace xnnpack {
namespace aarch64 {

struct XRegister {
  uint8_t code;
};

constexpr XRegister x0{0};
constexpr XRegister x1{1};
constexpr XRegister x2{2};
constexpr XRegister x3{3};
constexpr XRegister x4{4};
constexpr XRegister x5{5};
constexpr XRegister x6{6};
constexpr XRegister x7{7};
constexpr XRegister x8{8};
constexpr XRegister x9{9};
constexpr XRegister x10{10};
constexpr XRegister x11{11};
constexpr XRegister x12{12};
constexpr XRegister x13{13};
constexpr XRegister x14{14};
constexpr XRegister x15{15};
constexpr XRegister x16{16};
constexpr XRegister x17{17};
constexpr XRegister x18{18};
constexpr XRegister x19{19};
constexpr XRegister x20{20};
constexpr XRegister x21{21};
constexpr XRegister x22{22};
constexpr XRegister x23{23};
constexpr XRegister x24{24};
constexpr XRegister x25{25};
constexpr XRegister x26{26};
constexpr XRegister x27{27};
constexpr XRegister x28{28};
constexpr XRegister x29{29};
constexpr XRegister x30{30};
constexpr XRegister xzr{31};
constexpr XRegister sp{31};

// C1.3.3 Load/Store addressing modes
enum class AddressingMode {
  kOffset, // Base plus offset: [base{, #imm}] ; [base, Xm{, LSL #imm}].
  kPostIndex, // Post-index: [base], #imm ; [base], Xm.
};

struct MemOperand {
  MemOperand(XRegister xn): base(xn), mode(AddressingMode::kOffset), offset(0) {}
  MemOperand(XRegister xn, int32_t offset): base(xn), mode(AddressingMode::kOffset), offset(offset) {}
  MemOperand(XRegister xn, int32_t offset, AddressingMode mode): base(xn), mode(mode), offset(offset) {}

  XRegister base;
  AddressingMode mode;
  int32_t offset;
};

static inline MemOperand operator,(XRegister r, int32_t offset) {
  return MemOperand(r, offset);
}

// Helper struct for some syntax sugar to look like native assembly, see mem.
struct MemOperandHelper {
  const MemOperand operator[](MemOperand op) const { return op; }
  MemOperand operator[](XRegister r) const { return MemOperand(r, 0); }
};

// Use "mem" (and its overload of array subscript operator) to get some syntax
// that looks closer to native assembly when accessing memory. For example:
// - ldp(x0, x1, mem[rn, offset]); // offset
// - ldp(x0, x1, mem[rn], offset); // post-indexed
constexpr MemOperandHelper mem;

class Assembler : public AssemblerBase {
 public:
  using AssemblerBase::AssemblerBase;

  Assembler& ldp(XRegister xt1, XRegister xt2, MemOperand xn);
  Assembler& ldp(XRegister xt1, XRegister xt2, MemOperand xn, int32_t imm);

 private:
  Assembler& emit32(uint32_t value);

};

}  // namespace aarch64
}  // namespace xnnpack
