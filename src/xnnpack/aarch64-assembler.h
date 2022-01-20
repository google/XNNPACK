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

struct VRegister {
  uint8_t code;
  uint8_t size = 0;
  uint8_t q = 1;

  VRegister v8b() const { return {code, 0, 0}; }
  VRegister v16b() const { return {code, 0, 1}; }
  VRegister v4h() const { return {code, 1, 0}; }
  VRegister v8h() const { return {code, 1, 1}; }
  VRegister v2s() const { return {code, 2, 0}; }
  VRegister v4s() const { return {code, 2, 1}; }
  VRegister v1d() const { return {code, 3, 0}; }
  VRegister v2d() const { return {code, 3, 1}; }
};

constexpr VRegister v0{0};
constexpr VRegister v1{1};
constexpr VRegister v2{2};
constexpr VRegister v3{3};
constexpr VRegister v4{4};
constexpr VRegister v5{5};
constexpr VRegister v6{6};
constexpr VRegister v7{7};
constexpr VRegister v8{8};
constexpr VRegister v9{9};
constexpr VRegister v10{10};
constexpr VRegister v11{11};
constexpr VRegister v12{12};
constexpr VRegister v13{13};
constexpr VRegister v14{14};
constexpr VRegister v15{15};
constexpr VRegister v16{16};
constexpr VRegister v17{17};
constexpr VRegister v18{18};
constexpr VRegister v19{19};
constexpr VRegister v20{20};
constexpr VRegister v21{21};
constexpr VRegister v22{22};
constexpr VRegister v23{23};
constexpr VRegister v24{24};
constexpr VRegister v25{25};
constexpr VRegister v26{26};
constexpr VRegister v27{27};
constexpr VRegister v28{28};
constexpr VRegister v29{29};
constexpr VRegister v30{30};
constexpr VRegister v31{31};

struct VRegisterList {
  VRegisterList(VRegister vt1)
      : vt1(vt1), length(1) {}
  VRegisterList(VRegister vt1, VRegister vt2)
      : vt1(vt1), vt2(vt2), length(2) {}
  VRegisterList(VRegister vt1, VRegister vt2, VRegister vt3)
      : vt1(vt1), vt2(vt2), vt3(vt3), length(3) {}
  VRegisterList(VRegister vt1, VRegister vt2, VRegister vt3, VRegister vt4)
      : vt1(vt1), vt2(vt2), vt3(vt3), vt4(vt4), length(4) {}

  VRegister vt1;
  VRegister vt2;
  VRegister vt3;
  VRegister vt4;
  uint8_t length;
};

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

  Assembler& ld1(VRegisterList vs, MemOperand xn, int32_t imm);
  Assembler& ld2r(VRegisterList xs, MemOperand xn);
  Assembler& ldp(XRegister xt1, XRegister xt2, MemOperand xn);
  Assembler& ldp(XRegister xt1, XRegister xt2, MemOperand xn, int32_t imm);
  Assembler& ldr(XRegister xt, MemOperand xn);

 private:
  Assembler& emit32(uint32_t value);

};

}  // namespace aarch64
}  // namespace xnnpack
