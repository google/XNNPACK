// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembler.h>

namespace xnnpack {
namespace aarch64 {

constexpr size_t kInstructionSizeInBytesLog2 = 2;

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

struct VRegisterLane {
  uint8_t code;
  uint8_t size;
  uint8_t lane;
  const bool is_s() { return size == 2; };
};

struct ScalarVRegister{
  uint8_t code;
  uint8_t size = 0;

  const VRegisterLane operator[](std::size_t pos) const {
    return VRegisterLane{code, size, static_cast<uint8_t>(pos)};
  }
};

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

  ScalarVRegister s() const { return {code, 2}; }

  const bool is_s() { return size == 2; };
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

struct QRegister {
  uint8_t code;
};

constexpr QRegister q0{0};
constexpr QRegister q1{1};
constexpr QRegister q2{2};
constexpr QRegister q3{3};
constexpr QRegister q4{4};
constexpr QRegister q5{5};
constexpr QRegister q6{6};
constexpr QRegister q7{7};
constexpr QRegister q8{8};
constexpr QRegister q9{9};
constexpr QRegister q10{10};
constexpr QRegister q11{11};
constexpr QRegister q12{12};
constexpr QRegister q13{13};
constexpr QRegister q14{14};
constexpr QRegister q15{15};
constexpr QRegister q16{16};
constexpr QRegister q17{17};
constexpr QRegister q18{18};
constexpr QRegister q19{19};
constexpr QRegister q20{20};
constexpr QRegister q21{21};
constexpr QRegister q22{22};
constexpr QRegister q23{23};
constexpr QRegister q24{24};
constexpr QRegister q25{25};
constexpr QRegister q26{26};
constexpr QRegister q27{27};
constexpr QRegister q28{28};
constexpr QRegister q29{29};
constexpr QRegister q30{30};
constexpr QRegister q31{31};

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

enum class PrefetchOp {
  kPLDL1KEEP = 0
};

constexpr PrefetchOp PLDL1KEEP = PrefetchOp::kPLDL1KEEP;

enum Condition : uint32_t {
  kEQ = 0x0,
  kNE = 0x1,
  kCS = 0x2,
  kCC = 0x3,
  kMI = 0x4,
  kPL = 0x5,
  kVS = 0x6,
  kVC = 0x7,
  kHI = 0x8,
  kLS = 0x9,
  kGE = 0xa,
  kLT = 0xB,
  kGT = 0xC,
  kLE = 0xD,
  kAL = 0xE,
  kHS = kCS,
  kLO = kCC,
};

class Assembler : public AssemblerBase {
 public:
  using AssemblerBase::AssemblerBase;

  // Base instructions.
  Assembler& b_eq(Label& l) { return b(kEQ, l); }
  Assembler& b_hi(Label& l) { return b(kHI, l); }
  Assembler& b_hs(Label& l) { return b(kHS, l); }
  Assembler& b_lo(Label& l) { return b(kLO, l); }
  Assembler& b_ne(Label& l) { return b(kNE, l); }
  Assembler& ldp(XRegister xt1, XRegister xt2, MemOperand xn);
  Assembler& ldp(XRegister xt1, XRegister xt2, MemOperand xn, int32_t imm);
  Assembler& ldr(XRegister xt, MemOperand xn);
  Assembler& prfm(PrefetchOp prfop, MemOperand xn);
  Assembler& subs(XRegister xd, XRegister xn, uint16_t imm12);
  Assembler& tbnz(XRegister xd, uint8_t bit, Label& l);

  // SIMD instructions
  Assembler& fadd(VRegister vd, VRegister vn, VRegister vm);
  Assembler& fmax(VRegister vd, VRegister vn, VRegister vm);
  Assembler& fmin(VRegister vd, VRegister vn, VRegister vm);
  Assembler& fmla(VRegister vd, VRegister vn, VRegisterLane vm);
  Assembler& ld1(VRegisterList vs, MemOperand xn, int32_t imm);
  Assembler& ld2r(VRegisterList xs, MemOperand xn);
  Assembler& ldp(QRegister qt1, QRegister qt2, MemOperand xn, int32_t imm);
  Assembler& ldr(QRegister qt, MemOperand xn, int32_t imm);
  Assembler& movi(VRegister vd, uint8_t imm);

  // Binds Label l to the current location in the code buffer.
  Assembler& bind(Label& l);

 private:
  Assembler& emit32(uint32_t value);
  Assembler& b(Condition c, Label& l);

};

}  // namespace aarch64
}  // namespace xnnpack
