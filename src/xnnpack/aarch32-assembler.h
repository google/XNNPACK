// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include <xnnpack/assembler.h>

// MSVC defines these tokens using macros, causing name collisions. We use these name in our assembler, so undef them.
// These macros are pulled in via xnnpack/math.h -> intrin.h -> arm_neon.h.
#include <xnnpack/math.h>
#ifdef vabs_f32
  #undef vabs_f32
#endif
#ifdef vadd_f32
  #undef vadd_f32
#endif
#ifdef vcvt_f32_s32
  #undef vcvt_f32_s32
#endif
#ifdef vcvt_s32_f32
  #undef vcvt_s32_f32
#endif
#ifdef vcvtn_s32_f32
  #undef vcvtn_s32_f32
#endif
#ifdef vmax_f32
  #undef vmax_f32
#endif
#ifdef vmax_s8
  #undef vmax_s8
#endif
#ifdef vmin_f32
  #undef vmin_f32
#endif
#ifdef vmin_s8
  #undef vmin_s8
#endif
#ifdef vmla_f32
  #undef vmla_f32
#endif
#ifdef vmlal_s16
  #undef vmlal_s16
#endif
#ifdef vmovl_s8
  #undef vmovl_s8
#endif
#ifdef vmul_f32
  #undef vmul_f32
#endif
#ifdef vneg_f32
  #undef vneg_f32
#endif
#ifdef vqadd_s16
  #undef vqadd_s16
#endif
#ifdef vqdmulh_s32
  #undef vqdmulh_s32
#endif
#ifdef vqmovn_s16
  #undef vqmovn_s16
#endif
#ifdef vqmovn_s32
  #undef vqmovn_s32
#endif
#ifdef vqshl_s32
  #undef vqshl_s32
#endif
#ifdef vrshl_s32
  #undef vrshl_s32
#endif

namespace xnnpack {
namespace aarch32 {

// Special values used to check that callee-saved registers are properly saved.
// Low 8 bits should be 0 to encode register code.
constexpr uint32_t kRRegisterCorruptValue = UINT32_C(0xDEADBE00);
constexpr uint32_t kSRegisterCorruptValue = UINT32_C(0x7FF00000);
constexpr uint8_t kRegisterCorruptMask = UINT8_C(0xFF);

// Instruction used to align code, is a nop.
constexpr uint32_t kAlignInstruction = 0xE320F000;

enum class SpecialFPRegister {
  kFPSCR = 1,
};

constexpr SpecialFPRegister FPSCR = SpecialFPRegister::kFPSCR;

struct CoreRegister {
  uint8_t code;
};

constexpr CoreRegister r0{0};
constexpr CoreRegister r1{1};
constexpr CoreRegister r2{2};
constexpr CoreRegister r3{3};
constexpr CoreRegister r4{4};
constexpr CoreRegister r5{5};
constexpr CoreRegister r6{6};
constexpr CoreRegister r7{7};
constexpr CoreRegister r8{8};
constexpr CoreRegister r9{9};
constexpr CoreRegister r10{10};
constexpr CoreRegister r11{11};
constexpr CoreRegister r12{12};
constexpr CoreRegister r13{13};
constexpr CoreRegister r14{14};
constexpr CoreRegister r15{15};
constexpr CoreRegister sp = r13;
constexpr CoreRegister lr = r14;
constexpr CoreRegister pc = r15;
constexpr CoreRegister APSR_nzcv = r15;

static inline bool operator==(const CoreRegister lhs, const CoreRegister rhs) {
  return lhs.code == rhs.code;
}

struct CoreRegisterList {
  CoreRegisterList(std::initializer_list<CoreRegister> rs) {
    for (auto r : rs) {
      list |= 1 << r.code;
    }
  }

  bool has_more_than_one_register() { return (list & (list - 1)) != 0; }

  // Bit i is set if CoreRegister is in the list.
  uint16_t list = 0;
};

static inline bool operator==(int i, CoreRegisterList registers) {
  return i == registers.list;
}

struct SRegister {
  uint8_t code;
  uint8_t d() const { return code & 0x1; }
  uint8_t vd() const { return (code & 0x1e) >> 1; }
};

static inline bool operator==(const SRegister lhs, const SRegister rhs) {
  return lhs.code == rhs.code;
}

constexpr SRegister s0{0};
constexpr SRegister s1{1};
constexpr SRegister s2{2};
constexpr SRegister s3{3};
constexpr SRegister s4{4};
constexpr SRegister s5{5};
constexpr SRegister s6{6};
constexpr SRegister s7{7};
constexpr SRegister s8{8};
constexpr SRegister s9{9};
constexpr SRegister s10{10};
constexpr SRegister s11{11};
constexpr SRegister s12{12};
constexpr SRegister s13{13};
constexpr SRegister s14{14};
constexpr SRegister s15{15};
constexpr SRegister s16{16};
constexpr SRegister s17{17};
constexpr SRegister s18{18};
constexpr SRegister s19{19};
constexpr SRegister s20{20};
constexpr SRegister s21{21};
constexpr SRegister s22{22};
constexpr SRegister s23{23};
constexpr SRegister s24{24};
constexpr SRegister s25{25};
constexpr SRegister s26{26};
constexpr SRegister s27{27};
constexpr SRegister s28{28};
constexpr SRegister s29{29};
constexpr SRegister s30{30};
constexpr SRegister s31{31};

// Define DRegisterLane before DRegister so that we can have the operator[] overloading for nice syntax.
struct DRegisterLane {
  uint8_t code;
  uint8_t lane;

  uint8_t d() const { return (code & 0x10) >> 4; }
  uint8_t vd() const { return code & 0xf; }
};

static inline bool operator==(const DRegisterLane lhs, const DRegisterLane rhs) {
  return lhs.code == rhs.code && lhs.lane == rhs.lane;
}

struct DRegister {
  uint8_t code;

  uint8_t d() const { return (code & 0x10) >> 4; }
  uint8_t vd() const { return code & 0xf; }
  SRegister low() const { return SRegister{uint8_t(code * 2)}; }
  SRegister high() const { return SRegister{uint8_t(code * 2 + 1)}; }

  DRegisterLane operator[](std::size_t pos) const {
    return DRegisterLane{code, static_cast<uint8_t>(pos)};
  }
};

static inline bool operator==(const DRegister lhs, const DRegister rhs) {
  return lhs.code == rhs.code;
}

constexpr DRegister d0{0};
constexpr DRegister d1{1};
constexpr DRegister d2{2};
constexpr DRegister d3{3};
constexpr DRegister d4{4};
constexpr DRegister d5{5};
constexpr DRegister d6{6};
constexpr DRegister d7{7};
constexpr DRegister d8{8};
constexpr DRegister d9{9};
constexpr DRegister d10{10};
constexpr DRegister d11{11};
constexpr DRegister d12{12};
constexpr DRegister d13{13};
constexpr DRegister d14{14};
constexpr DRegister d15{15};
constexpr DRegister d16{16};
constexpr DRegister d17{17};
constexpr DRegister d18{18};
constexpr DRegister d19{19};
constexpr DRegister d20{20};
constexpr DRegister d21{21};
constexpr DRegister d22{22};
constexpr DRegister d23{23};
constexpr DRegister d24{24};
constexpr DRegister d25{25};
constexpr DRegister d26{26};
constexpr DRegister d27{27};
constexpr DRegister d28{28};
constexpr DRegister d29{29};
constexpr DRegister d30{30};
constexpr DRegister d31{31};

struct QRegister {
  uint8_t code;
  // Encode code * 2.
  uint8_t d() const { return (code & 0x8) >> 3; }
  uint8_t vd() const { return (code & 0x7) << 1; }
  DRegister low() const { return DRegister{uint8_t(code * 2)}; }
  DRegister high() const { return DRegister{uint8_t(code * 2 + 1)}; }
};

static inline bool operator==(const QRegister lhs, const QRegister rhs) {
  return lhs.code == rhs.code;
}

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

// SIMD register lists are used in a more restrictive way, compared to core
// registers, only consecutive registers are used as an operand to instruction.
template <typename RegType>
struct ConsecutiveRegisterList {
  // End must be >= start.
  ConsecutiveRegisterList(RegType s, RegType end)
      : start(s),
        length(end.code - s.code + 1) {}
  explicit ConsecutiveRegisterList(RegType s, int len)
      : start(s),
        length(len) {}
  // NOLINTNEXTLINE(google-explicit-constructor)
  ConsecutiveRegisterList(RegType start)
      : ConsecutiveRegisterList(start, start) {}

  RegType start;
  uint8_t length;
};

// Specific struct for VLD2 and VLD3 register list operand.
struct VLoadStoreRegList {
  VLoadStoreRegList(DRegister reg1, DRegister reg2)
      : reg1(reg1), reg2(reg2) {
    if (reg1.code == reg2.code - 2) {
      double_spaced = true;
    } else {
      double_spaced = false;
    }
  }
  VLoadStoreRegList(DRegister reg1, DRegister reg2, DRegister reg3)
      : reg1(reg1), reg2(reg2), reg3(reg3) {
    if (reg1.code == reg2.code - 2) {
      double_spaced = true;
    } else {
      double_spaced = false;
    }
  }

  DRegister reg1;
  DRegister reg2;
  DRegister reg3;
  bool double_spaced;
};

using SRegisterList = ConsecutiveRegisterList<SRegister>;
using DRegisterList = ConsecutiveRegisterList<DRegister>;

static inline SRegisterList operator-(const SRegister lhs, const SRegister rhs) {
  return SRegisterList(lhs, rhs);
}

static inline DRegisterList operator-(const DRegister lhs, const DRegister rhs) {
  return DRegisterList(lhs, rhs);
}

struct QRegisterList {
  // NOLINTNEXTLINE(google-explicit-constructor)
  QRegisterList(QRegister s) : start(s), length(1) {}
  QRegisterList(QRegister s, QRegister end) : start(s), length(end.code - s.code + 1) {}
  // Explicit conversion to DRegisterList.
  explicit operator DRegisterList() const {
    return DRegisterList({static_cast<uint8_t>(start.code * 2)}, length * 2);
  }

  QRegister start;
  uint8_t length;
};

static inline QRegisterList operator-(const QRegister lhs, const QRegister rhs) {
  return QRegisterList(lhs, rhs);
}

// A8.5 Addressing modes for memory access.
enum class AddressingMode {
  // [<Rn>, <offset>], offset applied to address in Rn.
  kOffset,
  // Pre-indexed not used, so not implemented.
  // [<Rn>], <offset>, address from Rn, offset applied, written back to Rn.
  kPostIndexed,
};

// Memory operands, operands for memory access instructions. See
// "MemOperandHelper mem" for a nicer syntax that is closer to assembly.
class MemOperand {
 public:
  MemOperand(CoreRegister rn, int32_t offset)
      : mode_(AddressingMode::kOffset),
        rn_(rn),
        offset_(offset) {}

  MemOperand(CoreRegister rn, int32_t offset, AddressingMode mode)
      : mode_(mode),
        rn_(rn),
        offset_(offset) {}

  CoreRegister base() const { return rn_; }
  int32_t offset() const { return offset_; }
  AddressingMode mode() const { return mode_; }

  // These are bits used for encoding, named based on the encoding description.
  int32_t u() { return static_cast<int32_t>(offset_ >= 0); }
  int32_t p() { return static_cast<int32_t>(mode_ != AddressingMode::kPostIndexed); }
  // Note, kPostIndexed will write back, but doesn't need to set bit w.
  int32_t w() { return 0; }

  // Overload postfix increment to indicate a post-indexed addressing mode for load/stores.
  MemOperand operator++(int) {
    mode_ = AddressingMode::kPostIndexed;
    return *this;
  }

 private:
  AddressingMode mode_;
  CoreRegister rn_;
  int32_t offset_;
};

static inline bool operator==(const MemOperand lhs, const MemOperand rhs) {
  return lhs.mode() == rhs.mode() && lhs.base() == rhs.base() && lhs.offset() == rhs.offset();
}

static inline MemOperand operator,(CoreRegister r, int32_t offset) {
  return MemOperand(r, offset);
}

// Helper struct for some syntax sugar to look like native assembly, see mem.
struct MemOperandHelper {
  MemOperand operator[](MemOperand op) const { return op; }
  MemOperand operator[](CoreRegister r) const { return MemOperand(r, 0); }
};

// Use "mem" (and its overload of array subscript operator) to get some syntax
// that looks closer to native assembly when accessing memory. For example:
// - ldr(r0, mem[rn, offset]); // offset
// - ldr(r0, mem[rn], offset); // post-indexed
constexpr MemOperandHelper mem;

// Conditional execution, only support AL (always) for now.
enum Condition : uint32_t {
  kEQ = 0x00000000,
  kNE = 0x10000000,
  kCS = 0x20000000,
  kCC = 0x30000000,
  kMI = 0x40000000,
  kPL = 0x50000000,
  kVS = 0x60000000,
  kVC = 0x70000000,
  kHI = 0x80000000,
  kLS = 0x90000000,
  kGE = 0xa0000000,
  kLT = 0xB0000000,
  kGT = 0xC0000000,
  kLE = 0xD0000000,
  kAL = 0xE0000000,
  kHS = kCS,
  kLO = kCC,
};

enum DataSize {
  k8 = 0,
  k16 = 1,
  k32 = 2,
};

// A simple AAarch32 assembler.
class Assembler : public AssemblerBase {
 public:
  using AssemblerBase::AssemblerBase;

  void add(CoreRegister rn, CoreRegister rm) { add(rn, rn, rm); }
  void add(CoreRegister rd, CoreRegister rn, CoreRegister rm);
  // Only support uint8_t immediates for now, it simplifies encoding.
  void add(CoreRegister rd, CoreRegister rn, uint8_t imm);
  void adds(CoreRegister rd, CoreRegister rn, uint8_t imm);
  void and_(CoreRegister rd, CoreRegister rn, uint8_t imm);
  void b(Label& l) { b(kAL, l); }
  void beq(Label& l) { b(kEQ, l); }
  void bne(Label& l) { b(kNE, l); }
  void bhi(Label& l) { b(kHI, l); }
  void bhs(Label& l) { b(kHS, l); }
  void blo(Label& l) { b(kLO, l); }
  void blx(CoreRegister rm);
  void bic(CoreRegister rd, CoreRegister rn, uint8_t imm);
  void bx(CoreRegister rm);
  // Cmp supports a subset of uint32_t offsets, see "A5.2.4 Modified immediate
  // constants in ARM instructions", for simplicity we start with uint8_t, which
  // is fully representation using a "rotation" of 0.
  void cmp(CoreRegister rn, uint8_t imm);
  void cmp(CoreRegister rn, CoreRegister rm);
  void ldr(CoreRegister rt, MemOperand operand, int32_t offset);
  void ldr(CoreRegister rt, MemOperand operand);
  // LDRD <Rt>, <Rt2>, [<Rn>{, #+/-<imm>}].
  void ldrd(CoreRegister rt, CoreRegister rt2, MemOperand op);
  void mov(CoreRegister rd, CoreRegister rm);
  void mov(CoreRegister rd, uint16_t imm);
  void movt(CoreRegister rd, uint16_t imm);
  void moveq(CoreRegister rd, CoreRegister rm) { mov(kEQ, rd, rm); }
  void movlo(CoreRegister rd, CoreRegister rm) { mov(kLO, rd, rm); }
  void movls(CoreRegister rd, CoreRegister rm) { mov(kLS, rd, rm); }
  void nop();
  void pld(MemOperand operand);
  void pop(CoreRegisterList regs);
  void push(CoreRegisterList regs);
  void str(CoreRegister rt, MemOperand op);
  void sub(CoreRegister rd, CoreRegister rn, uint8_t imm);
  void sub(CoreRegister rd, CoreRegister rn, CoreRegister rm);
  // Only support uint8_t immediates for now, it simplifies encoding.
  void subs(CoreRegister rd, CoreRegister rn, uint8_t imm);
  void tst(CoreRegister rn, uint8_t imm);

  // SIMD instructions.
  void vabs_f32(QRegister qd, QRegister qm);
  void vadd_f32(QRegister qd, QRegister qn, QRegister qm);
  void vcmpe_f32(SRegister sd, SRegister sm);
  void vcvt_f32_s32(QRegister qd, QRegister qm);
  void vcvt_s32_f32(QRegister qd, QRegister qm);
  void vcvtn_s32_f32(QRegister qd, QRegister qm);
  void vdup_8(QRegister qd, DRegisterLane dm) { vdup(k8, qd, dm); }
  void vdup_16(QRegister qd, DRegisterLane dm) { vdup(k16, qd, dm); }
  void vdup_32(QRegister qd, DRegisterLane dm) { vdup(k32, qd, dm); }
  void vext_8(QRegister qd, QRegister qn, QRegister qm, uint8_t imm4);
  // VLD1.8 <list>, [<Rn>]{!} (multiple single elements).
  void vld1_8(DRegisterList regs, MemOperand op) { vld1(k8, regs, op); }
  void vld1_8(DRegisterList regs, MemOperand op, CoreRegister rm) { vld1(k8, regs, op, rm); }
  void vld1_8(QRegisterList regs, MemOperand op) { vld1(k8, static_cast<DRegisterList>(regs), op); }
  // VLD1.32 <list>, [<Rn>]{!} (multiple single elements).
  void vld1_32(DRegisterList regs, MemOperand op) { vld1(k32, regs, op); }
  void vld1_32(QRegisterList regs, MemOperand op) { vld1(k32, static_cast<DRegisterList>(regs), op); }
  // VLD1.32 <list>, [<Rn>]{!} (single element to one lane).
  void vld1_32(DRegisterLane dd, MemOperand op);
  // VLD1.32 <list>, [<Rn>]{!} (single element to all lanes).
  // We cannot differentiate the register list in C++ syntax, so use an instruction name similar to AArch64 LD1R.
  void vld1r_32(DRegisterList regs, MemOperand op);
  void vld2r_32(VLoadStoreRegList regs, MemOperand op);
  void vld3r_32(VLoadStoreRegList regs, MemOperand op);
  // VLDM <Rn>{!}, <list> (IA).
  void vldm(MemOperand rn, SRegisterList regs);
  void vldm(MemOperand rn, DRegisterList regs);
  void vldr(SRegister sd, MemOperand op);
  void vldr(DRegister dd, MemOperand op);
  void vmax_f32(QRegister qd, QRegister qn, QRegister qm);
  void vmax_s8(QRegister qd, QRegister qn, QRegister qm);
  void vmin_f32(QRegister qd, QRegister qn, QRegister qm);
  void vmin_s8(QRegister qd, QRegister qn, QRegister qm);
  // VMLA.F32 <Sd>, <Sn>, <Sm>
  void vmla_f32(SRegister sd, SRegister sn, SRegister sm);
  // VMLA.F32 <Qd>, <Qn>, <Dm[x]>
  void vmla_f32(QRegister qd, QRegister qn, DRegisterLane dm);
  // VMLAL.S16 <Qd>, <Dn>, <Dm[x]>
  void vmlal_s16(QRegister qd, DRegister dn, DRegisterLane dm);
  // VMOV.I32 <Qd>, #<imm>; encoding A1
  void vmov_i32(QRegister qd, uint8_t imm);
  // VMOV.F32 <Qd>, #<imm>; encoding A1
  void vmov(QRegister qd, uint8_t imm);
  // VMOV <Rt>, <Sn>; encoding A1.
  void vmov(CoreRegister rt, SRegister sn);
  // VMOV <Sn>, <Rt>; encoding A1.
  void vmov(SRegister sn, CoreRegister rt);
  // VMOV.F32 <Sd>, <Sm>; encoding A2.
  void vmov(SRegister sd, SRegister sm);
  // VMOV <Dm>, <Rt>, <Rt2>; encoding A1.
  void vmov(DRegister dm, CoreRegister rt, CoreRegister rt2);
  // VMOV <Rt>, <Rt2>, <Dm>; encoding A1.
  void vmov(CoreRegister rt, CoreRegister rt2, DRegister dm);
  // VMOV <Dd>, <Dm>; encoding A1.
  void vmov(DRegister dd, DRegister dm);
  // VMOV <Qd>, <Qm>; encoding A1.
  void vmov(QRegister qd, QRegister qm);
  // VMOV_F32 <Sd>, <Sm>
  void vmov_f32(SRegister sd, SRegister sm) { vmov_f32(kAL, sd, sm); }
  void vmovpl_f32(SRegister sd, SRegister sm) { vmov_f32(kPL, sd, sm); }
  void vmovmi_f32(SRegister sd, SRegister sm) { vmov_f32(kMI, sd, sm); }
  // VMOV_F64 <Dd>, <Dm>
  void vmov_f64(DRegister dd, DRegister dm);
  // VMOVL.S8 <Qd>, <Dm>
  void vmovl_s8(QRegister qd, DRegister dm);
  void vmrs(CoreRegister rt, SpecialFPRegister spec_reg);
  void vmul_f32(QRegister qd, QRegister qn, QRegister qm);
  // VMUL.F32 <Qd>, <Qn>, <Dm[x]>
  void vmul_f32(QRegister qd, QRegister qn, DRegisterLane dm);
  void vneg_f32(QRegister qd, QRegister qm);
  void vpop(DRegisterList regs);
  void vpush(DRegisterList regs);
  void vpush(SRegisterList regs);
  void vqadd_s16(QRegister qd, QRegister qn, QRegister qm);
  void vqdmulh_s32(QRegister qd, QRegister qn, DRegisterLane dm);
  void vqmovn_s16(DRegister dd, QRegister qm);
  void vqmovn_s32(DRegister dd, QRegister qm);
  void vqshl_s32(QRegister qd, QRegister qm, QRegister qn);
  void vrshl_s32(QRegister qd, QRegister qm, QRegister qn);
  void vsdot_s8(QRegister qd, QRegister qn, DRegisterLane dm);
  // VST1.8 <list>, [<Rn>]{!} (multiple single elements).
  void vst1_8(DRegisterList regs, MemOperand op) { vst1(k8, regs, op); }
  // VST1.8 <list>, [<Rn>]{!}, <Rm> (multiple single elements).
  void vst1_8(DRegisterList regs, MemOperand op, CoreRegister rm) { vst1(k8, regs, op, rm); }
  // VST1.8 <list>, [<Rn>]{!} (single element form one lane).
  void vst1_8(DRegisterLane dd, MemOperand op) { vst1(k8, dd, op); }
  // VST1.16 <list>, [<Rn>]{!} (multiple single elements).
  void vst1_16(DRegisterList regs, MemOperand op) { vst1(k16, regs, op); }
  // VST1.16 <list>, [<Rn>]{!}, <Rm> (multiple single elements).
  void vst1_16(DRegisterList regs, MemOperand op, CoreRegister rm) { vst1(k16, regs, op, rm); }
  // VST1.16 <list>, [<Rn>]{!} (single element form one lane).
  void vst1_16(DRegisterLane dd, MemOperand op) { vst1(k16, dd, op); }
  // VST1.32 <list>, [<Rn>]{!} (multiple single elements).
  void vst1_32(DRegisterList regs, MemOperand op) { vst1(k32, regs, op); }
  // VST1.32 <list>, [<Rn>]{!}, <Rm> (multiple single elements).
  void vst1_32(DRegisterList regs, MemOperand op, CoreRegister rm) { vst1(k32, regs, op, rm); }
  // VST1.32 <list>, [<Rn>]{!} (single element form one lane).
  void vst1_32(DRegisterLane dd, MemOperand op) { vst1(k32, dd, op); }
  // VSTM <Rn>{!}, <list>, consecutive 64-bit registers.
  void vstm(MemOperand rn, DRegisterList regs);
  // VSTR <Sd>, [Rn{, #+/-<imm>}], store single extension register to memory.
  void vstr(SRegister rn, MemOperand op);

  // Binds Label l to the current location in the code buffer.
  void bind(Label& l);
  // Align the cursor to specified number of bytes, `n` must be a power of 2.
  void align(uint8_t n);

 private:
  void mov(Condition c, CoreRegister rd, CoreRegister rm);
  void b(Condition c, Label& l);
  void vdup(DataSize size, QRegister qd, DRegisterLane dm);
  void vmov_f32(Condition c, SRegister sd, SRegister sm);
  void vld1(DataSize size, DRegisterList regs, MemOperand op);
  void vld1(DataSize size, DRegisterList regs, MemOperand op, CoreRegister rm);
  void vst1(DataSize size, DRegisterList regs, MemOperand op);
  void vst1(DataSize size, DRegisterList regs, MemOperand op, CoreRegister rm);
  void vst1(DataSize size, DRegisterLane dd, MemOperand op);
};

class MacroAssembler : public Assembler {
  using Assembler::Assembler;
 public:
   void f32_hardswish(QRegister sixth, QRegister three, QRegister six,
                      QRegister zero, const QRegister *accs, size_t num_accs,
                      const QRegister *tmps, size_t num_tmps);
   void Mov(CoreRegister rd, uint32_t imm);
};

class TrampolineGenerator : public MacroAssembler {
  using MacroAssembler::MacroAssembler;

 public:
  void generate(size_t args_on_stack);
 private:
  // Helper functions to check that registers match. We keep the expected value inside of x0 and return early once we
  // have a mismatch. x0 then becomes the error code, if it is 0, there are no errors.
  void CheckRegisterMatch(DRegister actual, Label& exit);
  void CheckRegisterMatch(CoreRegister actual, Label& exit);
};

}  // namespace aarch32
}  // namespace xnnpack
