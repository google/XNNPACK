// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/allocator.h>
#include <xnnpack/assembler.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace xnnpack {
namespace aarch32 {

using namespace xnnpack;

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

  const DRegisterLane operator[](std::size_t pos) const {
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
  ConsecutiveRegisterList(RegType start)
      : ConsecutiveRegisterList(start, start) {}

  RegType start;
  uint8_t length;
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
  int32_t u() { return offset_ >= 0; }
  int32_t p() { return mode_ != AddressingMode::kPostIndexed; }
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
  const MemOperand operator[](MemOperand op) const { return op; }
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

  Assembler& add(CoreRegister rn, CoreRegister rm) { return add(rn, rn, rm); }
  Assembler& add(CoreRegister rd, CoreRegister rn, CoreRegister rm);
  // Only support uint8_t immediates for now, it simplifies encoding.
  Assembler& add(CoreRegister rd, CoreRegister rn, uint8_t imm);
  Assembler& adds(CoreRegister rd, CoreRegister rn, uint8_t imm);
  Assembler& and_(CoreRegister rd, CoreRegister rn, uint8_t imm);
  Assembler& b(Label& l) { return b(kAL, l); }
  Assembler& beq(Label& l) { return b(kEQ, l); }
  Assembler& bne(Label& l) { return b(kNE, l); }
  Assembler& bhi(Label& l) { return b(kHI, l); }
  Assembler& bhs(Label& l) { return b(kHS, l); }
  Assembler& blo(Label& l) { return b(kLO, l); }
  Assembler& bic(CoreRegister rd, CoreRegister rn, uint8_t imm);
  Assembler& bx(CoreRegister rm);
  // Cmp supports a subset of uint32_t offsets, see "A5.2.4 Modified immediate
  // constants in ARM instructions", for simplicity we start with uint8_t, which
  // is fully representation using a "rotation" of 0.
  Assembler& cmp(CoreRegister rn, uint8_t imm);
  Assembler& cmp(CoreRegister rn, CoreRegister rm);
  Assembler& ldr(CoreRegister rt, MemOperand operand, int32_t offset);
  Assembler& ldr(CoreRegister rt, MemOperand operand);
  // LDRD <Rt>, <Rt2>, [<Rn>{, #+/-<imm>}].
  Assembler& ldrd(CoreRegister rt, CoreRegister rt2, MemOperand op);
  Assembler& mov(CoreRegister rd, CoreRegister rm);
  Assembler& moveq(CoreRegister rd, CoreRegister rm) { return mov(kEQ, rd, rm); }
  Assembler& movlo(CoreRegister rd, CoreRegister rm) { return mov(kLO, rd, rm); }
  Assembler& movls(CoreRegister rd, CoreRegister rm) { return mov(kLS, rd, rm); }
  Assembler& nop();
  Assembler& pld(MemOperand operand);
  Assembler& pop(CoreRegisterList regs);
  Assembler& push(CoreRegisterList regs);
  Assembler& str(CoreRegister rt, MemOperand op);
  Assembler& sub(CoreRegister rd, CoreRegister rn, uint8_t imm);
  Assembler& sub(CoreRegister rd, CoreRegister rn, CoreRegister rm);
  // Only support uint8_t immediates for now, it simplifies encoding.
  Assembler& subs(CoreRegister rd, CoreRegister rn, uint8_t imm);
  Assembler& tst(CoreRegister rn, uint8_t imm);

  // SIMD instructions.
  Assembler& vcmpe_f32(SRegister sd, SRegister sm);
  Assembler& vcvt_f32_s32(QRegister qd, QRegister qm);
  Assembler& vcvt_s32_f32(QRegister qd, QRegister qm);
  Assembler& vcvtn_s32_f32(QRegister qd, QRegister qm);
  Assembler& vdup_8(QRegister qd, DRegisterLane dm) { return vdup(k8, qd, dm); }
  Assembler& vdup_16(QRegister qd, DRegisterLane dm) { return vdup(k16, qd, dm); }
  Assembler& vdup_32(QRegister qd, DRegisterLane dm) { return vdup(k32, qd, dm); }
  Assembler& vext_8(QRegister qd, QRegister qn, QRegister qm, uint8_t imm4);
  // VLD1.8 <list>, [<Rn>]{!} (multiple single elements).
  Assembler& vld1_8(DRegisterList regs, MemOperand op) { return vld1(k8, regs, op); }
  Assembler& vld1_8(DRegisterList regs, MemOperand op, CoreRegister rm) { return vld1(k8, regs, op, rm); }
  Assembler& vld1_8(QRegisterList regs, MemOperand op) { return vld1(k8, static_cast<DRegisterList>(regs), op); }
  // VLD1.32 <list>, [<Rn>]{!} (multiple single elements).
  Assembler& vld1_32(DRegisterList regs, MemOperand op) { return vld1(k32, regs, op); }
  Assembler& vld1_32(QRegisterList regs, MemOperand op) { return vld1(k32, static_cast<DRegisterList>(regs), op); }
  // VLD1.32 <list>, [<Rn>]{!} (single element to one lane).
  Assembler& vld1_32(DRegisterLane dd, MemOperand op);
  // VLD1.32 <list>, [<Rn>]{!} (single element to all lanes).
  // We cannot differentiate the register list in C++ syntax, so use an instruction name similar to AArch64 LD1R.
  Assembler& vld1r_32(DRegisterList regs, MemOperand op);
  // VLDM <Rn>{!}, <list>. {!} is indicated by setting `wb` argument.
  Assembler& vldm(CoreRegister rn, SRegisterList regs, bool wb);
  Assembler& vldm(CoreRegister rn, DRegisterList regs, bool wb);
  Assembler& vldm(CoreRegister rn, SRegisterList regs) { return vldm(rn, regs, false); }
  Assembler& vldm(CoreRegister rn, DRegisterList regs) { return vldm(rn, regs, false); }
  Assembler& vldr(SRegister sd, MemOperand op);
  Assembler& vldr(DRegister dd, MemOperand op);
  Assembler& vmax_f32(QRegister qd, QRegister qn, QRegister qm);
  Assembler& vmax_s8(QRegister qd, QRegister qn, QRegister qm);
  Assembler& vmin_f32(QRegister qd, QRegister qn, QRegister qm);
  Assembler& vmin_s8(QRegister qd, QRegister qn, QRegister qm);
  // VMLA.F32 <Sd>, <Sn>, <Sm>
  Assembler& vmla_f32(SRegister sd, SRegister sn, SRegister sm);
  // VMLA.F32 <Qd>, <Qn>, <Dm[x]>
  Assembler& vmla_f32(QRegister qd, QRegister qn, DRegisterLane dm);
  // VMLAL.S16 <Qd>, <Dn>, <Dm[x]>
  Assembler& vmlal_s16(QRegister qd, DRegister dn, DRegisterLane dm);
  // VMOV.F32 <Sd>, <Sm>; encoding A2.
  Assembler& vmov(SRegister sd, SRegister sm);
  // VMOV <Dm>, <Rt>, <Rt2>; encoding A1.
  Assembler& vmov(DRegister dm, CoreRegister rt, CoreRegister rt2);
  // VMOV <Dd>, <Dm>; encoding A1.
  Assembler& vmov(DRegister dd, DRegister dm);
  // VMOV <Qd>, <Qm>; encoding A1.
  Assembler& vmov(QRegister qd, QRegister qm);
  // VMOV_F32 <Sd>, <Sm>
  Assembler& vmov_f32(SRegister sd, SRegister sm) { return vmov_f32(kAL, sd, sm); }
  Assembler& vmovpl_f32(SRegister sd, SRegister sm) { return vmov_f32(kPL, sd, sm); }
  Assembler& vmovmi_f32(SRegister sd, SRegister sm) { return vmov_f32(kMI, sd, sm); }
  // VMOV_F64 <Dd>, <Dm>
  Assembler& vmov_f64(DRegister dd, DRegister dm);
  // VMOVL.S8 <Qd>, <Dm>
  Assembler& vmovl_s8(QRegister qd, DRegister dm);
  Assembler& vmrs(CoreRegister rt, SpecialFPRegister spec_reg);
  Assembler& vmul_f32(QRegister qd, QRegister qn, QRegister qm);
  Assembler& vpop(DRegisterList regs);
  Assembler& vpush(DRegisterList regs);
  Assembler& vpush(SRegisterList regs);
  Assembler& vqadd_s16(QRegister qd, QRegister qn, QRegister qm);
  Assembler& vqdmulh_s32(QRegister qd, QRegister qn, DRegisterLane dm);
  Assembler& vqmovn_s16(DRegister dd, QRegister qm);
  Assembler& vqmovn_s32(DRegister dd, QRegister qm);
  Assembler& vqshl_s32(QRegister qd, QRegister qm, QRegister qn);
  Assembler& vrshl_s32(QRegister qd, QRegister qm, QRegister qn);
  Assembler& vsdot_s8(QRegister qd, QRegister qn, DRegisterLane dm);
  // VST1.8 <list>, [<Rn>]{!} (multiple single elements).
  Assembler& vst1_8(DRegisterList regs, MemOperand op) { return vst1(k8, regs, op); }
  // VST1.8 <list>, [<Rn>]{!}, <Rm> (multiple single elements).
  Assembler& vst1_8(DRegisterList regs, MemOperand op, CoreRegister rm) { return vst1(k8, regs, op, rm); }
  // VST1.8 <list>, [<Rn>]{!} (single element form one lane).
  Assembler& vst1_8(DRegisterLane dd, MemOperand op) { return vst1(k8, dd, op); }
  // VST1.16 <list>, [<Rn>]{!} (multiple single elements).
  Assembler& vst1_16(DRegisterList regs, MemOperand op) { return vst1(k16, regs, op); }
  // VST1.16 <list>, [<Rn>]{!}, <Rm> (multiple single elements).
  Assembler& vst1_16(DRegisterList regs, MemOperand op, CoreRegister rm) { return vst1(k16, regs, op, rm); }
  // VST1.16 <list>, [<Rn>]{!} (single element form one lane).
  Assembler& vst1_16(DRegisterLane dd, MemOperand op) { return vst1(k16, dd, op); }
  // VST1.32 <list>, [<Rn>]{!} (multiple single elements).
  Assembler& vst1_32(DRegisterList regs, MemOperand op) { return vst1(k32, regs, op); }
  // VST1.32 <list>, [<Rn>]{!}, <Rm> (multiple single elements).
  Assembler& vst1_32(DRegisterList regs, MemOperand op, CoreRegister rm) { return vst1(k32, regs, op, rm); }
  // VST1.32 <list>, [<Rn>]{!} (single element form one lane).
  Assembler& vst1_32(DRegisterLane dd, MemOperand op) { return vst1(k32, dd, op); }
  // VSTM <Rn>{!}, <list>, consecutive 64-bit registers.
  Assembler& vstm(CoreRegister rn, DRegisterList regs, bool wb);
  // VSTR <Sd>, [Rn{, #+/-<imm>}], store single extension register to memory.
  Assembler& vstr(SRegister rn, MemOperand op);

  // Binds Label l to the current location in the code buffer.
  Assembler& bind(Label& l);
  // Align the cursor to specified number of bytes, `n` must be a power of 2.
  Assembler& align(uint8_t n);

 private:
  // Emits a 32-bit value to the code buffer.
  Assembler& emit32(uint32_t value);
  Assembler& mov(Condition c, CoreRegister rd, CoreRegister rm);
  Assembler& b(Condition c, Label& l);
  Assembler& vdup(DataSize size, QRegister qd, DRegisterLane dm);
  Assembler& vmov_f32(Condition c, SRegister sd, SRegister sm);
  Assembler& vld1(DataSize size, DRegisterList regs, MemOperand op);
  Assembler& vld1(DataSize size, DRegisterList regs, MemOperand op, CoreRegister rm);
  Assembler& vst1(DataSize size, DRegisterList regs, MemOperand op);
  Assembler& vst1(DataSize size, DRegisterList regs, MemOperand op, CoreRegister rm);
  Assembler& vst1(DataSize size, DRegisterLane dd, MemOperand op);
};

}  // namespace aarch32
}  // namespace xnnpack
