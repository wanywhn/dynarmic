/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/loongarch64/a32_jitstate.h"
#include "dynarmic/backend/loongarch64/abi.h"
#include "dynarmic/backend/loongarch64/emit_arm64.h"
#include "dynarmic/backend/loongarch64/emit_context.h"
#include "dynarmic/backend/loongarch64/fpsr_manager.h"
#include "dynarmic/backend/loongarch64/reg_alloc.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"
#include "xbyak_loongarch64.h"
#include "xbyak_loongarch64_util.h"

namespace Dynarmic::Backend::LoongArch64 {

using namespace Xbyak_loongarch64::util;

template<size_t bitsize, typename EmitFn>
static void EmitTwoOp(Xbyak_loongarch64::CodeGenerator&, EmitContext& ctx, IR::Inst* inst, EmitFn emit) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteVec<bitsize>(inst);
    auto Voperand = ctx.reg_alloc.ReadVec<bitsize>(args[0]);
    RegAlloc::Realize(Vresult, Voperand);
    ctx.fpsr.Load();

    emit(Vresult, Voperand);
}

template<size_t bitsize, typename EmitFn>
static void EmitThreeOp(Xbyak_loongarch64::CodeGenerator&, EmitContext& ctx, IR::Inst* inst, EmitFn emit) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteVec<bitsize>(inst);
    auto Va = ctx.reg_alloc.ReadVec<bitsize>(args[0]);
    auto Vb = ctx.reg_alloc.ReadVec<bitsize>(args[1]);
    RegAlloc::Realize(Vresult, Va, Vb);
    ctx.fpsr.Load();

    emit(Vresult, Va, Vb);
}

template<size_t bitsize, typename EmitFn>
static void EmitFourOp(Xbyak_loongarch64::CodeGenerator&, EmitContext& ctx, IR::Inst* inst, EmitFn emit) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteVec<bitsize>(inst);
    auto Va = ctx.reg_alloc.ReadVec<bitsize>(args[0]);
    auto Vb = ctx.reg_alloc.ReadVec<bitsize>(args[1]);
    auto Vc = ctx.reg_alloc.ReadVec<bitsize>(args[2]);
    RegAlloc::Realize(Vresult, Va, Vb, Vc);
    ctx.fpsr.Load();

    emit(Vresult, Va, Vb, Vc);
}

template<size_t bitsize_from, size_t bitsize_to, typename EmitFn>
static void EmitConvert(Xbyak_loongarch64::CodeGenerator&, EmitContext& ctx, IR::Inst* inst, EmitFn emit) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vto = ctx.reg_alloc.WriteVec<bitsize_to>(inst);
    auto Vfrom = ctx.reg_alloc.ReadVec<bitsize_from>(args[0]);
    const auto rounding_mode = static_cast<FP::RoundingMode>(args[1].GetImmediateU8());
    RegAlloc::Realize(Vto, Vfrom);
    ctx.fpsr.Load();

    ASSERT(rounding_mode == ctx.FPCR().RMode());

    emit(Vto, Vfrom);
}

template<size_t bitsize_from, size_t bitsize_to, bool is_signed>
static void EmitToFixed(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Rto = ctx.reg_alloc.WriteReg<std::max<size_t>(bitsize_to, 32)>(inst);
    auto Vfrom = ctx.reg_alloc.ReadVec<bitsize_from>(args[0]);
    const size_t fbits = args[1].GetImmediateU8();
    const auto rounding_mode = static_cast<FP::RoundingMode>(args[2].GetImmediateU8());
    RegAlloc::Realize(Rto, Vfrom);
    ctx.fpsr.Load();

    if (rounding_mode == FP::RoundingMode::TowardsZero) {
        if constexpr (is_signed) {
            if constexpr (bitsize_to == 16) {
                code.FCVTZS(Rto, Vfrom, fbits + 16);
                code.srai_w(Wscratch0, Rto, 31);
                code.ADD(Rto, Rto, Wscratch0, LSR, 16);  // Round towards zero when truncating
                code.srli_w(Rto, Rto, 16);
            } else if (fbits) {
                code.FCVTZS(Rto, Vfrom, fbits);
            } else {
                code.FCVTZS(Rto, Vfrom);
            }
        } else {
            if constexpr (bitsize_to == 16) {
                code.FCVTZU(Rto, Vfrom, fbits + 16);
                code.srli_w(Rto, Rto, 16);
            } else if (fbits) {
                code.FCVTZU(Rto, Vfrom, fbits);
            } else {
                code.FCVTZU(Rto, Vfrom);
            }
        }
    } else {
        ASSERT(fbits == 0);
        ASSERT(bitsize_to != 16);
        if constexpr (is_signed) {
            switch (rounding_mode) {
            case FP::RoundingMode::ToNearest_TieEven:
                code.FCVTNS(Rto, Vfrom);
                break;
            case FP::RoundingMode::TowardsPlusInfinity:
                code.FCVTPS(Rto, Vfrom);
                break;
            case FP::RoundingMode::TowardsMinusInfinity:
                code.FCVTMS(Rto, Vfrom);
                break;
            case FP::RoundingMode::TowardsZero:
                code.FCVTZS(Rto, Vfrom);
                break;
            case FP::RoundingMode::ToNearest_TieAwayFromZero:
                code.FCVTAS(Rto, Vfrom);
                break;
            case FP::RoundingMode::ToOdd:
                ASSERT_FALSE("Unimplemented");
                break;
            default:
                ASSERT_FALSE("Invalid RoundingMode");
                break;
            }
        } else {
            switch (rounding_mode) {
            case FP::RoundingMode::ToNearest_TieEven:
                code.FCVTNU(Rto, Vfrom);
                break;
            case FP::RoundingMode::TowardsPlusInfinity:
                code.FCVTPU(Rto, Vfrom);
                break;
            case FP::RoundingMode::TowardsMinusInfinity:
                code.FCVTMU(Rto, Vfrom);
                break;
            case FP::RoundingMode::TowardsZero:
                code.FCVTZU(Rto, Vfrom);
                break;
            case FP::RoundingMode::ToNearest_TieAwayFromZero:
                code.FCVTAU(Rto, Vfrom);
                break;
            case FP::RoundingMode::ToOdd:
                ASSERT_FALSE("Unimplemented");
                break;
            default:
                ASSERT_FALSE("Invalid RoundingMode");
                break;
            }
        }
    }
}

template<size_t bitsize_from, size_t bitsize_to, typename EmitFn>
static void EmitFromFixed(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, EmitFn emit) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vto = ctx.reg_alloc.WriteVec<bitsize_to>(inst);
    auto Rfrom = ctx.reg_alloc.ReadReg<std::max<size_t>(bitsize_from, 32)>(args[0]);
    const size_t fbits = args[1].GetImmediateU8();
    const auto rounding_mode = static_cast<FP::RoundingMode>(args[2].GetImmediateU8());
    RegAlloc::Realize(Vto, Rfrom);
    ctx.fpsr.Load();

    if (rounding_mode == ctx.FPCR().RMode()) {
        emit(Vto, Rfrom, fbits);
    } else {
        FP::FPCR new_fpcr = ctx.FPCR();
        new_fpcr.RMode(rounding_mode);

        code.add_d(Wscratch0, new_fpcr.Value(), code.zero);
        code.MSR(Xbyak_loongarch64::SystemReg::FPCR, Xscratch0);

        emit(Vto, Rfrom, fbits);

        code.add_d(Wscratch0, ctx.FPCR().Value(), code.zero);
        code.MSR(Xbyak_loongarch64::SystemReg::FPCR, Xscratch0);
    }
}

template<>
void EmitIR<IR::Opcode::FPAbs16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPAbs32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Soperand) { code.FABS(Sresult, Soperand); });
}

template<>
void EmitIR<IR::Opcode::FPAbs64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Doperand) { code.FABS(Dresult, Doperand); });
}

template<>
void EmitIR<IR::Opcode::FPAdd32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Sa, auto& Sb) { code.FADD(Sresult, Sa, Sb); });
}

template<>
void EmitIR<IR::Opcode::FPAdd64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Da, auto& Db) { code.FADD(Dresult, Da, Db); });
}

    static Xbyak::Reg64 SetFpscrNzcvFromFlags(BlockOfCode& code, EmitContext& ctx) {
        ctx.reg_alloc.ScratchGpr(HostLoc::RCX);  // shifting requires use of cl
        const Xbyak::Reg64 nzcv = ctx.reg_alloc.ScratchGpr();

        //               x64 flags    ARM flags
        //               ZF  PF  CF     NZCV
        // Unordered      1   1   1     0011
        // Greater than   0   0   0     0010
        // Less than      0   0   1     1000
        // Equal          1   0   0     0110
        //
        // Thus we can take use ZF:CF as an index into an array like so:
        //  x64      ARM      ARM as x64
        // ZF:CF     NZCV     NZ-----C-------V
        //   0       0010     0000000100000000 = 0x0100
        //   1       1000     1000000000000000 = 0x8000
        //   2       0110     0100000100000000 = 0x4100
        //   3       0011     0000000100000001 = 0x0101

        code.mov(nzcv, 0x0101'4100'8000'0100);
        code.sete(cl);
        code.rcl(cl, 5);  // cl = ZF:CF:0000
        code.shr(nzcv, cl);

        return nzcv;
    }

template<size_t size>
void EmitCompare(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto flags = ctx.reg_alloc.WriteFlags(inst);
    auto Va = ctx.reg_alloc.ReadVec<size>(args[0]);
    const bool exc_on_qnan = args[2].GetImmediateU1();

    if (args[1].IsImmediate() && args[1].GetImmediateU64() == 0) {
        RegAlloc::Realize(flags, Va);
        ctx.fpsr.Load();

        if (exc_on_qnan) {
            code.FCMPE(Va, 0);
        } else {
            code.FCMP(Va, 0);
        }
    } else {
        auto Vb = ctx.reg_alloc.ReadVec<size>(args[1]);
        RegAlloc::Realize(flags, Va, Vb);
        ctx.fpsr.Load();

        if (exc_on_qnan) {
            code.FCMPE(Va, Vb);
        } else {
            code.FCMP(Va, Vb);
        }
    }
}

template<>
void EmitIR<IR::Opcode::FPCompare32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitCompare<32>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPCompare64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitCompare<64>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPDiv32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Sa, auto& Sb) { code.FDIV(Sresult, Sa, Sb); });
}

template<>
void EmitIR<IR::Opcode::FPDiv64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Da, auto& Db) { code.FDIV(Dresult, Da, Db); });
}

template<>
void EmitIR<IR::Opcode::FPMax32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Sa, auto& Sb) { code.FMAX(Sresult, Sa, Sb); });
}

template<>
void EmitIR<IR::Opcode::FPMax64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Da, auto& Db) { code.FMAX(Dresult, Da, Db); });
}

template<>
void EmitIR<IR::Opcode::FPMaxNumeric32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Sa, auto& Sb) { code.FMAXNM(Sresult, Sa, Sb); });
}

template<>
void EmitIR<IR::Opcode::FPMaxNumeric64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Da, auto& Db) { code.FMAXNM(Dresult, Da, Db); });
}

template<>
void EmitIR<IR::Opcode::FPMin32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Sa, auto& Sb) { code.FMIN(Sresult, Sa, Sb); });
}

template<>
void EmitIR<IR::Opcode::FPMin64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Da, auto& Db) { code.FMIN(Dresult, Da, Db); });
}

template<>
void EmitIR<IR::Opcode::FPMinNumeric32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Sa, auto& Sb) { code.FMINNM(Sresult, Sa, Sb); });
}

template<>
void EmitIR<IR::Opcode::FPMinNumeric64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Da, auto& Db) { code.FMINNM(Dresult, Da, Db); });
}

template<>
void EmitIR<IR::Opcode::FPMul32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Sa, auto& Sb) { code.FMUL(Sresult, Sa, Sb); });
}

template<>
void EmitIR<IR::Opcode::FPMul64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Da, auto& Db) { code.FMUL(Dresult, Da, Db); });
}

template<>
void EmitIR<IR::Opcode::FPMulAdd16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPMulAdd32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFourOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Sa, auto& S1, auto& S2) { code.FMADD(Sresult, S1, S2, Sa); });
}

template<>
void EmitIR<IR::Opcode::FPMulAdd64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFourOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Da, auto& D1, auto& D2) { code.FMADD(Dresult, D1, D2, Da); });
}

template<>
void EmitIR<IR::Opcode::FPMulX32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Sa, auto& Sb) { code.FMULX(Sresult, Sa, Sb); });
}

template<>
void EmitIR<IR::Opcode::FPMulX64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Da, auto& Db) { code.FMULX(Dresult, Da, Db); });
}

template<>
void EmitIR<IR::Opcode::FPNeg16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPNeg32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Soperand) { code.FNEG(Sresult, Soperand); });
}

template<>
void EmitIR<IR::Opcode::FPNeg64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Doperand) { code.FNEG(Dresult, Doperand); });
}

template<>
void EmitIR<IR::Opcode::FPRecipEstimate16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPRecipEstimate32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Soperand) { code.FRECPE(Sresult, Soperand); });
}

template<>
void EmitIR<IR::Opcode::FPRecipEstimate64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Doperand) { code.FRECPE(Dresult, Doperand); });
}

template<>
void EmitIR<IR::Opcode::FPRecipExponent16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPRecipExponent32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Soperand) { code.FRECPX(Sresult, Soperand); });
}

template<>
void EmitIR<IR::Opcode::FPRecipExponent64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Doperand) { code.FRECPX(Dresult, Doperand); });
}

template<>
void EmitIR<IR::Opcode::FPRecipStepFused16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPRecipStepFused32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Sa, auto& Sb) { code.FRECPS(Sresult, Sa, Sb); });
}

template<>
void EmitIR<IR::Opcode::FPRecipStepFused64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Da, auto& Db) { code.FRECPS(Dresult, Da, Db); });
}

template<>
void EmitIR<IR::Opcode::FPRoundInt16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPRoundInt32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto rounding_mode = static_cast<FP::RoundingMode>(inst->GetArg(1).GetU8());
    const bool exact = inst->GetArg(2).GetU1();

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Sresult = ctx.reg_alloc.WriteS(inst);
    auto Soperand = ctx.reg_alloc.ReadS(args[0]);
    RegAlloc::Realize(Sresult, Soperand);
    ctx.fpsr.Load();

    if (exact) {
        ASSERT(ctx.FPCR().RMode() == rounding_mode);
        code.FRINTX(Sresult, Soperand);
    } else {
        switch (rounding_mode) {
        case FP::RoundingMode::ToNearest_TieEven:
            code.FRINTN(Sresult, Soperand);
            break;
        case FP::RoundingMode::TowardsPlusInfinity:
            code.FRINTP(Sresult, Soperand);
            break;
        case FP::RoundingMode::TowardsMinusInfinity:
            code.FRINTM(Sresult, Soperand);
            break;
        case FP::RoundingMode::TowardsZero:
            code.FRINTZ(Sresult, Soperand);
            break;
        case FP::RoundingMode::ToNearest_TieAwayFromZero:
            code.FRINTA(Sresult, Soperand);
            break;
        default:
            ASSERT_FALSE("Invalid RoundingMode");
        }
    }
}

template<>
void EmitIR<IR::Opcode::FPRoundInt64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto rounding_mode = static_cast<FP::RoundingMode>(inst->GetArg(1).GetU8());
    const bool exact = inst->GetArg(2).GetU1();

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Dresult = ctx.reg_alloc.WriteD(inst);
    auto Doperand = ctx.reg_alloc.ReadD(args[0]);
    RegAlloc::Realize(Dresult, Doperand);
    ctx.fpsr.Load();

    if (exact) {
        ASSERT(ctx.FPCR().RMode() == rounding_mode);
        code.FRINTX(Dresult, Doperand);
    } else {
        switch (rounding_mode) {
        case FP::RoundingMode::ToNearest_TieEven:
            code.FRINTN(Dresult, Doperand);
            break;
        case FP::RoundingMode::TowardsPlusInfinity:
            code.FRINTP(Dresult, Doperand);
            break;
        case FP::RoundingMode::TowardsMinusInfinity:
            code.FRINTM(Dresult, Doperand);
            break;
        case FP::RoundingMode::TowardsZero:
            code.FRINTZ(Dresult, Doperand);
            break;
        case FP::RoundingMode::ToNearest_TieAwayFromZero:
            code.FRINTA(Dresult, Doperand);
            break;
        default:
            ASSERT_FALSE("Invalid RoundingMode");
        }
    }
}

template<>
void EmitIR<IR::Opcode::FPRSqrtEstimate16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPRSqrtEstimate32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Soperand) { code.FRSQRTE(Sresult, Soperand); });
}

template<>
void EmitIR<IR::Opcode::FPRSqrtEstimate64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Doperand) { code.FRSQRTE(Dresult, Doperand); });
}

template<>
void EmitIR<IR::Opcode::FPRSqrtStepFused16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPRSqrtStepFused32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Sa, auto& Sb) { code.FRSQRTS(Sresult, Sa, Sb); });
}

template<>
void EmitIR<IR::Opcode::FPRSqrtStepFused64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Da, auto& Db) { code.FRSQRTS(Dresult, Da, Db); });
}

template<>
void EmitIR<IR::Opcode::FPSqrt32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Soperand) { code.FSQRT(Sresult, Soperand); });
}

template<>
void EmitIR<IR::Opcode::FPSqrt64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Doperand) { code.FSQRT(Dresult, Doperand); });
}

template<>
void EmitIR<IR::Opcode::FPSub32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Sa, auto& Sb) { code.FSUB(Sresult, Sa, Sb); });
}

template<>
void EmitIR<IR::Opcode::FPSub64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Da, auto& Db) { code.FSUB(Dresult, Da, Db); });
}

template<>
void EmitIR<IR::Opcode::FPHalfToDouble>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitConvert<16, 64>(code, ctx, inst, [&](auto& Dto, auto& Hfrom) { code.FCVT(Dto, Hfrom); });
}

template<>
void EmitIR<IR::Opcode::FPHalfToSingle>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitConvert<16, 32>(code, ctx, inst, [&](auto& Sto, auto& Hfrom) { code.FCVT(Sto, Hfrom); });
}

template<>
void EmitIR<IR::Opcode::FPSingleToDouble>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitConvert<32, 64>(code, ctx, inst, [&](auto& Dto, auto& Sfrom) { code.FCVT(Dto, Sfrom); });
}

template<>
void EmitIR<IR::Opcode::FPSingleToHalf>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitConvert<32, 16>(code, ctx, inst, [&](auto& Hto, auto& Sfrom) { code.FCVT(Hto, Sfrom); });
}

template<>
void EmitIR<IR::Opcode::FPDoubleToHalf>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitConvert<64, 16>(code, ctx, inst, [&](auto& Hto, auto& Dfrom) { code.FCVT(Hto, Dfrom); });
}

template<>
void EmitIR<IR::Opcode::FPDoubleToSingle>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto rounding_mode = static_cast<FP::RoundingMode>(inst->GetArg(1).GetU8());

    if (rounding_mode == FP::RoundingMode::ToOdd) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Sto = ctx.reg_alloc.WriteS(inst);
        auto Dfrom = ctx.reg_alloc.ReadD(args[0]);
        RegAlloc::Realize(Sto, Dfrom);
        ctx.fpsr.Load();

        code.FCVTXN(Sto, Dfrom);

        return;
    }

    EmitConvert<64, 32>(code, ctx, inst, [&](auto& Sto, auto& Dfrom) { code.FCVT(Sto, Dfrom); });
}

template<>
void EmitIR<IR::Opcode::FPDoubleToFixedS16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitToFixed<64, 16, true>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPDoubleToFixedS32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitToFixed<64, 32, true>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPDoubleToFixedS64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    // TODO: Consider fpr source
    EmitToFixed<64, 64, true>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPDoubleToFixedU16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitToFixed<64, 16, false>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPDoubleToFixedU32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitToFixed<64, 32, false>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPDoubleToFixedU64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    // TODO: Consider fpr source
    EmitToFixed<64, 64, false>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPHalfToFixedS16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPHalfToFixedS32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPHalfToFixedS64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPHalfToFixedU16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPHalfToFixedU32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPHalfToFixedU64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPSingleToFixedS16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitToFixed<32, 16, true>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPSingleToFixedS32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    // TODO: Consider fpr source
    EmitToFixed<32, 32, true>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPSingleToFixedS64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitToFixed<32, 64, true>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPSingleToFixedU16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitToFixed<32, 16, false>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPSingleToFixedU32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    // TODO: Consider fpr source
    EmitToFixed<32, 32, false>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPSingleToFixedU64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitToFixed<32, 64, false>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPFixedU16ToSingle>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<16, 32>(code, ctx, inst, [&](auto& Sto, auto& Wfrom, u8 fbits) {
        code.slli_w(Wscratch0, Wfrom, 16);
        code.UCVTF(Sto, Wscratch0, fbits + 16);
    });
}

template<>
void EmitIR<IR::Opcode::FPFixedS16ToSingle>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<16, 32>(code, ctx, inst, [&](auto& Sto, auto& Wfrom, u8 fbits) {
        code.slli_w(Wscratch0, Wfrom, 16);
        code.SCVTF(Sto, Wscratch0, fbits + 16);
    });
}

template<>
void EmitIR<IR::Opcode::FPFixedU16ToDouble>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<16, 64>(code, ctx, inst, [&](auto& Dto, auto& Wfrom, u8 fbits) {
        code.slli_w(Wscratch0, Wfrom, 16);
        code.UCVTF(Dto, Wscratch0, fbits + 16);
    });
}

template<>
void EmitIR<IR::Opcode::FPFixedS16ToDouble>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<16, 64>(code, ctx, inst, [&](auto& Dto, auto& Wfrom, u8 fbits) {
        code.slli_w(Wscratch0, Wfrom, 16);
        code.SCVTF(Dto, Wscratch0, fbits + 16);
    });
}

template<>
void EmitIR<IR::Opcode::FPFixedU32ToSingle>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    // TODO: Consider fpr source
    EmitFromFixed<32, 32>(code, ctx, inst, [&](auto& Sto, auto& Wfrom, u8 fbits) { fbits ? code.UCVTF(Sto, Wfrom, fbits) : code.UCVTF(Sto, Wfrom); });
}

template<>
void EmitIR<IR::Opcode::FPFixedS32ToSingle>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    // TODO: Consider fpr source
    EmitFromFixed<32, 32>(code, ctx, inst, [&](auto& Sto, auto& Wfrom, u8 fbits) { fbits ? code.SCVTF(Sto, Wfrom, fbits) : code.SCVTF(Sto, Wfrom); });
}

template<>
void EmitIR<IR::Opcode::FPFixedU32ToDouble>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<32, 64>(code, ctx, inst, [&](auto& Dto, auto& Wfrom, u8 fbits) { fbits ? code.UCVTF(Dto, Wfrom, fbits) : code.UCVTF(Dto, Wfrom); });
}

template<>
void EmitIR<IR::Opcode::FPFixedS32ToDouble>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<32, 64>(code, ctx, inst, [&](auto& Dto, auto& Wfrom, u8 fbits) { fbits ? code.SCVTF(Dto, Wfrom, fbits) : code.SCVTF(Dto, Wfrom); });
}

template<>
void EmitIR<IR::Opcode::FPFixedU64ToDouble>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    // TODO: Consider fpr source
    EmitFromFixed<64, 64>(code, ctx, inst, [&](auto& Dto, auto& Xfrom, u8 fbits) { fbits ? code.UCVTF(Dto, Xfrom, fbits) : code.UCVTF(Dto, Xfrom); });
}

template<>
void EmitIR<IR::Opcode::FPFixedU64ToSingle>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<64, 32>(code, ctx, inst, [&](auto& Sto, auto& Xfrom, u8 fbits) { fbits ? code.UCVTF(Sto, Xfrom, fbits) : code.UCVTF(Sto, Xfrom); });
}

template<>
void EmitIR<IR::Opcode::FPFixedS64ToDouble>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    // TODO: Consider fpr source
    EmitFromFixed<64, 64>(code, ctx, inst, [&](auto& Dto, auto& Xfrom, u8 fbits) { fbits ? code.SCVTF(Dto, Xfrom, fbits) : code.SCVTF(Dto, Xfrom); });
}

template<>
void EmitIR<IR::Opcode::FPFixedS64ToSingle>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<64, 32>(code, ctx, inst, [&](auto& Sto, auto& Xfrom, u8 fbits) { fbits ? code.SCVTF(Sto, Xfrom, fbits) : code.SCVTF(Sto, Xfrom); });
}

}  // namespace Dynarmic::Backend::LoongArch64
