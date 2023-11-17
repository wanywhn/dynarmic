/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <cstddef>

#include <fmt/ostream.h>

#include "dynarmic/backend/loongarch64/a32_jitstate.h"
#include "dynarmic/backend/loongarch64/abi.h"
#include "dynarmic/backend/loongarch64/emit_arm64.h"
#include "dynarmic/backend/loongarch64/emit_context.h"
#include "dynarmic/backend/loongarch64/reg_alloc.h"
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

    auto Rresult = ctx.reg_alloc.WriteReg<bitsize>(inst);
    auto Roperand = ctx.reg_alloc.ReadReg<bitsize>(args[0]);
    RegAlloc::Realize(Rresult, Roperand);

    emit(Rresult, Roperand);
}

template<size_t bitsize, typename EmitFn>
static void EmitThreeOp(Xbyak_loongarch64::CodeGenerator&, EmitContext& ctx, IR::Inst* inst, EmitFn emit) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Rresult = ctx.reg_alloc.WriteReg<bitsize>(inst);
    auto Ra = ctx.reg_alloc.ReadReg<bitsize>(args[0]);
    auto Rb = ctx.reg_alloc.ReadReg<bitsize>(args[1]);
    RegAlloc::Realize(Rresult, Ra, Rb);

    emit(Rresult, Ra, Rb);
}

template<>
void EmitIR<IR::Opcode::Pack2x32To1x64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Wlo = ctx.reg_alloc.ReadW(args[0]);
    auto Whi = ctx.reg_alloc.ReadW(args[1]);
    auto Xresult = ctx.reg_alloc.WriteX(inst);
    RegAlloc::Realize(Wlo, Whi, Xresult);

    code.add_d(Xresult->toW(), Wlo, code.zero);  // TODO: Move eliminiation
    code.BFI(Xresult, Whi->toX(), 32, 32);
}

template<>
void EmitIR<IR::Opcode::Pack2x64To1x128>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    if (args[0].IsInGpr() && args[1].IsInGpr()) {
        auto Xlo = ctx.reg_alloc.ReadX(args[0]);
        auto Xhi = ctx.reg_alloc.ReadX(args[1]);
        auto Qresult = ctx.reg_alloc.WriteQ(inst);
        RegAlloc::Realize(Xlo, Xhi, Qresult);

        code.FMOV(Qresult->toD(), Xlo);
        code.add_d(Xbyak_loongarch64::VRegSelector{Qresult->getIdx()}.D()[1], Xhi, code.zero);
    } else if (args[0].IsInGpr()) {
        auto Xlo = ctx.reg_alloc.ReadX(args[0]);
        auto Dhi = ctx.reg_alloc.ReadD(args[1]);
        auto Qresult = ctx.reg_alloc.WriteQ(inst);
        RegAlloc::Realize(Xlo, Dhi, Qresult);

        code.FMOV(Qresult->toD(), Xlo);
        code.add_d(Xbyak_loongarch64::VRegSelector{Qresult->getIdx()}.D()[1], Xbyak_loongarch64::VRegSelector{Dhi->getIdx()}.D()[0], code.zero);
    } else if (args[1].IsInGpr()) {
        auto Dlo = ctx.reg_alloc.ReadD(args[0]);
        auto Xhi = ctx.reg_alloc.ReadX(args[1]);
        auto Qresult = ctx.reg_alloc.WriteQ(inst);
        RegAlloc::Realize(Dlo, Xhi, Qresult);

        code.FMOV(Qresult->toD(), Dlo);  // TODO: Move eliminiation
        code.add_d(Xbyak_loongarch64::VRegSelector{Qresult->getIdx()))}.D()[1], Xhi, code.zero);
    } else {
        auto Dlo = ctx.reg_alloc.ReadD(args[0]);
        auto Dhi = ctx.reg_alloc.ReadD(args[1]);
        auto Qresult = ctx.reg_alloc.WriteQ(inst);
        RegAlloc::Realize(Dlo, Dhi, Qresult);

        code.FMOV(Qresult->toD(), Dlo);  // TODO: Move eliminiation
        code.add_d(Xbyak_loongarch64::VRegSelector{Qresult->getIdx()}.D()[1], Xbyak_loongarch64::VRegSelector{Dhi->getIdx()}.D()[0], code.zero);
    }
}

template<>
void EmitIR<IR::Opcode::LeastSignificantWord>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Wresult = ctx.reg_alloc.WriteW(inst);
    auto Xoperand = ctx.reg_alloc.ReadX(args[0]);
    RegAlloc::Realize(Wresult, Xoperand);

    code.add_d(Wresult, Xoperand->toW(), code.zero);  // TODO: Zext elimination
}

template<>
void EmitIR<IR::Opcode::LeastSignificantHalf>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Wresult = ctx.reg_alloc.WriteW(inst);
    auto Woperand = ctx.reg_alloc.ReadW(args[0]);
    RegAlloc::Realize(Wresult, Woperand);

    code.UXTH(Wresult, Woperand);  // TODO: Zext elimination
}

template<>
void EmitIR<IR::Opcode::LeastSignificantByte>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Wresult = ctx.reg_alloc.WriteW(inst);
    auto Woperand = ctx.reg_alloc.ReadW(args[0]);
    RegAlloc::Realize(Wresult, Woperand);

    code.UXTB(Wresult, Woperand);  // TODO: Zext elimination
}

template<>
void EmitIR<IR::Opcode::MostSignificantWord>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto carry_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetCarryFromOp);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Wresult = ctx.reg_alloc.WriteW(inst);
    auto Xoperand = ctx.reg_alloc.ReadX(args[0]);
    RegAlloc::Realize(Wresult, Xoperand);

    code.LSR(Wresult->toX(), Xoperand, 32);

    if (carry_inst) {
        auto Wcarry = ctx.reg_alloc.WriteW(carry_inst);
        RegAlloc::Realize(Wcarry);

        code.LSR(Wcarry, Xoperand->toW(), 31 - 29);
        code.andi(Wcarry, Wcarry, 1 << 29);
    }
}

template<>
void EmitIR<IR::Opcode::MostSignificantBit>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Wresult = ctx.reg_alloc.WriteW(inst);
    auto Woperand = ctx.reg_alloc.ReadW(args[0]);
    RegAlloc::Realize(Wresult, Woperand);

    code.LSR(Wresult, Woperand, 31);
}

template<>
void EmitIR<IR::Opcode::IsZero32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Wresult = ctx.reg_alloc.WriteW(inst);
    auto Woperand = ctx.reg_alloc.ReadW(args[0]);
    RegAlloc::Realize(Wresult, Woperand);
    ctx.reg_alloc.SpillFlags();

    code.CMP(Woperand, 0);
    code.CSET(Wresult, EQ);
}

template<>
void EmitIR<IR::Opcode::IsZero64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Wresult = ctx.reg_alloc.WriteW(inst);
    auto Xoperand = ctx.reg_alloc.ReadX(args[0]);
    RegAlloc::Realize(Wresult, Xoperand);
    ctx.reg_alloc.SpillFlags();

    code.CMP(Xoperand, 0);
    code.CSET(Wresult, EQ);
}

template<>
void EmitIR<IR::Opcode::TestBit>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xresult = ctx.reg_alloc.WriteX(inst);
    auto Xoperand = ctx.reg_alloc.ReadX(args[0]);
    RegAlloc::Realize(Xresult, Xoperand);
    ASSERT(args[1].IsImmediate());
    ASSERT(args[1].GetImmediateU8() < 64);

    code.UBFX(Xresult, Xoperand, args[1].GetImmediateU8(), 1);
}

template<>
void EmitIR<IR::Opcode::ConditionalSelect32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const IR::Cond cond = args[0].GetImmediateCond();
    auto Wresult = ctx.reg_alloc.WriteW(inst);
    auto Wthen = ctx.reg_alloc.ReadW(args[1]);
    auto Welse = ctx.reg_alloc.ReadW(args[2]);
    RegAlloc::Realize(Wresult, Wthen, Welse);
    ctx.reg_alloc.SpillFlags();

    // TODO: FSEL for fprs

    code.ld_d(Wscratch0, Xstate, ctx.conf.state_nzcv_offset);
    code.MSR(Xbyak_loongarch64::SystemReg::NZCV, Xscratch0);
    code.CSEL(Wresult, Wthen, Welse, static_cast<Xbyak_loongarch64::Cond>(cond));
}

template<>
void EmitIR<IR::Opcode::ConditionalSelect64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const IR::Cond cond = args[0].GetImmediateCond();
    auto Xresult = ctx.reg_alloc.WriteX(inst);
    auto Xthen = ctx.reg_alloc.ReadX(args[1]);
    auto Xelse = ctx.reg_alloc.ReadX(args[2]);
    RegAlloc::Realize(Xresult, Xthen, Xelse);
    ctx.reg_alloc.SpillFlags();

    // TODO: FSEL for fprs

    code.ld_d(Wscratch0, Xstate, ctx.conf.state_nzcv_offset);
    code.MSR(Xbyak_loongarch64::SystemReg::NZCV, Xscratch0);
    code.CSEL(Xresult, Xthen, Xelse, static_cast<Xbyak_loongarch64::Cond>(cond));
}

template<>
void EmitIR<IR::Opcode::ConditionalSelectNZCV>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitIR<IR::Opcode::ConditionalSelect32>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::LogicalShiftLeft32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto carry_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetCarryFromOp);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto& operand_arg = args[0];
    auto& shift_arg = args[1];
    auto& carry_arg = args[2];

    if (!carry_inst) {
        if (shift_arg.IsImmediate()) {
            const u8 shift = shift_arg.GetImmediateU8();
            auto Wresult = ctx.reg_alloc.WriteW(inst);
            auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
            RegAlloc::Realize(Wresult, Woperand);

            if (shift <= 31) {
                code.LSL(Wresult, Woperand, shift);
            } else {
                code.add_d(Wresult, WZR, code.zero);
            }
        } else {
            auto Wresult = ctx.reg_alloc.WriteW(inst);
            auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
            auto Wshift = ctx.reg_alloc.ReadW(shift_arg);
            RegAlloc::Realize(Wresult, Woperand, Wshift);
            ctx.reg_alloc.SpillFlags();

            code.andi(Wscratch0, Wshift, 0xff);
            code.LSL(Wresult, Woperand, Wscratch0);
            code.CMP(Wscratch0, 32);
            code.CSEL(Wresult, Wresult, WZR, LT);
        }
    } else {
        if (shift_arg.IsImmediate() && shift_arg.GetImmediateU8() == 0) {
            ctx.reg_alloc.DefineAsExisting(carry_inst, carry_arg);
            ctx.reg_alloc.DefineAsExisting(inst, operand_arg);
        } else if (shift_arg.IsImmediate()) {
            // TODO: Use RMIF
            const u8 shift = shift_arg.GetImmediateU8();

            if (shift < 32) {
                auto Wresult = ctx.reg_alloc.WriteW(inst);
                auto Wcarry_out = ctx.reg_alloc.WriteW(carry_inst);
                auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
                RegAlloc::Realize(Wresult, Wcarry_out, Woperand);

                code.UBFX(Wcarry_out, Woperand, 32 - shift, 1);
                code.LSL(Wcarry_out, Wcarry_out, 29);
                code.LSL(Wresult, Woperand, shift);
            } else if (shift > 32) {
                auto Wresult = ctx.reg_alloc.WriteW(inst);
                auto Wcarry_out = ctx.reg_alloc.WriteW(carry_inst);
                RegAlloc::Realize(Wresult, Wcarry_out);

                code.add_d(Wresult, WZR, code.zero);
                code.add_d(Wcarry_out, WZR, code.zero);
            } else {
                auto Wresult = ctx.reg_alloc.WriteW(inst);
                auto Wcarry_out = ctx.reg_alloc.WriteW(carry_inst);
                auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
                RegAlloc::Realize(Wresult, Wcarry_out, Woperand);

                code.UBFIZ(Wcarry_out, Woperand, 29, 1);
                code.add_d(Wresult, WZR, code.zero);
            }
        } else {
            auto Wresult = ctx.reg_alloc.WriteW(inst);
            auto Wcarry_out = ctx.reg_alloc.WriteW(carry_inst);
            auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
            auto Wshift = ctx.reg_alloc.ReadW(shift_arg);
            auto Wcarry_in = ctx.reg_alloc.ReadW(carry_arg);
            if (carry_arg.IsImmediate()) {
                RegAlloc::Realize(Wresult, Wcarry_out, Woperand, Wshift);
            } else {
                RegAlloc::Realize(Wresult, Wcarry_out, Woperand, Wshift, Wcarry_in);
            }
            ctx.reg_alloc.SpillFlags();

            // TODO: Use RMIF

            Xbyak_loongarch64::Label zero, end;

            code.ANDS(Wscratch1, Wshift, 0xff);
            code.B(EQ, zero);

            code.NEG(Wscratch0, Wshift);
            code.LSR(Wcarry_out, Woperand, Wscratch0);
            code.LSL(Wresult, Woperand, Wshift);
            code.UBFIZ(Wcarry_out, Wcarry_out, 29, 1);
            code.CMP(Wscratch1, 32);
            code.CSEL(Wresult, Wresult, WZR, LT);
            code.CSEL(Wcarry_out, Wcarry_out, WZR, LE);
            code.B(end);

            code.L(zero);
            code.add_d(*Wresult, Woperand, code.zero);
            if (carry_arg.IsImmediate()) {
                code.add_d(Wcarry_out, carry_arg.GetImmediateU32() << 29, code.zero);
            } else {
                code.add_d(*Wcarry_out, Wcarry_in, code.zero);
            }

            code.L(end);
        }
    }
}

template<>
void EmitIR<IR::Opcode::LogicalShiftLeft64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    if (args[1].IsImmediate()) {
        const u8 shift = args[1].GetImmediateU8();
        auto Xresult = ctx.reg_alloc.WriteX(inst);
        auto Xoperand = ctx.reg_alloc.ReadX(args[0]);
        RegAlloc::Realize(Xresult, Xoperand);

        if (shift <= 63) {
            code.LSL(Xresult, Xoperand, shift);
        } else {
            code.add_d(Xresult, XZR, code.zero);
        }
    } else {
        auto Xresult = ctx.reg_alloc.WriteX(inst);
        auto Xoperand = ctx.reg_alloc.ReadX(args[0]);
        auto Xshift = ctx.reg_alloc.ReadX(args[1]);
        RegAlloc::Realize(Xresult, Xoperand, Xshift);
        ctx.reg_alloc.SpillFlags();

        code.andi(Xscratch0, Xshift, 0xff);
        code.LSL(Xresult, Xoperand, Xscratch0);
        code.CMP(Xscratch0, 64);
        code.CSEL(Xresult, Xresult, XZR, LT);
    }
}

template<>
void EmitIR<IR::Opcode::LogicalShiftRight32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto carry_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetCarryFromOp);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto& operand_arg = args[0];
    auto& shift_arg = args[1];
    auto& carry_arg = args[2];

    if (!carry_inst) {
        if (shift_arg.IsImmediate()) {
            const u8 shift = shift_arg.GetImmediateU8();
            auto Wresult = ctx.reg_alloc.WriteW(inst);
            auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
            RegAlloc::Realize(Wresult, Woperand);

            if (shift <= 31) {
                code.LSR(Wresult, Woperand, shift);
            } else {
                code.add_d(Wresult, WZR, code.zero);
            }
        } else {
            auto Wresult = ctx.reg_alloc.WriteW(inst);
            auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
            auto Wshift = ctx.reg_alloc.ReadW(shift_arg);
            RegAlloc::Realize(Wresult, Woperand, Wshift);
            ctx.reg_alloc.SpillFlags();

            code.andi(Wscratch0, Wshift, 0xff);
            code.LSR(Wresult, Woperand, Wscratch0);
            code.CMP(Wscratch0, 32);
            code.CSEL(Wresult, Wresult, WZR, LT);
        }
    } else {
        if (shift_arg.IsImmediate() && shift_arg.GetImmediateU8() == 0) {
            ctx.reg_alloc.DefineAsExisting(carry_inst, carry_arg);
            ctx.reg_alloc.DefineAsExisting(inst, operand_arg);
        } else if (shift_arg.IsImmediate()) {
            // TODO: Use RMIF
            const u8 shift = shift_arg.GetImmediateU8();

            if (shift < 32) {
                auto Wresult = ctx.reg_alloc.WriteW(inst);
                auto Wcarry_out = ctx.reg_alloc.WriteW(carry_inst);
                auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
                RegAlloc::Realize(Wresult, Wcarry_out, Woperand);

                code.UBFX(Wcarry_out, Woperand, shift - 1, 1);
                code.LSL(Wcarry_out, Wcarry_out, 29);
                code.LSR(Wresult, Woperand, shift);
            } else if (shift > 32) {
                auto Wresult = ctx.reg_alloc.WriteW(inst);
                auto Wcarry_out = ctx.reg_alloc.WriteW(carry_inst);
                RegAlloc::Realize(Wresult, Wcarry_out);

                code.add_d(Wresult, WZR, code.zero);
                code.add_d(Wcarry_out, WZR, code.zero);
            } else {
                auto Wresult = ctx.reg_alloc.WriteW(inst);
                auto Wcarry_out = ctx.reg_alloc.WriteW(carry_inst);
                auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
                RegAlloc::Realize(Wresult, Wcarry_out, Woperand);

                code.LSR(Wcarry_out, Woperand, 31 - 29);
                code.andi(Wcarry_out, Wcarry_out, 1 << 29);
                code.add_d(Wresult, WZR, code.zero);
            }
        } else {
            auto Wresult = ctx.reg_alloc.WriteW(inst);
            auto Wcarry_out = ctx.reg_alloc.WriteW(carry_inst);
            auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
            auto Wshift = ctx.reg_alloc.ReadW(shift_arg);
            auto Wcarry_in = ctx.reg_alloc.ReadW(carry_arg);
            if (carry_arg.IsImmediate()) {
                RegAlloc::Realize(Wresult, Wcarry_out, Woperand, Wshift);
            } else {
                RegAlloc::Realize(Wresult, Wcarry_out, Woperand, Wshift, Wcarry_in);
            }
            ctx.reg_alloc.SpillFlags();

            // TODO: Use RMIF

            Xbyak_loongarch64::Label zero, end;

            code.ANDS(Wscratch1, Wshift, 0xff);
            code.B(EQ, zero);

            code.sub_imm(Wscratch0, Wshift, 1, code.t0);
            code.LSR(Wcarry_out, Woperand, Wscratch0);
            code.LSR(Wresult, Woperand, Wshift);
            code.UBFIZ(Wcarry_out, Wcarry_out, 29, 1);
            code.CMP(Wscratch1, 32);
            code.CSEL(Wresult, Wresult, WZR, LT);
            code.CSEL(Wcarry_out, Wcarry_out, WZR, LE);
            code.B(end);

            code.L(zero);
            code.add_d(*Wresult, Woperand, code.zero);
            if (carry_arg.IsImmediate()) {
                code.add_d(Wcarry_out, carry_arg.GetImmediateU32() << 29, code.zero);
            } else {
                code.add_d(*Wcarry_out, Wcarry_in, code.zero);
            }

            code.L(end);
        }
    }
}

template<>
void EmitIR<IR::Opcode::LogicalShiftRight64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    if (args[1].IsImmediate()) {
        const u8 shift = args[1].GetImmediateU8();
        auto Xresult = ctx.reg_alloc.WriteX(inst);
        auto Xoperand = ctx.reg_alloc.ReadX(args[0]);
        RegAlloc::Realize(Xresult, Xoperand);

        if (shift <= 63) {
            code.LSR(Xresult, Xoperand, shift);
        } else {
            code.add_d(Xresult, XZR, code.zero);
        }
    } else {
        auto Xresult = ctx.reg_alloc.WriteX(inst);
        auto Xoperand = ctx.reg_alloc.ReadX(args[0]);
        auto Xshift = ctx.reg_alloc.ReadX(args[1]);
        RegAlloc::Realize(Xresult, Xoperand, Xshift);
        ctx.reg_alloc.SpillFlags();

        code.andi(Xscratch0, Xshift, 0xff);
        code.LSR(Xresult, Xoperand, Xscratch0);
        code.CMP(Xscratch0, 64);
        code.CSEL(Xresult, Xresult, XZR, LT);
    }
}

template<>
void EmitIR<IR::Opcode::ArithmeticShiftRight32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto carry_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetCarryFromOp);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto& operand_arg = args[0];
    auto& shift_arg = args[1];
    auto& carry_arg = args[2];

    if (!carry_inst) {
        if (shift_arg.IsImmediate()) {
            const u8 shift = shift_arg.GetImmediateU8();
            auto Wresult = ctx.reg_alloc.WriteW(inst);
            auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
            RegAlloc::Realize(Wresult, Woperand);

            code.ASR(Wresult, Woperand, shift <= 31 ? shift : 31);
        } else {
            auto Wresult = ctx.reg_alloc.WriteW(inst);
            auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
            auto Wshift = ctx.reg_alloc.ReadW(shift_arg);
            RegAlloc::Realize(Wresult, Woperand, Wshift);
            ctx.reg_alloc.SpillFlags();

            code.andi(Wscratch0, Wshift, 0xff);
            code.add_d(Wscratch1, 31, code.zero);
            code.CMP(Wscratch0, 31);
            code.CSEL(Wscratch0, Wscratch0, Wscratch1, LS);
            code.ASR(Wresult, Woperand, Wscratch0);
        }
    } else {
        if (shift_arg.IsImmediate() && shift_arg.GetImmediateU8() == 0) {
            ctx.reg_alloc.DefineAsExisting(carry_inst, carry_arg);
            ctx.reg_alloc.DefineAsExisting(inst, operand_arg);
        } else if (shift_arg.IsImmediate()) {
            // TODO: Use RMIF

            const u8 shift = shift_arg.GetImmediateU8();

            if (shift <= 31) {
                auto Wresult = ctx.reg_alloc.WriteW(inst);
                auto Wcarry_out = ctx.reg_alloc.WriteW(carry_inst);
                auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
                RegAlloc::Realize(Wresult, Wcarry_out, Woperand);

                code.UBFX(Wcarry_out, Woperand, shift - 1, 1);
                code.LSL(Wcarry_out, Wcarry_out, 29);
                code.ASR(Wresult, Woperand, shift);
            } else {
                auto Wresult = ctx.reg_alloc.WriteW(inst);
                auto Wcarry_out = ctx.reg_alloc.WriteW(carry_inst);
                auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
                RegAlloc::Realize(Wresult, Wcarry_out, Woperand);

                code.ASR(Wresult, Woperand, 31);
                code.andi(Wcarry_out, Wresult, 1 << 29);
            }
        } else {
            auto Wresult = ctx.reg_alloc.WriteW(inst);
            auto Wcarry_out = ctx.reg_alloc.WriteW(carry_inst);
            auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
            auto Wshift = ctx.reg_alloc.ReadW(shift_arg);
            auto Wcarry_in = ctx.reg_alloc.ReadW(carry_arg);
            if (carry_arg.IsImmediate()) {
                RegAlloc::Realize(Wresult, Wcarry_out, Woperand, Wshift);
            } else {
                RegAlloc::Realize(Wresult, Wcarry_out, Woperand, Wshift, Wcarry_in);
            }
            ctx.reg_alloc.SpillFlags();

            // TODO: Use RMIF

            Xbyak_loongarch64::Label zero, end;

            code.ANDS(Wscratch0, Wshift, 0xff);
            code.B(EQ, zero);

            code.add_d(Wscratch1, 63, code.zero);
            code.CMP(Wscratch0, 63);
            code.CSEL(Wscratch0, Wscratch0, Wscratch1, LS);

            code.SXTW(Wresult->toX(), Woperand);
            code.sub_imm(Wscratch1, Wscratch0, 1, code.t0);

            code.ASR(Wcarry_out->toX(), Wresult->toX(), Xscratch1);
            code.ASR(Wresult->toX(), Wresult->toX(), Xscratch0);

            code.UBFIZ(Wcarry_out, Wcarry_out, 29, 1);
            code.add_d(*Wresult, Wresult, code.zero);

            code.B(end);

            code.L(zero);
            code.add_d(*Wresult, Woperand, code.zero);
            if (carry_arg.IsImmediate()) {
                code.add_d(Wcarry_out, carry_arg.GetImmediateU32() << 29, code.zero);
            } else {
                code.add_d(*Wcarry_out, Wcarry_in, code.zero);
            }

            code.L(end);
        }
    }
}

template<>
void EmitIR<IR::Opcode::ArithmeticShiftRight64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto& operand_arg = args[0];
    auto& shift_arg = args[1];

    if (shift_arg.IsImmediate()) {
        const u8 shift = shift_arg.GetImmediateU8();
        auto Xresult = ctx.reg_alloc.WriteX(inst);
        auto Xoperand = ctx.reg_alloc.ReadX(operand_arg);
        RegAlloc::Realize(Xresult, Xoperand);
        code.ASR(Xresult, Xoperand, shift <= 63 ? shift : 63);
    } else {
        auto Xresult = ctx.reg_alloc.WriteX(inst);
        auto Xoperand = ctx.reg_alloc.ReadX(operand_arg);
        auto Xshift = ctx.reg_alloc.ReadX(shift_arg);
        RegAlloc::Realize(Xresult, Xoperand, Xshift);
        code.ASR(Xresult, Xoperand, Xshift);
    }
}

template<>
void EmitIR<IR::Opcode::RotateRight32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto carry_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetCarryFromOp);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto& operand_arg = args[0];
    auto& shift_arg = args[1];
    auto& carry_arg = args[2];

    if (shift_arg.IsImmediate() && shift_arg.GetImmediateU8() == 0) {
        if (carry_inst) {
            ctx.reg_alloc.DefineAsExisting(carry_inst, carry_arg);
        }
        ctx.reg_alloc.DefineAsExisting(inst, operand_arg);
    } else if (shift_arg.IsImmediate()) {
        const u8 shift = shift_arg.GetImmediateU8() % 32;
        auto Wresult = ctx.reg_alloc.WriteW(inst);
        auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
        RegAlloc::Realize(Wresult, Woperand);

        code.ROR(Wresult, Woperand, shift);

        if (carry_inst) {
            auto Wcarry_out = ctx.reg_alloc.WriteW(carry_inst);
            RegAlloc::Realize(Wcarry_out);

            code.ROR(Wcarry_out, Woperand, ((shift + 31) - 29) % 32);
            code.andi(Wcarry_out, Wcarry_out, 1 << 29);
        }
    } else {
        auto Wresult = ctx.reg_alloc.WriteW(inst);
        auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
        auto Wshift = ctx.reg_alloc.ReadW(shift_arg);
        RegAlloc::Realize(Wresult, Woperand, Wshift);

        code.ROR(Wresult, Woperand, Wshift);

        if (carry_inst && carry_arg.IsImmediate()) {
            const u32 carry_in = carry_arg.GetImmediateU32() << 29;
            auto Wcarry_out = ctx.reg_alloc.WriteW(carry_inst);
            RegAlloc::Realize(Wcarry_out);
            ctx.reg_alloc.SpillFlags();

            code.TST(Wshift, 0xff);
            code.LSR(Wcarry_out, Wresult, 31 - 29);
            code.andi(Wcarry_out, Wcarry_out, 1 << 29);
            if (carry_in) {
                code.add_d(Wscratch0, carry_in, code.zero);
                code.CSEL(Wcarry_out, Wscratch0, Wcarry_out, EQ);
            } else {
                code.CSEL(Wcarry_out, WZR, Wcarry_out, EQ);
            }
        } else if (carry_inst) {
            auto Wcarry_in = ctx.reg_alloc.ReadW(carry_arg);
            auto Wcarry_out = ctx.reg_alloc.WriteW(carry_inst);
            RegAlloc::Realize(Wcarry_out, Wcarry_in);
            ctx.reg_alloc.SpillFlags();

            code.TST(Wshift, 0xff);
            code.LSR(Wcarry_out, Wresult, 31 - 29);
            code.andi(Wcarry_out, Wcarry_out, 1 << 29);
            code.CSEL(Wcarry_out, Wcarry_in, Wcarry_out, EQ);
        }
    }
}

template<>
void EmitIR<IR::Opcode::RotateRight64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto& operand_arg = args[0];
    auto& shift_arg = args[1];

    if (shift_arg.IsImmediate()) {
        const u8 shift = shift_arg.GetImmediateU8();
        auto Xresult = ctx.reg_alloc.WriteX(inst);
        auto Xoperand = ctx.reg_alloc.ReadX(operand_arg);
        RegAlloc::Realize(Xresult, Xoperand);
        code.ROR(Xresult, Xoperand, shift);
    } else {
        auto Xresult = ctx.reg_alloc.WriteX(inst);
        auto Xoperand = ctx.reg_alloc.ReadX(operand_arg);
        auto Xshift = ctx.reg_alloc.ReadX(shift_arg);
        RegAlloc::Realize(Xresult, Xoperand, Xshift);
        code.ROR(Xresult, Xoperand, Xshift);
    }
}

template<>
void EmitIR<IR::Opcode::RotateRightExtended>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto carry_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetCarryFromOp);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Wresult = ctx.reg_alloc.WriteW(inst);
    auto Woperand = ctx.reg_alloc.ReadW(args[0]);

    if (args[1].IsImmediate()) {
        RegAlloc::Realize(Wresult, Woperand);

        code.LSR(Wresult, Woperand, 1);
        if (args[1].GetImmediateU1()) {
            code.ORR(Wresult, Wresult, 0x8000'0000);
        }
    } else {
        auto Wcarry_in = ctx.reg_alloc.ReadW(args[1]);
        RegAlloc::Realize(Wresult, Woperand, Wcarry_in);

        code.LSR(Wscratch0, Wcarry_in, 29);
        code.EXTR(Wresult, Wscratch0, Woperand, 1);
    }

    if (carry_inst) {
        auto Wcarry_out = ctx.reg_alloc.WriteW(carry_inst);
        RegAlloc::Realize(Wcarry_out);
        code.UBFIZ(Wcarry_out, Woperand, 29, 1);
    }
}

template<typename ShiftI, typename ShiftR>
static void EmitMaskedShift32(Xbyak_loongarch64::CodeGenerator&, EmitContext& ctx, IR::Inst* inst, ShiftI si_fn, ShiftR sr_fn) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto& operand_arg = args[0];
    auto& shift_arg = args[1];

    if (shift_arg.IsImmediate()) {
        auto Wresult = ctx.reg_alloc.WriteW(inst);
        auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
        RegAlloc::Realize(Wresult, Woperand);
        const u32 shift = shift_arg.GetImmediateU32();

        si_fn(Wresult, Woperand, static_cast<int>(shift & 0x1F));
    } else {
        auto Wresult = ctx.reg_alloc.WriteW(inst);
        auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
        auto Wshift = ctx.reg_alloc.ReadW(shift_arg);
        RegAlloc::Realize(Wresult, Woperand, Wshift);

        sr_fn(Wresult, Woperand, Wshift);
    }
}

template<typename ShiftI, typename ShiftR>
static void EmitMaskedShift64(Xbyak_loongarch64::CodeGenerator&, EmitContext& ctx, IR::Inst* inst, ShiftI si_fn, ShiftR sr_fn) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto& operand_arg = args[0];
    auto& shift_arg = args[1];

    if (shift_arg.IsImmediate()) {
        auto Xresult = ctx.reg_alloc.WriteX(inst);
        auto Xoperand = ctx.reg_alloc.ReadX(operand_arg);
        RegAlloc::Realize(Xresult, Xoperand);
        const u32 shift = shift_arg.GetImmediateU64();

        si_fn(Xresult, Xoperand, static_cast<int>(shift & 0x3F));
    } else {
        auto Xresult = ctx.reg_alloc.WriteX(inst);
        auto Xoperand = ctx.reg_alloc.ReadX(operand_arg);
        auto Xshift = ctx.reg_alloc.ReadX(shift_arg);
        RegAlloc::Realize(Xresult, Xoperand, Xshift);

        sr_fn(Xresult, Xoperand, Xshift);
    }
}

template<>
void EmitIR<IR::Opcode::LogicalShiftLeftMasked32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitMaskedShift32(
        code, ctx, inst,
        [&](auto& Wresult, auto& Woperand, auto shift) { code.LSL(Wresult, Woperand, shift); },
        [&](auto& Wresult, auto& Woperand, auto& Wshift) { code.LSL(Wresult, Woperand, Wshift); });
}

template<>
void EmitIR<IR::Opcode::LogicalShiftLeftMasked64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitMaskedShift64(
        code, ctx, inst,
        [&](auto& Xresult, auto& Xoperand, auto shift) { code.LSL(Xresult, Xoperand, shift); },
        [&](auto& Xresult, auto& Xoperand, auto& Xshift) { code.LSL(Xresult, Xoperand, Xshift); });
}

template<>
void EmitIR<IR::Opcode::LogicalShiftRightMasked32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitMaskedShift32(
        code, ctx, inst,
        [&](auto& Wresult, auto& Woperand, auto shift) { code.LSR(Wresult, Woperand, shift); },
        [&](auto& Wresult, auto& Woperand, auto& Wshift) { code.LSR(Wresult, Woperand, Wshift); });
}

template<>
void EmitIR<IR::Opcode::LogicalShiftRightMasked64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitMaskedShift64(
        code, ctx, inst,
        [&](auto& Xresult, auto& Xoperand, auto shift) { code.LSR(Xresult, Xoperand, shift); },
        [&](auto& Xresult, auto& Xoperand, auto& Xshift) { code.LSR(Xresult, Xoperand, Xshift); });
}

template<>
void EmitIR<IR::Opcode::ArithmeticShiftRightMasked32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitMaskedShift32(
        code, ctx, inst,
        [&](auto& Wresult, auto& Woperand, auto shift) { code.ASR(Wresult, Woperand, shift); },
        [&](auto& Wresult, auto& Woperand, auto& Wshift) { code.ASR(Wresult, Woperand, Wshift); });
}

template<>
void EmitIR<IR::Opcode::ArithmeticShiftRightMasked64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitMaskedShift64(
        code, ctx, inst,
        [&](auto& Xresult, auto& Xoperand, auto shift) { code.ASR(Xresult, Xoperand, shift); },
        [&](auto& Xresult, auto& Xoperand, auto& Xshift) { code.ASR(Xresult, Xoperand, Xshift); });
}

template<>
void EmitIR<IR::Opcode::RotateRightMasked32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitMaskedShift32(
        code, ctx, inst,
        [&](auto& Wresult, auto& Woperand, auto shift) { code.ROR(Wresult, Woperand, shift); },
        [&](auto& Wresult, auto& Woperand, auto& Wshift) { code.ROR(Wresult, Woperand, Wshift); });
}

template<>
void EmitIR<IR::Opcode::RotateRightMasked64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitMaskedShift64(
        code, ctx, inst,
        [&](auto& Xresult, auto& Xoperand, auto shift) { code.ROR(Xresult, Xoperand, shift); },
        [&](auto& Xresult, auto& Xoperand, auto& Xshift) { code.ROR(Xresult, Xoperand, Xshift); });
}

template<size_t bitsize, typename EmitFn>
static void MaybeAddSubImm(Xbyak_loongarch64::CodeGenerator& code, u64 imm, EmitFn emit_fn) {
    static_assert(bitsize == 32 || bitsize == 64);
    if constexpr (bitsize == 32) {
        imm = static_cast<u32>(imm);
    }
    if (Xbyak_loongarch64::AddSubImm::is_valid(imm)) {
        emit_fn(imm);
    } else {
        code.add_d(Rscratch0<bitsize>(), imm, code.zero);
        emit_fn(Rscratch0<bitsize>());
    }
}

template<size_t bitsize, bool sub>
static void EmitAddSub(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto nzcv_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetNZCVFromOp);
    const auto overflow_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetOverflowFromOp);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Rresult = ctx.reg_alloc.WriteReg<bitsize>(inst);
    auto Ra = ctx.reg_alloc.ReadReg<bitsize>(args[0]);

    if (overflow_inst) {
        // There is a limited set of circumstances where this is required, so assert for this.
        ASSERT(!sub);
        ASSERT(!nzcv_inst);
        ASSERT(args[2].IsImmediate() && args[2].GetImmediateU1() == false);

        auto Rb = ctx.reg_alloc.ReadReg<bitsize>(args[1]);
        auto Woverflow = ctx.reg_alloc.WriteW(overflow_inst);
        ctx.reg_alloc.SpillFlags();
        RegAlloc::Realize(Rresult, Ra, Rb, Woverflow);

        code.ADDS(Rresult, *Ra, Rb);
        code.CSET(Woverflow, VS);
    } else if (nzcv_inst) {
        if (args[1].IsImmediate()) {
            const u64 imm = args[1].GetImmediateU64();

            if (args[2].IsImmediate()) {
                auto flags = ctx.reg_alloc.WriteFlags(nzcv_inst);
                RegAlloc::Realize(Rresult, Ra, flags);

                if (args[2].GetImmediateU1()) {
                    MaybeAddSubImm<bitsize>(code, sub ? imm : ~imm, [&](const auto b) { code.SUBS(Rresult, *Ra, b); });
                } else {
                    MaybeAddSubImm<bitsize>(code, sub ? ~imm : imm, [&](const auto b) { code.ADDS(Rresult, *Ra, b); });
                }
            } else {
                RegAlloc::Realize(Rresult, Ra);
                ctx.reg_alloc.ReadWriteFlags(args[2], nzcv_inst);

                if (imm == 0) {
                    if constexpr (bitsize == 32) {
                        sub ? code.SBCS(Rresult, Ra, WZR) : code.ADCS(Rresult, Ra, WZR);
                    } else {
                        sub ? code.SBCS(Rresult, Ra, XZR) : code.ADCS(Rresult, Ra, XZR);
                    }
                } else {
                    code.add_d(Rscratch0<bitsize>(), imm, code.zero);
                    sub ? code.SBCS(Rresult, Ra, Rscratch0<bitsize>()) : code.ADCS(Rresult, Ra, Rscratch0<bitsize>());
                }
            }
        } else {
            auto Rb = ctx.reg_alloc.ReadReg<bitsize>(args[1]);

            if (args[2].IsImmediate()) {
                auto flags = ctx.reg_alloc.WriteFlags(nzcv_inst);
                RegAlloc::Realize(Rresult, Ra, Rb, flags);

                if (args[2].GetImmediateU1()) {
                    if (sub) {
                        code.SUBS(Rresult, *Ra, Rb);
                    } else {
                        code.MVN(Rscratch0<bitsize>(), Rb);
                        code.SUBS(Rresult, *Ra, Rscratch0<bitsize>());
                    }
                } else {
                    if (sub) {
                        code.MVN(Rscratch0<bitsize>(), Rb);
                        code.ADDS(Rresult, *Ra, Rscratch0<bitsize>());
                    } else {
                        code.ADDS(Rresult, *Ra, Rb);
                    }
                }
            } else {
                RegAlloc::Realize(Rresult, Ra, Rb);
                ctx.reg_alloc.ReadWriteFlags(args[2], nzcv_inst);

                sub ? code.SBCS(Rresult, Ra, Rb) : code.ADCS(Rresult, Ra, Rb);
            }
        }
    } else {
        if (args[1].IsImmediate()) {
            const u64 imm = args[1].GetImmediateU64();

            RegAlloc::Realize(Rresult, Ra);

            if (args[2].IsImmediate()) {
                if (args[2].GetImmediateU1()) {
                    MaybeAddSubImm<bitsize>(
                        code, sub ? imm : ~imm, [&](const auto b) { code.sub_imm(Rresult, *Ra, b); }, code.t0);
                } else {
                    MaybeAddSubImm<bitsize>(code, sub ? ~imm : imm, [&](const auto b) { code.ADD(Rresult, *Ra, b); });
                }
            } else {
                ctx.reg_alloc.ReadWriteFlags(args[2], nullptr);

                if (imm == 0) {
                    if constexpr (bitsize == 32) {
                        sub ? code.SBC(Rresult, Ra, WZR) : code.ADC(Rresult, Ra, WZR);
                    } else {
                        sub ? code.SBC(Rresult, Ra, XZR) : code.ADC(Rresult, Ra, XZR);
                    }
                } else {
                    code.add_d(Rscratch0<bitsize>(), imm, code.zero);
                    sub ? code.SBC(Rresult, Ra, Rscratch0<bitsize>()) : code.ADC(Rresult, Ra, Rscratch0<bitsize>());
                }
            }
        } else {
            auto Rb = ctx.reg_alloc.ReadReg<bitsize>(args[1]);

            RegAlloc::Realize(Rresult, Ra, Rb);

            if (args[2].IsImmediate()) {
                if (args[2].GetImmediateU1()) {
                    if (sub) {
                        code.sub_imm(Rresult, *Ra, Rb, code.t0);
                    } else {
                        code.MVN(Rscratch0<bitsize>(), Rb);
                        code.sub_imm(Rresult, *Ra, Rscratch0<bitsize>(), code.t0);
                    }
                } else {
                    if (sub) {
                        code.MVN(Rscratch0<bitsize>(), Rb);
                        code.ADD(Rresult, *Ra, Rscratch0<bitsize>());
                    } else {
                        code.ADD(Rresult, *Ra, Rb);
                    }
                }
            } else {
                ctx.reg_alloc.ReadWriteFlags(args[2], nullptr);

                sub ? code.SBC(Rresult, Ra, Rb) : code.ADC(Rresult, Ra, Rb);
            }
        }
    }
}

template<>
void EmitIR<IR::Opcode::Add32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitAddSub<32, false>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::Add64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitAddSub<64, false>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::Sub32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitAddSub<32, true>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::Sub64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitAddSub<64, true>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::Mul32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<32>(
        code, ctx, inst,
        [&](auto& Wresult, auto& Wa, auto& Wb) { code.MUL(Wresult, Wa, Wb); });
}

template<>
void EmitIR<IR::Opcode::Mul64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<64>(
        code, ctx, inst,
        [&](auto& Xresult, auto& Xa, auto& Xb) { code.MUL(Xresult, Xa, Xb); });
}

template<>
void EmitIR<IR::Opcode::SignedMultiplyHigh64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xresult = ctx.reg_alloc.WriteX(inst);
    auto Xop1 = ctx.reg_alloc.ReadX(args[0]);
    auto Xop2 = ctx.reg_alloc.ReadX(args[1]);
    RegAlloc::Realize(Xresult, Xop1, Xop2);

    code.SMULH(Xresult, Xop1, Xop2);
}

template<>
void EmitIR<IR::Opcode::UnsignedMultiplyHigh64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xresult = ctx.reg_alloc.WriteX(inst);
    auto Xop1 = ctx.reg_alloc.ReadX(args[0]);
    auto Xop2 = ctx.reg_alloc.ReadX(args[1]);
    RegAlloc::Realize(Xresult, Xop1, Xop2);

    code.UMULH(Xresult, Xop1, Xop2);
}

template<>
void EmitIR<IR::Opcode::UnsignedDiv32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<32>(
        code, ctx, inst,
        [&](auto& Wresult, auto& Wa, auto& Wb) { code.UDIV(Wresult, Wa, Wb); });
}

template<>
void EmitIR<IR::Opcode::UnsignedDiv64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<64>(
        code, ctx, inst,
        [&](auto& Xresult, auto& Xa, auto& Xb) { code.UDIV(Xresult, Xa, Xb); });
}

template<>
void EmitIR<IR::Opcode::SignedDiv32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<32>(
        code, ctx, inst,
        [&](auto& Wresult, auto& Wa, auto& Wb) { code.SDIV(Wresult, Wa, Wb); });
}

template<>
void EmitIR<IR::Opcode::SignedDiv64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<64>(
        code, ctx, inst,
        [&](auto& Xresult, auto& Xa, auto& Xb) { code.SDIV(Xresult, Xa, Xb); });
}

template<size_t bitsize>
static bool IsValidBitImm(u64 imm) {
    static_assert(bitsize == 32 || bitsize == 64);
    if constexpr (bitsize == 32) {
        return static_cast<bool>(Xbyak_loongarch64::detail::encode_bit_imm(static_cast<u32>(imm)));
    } else {
        return static_cast<bool>(Xbyak_loongarch64::detail::encode_bit_imm(imm));
    }
}

template<size_t bitsize, typename EmitFn>
static void MaybeBitImm(Xbyak_loongarch64::CodeGenerator& code, u64 imm, EmitFn emit_fn) {
    static_assert(bitsize == 32 || bitsize == 64);
    if constexpr (bitsize == 32) {
        imm = static_cast<u32>(imm);
    }
    if (IsValidBitImm<bitsize>(imm)) {
        emit_fn(imm);
    } else {
        code.add_d(Rscratch0<bitsize>(), imm, code.zero);
        emit_fn(Rscratch0<bitsize>());
    }
}

template<size_t bitsize, typename EmitFn1, typename EmitFn2 = std::nullptr_t>
static void EmitBitOp(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, EmitFn1 emit_without_flags, EmitFn2 emit_with_flags = nullptr) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Rresult = ctx.reg_alloc.WriteReg<bitsize>(inst);
    auto Ra = ctx.reg_alloc.ReadReg<bitsize>(args[0]);

    if constexpr (!std::is_same_v<EmitFn2, std::nullptr_t>) {
        const auto nz_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetNZFromOp);
        const auto nzcv_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetNZCVFromOp);
        ASSERT(!(nz_inst && nzcv_inst));
        const auto flag_inst = nz_inst ? nz_inst : nzcv_inst;

        if (flag_inst) {
            auto Wflags = ctx.reg_alloc.WriteFlags(flag_inst);

            if (args[1].IsImmediate()) {
                RegAlloc::Realize(Rresult, Ra, Wflags);

                MaybeBitImm<bitsize>(code, args[1].GetImmediateU64(), [&](const auto& b) { emit_with_flags(Rresult, Ra, b); });
            } else {
                auto Rb = ctx.reg_alloc.ReadReg<bitsize>(args[1]);
                RegAlloc::Realize(Rresult, Ra, Rb, Wflags);

                emit_with_flags(Rresult, Ra, Rb);
            }

            return;
        }
    }

    if (args[1].IsImmediate()) {
        RegAlloc::Realize(Rresult, Ra);

        MaybeBitImm<bitsize>(code, args[1].GetImmediateU64(), [&](const auto& b) { emit_without_flags(Rresult, Ra, b); });
    } else {
        auto Rb = ctx.reg_alloc.ReadReg<bitsize>(args[1]);
        RegAlloc::Realize(Rresult, Ra, Rb);

        emit_without_flags(Rresult, Ra, Rb);
    }
}

template<size_t bitsize>
static void EmitAndNot(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto nz_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetNZFromOp);
    const auto nzcv_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetNZCVFromOp);
    ASSERT(!(nz_inst && nzcv_inst));
    const auto flag_inst = nz_inst ? nz_inst : nzcv_inst;

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Rresult = ctx.reg_alloc.WriteReg<bitsize>(inst);
    auto Ra = ctx.reg_alloc.ReadReg<bitsize>(args[0]);

    if (flag_inst) {
        auto Wflags = ctx.reg_alloc.WriteFlags(flag_inst);

        if (args[1].IsImmediate()) {
            RegAlloc::Realize(Rresult, Ra, Wflags);

            const u64 not_imm = bitsize == 32 ? static_cast<u32>(~args[1].GetImmediateU64()) : ~args[1].GetImmediateU64();

            if (IsValidBitImm<bitsize>(not_imm)) {
                code.ANDS(Rresult, Ra, not_imm);
            } else {
                code.add_d(Rscratch0<bitsize>(), args[1].GetImmediateU64(), code.zero);
                code.BICS(Rresult, Ra, Rscratch0<bitsize>());
            }
        } else {
            auto Rb = ctx.reg_alloc.ReadReg<bitsize>(args[1]);
            RegAlloc::Realize(Rresult, Ra, Rb, Wflags);

            code.BICS(Rresult, Ra, Rb);
        }

        return;
    }

    if (args[1].IsImmediate()) {
        RegAlloc::Realize(Rresult, Ra);

        const u64 not_imm = bitsize == 32 ? static_cast<u32>(~args[1].GetImmediateU64()) : ~args[1].GetImmediateU64();

        if (IsValidBitImm<bitsize>(not_imm)) {
            code.andi(Rresult, Ra, not_imm);
        } else {
            code.add_d(Rscratch0<bitsize>(), args[1].GetImmediateU64(), code.zero);
            code.BIC(Rresult, Ra, Rscratch0<bitsize>());
        }
    } else {
        auto Rb = ctx.reg_alloc.ReadReg<bitsize>(args[1]);
        RegAlloc::Realize(Rresult, Ra, Rb);

        code.BIC(Rresult, Ra, Rb);
    }
}

template<>
void EmitIR<IR::Opcode::And32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitBitOp<32>(
        code, ctx, inst,
        [&](auto& result, auto& a, auto& b) { code.andi(result, a, b); },
        [&](auto& result, auto& a, auto& b) { code.ANDS(result, a, b); });
}

template<>
void EmitIR<IR::Opcode::And64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitBitOp<64>(
        code, ctx, inst,
        [&](auto& result, auto& a, auto& b) { code.andi(result, a, b); },
        [&](auto& result, auto& a, auto& b) { code.ANDS(result, a, b); });
}

template<>
void EmitIR<IR::Opcode::AndNot32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitAndNot<32>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::AndNot64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitAndNot<64>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::Eor32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitBitOp<32>(
        code, ctx, inst,
        [&](auto& result, auto& a, auto& b) { code.EOR(result, a, b); });
}

template<>
void EmitIR<IR::Opcode::Eor64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitBitOp<64>(
        code, ctx, inst,
        [&](auto& result, auto& a, auto& b) { code.EOR(result, a, b); });
}

template<>
void EmitIR<IR::Opcode::Or32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitBitOp<32>(
        code, ctx, inst,
        [&](auto& result, auto& a, auto& b) { code.ORR(result, a, b); });
}

template<>
void EmitIR<IR::Opcode::Or64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitBitOp<64>(
        code, ctx, inst,
        [&](auto& result, auto& a, auto& b) { code.ORR(result, a, b); });
}

template<>
void EmitIR<IR::Opcode::Not32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<32>(
        code, ctx, inst,
        [&](auto& Wresult, auto& Woperand) { code.MVN(Wresult, Woperand); });
}

template<>
void EmitIR<IR::Opcode::Not64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<64>(
        code, ctx, inst,
        [&](auto& Xresult, auto& Xoperand) { code.MVN(Xresult, Xoperand); });
}

template<>
void EmitIR<IR::Opcode::SignExtendByteToWord>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<32>(
        code, ctx, inst,
        [&](auto& Wresult, auto& Woperand) { code.SXTB(Wresult, Woperand); });
}

template<>
void EmitIR<IR::Opcode::SignExtendHalfToWord>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<32>(
        code, ctx, inst,
        [&](auto& Wresult, auto& Woperand) { code.SXTH(Wresult, Woperand); });
}

template<>
void EmitIR<IR::Opcode::SignExtendByteToLong>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<64>(
        code, ctx, inst,
        [&](auto& Xresult, auto& Xoperand) { code.SXTB(Xresult, Xoperand->toW()); });
}

template<>
void EmitIR<IR::Opcode::SignExtendHalfToLong>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<64>(
        code, ctx, inst,
        [&](auto& Xresult, auto& Xoperand) { code.SXTH(Xresult, Xoperand->toW()); });
}

template<>
void EmitIR<IR::Opcode::SignExtendWordToLong>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<64>(
        code, ctx, inst,
        [&](auto& Xresult, auto& Xoperand) { code.SXTW(Xresult, Xoperand->toW()); });
}

template<>
void EmitIR<IR::Opcode::ZeroExtendByteToWord>(Xbyak_loongarch64::CodeGenerator&, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.DefineAsExisting(inst, args[0]);
}

template<>
void EmitIR<IR::Opcode::ZeroExtendHalfToWord>(Xbyak_loongarch64::CodeGenerator&, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.DefineAsExisting(inst, args[0]);
}

template<>
void EmitIR<IR::Opcode::ZeroExtendByteToLong>(Xbyak_loongarch64::CodeGenerator&, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.DefineAsExisting(inst, args[0]);
}

template<>
void EmitIR<IR::Opcode::ZeroExtendHalfToLong>(Xbyak_loongarch64::CodeGenerator&, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.DefineAsExisting(inst, args[0]);
}

template<>
void EmitIR<IR::Opcode::ZeroExtendWordToLong>(Xbyak_loongarch64::CodeGenerator&, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.DefineAsExisting(inst, args[0]);
}

template<>
void EmitIR<IR::Opcode::ZeroExtendLongToQuad>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xvalue = ctx.reg_alloc.ReadX(args[0]);
    auto Qresult = ctx.reg_alloc.WriteQ(inst);
    RegAlloc::Realize(Xvalue, Qresult);

    code.FMOV(Qresult->toD(), Xvalue);
}

template<>
void EmitIR<IR::Opcode::ByteReverseWord>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<32>(
        code, ctx, inst,
        [&](auto& Wresult, auto& Woperand) { code.REV(Wresult, Woperand); });
}

template<>
void EmitIR<IR::Opcode::ByteReverseHalf>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<32>(
        code, ctx, inst,
        [&](auto& Wresult, auto& Woperand) { code.REV16(Wresult, Woperand); });
}

template<>
void EmitIR<IR::Opcode::ByteReverseDual>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<64>(
        code, ctx, inst,
        [&](auto& Xresult, auto& Xoperand) { code.REV(Xresult, Xoperand); });
}

template<>
void EmitIR<IR::Opcode::CountLeadingZeros32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<32>(
        code, ctx, inst,
        [&](auto& Wresult, auto& Woperand) { code.CLZ(Wresult, Woperand); });
}

template<>
void EmitIR<IR::Opcode::CountLeadingZeros64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<64>(
        code, ctx, inst,
        [&](auto& Xresult, auto& Xoperand) { code.CLZ(Xresult, Xoperand); });
}

template<>
void EmitIR<IR::Opcode::ExtractRegister32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ASSERT(args[2].IsImmediate());

    auto Wresult = ctx.reg_alloc.WriteW(inst);
    auto Wop1 = ctx.reg_alloc.ReadW(args[0]);
    auto Wop2 = ctx.reg_alloc.ReadW(args[1]);
    RegAlloc::Realize(Wresult, Wop1, Wop2);
    const u8 lsb = args[2].GetImmediateU8();

    code.EXTR(Wresult, Wop2, Wop1, lsb);  // NB: flipped
}

template<>
void EmitIR<IR::Opcode::ExtractRegister64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ASSERT(args[2].IsImmediate());

    auto Xresult = ctx.reg_alloc.WriteX(inst);
    auto Xop1 = ctx.reg_alloc.ReadX(args[0]);
    auto Xop2 = ctx.reg_alloc.ReadX(args[1]);
    RegAlloc::Realize(Xresult, Xop1, Xop2);
    const u8 lsb = args[2].GetImmediateU8();

    code.EXTR(Xresult, Xop2, Xop1, lsb);  // NB: flipped
}

template<>
void EmitIR<IR::Opcode::ReplicateBit32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ASSERT(args[1].IsImmediate());

    auto Wresult = ctx.reg_alloc.WriteW(inst);
    auto Wvalue = ctx.reg_alloc.ReadW(args[0]);
    const u8 bit = args[1].GetImmediateU8();
    RegAlloc::Realize(Wresult, Wvalue);

    code.LSL(Wresult, Wvalue, 31 - bit);
    code.ASR(Wresult, Wresult, 31);
}

template<>
void EmitIR<IR::Opcode::ReplicateBit64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ASSERT(args[1].IsImmediate());

    auto Xresult = ctx.reg_alloc.WriteX(inst);
    auto Xvalue = ctx.reg_alloc.ReadX(args[0]);
    const u8 bit = args[1].GetImmediateU8();
    RegAlloc::Realize(Xresult, Xvalue);

    code.LSL(Xresult, Xvalue, 63 - bit);
    code.ASR(Xresult, Xresult, 63);
}

static void EmitMaxMin32(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, Xbyak_loongarch64::Cond cond) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Wresult = ctx.reg_alloc.WriteW(inst);
    auto Wop1 = ctx.reg_alloc.ReadW(args[0]);
    auto Wop2 = ctx.reg_alloc.ReadW(args[1]);
    RegAlloc::Realize(Wresult, Wop1, Wop2);
    ctx.reg_alloc.SpillFlags();

    code.CMP(Wop1->toW(), Wop2);
    code.CSEL(Wresult, Wop1, Wop2, cond);
}

static void EmitMaxMin64(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, Xbyak_loongarch64::Cond cond) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Xresult = ctx.reg_alloc.WriteX(inst);
    auto Xop1 = ctx.reg_alloc.ReadX(args[0]);
    auto Xop2 = ctx.reg_alloc.ReadX(args[1]);
    RegAlloc::Realize(Xresult, Xop1, Xop2);
    ctx.reg_alloc.SpillFlags();

    code.CMP(Xop1->toX(), Xop2);
    code.CSEL(Xresult, Xop1, Xop2, cond);
}

template<>
void EmitIR<IR::Opcode::MaxSigned32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitMaxMin32(code, ctx, inst, GT);
}

template<>
void EmitIR<IR::Opcode::MaxSigned64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitMaxMin64(code, ctx, inst, GT);
}

template<>
void EmitIR<IR::Opcode::MaxUnsigned32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitMaxMin32(code, ctx, inst, HI);
}

template<>
void EmitIR<IR::Opcode::MaxUnsigned64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitMaxMin64(code, ctx, inst, HI);
}

template<>
void EmitIR<IR::Opcode::MinSigned32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitMaxMin32(code, ctx, inst, LT);
}

template<>
void EmitIR<IR::Opcode::MinSigned64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitMaxMin64(code, ctx, inst, LT);
}

template<>
void EmitIR<IR::Opcode::MinUnsigned32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitMaxMin32(code, ctx, inst, LO);
}

template<>
void EmitIR<IR::Opcode::MinUnsigned64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitMaxMin64(code, ctx, inst, LO);
}

}  // namespace Dynarmic::Backend::LoongArch64
