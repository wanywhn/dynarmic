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
#include "a64_jitstate.h"

namespace Dynarmic::Backend::LoongArch64 {

    using namespace Xbyak_loongarch64::util;

    template<size_t bitsize, typename EmitFn>
    static void EmitTwoOp(BlockOfCode &, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        auto Rresult = ctx.reg_alloc.WriteReg<bitsize>(inst);
        auto Roperand = ctx.reg_alloc.ReadReg<bitsize>(args[0]);
        RegAlloc::Realize(Rresult, Roperand);

        emit(Rresult, Roperand);
    }

    template<size_t bitsize, typename EmitFn>
    static void EmitThreeOp(BlockOfCode &, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        auto Rresult = ctx.reg_alloc.WriteReg<bitsize>(inst);
        auto Ra = ctx.reg_alloc.ReadReg<bitsize>(args[0]);
        auto Rb = ctx.reg_alloc.ReadReg<bitsize>(args[1]);
        RegAlloc::Realize(Rresult, Ra, Rb);

        emit(Rresult, Ra, Rb);
    }

    template<>
    void EmitIR<IR::Opcode::Pack2x32To1x64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        auto Wlo = ctx.reg_alloc.ReadW(args[0]);
        auto Whi = ctx.reg_alloc.ReadW(args[1]);
        auto Xresult = ctx.reg_alloc.WriteX(inst);
        RegAlloc::Realize(Wlo, Whi, Xresult);

        code.add_d(Xresult, Wlo, code.zero);  // TODO: Move eliminiation
        code.bstrins_d(Xresult, Whi, 63, 32);
    }

    template<>
    void EmitIR<IR::Opcode::Pack2x64To1x128>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        if (args[0].IsInGpr() && args[1].IsInGpr()) {
            auto Xlo = ctx.reg_alloc.ReadX(args[0]);
            auto Xhi = ctx.reg_alloc.ReadX(args[1]);
            auto Qresult = ctx.reg_alloc.WriteQ(inst);
            RegAlloc::Realize(Xlo, Xhi, Qresult);

            code.vinsgr2vr_w(Qresult, Xlo, 0);
            code.vinsgr2vr_w(Qresult, Xhi, 1);
        } else if (args[0].IsInGpr()) {
            auto Xlo = ctx.reg_alloc.ReadX(args[0]);
            auto Dhi = ctx.reg_alloc.ReadD(args[1]);
            auto Qresult = ctx.reg_alloc.WriteQ(inst);
            RegAlloc::Realize(Xlo, Dhi, Qresult);

            code.vslli_d(Qresult, Dhi, 64);
            code.vinsgr2vr_w(Qresult, Xlo, 0);
        } else if (args[1].IsInGpr()) {
            auto Dlo = ctx.reg_alloc.ReadD(args[0]);
            auto Xhi = ctx.reg_alloc.ReadX(args[1]);
            auto Qresult = ctx.reg_alloc.WriteQ(inst);
            RegAlloc::Realize(Dlo, Xhi, Qresult);

            code.vinsgr2vr_d(Qresult, Xhi, 1);
            code.vinsgr2vr_d(Qresult, code.zero, 0);
            code.vadd_d(Qresult, Qresult, Dlo);
//        code.FMOV(Qresult->toD(), Dlo);  // TODO: Move eliminiation
//        code.add_d(Xbyak_loongarch64::VRegSelector{Qresult->getIdx()))}.D()[1], Xhi, code.zero);
        } else {
            auto Dlo = ctx.reg_alloc.ReadD(args[0]);
            auto Dhi = ctx.reg_alloc.ReadD(args[1]);
            auto Qresult = ctx.reg_alloc.WriteQ(inst);
            RegAlloc::Realize(Dlo, Dhi, Qresult);
            code.vslli_d(Qresult, Dhi, 64);
            code.vadd_d(Qresult, Dlo, Qresult);
            // TODO is this instrc ok?
//        code.FMOV(Qresult->toD(), Dlo);  // TODO: Move eliminiation
//        code.add_d(Xbyak_loongarch64::VRegSelector{Qresult->getIdx()}.D()[1], Xbyak_loongarch64::VRegSelector{Dhi->getIdx()}.D()[0], code.zero);
        }
    }

    template<>
    void
    EmitIR<IR::Opcode::LeastSignificantWord>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        auto Wresult = ctx.reg_alloc.WriteW(inst);
        auto Xoperand = ctx.reg_alloc.ReadX(args[0]);
        RegAlloc::Realize(Wresult, Xoperand);
        code.add_w(Wresult, code.zero, Xoperand);

//    code.add_d(Wresult, Xoperand->toW(), code.zero);  // TODO: Zext elimination
    }

    template<>
    void
    EmitIR<IR::Opcode::LeastSignificantHalf>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        auto Wresult = ctx.reg_alloc.WriteW(inst);
        auto Woperand = ctx.reg_alloc.ReadW(args[0]);
        RegAlloc::Realize(Wresult, Woperand);
        code.bstrpick_w(Wresult, Woperand, 15, 0);

//    code.UXTH(Wresult, Woperand);  // TODO: Zext elimination
    }

    template<>
    void
    EmitIR<IR::Opcode::LeastSignificantByte>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        auto Wresult = ctx.reg_alloc.WriteW(inst);
        auto Woperand = ctx.reg_alloc.ReadW(args[0]);
        RegAlloc::Realize(Wresult, Woperand);
        code.bstrpick_w(Wresult, Woperand, 7, 0);

//    code.UXTB(Wresult, Woperand);  // TODO: Zext elimination
    }

    template<>
    void
    EmitIR<IR::Opcode::MostSignificantWord>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        const auto carry_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetCarryFromOp);

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        auto Wresult = ctx.reg_alloc.WriteW(inst);
        auto Xoperand = ctx.reg_alloc.ReadX(args[0]);
        RegAlloc::Realize(Wresult, Xoperand);

        code.srli_d(Wresult, Xoperand, 32);

        if (carry_inst) {
            auto Wcarry = ctx.reg_alloc.WriteW(carry_inst);
            RegAlloc::Realize(Wcarry);

            code.srli_w(Wcarry, Xoperand, 31 - 29);
            code.add_imm(Wscratch0, code.zero, 1 << 29, Wscratch1);
            // TODO what does this mean?
            code.and_(Wcarry, Wcarry, Wscratch0);
        }
    }

    template<>
    void
    EmitIR<IR::Opcode::MostSignificantBit>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        auto Wresult = ctx.reg_alloc.WriteW(inst);
        auto Woperand = ctx.reg_alloc.ReadW(args[0]);
        RegAlloc::Realize(Wresult, Woperand);

        code.srli_w(Wresult, Woperand, 31);
    }

    template<>
    void EmitIR<IR::Opcode::IsZero32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        auto Wresult = ctx.reg_alloc.WriteW(inst);
        auto Woperand = ctx.reg_alloc.ReadW(args[0]);
        RegAlloc::Realize(Wresult, Woperand);
        ctx.reg_alloc.SpillFlags();
        code.sltui(Wresult, Woperand, 0x1);
//    code.xori(Wresult, Woperand, 0x1);
//    code.CMP(Woperand, 0);
//    code.CSET(Wresult, EQ);
    }

    template<>
    void EmitIR<IR::Opcode::IsZero64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        auto Wresult = ctx.reg_alloc.WriteW(inst);
        auto Xoperand = ctx.reg_alloc.ReadX(args[0]);
        RegAlloc::Realize(Wresult, Xoperand);
        ctx.reg_alloc.SpillFlags();
        code.sltui(Wresult, Xoperand, 0x1);
//    code.CMP(Xoperand, 0);
//    code.CSET(Wresult, EQ);
    }

    template<>
    void EmitIR<IR::Opcode::TestBit>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Xresult = ctx.reg_alloc.WriteX(inst);
        auto Xoperand = ctx.reg_alloc.ReadX(args[0]);
        RegAlloc::Realize(Xresult, Xoperand);
        ASSERT(args[1].IsImmediate());
        ASSERT(args[1].GetImmediateU8() < 64);
//    code.srli_d(Xresult, Xoperand, args[1].GetImmediateU8());
//    code.andi(Xresult, Xresult, 0x1);

        code.bstrpick_d(Xresult, Xoperand, args[1].GetImmediateU8(), args[1].GetImmediateU8());
    }

// bind to EmitA32Cond
    template<size_t bitsize>
    static void EmitConditionalSelect(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        const IR::Cond cond = args[0].GetImmediateCond();
        auto Xresult = ctx.reg_alloc.WriteX(inst);
        auto Xthen = ctx.reg_alloc.ReadReg<bitsize>(args[1]);
        auto Xelse = ctx.reg_alloc.ReadReg<bitsize>(args[2]);
        RegAlloc::Realize(Xresult, Xthen, Xelse);
        Xbyak_loongarch64::Label then_rst;

        code.ld_d(Xscratch0, Xstate, offsetof(A64JitState, cpsr_nzcv));
        // FIXME
//        LoadRequiredFlagsForCondFromReg(code, Xscratch2, Xscratch0, args[0].GetImmediateCond());

        switch (args[0].GetImmediateCond()) {
            case IR::Cond::EQ:
                code.andi(Xscratch0, Xscratch0, NZCV::arm_n_flag_mask);
                code.bnez(Xscratch0, then_rst);
                break;
            case IR::Cond::NE:
                code.andi(Xscratch0, Xscratch0, NZCV::arm_n_flag_mask);
                code.beqz(Xscratch0, then_rst);
                break;
            case IR::Cond::CS:
                code.andi(Xscratch0, Xscratch0, NZCV::arm_c_flag_mask);
                code.bnez(Xscratch0, then_rst);
                break;
            case IR::Cond::CC:
                code.andi(Xscratch0, Xscratch0, NZCV::arm_c_flag_mask);
                code.beqz(Xscratch0, then_rst);
                break;
            case IR::Cond::MI:
                code.andi(Xscratch0, Xscratch0, NZCV::arm_n_flag_mask);
                code.bnez(Xscratch0, then_rst);
                break;
            case IR::Cond::PL:
                code.andi(Xscratch0, Xscratch0, NZCV::arm_n_flag_mask);
                code.beqz(Xscratch0, then_rst);
                break;
            case IR::Cond::VS:
                code.andi(Xscratch0, Xscratch0, NZCV::arm_v_flag_mask);
                code.bnez(Xscratch0, then_rst);
                break;
            case IR::Cond::VC:
                code.andi(Xscratch0, Xscratch0, NZCV::arm_v_flag_mask);
                code.beqz(Xscratch0, then_rst);
                break;
            case IR::Cond::HI:
                code.andi(Xscratch0, Xscratch0, NZCV::arm_c_flag_mask | NZCV::arm_z_flag_mask);
                code.addi_d(Xscratch1, code.zero, NZCV::arm_hi_flag_mask);
                code.beq(Xscratch0, Xscratch1, then_rst);
                break;
            case IR::Cond::LS:
                code.andi(Xscratch0, Xscratch0, NZCV::arm_c_flag_mask | NZCV::arm_z_flag_mask);
                code.addi_d(Xscratch1, code.zero, NZCV::arm_ls_flag_mask);
                code.beq(Xscratch0, Xscratch1, then_rst);
                break;
            case IR::Cond::GT:
            case IR::Cond::GE:
                code.andi(Xscratch0, Xscratch0, NZCV::arm_gt_flag_mask1);
                if (cond == IR::Cond::GE) {
                    code.andi(Xscratch0, Xscratch0, NZCV::arm_ge_flag_mask);
                }
                code.beqz(Xscratch0, then_rst);
                code.addi_d(Xscratch1, code.zero, NZCV::arm_ge_flag_mask);
                code.beq(Xscratch0, Xscratch1, then_rst);
                break;
            case IR::Cond::LE:
                code.andi(Xscratch1, Xscratch0, NZCV::arm_z_flag_mask);
                code.bnez(Xscratch1, then_rst);
                break;
            case IR::Cond::LT:
                code.andi(Xscratch0, Xscratch0, NZCV::arm_n_flag_mask | NZCV::arm_v_flag_mask);
                code.addi_d(Xscratch1, code.zero, NZCV::arm_v_flag_mask);
                code.beq(Xscratch0, Xscratch1, then_rst);
                code.addi_d(Xscratch1, code.zero, NZCV::arm_n_flag_mask);
                code.beq(Xscratch0, Xscratch1, then_rst);
                break;
            default:
                ASSERT_MSG(false, "Unknown cond {}", static_cast<size_t>(cond));
                break;
        }
        code.add_d(Xthen, code.zero, Xelse);
        code.align(8);
        code.L(then_rst);
        code.add_d(Xelse, code.zero, Xthen);

        code.add_d(Xresult, code.zero, Xelse);

//        ctx.reg_alloc.DefineValue(inst, else_);
    }

    template<>
    void
    EmitIR<IR::Opcode::ConditionalSelect32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitConditionalSelect<32>(code, ctx, inst);
    }

    template<>
    void
    EmitIR<IR::Opcode::ConditionalSelect64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitConditionalSelect<64>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::ConditionalSelectNZCV>(BlockOfCode &code, EmitContext &ctx,
                                                   IR::Inst *inst) {
        EmitIR<IR::Opcode::ConditionalSelect32>(code, ctx, inst);
    }

    template<>
    void
    EmitIR<IR::Opcode::LogicalShiftLeft32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        const auto carry_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetCarryFromOp);

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto &operand_arg = args[0];
        auto &shift_arg = args[1];
        auto &carry_arg = args[2];

        if (!carry_inst) {
            if (shift_arg.IsImmediate()) {
                const u8 shift = shift_arg.GetImmediateU8();
                auto Wresult = ctx.reg_alloc.WriteW(inst);
                auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
                RegAlloc::Realize(Wresult, Woperand);

                if (shift <= 31) {
                    code.slli_w(Wresult, Woperand, shift);
                } else {
                    code.add_d(Wresult, code.zero, code.zero);
                }
            } else {
                auto Wresult = ctx.reg_alloc.WriteW(inst);
                auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
                auto Wshift = ctx.reg_alloc.ReadW(shift_arg);
                RegAlloc::Realize(Wresult, Woperand, Wshift);
                ctx.reg_alloc.SpillFlags();

                code.andi(Wscratch0, Wshift, 0xff);
                code.sll_w(Wresult, Woperand, Wscratch0);
                code.srli_w(Wscratch0, Wscratch0, 32);
                code.masknez(Wresult, Wresult, Wscratch0);

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

                    code.bstrpick_w(Wcarry_out, Woperand, 32 - shift, 32 - shift);
//                code.UBFX(Wcarry_out, Woperand, 32 - shift, 1);
                    code.slli_w(Wcarry_out, Wcarry_out, 29);
                    code.slli_w(Wresult, Woperand, shift);
                } else if (shift > 32) {
                    auto Wresult = ctx.reg_alloc.WriteW(inst);
                    auto Wcarry_out = ctx.reg_alloc.WriteW(carry_inst);
                    RegAlloc::Realize(Wresult, Wcarry_out);

                    code.add_d(Wresult, code.zero, code.zero);
                    code.add_d(Wcarry_out, code.zero, code.zero);
                } else {
                    auto Wresult = ctx.reg_alloc.WriteW(inst);
                    auto Wcarry_out = ctx.reg_alloc.WriteW(carry_inst);
                    auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
                    RegAlloc::Realize(Wresult, Wcarry_out, Woperand);
                    code.add_d(Wcarry_out, code.zero, code.zero);
                    code.bstrins_w(Wcarry_out, Woperand, 29, 29);
//                code.UBFIZ(Wcarry_out, Woperand, 29, 1);
                    code.add_d(Wresult, code.zero, code.zero);
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

                code.andi(Wscratch1, Wshift, 0xff);
                code.beqz(Wscratch1, zero);

                code.sub_w(Wscratch0, code.zero, Wshift);
                code.srl_w(Wcarry_out, Woperand, Wscratch0);
                code.sll_w(Wresult, Woperand, Wshift);
                code.bstrins_w(Wcarry_out, Wcarry_out, 29, 29);
                code.slti(Wscratch2, Wscratch1, 32);
                code.maskeqz(Wresult, Wresult, Wscratch2);
                code.slti(Wscratch2, Wscratch1, 33);
                code.maskeqz(Wcarry_out, Wcarry_out, Wscratch2);
                code.b(end);

                code.L(zero);
                code.add_d(*Wresult, Woperand, code.zero);
                if (carry_arg.IsImmediate()) {

                    code.add_imm(Wcarry_out, code.zero, carry_arg.GetImmediateU32() << 29, Wscratch0);
                } else {
                    code.add_d(*Wcarry_out, Wcarry_in, code.zero);
                }

                code.L(end);
            }
        }
    }

    template<>
    void
    EmitIR<IR::Opcode::LogicalShiftLeft64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        if (args[1].IsImmediate()) {
            const u8 shift = args[1].GetImmediateU8();
            auto Xresult = ctx.reg_alloc.WriteX(inst);
            auto Xoperand = ctx.reg_alloc.ReadX(args[0]);
            RegAlloc::Realize(Xresult, Xoperand);

            if (shift <= 63) {
                code.slli_d(Xresult, Xoperand, shift);
            } else {
                code.add_d(Xresult, code.zero, code.zero);
            }
        } else {
            auto Xresult = ctx.reg_alloc.WriteX(inst);
            auto Xoperand = ctx.reg_alloc.ReadX(args[0]);
            auto Xshift = ctx.reg_alloc.ReadX(args[1]);
            RegAlloc::Realize(Xresult, Xoperand, Xshift);
            ctx.reg_alloc.SpillFlags();

            code.andi(Xscratch0, Xshift, 0xff);
            code.sll_d(Xresult, Xoperand, Xscratch0);
            code.slti(Xscratch0, Xscratch0, 64);
            code.maskeqz(Xresult, Xresult, Xscratch0);
        }
    }

    template<>
    void
    EmitIR<IR::Opcode::LogicalShiftRight32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        const auto carry_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetCarryFromOp);

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto &operand_arg = args[0];
        auto &shift_arg = args[1];
        auto &carry_arg = args[2];

        if (!carry_inst) {
            if (shift_arg.IsImmediate()) {
                const u8 shift = shift_arg.GetImmediateU8();
                auto Wresult = ctx.reg_alloc.WriteW(inst);
                auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
                RegAlloc::Realize(Wresult, Woperand);

                if (shift <= 31) {
                    code.srli_w(Wresult, Woperand, shift);
                } else {
                    code.add_d(Wresult, code.zero, code.zero);
                }
            } else {
                auto Wresult = ctx.reg_alloc.WriteW(inst);
                auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
                auto Wshift = ctx.reg_alloc.ReadW(shift_arg);
                RegAlloc::Realize(Wresult, Woperand, Wshift);
                ctx.reg_alloc.SpillFlags();

                code.andi(Wscratch0, Wshift, 0xff);
                code.srl_w(Wresult, Woperand, Wscratch0);
                code.slti(Wscratch0, Wscratch0, 32);
                code.maskeqz(Wresult, Wresult, Wscratch0);

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
                    code.bstrpick_w(Wcarry_out, Woperand, shift - 1, shift - 1);
                    code.slli_w(Wcarry_out, Wcarry_out, 29);
                    code.srli_w(Wresult, Woperand, shift);
                } else if (shift > 32) {
                    auto Wresult = ctx.reg_alloc.WriteW(inst);
                    auto Wcarry_out = ctx.reg_alloc.WriteW(carry_inst);
                    RegAlloc::Realize(Wresult, Wcarry_out);

                    code.add_d(Wresult, code.zero, code.zero);
                    code.add_d(Wcarry_out, code.zero, code.zero);
                } else {
                    auto Wresult = ctx.reg_alloc.WriteW(inst);
                    auto Wcarry_out = ctx.reg_alloc.WriteW(carry_inst);
                    auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
                    RegAlloc::Realize(Wresult, Wcarry_out, Woperand);

                    code.srli_w(Wcarry_out, Woperand, 31 - 29);
                    code.andi(Wcarry_out, Wcarry_out, 1 << 29);
                    code.add_d(Wresult, code.zero, code.zero);
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

                code.andi(Wscratch1, Wshift, 0xff);
                code.beqz(Wscratch1, zero);

                code.sub_imm(Wscratch0, Wshift, 1, code.t0);
                code.srl_w(Wcarry_out, Woperand, Wscratch0);
                code.srl_w(Wresult, Woperand, Wshift);
                code.bstrins_w(Wcarry_out, Wcarry_out, 29, 29);
                code.slti(Wscratch2, Wscratch1, 32);
                code.maskeqz(Wresult, Wresult, Wscratch2);
                code.slti(Wscratch2, Wscratch1, 33);
                code.maskeqz(Wcarry_out, Wcarry_out, Wscratch2);
                code.b(end);

                code.L(zero);
                code.add_d(*Wresult, Woperand, code.zero);
                if (carry_arg.IsImmediate()) {
                    code.add_imm(Wcarry_out, code.zero, carry_arg.GetImmediateU32() << 29, Wscratch0);
                } else {
                    code.add_d(*Wcarry_out, Wcarry_in, code.zero);
                }

                code.L(end);
            }
        }
    }

    template<>
    void
    EmitIR<IR::Opcode::LogicalShiftRight64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        if (args[1].IsImmediate()) {
            const u8 shift = args[1].GetImmediateU8();
            auto Xresult = ctx.reg_alloc.WriteX(inst);
            auto Xoperand = ctx.reg_alloc.ReadX(args[0]);
            RegAlloc::Realize(Xresult, Xoperand);

            if (shift <= 63) {
                code.srli_d(Xresult, Xoperand, shift);
            } else {
                code.add_d(Xresult, code.zero, code.zero);
            }
        } else {
            auto Xresult = ctx.reg_alloc.WriteX(inst);
            auto Xoperand = ctx.reg_alloc.ReadX(args[0]);
            auto Xshift = ctx.reg_alloc.ReadX(args[1]);
            RegAlloc::Realize(Xresult, Xoperand, Xshift);
            ctx.reg_alloc.SpillFlags();

            code.andi(Xscratch0, Xshift, 0xff);
            code.srl_d(Xresult, Xoperand, Xscratch0);
            code.slti(Xscratch0, Xscratch0, 64);
            code.maskeqz(Xresult, Xresult, Xscratch0);

        }
    }

    template<>
    void EmitIR<IR::Opcode::ArithmeticShiftRight32>(BlockOfCode &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        const auto carry_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetCarryFromOp);

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto &operand_arg = args[0];
        auto &shift_arg = args[1];
        auto &carry_arg = args[2];

        if (!carry_inst) {
            if (shift_arg.IsImmediate()) {
                const u8 shift = shift_arg.GetImmediateU8();
                auto Wresult = ctx.reg_alloc.WriteW(inst);
                auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
                RegAlloc::Realize(Wresult, Woperand);
                code.srai_w(Wresult, Woperand, shift <= 31 ? shift : 31);
            } else {
                auto Wresult = ctx.reg_alloc.WriteW(inst);
                auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
                auto Wshift = ctx.reg_alloc.ReadW(shift_arg);
                RegAlloc::Realize(Wresult, Woperand, Wshift);
                ctx.reg_alloc.SpillFlags();

                code.andi(Wscratch0, Wshift, 0xff);
                code.addi_d(Wscratch1, code.zero, 31);
//            code.add_d(Wscratch1, 31, code.zero);
                code.andi(Wscratch0, Wscratch0, 0x1F);
//            code.CMP(Wscratch0, 31);
//            code.CSEL(Wscratch0, Wscratch0, Wscratch1, LS);
                code.sra_w(Wresult, Woperand, Wscratch0);
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

                    code.bstrpick_w(Wcarry_out, Woperand, shift - 1, shift - 1);
                    code.slli_w(Wcarry_out, Wcarry_out, 29);
                    code.srai_w(Wresult, Woperand, shift);
                } else {
                    auto Wresult = ctx.reg_alloc.WriteW(inst);
                    auto Wcarry_out = ctx.reg_alloc.WriteW(carry_inst);
                    auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
                    RegAlloc::Realize(Wresult, Wcarry_out, Woperand);

                    code.srai_w(Wresult, Woperand, 31);
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

                code.andi(Wscratch0, Wshift, 0xff);
                code.beqz(Wscratch0, zero);
//            code.B(EQ, zero);
// TODO is this right?
                code.andi(Wscratch0, Wscratch0, 0x3F);
//            code.add_d(Wscratch1, 63, code.zero);
//            code.CMP(Wscratch0, 63);
//            code.CSEL(Wscratch0, Wscratch0, Wscratch1, LS);

                code.bstrins_d(Wresult, Woperand, 63, 0);
//            code.SXTW(Wresult->toX(), Woperand);
                code.sub_imm(Wscratch1, Wscratch0, 1, code.t0);

                code.sra_d(Wcarry_out, Wresult, Xscratch1);
                code.sra_d(Wresult, Wresult, Xscratch0);
                code.bstrins_w(Wcarry_out, Wcarry_out, 29, 29);
//            code.UBFIZ(Wcarry_out, Wcarry_out, 29, 1);
                code.add_d(*Wresult, Wresult, code.zero);

                code.b(end);

                code.L(zero);
                code.add_d(*Wresult, Woperand, code.zero);
                if (carry_arg.IsImmediate()) {
                    code.add_imm(Wcarry_out, code.zero, carry_arg.GetImmediateU32() << 29, Wscratch0);
                } else {
                    code.add_d(*Wcarry_out, Wcarry_in, code.zero);
                }

                code.L(end);
            }
        }
    }

    template<>
    void EmitIR<IR::Opcode::ArithmeticShiftRight64>(BlockOfCode &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto &operand_arg = args[0];
        auto &shift_arg = args[1];

        if (shift_arg.IsImmediate()) {
            const u8 shift = shift_arg.GetImmediateU8();
            auto Xresult = ctx.reg_alloc.WriteX(inst);
            auto Xoperand = ctx.reg_alloc.ReadX(operand_arg);
            RegAlloc::Realize(Xresult, Xoperand);
            code.srai_d(Xresult, Xoperand, shift <= 63 ? shift : 63);
        } else {
            auto Xresult = ctx.reg_alloc.WriteX(inst);
            auto Xoperand = ctx.reg_alloc.ReadX(operand_arg);
            auto Xshift = ctx.reg_alloc.ReadX(shift_arg);
            RegAlloc::Realize(Xresult, Xoperand, Xshift);
            code.sra_d(Xresult, Xoperand, Xshift);
        }
    }

    template<>
    void EmitIR<IR::Opcode::RotateRight32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        const auto carry_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetCarryFromOp);

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto &operand_arg = args[0];
        auto &shift_arg = args[1];
        auto &carry_arg = args[2];

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

            code.rotri_w(Wresult, Woperand, shift);

            if (carry_inst) {
                auto Wcarry_out = ctx.reg_alloc.WriteW(carry_inst);
                RegAlloc::Realize(Wcarry_out);

                code.rotri_w(Wcarry_out, Woperand, ((shift + 31) - 29) % 32);
                code.andi(Wcarry_out, Wcarry_out, 1 << 29);
            }
        } else {
            auto Wresult = ctx.reg_alloc.WriteW(inst);
            auto Woperand = ctx.reg_alloc.ReadW(operand_arg);
            auto Wshift = ctx.reg_alloc.ReadW(shift_arg);
            RegAlloc::Realize(Wresult, Woperand, Wshift);

            code.rotr_w(Wresult, Woperand, Wshift);

            if (carry_inst && carry_arg.IsImmediate()) {
                const u32 carry_in = carry_arg.GetImmediateU32() << 29;
                auto Wcarry_out = ctx.reg_alloc.WriteW(carry_inst);
                RegAlloc::Realize(Wcarry_out);
                ctx.reg_alloc.SpillFlags();

//            code.TST(Wshift, 0xff);
                code.addi_w(Wscratch2, code.zero, 0xff);
                code.srli_w(Wcarry_out, Wresult, 31 - 29);
                code.andi(Wcarry_out, Wcarry_out, 1 << 29);
                if (carry_in) {
                    Xbyak_loongarch64::Label end;
                    code.add_imm(Wscratch0, code.zero, carry_in, Wscratch1);

                    code.bne(Wshift, Wscratch2, end);
                    code.add_d(Wcarry_out, code.zero, Wscratch0);
                    code.L(end);
                } else {
                    code.sub_w(Wscratch0, Wshift, Wscratch2);
                    code.maskeqz(Wcarry_out, Wcarry_out, Wscratch0);
                }
            } else if (carry_inst) {
                auto Wcarry_in = ctx.reg_alloc.ReadW(carry_arg);
                auto Wcarry_out = ctx.reg_alloc.WriteW(carry_inst);
                RegAlloc::Realize(Wcarry_out, Wcarry_in);
                ctx.reg_alloc.SpillFlags();

                Xbyak_loongarch64::Label wciend;
                code.srli_w(Wcarry_out, Wresult, 31 - 29);
                code.andi(Wcarry_out, Wcarry_out, 1 << 29);
                code.addi_w(Wscratch0, code.zero, 0xff);
                code.bne(Wshift, Wscratch0, wciend);
                code.add_w(Wcarry_out, code.zero, Wcarry_in);
                code.L(wciend);
            }
        }
    }

    template<>
    void EmitIR<IR::Opcode::RotateRight64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto &operand_arg = args[0];
        auto &shift_arg = args[1];

        if (shift_arg.IsImmediate()) {
            const u8 shift = shift_arg.GetImmediateU8();
            auto Xresult = ctx.reg_alloc.WriteX(inst);
            auto Xoperand = ctx.reg_alloc.ReadX(operand_arg);
            RegAlloc::Realize(Xresult, Xoperand);
            code.rotri_d(Xresult, Xoperand, shift);
        } else {
            auto Xresult = ctx.reg_alloc.WriteX(inst);
            auto Xoperand = ctx.reg_alloc.ReadX(operand_arg);
            auto Xshift = ctx.reg_alloc.ReadX(shift_arg);
            RegAlloc::Realize(Xresult, Xoperand, Xshift);
            code.rotr_d(Xresult, Xoperand, Xshift);
        }
    }

    template<>
    void
    EmitIR<IR::Opcode::RotateRightExtended>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        const auto carry_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetCarryFromOp);

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Wresult = ctx.reg_alloc.WriteW(inst);
        auto Woperand = ctx.reg_alloc.ReadW(args[0]);

        if (args[1].IsImmediate()) {
            RegAlloc::Realize(Wresult, Woperand);

            code.srli_w(Wresult, Woperand, 1);
            if (args[1].GetImmediateU1()) {
                code.addi_w(Wscratch0, code.zero, 0x1);
                code.bstrins_w(Wresult, Wscratch0, 31, 31);
//            code.ORR(Wresult, Wresult, 0x8000'0000);
            }
        } else {
            auto Wcarry_in = ctx.reg_alloc.ReadW(args[1]);
            RegAlloc::Realize(Wresult, Woperand, Wcarry_in);
            // TODO WHY 29?
            code.srli_w(Wscratch0, Wcarry_in, 29);
            code.alsl_d(Wresult, Wscratch0, Woperand, 31);
            code.rotri_d(Wresult, Wresult, 1);
//            code.EXTR(Wresult, Wscratch0, Woperand, 1);
        }

        if (carry_inst) {
            auto Wcarry_out = ctx.reg_alloc.WriteW(carry_inst);
            RegAlloc::Realize(Wcarry_out);
            code.bstrins_w(Wcarry_out, Woperand, 29, 29);
//        code.UBFIZ(Wcarry_out, Woperand, 29, 1);
        }
    }

    template<typename ShiftI, typename ShiftR>
    static void EmitMaskedShift32(BlockOfCode &, EmitContext &ctx, IR::Inst *inst, ShiftI si_fn,
                                  ShiftR sr_fn) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto &operand_arg = args[0];
        auto &shift_arg = args[1];

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
    static void EmitMaskedShift64(BlockOfCode &, EmitContext &ctx, IR::Inst *inst, ShiftI si_fn,
                                  ShiftR sr_fn) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto &operand_arg = args[0];
        auto &shift_arg = args[1];

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
    void EmitIR<IR::Opcode::LogicalShiftLeftMasked32>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitMaskedShift32(
                code, ctx, inst,
                [&](auto &Wresult, auto &Woperand, auto shift) { code.slli_w(Wresult, Woperand, shift); },
                [&](auto &Wresult, auto &Woperand, auto &Wshift) { code.sll_w(Wresult, Woperand, Wshift); });
    }

    template<>
    void EmitIR<IR::Opcode::LogicalShiftLeftMasked64>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitMaskedShift64(
                code, ctx, inst,
                [&](auto &Xresult, auto &Xoperand, auto shift) { code.slli_d(Xresult, Xoperand, shift); },
                [&](auto &Xresult, auto &Xoperand, auto &Xshift) { code.sll_d(Xresult, Xoperand, Xshift); });
    }

    template<>
    void EmitIR<IR::Opcode::LogicalShiftRightMasked32>(BlockOfCode &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitMaskedShift32(
                code, ctx, inst,
                [&](auto &Wresult, auto &Woperand, auto shift) { code.srli_w(Wresult, Woperand, shift); },
                [&](auto &Wresult, auto &Woperand, auto &Wshift) { code.srl_w(Wresult, Woperand, Wshift); });
    }

    template<>
    void EmitIR<IR::Opcode::LogicalShiftRightMasked64>(BlockOfCode &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitMaskedShift64(
                code, ctx, inst,
                [&](auto &Xresult, auto &Xoperand, auto shift) { code.srli_d(Xresult, Xoperand, shift); },
                [&](auto &Xresult, auto &Xoperand, auto &Xshift) { code.srl_d(Xresult, Xoperand, Xshift); });
    }

    template<>
    void EmitIR<IR::Opcode::ArithmeticShiftRightMasked32>(BlockOfCode &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitMaskedShift32(
                code, ctx, inst,
                [&](auto &Wresult, auto &Woperand, auto shift) { code.srai_w(Wresult, Woperand, shift); },
                [&](auto &Wresult, auto &Woperand, auto &Wshift) { code.sra_w(Wresult, Woperand, Wshift); });
    }

    template<>
    void EmitIR<IR::Opcode::ArithmeticShiftRightMasked64>(BlockOfCode &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitMaskedShift64(
                code, ctx, inst,
                [&](auto &Xresult, auto &Xoperand, auto shift) { code.srai_d(Xresult, Xoperand, shift); },
                [&](auto &Xresult, auto &Xoperand, auto &Xshift) { code.sra_d(Xresult, Xoperand, Xshift); });
    }

    template<>
    void
    EmitIR<IR::Opcode::RotateRightMasked32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitMaskedShift32(
                code, ctx, inst,
                [&](auto &Wresult, auto &Woperand, auto shift) { code.rotri_w(Wresult, Woperand, shift); },
                [&](auto &Wresult, auto &Woperand, auto &Wshift) { code.rotr_w(Wresult, Woperand, Wshift); });
    }

    template<>
    void
    EmitIR<IR::Opcode::RotateRightMasked64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitMaskedShift64(
                code, ctx, inst,
                [&](auto &Xresult, auto &Xoperand, auto shift) { code.rotri_d(Xresult, Xoperand, shift); },
                [&](auto &Xresult, auto &Xoperand, auto &Xshift) { code.rotr_d(Xresult, Xoperand, Xshift); });
    }

    template<size_t bitsize, typename EmitFn>
    static void MaybeAddSubImm(BlockOfCode &code, u64 imm, EmitFn emit_fn) {
        static_assert(bitsize == 32 || bitsize == 64);
        if constexpr (bitsize == 32) {
            imm = static_cast<u32>(imm);
        }
//        if (Xbyak_loongarch64::AddSubImm::is_valid(imm)) {
//            emit_fn(imm);
//        } else {
        code.add_imm(Xscratch0, code.zero, imm, Xscratch1);
        emit_fn(Xscratch0);
//            code.add_d(Rscratch0<bitsize>(), imm, code.zero);
//            emit_fn(Rscratch0<bitsize>());
//        }
    }

    template<size_t bitsize>
    static void EmitAdd(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        const auto carry_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetCarryFromOp);
        const auto nzcv_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetNZCVFromOp);
        const auto overflow_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetOverflowFromOp);

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        auto Rresult = ctx.reg_alloc.WriteReg<bitsize>(inst);
        auto Ra = ctx.reg_alloc.ReadReg<bitsize>(args[0]);
        auto Rb = ctx.reg_alloc.ReadReg<bitsize>(args[1]);
        auto carry_in = ctx.reg_alloc.ReadReg<bitsize>(args[2]);
        RegAlloc::Realize(Rresult, Ra, Rb, carry_in);

        decltype(&BlockOfCode::add_d) addfn;
        if constexpr (bitsize == 32) {
            addfn = &BlockOfCode::add_w;
        } else if constexpr (bitsize == 64) {
            addfn = &BlockOfCode::add_d;
        }
        (code.*addfn)(Rresult, Ra, Rb);
        (code.*addfn)(Rresult, Rresult, carry_in);

        if (overflow_inst) {
            code.xor_(Xscratch0, Rb, Ra);
            code.xor_(Xscratch1, Rresult, Ra);
            code.and_(Xscratch0, Xscratch0, Xscratch1);

            auto Woverflow = ctx.reg_alloc.WriteW(overflow_inst);
            RegAlloc::Realize(Woverflow);
            code.slt(Woverflow, Xscratch0, code.zero);
        }
        if (carry_inst) {
            auto Wcarry = ctx.reg_alloc.WriteW(carry_inst);
            RegAlloc::Realize(Wcarry);
            code.sltu(Wcarry, Rresult, Ra);
        }
        if (nzcv_inst) {
            auto Wflags = ctx.reg_alloc.WriteFlags(nzcv_inst);
            // TODO how to impl?
            RegAlloc::Realize(Wflags);
        }

    }

    template<size_t bitsize>
    static void EmitSub(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        const auto carry_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetCarryFromOp);
        const auto nzcv_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetNZCVFromOp);
        const auto overflow_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetOverflowFromOp);

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        auto Rresult = ctx.reg_alloc.WriteReg<bitsize>(inst);
        auto Ra = ctx.reg_alloc.ReadReg<bitsize>(args[0]);
        auto Rb = ctx.reg_alloc.ReadReg<bitsize>(args[1]);
        auto carry_in = ctx.reg_alloc.ReadReg<bitsize>(args[2]);
        RegAlloc::Realize(Rresult, Ra, Rb, carry_in);

        code.add_d(Rb, Rb, carry_in);
        code.sub_d(Rresult, Wscratch2, Ra);


        if (overflow_inst) {
            auto Woverflow = ctx.reg_alloc.WriteW(overflow_inst);
            RegAlloc::Realize(Woverflow);

            code.xor_(Xscratch0, Rb, Ra);
            code.xor_(Xscratch1, Rresult, Ra);
            code.and_(Xscratch0, Xscratch0, Xscratch1);
            code.slt(Woverflow, Xscratch0, code.zero);
        }
        if (carry_inst) {
            auto Wcarry = ctx.reg_alloc.WriteW(carry_inst);
            RegAlloc::Realize(Wcarry);
            code.sltu(Wcarry, Rb, Ra);
        }
        if (nzcv_inst) {
            auto Wflags = ctx.reg_alloc.WriteFlags(nzcv_inst);
            // TODO how to impl?
            RegAlloc::Realize(Wflags);
        }

    }


    template<>
    void EmitIR<IR::Opcode::Add32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitAdd<32>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::Add64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitAdd<64>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::Sub32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitSub<32>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::Sub64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitSub<64>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::Mul32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp<32>(
                code, ctx, inst,
                [&](auto &Wresult, auto &Wa, auto &Wb) { code.mul_w(Wresult, Wa, Wb); });
    }

    template<>
    void EmitIR<IR::Opcode::Mul64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp<64>(
                code, ctx, inst,
                [&](auto &Xresult, auto &Xa, auto &Xb) { code.mul_d(Xresult, Xa, Xb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::SignedMultiplyHigh64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Xresult = ctx.reg_alloc.WriteX(inst);
        auto Xop1 = ctx.reg_alloc.ReadX(args[0]);
        auto Xop2 = ctx.reg_alloc.ReadX(args[1]);
        RegAlloc::Realize(Xresult, Xop1, Xop2);

        code.mulh_d(Xresult, Xop1, Xop2);
    }

    template<>
    void EmitIR<IR::Opcode::UnsignedMultiplyHigh64>(BlockOfCode &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Xresult = ctx.reg_alloc.WriteX(inst);
        auto Xop1 = ctx.reg_alloc.ReadX(args[0]);
        auto Xop2 = ctx.reg_alloc.ReadX(args[1]);
        RegAlloc::Realize(Xresult, Xop1, Xop2);

        code.mulh_du(Xresult, Xop1, Xop2);
    }

    template<>
    void EmitIR<IR::Opcode::UnsignedDiv32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp<32>(
                code, ctx, inst,
                [&](auto &Wresult, auto &Wa, auto &Wb) { code.div_wu(Wresult, Wa, Wb); });
    }

    template<>
    void EmitIR<IR::Opcode::UnsignedDiv64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp<64>(
                code, ctx, inst,
                [&](auto &Xresult, auto &Xa, auto &Xb) { code.div_du(Xresult, Xa, Xb); });
    }

    template<>
    void EmitIR<IR::Opcode::SignedDiv32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp<32>(
                code, ctx, inst,
                [&](auto &Wresult, auto &Wa, auto &Wb) { code.div_w(Wresult, Wa, Wb); });
    }

    template<>
    void EmitIR<IR::Opcode::SignedDiv64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp<64>(
                code, ctx, inst,
                [&](auto &Xresult, auto &Xa, auto &Xb) { code.div_d(Xresult, Xa, Xb); });
    }


    template<size_t bitsize, typename EmitFn>
    static void MaybeBitImm(BlockOfCode &code, u64 imm, EmitFn emit_fn) {
        static_assert(bitsize == 32 || bitsize == 64);
        if constexpr (bitsize == 32) {
            imm = static_cast<u32>(imm);
        }
        code.add_imm(Rscratch0<bitsize>(), code.zero, imm, Rscratch1<bitsize>());
//            code.add_d(Rscratch0<bitsize>(), imm, code.zero);
        emit_fn(Rscratch0<bitsize>());
    }

    template<size_t bitsize, typename EmitFn1, typename EmitFn2 = std::nullptr_t>
    static void
    EmitBitOp(BlockOfCode &, EmitContext &ctx, IR::Inst *inst, EmitFn1 emit_without_flags,
              EmitFn2  = nullptr) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Rresult = ctx.reg_alloc.WriteReg<bitsize>(inst);
        auto Ra = ctx.reg_alloc.ReadReg<bitsize>(args[0]);
        auto Rb = ctx.reg_alloc.ReadW(args[1]);
        RegAlloc::Realize(Rresult, Ra, Rb);

//        if constexpr (!std::is_same_v<EmitFn2, std::nullptr_t>) {
//            const auto nz_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetNZFromOp);
//            const auto nzcv_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetNZCVFromOp);
//            ASSERT(!(nz_inst && nzcv_inst));
//            const auto flag_inst = nz_inst ? nz_inst : nzcv_inst;
//
//            if (flag_inst) {
//                auto Wflags = ctx.reg_alloc.WriteW(flag_inst);
//                    emit_with_flags(Rresult, Ra, Rb);
//                return;
//            }
//        }

        emit_without_flags(Rresult, Ra, Rb);
    }

    template<size_t bitsize>
    static void EmitAndNot(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        const auto nz_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetNZFromOp);
        const auto nzcv_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetNZCVFromOp);
        ASSERT(!(nz_inst && nzcv_inst));
        const auto flag_inst = nz_inst ? nz_inst : nzcv_inst;

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Rresult = ctx.reg_alloc.WriteReg<bitsize>(inst);
        auto Ra = ctx.reg_alloc.ReadReg<bitsize>(args[0]);
        RegAlloc::Realize(Rresult, Ra);

        if (args[1].IsImmediate()) {
            code.add_imm(Xscratch0, code.zero,
                         bitsize == 32 ?
                         static_cast<u32>(args[1].GetImmediateU64()) : args[1].GetImmediateU64(),
                         Wscratch2);
        }
        code.andn(Rresult, Ra, Xscratch0);

        // TODO how to impl flag
        if (flag_inst) {
            auto Wflags = ctx.reg_alloc.WriteFlags(flag_inst);

            RegAlloc::Realize(Wflags);
        }

    }

    // TODO flags
    template<>
    void EmitIR<IR::Opcode::And32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitBitOp<32>(
                code, ctx, inst,
                [&](auto &result, auto &a, auto &b) {
                    code.bstrpick_d(a, a, 31, 0);
                    code.and_(result, a, b);
                }
                // FIXME
//                ,[&](auto &result, auto &a, auto &b) { code.ANDS(result, a, b); }
        );
    }

    template<>
    void EmitIR<IR::Opcode::And64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitBitOp<64>(
                code, ctx, inst,
                [&](auto &result, auto &a, auto &b) { code.and_(result, a, b); }
//               FIXME
//                ,[&](auto &result, auto &a, auto &b) { code.ANDS(result, a, b); }
        );
    }

    template<>
    void EmitIR<IR::Opcode::AndNot32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitAndNot<32>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::AndNot64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitAndNot<64>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::Eor32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitBitOp<32>(
                code, ctx, inst,
                [&](auto &result, auto &a, auto &b) { code.xor_(result, a, b); });
    }

    template<>
    void EmitIR<IR::Opcode::Eor64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitBitOp<64>(
                code, ctx, inst,
                [&](auto &result, auto &a, auto &b) { code.xor_(result, a, b); });
    }

    template<>
    void EmitIR<IR::Opcode::Or32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitBitOp<32>(
                code, ctx, inst,
                [&](auto &result, auto &a, auto &b) { code.or_(result, a, b); });
    }

    template<>
    void EmitIR<IR::Opcode::Or64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitBitOp<64>(
                code, ctx, inst,
                [&](auto &result, auto &a, auto &b) { code.or_(result, a, b); });
    }

    template<>
    void EmitIR<IR::Opcode::Not32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp<32>(
                code, ctx, inst,
                [&](auto &Wresult, auto &ra, auto &rb) { code.xor_(Wresult, ra, rb); });
    }

    template<>
    void EmitIR<IR::Opcode::Not64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp<64>(
                code, ctx, inst,
                [&](auto &Xresult, auto &ra, auto &rb) { code.xor_(Xresult, ra, rb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::SignExtendByteToWord>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp<32>(
                code, ctx, inst,
                [&](auto &Wresult, auto &Woperand) { code.ext_w_b(Wresult, Woperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::SignExtendHalfToWord>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp<32>(
                code, ctx, inst,
                [&](auto &Wresult, auto &Woperand) { code.ext_w_h(Wresult, Woperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::SignExtendByteToLong>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp<64>(
                code, ctx, inst,
                [&](auto &Xresult, auto &Xoperand) {
                    code.ext_w_b(Wscratch0, Xoperand);
                    code.add_w(Xresult, code.zero, Wscratch0);
                });
    }

    template<>
    void
    EmitIR<IR::Opcode::SignExtendHalfToLong>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp<64>(
                code, ctx, inst,
                [&](auto &Xresult, auto &Xoperand) {
                    code.ext_w_h(Wscratch0, Xoperand);
                    code.add_w(Xresult, code.zero, Wscratch0);
                });
    }

    template<>
    void
    EmitIR<IR::Opcode::SignExtendWordToLong>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp<64>(
                code, ctx, inst,
                [&](auto &Xresult, auto &Xoperand) {
                    code.add_w(Xresult, code.zero, Xoperand);
                });
    }

    template<>
    void
    EmitIR<IR::Opcode::ZeroExtendByteToWord>(BlockOfCode &, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        ctx.reg_alloc.DefineAsExisting(inst, args[0]);
    }

    template<>
    void
    EmitIR<IR::Opcode::ZeroExtendHalfToWord>(BlockOfCode &, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        ctx.reg_alloc.DefineAsExisting(inst, args[0]);
    }

    template<>
    void
    EmitIR<IR::Opcode::ZeroExtendByteToLong>(BlockOfCode &, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        ctx.reg_alloc.DefineAsExisting(inst, args[0]);
    }

    template<>
    void
    EmitIR<IR::Opcode::ZeroExtendHalfToLong>(BlockOfCode &, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        // FIXME ZeroExtend should do as bellow
        ctx.reg_alloc.DefineAsExisting(inst, args[0]);
    }

    template<>
    void
    EmitIR<IR::Opcode::ZeroExtendWordToLong>(BlockOfCode &, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        ctx.reg_alloc.DefineAsExisting(inst, args[0]);
    }

    template<>
    void
    EmitIR<IR::Opcode::ZeroExtendLongToQuad>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Xvalue = ctx.reg_alloc.ReadX(args[0]);
        auto Qresult = ctx.reg_alloc.WriteQ(inst);
        RegAlloc::Realize(Xvalue, Qresult);
        code.vxor_v(Qresult, Qresult, Qresult);
        code.vinsgr2vr_d(Qresult, Xvalue, 0);
    }

    template<>
    void EmitIR<IR::Opcode::ByteReverseWord>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp<32>(
                code, ctx, inst,
                [&](auto &Wresult, auto &Woperand) { code.revb_d(Wresult, Woperand); });
    }

    template<>
    void EmitIR<IR::Opcode::ByteReverseHalf>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp<32>(
                code, ctx, inst,
                [&](auto &Wresult, auto &Woperand) { code.revh_2w(Wresult, Woperand); });
    }

    template<>
    void EmitIR<IR::Opcode::ByteReverseDual>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp<64>(
                code, ctx, inst,
                [&](auto &Xresult, auto &Xoperand) { code.revb_d(Xresult, Xoperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::CountLeadingZeros32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp<32>(
                code, ctx, inst,
                [&](auto &Wresult, auto &Woperand) { code.clz_w(Wresult, Woperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::CountLeadingZeros64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp<64>(
                code, ctx, inst,
                [&](auto &Xresult, auto &Xoperand) { code.clz_d(Xresult, Xoperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::ExtractRegister32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        ASSERT(args[2].IsImmediate());

        auto Wresult = ctx.reg_alloc.WriteW(inst);
        auto Wop1 = ctx.reg_alloc.ReadW(args[0]);
        auto Wop2 = ctx.reg_alloc.ReadW(args[1]);
        RegAlloc::Realize(Wresult, Wop1, Wop2);
        const u8 lsb = args[2].GetImmediateU8();
        code.alsl_d(Wop2, Wop2, Wop1, 31);
        code.bstrpick_d(Wresult, Wop2, lsb + 31, lsb);
    }

    template<>
    void
    EmitIR<IR::Opcode::ExtractRegister64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        ASSERT(args[2].IsImmediate());

        auto Xresult = ctx.reg_alloc.WriteX(inst);
        auto Xop1 = ctx.reg_alloc.ReadX(args[0]);
        auto Xop2 = ctx.reg_alloc.ReadX(args[1]);
        RegAlloc::Realize(Xresult, Xop1, Xop2);
        const u8 lsb = args[2].GetImmediateU8();
        code.srli_d(Xop1, Xop1, lsb);
        code.slli_d(Xop2, Xop2, 64 - lsb);
        code.add_d(Xresult, Xop2, Xop1);
//        code.EXTR(Xresult, Xop2, Xop1, lsb);  // NB: flipped
    }

    template<>
    void EmitIR<IR::Opcode::ReplicateBit32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        ASSERT(args[1].IsImmediate());

        auto Wresult = ctx.reg_alloc.WriteW(inst);
        auto Wvalue = ctx.reg_alloc.ReadW(args[0]);
        const u8 bit = args[1].GetImmediateU8();
        RegAlloc::Realize(Wresult, Wvalue);

        code.slli_w(Wresult, Wvalue, 31 - bit);
        code.srai_w(Wresult, Wresult, 31);
    }

    template<>
    void EmitIR<IR::Opcode::ReplicateBit64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        ASSERT(args[1].IsImmediate());

        auto Xresult = ctx.reg_alloc.WriteX(inst);
        auto Xvalue = ctx.reg_alloc.ReadX(args[0]);
        const u8 bit = args[1].GetImmediateU8();
        RegAlloc::Realize(Xresult, Xvalue);

        code.slli_d(Xresult, Xvalue, 63 - bit);
        code.srai_d(Xresult, Xresult, 63);
    }

    static void EmitMaxMin32(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst,
                             Xbyak_loongarch64::Cond cond) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        auto Wresult = ctx.reg_alloc.WriteW(inst);
        auto Wop1 = ctx.reg_alloc.ReadW(args[0]);
        auto Wop2 = ctx.reg_alloc.ReadW(args[1]);
        RegAlloc::Realize(Wresult, Wop1, Wop2);
        ctx.reg_alloc.SpillFlags();

//        code.add_d(Wresult, Wop1, code.zero);
        if (cond == GT) {
            code.slt(Wresult, Wop2, Wop1);
            code.masknez(Wscratch0, Wop2, Wresult);
            code.maskeqz(Wscratch1, Wop1, Wresult);
            code.add_d(Wresult, Wscratch0, Wscratch1);
        } else if (cond == HI) {
            code.sltu(Wresult, Wop2, Wop1);
            code.masknez(Wscratch0, Wop2, Wresult);
            code.maskeqz(Wscratch1, Wop1, Wresult);
            code.add_d(Wresult, Wscratch0, Wscratch1);
        } else if (cond == LT) {
            code.slt(Wresult, Wop1, Wop2);
            code.masknez(Wscratch0, Wop2, Wresult);
            code.maskeqz(Wscratch1, Wop1, Wresult);
            code.add_d(Wresult, Wscratch0, Wscratch1);
        } else if (cond == LO) {
            code.sltu(Wresult, Wop1, Wop2);
            code.masknez(Wscratch0, Wop2, Wresult);
            code.maskeqz(Wscratch1, Wop1, Wresult);
            code.add_d(Wresult, Wscratch0, Wscratch1);
        }

    }

    template<>
    void EmitIR<IR::Opcode::MaxSigned32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitMaxMin32(code, ctx, inst, GT);
    }

    template<>
    void EmitIR<IR::Opcode::MaxSigned64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitMaxMin32(code, ctx, inst, GT);
    }

    template<>
    void EmitIR<IR::Opcode::MaxUnsigned32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitMaxMin32(code, ctx, inst, HI);
    }

    template<>
    void EmitIR<IR::Opcode::MaxUnsigned64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitMaxMin32(code, ctx, inst, HI);
    }

    template<>
    void EmitIR<IR::Opcode::MinSigned32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitMaxMin32(code, ctx, inst, LT);
    }

    template<>
    void EmitIR<IR::Opcode::MinSigned64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitMaxMin32(code, ctx, inst, LT);
    }

    template<>
    void EmitIR<IR::Opcode::MinUnsigned32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitMaxMin32(code, ctx, inst, LO);
    }

    template<>
    void EmitIR<IR::Opcode::MinUnsigned64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitMaxMin32(code, ctx, inst, LO);
    }

}  // namespace Dynarmic::Backend::LoongArch64
