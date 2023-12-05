/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <mcl/bit/bit_field.hpp>

#include "nzcv_util.h"
#include "dynarmic/backend/loongarch64/a32_jitstate.h"
#include "dynarmic/backend/loongarch64/abi.h"
#include "dynarmic/backend/loongarch64/emit_arm64.h"
#include "dynarmic/backend/loongarch64/emit_context.h"
#include "dynarmic/backend/loongarch64/fpsr_manager.h"
#include "dynarmic/backend/loongarch64/reg_alloc.h"
#include "dynarmic/frontend/A32/a32_types.h"
#include "dynarmic/interface/halt_reason.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"
#include "xbyak_loongarch64.h"
#include "xbyak_loongarch64_util.h"
#include "dynarmic/frontend/A32/a32_location_descriptor.h"

namespace Dynarmic::Backend::LoongArch64 {
    using namespace Xbyak_loongarch64::util;

    Xbyak_loongarch64::Label EmitA32Cond(Xbyak_loongarch64::CodeGenerator &code, EmitContext &, IR::Cond cond) {
        Xbyak_loongarch64::Label pass;
        // TODO: Flags in host flags
        code.ld_d(Xscratch0, Xstate, offsetof(A32JitState, cpsr_nzcv));
        switch (cond) {
            case IR::Cond::EQ:
                code.andi(Xscratch0, Xscratch0, NZCV::arm_n_flag_mask);
                code.bnez(Xscratch0, pass);
                break;
            case IR::Cond::NE:
                code.andi(Xscratch0, Xscratch0, NZCV::arm_n_flag_mask);
                code.beqz(Xscratch0, pass);
                break;
            case IR::Cond::CS:
                code.andi(Xscratch0, Xscratch0, NZCV::arm_c_flag_mask);
                code.bnez(Xscratch0, pass);
                break;
            case IR::Cond::CC:
                code.andi(Xscratch0, Xscratch0, NZCV::arm_c_flag_mask);
                code.beqz(Xscratch0, pass);
                break;
            case IR::Cond::MI:
                code.andi(Xscratch0, Xscratch0, NZCV::arm_n_flag_mask);
                code.bnez(Xscratch0, pass);
                break;
            case IR::Cond::PL:
                code.andi(Xscratch0, Xscratch0, NZCV::arm_n_flag_mask);
                code.beqz(Xscratch0, pass);
                break;
            case IR::Cond::VS:
                code.andi(Xscratch0, Xscratch0, NZCV::arm_v_flag_mask);
                code.bnez(Xscratch0, pass);
                break;
            case IR::Cond::VC:
                code.andi(Xscratch0, Xscratch0, NZCV::arm_v_flag_mask);
                code.beqz(Xscratch0, pass);
                break;
            case IR::Cond::HI:
                code.andi(Xscratch0, Xscratch0, NZCV::arm_c_flag_mask | NZCV::arm_z_flag_mask);
                code.addi_d(Xscratch1, code.zero, NZCV::arm_hi_flag_mask);
                code.beq(Xscratch0, Xscratch1, pass);
                break;
            case IR::Cond::LS:
                code.andi(Xscratch0, Xscratch0, NZCV::arm_c_flag_mask | NZCV::arm_z_flag_mask);
                code.addi_d(Xscratch1, code.zero, NZCV::arm_ls_flag_mask);
                code.beq(Xscratch0, Xscratch1, pass);
                break;
            case IR::Cond::GT:
            case IR::Cond::GE:
                code.andi(Xscratch0, Xscratch0, NZCV::arm_gt_flag_mask1);
                if (cond == IR::Cond::GE) {
                    code.andi(Xscratch0, Xscratch0, NZCV::arm_ge_flag_mask);
                }
                code.beqz(Xscratch0, pass);
                code.addi_d(Xscratch1, code.zero, NZCV::arm_ge_flag_mask);
                code.beq(Xscratch0, Xscratch1, pass);
                break;
            case IR::Cond::LE:
                code.andi(Xscratch1, Xscratch0, NZCV::arm_z_flag_mask);
                code.bnez(Xscratch1, pass);
            case IR::Cond::LT:
                code.andi(Xscratch0, Xscratch0, NZCV::arm_n_flag_mask | NZCV::arm_v_flag_mask);
                code.addi_d(Xscratch1, code.zero, NZCV::arm_v_flag_mask);
                code.beq(Xscratch0, Xscratch1, pass);
                code.addi_d(Xscratch1, code.zero, NZCV::arm_n_flag_mask);
                code.beq(Xscratch0, Xscratch1, pass);
                break;
            default:
                ASSERT_MSG(false, "Unknown cond {}", static_cast<size_t>(cond));
                break;
        }
        return pass;
    }

    void EmitA32Terminal(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Term::Terminal terminal,
                         IR::LocationDescriptor initial_location, bool is_single_step);

    void EmitA32Terminal(Xbyak_loongarch64::CodeGenerator &, EmitContext &, IR::Term::Interpret, IR::LocationDescriptor,
                         bool) {
        ASSERT_FALSE("Interpret should never be emitted.");
    }

    void EmitA32Terminal(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Term::ReturnToDispatch,
                         IR::LocationDescriptor, bool) {
        EmitRelocation(code, ctx, LinkTarget::ReturnToDispatcher);
    }

    static void EmitSetUpperLocationDescriptor(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                               IR::LocationDescriptor new_location,
                                               IR::LocationDescriptor old_location) {
        auto get_upper = [](const IR::LocationDescriptor &desc) -> u32 {
            return static_cast<u32>(A32::LocationDescriptor{desc}.SetSingleStepping(false).UniqueHash() >> 32);
        };

        const u32 old_upper = get_upper(old_location);
        const u32 new_upper = [&] {
            const u32 mask = ~u32(ctx.conf.always_little_endian ? 0x2 : 0);
            return get_upper(new_location) & mask;
        }();

        if (old_upper != new_upper) {
            code.add_imm(Wscratch0, code.zero, new_upper, Xscratch0);
//            code.add_d(Wscratch0, new_upper, code.zero);
            code.st_w(Wscratch0, Xstate, offsetof(A32JitState, upper_location_descriptor));
        }
    }

    void EmitA32Terminal(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Term::LinkBlock terminal,
                         IR::LocationDescriptor initial_location, bool is_single_step) {
        EmitSetUpperLocationDescriptor(code, ctx, terminal.next, initial_location);

        Xbyak_loongarch64::Label fail;

        if (ctx.conf.HasOptimization(OptimizationFlag::BlockLinking) && !is_single_step) {
            if (ctx.conf.enable_cycle_counting) {
                code.beqz(Xticks, fail);
                code.blt(Xticks, code.zero, fail);
                EmitBlockLinkRelocation(code, ctx, terminal.next, BlockRelocationType::Branch);
            } else {
                code.ll_acq_w(Wscratch0, Xhalt);
                code.bnez(Wscratch0, fail);
                EmitBlockLinkRelocation(code, ctx, terminal.next, BlockRelocationType::Branch);
            }
        }

        code.L(fail);
        code.add_imm(Wscratch0, code.zero, A32::LocationDescriptor{terminal.next}.PC(), Wscratch1);

//        code.add_d(Wscratch0, A32::LocationDescriptor{terminal.next}.PC(), code.zero);
        code.st_d(Wscratch0, Xstate, offsetof(A32JitState, regs) + sizeof(u32) * 15);
        EmitRelocation(code, ctx, LinkTarget::ReturnToDispatcher);
    }

    void EmitA32Terminal(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Term::LinkBlockFast terminal,
                         IR::LocationDescriptor initial_location, bool is_single_step) {
        EmitSetUpperLocationDescriptor(code, ctx, terminal.next, initial_location);

        if (ctx.conf.HasOptimization(OptimizationFlag::BlockLinking) && !is_single_step) {
            EmitBlockLinkRelocation(code, ctx, terminal.next, BlockRelocationType::Branch);
        }

        code.add_imm(Wscratch0, code.zero, A32::LocationDescriptor{terminal.next}.PC(), Xscratch1);
//        code.add_d(Wscratch0, A32::LocationDescriptor{terminal.next}.PC(), code.zero);
        code.st_d(Wscratch0, Xstate, offsetof(A32JitState, regs) + sizeof(u32) * 15);
        EmitRelocation(code, ctx, LinkTarget::ReturnToDispatcher);
    }

    void EmitA32Terminal(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Term::PopRSBHint,
                         IR::LocationDescriptor, bool is_single_step) {
        if (ctx.conf.HasOptimization(OptimizationFlag::ReturnStackBuffer) && !is_single_step) {
            Xbyak_loongarch64::Label fail;

            code.ld_d(Wscratch2, code.sp, offsetof(StackLayout, rsb_ptr));
            code.andi(Wscratch2, Wscratch2, RSBIndexMask);
            code.add_d(code.a2, code.sp, Xscratch2);
            code.sub_imm(Wscratch2, Wscratch2, sizeof(RSBEntry), code.t0);
            code.st_d(Wscratch2, code.sp, offsetof(StackLayout, rsb_ptr));

            code.ld_d(Xscratch0, code.a2, offsetof(StackLayout, rsb));
            code.ld_d(Xscratch1, code.a2, offsetof(StackLayout, rsb) + 8);
//            code.LDP(Xscratch0, Xscratch1, code.a2, offsetof(StackLayout, rsb));

            static_assert(
                    offsetof(A32JitState, regs) + 16 * sizeof(u32) == offsetof(A32JitState, upper_location_descriptor));
//            code.LDUR(code.a0, Xstate, offsetof(A32JitState, regs) + 15 * sizeof(u32));
            code.ld_hu(code.a0, Xstate, offsetof(A32JitState, regs) + 15 * sizeof(u32));

//            code.CMP(code.a0, Xscratch0);
//            code.B(NE, fail);
            code.bne(code.a0, Xscratch0, fail);
            code.jirl(code.zero, Xscratch1, 0);

            code.L(fail);
        }

        EmitRelocation(code, ctx, LinkTarget::ReturnToDispatcher);
    }

    void EmitA32Terminal(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Term::FastDispatchHint,
                         IR::LocationDescriptor, bool) {
        EmitRelocation(code, ctx, LinkTarget::ReturnToDispatcher);

        // TODO: Implement FastDispatchHint optimization
    }

    void EmitA32Terminal(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Term::If terminal,
                         IR::LocationDescriptor initial_location, bool is_single_step) {
        Xbyak_loongarch64::Label pass = EmitA32Cond(code, ctx, terminal.if_);
        EmitA32Terminal(code, ctx, terminal.else_, initial_location, is_single_step);
        code.L(pass);
        EmitA32Terminal(code, ctx, terminal.then_, initial_location, is_single_step);
    }

    void EmitA32Terminal(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Term::CheckBit terminal,
                         IR::LocationDescriptor initial_location, bool is_single_step) {
        Xbyak_loongarch64::Label fail;
        code.ld_b(Wscratch0, code.sp, offsetof(StackLayout, check_bit));
        code.beqz(Wscratch0, fail);
        EmitA32Terminal(code, ctx, terminal.then_, initial_location, is_single_step);
        code.L(fail);
        EmitA32Terminal(code, ctx, terminal.else_, initial_location, is_single_step);
    }

    void EmitA32Terminal(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Term::CheckHalt terminal,
                         IR::LocationDescriptor initial_location, bool is_single_step) {
        Xbyak_loongarch64::Label fail;
        code.ll_acq_w(Wscratch0, Xhalt);
        code.bnez(Wscratch0, fail);
        EmitA32Terminal(code, ctx, terminal.else_, initial_location, is_single_step);
        code.L(fail);
        EmitRelocation(code, ctx, LinkTarget::ReturnToDispatcher);
    }

    void EmitA32Terminal(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Term::Terminal terminal,
                         IR::LocationDescriptor initial_location, bool is_single_step) {
        boost::apply_visitor([&](const auto &t) { EmitA32Terminal(code, ctx, t, initial_location, is_single_step); },
                             terminal);
    }

    void EmitA32Terminal(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx) {
        const A32::LocationDescriptor location{ctx.block.Location()};
        EmitA32Terminal(code, ctx, ctx.block.GetTerminal(), location.SetSingleStepping(false),
                        location.SingleStepping());
    }

    void EmitA32ConditionFailedTerminal(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx) {
        const A32::LocationDescriptor location{ctx.block.Location()};
        EmitA32Terminal(code, ctx, IR::Term::LinkBlock{ctx.block.ConditionFailedLocation()},
                        location.SetSingleStepping(false), location.SingleStepping());
    }

    void EmitA32CheckMemoryAbort(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst,
                                 Xbyak_loongarch64::Label &end) {
        if (!ctx.conf.check_halt_on_memory_access) {
            return;
        }

        const A32::LocationDescriptor current_location{IR::LocationDescriptor{inst->GetArg(0).GetU64()}};

        code.ll_acq_w(Xscratch0, Xhalt);
//        code.TST(Xscratch0, static_cast<u32>(HaltReason::MemoryAbort));
//        code.B(EQ, end);
        code.add_imm(Xscratch1, code.zero, static_cast<u32>(HaltReason::MemoryAbort), Xscratch2);
        code.beq(Xscratch0, Xscratch1, end);
        EmitSetUpperLocationDescriptor(code, ctx, current_location, ctx.block.Location());
        code.add_imm(Wscratch0, code.zero, current_location.PC(), Wscratch1);
//        code.add_d(Wscratch0, current_location.PC(), code.zero);
        code.st_w(Wscratch0, Xstate, offsetof(A32JitState, regs) + sizeof(u32) * 15);
        EmitRelocation(code, ctx, LinkTarget::ReturnFromRunCode);
    }

    template<>
    void EmitIR<IR::Opcode::A32SetCheckBit>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        if (args[0].IsImmediate()) {
            if (args[0].GetImmediateU1()) {
                code.addi_d(Xscratch0, code.zero, 1);
                code.st_b(Xscratch0, code.sp, offsetof(StackLayout, check_bit));
            } else {
                code.st_b(code.zero, code.sp, offsetof(StackLayout, check_bit));
            }
        } else {
            auto Wbit = ctx.reg_alloc.ReadW(args[0]);
            RegAlloc::Realize(Wbit);
            code.st_b(Wbit, code.sp, offsetof(StackLayout, check_bit));
        }
    }

    template<>
    void EmitIR<IR::Opcode::A32GetRegister>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        const A32::Reg reg = inst->GetArg(0).GetA32RegRef();

        auto Wresult = ctx.reg_alloc.WriteW(inst);
        RegAlloc::Realize(Wresult);

        // TODO: Detect if Gpr vs Fpr is more appropriate

        code.ld_d(Wresult, Xstate, offsetof(A32JitState, regs) + sizeof(u32) * static_cast<size_t>(reg));
    }

    template<>
    void EmitIR<IR::Opcode::A32GetExtendedRegister32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        const A32::ExtReg reg = inst->GetArg(0).GetA32ExtRegRef();
        ASSERT(A32::IsSingleExtReg(reg));
        const size_t index = static_cast<size_t>(reg) - static_cast<size_t>(A32::ExtReg::S0);

        auto Sresult = ctx.reg_alloc.WriteS(inst);
        RegAlloc::Realize(Sresult);

        // TODO: Detect if Gpr vs Fpr is more appropriate
        // FIXME: this assume fp is in vr
        code.ld_w(Xscratch0, Xstate, offsetof(A32JitState, ext_regs) + sizeof(u32) * index);
        code.vinsgr2vr_w(Sresult, Xscratch0, 0);
//        code.ld_d(Sresult, Xstate, offsetof(A32JitState, ext_regs) + sizeof(u32) * index);
    }

    template<>
    void EmitIR<IR::Opcode::A32GetVector>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        const A32::ExtReg reg = inst->GetArg(0).GetA32ExtRegRef();
        ASSERT(A32::IsDoubleExtReg(reg) || A32::IsQuadExtReg(reg));

        if (A32::IsDoubleExtReg(reg)) {
            const size_t index = static_cast<size_t>(reg) - static_cast<size_t>(A32::ExtReg::D0);
            auto Dresult = ctx.reg_alloc.WriteD(inst);
            RegAlloc::Realize(Dresult);
            code.ld_d(Xscratch0, Xstate, offsetof(A32JitState, ext_regs) + sizeof(u64) * index);
            code.vinsgr2vr_d(Dresult, Xscratch0, 0);
//            code.ld_d(Dresult, Xstate, offsetof(A32JitState, ext_regs) + sizeof(u64) * index);
        } else {
            const size_t index = static_cast<size_t>(reg) - static_cast<size_t>(A32::ExtReg::Q0);
            auto Qresult = ctx.reg_alloc.WriteQ(inst);
            RegAlloc::Realize(Qresult);
            code.vld(Qresult, Xstate, offsetof(A32JitState, ext_regs) + 2 * sizeof(u64) * index);
//            code.ld_d(Qresult, Xstate, offsetof(A32JitState, ext_regs) + 2 * sizeof(u64) * index);
        }
    }

    template<>
    void EmitIR<IR::Opcode::A32GetExtendedRegister64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        const A32::ExtReg reg = inst->GetArg(0).GetA32ExtRegRef();
        ASSERT(A32::IsDoubleExtReg(reg));
        const size_t index = static_cast<size_t>(reg) - static_cast<size_t>(A32::ExtReg::D0);

        auto Dresult = ctx.reg_alloc.WriteD(inst);
        RegAlloc::Realize(Dresult);

        // TODO: Detect if Gpr vs Fpr is more appropriate
        code.ld_d(Xscratch0, Xstate, offsetof(A32JitState, ext_regs) + 2 * sizeof(u32) * index);
        code.vinsgr2vr_d(Dresult, Xscratch0, 0);

//        code.ld_d(Dresult, Xstate, offsetof(A32JitState, ext_regs) + 2 * sizeof(u32) * index);
    }

    template<>
    void EmitIR<IR::Opcode::A32SetRegister>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        const A32::Reg reg = inst->GetArg(0).GetA32RegRef();

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        auto Wvalue = ctx.reg_alloc.ReadW(args[1]);
        RegAlloc::Realize(Wvalue);

        // FIXME: if this should detect args[1] is imm
        // TODO: Detect if Gpr vs Fpr is more appropriate
//        if (args[1].IsImmediate()) {
//            code.
//        }

        code.st_d(Wvalue, Xstate, offsetof(A32JitState, regs) + sizeof(u32) * static_cast<size_t>(reg));
    }

    template<>
    void EmitIR<IR::Opcode::A32SetExtendedRegister32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        const A32::ExtReg reg = inst->GetArg(0).GetA32ExtRegRef();
        ASSERT(A32::IsSingleExtReg(reg));
        const size_t index = static_cast<size_t>(reg) - static_cast<size_t>(A32::ExtReg::S0);

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Svalue = ctx.reg_alloc.ReadS(args[1]);
        RegAlloc::Realize(Svalue);

        // TODO: Detect if Gpr vs Fpr is more appropriate
        code.ld_d(Xscratch0, Xstate, offsetof(A32JitState, ext_regs) + sizeof(u32) * index);
        code.vinsgr2vr_w(Svalue, Xscratch0, 0);
//        code.st_d(Svalue, Xstate, offsetof(A32JitState, ext_regs) + sizeof(u32) * index);
    }

    template<>
    void EmitIR<IR::Opcode::A32SetExtendedRegister64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        const A32::ExtReg reg = inst->GetArg(0).GetA32ExtRegRef();
        ASSERT(A32::IsDoubleExtReg(reg));
        const size_t index = static_cast<size_t>(reg) - static_cast<size_t>(A32::ExtReg::D0);

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Dvalue = ctx.reg_alloc.ReadD(args[1]);
        RegAlloc::Realize(Dvalue);

        // TODO: Detect if Gpr vs Fpr is more appropriate
        code.ld_d(Xscratch0, Xstate, offsetof(A32JitState, ext_regs) + 2 * sizeof(u32) * index);
        code.vinsgr2vr_d(Dvalue, Xscratch0, 0);
//        code.st_d(Dvalue, Xstate, offsetof(A32JitState, ext_regs) + 2 * sizeof(u32) * index);
    }

    template<>
    void EmitIR<IR::Opcode::A32SetVector>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        const A32::ExtReg reg = inst->GetArg(0).GetA32ExtRegRef();
        ASSERT(A32::IsDoubleExtReg(reg) || A32::IsQuadExtReg(reg));
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        if (A32::IsDoubleExtReg(reg)) {
            const size_t index = static_cast<size_t>(reg) - static_cast<size_t>(A32::ExtReg::D0);
            auto Dvalue = ctx.reg_alloc.ReadD(args[1]);
            RegAlloc::Realize(Dvalue);
            code.ld_d(Xscratch0, Xstate, offsetof(A32JitState, ext_regs) + sizeof(u64) * index);
            code.vinsgr2vr_d(Dvalue, Xscratch0, 0);
//            code.st_d(Dvalue, Xstate, offsetof(A32JitState, ext_regs) + sizeof(u64) * index);
        } else {
            const size_t index = static_cast<size_t>(reg) - static_cast<size_t>(A32::ExtReg::Q0);
            auto Qvalue = ctx.reg_alloc.ReadQ(args[1]);
            RegAlloc::Realize(Qvalue);
            code.vld(Qvalue, Xstate, offsetof(A32JitState, ext_regs) + 2 * sizeof(u64) * index);
//            code.st_d(Qvalue, Xstate, offsetof(A32JitState, ext_regs) + 2 * sizeof(u64) * index);
        }
    }

    template<>
    void EmitIR<IR::Opcode::A32GetCpsr>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto Wcpsr = ctx.reg_alloc.WriteW(inst);
        RegAlloc::Realize(Wcpsr);

        static_assert(offsetof(A32JitState, cpsr_nzcv) + sizeof(u32) == offsetof(A32JitState, cpsr_q));

        code.ld_w(Wscratch0, Xstate, offsetof(A32JitState, cpsr_nzcv));
        code.ld_w(Wscratch1, Xstate, offsetof(A32JitState, cpsr_nzcv) + sizeof(u32));
//        code.LDP(Wscratch0, Wscratch1, Xstate, offsetof(A32JitState, cpsr_nzcv));
        code.ld_d(Wcpsr, Xstate, offsetof(A32JitState, cpsr_jaifm));
        code.or_(Wcpsr, Wcpsr, Wscratch0);
        code.or_(Wcpsr, Wcpsr, Wscratch1);

        code.ld_d(Wscratch0, Xstate, offsetof(A32JitState, cpsr_ge));
        code.add_imm(Wscratch1, code.zero, 0x80808080, Wscratch2);
        code.and_(Wscratch0, Wscratch0, Wscratch1);
        code.add_imm(Wscratch1, code.zero, 0x00204081, Wscratch2);
//        code.add_d(Wscratch1, 0x00204081, code.zero);
        code.mul_w(Wscratch0, Wscratch0, Wscratch1);
        code.add_imm(Wscratch1, code.zero, 0xf0000000, Wscratch2);
//        code.andi(Wscratch0, Wscratch0,    0xf0000000);
        code.and_(Wscratch0, Wscratch0, Wscratch1);
        code.srli_w(Wscratch0, Wscratch0, 12);
        code.or_(Wcpsr, Wcpsr, Wscratch0);
//        code.ORR(Wcpsr, Wcpsr, Wscratch0, LSR, 12);

        code.ld_d(Wscratch0, Xstate, offsetof(A32JitState, upper_location_descriptor));
        code.andi(Wscratch0, Wscratch0, 0b11);
        // 9 8 7 6 5
        //       E T
        code.slli_w(Wscratch1, Wscratch0, 3);
        code.or_(Wscratch0, Wscratch0, Wscratch1);
//        code.ORR(Wscratch0, Wscratch0, Wscratch0, LSL, 3);
        code.add_imm(Wscratch1, code.zero, 0x11111111, Wscratch2);
        code.and_(Wscratch0, Wscratch0, Wscratch1);
//        code.andi(Wscratch0, Wscratch0, 0x11111111);
        code.slli_w(Wscratch0, Wscratch0, 3);
        code.or_(Wcpsr, Wcpsr, Wscratch1);
//        code.ORR(Wcpsr, Wcpsr, Wscratch0, LSL, 5);
    }

    template<>
    void EmitIR<IR::Opcode::A32SetCpsr>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Wcpsr = ctx.reg_alloc.ReadW(args[0]);
        RegAlloc::Realize(Wcpsr);

        // NZCV, Q flags
        code.srli_w(Wscratch0, Wcpsr, NZCV::arm_nzcv_shift);
        code.and_(Wscratch0, Wcpsr, Wscratch1);
        code.bstrpick_w(Wscratch1, Wcpsr, 27, 27);
//        code.andi(Wscratch1, Wcpsr, 1 << 27);

        static_assert(offsetof(A32JitState, cpsr_nzcv) + sizeof(u32) == offsetof(A32JitState, cpsr_q));
        code.st_w(Wscratch0, Xstate, offsetof(A32JitState, cpsr_nzcv));
        code.st_w(Wscratch1, Xstate, offsetof(A32JitState, cpsr_nzcv) + sizeof(u32));

//        code.STP(Wscratch0, Wscratch1, Xstate, offsetof(A32JitState, cpsr_nzcv));

        // GE flags
        // this does the following:
        // cpsr_ge |= mcl::bit::get_bit<19>(cpsr) ? 0xFF000000 : 0;
        // cpsr_ge |= mcl::bit::get_bit<18>(cpsr) ? 0x00FF0000 : 0;
        // cpsr_ge |= mcl::bit::get_bit<17>(cpsr) ? 0x0000FF00 : 0;
        // cpsr_ge |= mcl::bit::get_bit<16>(cpsr) ? 0x000000FF : 0;
        code.xor_(Wscratch0, Wscratch0, Wscratch0);
        code.bstrins_w(Wscratch0, Wcpsr, 19, 16);
//        code.UBFX(Wscratch0, Wcpsr, 16, 4);
        code.add_imm(Wscratch1, code.zero, 0x00204081, Wscratch2);
//        code.add_d(Wscratch1, 0x00204081, code.zero);
        code.mul_w(Wscratch0, Wscratch0, Wscratch1);
        code.add_imm(Wscratch1, code.zero, 0x01010101, Wscratch2);
        code.and_(Wscratch0, Wscratch0, Wscratch1);
        code.slli_w(Wscratch1, Wscratch0, 8);
        code.sub_d(Wscratch0, Wscratch1, Wscratch0);

        // Other flags
        code.add_imm(Wscratch1, code.zero, 0x010001DF, Wscratch2);
        code.and_(Wscratch1, Wcpsr, Wscratch1);

        static_assert(offsetof(A32JitState, cpsr_jaifm) + sizeof(u32) == offsetof(A32JitState, cpsr_ge));
        code.st_w(Wscratch1, Xstate, offsetof(A32JitState, cpsr_jaifm));
        code.st_w(Wscratch0, Xstate, offsetof(A32JitState, cpsr_jaifm) + sizeof(u32));

//        code.STP(Wscratch1, Wscratch0, Xstate, offsetof(A32JitState, cpsr_jaifm));

        // IT state
        code.add_imm(Wscratch1, code.zero, 0xFC00, Wscratch2);
        code.and_(Wscratch0, Wcpsr, Wscratch1);
        code.srli_w(Wscratch1, Wcpsr, 17);
        code.andi(Wscratch1, Wscratch1, 0x300);
        code.or_(Wscratch0, Wscratch0, Wscratch1);

        // E flag, T flag
        code.srli_w(Wscratch1, Wcpsr, 8);
        code.andi(Wscratch1, Wscratch1, 0x2);
        code.or_(Wscratch0, Wscratch0, Wscratch1);

        code.srli_w(Wscratch2, Wcpsr, 5);
        code.andi(Wscratch2, Wscratch2, 0x1);
        code.or_(Wscratch0, Wscratch0, Wscratch2);
//        code.BFXIL(Wscratch0, Wcpsr, 5, 1);
        code.add_imm(Wscratch2, code.zero, 0xFFFF0000, Wscratch1);

        code.ld_d(Wscratch1, Xstate, offsetof(A32JitState, upper_location_descriptor));
        code.and_(Wscratch1, Wscratch1, Wscratch2);
        code.or_(Wscratch0, Wscratch0, Wscratch1);
        code.st_d(Wscratch0, Xstate, offsetof(A32JitState, upper_location_descriptor));
    }

    template<>
    void EmitIR<IR::Opcode::A32SetCpsrNZCV>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Wnzcv = ctx.reg_alloc.ReadW(args[0]);
        RegAlloc::Realize(Wnzcv);

        code.st_w(Wnzcv, Xstate, offsetof(A32JitState, cpsr_nzcv));
    }

    template<>
    void
    EmitIR<IR::Opcode::A32SetCpsrNZCVRaw>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Wnzcv = ctx.reg_alloc.ReadW(args[0]);
        RegAlloc::Realize(Wnzcv);
        // u32 ToLoongArch64(u32 nzcv)
        // TODO if Wnzcv is imm
//        code.addi_w(Wscratch0, code.zero, 28);
        code.srli_w(Wnzcv, Wnzcv, NZCV::arm_nzcv_shift);

        code.st_d(Wnzcv, Xstate, offsetof(A32JitState, cpsr_nzcv));
    }

    template<>
    void EmitIR<IR::Opcode::A32SetCpsrNZCVQ>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Wnzcv = ctx.reg_alloc.ReadW(args[0]);
        RegAlloc::Realize(Wnzcv);

        static_assert(offsetof(A32JitState, cpsr_nzcv) + sizeof(u32) == offsetof(A32JitState, cpsr_q));
        code.srli_w(Wscratch0, Wnzcv, NZCV::arm_nzcv_shift);
        code.bstrpick_w(Wscratch1, Wnzcv, 27, 27);
        code.st_w(Wscratch0, Xstate, offsetof(A32JitState, cpsr_nzcv));
        code.st_w(Wscratch1, Xstate, offsetof(A32JitState, cpsr_q));

//        code.andi(Wscratch0, Wnzcv, 0xf000'0000);
//        code.andi(Wscratch1, Wnzcv, 0x0800'0000);
//        code.STP(Wscratch0, Wscratch1, Xstate, offsetof(A32JitState, cpsr_nzcv));
    }

    template<>
    void EmitIR<IR::Opcode::A32SetCpsrNZ>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        auto Wnz = ctx.reg_alloc.ReadW(args[0]);
        RegAlloc::Realize(Wnz);

        // TODO: Track latent value

        code.ld_d(Wscratch0, Xstate, offsetof(A32JitState, cpsr_nzcv));
//        code.add_imm(Wscratch1, code.zero, 0x3, Wscratch2);
        code.andi(Wscratch0, Wscratch0, 0x3);
        code.srli_w(Wnz, Wnz, NZCV::arm_nzcv_shift);
        code.or_(Wscratch0, Wscratch0, Wnz);
        code.st_d(Wscratch0, Xstate, offsetof(A32JitState, cpsr_nzcv));
    }

    template<>
    void EmitIR<IR::Opcode::A32SetCpsrNZC>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        // TODO: Track latent value

        if (args[0].IsImmediate()) {
            if (args[1].IsImmediate()) {
                const u32 carry = args[1].GetImmediateU1() ? 0x2 : 0;

                code.ld_d(Wscratch0, Xstate, offsetof(A32JitState, cpsr_nzcv));
//                code.add_imm(Wscratch1, code.zero, 0x10000000, Wscratch2);

                code.andi(Wscratch0, Wscratch0, 0x1);
                if (carry) {
//                    code.add_imm(Wscratch1, code.zero, carry, Wscratch2);
                    code.ori(Wscratch0, Wscratch0, carry);
                }
                code.st_d(Wscratch0, Xstate, offsetof(A32JitState, cpsr_nzcv));
            } else {
                auto Wc = ctx.reg_alloc.ReadW(args[1]);
                RegAlloc::Realize(Wc);

                code.ld_d(Wscratch0, Xstate, offsetof(A32JitState, cpsr_nzcv));
                code.andi(Wscratch0, Wscratch0, 0x1);
                code.slli_w(Wc, Wc, NZCV::arm_nzcv_shift);
                code.andi(Wc, Wc, 0x2);
                code.or_(Wscratch0, Wscratch0, Wc);
                code.st_d(Wscratch0, Xstate, offsetof(A32JitState, cpsr_nzcv));
            }
        } else {
            if (args[1].IsImmediate()) {
                const u32 carry = args[1].GetImmediateU1() ? 0x2 : 0;
                auto Wnz = ctx.reg_alloc.ReadW(args[0]);
                RegAlloc::Realize(Wnz);

                code.ld_d(Wscratch0, Xstate, offsetof(A32JitState, cpsr_nzcv));
                code.andi(Wscratch0, Wscratch0, 0x1);
                code.srli_w(Wnz, Wnz, NZCV::arm_nzcv_shift);
                code.or_(Wscratch0, Wscratch0, Wnz);
                if (carry) {
                    code.ori(Wscratch0, Wscratch0, carry);
                }
                code.st_d(Wscratch0, Xstate, offsetof(A32JitState, cpsr_nzcv));
            } else {
                auto Wnz = ctx.reg_alloc.ReadW(args[0]);
                auto Wc = ctx.reg_alloc.ReadW(args[1]);
                RegAlloc::Realize(Wnz, Wc);

                code.ld_d(Wscratch0, Xstate, offsetof(A32JitState, cpsr_nzcv));
                code.andi(Wscratch0, Wscratch0, 0x1);
                code.srli_w(Wnz, Wnz, NZCV::arm_nzcv_shift);
                code.srli_w(Wc, Wc, NZCV::arm_nzcv_shift);
                code.andi(Wc, Wc, 0x2);
                code.or_(Wscratch0, Wscratch0, Wnz);
                code.or_(Wscratch0, Wscratch0, Wc);
                code.st_d(Wscratch0, Xstate, offsetof(A32JitState, cpsr_nzcv));
            }
        }
    }

    template<>
    void EmitIR<IR::Opcode::A32GetCFlag>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto Wflag = ctx.reg_alloc.WriteW(inst);
        RegAlloc::Realize(Wflag);

        code.ld_w(Wflag, Xstate, offsetof(A32JitState, cpsr_nzcv));
        code.andi(Wflag, Wflag, 1 << 1);
    }

    template<>
    void EmitIR<IR::Opcode::A32OrQFlag>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Wflag = ctx.reg_alloc.ReadW(args[0]);
        RegAlloc::Realize(Wflag);

        code.ld_d(Wscratch0, Xstate, offsetof(A32JitState, cpsr_q));
        code.or_(Wscratch0, Wscratch0, Wflag);
        code.st_d(Wscratch0, Xstate, offsetof(A32JitState, cpsr_q));
    }

    template<>
    void EmitIR<IR::Opcode::A32GetGEFlags>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto Snzcv = ctx.reg_alloc.WriteS(inst);
        RegAlloc::Realize(Snzcv);
        code.fld_s(Snzcv, Xstate, offsetof(A32JitState, cpsr_ge));
//        code.ld_d(Snzcv, Xstate, offsetof(A32JitState, cpsr_ge));
    }

    template<>
    void EmitIR<IR::Opcode::A32SetGEFlags>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        auto Snzcv = ctx.reg_alloc.ReadS(args[0]);
        RegAlloc::Realize(Snzcv);
        code.fst_s(Snzcv, Xstate, offsetof(A32JitState, cpsr_ge));
//        code.st_d(Snzcv, Xstate, offsetof(A32JitState, cpsr_ge));
    }

    template<>
    void EmitIR<IR::Opcode::A32SetGEFlagsCompressed>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Wge = ctx.reg_alloc.ReadW(args[0]);
        RegAlloc::Realize(Wge);

        code.srli_w(Wscratch0, Wge, 16);
        code.add_imm(Wscratch1, code.zero, 0x00204081, Wscratch2);
//        code.add_d(Wscratch1, 0x00204081, code.zero);
        code.mul_w(Wscratch0, Wscratch0, Wscratch1);
        code.add_imm(Wscratch1, code.zero, 0x01010101, Wscratch2);
        code.and_(Wscratch0, Wscratch0, Wscratch1);
        code.slli_w(Wscratch1, Wscratch0, 8);
        code.sub_w(Wscratch0, Wscratch1, Wscratch0);
        code.st_w(Wscratch0, Xstate, offsetof(A32JitState, cpsr_ge));
    }

    template<>
    void EmitIR<IR::Opcode::A32BXWritePC>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        const u32 upper_without_t = (A32::LocationDescriptor{ctx.block.EndLocation()}.SetSingleStepping(false).
                UniqueHash() >> 32) & 0xFFFFFFFE;

        static_assert(
                offsetof(A32JitState, regs) + 16 * sizeof(u32) == offsetof(A32JitState, upper_location_descriptor));

        if (args[0].IsImmediate()) {
            const u32 new_pc = args[0].GetImmediateU32();
            const u32 mask = mcl::bit::get_bit<0>(new_pc) ? 0xFFFFFFFE : 0xFFFFFFFC;
            const u32 new_upper = upper_without_t | (mcl::bit::get_bit<0>(new_pc) ? 1 : 0);
            // TODO check if eq to stp
            code.add_imm(Xscratch0, code.zero, (u64{new_upper} << 32) | (new_pc & mask), Xscratch1);
            code.st_d(Xscratch0, Xstate, offsetof(A32JitState, regs) + 15 * sizeof(u32));
//            code.STUR(Xscratch0, Xstate, offsetof(A32JitState, regs) + 15 * sizeof(u32));
        } else {
            auto Wpc = ctx.reg_alloc.ReadW(args[0]);
            RegAlloc::Realize(Wpc);
            ctx.reg_alloc.SpillFlags();

            code.add_imm(Wscratch2,  code.zero, upper_without_t, Wscratch1);
            code.andi(Wscratch0, Wpc, 1);
//            code.bstrpick_d(Wscratch1, Wscratch0, 0, 0);
            code.or_(Wscratch2, Wscratch2, Wscratch0);
            code.st_w(Wscratch2, Xstate, offsetof(A32JitState, regs) + 15 * sizeof(u32) + sizeof(u32));

            code.addi_w(Wscratch0, Wscratch0, 1);
            code.andi(Wscratch0, Wscratch0, 0x1);
            code.slli_w(Wscratch0, Wscratch0, 1);

            code.bstrins_d(Wpc, code.zero, 0, 0);
            code.orn(Wpc, Wpc, Wscratch0);
            code.st_w(Wpc, Xstate, offsetof(A32JitState, regs) + 15 * sizeof(u32));
        }
    }

    template<>
    void EmitIR<IR::Opcode::A32UpdateUpperLocationDescriptor>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                              IR::Inst *) {
        for (auto &inst: ctx.block) {
            if (inst.GetOpcode() == IR::Opcode::A32BXWritePC) {
                return;
            }
        }
        EmitSetUpperLocationDescriptor(code, ctx, ctx.block.EndLocation(), ctx.block.Location());
    }

    template<>
    void
    EmitIR<IR::Opcode::A32CallSupervisor>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        ctx.reg_alloc.PrepareForCall();

        if (ctx.conf.enable_cycle_counting) {
            code.ld_d(code.a1, code.sp, offsetof(StackLayout, cycles_to_run));
            code.sub_d(code.a1, code.a1, Xticks);
            EmitRelocation(code, ctx, LinkTarget::AddTicks);
        }

        code.add_imm(code.a1, code.zero, args[0].GetImmediateU32(), Wscratch0);
        EmitRelocation(code, ctx, LinkTarget::CallSVC);

        if (ctx.conf.enable_cycle_counting) {
            EmitRelocation(code, ctx, LinkTarget::GetTicksRemaining);
            code.st_d(code.a0, code.sp, offsetof(StackLayout, cycles_to_run));
            code.add_d(Xticks, code.a0, code.zero);
        }
        // todo A32EmitX64::EmitA32CallSupervisor call SwitchMxcsrOnExit?
    }

    template<>
    void EmitIR<IR::Opcode::A32ExceptionRaised>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        ctx.reg_alloc.PrepareForCall();

        if (ctx.conf.enable_cycle_counting) {
            code.ld_d(code.a1, code.sp, offsetof(StackLayout, cycles_to_run));
            code.sub_d(code.a1, code.a1, Xticks);
            EmitRelocation(code, ctx, LinkTarget::AddTicks);
        }

        code.add_imm(code.a1, code.zero, args[0].GetImmediateU32(), Wscratch0);
        code.add_imm(code.a2, code.zero, args[1].GetImmediateU32(), Wscratch0);
        EmitRelocation(code, ctx, LinkTarget::ExceptionRaised);

        if (ctx.conf.enable_cycle_counting) {
            EmitRelocation(code, ctx, LinkTarget::GetTicksRemaining);
            code.st_d(code.a0, code.sp, offsetof(StackLayout, cycles_to_run));
            code.add_d(Xticks, code.a0, code.zero);
        }
    }

    template<>
    void EmitIR<IR::Opcode::A32DataSynchronizationBarrier>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &,
                                                           IR::Inst *) {
        code.dbar(0x0);
    }

    template<>
    void EmitIR<IR::Opcode::A32DataMemoryBarrier>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &, IR::Inst *) {
        code.dbar(0x700);
    }

    template<>
    void EmitIR<IR::Opcode::A32InstructionSynchronizationBarrier>(Xbyak_loongarch64::CodeGenerator &code,
                                                                  EmitContext &ctx, IR::Inst *) {
        if (!ctx.conf.hook_isb) {
            return;
        }

        ctx.reg_alloc.PrepareForCall();
        EmitRelocation(code, ctx, LinkTarget::InstructionSynchronizationBarrierRaised);
    }
    static u32 GetFpscrImpl(A32JitState* jit_state) {
        return jit_state->Fpscr();
    }
    template<>
    void EmitIR<IR::Opcode::A32GetFpscr>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto Wfpscr = ctx.reg_alloc.WriteW(inst);
        RegAlloc::Realize(Wfpscr);
        ctx.fpsr.Spill();

        static_assert(offsetof(A32JitState, fpsr) + sizeof(u32) == offsetof(A32JitState, fpsr_nzcv));

        code.ld_d(Wfpscr, Xstate, offsetof(A32JitState, upper_location_descriptor));
        code.LDP(Wscratch0, Wscratch1, Xstate, offsetof(A32JitState, fpsr));
        code.andi(Wfpscr, Wfpscr, 0xffff'0000);
        code.or_(Wscratch0, Wscratch0, Wscratch1);
        code.or_(Wfpscr, Wfpscr, Wscratch0);
        code.call(GetFpscrImpl);
        // FIXME
    }
    static void SetFpscrImpl(u32 value, A32JitState* jit_state) {
        jit_state->SetFpscr(value);
    }
    template<>
    void EmitIR<IR::Opcode::A32SetFpscr>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Wfpscr = ctx.reg_alloc.ReadW(args[0]);
        RegAlloc::Realize(Wfpscr);
        ctx.fpsr.Overwrite();

        static_assert(offsetof(A32JitState, fpsr) + sizeof(u32) == offsetof(A32JitState, fpsr_nzcv));

        code.ld_d(Wscratch0, Xstate, offsetof(A32JitState, upper_location_descriptor));
        code.add_d(Wscratch1, 0x07f7'0000, code.zero);
        code.andi(Wscratch1, Wfpscr, Wscratch1);
        code.andi(Wscratch0, Wscratch0, 0x0000'ffff);
        code.or_(Wscratch0, Wscratch0, Wscratch1);
        code.st_d(Wscratch0, Xstate, offsetof(A32JitState, upper_location_descriptor));

        code.add_d(Wscratch0, 0x0800'009f, code.zero);
        code.andi(Wscratch0, Wfpscr, Wscratch0);
        code.andi(Wscratch1, Wfpscr, 0xf000'0000);
        code.STP(Wscratch0, Wscratch1, Xstate, offsetof(A32JitState, fpsr));
        code.call(SetFpscrImpl);
        // FIXME
    }

    template<>
    void EmitIR<IR::Opcode::A32GetFpscrNZCV>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto Wnzcv = ctx.reg_alloc.WriteW(inst);
        RegAlloc::Realize(Wnzcv);

        code.ld_d(Wnzcv, Xstate, offsetof(A32JitState, fpsr_nzcv));
    }

    template<>
    void EmitIR<IR::Opcode::A32SetFpscrNZCV>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Wnzcv = ctx.reg_alloc.ReadW(args[0]);
        RegAlloc::Realize(Wnzcv);

        code.st_d(Wnzcv, Xstate, offsetof(A32JitState, fpsr_nzcv));
    }
} // namespace Dynarmic::Backend::LoongArch64
