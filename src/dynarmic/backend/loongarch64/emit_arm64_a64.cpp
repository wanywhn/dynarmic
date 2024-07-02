/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <mcl/bit_cast.hpp>

#include "nzcv_util.h"
#include "dynarmic/backend/loongarch64/a64_jitstate.h"
#include "dynarmic/backend/loongarch64/abi.h"
#include "dynarmic/backend/loongarch64/emit_arm64.h"
#include "dynarmic/backend/loongarch64/emit_context.h"
#include "dynarmic/backend/loongarch64/reg_alloc.h"
#include "dynarmic/interface/halt_reason.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"
#include "xbyak_loongarch64.h"
#include "xbyak_loongarch64_util.h"

namespace Dynarmic::Backend::LoongArch64 {

    using namespace Xbyak_loongarch64::util;

    Xbyak_loongarch64::Label EmitA64Cond(BlockOfCode &code, EmitContext &, IR::Cond cond) {
        Xbyak_loongarch64::Label pass;
        // TODO: Flags in host flags
        code.ld_d(Xscratch0, Xstate, offsetof(A64JitState, cpsr_nzcv));
//        void LoadRequiredFlagsForCondFromRax(IR::Cond cond);
// FIXME

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
                break;
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

    void EmitA64Terminal(BlockOfCode &code, EmitContext &ctx, IR::Term::Terminal terminal,
                         IR::LocationDescriptor initial_location, bool is_single_step);

    void EmitA64Terminal(BlockOfCode &, EmitContext &, IR::Term::Interpret, IR::LocationDescriptor,
                         bool) {
        ASSERT_FALSE("Interpret should never be emitted.");
    }

    void EmitA64Terminal(BlockOfCode &code, EmitContext &ctx, IR::Term::ReturnToDispatch,
                         IR::LocationDescriptor, bool) {
        EmitRelocation(code, ctx, LinkTarget::ReturnToDispatcher);
    }

    void EmitA64Terminal(BlockOfCode &code, EmitContext &ctx, IR::Term::LinkBlock terminal,
                         IR::LocationDescriptor, bool is_single_step) {
        Xbyak_loongarch64::Label fail;

        if (ctx.conf.HasOptimization(OptimizationFlag::BlockLinking) && !is_single_step) {
            if (ctx.conf.enable_cycle_counting) {
                code.beqz(Xticks, fail);
                code.blt(Xticks, code.zero, fail);
//                code.CMP(Xticks, 0);
//                code.B(LE, fail);
                EmitBlockLinkRelocation(code, ctx, terminal.next, BlockRelocationType::Branch);
            } else {
                code.ll_w(Wscratch0, Xhalt, 0);
                code.bnez(Wscratch0, fail);
                EmitBlockLinkRelocation(code, ctx, terminal.next, BlockRelocationType::Branch);
            }
        }

        code.L(fail);
        code.add_imm(Xscratch0, code.zero, A64::LocationDescriptor{terminal.next}.PC(), Xscratch1);
//        code.add_d(Xscratch0, A64::LocationDescriptor{terminal.next}.PC(), code.zero);
        code.st_d(Xscratch0, Xstate, offsetof(A64JitState, pc));
        EmitRelocation(code, ctx, LinkTarget::ReturnToDispatcher);
    }

    void EmitA64Terminal(BlockOfCode &code, EmitContext &ctx, IR::Term::LinkBlockFast terminal,
                         IR::LocationDescriptor, bool is_single_step) {
        if (ctx.conf.HasOptimization(OptimizationFlag::BlockLinking) && !is_single_step) {
            EmitBlockLinkRelocation(code, ctx, terminal.next, BlockRelocationType::Branch);
        }
        code.add_imm(Xscratch0, code.zero, A64::LocationDescriptor{terminal.next}.PC(), Xscratch1);
//        code.add_d(Xscratch0, A64::LocationDescriptor{terminal.next}.PC(), code.zero);
        code.st_d(Xscratch0, Xstate, offsetof(A64JitState, pc));
        EmitRelocation(code, ctx, LinkTarget::ReturnToDispatcher);
    }

    void EmitA64Terminal(BlockOfCode &code, EmitContext &ctx, IR::Term::PopRSBHint,
                         IR::LocationDescriptor, bool is_single_step) {
        if (ctx.conf.HasOptimization(OptimizationFlag::ReturnStackBuffer) && !is_single_step) {
            Xbyak_loongarch64::Label fail;
            code.add_imm(Wscratch0, code.zero, A64::LocationDescriptor::fpcr_mask, Xscratch1);
//            code.add_d(Wscratch0, A64::LocationDescriptor::fpcr_mask, code.zero);
            code.ld_d(code.a0, Xstate, offsetof(A64JitState, fpcr));
            code.ld_d(code.a1, Xstate, offsetof(A64JitState, pc));
            code.and_(code.a0, code.a0, Wscratch0);
            code.add_imm(Wscratch0, code.zero, A64::LocationDescriptor::pc_mask, Xscratch1);
            code.and_(code.a1, code.a1, Wscratch0);
            code.slli_w(code.a0, code.a0, A64::LocationDescriptor::fpcr_shift);
            code.or_(code.a0, code.a0, code.a1);

            code.ld_d(Wscratch2, code.sp, offsetof(StackLayout, rsb_ptr));
            code.andi(Wscratch2, Wscratch2, RSBIndexMask);
            code.add_d(code.a2, code.sp, Xscratch2);
//            code.ADD(code.a2, code.sp, Xscratch2);
            code.sub_imm(Wscratch2, Wscratch2, sizeof(RSBEntry), code.t0);
            code.st_d(Wscratch2, code.sp, offsetof(StackLayout, rsb_ptr));
            code.ld_d(Xscratch0, code.a2, offsetof(StackLayout, rsb));
            code.ld_d(Xscratch1, code.a2, offsetof(StackLayout, rsb) + 8);
//            code.LDP(Xscratch0, Xscratch1, code.a2, offsetof(StackLayout, rsb));
            code.bne(code.a0, Xscratch0, fail);
//            code.CMP(code.a0, Xscratch0);
//            code.B(NE, fail);
            code.jirl(code.zero, Xscratch1, 0);

            code.L(fail);
        }

        EmitRelocation(code, ctx, LinkTarget::ReturnToDispatcher);
    }

    void EmitA64Terminal(BlockOfCode &code, EmitContext &ctx, IR::Term::FastDispatchHint,
                         IR::LocationDescriptor, bool) {
        EmitRelocation(code, ctx, LinkTarget::ReturnToDispatcher);

        // TODO: Implement FastDispatchHint optimization
    }

    void EmitA64Terminal(BlockOfCode &code, EmitContext &ctx, IR::Term::If terminal,
                         IR::LocationDescriptor initial_location, bool is_single_step) {
        Xbyak_loongarch64::Label pass = EmitA64Cond(code, ctx, terminal.if_);
        EmitA64Terminal(code, ctx, terminal.else_, initial_location, is_single_step);
        code.L(pass);
        EmitA64Terminal(code, ctx, terminal.then_, initial_location, is_single_step);
    }

    void EmitA64Terminal(BlockOfCode &code, EmitContext &ctx, IR::Term::CheckBit terminal,
                         IR::LocationDescriptor initial_location, bool is_single_step) {
        Xbyak_loongarch64::Label fail;
        code.ld_b(Wscratch0, code.sp, offsetof(StackLayout, check_bit));
        code.beqz(Wscratch0, fail);
        EmitA64Terminal(code, ctx, terminal.then_, initial_location, is_single_step);
        code.L(fail);
        EmitA64Terminal(code, ctx, terminal.else_, initial_location, is_single_step);
    }

    void EmitA64Terminal(BlockOfCode &code, EmitContext &ctx, IR::Term::CheckHalt terminal,
                         IR::LocationDescriptor initial_location, bool is_single_step) {
        Xbyak_loongarch64::Label fail;
        code.ll_w(Wscratch0, Xhalt, 0);
        code.bnez(Wscratch0, fail);
        EmitA64Terminal(code, ctx, terminal.else_, initial_location, is_single_step);
        code.L(fail);
        EmitRelocation(code, ctx, LinkTarget::ReturnToDispatcher);
    }

    void EmitA64Terminal(BlockOfCode &code, EmitContext &ctx, IR::Term::Terminal terminal,
                         IR::LocationDescriptor initial_location, bool is_single_step) {
        boost::apply_visitor([&](const auto &t) { EmitA64Terminal(code, ctx, t, initial_location, is_single_step); },
                             terminal);
    }

    void EmitA64Terminal(BlockOfCode &code, EmitContext &ctx) {
        const A64::LocationDescriptor location{ctx.block.Location()};
        EmitA64Terminal(code, ctx, ctx.block.GetTerminal(), location.SetSingleStepping(false),
                        location.SingleStepping());
    }

    void EmitA64ConditionFailedTerminal(BlockOfCode &code, EmitContext &ctx) {
        const A64::LocationDescriptor location{ctx.block.Location()};
        EmitA64Terminal(code, ctx, IR::Term::LinkBlock{ctx.block.ConditionFailedLocation()},
                        location.SetSingleStepping(false), location.SingleStepping());
    }

    void EmitA64CheckMemoryAbort(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst,
                                 Xbyak_loongarch64::Label &end) {
        if (!ctx.conf.check_halt_on_memory_access) {
            return;
        }

        const A64::LocationDescriptor current_location{IR::LocationDescriptor{inst->GetArg(0).GetU64()}};

        code.ll_w(Xscratch0, Xhalt, 0);
        code.add_imm(Xscratch1, code.zero, static_cast<u32>(HaltReason::MemoryAbort), Xscratch2);
        code.beq(Xscratch0, Xscratch1, end);
//        code.B(EQ, end);
        code.add_imm(Xscratch0, code.zero, current_location.PC(), Xscratch1);
//        code.add_d(Xscratch0, current_location.PC(), code.zero);
        code.st_d(Xscratch0, Xstate, offsetof(A64JitState, pc));
        EmitRelocation(code, ctx, LinkTarget::ReturnFromRunCode);
    }

    template<>
    void EmitIR<IR::Opcode::A64SetCheckBit>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        if (args[0].IsImmediate()) {
            if (args[0].GetImmediateU1()) {
                code.addi_w(Wscratch0, code.zero, 1);
//                code.add_d(Wscratch0, 1, code.zero);
                code.st_b(Wscratch0, code.sp, offsetof(StackLayout, check_bit));
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
    void EmitIR<IR::Opcode::A64GetCFlag>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto Wflag = ctx.reg_alloc.WriteW(inst);
        RegAlloc::Realize(Wflag);
        code.ld_d(Wflag, Xstate, offsetof(A64JitState, cpsr_nzcv));
        code.andi(Wflag, Wflag, 1 << 29);
    }

    template<>
    void EmitIR<IR::Opcode::A64GetNZCVRaw>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto Wnzcv = ctx.reg_alloc.WriteW(inst);
        RegAlloc::Realize(Wnzcv);

        code.ld_d(Wnzcv, Xstate, offsetof(A64JitState, cpsr_nzcv));
    }

    template<>
    void EmitIR<IR::Opcode::A64SetNZCVRaw>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Wnzcv = ctx.reg_alloc.ReadW(args[0]);
        RegAlloc::Realize(Wnzcv);

        code.st_d(Wnzcv, Xstate, offsetof(A64JitState, cpsr_nzcv));
    }

    template<>
    void EmitIR<IR::Opcode::A64SetNZCV>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Wnzcv = ctx.reg_alloc.ReadW(args[0]);
        RegAlloc::Realize(Wnzcv);

        code.st_d(Wnzcv, Xstate, offsetof(A64JitState, cpsr_nzcv));
    }

    template<>
    void EmitIR<IR::Opcode::A64GetW>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        const A64::Reg reg = inst->GetArg(0).GetA64RegRef();

        auto Wresult = ctx.reg_alloc.WriteW(inst);
        RegAlloc::Realize(Wresult);

        // TODO: Detect if Gpr vs Fpr is more appropriate

        code.ld_w(Wresult, Xstate, offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg));
    }

    template<>
    void EmitIR<IR::Opcode::A64GetX>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        const A64::Reg reg = inst->GetArg(0).GetA64RegRef();

        auto Xresult = ctx.reg_alloc.WriteX(inst);
        RegAlloc::Realize(Xresult);

        // TODO: Detect if Gpr vs Fpr is more appropriate

        code.ld_d(Xresult, Xstate, offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg));
    }

    template<>
    void EmitIR<IR::Opcode::A64GetS>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
        auto Sresult = ctx.reg_alloc.WriteQ(inst);
        RegAlloc::Realize(Sresult);
        code.vld(Sresult, Xstate, offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec));
    }

    template<>
    void EmitIR<IR::Opcode::A64GetD>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
        auto Dresult = ctx.reg_alloc.WriteQ(inst);
        RegAlloc::Realize(Dresult);
        code.vld(Dresult, Xstate, offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec));
    }

    template<>
    void EmitIR<IR::Opcode::A64GetQ>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
        auto Qresult = ctx.reg_alloc.WriteQ(inst);
        RegAlloc::Realize(Qresult);
        code.vld(Qresult, Xstate, offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec));
    }

    template<>
    void EmitIR<IR::Opcode::A64GetSP>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto Xresult = ctx.reg_alloc.WriteX(inst);
        RegAlloc::Realize(Xresult);

        code.ld_d(Xresult, Xstate, offsetof(A64JitState, sp));
    }

    template<>
    void EmitIR<IR::Opcode::A64GetFPCR>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto Wresult = ctx.reg_alloc.WriteW(inst);
        RegAlloc::Realize(Wresult);

        code.ld_d(Wresult, Xstate, offsetof(A64JitState, fpcr));
    }

    static u32 GetFPSRImpl(A64JitState *jit_state) {
        return jit_state->GetFpsr();
    }

    template<>
    void EmitIR<IR::Opcode::A64GetFPSR>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto Wresult = ctx.reg_alloc.WriteW(inst);
        RegAlloc::Realize(Wresult);

        code.movfcsr2gr(Xscratch0, code.fcsr0);
        code.st_w(Xscratch0, Xstate, offsetof(A64JitState, guest_FCSR));
        code.add_d(code.a0, Xstate, code.zero);
        code.CallFunction(GetFPSRImpl);
        code.add_d(Wresult, code.zero, code.a0);

//    code.ld_d(Wresult, Xstate, offsetof(A64JitState, guest_FCSR));
    }

    template<>
    void EmitIR<IR::Opcode::A64SetW>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        const A64::Reg reg = inst->GetArg(0).GetA64RegRef();

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        auto Wvalue = ctx.reg_alloc.ReadW(args[1]);
        RegAlloc::Realize(Wvalue);

        code.st_w(Wvalue, Xstate, offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg));
    }

    template<>
    void EmitIR<IR::Opcode::A64SetX>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        const A64::Reg reg = inst->GetArg(0).GetA64RegRef();

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        auto Xvalue = ctx.reg_alloc.ReadX(args[1]);
        RegAlloc::Realize(Xvalue);

        code.st_d(Xvalue, Xstate, offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg));
    }

    template<>
    void EmitIR<IR::Opcode::A64SetS>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
        auto Svalue = ctx.reg_alloc.ReadS(args[1]);
        RegAlloc::Realize(Svalue);

        code.vstelm_w(Svalue, Xstate, offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec), 0);
    }

    template<>
    void EmitIR<IR::Opcode::A64SetD>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
        auto Dvalue = ctx.reg_alloc.ReadD(args[1]);
        RegAlloc::Realize(Dvalue);

        code.vstelm_d(Dvalue, Xstate, offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec), 0);

    }

    template<>
    void EmitIR<IR::Opcode::A64SetQ>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
        auto Qvalue = ctx.reg_alloc.ReadQ(args[1]);
        RegAlloc::Realize(Qvalue);
        code.vst(Qvalue, Xstate, offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec));

//    code.st_d(Qvalue, Xstate, offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec));
    }

    template<>
    void EmitIR<IR::Opcode::A64SetSP>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Xvalue = ctx.reg_alloc.ReadX(args[0]);
        RegAlloc::Realize(Xvalue);
        code.st_d(Xvalue, Xstate, offsetof(A64JitState, sp));
    }

    static void SetFPCRImpl(A64JitState *jit_state, u32 value) {
        jit_state->SetFpcr(value);
    }


    template<>
    void EmitIR<IR::Opcode::A64SetFPCR>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Wvalue = ctx.reg_alloc.ReadW(args[0]);
        RegAlloc::Realize(Wvalue);
        ctx.reg_alloc.PrepareForCall({}, args[0]);
        code.add_d(code.a0, Xstate, code.zero);
        code.CallFunction(SetFPCRImpl);
        code.st_w(Wvalue, Xstate, offsetof(A64JitState, guest_FCSR));
        code.movgr2fcsr(code.fcsr0, Wvalue);
//    code.MSR(Xbyak_loongarch64::SystemReg::FPCR, Wvalue->toX());
    }

    static void SetFPSRImpl(A64JitState *jit_state, u32 value) {
        jit_state->SetFpsr(value);
    }

    template<>
    void EmitIR<IR::Opcode::A64SetFPSR>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Wvalue = ctx.reg_alloc.ReadW(args[0]);
        RegAlloc::Realize(Wvalue);
        code.add_d(code.a0, Xstate, code.zero);
        code.CallFunction(SetFPSRImpl);
        // TODO lijie guoji
        code.st_w(Wvalue, Xstate, offsetof(A64JitState, guest_FCSR));
        // FIXME convert to arch spec
        code.movgr2fcsr(code.fcsr0, Wvalue);
//    code.MSR(Xbyak_loongarch64::SystemReg::FPSR, Wvalue->toX());
    }

    template<>
    void EmitIR<IR::Opcode::A64SetPC>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Xvalue = ctx.reg_alloc.ReadX(args[0]);
        RegAlloc::Realize(Xvalue);
        code.st_d(Xvalue, Xstate, offsetof(A64JitState, pc));
    }

    template<>
    void
    EmitIR<IR::Opcode::A64CallSupervisor>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        ctx.reg_alloc.PrepareForCall();

        if (ctx.conf.enable_cycle_counting) {
            code.ld_d(code.a1, code.sp, offsetof(StackLayout, cycles_to_run));
            code.sub_d(code.a1, code.a1, Xticks);
            EmitRelocation(code, ctx, LinkTarget::AddTicks);
        }

        code.add_imm(code.a1, code.zero, args[0].GetImmediateU32(), Xscratch0);
//    code.add_d(code.a1, args[0].GetImmediateU32(), code.zero);
        EmitRelocation(code, ctx, LinkTarget::CallSVC);

        if (ctx.conf.enable_cycle_counting) {
            EmitRelocation(code, ctx, LinkTarget::GetTicksRemaining);
            code.st_d(code.a0, code.sp, offsetof(StackLayout, cycles_to_run));
            code.add_d(Xticks, code.a0, code.zero);
        }
    }

    template<>
    void
    EmitIR<IR::Opcode::A64ExceptionRaised>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        ctx.reg_alloc.PrepareForCall();

        if (ctx.conf.enable_cycle_counting) {
            code.ld_d(code.a1, code.sp, offsetof(StackLayout, cycles_to_run));
            code.sub_d(code.a1, code.a1, Xticks);
            EmitRelocation(code, ctx, LinkTarget::AddTicks);
        }
        code.add_imm(code.a1, code.zero, args[0].GetImmediateU32(), Xscratch0);
        code.add_imm(code.a2, code.zero, args[1].GetImmediateU32(), Xscratch0);
        EmitRelocation(code, ctx, LinkTarget::ExceptionRaised);

        if (ctx.conf.enable_cycle_counting) {
            EmitRelocation(code, ctx, LinkTarget::GetTicksRemaining);
            code.st_d(code.a0, code.sp, offsetof(StackLayout, cycles_to_run));
            code.add_d(Xticks, code.a0, code.zero);
        }
    }

    template<>
    void EmitIR<IR::Opcode::A64DataCacheOperationRaised>(BlockOfCode &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        ctx.reg_alloc.PrepareForCall({}, args[1], args[2]);
        EmitRelocation(code, ctx, LinkTarget::DataCacheOperationRaised);
    }

    template<>
    void
    EmitIR<IR::Opcode::A64InstructionCacheOperationRaised>(BlockOfCode &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        ctx.reg_alloc.PrepareForCall({}, args[0], args[1]);
        EmitRelocation(code, ctx, LinkTarget::InstructionCacheOperationRaised);
    }

    template<>
    void EmitIR<IR::Opcode::A64DataSynchronizationBarrier>(BlockOfCode &code, EmitContext &,
                                                           IR::Inst *) {
        code.dbar(0);
    }

    template<>
    void EmitIR<IR::Opcode::A64DataMemoryBarrier>(BlockOfCode &code, EmitContext &, IR::Inst *) {
        code.dbar(0x700);
    }

    template<>
    void
    EmitIR<IR::Opcode::A64InstructionSynchronizationBarrier>(BlockOfCode &code, EmitContext &ctx,
                                                             IR::Inst *) {
        if (!ctx.conf.hook_isb) {
            return;
        }

        ctx.reg_alloc.PrepareForCall();
        EmitRelocation(code, ctx, LinkTarget::InstructionSynchronizationBarrierRaised);
    }

    template<>
    void EmitIR<IR::Opcode::A64GetCNTFRQ>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto Xvalue = ctx.reg_alloc.WriteX(inst);
        RegAlloc::Realize(Xvalue);
        code.add_imm(Xvalue, code.zero, ctx.conf.cntfreq_el0, Xscratch0);
    }

    template<>
    void EmitIR<IR::Opcode::A64GetCNTPCT>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        ctx.reg_alloc.PrepareForCall();
        if (!ctx.conf.wall_clock_cntpct && ctx.conf.enable_cycle_counting) {
            code.ld_d(code.a1, code.sp, offsetof(StackLayout, cycles_to_run));
            code.sub_d(code.a1, code.a1, Xticks);
            EmitRelocation(code, ctx, LinkTarget::AddTicks);
            EmitRelocation(code, ctx, LinkTarget::GetTicksRemaining);
            code.st_d(code.a0, code.sp, offsetof(StackLayout, cycles_to_run));
            code.add_d(Xticks, code.a0, code.zero);
        }
        EmitRelocation(code, ctx, LinkTarget::GetCNTPCT);
        ctx.reg_alloc.DefineAsRegister(inst, code.a0);
    }

    template<>
    void EmitIR<IR::Opcode::A64GetCTR>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto Wvalue = ctx.reg_alloc.WriteW(inst);
        RegAlloc::Realize(Wvalue);
        code.add_imm(Wvalue, code.zero, ctx.conf.ctr_el0, Xscratch0);

    }

    template<>
    void EmitIR<IR::Opcode::A64GetDCZID>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto Wvalue = ctx.reg_alloc.WriteW(inst);
        RegAlloc::Realize(Wvalue);
        code.add_imm(Wvalue, code.zero, ctx.conf.dczid_el0, Xscratch0);
    }

    template<>
    void EmitIR<IR::Opcode::A64GetTPIDR>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto Xvalue = ctx.reg_alloc.WriteX(inst);
        RegAlloc::Realize(Xvalue);
        code.add_imm(Xscratch0, code.zero, mcl::bit_cast<u64>(ctx.conf.tpidr_el0), Xscratch0);
        code.ld_d(Xvalue, Xscratch0, 0);
    }

    template<>
    void EmitIR<IR::Opcode::A64GetTPIDRRO>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto Xvalue = ctx.reg_alloc.WriteX(inst);
        RegAlloc::Realize(Xvalue);
        code.add_imm(Xscratch0, code.zero, mcl::bit_cast<u64>(ctx.conf.tpidrro_el0), Xscratch0);
        code.ld_d(Xvalue, Xscratch0, 0);
    }

    template<>
    void EmitIR<IR::Opcode::A64SetTPIDR>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Xvalue = ctx.reg_alloc.ReadX(args[0]);
        RegAlloc::Realize(Xvalue);
        code.add_imm(Xscratch0, code.zero, mcl::bit_cast<u64>(ctx.conf.tpidr_el0), Xscratch0);
        code.stx_d(Xvalue, Xscratch0, code.zero);
    }

}  // namespace Dynarmic::Backend::LoongArch64
