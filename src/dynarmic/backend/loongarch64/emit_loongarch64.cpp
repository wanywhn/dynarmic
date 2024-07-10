/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/loongarch64/emit_loongarch64.h"

#include "dynarmic/backend/loongarch64/abi.h"
#include "dynarmic/backend/loongarch64/emit_context.h"
#include "dynarmic/backend/loongarch64/fpsr_manager.h"
#include "dynarmic/backend/loongarch64/reg_alloc.h"
#include "dynarmic/backend/loongarch64/verbose_debugging_output.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"
#include "xbyak_loongarch64.h"
#include "xbyak_loongarch64_util.h"
#include "nzcv_util.h"

namespace Dynarmic::Backend::LoongArch64 {

    using namespace Xbyak_loongarch64::util;

    template<>
    void EmitIR<IR::Opcode::Void>(BlockOfCode &, EmitContext &, IR::Inst *) {}

    template<>
    void EmitIR<IR::Opcode::Identity>(BlockOfCode &, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        ctx.reg_alloc.DefineAsExisting(inst, args[0]);
    }

    template<>
    void EmitIR<IR::Opcode::Breakpoint>(BlockOfCode &code, EmitContext &, IR::Inst *) {
        code.break_(0);
    }

    template<>
    void EmitIR<IR::Opcode::CallHostFunction>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        ctx.reg_alloc.PrepareForCall(args[1], args[2], args[3]);
        code.add_imm(Xscratch0, code.zero, args[0].GetImmediateU64(), Xscratch1);
        code.jirl(code.ra, Xscratch0, 0);
    }

    template<>
    void EmitIR<IR::Opcode::PushRSB>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        if (!ctx.conf.HasOptimization(OptimizationFlag::ReturnStackBuffer)) {
            return;
        }

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        ASSERT(args[0].IsImmediate());
        const IR::LocationDescriptor target{args[0].GetImmediateU64()};

        code.ld_w(Wscratch2, code.sp, offsetof(StackLayout, rsb_ptr));
        code.addi_w(Wscratch2, Wscratch2, sizeof(RSBEntry));
        code.andi(Wscratch2, Wscratch2, RSBIndexMask);
        code.st_w(Wscratch2, code.sp, offsetof(StackLayout, rsb_ptr));
        code.add_d(Xscratch2, code.sp, Xscratch2);

        code.add_imm(Xscratch0, code.zero, target.Value(), Wscratch1);
        code.st_d(Xscratch0, Xscratch2, offsetof(StackLayout, rsb));

        EmitBlockLinkRelocation(code, ctx, target, BlockRelocationType::MoveToScratch1);
        code.st_d(Xscratch1, Xscratch2, offsetof(StackLayout, rsb) + sizeof(RSBEntry::code_ptr));
    }

    template<>
    void EmitIR<IR::Opcode::GetCarryFromOp>(BlockOfCode &, EmitContext &ctx, IR::Inst *inst) {
        [[maybe_unused]] auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        ASSERT(ctx.reg_alloc.WasValueDefined(inst));
    }

    template<>
    void EmitIR<IR::Opcode::GetOverflowFromOp>(BlockOfCode &, EmitContext &ctx, IR::Inst *inst) {
        [[maybe_unused]] auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        ASSERT(ctx.reg_alloc.WasValueDefined(inst));
    }

    template<>
    void EmitIR<IR::Opcode::GetGEFromOp>(BlockOfCode &, EmitContext &ctx, IR::Inst *inst) {
        [[maybe_unused]] auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        ASSERT(ctx.reg_alloc.WasValueDefined(inst));
    }

    template<>
    void EmitIR<IR::Opcode::GetNZCVFromOp>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        if (ctx.reg_alloc.WasValueDefined(inst)) {
            return;
        }
        auto Wvalue = ctx.reg_alloc.ReadW(args[0]);
        auto nzcv = ctx.reg_alloc.WriteW(inst);
        RegAlloc::Realize(Wvalue, nzcv);

        switch (args[0].GetType()) {
            case IR::Type::U32: {
                code.srli_w(Wscratch0, Wvalue, NZCV::arm_nzcv_shift);
                break;
            }
            case IR::Type::U64: {
                code.srli_d(Wscratch0, Wvalue, NZCV::arm_nzcv_shift + 32);
                break;
                default:
                    ASSERT_FALSE("Invalid type for GetNZCVFromOp");
                break;
            }
        }
        code.bstrpick_w(nzcv, Wscratch0, NZCV::arm_n_flag_inner_sft, NZCV::arm_n_flag_inner_sft);
        code.slli_w(nzcv, nzcv, NZCV::arm_n_flag_inner_sft);

        code.addi_w(Wscratch2, code.zero, 1);
        code.masknez(Wscratch0, Wscratch2, Wvalue);
        code.bstrins_w(nzcv, Wscratch0, NZCV::arm_z_flag_inner_sft, NZCV::arm_z_flag_inner_sft);
    }

    template<>
    void EmitIR<IR::Opcode::GetNZFromOp>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitIR<IR::Opcode::GetNZCVFromOp>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::GetUpperFromOp>(BlockOfCode &, EmitContext &ctx, IR::Inst *inst) {
        [[maybe_unused]] auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        ASSERT(ctx.reg_alloc.WasValueDefined(inst));
    }

    template<>
    void EmitIR<IR::Opcode::GetLowerFromOp>(BlockOfCode &, EmitContext &ctx, IR::Inst *inst) {
        [[maybe_unused]] auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        ASSERT(ctx.reg_alloc.WasValueDefined(inst));
    }

    template<>
    void EmitIR<IR::Opcode::GetCFlagFromNZCV>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        auto Wc = ctx.reg_alloc.WriteW(inst);
        auto Wnzcv = ctx.reg_alloc.ReadW(args[0]);
        RegAlloc::Realize(Wc, Wnzcv);
        code.bstrpick_d(Wc, Wnzcv, NZCV::arm_c_flag_inner_sft, NZCV::arm_c_flag_inner_sft);
    }

    template<>
    void EmitIR<IR::Opcode::NZCVFromPackedFlags>(BlockOfCode &, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        ctx.reg_alloc.DefineAsExisting(inst, args[0]);
    }

    static void EmitAddCycles(BlockOfCode &code, EmitContext &ctx, size_t cycles_to_add) {
        if (!ctx.conf.enable_cycle_counting) {
            return;
        }
        if (cycles_to_add == 0) {
            return;
        }
        // TODO use subi more efficient?
        code.sub_imm(Xticks, Xticks, cycles_to_add, Xscratch0);
    }

    EmittedBlockInfo
    EmitArm64(BlockOfCode &code, IR::Block block, const EmitConfig &conf, FastmemManager &fastmem_manager) {
        if (conf.very_verbose_debugging_output) {
            std::puts(IR::DumpBlock(block).c_str());
        }

        EmittedBlockInfo ebi;

        FpsrManager fpsr_manager{code, conf.state_fpsr_offset};
        // TODO change to loongarch fr
        RegAlloc reg_alloc{code, fpsr_manager, GPR_ORDER, FPR_ORDER};
        EmitContext ctx{block, reg_alloc, conf, ebi, fpsr_manager, fastmem_manager, {}};

        ebi.entry_point = code.getCurr<CodePtr>();

        if (ctx.block.GetCondition() == IR::Cond::AL) {
            ASSERT(!ctx.block.HasConditionFailedLocation());
        } else {
            ASSERT(ctx.block.HasConditionFailedLocation());
            Xbyak_loongarch64::Label pass;

            pass = conf.emit_cond(code, ctx, ctx.block.GetCondition());
            EmitAddCycles(code, ctx, ctx.block.ConditionFailedCycleCount());
            conf.emit_condition_failed_terminal(code, ctx);

            code.L(pass);
        }

        for (auto iter = block.begin(); iter != block.end(); ++iter) {
            IR::Inst *inst = &*iter;

            switch (inst->GetOpcode()) {
#define OPCODE(name, type, ...)                    \
    case IR::Opcode::name:                         \
        EmitIR<IR::Opcode::name>(code, ctx, inst); \
        break;
#define A32OPC(name, type, ...)                         \
    case IR::Opcode::A32##name:                         \
        EmitIR<IR::Opcode::A32##name>(code, ctx, inst); \
        break;
#define A64OPC(name, type, ...)                         \
    case IR::Opcode::A64##name:                         \
        EmitIR<IR::Opcode::A64##name>(code, ctx, inst); \
        break;

#include "dynarmic/ir/opcodes.inc"

#undef OPCODE
#undef A32OPC
#undef A64OPC
                default:
                    ASSERT_FALSE("Invalid opcode: {}", inst->GetOpcode());
                    break;
            }

            reg_alloc.UpdateAllUses();
            reg_alloc.AssertAllUnlocked();

            if (conf.very_verbose_debugging_output) {
                // FIXME test this func
//            EmitVerboseDebuggingOutput(code, ctx);
            }
        }

        // TODO : is this need or how to implement?
//    fpsr_manager.Spill();
        reg_alloc.AssertNoMoreUses();

        EmitAddCycles(code, ctx, block.CycleCount());
        conf.emit_terminal(code, ctx);
        code.break_(0);

        for (const auto &deferred_emit: ctx.deferred_emits) {
            deferred_emit();
        }
        code.break_(0);

        ebi.size = code.getCurr<CodePtr>() - ebi.entry_point;
        return ebi;
    }

    void EmitRelocation(BlockOfCode &code, EmitContext &ctx, LinkTarget link_target) {
        ctx.ebi.relocations.emplace_back(Relocation{code.getCurr<CodePtr>() - ctx.ebi.entry_point, link_target});
        code.nop();
    }

    void EmitBlockLinkRelocation(BlockOfCode &code, EmitContext &ctx, const IR::LocationDescriptor &descriptor,
                                 BlockRelocationType type) {
        ctx.ebi.block_relocations[descriptor].emplace_back(
                BlockRelocation{code.getCurr<CodePtr>() - ctx.ebi.entry_point, type});
        switch (type) {
            case BlockRelocationType::Branch:
                code.nop();
                break;
            case BlockRelocationType::MoveToScratch1:
                code.break_(0);
                code.nop();
                code.nop();
                code.nop();
                code.nop();
                code.nop();
                // this should enough. it depedent on how add_imm imp
                break;
            default:
                UNREACHABLE();
        }
    }

}  // namespace Dynarmic::Backend::LoongArch64
