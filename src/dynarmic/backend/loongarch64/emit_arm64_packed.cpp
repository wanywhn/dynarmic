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
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"
#include "xbyak_loongarch64.h"
#include "xbyak_loongarch64_util.h"

namespace Dynarmic::Backend::LoongArch64 {

using namespace Xbyak_loongarch64::util;

template<typename EmitFn>
static void EmitPackedOp(Xbyak_loongarch64::CodeGenerator&, EmitContext& ctx, IR::Inst* inst, EmitFn emit) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Vresult = ctx.reg_alloc.WriteD(inst);
    auto Va = ctx.reg_alloc.ReadD(args[0]);
    auto Vb = ctx.reg_alloc.ReadD(args[1]);
    RegAlloc::Realize(Vresult, Va, Vb);

    emit(Vresult, Va, Vb);
}

template<typename EmitFn>
static void EmitSaturatedPackedOp(Xbyak_loongarch64::CodeGenerator&, EmitContext& ctx, IR::Inst* inst, EmitFn emit) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Vresult = ctx.reg_alloc.WriteD(inst);
    auto Va = ctx.reg_alloc.ReadD(args[0]);
    auto Vb = ctx.reg_alloc.ReadD(args[1]);
    RegAlloc::Realize(Vresult, Va, Vb);
    ctx.fpsr.Spill();

    emit(Vresult, Va, Vb);
}

template<>
void EmitIR<IR::Opcode::PackedAddU8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto ge_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetGEFromOp);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteD(inst);
    auto Va = ctx.reg_alloc.ReadD(args[0]);
    auto Vb = ctx.reg_alloc.ReadD(args[1]);
    RegAlloc::Realize(Vresult, Va, Vb);

    code.vadd_b(Vresult, Va, Vb);
//    code.ADD(Vresult->B8(), Va->B8(), Vb->B8());

    if (ge_inst) {
        auto Vge = ctx.reg_alloc.WriteD(ge_inst);
        RegAlloc::Realize(Vge);
        // TODO how?
        code.CMHI(Vge->B8(), Va->B8(), Vresult->B8());
    }
}

template<>
void EmitIR<IR::Opcode::PackedAddS8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto ge_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetGEFromOp);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteD(inst);
    auto Va = ctx.reg_alloc.ReadD(args[0]);
    auto Vb = ctx.reg_alloc.ReadD(args[1]);
    RegAlloc::Realize(Vresult, Va, Vb);

    code.vadd_b(Vresult, Va, Vb);

//    code.ADD(Vresult->B8(), Va->B8(), Vb->B8());

    if (ge_inst) {
        auto Vge = ctx.reg_alloc.WriteD(ge_inst);
        RegAlloc::Realize(Vge);

        code.SHADD(Vge->B8(), Va->B8(), Vb->B8());
        code.CMGE(Vge->B8(), Vge->B8(), 0);
    }
}

template<>
void EmitIR<IR::Opcode::PackedSubU8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto ge_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetGEFromOp);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteD(inst);
    auto Va = ctx.reg_alloc.ReadD(args[0]);
    auto Vb = ctx.reg_alloc.ReadD(args[1]);
    RegAlloc::Realize(Vresult, Va, Vb);

    code.sub_imm(Vresult->B8(), Va->B8(), Vb->B8(), code.t0);

    if (ge_inst) {
        auto Vge = ctx.reg_alloc.WriteD(ge_inst);
        RegAlloc::Realize(Vge);

        code.UHSUB(Vge->B8(), Va->B8(), Vb->B8());
        code.CMGE(Vge->B8(), Vge->B8(), 0);
    }
}

template<>
void EmitIR<IR::Opcode::PackedSubS8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto ge_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetGEFromOp);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteD(inst);
    auto Va = ctx.reg_alloc.ReadD(args[0]);
    auto Vb = ctx.reg_alloc.ReadD(args[1]);
    RegAlloc::Realize(Vresult, Va, Vb);

    code.sub_imm(Vresult->B8(), Va->B8(), Vb->B8(), code.t0);

    if (ge_inst) {
        auto Vge = ctx.reg_alloc.WriteD(ge_inst);
        RegAlloc::Realize(Vge);

        code.SHSUB(Vge->B8(), Va->B8(), Vb->B8());
        code.CMGE(Vge->B8(), Vge->B8(), 0);
    }
}

template<>
void EmitIR<IR::Opcode::PackedAddU16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto ge_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetGEFromOp);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteD(inst);
    auto Va = ctx.reg_alloc.ReadD(args[0]);
    auto Vb = ctx.reg_alloc.ReadD(args[1]);
    RegAlloc::Realize(Vresult, Va, Vb);

    code.ADD(Vresult->H4(), Va->H4(), Vb->H4());

    if (ge_inst) {
        auto Vge = ctx.reg_alloc.WriteD(ge_inst);
        RegAlloc::Realize(Vge);

        code.CMHI(Vge->H4(), Va->H4(), Vresult->H4());
    }
}

template<>
void EmitIR<IR::Opcode::PackedAddS16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto ge_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetGEFromOp);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteD(inst);
    auto Va = ctx.reg_alloc.ReadD(args[0]);
    auto Vb = ctx.reg_alloc.ReadD(args[1]);
    RegAlloc::Realize(Vresult, Va, Vb);

    code.ADD(Vresult->H4(), Va->H4(), Vb->H4());

    if (ge_inst) {
        auto Vge = ctx.reg_alloc.WriteD(ge_inst);
        RegAlloc::Realize(Vge);

        code.SHADD(Vge->H4(), Va->H4(), Vb->H4());
        code.CMGE(Vge->H4(), Vge->H4(), 0);
    }
}

template<>
void EmitIR<IR::Opcode::PackedSubU16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto ge_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetGEFromOp);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteD(inst);
    auto Va = ctx.reg_alloc.ReadD(args[0]);
    auto Vb = ctx.reg_alloc.ReadD(args[1]);
    RegAlloc::Realize(Vresult, Va, Vb);

    code.sub_imm(Vresult->H4(), Va->H4(), Vb->H4(), code.t0);

    if (ge_inst) {
        auto Vge = ctx.reg_alloc.WriteD(ge_inst);
        RegAlloc::Realize(Vge);

        code.UHSUB(Vge->H4(), Va->H4(), Vb->H4());
        code.CMGE(Vge->H4(), Vge->H4(), 0);
    }
}

template<>
void EmitIR<IR::Opcode::PackedSubS16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto ge_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetGEFromOp);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteD(inst);
    auto Va = ctx.reg_alloc.ReadD(args[0]);
    auto Vb = ctx.reg_alloc.ReadD(args[1]);
    RegAlloc::Realize(Vresult, Va, Vb);

    code.sub_imm(Vresult->H4(), Va->H4(), Vb->H4(), code.t0);

    if (ge_inst) {
        auto Vge = ctx.reg_alloc.WriteD(ge_inst);
        RegAlloc::Realize(Vge);

        code.SHSUB(Vge->H4(), Va->H4(), Vb->H4());
        code.CMGE(Vge->H4(), Vge->H4(), 0);
    }
}

template<bool add_is_hi, bool is_signed, bool is_halving>
static void EmitPackedAddSub(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto ge_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetGEFromOp);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteX(inst);
    auto reg_a_hi = ctx.reg_alloc.ReadX(args[0]);
    auto reg_b_hi = ctx.reg_alloc.ReadX(args[1]);
    RegAlloc::Realize(Vresult, reg_a_hi, reg_b_hi);
    auto reg_a_lo = Wscratch0;
    auto reg_b_lo = Wscratch1;
    auto reg_diff = reg_a_lo;
    auto reg_sum = reg_a_lo;


    if (is_signed) {
        code.ext_w_h(reg_a_lo, reg_a_hi);
        code.ext_w_h(reg_b_lo, reg_b_hi);
        code.srai_w(reg_a_hi, reg_a_hi, 16);
        code.srai_w(reg_b_hi, reg_b_hi, 16);
//        code.SXTL(V0.S4(), Va->H4());
//        code.SXTL(V1.S4(), Vb->H4());
    } else {
        code.bstrins_w(reg_a_lo, reg_a_hi, 15, 0);
        code.bstrins_w(reg_b_lo, reg_b_hi, 15, 0);
        code.srli_d(reg_a_hi, reg_a_hi, 16);
        code.srli_d(reg_b_hi, reg_b_hi, 16);
//        code.UXTL(V0.S4(), Va->H4());
//        code.UXTL(V1.S4(), Vb->H4());
    }

    if (add_is_hi) {
        code.sub_w(reg_a_lo, reg_a_lo, reg_b_hi);
        code.add_w(reg_a_hi, reg_a_hi, reg_b_lo);
        reg_diff = reg_a_lo;
        reg_sum = reg_a_hi;
    } else {
        code.add_w(reg_a_lo, reg_a_lo, reg_b_hi);
        code.sub_w(reg_a_hi, reg_a_hi, reg_b_lo);
        reg_diff = reg_a_hi;
        reg_sum = reg_a_lo;
    }

    if (ge_inst){
        auto Vge = ctx.reg_alloc.WriteW(ge_inst);
        RegAlloc::Realize(Vge);
        Xbyak_loongarch64::XReg ge_sum = reg_b_hi;
        auto ge_diff = reg_b_lo;
        code.add_w(ge_sum, code.zero, reg_sum);
        code.add_w(ge_diff, code.zero, reg_diff);

        if(!is_signed){
            code.slli_w(ge_sum, ge_sum, 15);
            code.srai_w(ge_sum, ge_sum, 31);
        } else {
            code.nor(ge_sum, ge_sum, code.zero);
            code.srai_w(ge_sum, ge_sum, 31);
        }
        code.nor(ge_diff, ge_diff, code.zero);
        code.srai_w(ge_diff, ge_diff, 31);

        code.add_imm(Vresult, code.zero, add_is_hi? 0xFFFF0000:0x0000FFFF,Wscratch2);
        code.and_(ge_sum, ge_sum, Vresult);
        code.add_imm(Vresult, code.zero, add_is_hi? 0x0000FFFF:0xFFFF0000,Wscratch2);
        code.and_(ge_diff, ge_diff, Vresult);
        code.or_(Vge, ge_sum, ge_diff);
    }

    code.bstrins_w(reg_a_hi, reg_a_lo, 15, 0);
    if(is_halving) {
        code.srli_w(Vresult, reg_a_hi, 1);
    } else {
        code.add_w(Vresult, code.zero, reg_a_hi);
    }
}

template<>
void EmitIR<IR::Opcode::PackedAddSubU16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddSub<true, false, false>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::PackedAddSubS16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddSub<true, true, false>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::PackedSubAddU16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddSub<false, false, false>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::PackedSubAddS16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddSub<false, true, false>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::PackedHalvingAddU8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) { code.UHADD(Vresult->B8(), Va->B8(), Vb->B8()); });
}

template<>
void EmitIR<IR::Opcode::PackedHalvingAddS8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) { code.SHADD(Vresult->B8(), Va->B8(), Vb->B8()); });
}

template<>
void EmitIR<IR::Opcode::PackedHalvingSubU8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) { code.UHSUB(Vresult->B8(), Va->B8(), Vb->B8()); });
}

template<>
void EmitIR<IR::Opcode::PackedHalvingSubS8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) { code.SHSUB(Vresult->B8(), Va->B8(), Vb->B8()); });
}

template<>
void EmitIR<IR::Opcode::PackedHalvingAddU16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) { code.UHADD(Vresult->H4(), Va->H4(), Vb->H4()); });
}

template<>
void EmitIR<IR::Opcode::PackedHalvingAddS16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) { code.SHADD(Vresult->H4(), Va->H4(), Vb->H4()); });
}

template<>
void EmitIR<IR::Opcode::PackedHalvingSubU16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) { code.UHSUB(Vresult->H4(), Va->H4(), Vb->H4()); });
}

template<>
void EmitIR<IR::Opcode::PackedHalvingSubS16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) { code.SHSUB(Vresult->H4(), Va->H4(), Vb->H4()); });
}

template<>
void EmitIR<IR::Opcode::PackedHalvingAddSubU16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddSub<true, false, true>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::PackedHalvingAddSubS16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddSub<true, true, true>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::PackedHalvingSubAddU16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddSub<false, false, true>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::PackedHalvingSubAddS16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddSub<false, true, true>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::PackedSaturatedAddU8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitSaturatedPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) { code.UQADD(Vresult->B8(), Va->B8(), Vb->B8()); });
}

template<>
void EmitIR<IR::Opcode::PackedSaturatedAddS8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitSaturatedPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) { code.SQADD(Vresult->B8(), Va->B8(), Vb->B8()); });
}

template<>
void EmitIR<IR::Opcode::PackedSaturatedSubU8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitSaturatedPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) { code.UQSUB(Vresult->B8(), Va->B8(), Vb->B8()); });
}

template<>
void EmitIR<IR::Opcode::PackedSaturatedSubS8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitSaturatedPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) { code.SQSUB(Vresult->B8(), Va->B8(), Vb->B8()); });
}

template<>
void EmitIR<IR::Opcode::PackedSaturatedAddU16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitSaturatedPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) { code.UQADD(Vresult->H4(), Va->H4(), Vb->H4()); });
}

template<>
void EmitIR<IR::Opcode::PackedSaturatedAddS16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitSaturatedPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) { code.SQADD(Vresult->H4(), Va->H4(), Vb->H4()); });
}

template<>
void EmitIR<IR::Opcode::PackedSaturatedSubU16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitSaturatedPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) { code.UQSUB(Vresult->H4(), Va->H4(), Vb->H4()); });
}

template<>
void EmitIR<IR::Opcode::PackedSaturatedSubS16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitSaturatedPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) { code.SQSUB(Vresult->H4(), Va->H4(), Vb->H4()); });
}

template<>
void EmitIR<IR::Opcode::PackedAbsDiffSumU8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) {
        code.MOVI(D2, Xbyak_loongarch64::RepImm{0b00001111});
        code.UABD(Vresult->B8(), Va->B8(), Vb->B8());
        code.andi(Vresult->B8(), Vresult->B8(), V2.B8());  // TODO: Zext tracking
        code.UADDLV(Vresult->toH(), Vresult->B8());
    });
}

template<>
void EmitIR<IR::Opcode::PackedSelect>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Vresult = ctx.reg_alloc.WriteD(inst);
    auto Vge = ctx.reg_alloc.ReadD(args[0]);
    auto Va = ctx.reg_alloc.ReadD(args[1]);
    auto Vb = ctx.reg_alloc.ReadD(args[2]);
    RegAlloc::Realize(Vresult, Vge, Va, Vb);

    code.FMOV(Vresult, Vge);  // TODO: Move elimination
    code.BSL(Vresult->B8(), Vb->B8(), Va->B8());
}

}  // namespace Dynarmic::Backend::LoongArch64
