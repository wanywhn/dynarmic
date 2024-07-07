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
#include "xbyak_loongarch64_reg.h"

namespace Dynarmic::Backend::LoongArch64 {

using namespace Xbyak_loongarch64::util;

template<typename EmitFn>
static void EmitPackedOp(BlockOfCode&, EmitContext& ctx, IR::Inst* inst, EmitFn emit) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Vresult = ctx.reg_alloc.WriteQ(inst);
    auto Va = ctx.reg_alloc.ReadQ(args[0]);
    auto Vb = ctx.reg_alloc.ReadQ(args[1]);
    RegAlloc::Realize(Vresult, Va, Vb);

    emit(Vresult, Va, Vb);
}

template<typename EmitFn>
static void EmitSaturatedPackedOp(BlockOfCode&, EmitContext& ctx, IR::Inst* inst, EmitFn emit) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Vresult = ctx.reg_alloc.WriteD(inst);
    auto Va = ctx.reg_alloc.ReadD(args[0]);
    auto Vb = ctx.reg_alloc.ReadD(args[1]);
    RegAlloc::Realize(Vresult, Va, Vb);
    ctx.fpsr.Spill();

    emit(Vresult, Va, Vb);
}

template<>
void EmitIR<IR::Opcode::PackedAddU8>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
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
        // FIXME vslt_bu how?
//        code.CMHI(Vge->B8(), Va->B8(), Vresult->B8());
//        code.vslt_bu(Vge, Vb, Va);
    }
}

template<>
void EmitIR<IR::Opcode::PackedAddS8>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
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
//        code.vsadd_b(Va, Va, Vb);
//        code.vseq_b(Vb, Vb, Vb);
        // TODO is this right?
//        code.vslt_b(Vge, Vb, Vge);
//        code.SHADD(Vge->B8(), Va->B8(), Vb->B8());
//        code.CMGE(Vge->B8(), Vge->B8(), 0);
// FIXME vslt_b
//            code.vslt_b(Vge, Vb, Va);
    }
}

template<>
void EmitIR<IR::Opcode::PackedSubU8>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    const auto ge_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetGEFromOp);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteD(inst);
    auto Va = ctx.reg_alloc.ReadD(args[0]);
    auto Vb = ctx.reg_alloc.ReadD(args[1]);
    RegAlloc::Realize(Vresult, Va, Vb);

    code.nop();
    // FIXME vsub_b
//    code.vsub_b(Vresult, Va, Vb);
//    code.sub_imm(Vresult->B8(), Va->B8(), Vb->B8(), code.t0);

    if (ge_inst) {
        auto Vge = ctx.reg_alloc.WriteD(ge_inst);
        RegAlloc::Realize(Vge);
        // FIXME vslt_bu
//        code.vslt_bu(Vge, Vb, Va);
//        code.vmax_bu(Vge, Va, Vb);
//        code.vseq_b(Vge, Vge, Va);
//        code.UHSUB(Vge->B8(), Va->B8(), Vb->B8());
//        code.CMGE(Vge->B8(), Vge->B8(), 0);
    }
}

template<>
void EmitIR<IR::Opcode::PackedSubS8>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    const auto ge_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetGEFromOp);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteD(inst);
    auto Va = ctx.reg_alloc.ReadD(args[0]);
    auto Vb = ctx.reg_alloc.ReadD(args[1]);
    RegAlloc::Realize(Vresult, Va, Vb);

    // FIXME vsub_b
//    code.vsub_b(Vresult, Va, Vb);
//    code.sub_imm(Vresult->B8(), Va->B8(), Vb->B8(), code.t0);
    code.nop();

    if (ge_inst) {
        auto Vge = ctx.reg_alloc.WriteD(ge_inst);
        RegAlloc::Realize(Vge);
        // FIXME vslt_b
//        code.vslt_b(Vge, Va, Vb);
//        code.SHSUB(Vge->B8(), Va->B8(), Vb->B8());
//        code.CMGE(Vge->B8(), Vge->B8(), 0);
    }
}

template<>
void EmitIR<IR::Opcode::PackedAddU16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    const auto ge_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetGEFromOp);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteD(inst);
    auto Va = ctx.reg_alloc.ReadD(args[0]);
    auto Vb = ctx.reg_alloc.ReadD(args[1]);
    RegAlloc::Realize(Vresult, Va, Vb);

    code.vadd_h(Vresult, Va, Vb);
//    code.ADD(Vresult->H4(), Va->H4(), Vb->H4());

    if (ge_inst) {
        auto Vge = ctx.reg_alloc.WriteD(ge_inst);
        RegAlloc::Realize(Vge);
        // FIXME vslt_hu
//        code.vslt_hu(Vge, Vresult, Va);
//        code.CMHI(Vge->H4(), Va->H4(), Vresult->H4());
    }
}

template<>
void EmitIR<IR::Opcode::PackedAddS16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    const auto ge_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetGEFromOp);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteD(inst);
    auto Va = ctx.reg_alloc.ReadD(args[0]);
    auto Vb = ctx.reg_alloc.ReadD(args[1]);
    RegAlloc::Realize(Vresult, Va, Vb);

    code.vadd_h(Vresult, Va, Vb);
//    code.ADD(Vresult->H4(), Va->H4(), Vb->H4());

    if (ge_inst) {
        auto Vge = ctx.reg_alloc.WriteD(ge_inst);
        RegAlloc::Realize(Vge);
        // FIXME vslt_h
//        code.vslt_h(Vge, Vb, Va);

//        code.SHADD(Vge->H4(), Va->H4(), Vb->H4());
//        code.CMGE(Vge->H4(), Vge->H4(), 0);
    }
}

template<>
void EmitIR<IR::Opcode::PackedSubU16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    const auto ge_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetGEFromOp);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteD(inst);
    auto Va = ctx.reg_alloc.ReadD(args[0]);
    auto Vb = ctx.reg_alloc.ReadD(args[1]);
    RegAlloc::Realize(Vresult, Va, Vb);

//    code.vsub_h(Vresult, Va, Vb);
//    code.sub_imm(Vresult->H4(), Va->H4(), Vb->H4(), code.t0);
    code.nop();

    if (ge_inst) {
        auto Vge = ctx.reg_alloc.WriteD(ge_inst);
        RegAlloc::Realize(Vge);
        // FIXME vslt_hu
//        code.vslt_hu(Vge, Vb, Va);
//        code.UHSUB(Vge->H4(), Va->H4(), Vb->H4());
//        code.CMGE(Vge->H4(), Vge->H4(), 0);
    }
}

template<>
void EmitIR<IR::Opcode::PackedSubS16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    const auto ge_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetGEFromOp);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteD(inst);
    auto Va = ctx.reg_alloc.ReadD(args[0]);
    auto Vb = ctx.reg_alloc.ReadD(args[1]);
    RegAlloc::Realize(Vresult, Va, Vb);

    code.nop();
    // FIXME vsub_h
//    code.vsub_h(Vresult, Va, Vb);
//    code.sub_imm(Vresult->H4(), Va->H4(), Vb->H4(), code.t0);

    if (ge_inst) {
        auto Vge = ctx.reg_alloc.WriteD(ge_inst);
        RegAlloc::Realize(Vge);
        // FIXME vslt_h
//        code.vslt_h(ge, Vb, Va);
//        code.SHSUB(Vge->H4(), Va->H4(), Vb->H4());
//        code.CMGE(Vge->H4(), Vge->H4(), 0);
    }
}

template<bool add_is_hi, bool is_signed, bool is_halving>
static void EmitPackedAddSub(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    const auto ge_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetGEFromOp);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteX(inst);
    auto reg_a_hi = ctx.reg_alloc.ReadX(args[0]);
    auto reg_b_hi = ctx.reg_alloc.ReadX(args[1]);
    RegAlloc::Realize(Vresult, reg_a_hi, reg_b_hi);
    auto reg_a_lo = Xscratch0;
    auto reg_b_lo = Xscratch1;
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
void EmitIR<IR::Opcode::PackedAddSubU16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddSub<true, false, false>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::PackedAddSubS16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddSub<true, true, false>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::PackedSubAddU16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddSub<false, false, false>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::PackedSubAddS16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddSub<false, true, false>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::PackedHalvingAddU8>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) {
        // TODO diff bettwen avgr?
//        code.xvavg_bu(Vresult, Va, Vb);
        // FIXME
        code.vavg_bu(Vresult, Va, Vb);
    });
}

template<>
void EmitIR<IR::Opcode::PackedHalvingAddS8>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) {
        code.vavg_b(Vresult, Va, Vb);
        // FIXME
    });
}

template<>
void EmitIR<IR::Opcode::PackedHalvingSubU8>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) {
        code.vavg_bu(Vresult, Va, Vb);
        code.vsub_b(Vresult, Vresult, Vb);
//        code.nop();
        // FIXME
    });
}

template<>
void EmitIR<IR::Opcode::PackedHalvingSubS8>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) {
        code.vavg_b(Vresult, Va, Vb);
        code.vsub_b(Vresult, Vresult, Vb);

    });
}

template<>
void EmitIR<IR::Opcode::PackedHalvingAddU16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) {
        code.vavg_hu(Vresult, Va, Vb);
        // FIXME
    });
}

template<>
void EmitIR<IR::Opcode::PackedHalvingAddS16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) {
        code.vavg_h(Vresult, Va, Vb);

    });
}

template<>
void EmitIR<IR::Opcode::PackedHalvingSubU16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) {
        code.vavg_hu(Vresult, Va, Vb);
        code.vsub_h(Vresult, Vresult, Vb);
        // FIXME

//        code.UHSUB(Vresult->H4(), Va->H4(), Vb->H4());
    });
}

template<>
void EmitIR<IR::Opcode::PackedHalvingSubS16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) {
        code.vavg_hu(Vresult, Va, Vb);
        code.vsub_h(Vresult, Vresult, Vb);
        // FIXME

//        code.SHSUB(Vresult->H4(), Va->H4(), Vb->H4());
    });
}

template<>
void EmitIR<IR::Opcode::PackedHalvingAddSubU16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddSub<true, false, true>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::PackedHalvingAddSubS16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddSub<true, true, true>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::PackedHalvingSubAddU16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddSub<false, false, true>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::PackedHalvingSubAddS16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddSub<false, true, true>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::PackedSaturatedAddU8>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitSaturatedPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) {
        code.vsadd_bu(Vresult, Va, Vb);
    });
}

template<>
void EmitIR<IR::Opcode::PackedSaturatedAddS8>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitSaturatedPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) {
                code.vsadd_b(Vresult, Va, Vb);
    });
}

template<>
void EmitIR<IR::Opcode::PackedSaturatedSubU8>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitSaturatedPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) {
        code.vssub_bu(Vresult, Va, Vb);
    });
}

template<>
void EmitIR<IR::Opcode::PackedSaturatedSubS8>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitSaturatedPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) {
        code.vssub_b(Vresult, Va, Vb);
    });
}

template<>
void EmitIR<IR::Opcode::PackedSaturatedAddU16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitSaturatedPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) {
        code.vsadd_hu(Vresult, Va, Vb);
    });
}

template<>
void EmitIR<IR::Opcode::PackedSaturatedAddS16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitSaturatedPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) {

        code.vsadd_h(Vresult, Va, Vb);
    });
}

template<>
void EmitIR<IR::Opcode::PackedSaturatedSubU16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitSaturatedPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) {
        code.vssub_hu(Vresult, Va, Vb);
    });
}

template<>
void EmitIR<IR::Opcode::PackedSaturatedSubS16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitSaturatedPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) {
        code.vssub_h(Vresult, Va, Vb);
    });
}

template<>
void EmitIR<IR::Opcode::PackedAbsDiffSumU8>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) {
        code.vxor_v(Vresult, Vresult, Vresult);
        code.add_imm(Xscratch0, code.zero, 0xffff'ffff, Xscratch1);
        code.vinsgr2vr_w(Vresult, Xscratch0, 0);
        code.vand_v(Va, Va, Vresult);
        code.vand_v(Vb, Vb, Vresult);
        code.vabsd_bu(Vresult, Va, Vb);
        // FIXME add into xybak
        code.vhaddw_hu_bu(Vresult, Vresult, Vresult);
        code.vhaddw_wu_hu(Vresult, Vresult, Vresult);
    });
}

template<>
void EmitIR<IR::Opcode::PackedSelect>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Vresult = ctx.reg_alloc.WriteD(inst);
    auto Vge = ctx.reg_alloc.ReadX(args[0]);
    auto Va = ctx.reg_alloc.ReadX(args[1]);
    auto Vb = ctx.reg_alloc.ReadX(args[2]);
    RegAlloc::Realize(Vresult, Vge, Va, Vb);

    code.and_(Vb, Vb, Vge);
    code.andn(Va, Vge, Va);
    code.or_(Vb, Vb, Va);
//    code.FMOV(Vresult, Vge);  // TODO: Move elimination
//    code.BSL(Vresult->B8(), Vb->B8(), Va->B8());
}

}  // namespace Dynarmic::Backend::LoongArch64
