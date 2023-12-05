/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <mcl/mp/metavalue/lift_value.hpp>

#include "dynarmic/backend/loongarch64/a32_jitstate.h"
#include "dynarmic/backend/loongarch64/abi.h"
#include "dynarmic/backend/loongarch64/emit_arm64.h"
#include "dynarmic/backend/loongarch64/emit_context.h"
#include "dynarmic/backend/loongarch64/fpsr_manager.h"
#include "dynarmic/backend/loongarch64/reg_alloc.h"
#include "dynarmic/common/always_false.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"
#include "xbyak_loongarch64.h"
#include "xbyak_loongarch64_util.h"

namespace Dynarmic::Backend::LoongArch64 {

using namespace Xbyak_loongarch64::util;

template<size_t size, typename EmitFn>
static void Emit(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, EmitFn emit) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Qresult = ctx.reg_alloc.WriteQ(inst);
    auto Qa = ctx.reg_alloc.ReadQ(args[0]);
    auto Qb = ctx.reg_alloc.ReadQ(args[1]);
    RegAlloc::Realize(Qresult, Qa, Qb);
    ctx.fpsr.Load();

    emit(*Qresult, *Qa, *Qb);
}

template<>
void EmitIR<IR::Opcode::VectorSignedSaturatedAdd8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vadd_b(Vresult, Va, Vb); });
}

template<>
void EmitIR<IR::Opcode::VectorSignedSaturatedAdd16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vadd_h(Vresult, Va, Vb); });
}

template<>
void EmitIR<IR::Opcode::VectorSignedSaturatedAdd32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vadd_w(Vresult, Va, Vb); });
}

template<>
void EmitIR<IR::Opcode::VectorSignedSaturatedAdd64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vadd_w(Vresult, Va, Vb); });
}

template<>
void EmitIR<IR::Opcode::VectorSignedSaturatedSub8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
        // FIXME use which sch reg?
        code.vxor_v(code.vr6, code.vr6, code.vr6);
        code.vorn_v(Vb, Vb, code.vr6);
        code.addi_w(Wscratch0, code.zero, 1);
        code.vreplgr2vr_b(code.vr6, Wscratch0);
        code.vadd_b(Vb, Vb,code.vr6);
        code.vadd_b(Vresult, Va, Vb);
//        code.SQSUB(Vresult, Va, Vb);
        });
}

template<>
void EmitIR<IR::Opcode::VectorSignedSaturatedSub16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
        // FIXME use which sch reg?
        code.vxor_v(code.vr6, code.vr6, code.vr6);
        code.vorn_v(Vb, Vb, code.vr6);
        code.addi_w(Wscratch0, code.zero, 1);
        code.vreplgr2vr_b(code.vr6, Wscratch0);
        code.vadd_h(Vb, Vb,code.vr6);
        code.vadd_h(Vresult, Va, Vb);
//        code.SQSUB(Vresult, Va, Vb);
    });
}

template<>
void EmitIR<IR::Opcode::VectorSignedSaturatedSub32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
        // FIXME use which sch reg?
        code.vxor_v(code.vr6, code.vr6, code.vr6);
        code.vorn_v(Vb, Vb, code.vr6);
        code.addi_w(Wscratch0, code.zero, 1);
        code.vreplgr2vr_b(code.vr6, Wscratch0);
        code.vadd_w(Vb, Vb,code.vr6);
        code.vadd_w(Vresult, Va, Vb);
//        code.SQSUB(Vresult, Va, Vb);
    });
}

template<>
void EmitIR<IR::Opcode::VectorSignedSaturatedSub64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
        // FIXME use which sch reg?
        code.vxor_v(code.vr6, code.vr6, code.vr6);
        code.vorn_v(Vb, Vb, code.vr6);
        code.addi_w(Wscratch0, code.zero, 1);
        code.vreplgr2vr_b(code.vr6, Wscratch0);
        code.vadd_d(Vb, Vb,code.vr6);
        code.vadd_d(Vresult, Va, Vb);
//        code.SQSUB(Vresult, Va, Vb);
    });
}

template<>
void EmitIR<IR::Opcode::VectorUnsignedSaturatedAdd8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vadd_b(Vresult, Va, Vb); });
}

template<>
void EmitIR<IR::Opcode::VectorUnsignedSaturatedAdd16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vadd_h(Vresult, Va, Vb); });
}

template<>
void EmitIR<IR::Opcode::VectorUnsignedSaturatedAdd32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vadd_w(Vresult, Va, Vb); });
}

template<>
void EmitIR<IR::Opcode::VectorUnsignedSaturatedAdd64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vadd_d(Vresult, Va, Vb); });
}

template<>
void EmitIR<IR::Opcode::VectorUnsignedSaturatedSub8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
        // FIXME use which sch reg?
        code.vxor_v(code.vr6, code.vr6, code.vr6);
        code.vorn_v(Vb, Vb, code.vr6);
        code.addi_w(Wscratch0, code.zero, 1);
        code.vreplgr2vr_b(code.vr6, Wscratch0);
        code.vadd_b(Vb, Vb,code.vr6);
        code.vadd_b(Vresult, Va, Vb);
//        code.UQSUB(Vresult, Va, Vb);
    });
}

template<>
void EmitIR<IR::Opcode::VectorUnsignedSaturatedSub16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
        // FIXME use which sch reg?
        code.vxor_v(code.vr6, code.vr6, code.vr6);
        code.vorn_v(Vb, Vb, code.vr6);
        code.addi_w(Wscratch0, code.zero, 1);
        code.vreplgr2vr_b(code.vr6, Wscratch0);
        code.vadd_h(Vb, Vb,code.vr6);
        code.vadd_h(Vresult, Va, Vb);
//        code.UQSUB(Vresult, Va, Vb);
    });
}

template<>
void EmitIR<IR::Opcode::VectorUnsignedSaturatedSub32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
        // FIXME use which sch reg?
        code.vxor_v(code.vr6, code.vr6, code.vr6);
        code.vorn_v(Vb, Vb, code.vr6);
        code.addi_w(Wscratch0, code.zero, 1);
        code.vreplgr2vr_b(code.vr6, Wscratch0);
        code.vadd_w(Vb, Vb,code.vr6);
        code.vadd_w(Vresult, Va, Vb);
//        code.UQSUB(Vresult, Va, Vb);
    });
}

template<>
void EmitIR<IR::Opcode::VectorUnsignedSaturatedSub64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
        // FIXME use which sch reg?
        code.vxor_v(code.vr6, code.vr6, code.vr6);
        code.vorn_v(Vb, Vb, code.vr6);
        code.addi_w(Wscratch0, code.zero, 1);
        code.vreplgr2vr_b(code.vr6, Wscratch0);
        code.vadd_d(Vb, Vb,code.vr6);
        code.vadd_d(Vresult, Va, Vb);
//        code.UQSUB(Vresult, Va, Vb);
    });
}

}  // namespace Dynarmic::Backend::LoongArch64
