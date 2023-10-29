/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <mcl/mp/metavalue/lift_value.hpp>

#include "dynarmic/backend/loongarch64/a32_jitstate.h"
#include "dynarmic/backend/loongarch64/abi.h"
#include "dynarmic/backend/loongarch64/emit_loongarch64.h"
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
    static void Emit(BlockOfCode &, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qresult = ctx.reg_alloc.WriteQ(inst);
        auto Qa = ctx.reg_alloc.ReadQ(args[0]);
        auto Qb = ctx.reg_alloc.ReadQ(args[1]);
        RegAlloc::Realize(Qresult, Qa, Qb);
        ctx.fpsr.Load();

        emit(*Qresult, *Qa, *Qb);
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedAdd8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        // FIX this all
        Emit<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vsadd_b(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedAdd16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        Emit<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vsadd_h(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedAdd32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        Emit<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vsadd_w(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedAdd64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        Emit<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vsadd_d(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedSub8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        Emit<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            // FIXME use which sch reg?
//        code.vxor_v(code.vr6, code.vr6, code.vr6);
//        code.vorn_v(Vb, Vb, code.vr6);
//        code.addi_w(Wscratch0, code.zero, 1);
//        code.vreplgr2vr_b(code.vr6, Wscratch0);
//        code.vadd_b(Vb, Vb,code.vr6);
//        code.vadd_b(Vresult, Va, Vb);
//        code.SQSUB(Vresult, Va, Vb);
            code.vssub_b(Vresult, Va, Vb);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedSub16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        Emit<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            // FIXME use which sch reg?
//        code.vxor_v(code.vr6, code.vr6, code.vr6);
//        code.vorn_v(Vb, Vb, code.vr6);
//        code.addi_w(Wscratch0, code.zero, 1);
//        code.vreplgr2vr_b(code.vr6, Wscratch0);
//        code.vadd_h(Vb, Vb,code.vr6);
//        code.vadd_h(Vresult, Va, Vb);
//        code.SQSUB(Vresult, Va, Vb);
            code.vssub_h(Vresult, Va, Vb);

        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedSub32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        Emit<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            // FIXME use which sch reg?
//        code.vxor_v(code.vr6, code.vr6, code.vr6);
//        code.vorn_v(Vb, Vb, code.vr6);
//        code.addi_w(Wscratch0, code.zero, 1);
//        code.vreplgr2vr_b(code.vr6, Wscratch0);
//        code.vadd_w(Vb, Vb,code.vr6);
//        code.vadd_w(Vresult, Va, Vb);
//        code.SQSUB(Vresult, Va, Vb);
            code.vssub_w(Vresult, Va, Vb);

        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedSub64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        Emit<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            // FIXME use which sch reg?
//        code.vxor_v(code.vr6, code.vr6, code.vr6);
//        code.vorn_v(Vb, Vb, code.vr6);
//        code.addi_w(Wscratch0, code.zero, 1);
//        code.vreplgr2vr_b(code.vr6, Wscratch0);
//        code.vadd_d(Vb, Vb,code.vr6);
//        code.vadd_d(Vresult, Va, Vb);
//        code.SQSUB(Vresult, Va, Vb);
            code.vssub_d(Vresult, Va, Vb);

        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorUnsignedSaturatedAdd8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        Emit<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vsadd_bu(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorUnsignedSaturatedAdd16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        Emit<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vsadd_hu(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorUnsignedSaturatedAdd32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        Emit<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vsadd_wu(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorUnsignedSaturatedAdd64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        Emit<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vsadd_du(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorUnsignedSaturatedSub8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        Emit<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            // FIXME use which scr reg?
//        code.vxor_v(code.vr6, code.vr6, code.vr6);
//        code.vorn_v(Vb, Vb, code.vr6);
//        code.addi_w(Wscratch0, code.zero, 1);
//        code.vreplgr2vr_b(code.vr6, Wscratch0);
//        code.vadd_b(Vb, Vb,code.vr6);
//        code.vadd_b(Vresult, Va, Vb);
//        code.UQSUB(Vresult, Va, Vb);
            code.vssub_bu(Vresult, Va, Vb);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorUnsignedSaturatedSub16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        Emit<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            // FIXME use which sch reg?
//        code.vxor_v(code.vr6, code.vr6, code.vr6);
//        code.vorn_v(Vb, Vb, code.vr6);
//        code.addi_w(Wscratch0, code.zero, 1);
//        code.vreplgr2vr_b(code.vr6, Wscratch0);
//        code.vadd_h(Vb, Vb,code.vr6);
//        code.vadd_h(Vresult, Va, Vb);
//        code.UQSUB(Vresult, Va, Vb);
            code.vssub_hu(Vresult, Va, Vb);

        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorUnsignedSaturatedSub32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        Emit<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            // FIXME use which sch reg?
//        code.vxor_v(code.vr6, code.vr6, code.vr6);
//        code.vorn_v(Vb, Vb, code.vr6);
//        code.addi_w(Wscratch0, code.zero, 1);
//        code.vreplgr2vr_b(code.vr6, Wscratch0);
//        code.vadd_w(Vb, Vb,code.vr6);
//        code.vadd_w(Vresult, Va, Vb);
//        code.UQSUB(Vresult, Va, Vb);
            code.vssub_wu(Vresult, Va, Vb);

        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorUnsignedSaturatedSub64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        Emit<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            // FIXME use which sch reg?
//        code.vxor_v(code.vr6, code.vr6, code.vr6);
//        code.vorn_v(Vb, Vb, code.vr6);
//        code.addi_w(Wscratch0, code.zero, 1);
//        code.vreplgr2vr_b(code.vr6, Wscratch0);
//        code.vadd_d(Vb, Vb,code.vr6);
//        code.vadd_d(Vresult, Va, Vb);
//        code.UQSUB(Vresult, Va, Vb);
            code.vssub_du(Vresult, Va, Vb);

        });
    }

}  // namespace Dynarmic::Backend::LoongArch64
