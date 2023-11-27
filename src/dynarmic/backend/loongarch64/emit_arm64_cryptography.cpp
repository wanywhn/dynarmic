/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

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
#include "dynarmic/common/crypto/aes.h"


namespace Dynarmic::Backend::LoongArch64 {
    namespace AES = Common::Crypto::AES;

    using AESFn = void(AES::State &, const AES::State &);

    using namespace Xbyak_loongarch64::util;

    static void
    EmitAESFunction(RegAlloc::ArgumentInfo args, EmitContext &ctx, BlockOfCode &code, IR::Inst *inst, AESFn fn) {
        constexpr u32 stack_space = static_cast<u32>(sizeof(AES::State)) * 2;
        auto Qinpt = ctx.reg_alloc.WriteQ(inst);
        auto Qresult = ctx.reg_alloc.ReadQ(args[0]);
//        const Xbyak::Xmm input = ctx.reg_alloc.UseXmm(args[0]);
//        const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();
        ctx.reg_alloc.EndOfAllocScope();

        ctx.reg_alloc.HostCall(nullptr);

        ctx.reg_alloc.AllocStackSpace(stack_space + ABI_SHADOW_SPACE);

        code.lea(code.ABI_PARAM1, ptr[rsp + ABI_SHADOW_SPACE]);
        code.lea(code.ABI_PARAM2, ptr[rsp + ABI_SHADOW_SPACE + sizeof(AES::State)]);
        code.movaps(xword[code.ABI_PARAM2], Qinpt);
        code.CallFunction(fn);
        code.movaps(Qresult, xword[rsp + ABI_SHADOW_SPACE]);

        ctx.reg_alloc.ReleaseStackSpace(stack_space + ABI_SHADOW_SPACE);

        ctx.reg_alloc.DefineValue(inst, Qresult);
    }

    template<size_t bitsize, typename EmitFn>
    static void EmitCRC(Xbyak_loongarch64::CodeGenerator &, EmitContext &ctx, IR::Inst *inst, EmitFn emit_fn) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        auto Woutput = ctx.reg_alloc.WriteW(inst);
        auto Winput = ctx.reg_alloc.ReadW(args[0]);
        auto Rdata = ctx.reg_alloc.ReadReg<bitsize>(args[1]);
        RegAlloc::Realize(Woutput, Winput, Rdata);

        emit_fn(Woutput, Winput, Rdata);
    }

    template<>
    void
    EmitIR<IR::Opcode::CRC32Castagnoli8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitCRC<32>(code, ctx, inst,
                    [&](auto &Woutput, auto &Winput, auto &Wdata) { code.crc_w_b_w(Woutput, Wdata, Winput); });
    }

    template<>
    void
    EmitIR<IR::Opcode::CRC32Castagnoli16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitCRC<32>(code, ctx, inst,
                    [&](auto &Woutput, auto &Winput, auto &Wdata) { code.crc_w_h_w(Woutput, Wdata, Winput); });
    }

    template<>
    void
    EmitIR<IR::Opcode::CRC32Castagnoli32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitCRC<32>(code, ctx, inst,
                    [&](auto &Woutput, auto &Winput, auto &Wdata) { code.crc_w_w_w(Woutput, Wdata, Winput); });
    }

    template<>
    void
    EmitIR<IR::Opcode::CRC32Castagnoli64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitCRC<64>(code, ctx, inst,
                    [&](auto &Woutput, auto &Winput, auto &Xdata) { code.crc_w_d_w(Woutput, Xdata, Winput); });
    }

    template<>
    void EmitIR<IR::Opcode::CRC32ISO8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitCRC<32>(code, ctx, inst,
                    [&](auto &Woutput, auto &Winput, auto &Wdata) { code.crcc_w_b_w(Woutput, Wdata, Winput); });
    }

    template<>
    void EmitIR<IR::Opcode::CRC32ISO16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitCRC<32>(code, ctx, inst,
                    [&](auto &Woutput, auto &Winput, auto &Wdata) { code.crcc_w_h_w(Woutput, Wdata, Winput); });
    }

    template<>
    void EmitIR<IR::Opcode::CRC32ISO32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitCRC<32>(code, ctx, inst,
                    [&](auto &Woutput, auto &Winput, auto &Wdata) { code.crcc_w_w_w(Woutput, Wdata, Winput); });
    }

    template<>
    void EmitIR<IR::Opcode::CRC32ISO64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitCRC<64>(code, ctx, inst,
                    [&](auto &Woutput, auto &Winput, auto &Xdata) { code.crcc_w_d_w(Woutput, Xdata, Winput); });
    }

    template<>
    void EmitIR<IR::Opcode::AESDecryptSingleRound>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                   IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        EmitAESFunction(args, ctx, code, inst, AES::DecryptSingleRound);

    }

    template<>
    void EmitIR<IR::Opcode::AESEncryptSingleRound>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                   IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        EmitAESFunction(args, ctx, code, inst, AES::EncryptSingleRound);
    }

    template<>
    void
    EmitIR<IR::Opcode::AESInverseMixColumns>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        EmitAESFunction(args, ctx, code, inst, AES::InverseMixColumns);

    }

    template<>
    void EmitIR<IR::Opcode::AESMixColumns>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        EmitAESFunction(args, ctx, code, inst, AES::MixColumns);

    }

    template<>
    void EmitIR<IR::Opcode::SM4AccessSubstitutionBox>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        ctx.reg_alloc.HostCall(inst, args[0]);
        code.CallFunction(&Common::Crypto::SM4::AccessSubstitutionBox);
        code.movzx(code.ABI_RETURN.cvt32(), code.ABI_RETURN.cvt8());
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
    }

    template<>
    void EmitIR<IR::Opcode::SHA256Hash>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        const bool part1 = args[3].GetImmediateU1();

        if (part1) {
            auto Qx = ctx.reg_alloc.ReadWriteQ(args[0], inst);
            auto Qy = ctx.reg_alloc.ReadQ(args[1]);
            auto Qz = ctx.reg_alloc.ReadQ(args[2]);
            RegAlloc::Realize(Qx, Qy, Qz);

            code.SHA256H(Qx, Qy, Qz->S4());
        } else {
            auto Qx = ctx.reg_alloc.ReadQ(args[0]);
            auto Qy = ctx.reg_alloc.ReadWriteQ(args[1], inst);
            auto Qz = ctx.reg_alloc.ReadQ(args[2]);
            RegAlloc::Realize(Qx, Qy, Qz);

            code.SHA256H2(Qy, Qx, Qz->S4());  // Yes x and y are swapped
        }
    }

    template<>
    void EmitIR<IR::Opcode::SHA256MessageSchedule0>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qa = ctx.reg_alloc.ReadWriteQ(args[0], inst);
        auto Qb = ctx.reg_alloc.ReadQ(args[1]);
        RegAlloc::Realize(Qa, Qb);

        code.SHA256SU0(Qa->S4(), Qb->S4());
    }

    template<>
    void EmitIR<IR::Opcode::SHA256MessageSchedule1>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qa = ctx.reg_alloc.ReadWriteQ(args[0], inst);
        auto Qb = ctx.reg_alloc.ReadQ(args[1]);
        auto Qc = ctx.reg_alloc.ReadQ(args[2]);
        RegAlloc::Realize(Qa, Qb, Qc);

        code.SHA256SU1(Qa->S4(), Qb->S4(), Qc->S4());
    }

}  // namespace Dynarmic::Backend::LoongArch64
