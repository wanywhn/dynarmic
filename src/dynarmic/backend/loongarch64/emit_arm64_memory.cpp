/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/loongarch64/emit_arm64_memory.h"

#include <optional>
#include <utility>

#include <mcl/bit_cast.hpp>

#include "dynarmic/backend/loongarch64/abi.h"
#include "dynarmic/backend/loongarch64/emit_arm64.h"
#include "dynarmic/backend/loongarch64/emit_context.h"
#include "dynarmic/backend/loongarch64/fastmem.h"
#include "dynarmic/backend/loongarch64/fpsr_manager.h"
#include "dynarmic/backend/loongarch64/reg_alloc.h"
#include "dynarmic/ir/acc_type.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"
#include "xbyak_loongarch64.h"
#include "xbyak_loongarch64_util.h"

namespace Dynarmic::Backend::LoongArch64 {

using namespace Xbyak_loongarch64::util;

namespace {

bool IsOrdered(IR::AccType acctype) {
    return acctype == IR::AccType::ORDERED || acctype == IR::AccType::ORDEREDRW || acctype == IR::AccType::LIMITEDORDERED;
}

LinkTarget ReadMemoryLinkTarget(size_t bitsize) {
    switch (bitsize) {
    case 8:
        return LinkTarget::ReadMemory8;
    case 16:
        return LinkTarget::ReadMemory16;
    case 32:
        return LinkTarget::ReadMemory32;
    case 64:
        return LinkTarget::ReadMemory64;
    case 128:
        return LinkTarget::ReadMemory128;
    }
    UNREACHABLE();
}

LinkTarget WriteMemoryLinkTarget(size_t bitsize) {
    switch (bitsize) {
    case 8:
        return LinkTarget::WriteMemory8;
    case 16:
        return LinkTarget::WriteMemory16;
    case 32:
        return LinkTarget::WriteMemory32;
    case 64:
        return LinkTarget::WriteMemory64;
    case 128:
        return LinkTarget::WriteMemory128;
    }
    UNREACHABLE();
}

LinkTarget WrappedReadMemoryLinkTarget(size_t bitsize) {
    switch (bitsize) {
    case 8:
        return LinkTarget::WrappedReadMemory8;
    case 16:
        return LinkTarget::WrappedReadMemory16;
    case 32:
        return LinkTarget::WrappedReadMemory32;
    case 64:
        return LinkTarget::WrappedReadMemory64;
    case 128:
        return LinkTarget::WrappedReadMemory128;
    }
    UNREACHABLE();
}

LinkTarget WrappedWriteMemoryLinkTarget(size_t bitsize) {
    switch (bitsize) {
    case 8:
        return LinkTarget::WrappedWriteMemory8;
    case 16:
        return LinkTarget::WrappedWriteMemory16;
    case 32:
        return LinkTarget::WrappedWriteMemory32;
    case 64:
        return LinkTarget::WrappedWriteMemory64;
    case 128:
        return LinkTarget::WrappedWriteMemory128;
    }
    UNREACHABLE();
}

LinkTarget ExclusiveReadMemoryLinkTarget(size_t bitsize) {
    switch (bitsize) {
    case 8:
        return LinkTarget::ExclusiveReadMemory8;
    case 16:
        return LinkTarget::ExclusiveReadMemory16;
    case 32:
        return LinkTarget::ExclusiveReadMemory32;
    case 64:
        return LinkTarget::ExclusiveReadMemory64;
    case 128:
        return LinkTarget::ExclusiveReadMemory128;
    }
    UNREACHABLE();
}

LinkTarget ExclusiveWriteMemoryLinkTarget(size_t bitsize) {
    switch (bitsize) {
    case 8:
        return LinkTarget::ExclusiveWriteMemory8;
    case 16:
        return LinkTarget::ExclusiveWriteMemory16;
    case 32:
        return LinkTarget::ExclusiveWriteMemory32;
    case 64:
        return LinkTarget::ExclusiveWriteMemory64;
    case 128:
        return LinkTarget::ExclusiveWriteMemory128;
    }
    UNREACHABLE();
}

template<size_t bitsize>
void CallbackOnlyEmitReadMemory(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.PrepareForCall({}, args[1]);
    const bool ordered = IsOrdered(args[2].GetImmediateAccType());

    EmitRelocation(code, ctx, ReadMemoryLinkTarget(bitsize));
    if (ordered) {
        code.dbar(0);
    }

    if constexpr (bitsize == 128) {
        code.add_d(Q8.B16(), Q0.B16(), code.zero);
        ctx.reg_alloc.DefineAsRegister(inst, Q8);
    } else {
        ctx.reg_alloc.DefineAsRegister(inst, code.a0);
    }
}

template<size_t bitsize>
void CallbackOnlyEmitExclusiveReadMemory(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.PrepareForCall({}, args[1]);
    const bool ordered = IsOrdered(args[2].GetImmediateAccType());

    code.add_d(Wscratch0, 1, code.zero);
    code.st_b(Wscratch0, Xstate, ctx.conf.state_exclusive_state_offset);
    EmitRelocation(code, ctx, ExclusiveReadMemoryLinkTarget(bitsize));
    if (ordered) {
        code.dbar(0);
    }

    if constexpr (bitsize == 128) {
        code.add_d(Q8.B16(), Q0.B16(), code.zero);
        ctx.reg_alloc.DefineAsRegister(inst, Q8);
    } else {
        ctx.reg_alloc.DefineAsRegister(inst, code.a0);
    }
}

template<size_t bitsize>
void CallbackOnlyEmitWriteMemory(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.PrepareForCall({}, args[1], args[2]);
    const bool ordered = IsOrdered(args[3].GetImmediateAccType());

    if (ordered) {
        code.dbar(0);
    }
    EmitRelocation(code, ctx, WriteMemoryLinkTarget(bitsize));
    if (ordered) {
        code.dbar(0);
    }
}

template<size_t bitsize>
void CallbackOnlyEmitExclusiveWriteMemory(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.PrepareForCall({}, args[1], args[2]);
    const bool ordered = IsOrdered(args[3].GetImmediateAccType());

    Xbyak_loongarch64::Label end;

    if (ordered) {
        code.dbar(0x700);
    }
    code.add_d(W0, 1, code.zero);
    code.ld_b(Wscratch0, Xstate, ctx.conf.state_exclusive_state_offset);
    code.beqz(Wscratch0, end);
    code.st_b(code.zero, Xstate, ctx.conf.state_exclusive_state_offset);
    EmitRelocation(code, ctx, ExclusiveWriteMemoryLinkTarget(bitsize));
    if (ordered) {
        code.dbar(0x700);
    }
    code.L(end);
    ctx.reg_alloc.DefineAsRegister(inst, code.a0);
}

constexpr size_t page_bits = 12;
constexpr size_t page_size = 1 << page_bits;
constexpr size_t page_mask = (1 << page_bits) - 1;

// This function may use Xscratch0 as a scratch register
// Trashes NZCV
template<size_t bitsize>
void EmitDetectMisalignedVAddr(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, Xbyak_loongarch64::XReg Xaddr, const SharedLabel& fallback) {
    static_assert(bitsize == 8 || bitsize == 16 || bitsize == 32 || bitsize == 64 || bitsize == 128);

    if (bitsize == 8 || (ctx.conf.detect_misaligned_access_via_page_table & bitsize) == 0) {
        return;
    }

    if (!ctx.conf.only_detect_misalignment_via_page_table_on_page_boundary) {
        const u64 align_mask = []() -> u64 {
            switch (bitsize) {
            case 16:
                return 0b1;
            case 32:
                return 0b11;
            case 64:
                return 0b111;
            case 128:
                return 0b1111;
            default:
                UNREACHABLE();
            }
        }();
        code.addi_d(Xscratch0, code.zero, align_mask);
        code.bne(Xaddr, Xscratch0, *fallback);
    } else {
        // If (addr & page_mask) > page_size - byte_size, use fallback.
        code.andi(Xscratch0, Xaddr, page_mask);
        code.addi_d(Xscratch1, code.zero, page_size - bitsize / 8);
        code.bltu(Xscratch1, Xscratch0, *fallback);
    }
}

// Outputs Xscratch0 = page_table[addr >> page_bits]
// May use Xscratch1 as scratch register
// Address to read/write = [ret0 + ret1], ret0 is always Xscratch0 and ret1 is either Xaddr or Xscratch1
// Trashes NZCV
template<size_t bitsize>
std::pair<Xbyak_loongarch64::XReg, Xbyak_loongarch64::XReg> InlinePageTableEmitVAddrLookup(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, Xbyak_loongarch64::XReg Xaddr, const SharedLabel& fallback) {
    const size_t valid_page_index_bits = ctx.conf.page_table_address_space_bits - page_bits;
    const size_t unused_top_bits = 64 - ctx.conf.page_table_address_space_bits;

    EmitDetectMisalignedVAddr<bitsize>(code, ctx, Xaddr, fallback);

    if (ctx.conf.silently_mirror_page_table || unused_top_bits == 0) {
        code.bstrpick_w(Xscratch0, Xaddr, page_bits + valid_page_index_bits -1, page_bits);
    } else {
        code.srli_d(Xscratch0, Xaddr, page_bits);
        code.addi_d(Xscratch1, code.zero, u64(~u64(0)) << valid_page_index_bits);
        code.bne(Xscratch1, Xscratch0, *fallback);
    }

    code.ld_d(Xscratch0, Xpagetable, Xscratch0, LSL, 3);

    if (ctx.conf.page_table_pointer_mask_bits != 0) {
        const u64 mask = u64(~u64(0)) << ctx.conf.page_table_pointer_mask_bits;
        code.andi(Xscratch0, Xscratch0, mask);
    }

    code.beqz(Xscratch0, *fallback);

    if (ctx.conf.absolute_offset_page_table) {
        return std::make_pair(Xscratch0, Xaddr);
    }
    code.andi(Xscratch1, Xaddr, page_mask);
    return std::make_pair(Xscratch0, Xscratch1);
}

template<std::size_t bitsize>
CodePtr EmitMemoryLdr(Xbyak_loongarch64::CodeGenerator& code, int value_idx, Xbyak_loongarch64::XReg Xbase, Xbyak_loongarch64::XReg Xoffset, bool ordered, bool extend32 = false) {
    const auto index_ext = extend32 ? Xbyak_loongarch64::IndexExt::UXTW : Xbyak_loongarch64::IndexExt::LSL;
    const auto add_ext = extend32 ? Xbyak_loongarch64::AddSubExt::UXTW : Xbyak_loongarch64::AddSubExt::LSL;
    const auto Roffset = extend32 ? Xbyak_loongarch64::RReg{Xoffset.toW()} : Xbyak_loongarch64::RReg{Xoffset};

    CodePtr fastmem_location = code.getCurr<CodePtr>();

    if (ordered) {
        code.ADD(Xscratch0, Xbase, Roffset, add_ext);

        fastmem_location = code.getCurr<CodePtr>();

        switch (bitsize) {
        case 8:
            code.LDARB(Xbyak_loongarch64::WReg{value_idx}, Xscratch0);
            break;
        case 16:
            code.LDARH(Xbyak_loongarch64::WReg{value_idx}, Xscratch0);
            break;
        case 32:
            code.ll_acq_w(Xbyak_loongarch64::WReg{value_idx}, Xscratch0);
            break;
        case 64:
            code.ll_acq_w(Xbyak_loongarch64::XReg{value_idx}, Xscratch0);
            break;
        case 128:
            code.ld_d(Xbyak_loongarch64::VReg{value_idx}, Xscratch0, 0);
            code.dbar(0x700);
            break;
        default:
            ASSERT_FALSE("Invalid bitsize");
        }
    } else {
        fastmem_location = code.getCurr<CodePtr>();

        switch (bitsize) {
        case 8:
            code.LDRB(Xbyak_loongarch64::WReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        case 16:
            code.LDRH(Xbyak_loongarch64::WReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        case 32:
            code.ld_d(Xbyak_loongarch64::WReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        case 64:
            code.ld_d(Xbyak_loongarch64::XReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        case 128:
            code.ld_d(Xbyak_loongarch64::VReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        default:
            ASSERT_FALSE("Invalid bitsize");
        }
    }

    return fastmem_location;
}

template<std::size_t bitsize>
CodePtr EmitMemoryStr(Xbyak_loongarch64::CodeGenerator& code, int value_idx, Xbyak_loongarch64::XReg Xbase, Xbyak_loongarch64::XReg Xoffset, bool ordered, bool extend32 = false) {
    const auto index_ext = extend32 ? Xbyak_loongarch64::IndexExt::UXTW : Xbyak_loongarch64::IndexExt::LSL;
    const auto add_ext = extend32 ? Xbyak_loongarch64::AddSubExt::UXTW : Xbyak_loongarch64::AddSubExt::LSL;
    const auto Roffset = extend32 ? Xbyak_loongarch64::RReg{Xoffset.toW()} : Xbyak_loongarch64::RReg{Xoffset};

    CodePtr fastmem_location;

    if (ordered) {
        code.ADD(Xscratch0, Xbase, Roffset, add_ext);

        fastmem_location = code.getCurr<CodePtr>();

        switch (bitsize) {
        case 8:
            code.STLRB(Xbyak_loongarch64::WReg{value_idx}, Xscratch0);
            break;
        case 16:
            code.STLRH(Xbyak_loongarch64::WReg{value_idx}, Xscratch0);
            break;
        case 32:
            code.sc_rel_w(Xbyak_loongarch64::WReg{value_idx}, Xscratch0);
            break;
        case 64:
            code.sc_rel_w(Xbyak_loongarch64::XReg{value_idx}, Xscratch0);
            break;
        case 128:
            code.dbar(0x700);
            code.stx_d(Xbyak_loongarch64::VReg{value_idx}, Xscratch0, code.zero);
            code.dbar(0x700);
            break;
        default:
            ASSERT_FALSE("Invalid bitsize");
        }
    } else {
        fastmem_location = code.getCurr<CodePtr>();

        switch (bitsize) {
        case 8:
            code.st_b(Xbyak_loongarch64::WReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        case 16:
            code.STRH(Xbyak_loongarch64::WReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        case 32:
            code.STR(Xbyak_loongarch64::WReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        case 64:
            code.STR(Xbyak_loongarch64::XReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        case 128:
            code.STR(Xbyak_loongarch64::VReg{value_idx}, Xbase, Roffset, index_ext);
            break;
        default:
            ASSERT_FALSE("Invalid bitsize");
        }
    }

    return fastmem_location;
}

template<size_t bitsize>
void InlinePageTableEmitReadMemory(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xaddr = ctx.reg_alloc.ReadX(args[1]);
    auto Rvalue = [&] {
        if constexpr (bitsize == 128) {
            return ctx.reg_alloc.WriteQ(inst);
        } else {
            return ctx.reg_alloc.WriteReg<std::max<std::size_t>(bitsize, 32)>(inst);
        }
    }();
    const bool ordered = IsOrdered(args[2].GetImmediateAccType());
    ctx.fpsr.Spill();
    ctx.reg_alloc.SpillFlags();
    RegAlloc::Realize(Xaddr, Rvalue);

    SharedLabel fallback = GenSharedLabel(), end = GenSharedLabel();

    const auto [Xbase, Xoffset] = InlinePageTableEmitVAddrLookup<bitsize>(code, ctx, Xaddr, fallback);
    EmitMemoryLdr<bitsize>(code, Rvalue->getIdx(), Xbase, Xoffset, ordered);

    ctx.deferred_emits.emplace_back([&code, &ctx, inst, Xaddr = *Xaddr, Rvalue = *Rvalue, ordered, fallback, end] {
        code.L(*fallback);
        code.add_d(Xscratch0, Xaddr, code.zero);
        EmitRelocation(code, ctx, WrappedReadMemoryLinkTarget(bitsize));
        if (ordered) {
            code.dbar(0x700);
        }
        if constexpr (bitsize == 128) {
            code.add_d(Rvalue.B16(), Q0.B16(), code.zero);
        } else {
            code.add_d(Rvalue.toX(), Xscratch0, code.zero);
        }
        ctx.conf.emit_check_memory_abort(code, ctx, inst, *end);
        code.b(*end);
    });

    code.L(*end);
}

template<size_t bitsize>
void InlinePageTableEmitWriteMemory(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xaddr = ctx.reg_alloc.ReadX(args[1]);
    auto Rvalue = [&] {
        if constexpr (bitsize == 128) {
            return ctx.reg_alloc.ReadQ(args[2]);
        } else {
            return ctx.reg_alloc.ReadReg<std::max<std::size_t>(bitsize, 32)>(args[2]);
        }
    }();
    const bool ordered = IsOrdered(args[3].GetImmediateAccType());
    ctx.fpsr.Spill();
    ctx.reg_alloc.SpillFlags();
    RegAlloc::Realize(Xaddr, Rvalue);

    SharedLabel fallback = GenSharedLabel(), end = GenSharedLabel();

    const auto [Xbase, Xoffset] = InlinePageTableEmitVAddrLookup<bitsize>(code, ctx, Xaddr, fallback);
    EmitMemoryStr<bitsize>(code, Rvalue->getIdx(), Xbase, Xoffset, ordered);

    ctx.deferred_emits.emplace_back([&code, &ctx, inst, Xaddr = *Xaddr, Rvalue = *Rvalue, ordered, fallback, end] {
        code.L(*fallback);
        if constexpr (bitsize == 128) {
            code.add_d(Xscratch0, Xaddr, code.zero);
            code.add_d(Q0.B16(), Rvalue.B16(), code.zero);
        } else {
            code.add_d(Xscratch0, Xaddr, code.zero);
            code.add_d(Xscratch1, Rvalue.toX(), code.zero);
        }
        if (ordered) {
            code.dbar(0x700);
        }
        EmitRelocation(code, ctx, WrappedWriteMemoryLinkTarget(bitsize));
        if (ordered) {
            code.dbar(0x700);
        }
        ctx.conf.emit_check_memory_abort(code, ctx, inst, *end);
        code.b(*end);
    });

    code.L(*end);
}

std::optional<DoNotFastmemMarker> ShouldFastmem(EmitContext& ctx, IR::Inst* inst) {
    if (!ctx.conf.fastmem_pointer || !ctx.fastmem.SupportsFastmem()) {
        return std::nullopt;
    }

    const auto marker = std::make_tuple(ctx.block.Location(), inst->GetName());
    if (ctx.fastmem.ShouldFastmem(marker)) {
        return marker;
    }
    return std::nullopt;
}

inline bool ShouldExt32(EmitContext& ctx) {
    return ctx.conf.fastmem_address_space_bits == 32 && ctx.conf.silently_mirror_fastmem;
}

// May use Xscratch0 as scratch register
// Address to read/write = [ret0 + ret1], ret0 is always Xfastmem and ret1 is either Xaddr or Xscratch0
// Trashes NZCV
template<size_t bitsize>
std::pair<Xbyak_loongarch64::XReg, Xbyak_loongarch64::XReg> FastmemEmitVAddrLookup(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, Xbyak_loongarch64::XReg Xaddr, const SharedLabel& fallback) {
    if (ctx.conf.fastmem_address_space_bits == 64 || ShouldExt32(ctx)) {
        return std::make_pair(Xfastmem, Xaddr);
    }

    if (ctx.conf.silently_mirror_fastmem) {
        code.bstrpick_w(Xscratch0, Xaddr, ctx.conf.fastmem_address_space_bits - 1, 0);
        return std::make_pair(Xfastmem, Xscratch0);
    }

    code.srli_d(Xscratch0, Xaddr, ctx.conf.fastmem_address_space_bits);
    code.bnez(Xscratch0, *fallback);
    return std::make_pair(Xfastmem, Xaddr);
}

template<size_t bitsize>
void FastmemEmitReadMemory(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, DoNotFastmemMarker marker) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xaddr = ctx.reg_alloc.ReadX(args[1]);
    auto Rvalue = [&] {
        if constexpr (bitsize == 128) {
            return ctx.reg_alloc.WriteQ(inst);
        } else {
            return ctx.reg_alloc.WriteReg<std::max<std::size_t>(bitsize, 32)>(inst);
        }
    }();
    const bool ordered = IsOrdered(args[2].GetImmediateAccType());
    ctx.fpsr.Spill();
    ctx.reg_alloc.SpillFlags();
    RegAlloc::Realize(Xaddr, Rvalue);

    SharedLabel fallback = GenSharedLabel(), end = GenSharedLabel();

    const auto [Xbase, Xoffset] = FastmemEmitVAddrLookup<bitsize>(code, ctx, Xaddr, fallback);
    const auto fastmem_location = EmitMemoryLdr<bitsize>(code, Rvalue->getIdx(), Xbase, Xoffset, ordered, ShouldExt32(ctx));

    ctx.deferred_emits.emplace_back([&code, &ctx, inst, marker, Xaddr = *Xaddr, Rvalue = *Rvalue, ordered, fallback, end, fastmem_location] {
        ctx.ebi.fastmem_patch_info.emplace(
            fastmem_location - ctx.ebi.entry_point,
            FastmemPatchInfo{
                .marker = marker,
                .fc = FakeCall{
                    .call_pc = mcl::bit_cast<u64>(code.getCurr<void (*)()>()),
                },
                .recompile = ctx.conf.recompile_on_fastmem_failure,
            });

        code.L(*fallback);
        code.add_d(Xscratch0, Xaddr, code.zero);
        EmitRelocation(code, ctx, WrappedReadMemoryLinkTarget(bitsize));
        if (ordered) {
            code.dbar(0x700);
        }
        if constexpr (bitsize == 128) {
            code.add_d(Rvalue.B16(), Q0.B16(), code.zero);
        } else {
            code.add_d(Rvalue.toX(), Xscratch0, code.zero);
        }
        ctx.conf.emit_check_memory_abort(code, ctx, inst, *end);
        code.b(*end);
    });

    code.L(*end);
}

template<size_t bitsize>
void FastmemEmitWriteMemory(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, DoNotFastmemMarker marker) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xaddr = ctx.reg_alloc.ReadX(args[1]);
    auto Rvalue = [&] {
        if constexpr (bitsize == 128) {
            return ctx.reg_alloc.ReadQ(args[2]);
        } else {
            return ctx.reg_alloc.ReadReg<std::max<std::size_t>(bitsize, 32)>(args[2]);
        }
    }();
    const bool ordered = IsOrdered(args[3].GetImmediateAccType());
    ctx.fpsr.Spill();
    ctx.reg_alloc.SpillFlags();
    RegAlloc::Realize(Xaddr, Rvalue);

    SharedLabel fallback = GenSharedLabel(), end = GenSharedLabel();

    const auto [Xbase, Xoffset] = FastmemEmitVAddrLookup<bitsize>(code, ctx, Xaddr, fallback);
    const auto fastmem_location = EmitMemoryStr<bitsize>(code, Rvalue->getIdx(), Xbase, Xoffset, ordered, ShouldExt32(ctx));

    ctx.deferred_emits.emplace_back([&code, &ctx, inst, marker, Xaddr = *Xaddr, Rvalue = *Rvalue, ordered, fallback, end, fastmem_location] {
        ctx.ebi.fastmem_patch_info.emplace(
            fastmem_location - ctx.ebi.entry_point,
            FastmemPatchInfo{
                .marker = marker,
                .fc = FakeCall{
                    .call_pc = mcl::bit_cast<u64>(code.getCurr<void (*)()>()),
                },
                .recompile = ctx.conf.recompile_on_fastmem_failure,
            });

        code.L(*fallback);
        if constexpr (bitsize == 128) {
            code.add_d(Xscratch0, Xaddr, code.zero);
            code.add_d(Q0.B16(), Rvalue.B16(), code.zero);
        } else {
            code.add_d(Xscratch0, Xaddr, code.zero);
            code.add_d(Xscratch1, Rvalue.toX(), code.zero);
        }
        if (ordered) {
            code.dbar(0x700);
        }
        EmitRelocation(code, ctx, WrappedWriteMemoryLinkTarget(bitsize));
        if (ordered) {
            code.dbar(0x700);
        }
        ctx.conf.emit_check_memory_abort(code, ctx, inst, *end);
        code.b(*end);
    });

    code.L(*end);
}

}  // namespace

template<size_t bitsize>
void EmitReadMemory(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if (const auto marker = ShouldFastmem(ctx, inst)) {
        FastmemEmitReadMemory<bitsize>(code, ctx, inst, *marker);
    } else if (ctx.conf.page_table_pointer != 0) {
        InlinePageTableEmitReadMemory<bitsize>(code, ctx, inst);
    } else {
        CallbackOnlyEmitReadMemory<bitsize>(code, ctx, inst);
    }
}

template<size_t bitsize>
void EmitExclusiveReadMemory(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    CallbackOnlyEmitExclusiveReadMemory<bitsize>(code, ctx, inst);
}

template<size_t bitsize>
void EmitWriteMemory(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if (const auto marker = ShouldFastmem(ctx, inst)) {
        FastmemEmitWriteMemory<bitsize>(code, ctx, inst, *marker);
    } else if (ctx.conf.page_table_pointer != 0) {
        InlinePageTableEmitWriteMemory<bitsize>(code, ctx, inst);
    } else {
        CallbackOnlyEmitWriteMemory<bitsize>(code, ctx, inst);
    }
}

template<size_t bitsize>
void EmitExclusiveWriteMemory(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    CallbackOnlyEmitExclusiveWriteMemory<bitsize>(code, ctx, inst);
}

template void EmitReadMemory<8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitReadMemory<16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitReadMemory<32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitReadMemory<64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitReadMemory<128>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveReadMemory<8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveReadMemory<16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveReadMemory<32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveReadMemory<64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveReadMemory<128>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitWriteMemory<8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitWriteMemory<16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitWriteMemory<32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitWriteMemory<64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitWriteMemory<128>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveWriteMemory<8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveWriteMemory<16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveWriteMemory<32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveWriteMemory<64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveWriteMemory<128>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);

}  // namespace Dynarmic::Backend::LoongArch64
