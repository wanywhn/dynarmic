/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/loongarch64/a64_address_space.h"

#include "dynarmic/backend/loongarch64/a64_jitstate.h"
#include "dynarmic/backend/loongarch64/abi.h"
#include "dynarmic/backend/loongarch64/devirtualize.h"
#include "dynarmic/backend/loongarch64/emit_arm64.h"
#include "dynarmic/backend/loongarch64/stack_layout.h"
#include "dynarmic/common/cast_util.h"
#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/frontend/A64/translate/a64_translate.h"
#include "dynarmic/interface/A64/config.h"
#include "dynarmic/interface/exclusive_monitor.h"
#include "dynarmic/ir/opt/passes.h"

namespace Dynarmic::Backend::LoongArch64 {

template<auto mfp, typename T>
static void* EmitCallTrampoline(BlockOfCode& code, T* this_) {
    using namespace Xbyak_loongarch64::util;

    const auto info = Devirtualize<mfp>(this_);

    Xbyak_loongarch64::Label l_addr, l_this;

    void* target = code.getCurr<void*>();
    code.LDLableData_d(code.a0, l_this);

    code.LDLableData_d(Xscratch0, l_addr);

    code.jirl(code.zero, Xscratch0, 0);

    code.align(8);
    code.L(l_this);
    code.dx(info.this_ptr);
    code.L(l_addr);
    code.dx(info.fn_ptr);

    return target;
}

template<auto mfp, typename T>
static void* EmitWrappedReadCallTrampoline(BlockOfCode& code, T* this_) {
    using namespace Xbyak_loongarch64::util;

    const auto info = Devirtualize<mfp>(this_);

    Xbyak_loongarch64::Label l_addr, l_this;

    constexpr u64 save_regs = ABI_CALLER_SAVE & ~ToRegList(Xscratch0);

    void *target = code.getCode<void *>();
    ABI_PushRegisters(code, save_regs, 0);
    code.LDLableData_d(code.a0, l_this);
    code.add_d(code.a1, Xscratch0, code.zero);
    code.LDLableData_d(Xscratch0, l_addr);
    code.jirl(code.ra, Xscratch0, 0);
    code.add_d(Xscratch0, code.a0, code.zero);
    ABI_PopRegisters(code, save_regs, 0);
    code.jirl(code.zero, code.ra, 0);

    code.align(8);
    code.L(l_this);
    code.dx(info.this_ptr);
    code.L(l_addr);
    code.dx(info.fn_ptr);

    return target;
}

template<auto callback, typename T>
static void* EmitExclusiveReadCallTrampoline(BlockOfCode& code, const A64::UserConfig& conf) {
    using namespace Xbyak_loongarch64::util;

    Xbyak_loongarch64::Label l_addr, l_this;

    auto fn = [](const A64::UserConfig& conf, A64::VAddr vaddr) -> T {
        return conf.global_monitor->ReadAndMark<T>(conf.processor_id, vaddr, [&]() -> T {
            return (conf.callbacks->*callback)(vaddr);
        });
    };

    void *target = code.getCode<void *>();
    code.LDLableData_d(code.a0, l_this);
    code.LDLableData_d(Xscratch0, l_addr);
    code.jirl(code.zero, Xscratch0, 0);

    code.align(8);
    code.L(l_this);
    code.dx(mcl::bit_cast<u64>(&conf));
    code.L(l_addr);
    code.dx(mcl::bit_cast<u64>(Common::FptrCast(fn)));

    return target;
}

template<auto mfp, typename T>
static void* EmitWrappedWriteCallTrampoline(BlockOfCode& code, T* this_) {
    using namespace Xbyak_loongarch64::util;

    const auto info = Devirtualize<mfp>(this_);

    Xbyak_loongarch64::Label l_addr, l_this;

    constexpr u64 save_regs = ABI_CALLER_SAVE;

    void *target = code.getCode<void *>();
    ABI_PushRegisters(code, save_regs, 0);
    code.LDLableData_d(code.a0, l_this);
    code.add_d(code.a1, Xscratch0, code.zero);
    code.add_d(code.a2, Xscratch1, code.zero);
    code.LDLableData_d(Xscratch0, l_addr);
    code.jirl(code.ra, Xscratch0, 0);
    ABI_PopRegisters(code, save_regs, 0);
    code.jirl(code.zero, code.ra, 0);

    code.align(8);
    code.L(l_this);
    code.dx(info.this_ptr);
    code.L(l_addr);
    code.dx(info.fn_ptr);

    return target;
}

template<auto callback, typename T>
static void* EmitExclusiveWriteCallTrampoline(BlockOfCode& code, const A64::UserConfig& conf) {
    using namespace Xbyak_loongarch64::util;

    Xbyak_loongarch64::Label l_addr, l_this;

    auto fn = [](const A64::UserConfig& conf, A64::VAddr vaddr, T value) -> u32 {
        return conf.global_monitor->DoExclusiveOperation<T>(conf.processor_id, vaddr,
                                                            [&](T expected) -> bool {
                                                                return (conf.callbacks->*callback)(vaddr, value, expected);
                                                            })
                 ? 0
                 : 1;
    };

    void* target = code.getCode<void *>();
    code.LDLableData_d(code.a0, l_this);
    code.LDLableData_d(Xscratch0, l_addr);
    code.jirl(code.zero, Xscratch0, 0);

    code.align(8);
    code.L(l_this);
    code.dx(mcl::bit_cast<u64>(&conf));
    code.L(l_addr);
    code.dx(mcl::bit_cast<u64>(Common::FptrCast(fn)));

    return target;
}

static void* EmitRead128CallTrampoline(BlockOfCode& code, A64::UserCallbacks* this_) {
    using namespace Xbyak_loongarch64::util;

    const auto info = Devirtualize<&A64::UserCallbacks::MemoryRead128>(this_);

    Xbyak_loongarch64::Label l_addr, l_this;
    void * target = code.getCode<void *>();
    ABI_PushRegisters(code, (1ull << 29) | (1ull << 30), 0);
    code.LDLableData_d(code.a0, l_this);
    code.LDLableData_d(Xscratch0, l_addr);
    code.jirl(code.ra, Xscratch0, 0);
    code.vinsgr2vr_d(Vscratch2, code.a0, 0);
    code.vinsgr2vr_d(Vscratch2, code.a1, 1);
    ABI_PopRegisters(code, (1ull << 29) | (1ull << 30), 0);
    code.jirl(code.zero, code.ra, 0);

    code.align(8);
    code.L(l_this);
    code.dx(info.this_ptr);
    code.L(l_addr);
    code.dx(info.fn_ptr);

    return target;
}

static void* EmitWrappedRead128CallTrampoline(BlockOfCode& code, A64::UserCallbacks* this_) {
    using namespace Xbyak_loongarch64::util;

    const auto info = Devirtualize<&A64::UserCallbacks::MemoryRead128>(this_);

    Xbyak_loongarch64::Label l_addr, l_this;

    constexpr u64 save_regs = ABI_CALLER_SAVE & ~ToRegList(Xscratch0);

    void *target = code.getCode<void *>();
    ABI_PushRegisters(code, save_regs, 0);
    code.LDLableData_d(code.a0, l_this);
    code.add_d(code.a1, Xscratch0, code.zero);
    code.LDLableData_d(Xscratch0, l_addr);
    code.jirl(code.ra, Xscratch0, 0);
    code.vinsgr2vr_d(Vscratch2, code.a0, 0);
    code.vinsgr2vr_d(Vscratch2, code.a1, 1);
    ABI_PopRegisters(code, save_regs, 0);
    code.jirl(code.zero, code.ra, 0);

    code.align(8);
    code.L(l_this);
    code.dx(info.this_ptr);
    code.L(l_addr);
    code.dx(info.fn_ptr);

    return target;
}

static void* EmitExclusiveRead128CallTrampoline(BlockOfCode& code, const A64::UserConfig& conf) {
    using namespace Xbyak_loongarch64::util;

    Xbyak_loongarch64::Label l_addr, l_this;

    auto fn = [](const A64::UserConfig& conf, A64::VAddr vaddr) -> Vector {
        return conf.global_monitor->ReadAndMark<Vector>(conf.processor_id, vaddr, [&]() -> Vector {
            return conf.callbacks->MemoryRead128(vaddr);
        });
    };

    void *target = code.getCode<void *>();
    ABI_PushRegisters(code, (1ull << 29) | (1ull << 30), 0);
    code.LDLableData_d(code.a0, l_this);
    code.LDLableData_d(Xscratch0, l_addr);
    code.jirl(code.ra, Xscratch0, 0);
    code.vinsgr2vr_d(Vscratch2, code.a0, 0);
    code.vinsgr2vr_d(Vscratch2, code.a1, 1);
    ABI_PopRegisters(code, (1ull << 29) | (1ull << 30), 0);
    code.jirl(code.zero, code.ra, 0);

    code.align(8);
    code.L(l_this);
    code.dx(mcl::bit_cast<u64>(&conf));
    code.L(l_addr);
    code.dx(mcl::bit_cast<u64>(Common::FptrCast(fn)));

    return target;
}

static void* EmitWrite128CallTrampoline(BlockOfCode& code, A64::UserCallbacks* this_) {
    using namespace Xbyak_loongarch64::util;

    const auto info = Devirtualize<&A64::UserCallbacks::MemoryWrite128>(this_);

    Xbyak_loongarch64::Label l_addr, l_this;

    void *target = code.getCode<void *>();
    code.LDLableData_d(code.a0, l_this);
    code.vpickve2gr_d(code.a2, Vscratch2, 0);
    code.vpickve2gr_d(code.a3, Vscratch2, 1);
    code.LDLableData_d(Xscratch0, l_addr);
    code.jirl(code.zero, Xscratch0, 0);

    code.align(8);
    code.L(l_this);
    code.dx(info.this_ptr);
    code.L(l_addr);
    code.dx(info.fn_ptr);

    return target;
}

static void* EmitWrappedWrite128CallTrampoline(BlockOfCode& code, A64::UserCallbacks* this_) {
    using namespace Xbyak_loongarch64::util;

    const auto info = Devirtualize<&A64::UserCallbacks::MemoryWrite128>(this_);

    Xbyak_loongarch64::Label l_addr, l_this;

    constexpr u64 save_regs = ABI_CALLER_SAVE;

    void *target = code.getCode<void *>();
    ABI_PushRegisters(code, save_regs, 0);
    code.LDLableData_d(code.a0, l_this);
    code.add_d(code.a1, Xscratch0, code.zero);
    code.vpickve2gr_d(code.a2, Vscratch2, 0);
    code.vpickve2gr_d(code.a3, Vscratch2, 1);
    code.LDLableData_d(Xscratch0, l_addr);
    code.jirl(code.ra, Xscratch0, 0);
    ABI_PopRegisters(code, save_regs, 0);
    code.jirl(code.zero, code.ra, 0);

    code.align(8);
    code.L(l_this);
    code.dx(info.this_ptr);
    code.L(l_addr);
    code.dx(info.fn_ptr);

    return target;
}

static void* EmitExclusiveWrite128CallTrampoline(BlockOfCode& code, const A64::UserConfig& conf) {
    using namespace Xbyak_loongarch64::util;

    Xbyak_loongarch64::Label l_addr, l_this;

    auto fn = [](const A64::UserConfig& conf, A64::VAddr vaddr, Vector value) -> u32 {
        return conf.global_monitor->DoExclusiveOperation<Vector>(conf.processor_id, vaddr,
                                                                 [&](Vector expected) -> bool {
                                                                     return conf.callbacks->MemoryWriteExclusive128(vaddr, value, expected);
                                                                 })
                 ? 0
                 : 1;
    };

    void *target = code.getCode<void *>();
    code.LDLableData_d(code.a0, l_this);
    code.vpickve2gr_d(code.a2, Vscratch2, 0);
    code.vpickve2gr_d(code.a3, Vscratch2, 1);
    code.LDLableData_d(Xscratch0, l_addr);
    code.jirl(code.zero, Xscratch0, 0);

    code.align(8);
    code.L(l_this);
    code.dx(mcl::bit_cast<u64>(&conf));
    code.L(l_addr);
    code.dx(mcl::bit_cast<u64>(Common::FptrCast(fn)));

    return target;
}

A64AddressSpace::A64AddressSpace(const A64::UserConfig& conf, JitStateInfo jsi)
        : AddressSpace(conf.code_cache_size, jsi)
        , conf(conf) {
    EmitPrelude();
}

IR::Block A64AddressSpace::GenerateIR(IR::LocationDescriptor descriptor) const {
    const auto get_code = [this](u64 vaddr) { return conf.callbacks->MemoryReadCode(vaddr); };
    IR::Block ir_block = A64::Translate(A64::LocationDescriptor{descriptor}, get_code,
                                        {conf.define_unpredictable_behaviour, conf.wall_clock_cntpct});

    Optimization::A64CallbackConfigPass(ir_block, conf);
    Optimization::NamingPass(ir_block);
    if (conf.HasOptimization(OptimizationFlag::GetSetElimination) && !conf.check_halt_on_memory_access) {
        Optimization::A64GetSetElimination(ir_block);
        Optimization::DeadCodeElimination(ir_block);
    }
    if (conf.HasOptimization(OptimizationFlag::ConstProp)) {
        Optimization::ConstantPropagation(ir_block);
        Optimization::DeadCodeElimination(ir_block);
    }
    if (conf.HasOptimization(OptimizationFlag::MiscIROpt)) {
        Optimization::A64MergeInterpretBlocksPass(ir_block, conf.callbacks);
    }
    Optimization::VerificationPass(ir_block);

    return ir_block;
}

void A64AddressSpace::InvalidateCacheRanges(const boost::icl::interval_set<u64>& ranges) {
    InvalidateBasicBlocks(block_ranges.InvalidateRanges(ranges));
}

void A64AddressSpace::EmitPrelude() {
    using namespace Xbyak_loongarch64::util;

    //    mem.unprotect();

    prelude_info.read_memory_8 = EmitCallTrampoline<&A64::UserCallbacks::MemoryRead8>(code, conf.callbacks);
    prelude_info.read_memory_16 = EmitCallTrampoline<&A64::UserCallbacks::MemoryRead16>(code, conf.callbacks);
    prelude_info.read_memory_32 = EmitCallTrampoline<&A64::UserCallbacks::MemoryRead32>(code, conf.callbacks);
    prelude_info.read_memory_64 = EmitCallTrampoline<&A64::UserCallbacks::MemoryRead64>(code, conf.callbacks);
    prelude_info.read_memory_128 = EmitRead128CallTrampoline(code, conf.callbacks);
    prelude_info.wrapped_read_memory_8 = EmitWrappedReadCallTrampoline<&A64::UserCallbacks::MemoryRead8>(code, conf.callbacks);
    prelude_info.wrapped_read_memory_16 = EmitWrappedReadCallTrampoline<&A64::UserCallbacks::MemoryRead16>(code, conf.callbacks);
    prelude_info.wrapped_read_memory_32 = EmitWrappedReadCallTrampoline<&A64::UserCallbacks::MemoryRead32>(code, conf.callbacks);
    prelude_info.wrapped_read_memory_64 = EmitWrappedReadCallTrampoline<&A64::UserCallbacks::MemoryRead64>(code, conf.callbacks);
    prelude_info.wrapped_read_memory_128 = EmitWrappedRead128CallTrampoline(code, conf.callbacks);
    prelude_info.exclusive_read_memory_8 = EmitExclusiveReadCallTrampoline<&A64::UserCallbacks::MemoryRead8, u8>(code, conf);
    prelude_info.exclusive_read_memory_16 = EmitExclusiveReadCallTrampoline<&A64::UserCallbacks::MemoryRead16, u16>(code, conf);
    prelude_info.exclusive_read_memory_32 = EmitExclusiveReadCallTrampoline<&A64::UserCallbacks::MemoryRead32, u32>(code, conf);
    prelude_info.exclusive_read_memory_64 = EmitExclusiveReadCallTrampoline<&A64::UserCallbacks::MemoryRead64, u64>(code, conf);
    prelude_info.exclusive_read_memory_128 = EmitExclusiveRead128CallTrampoline(code, conf);
    prelude_info.write_memory_8 = EmitCallTrampoline<&A64::UserCallbacks::MemoryWrite8>(code, conf.callbacks);
    prelude_info.write_memory_16 = EmitCallTrampoline<&A64::UserCallbacks::MemoryWrite16>(code, conf.callbacks);
    prelude_info.write_memory_32 = EmitCallTrampoline<&A64::UserCallbacks::MemoryWrite32>(code, conf.callbacks);
    prelude_info.write_memory_64 = EmitCallTrampoline<&A64::UserCallbacks::MemoryWrite64>(code, conf.callbacks);
    prelude_info.write_memory_128 = EmitWrite128CallTrampoline(code, conf.callbacks);
    prelude_info.wrapped_write_memory_8 = EmitWrappedWriteCallTrampoline<&A64::UserCallbacks::MemoryWrite8>(code, conf.callbacks);
    prelude_info.wrapped_write_memory_16 = EmitWrappedWriteCallTrampoline<&A64::UserCallbacks::MemoryWrite16>(code, conf.callbacks);
    prelude_info.wrapped_write_memory_32 = EmitWrappedWriteCallTrampoline<&A64::UserCallbacks::MemoryWrite32>(code, conf.callbacks);
    prelude_info.wrapped_write_memory_64 = EmitWrappedWriteCallTrampoline<&A64::UserCallbacks::MemoryWrite64>(code, conf.callbacks);
    prelude_info.wrapped_write_memory_128 = EmitWrappedWrite128CallTrampoline(code, conf.callbacks);
    prelude_info.exclusive_write_memory_8 = EmitExclusiveWriteCallTrampoline<&A64::UserCallbacks::MemoryWriteExclusive8, u8>(code, conf);
    prelude_info.exclusive_write_memory_16 = EmitExclusiveWriteCallTrampoline<&A64::UserCallbacks::MemoryWriteExclusive16, u16>(code, conf);
    prelude_info.exclusive_write_memory_32 = EmitExclusiveWriteCallTrampoline<&A64::UserCallbacks::MemoryWriteExclusive32, u32>(code, conf);
    prelude_info.exclusive_write_memory_64 = EmitExclusiveWriteCallTrampoline<&A64::UserCallbacks::MemoryWriteExclusive64, u64>(code, conf);
    prelude_info.exclusive_write_memory_128 = EmitExclusiveWrite128CallTrampoline(code, conf);
    prelude_info.call_svc = EmitCallTrampoline<&A64::UserCallbacks::CallSVC>(code, conf.callbacks);
    prelude_info.exception_raised = EmitCallTrampoline<&A64::UserCallbacks::ExceptionRaised>(code, conf.callbacks);
    prelude_info.isb_raised = EmitCallTrampoline<&A64::UserCallbacks::InstructionSynchronizationBarrierRaised>(code, conf.callbacks);
    prelude_info.ic_raised = EmitCallTrampoline<&A64::UserCallbacks::InstructionCacheOperationRaised>(code, conf.callbacks);
    prelude_info.dc_raised = EmitCallTrampoline<&A64::UserCallbacks::DataCacheOperationRaised>(code, conf.callbacks);
    prelude_info.get_cntpct = EmitCallTrampoline<&A64::UserCallbacks::GetCNTPCT>(code, conf.callbacks);
    prelude_info.add_ticks = EmitCallTrampoline<&A64::UserCallbacks::AddTicks>(code, conf.callbacks);
    prelude_info.get_ticks_remaining = EmitCallTrampoline<&A64::UserCallbacks::GetTicksRemaining>(code, conf.callbacks);

    Xbyak_loongarch64::Label return_from_run_code, l_return_to_dispatcher;

    prelude_info.run_code = code.getCurr<PreludeInfo::RunCodeFuncType>();
    {
        // TODO 30?
        ABI_PushRegisters(code, ABI_CALLEE_SAVE | (1 << code.ra.getIdx()), sizeof(StackLayout));

        code.add_d(code.s0, code.a0, code.zero);
        code.add_d(Xstate, code.a1, code.zero);
        code.add_d(Xhalt, code.a2, code.zero);
        if (conf.page_table) {
            code.add_imm(Xpagetable, code.zero, mcl::bit_cast<u64>(conf.page_table), Xscratch0);
        }
        if (conf.fastmem_pointer) {
            code.add_imm(Xfastmem, code.zero, mcl::bit_cast<u64>(conf.fastmem_pointer), Xscratch0);
        }

        if (conf.HasOptimization(OptimizationFlag::ReturnStackBuffer)) {
            code.LDLableData_d(Xscratch0, l_return_to_dispatcher);
            for (size_t i = 0; i < RSBCount; i++) {
                code.st_d(Xscratch0, code.sp, offsetof(StackLayout, rsb) + offsetof(RSBEntry, code_ptr) + i * sizeof(RSBEntry));
            }
        }

        if (conf.enable_cycle_counting) {
            code.BL(prelude_info.get_ticks_remaining);
            code.add_d(Xticks, code.a0, code.zero);
            code.st_d(Xticks, code.sp, offsetof(StackLayout, cycles_to_run));
        }

        SwitchFcsrOnEntry();

        code.ll_w(Wscratch0, Xhalt, 0);
        code.bnez(Wscratch0, return_from_run_code);

        code.jirl(code.zero, code.s0, 0);
    }

    prelude_info.step_code = code.getCurr<PreludeInfo::RunCodeFuncType>();
    {
        ABI_PushRegisters(code, ABI_CALLEE_SAVE | (1 << code.ra.getIdx()), sizeof(StackLayout));

        code.add_d(code.s0, code.a0, code.zero);
        code.add_d(Xstate, code.a1, code.zero);
        code.add_d(Xhalt, code.a2, code.zero);
        if (conf.page_table) {
            code.add_imm(Xpagetable, code.zero, mcl::bit_cast<u64>(conf.page_table), Xscratch0);
        }
        if (conf.fastmem_pointer) {
            code.add_imm(Xfastmem,code.zero, mcl::bit_cast<u64>(conf.fastmem_pointer), Xscratch0);
        }

        if (conf.HasOptimization(OptimizationFlag::ReturnStackBuffer)) {
            code.LDLableData_d(Xscratch0, l_return_to_dispatcher);
            for (size_t i = 0; i < RSBCount; i++) {
                code.st_d(Xscratch0, code.sp, offsetof(StackLayout, rsb) + offsetof(RSBEntry, code_ptr) + i * sizeof(RSBEntry));
            }
        }

        if (conf.enable_cycle_counting) {
            code.addi_d(Xticks, Xticks, 1);
            code.st_d(Xticks, code.sp, offsetof(StackLayout, cycles_to_run));
        }

        SwitchFcsrOnEntry();

        Xbyak_loongarch64::Label step_hr_loop;
        code.L(step_hr_loop);
        code.ll_d(Wscratch0, Xhalt, 0);
        code.bnez(Wscratch0, return_from_run_code);
        code.add_imm(Wscratch1, code.zero, static_cast<u32>(HaltReason::Step), Xscratch2);
        code.or_(Wscratch0, Wscratch0, Wscratch1);
        code.add_d(Wscratch1, code.zero, Wscratch0);
        code.sc_d(Wscratch1, Xhalt, 0);
        code.bnez(Wscratch1, step_hr_loop);

        code.jirl(code.zero, code.s0, 0);
    }

    prelude_info.return_to_dispatcher = code.getCurr<void *>();
    {
        Xbyak_loongarch64::Label l_this, l_addr;

        code.ll_w(Wscratch0, Xhalt, 0);
        code.bnez(Wscratch0, return_from_run_code);

        if (conf.enable_cycle_counting) {
            code.beqz(Xticks, return_from_run_code);
            code.blt(Xticks, code.zero, return_from_run_code);
        }

        code.LDLableData_d(code.a0, l_this);
        code.add_d(code.a1, Xstate, code.zero);
        code.LDLableData_d(Xscratch0, l_addr);
        code.jirl(code.ra, Xscratch0, 0);
        code.jirl(code.zero, code.a0, 0);

        const auto fn = [](A64AddressSpace& self, A64JitState& context) -> CodePtr {
            return self.GetOrEmit(context.GetLocationDescriptor());
        };

        code.align(8);
        code.L(l_this);
        code.dx(mcl::bit_cast<u64>(this));
        code.L(l_addr);
        code.dx(mcl::bit_cast<u64>(Common::FptrCast(fn)));
    }

    prelude_info.return_from_run_code = code.getCurr<void*>();
    {
        code.L(return_from_run_code);

        if (conf.enable_cycle_counting) {
            code.ld_d(code.a1, code.sp, offsetof(StackLayout, cycles_to_run));
            code.sub_d(code.a1, code.a1, Xticks);
            // FIXME is this ok?
            code.BL((prelude_info.add_ticks));
        }

        SwitchFcsrOnExit();

        Xbyak_loongarch64::Label exit_hr_loop;
        code.L(exit_hr_loop);
        code.ll_d(code.a0, Xhalt, 0);
        code.add_d(Wscratch0, code.zero, code.zero);
//        code.dbar(0); FIXME no llacq ,is this ok ?
        code.sc_d(Wscratch0, Xhalt, 0);
        code.beqz(Wscratch0, exit_hr_loop);

        ABI_PopRegisters(code, ABI_CALLEE_SAVE | (1 << code.ra.getIdx()), sizeof(StackLayout));
        code.jirl(code.zero, code.ra, 0);
    }

    code.align(8);
    code.L(l_return_to_dispatcher);
    code.dx(mcl::bit_cast<u64>(prelude_info.return_to_dispatcher));

    prelude_info.end_of_prelude = code.getCurr<u32*>();

    // FIXME how?
    //    mem.invalidate_all();
    //    mem.protect();
}

    void A64AddressSpace::SwitchFcsrOnExit() {
        code.ld_d(Wscratch0, code.sp, offsetof(StackLayout, save_host_fpcr));
        code.movgr2fcsr(code.fcsr0, Wscratch0);
    }

    void A64AddressSpace::SwitchFcsrOnEntry() {
        code.movfcsr2gr(Xscratch1, code.fcsr0);
        code.st_d(Wscratch1, code.sp, offsetof(StackLayout, save_host_fpcr));
        code.ld_d(Wscratch0, Xstate, offsetof(A64JitState, fpcr));
        code.movgr2fcsr(code.fcsr0, Xscratch0);
    }

    EmitConfig A64AddressSpace::GetEmitConfig() {
    return EmitConfig{
        .optimizations = conf.unsafe_optimizations ? conf.optimizations : conf.optimizations & all_safe_optimizations,

        .hook_isb = conf.hook_isb,

        .cntfreq_el0 = conf.cntfrq_el0,
        .ctr_el0 = conf.ctr_el0,
        .dczid_el0 = conf.dczid_el0,
        .tpidrro_el0 = conf.tpidrro_el0,
        .tpidr_el0 = conf.tpidr_el0,

        .check_halt_on_memory_access = conf.check_halt_on_memory_access,

        .page_table_pointer = mcl::bit_cast<u64>(conf.page_table),
        .page_table_address_space_bits = conf.page_table_address_space_bits,
        .page_table_pointer_mask_bits = conf.page_table_pointer_mask_bits,
        .silently_mirror_page_table = conf.silently_mirror_page_table,
        .absolute_offset_page_table = conf.absolute_offset_page_table,
        .detect_misaligned_access_via_page_table = conf.detect_misaligned_access_via_page_table,
        .only_detect_misalignment_via_page_table_on_page_boundary = conf.only_detect_misalignment_via_page_table_on_page_boundary,

        .fastmem_pointer = mcl::bit_cast<u64>(conf.fastmem_pointer),
        .recompile_on_fastmem_failure = conf.recompile_on_fastmem_failure,
        .fastmem_address_space_bits = conf.fastmem_address_space_bits,
        .silently_mirror_fastmem = conf.silently_mirror_fastmem,

        .wall_clock_cntpct = conf.wall_clock_cntpct,
        .enable_cycle_counting = conf.enable_cycle_counting,

        .always_little_endian = true,

        .descriptor_to_fpcr = [](const IR::LocationDescriptor& location) { return A64::LocationDescriptor{location}.FPCR(); },
        .emit_cond = EmitA64Cond,
        .emit_condition_failed_terminal = EmitA64ConditionFailedTerminal,
        .emit_terminal = EmitA64Terminal,
        .emit_check_memory_abort = EmitA64CheckMemoryAbort,

        .state_nzcv_offset = offsetof(A64JitState, cpsr_nzcv),
        .state_fpsr_offset = offsetof(A64JitState, fpcr),
        .state_exclusive_state_offset = offsetof(A64JitState, exclusive_state),

        .coprocessors{},

        .very_verbose_debugging_output = conf.very_verbose_debugging_output,
    };
}

void A64AddressSpace::RegisterNewBasicBlock(const IR::Block& block, const EmittedBlockInfo&) {
    const A64::LocationDescriptor descriptor{block.Location()};
    const A64::LocationDescriptor end_location{block.EndLocation()};
    const auto range = boost::icl::discrete_interval<u64>::closed(descriptor.PC(), end_location.PC() - 1);
    block_ranges.AddRange(range, descriptor);
}

}  // namespace Dynarmic::Backend::LoongArch64
