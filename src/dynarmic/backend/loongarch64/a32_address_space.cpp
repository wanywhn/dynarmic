/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/loongarch64/a32_address_space.h"

#include <cstdint>

#include "dynarmic/backend/loongarch64/a32_jitstate.h"
#include "dynarmic/backend/loongarch64/abi.h"
#include "dynarmic/backend/loongarch64/devirtualize.h"
#include "dynarmic/backend/loongarch64/emit_arm64.h"
#include "dynarmic/backend/loongarch64/stack_layout.h"
#include "dynarmic/common/cast_util.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/frontend/A32/a32_location_descriptor.h"
#include "dynarmic/frontend/A32/translate/a32_translate.h"
#include "dynarmic/interface/A32/config.h"
#include "dynarmic/interface/exclusive_monitor.h"
#include "dynarmic/ir/opt/passes.h"
#include "xbyak_loongarch64.h"
#include "xbyak_loongarch64_util.h"

namespace Dynarmic::Backend::LoongArch64 {

template<auto mfp, typename T>
static void* EmitCallTrampoline(Xbyak_loongarch64::CodeGenerator& code, T* this_) {
    using namespace Xbyak_loongarch64::util;

    const auto info = Devirtualize<mfp>(this_);

    Xbyak_loongarch64::Label l_addr, l_this;

    void* target = (void*)code.getCurr();
    code.pcaddi(code.a0, l_this);
    code.pcaddi(Xscratch0, l_addr);
    code.jirl(code.zero, Xscratch0, 0);

    code.align(8);
    code.L(l_this);
    code.dx(info.this_ptr);
    code.L(l_addr);
    code.dx(info.fn_ptr);

    return target;
}

template<auto mfp, typename T>
static void* EmitWrappedReadCallTrampoline(Xbyak_loongarch64::CodeGenerator& code, T* this_) {
    using namespace Xbyak_loongarch64::util;

    const auto info = Devirtualize<mfp>(this_);

    Xbyak_loongarch64::Label l_addr, l_this;
    // ! FIXME where Xscratch0 value from ?
    constexpr u64 save_regs = ABI_CALLER_SAVE & ~ToRegList(Xscratch0);

    void* target = (void*)code.getCurr();
    ABI_PushRegisters(code, save_regs, 0);
    code.pcaddi(code.a0, l_this);
    code.add_d(code.a1, Xscratch0, code.zero);
    code.pcaddi(Xscratch0, l_addr);
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
static void* EmitExclusiveReadCallTrampoline(Xbyak_loongarch64::CodeGenerator& code, const A32::UserConfig& conf) {
    using namespace Xbyak_loongarch64::util;

    Xbyak_loongarch64::Label l_addr, l_this;

    auto fn = [](const A32::UserConfig& conf, A32::VAddr vaddr) -> T {
        return conf.global_monitor->ReadAndMark<T>(conf.processor_id, vaddr, [&]() -> T {
            return (conf.callbacks->*callback)(vaddr);
        });
    };

    void* target = (void*)code.getCurr();
    code.pcaddi(code.a0, l_this);
    code.pcaddi(Xscratch0, l_addr);
    code.jirl(code.zero, Xscratch0, 0);

    code.align(8);
    code.L(l_this);
    code.dx(mcl::bit_cast<u64>(&conf));
    code.L(l_addr);
    code.dx(mcl::bit_cast<u64>(Common::FptrCast(fn)));

    return target;
}

template<auto mfp, typename T>
static void* EmitWrappedWriteCallTrampoline(Xbyak_loongarch64::CodeGenerator& code, T* this_) {
    using namespace Xbyak_loongarch64::util;

    const auto info = Devirtualize<mfp>(this_);

    Xbyak_loongarch64::Label l_addr, l_this;

    constexpr u64 save_regs = ABI_CALLER_SAVE;

    void* target = (void*)code.getCurr();
    ABI_PushRegisters(code, save_regs, 0);
    code.pcaddi(code.a0, l_this);
    code.add_d(code.a1, Xscratch0, code.zero);
    code.add_d(code.a2, Xscratch1, code.zero);
    code.pcaddi(Xscratch0, l_addr);
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
static void* EmitExclusiveWriteCallTrampoline(Xbyak_loongarch64::CodeGenerator& code, const A32::UserConfig& conf) {
    using namespace Xbyak_loongarch64::util;

    Xbyak_loongarch64::Label l_addr, l_this;

    auto fn = [](const A32::UserConfig& conf, A32::VAddr vaddr, T value) -> u32 {
        return conf.global_monitor->DoExclusiveOperation<T>(conf.processor_id, vaddr,
                                                            [&](T expected) -> bool {
                                                                return (conf.callbacks->*callback)(vaddr, value, expected);
                                                            })
                 ? 0
                 : 1;
    };

    void* target = (void*)code.getCurr();
    code.pcaddi(code.a0, l_this);
    code.pcaddi(Xscratch0, l_addr);
    code.jirl(code.zero, Xscratch0, 0);

    code.align(8);
    code.L(l_this);
    code.dx(mcl::bit_cast<u64>(&conf));
    code.L(l_addr);
    code.dx(mcl::bit_cast<u64>(Common::FptrCast(fn)));

    return target;
}

A32AddressSpace::A32AddressSpace(const A32::UserConfig& conf)
        : AddressSpace(conf.code_cache_size)
        , conf(conf) {
    EmitPrelude();
}

IR::Block A32AddressSpace::GenerateIR(IR::LocationDescriptor descriptor) const {
    IR::Block ir_block = A32::Translate(A32::LocationDescriptor{descriptor}, conf.callbacks, {conf.arch_version, conf.define_unpredictable_behaviour, conf.hook_hint_instructions});

    Optimization::PolyfillPass(ir_block, {});
    Optimization::NamingPass(ir_block);
    if (conf.HasOptimization(OptimizationFlag::GetSetElimination)) {
        Optimization::A32GetSetElimination(ir_block, {.convert_nzc_to_nz = true});
        Optimization::DeadCodeElimination(ir_block);
    }
    if (conf.HasOptimization(OptimizationFlag::ConstProp)) {
        Optimization::A32ConstantMemoryReads(ir_block, conf.callbacks);
        Optimization::ConstantPropagation(ir_block);
        Optimization::DeadCodeElimination(ir_block);
    }
    Optimization::IdentityRemovalPass(ir_block);
    Optimization::VerificationPass(ir_block);

    return ir_block;
}

void A32AddressSpace::InvalidateCacheRanges(const boost::icl::interval_set<u32>& ranges) {
    InvalidateBasicBlocks(block_ranges.InvalidateRanges(ranges));
}

void A32AddressSpace::EmitPrelude() {
    using namespace Xbyak_loongarch64::util;

    //    mem.unprotect();

    prelude_info.read_memory_8 = EmitCallTrampoline<&A32::UserCallbacks::MemoryRead8>(code, conf.callbacks);
    prelude_info.read_memory_16 = EmitCallTrampoline<&A32::UserCallbacks::MemoryRead16>(code, conf.callbacks);
    prelude_info.read_memory_32 = EmitCallTrampoline<&A32::UserCallbacks::MemoryRead32>(code, conf.callbacks);
    prelude_info.read_memory_64 = EmitCallTrampoline<&A32::UserCallbacks::MemoryRead64>(code, conf.callbacks);
    prelude_info.wrapped_read_memory_8 = EmitWrappedReadCallTrampoline<&A32::UserCallbacks::MemoryRead8>(code, conf.callbacks);
    prelude_info.wrapped_read_memory_16 = EmitWrappedReadCallTrampoline<&A32::UserCallbacks::MemoryRead16>(code, conf.callbacks);
    prelude_info.wrapped_read_memory_32 = EmitWrappedReadCallTrampoline<&A32::UserCallbacks::MemoryRead32>(code, conf.callbacks);
    prelude_info.wrapped_read_memory_64 = EmitWrappedReadCallTrampoline<&A32::UserCallbacks::MemoryRead64>(code, conf.callbacks);
    prelude_info.exclusive_read_memory_8 = EmitExclusiveReadCallTrampoline<&A32::UserCallbacks::MemoryRead8, u8>(code, conf);
    prelude_info.exclusive_read_memory_16 = EmitExclusiveReadCallTrampoline<&A32::UserCallbacks::MemoryRead16, u16>(code, conf);
    prelude_info.exclusive_read_memory_32 = EmitExclusiveReadCallTrampoline<&A32::UserCallbacks::MemoryRead32, u32>(code, conf);
    prelude_info.exclusive_read_memory_64 = EmitExclusiveReadCallTrampoline<&A32::UserCallbacks::MemoryRead64, u64>(code, conf);
    prelude_info.write_memory_8 = EmitCallTrampoline<&A32::UserCallbacks::MemoryWrite8>(code, conf.callbacks);
    prelude_info.write_memory_16 = EmitCallTrampoline<&A32::UserCallbacks::MemoryWrite16>(code, conf.callbacks);
    prelude_info.write_memory_32 = EmitCallTrampoline<&A32::UserCallbacks::MemoryWrite32>(code, conf.callbacks);
    prelude_info.write_memory_64 = EmitCallTrampoline<&A32::UserCallbacks::MemoryWrite64>(code, conf.callbacks);
    prelude_info.wrapped_write_memory_8 = EmitWrappedWriteCallTrampoline<&A32::UserCallbacks::MemoryWrite8>(code, conf.callbacks);
    prelude_info.wrapped_write_memory_16 = EmitWrappedWriteCallTrampoline<&A32::UserCallbacks::MemoryWrite16>(code, conf.callbacks);
    prelude_info.wrapped_write_memory_32 = EmitWrappedWriteCallTrampoline<&A32::UserCallbacks::MemoryWrite32>(code, conf.callbacks);
    prelude_info.wrapped_write_memory_64 = EmitWrappedWriteCallTrampoline<&A32::UserCallbacks::MemoryWrite64>(code, conf.callbacks);
    prelude_info.exclusive_write_memory_8 = EmitExclusiveWriteCallTrampoline<&A32::UserCallbacks::MemoryWriteExclusive8, u8>(code, conf);
    prelude_info.exclusive_write_memory_16 = EmitExclusiveWriteCallTrampoline<&A32::UserCallbacks::MemoryWriteExclusive16, u16>(code, conf);
    prelude_info.exclusive_write_memory_32 = EmitExclusiveWriteCallTrampoline<&A32::UserCallbacks::MemoryWriteExclusive32, u32>(code, conf);
    prelude_info.exclusive_write_memory_64 = EmitExclusiveWriteCallTrampoline<&A32::UserCallbacks::MemoryWriteExclusive64, u64>(code, conf);
    prelude_info.call_svc = EmitCallTrampoline<&A32::UserCallbacks::CallSVC>(code, conf.callbacks);
    prelude_info.exception_raised = EmitCallTrampoline<&A32::UserCallbacks::ExceptionRaised>(code, conf.callbacks);
    prelude_info.isb_raised = EmitCallTrampoline<&A32::UserCallbacks::InstructionSynchronizationBarrierRaised>(code, conf.callbacks);
    prelude_info.add_ticks = EmitCallTrampoline<&A32::UserCallbacks::AddTicks>(code, conf.callbacks);
    prelude_info.get_ticks_remaining = EmitCallTrampoline<&A32::UserCallbacks::GetTicksRemaining>(code, conf.callbacks);

    Xbyak_loongarch64::Label return_from_run_code, l_return_to_dispatcher;

    prelude_info.run_code = code.getCurr<PreludeInfo::RunCodeFuncType>();
    {
        ABI_PushRegisters(code, ABI_CALLEE_SAVE | (1 << 30), sizeof(StackLayout));

        code.add_d(code.s0, code.a0, code.zero);
        code.add_d(Xstate, code.a1, code.zero);
        code.add_d(Xhalt, code.a2, code.zero);
        if (conf.page_table) {
            code.add_imm(Xpagetable, code.zero, mcl::bit_cast<u64>(conf.page_table), code.t0);
        }
        if (conf.fastmem_pointer) {
            code.add_imm(Xfastmem, code.zero, mcl::bit_cast<u64>(conf.fastmem_pointer), code.t0);
        }

        if (conf.HasOptimization(OptimizationFlag::ReturnStackBuffer)) {
            code.pcaddi(Xscratch0, l_return_to_dispatcher);
            for (size_t i = 0; i < RSBCount; i++) {
                code.st_d(Xscratch0, code.sp, offsetof(StackLayout, rsb) + offsetof(RSBEntry, code_ptr) + i * sizeof(RSBEntry));
            }
        }

        if (conf.enable_cycle_counting) {
            code.bl((uint64_t)prelude_info.get_ticks_remaining);
            code.add_d(Xticks, code.a0, code.zero);
            code.st_d(Xticks, code.sp, offsetof(StackLayout, cycles_to_run));
        }

        code.ld_d(Wscratch0, Xstate, offsetof(A32JitState, upper_location_descriptor));
        code.andi(Wscratch0, Wscratch0, 0xffff0000);
        code.MRS(Xscratch1, Xbyak_loongarch64::SystemReg::FPCR);
        code.st_w(Wscratch1, code.sp, offsetof(StackLayout, save_host_fpcr));
        code.MSR(Xbyak_loongarch64::SystemReg::FPCR, Xscratch0);

        code.ll_acq_w(Wscratch0, Xhalt);
        code.bnez(Wscratch0, return_from_run_code);

        code.jirl(code.zero, code.s0, 0);
    }

    prelude_info.step_code = code.getCurr<PreludeInfo::RunCodeFuncType>();
    {
        ABI_PushRegisters(code, ABI_CALLEE_SAVE | (1 << 30), sizeof(StackLayout));

        code.add_d(code.s0, code.a0, code.zero);
        code.add_d(Xstate, code.a1, code.zero);
        code.add_d(Xhalt, code.a2, code.zero);
        if (conf.page_table) {
            code.add_imm(Xpagetable, code.zero, mcl::bit_cast<u64>(conf.page_table), code.t0);
        }
        if (conf.fastmem_pointer) {
            code.add_imm(Xfastmem, code.zero, mcl::bit_cast<u64>(conf.fastmem_pointer), code.t0);
        }

        if (conf.HasOptimization(OptimizationFlag::ReturnStackBuffer)) {
            code.pcaddi(Xscratch0, l_return_to_dispatcher);
            for (size_t i = 0; i < RSBCount; i++) {
                code.st_d(Xscratch0, code.sp, offsetof(StackLayout, rsb) + offsetof(RSBEntry, code_ptr) + i * sizeof(RSBEntry));
            }
        }

        if (conf.enable_cycle_counting) {
            code.addi_d(Xticks, Xticks, 1);
            code.st_d(Xticks, code.sp, offsetof(StackLayout, cycles_to_run));
        }

        code.ld_d(Wscratch0, Xstate, offsetof(A32JitState, upper_location_descriptor));
        code.andi(Wscratch0, Wscratch0, 0xffff0000);
        code.MRS(Xscratch1, Xbyak_loongarch64::SystemReg::FPCR);
        code.st_d(Wscratch1, code.sp, offsetof(StackLayout, save_host_fpcr));
        code.MSR(Xbyak_loongarch64::SystemReg::FPCR, Xscratch0);

        Xbyak_loongarch64::Label step_hr_loop;
        code.L(step_hr_loop);
        code.LDAXR(Wscratch0, Xhalt);
        code.bnez(Wscratch0, return_from_run_code);
        code.ORR(Wscratch0, Wscratch0, static_cast<u32>(HaltReason::Step));
        code.STLXR(Wscratch1, Wscratch0, Xhalt);
        code.bnez(Wscratch1, step_hr_loop);

        code.jirl(code.zero, code.s0, 0);
    }

    prelude_info.return_to_dispatcher = code.getCurr<void*>();
    {
        Xbyak_loongarch64::Label l_this, l_addr;

        code.ll_acq_w(Wscratch0, Xhalt);
        code.bnez(Wscratch0, return_from_run_code);

        if (conf.enable_cycle_counting) {
            code.blt(Xticks, code.zero, return_from_run_code);
            code.beqz(Xticks, return_from_run_code);
        }

        code.pcaddi(code.a0, l_this);
        code.add_d(code.a1, Xstate, code.zero);
        code.pcaddi(Xscratch0, l_addr);
        code.jirl(code.ra, Xscratch0, 0);
        code.jirl(code.zero, code.a0, 0);

        const auto fn = [](A32AddressSpace& self, A32JitState& context) -> CodePtr {
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
            code.bl(prelude_info.add_ticks);
        }

        code.ld_d(Wscratch0, code.sp, offsetof(StackLayout, save_host_fpcr));
        code.MSR(Xbyak_loongarch64::SystemReg::FPCR, Xscratch0);

        Xbyak_loongarch64::Label exit_hr_loop;
        code.L(exit_hr_loop);
        code.LDAXR(W0, Xhalt);
        code.STLXR(Wscratch0, WZR, Xhalt);
        code.bnez(Wscratch0, exit_hr_loop);

        ABI_PopRegisters(code, ABI_CALLEE_SAVE | (1 << 30), sizeof(StackLayout));
        code.jirl(code.zero, code.ra, 0);
        ();
    }

    code.align(8);
    code.L(l_return_to_dispatcher);
    code.dx(mcl::bit_cast<u64>(prelude_info.return_to_dispatcher));

    prelude_info.end_of_prelude = code.getCurr<u32*>();

    //    mem.invalidate_all();
    //    mem.protect();
}

EmitConfig A32AddressSpace::GetEmitConfig() {
    return EmitConfig{
        .optimizations = conf.unsafe_optimizations ? conf.optimizations : conf.optimizations & all_safe_optimizations,

        .hook_isb = conf.hook_isb,

        .cntfreq_el0{},
        .ctr_el0{},
        .dczid_el0{},
        .tpidrro_el0{},
        .tpidr_el0{},

        .check_halt_on_memory_access = conf.check_halt_on_memory_access,

        .page_table_pointer = mcl::bit_cast<u64>(conf.page_table),
        .page_table_address_space_bits = 32,
        .page_table_pointer_mask_bits = conf.page_table_pointer_mask_bits,
        .silently_mirror_page_table = true,
        .absolute_offset_page_table = conf.absolute_offset_page_table,
        .detect_misaligned_access_via_page_table = conf.detect_misaligned_access_via_page_table,
        .only_detect_misalignment_via_page_table_on_page_boundary = conf.only_detect_misalignment_via_page_table_on_page_boundary,

        .fastmem_pointer = mcl::bit_cast<u64>(conf.fastmem_pointer),
        .recompile_on_fastmem_failure = conf.recompile_on_fastmem_failure,
        .fastmem_address_space_bits = 32,
        .silently_mirror_fastmem = true,

        .wall_clock_cntpct = conf.wall_clock_cntpct,
        .enable_cycle_counting = conf.enable_cycle_counting,

        .always_little_endian = conf.always_little_endian,

        .descriptor_to_fpcr = [](const IR::LocationDescriptor& location) { return FP::FPCR{A32::LocationDescriptor{location}.FPSCR().Value()}; },
        .emit_cond = EmitA32Cond,
        .emit_condition_failed_terminal = EmitA32ConditionFailedTerminal,
        .emit_terminal = EmitA32Terminal,
        .emit_check_memory_abort = EmitA32CheckMemoryAbort,

        .state_nzcv_offset = offsetof(A32JitState, cpsr_nzcv),
        .state_fpsr_offset = offsetof(A32JitState, fpsr),
        .state_exclusive_state_offset = offsetof(A32JitState, exclusive_state),

        .coprocessors = conf.coprocessors,

        .very_verbose_debugging_output = conf.very_verbose_debugging_output,
    };
}

void A32AddressSpace::RegisterNewBasicBlock(const IR::Block& block, const EmittedBlockInfo&) {
    const A32::LocationDescriptor descriptor{block.Location()};
    const A32::LocationDescriptor end_location{block.EndLocation()};
    const auto range = boost::icl::discrete_interval<u32>::closed(descriptor.PC(), end_location.PC() - 1);
    block_ranges.AddRange(range, descriptor);
}

}  // namespace Dynarmic::Backend::LoongArch64
