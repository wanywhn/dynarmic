/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/loongarch64/reg_alloc.h"

#include <algorithm>
#include <array>
#include <iterator>

#include <mcl/assert.hpp>
#include <mcl/bit/bit_field.hpp>
#include <mcl/bit_cast.hpp>
#include <mcl/mp/metavalue/lift_value.hpp>
#include <mcl/stdint.hpp>

#include "dynarmic/backend/loongarch64/abi.h"
#include "dynarmic/backend/loongarch64/emit_context.h"
#include "dynarmic/backend/loongarch64/fpsr_manager.h"
#include "dynarmic/backend/loongarch64/verbose_debugging_output.h"
#include "dynarmic/common/always_false.h"

namespace Dynarmic::Backend::LoongArch64 {

constexpr size_t spill_offset = offsetof(StackLayout, spill);
constexpr size_t spill_slot_size = sizeof(decltype(StackLayout::spill)::value_type);

static bool IsValuelessType(IR::Type type) {
    switch (type) {
    case IR::Type::Table:
        return true;
    default:
        return false;
    }
}

IR::Type Argument::GetType() const {
    return value.GetType();
}

bool Argument::IsImmediate() const {
    return value.IsImmediate();
}

bool Argument::GetImmediateU1() const {
    return value.GetU1();
}

u8 Argument::GetImmediateU8() const {
    const u64 imm = value.GetImmediateAsU64();
    ASSERT(imm < 0x100);
    return u8(imm);
}

u16 Argument::GetImmediateU16() const {
    const u64 imm = value.GetImmediateAsU64();
    ASSERT(imm < 0x10000);
    return u16(imm);
}

u32 Argument::GetImmediateU32() const {
    const u64 imm = value.GetImmediateAsU64();
    ASSERT(imm < 0x100000000);
    return u32(imm);
}

u64 Argument::GetImmediateU64() const {
    return value.GetImmediateAsU64();
}

IR::Cond Argument::GetImmediateCond() const {
    ASSERT(IsImmediate() && GetType() == IR::Type::Cond);
    return value.GetCond();
}

IR::AccType Argument::GetImmediateAccType() const {
    ASSERT(IsImmediate() && GetType() == IR::Type::AccType);
    return value.GetAccType();
}

HostLoc::Kind Argument::CurrentLocationKind() const {
    return reg_alloc.ValueLocation(value.GetInst())->kind;
}

bool HostLocInfo::Contains(const IR::Inst* value) const {
    return std::find(values.begin(), values.end(), value) != values.end();
}

void HostLocInfo::SetupScratchLocation() {
    ASSERT(IsCompletelyEmpty());
    realized = true;
}

void HostLocInfo::SetupLocation(const IR::Inst* value) {
    ASSERT(IsCompletelyEmpty());
    values.clear();
    values.emplace_back(value);
    realized = true;
    uses_this_inst = 0;
    accumulated_uses = 0;
    expected_uses = value->UseCount();
}

bool HostLocInfo::IsCompletelyEmpty() const {
    return values.empty() && !locked && !realized && !accumulated_uses && !expected_uses && !uses_this_inst;
}

bool HostLocInfo::MaybeAllocatable() const {
    return !locked && !realized;
}

bool HostLocInfo::IsOneRemainingUse() const {
    return accumulated_uses + 1 == expected_uses && uses_this_inst == 1;
}

void HostLocInfo::UpdateUses() {
    accumulated_uses += uses_this_inst;
    uses_this_inst = 0;

    if (accumulated_uses == expected_uses) {
        values.clear();
        accumulated_uses = 0;
        expected_uses = 0;
    }
}

RegAlloc::ArgumentInfo RegAlloc::GetArgumentInfo(IR::Inst* inst) {
    ArgumentInfo ret = {Argument{*this}, Argument{*this}, Argument{*this}, Argument{*this}};
    for (size_t i = 0; i < inst->NumArgs(); i++) {
        const IR::Value arg = inst->GetArg(i);
        ret[i].value = arg;
        if (!arg.IsImmediate() && !IsValuelessType(arg.GetType())) {
            ASSERT_MSG(ValueLocation(arg.GetInst()), "argument must already been defined");
            ValueInfo(arg.GetInst()).uses_this_inst++;
        }
    }
    return ret;
}

bool RegAlloc::WasValueDefined(IR::Inst* inst) const {
    return defined_insts.count(inst) > 0;
}

void RegAlloc::PrepareForCall(std::optional<Argument::copyable_reference> arg0, std::optional<Argument::copyable_reference> arg1, std::optional<Argument::copyable_reference> arg2, std::optional<Argument::copyable_reference> arg3) {
    fpsr_manager.Spill();
    SpillFlags();

    // TODO: Spill into callee-save registers

    for (int i = 0; i < 32; i++) {
        if (mcl::bit::get_bit(i, static_cast<u32>(ABI_CALLER_SAVE))) {
            SpillGpr(i);
        }
    }

    for (int i = 0; i < 32; i++) {
        if (mcl::bit::get_bit(i, static_cast<u32>(ABI_CALLER_SAVE >> 32))) {
            SpillFpr(i);
        }
    }

    const std::array<std::optional<Argument::copyable_reference>, 4> args{arg0, arg1, arg2, arg3};

    // FIXME is this is args reg, should change to a0?
    // AAPCS64 Next General-purpose Register Number
    unsigned int ngrn = 4;
    // AAPCS64 Next SIMD and Floating-point Register Number
    unsigned int nsrn = 0;

    for (int i = 0; i < 4; i++) {
        if (args[i]) {
            if (args[i]->get().GetType() == IR::Type::U128) {
                ASSERT(fprs[nsrn].IsCompletelyEmpty());
                LoadCopyInto(args[i]->get().value, Xbyak_loongarch64::VReg{nsrn});
                nsrn++;
            } else {
                ASSERT(gprs[ngrn].IsCompletelyEmpty());
                LoadCopyInto(args[i]->get().value, Xbyak_loongarch64::XReg{ngrn});
                ngrn++;
            }
        } else {
            // Gaps are assumed to be in general-purpose registers
            // TODO: should there be a separate list passed for FPRs instead?
            ngrn++;
        }
    }
}

void RegAlloc::DefineAsExisting(IR::Inst* inst, Argument& arg) {
    defined_insts.emplace(inst);

    ASSERT(!ValueLocation(inst));

    if (arg.value.IsImmediate()) {
        inst->ReplaceUsesWith(arg.value);
        return;
    }

    auto& info = ValueInfo(arg.value.GetInst());
    info.values.emplace_back(inst);
    info.expected_uses += inst->UseCount();
}

void RegAlloc::DefineAsRegister(IR::Inst* inst, Xbyak_loongarch64::Reg reg) {
    defined_insts.emplace(inst);

    ASSERT(!ValueLocation(inst));
    auto& info = reg.isVReg() ? fprs[reg.getIdx()] : gprs[reg.getIdx()];
    ASSERT(info.IsCompletelyEmpty());
    info.values.emplace_back(inst);
    info.expected_uses += inst->UseCount();
}

void RegAlloc::UpdateAllUses() {
    for (auto& gpr : gprs) {
        gpr.UpdateUses();
    }
    for (auto& fpr : fprs) {
        fpr.UpdateUses();
    }
    flags.UpdateUses();
    for (auto& spill : spills) {
        spill.UpdateUses();
    }
}

void RegAlloc::AssertAllUnlocked() const {
    const auto is_unlocked = [](const auto& i) { return !i.locked && !i.realized; };
    ASSERT(std::all_of(gprs.begin(), gprs.end(), is_unlocked));
    ASSERT(std::all_of(fprs.begin(), fprs.end(), is_unlocked));
    ASSERT(is_unlocked(flags));
    ASSERT(std::all_of(spills.begin(), spills.end(), is_unlocked));
}

void RegAlloc::AssertNoMoreUses() const {
    const auto is_empty = [](const auto& i) { return i.IsCompletelyEmpty(); };
    ASSERT(std::all_of(gprs.begin(), gprs.end(), is_empty));
    ASSERT(std::all_of(fprs.begin(), fprs.end(), is_empty));
    ASSERT(is_empty(flags));
    ASSERT(std::all_of(spills.begin(), spills.end(), is_empty));
}

void RegAlloc::EmitVerboseDebuggingOutput() {
    code.add_imm(code.s0, code.zero,mcl::bit_cast<u64>(&PrintVerboseDebuggingOutputLine), Xscratch0);  // Non-volatile register

    const auto do_location = [&](HostLocInfo& info, HostLocType type, size_t index) {
        for (const IR::Inst* value : info.values) {
            code.add_d(code.a0, code.sp, code.zero);
            code.add_imm(code.a1, code.zero, static_cast<u64>(type), Xscratch0);
            code.add_imm(code.a2, code.zero, index, Xscratch0);
            code.add_imm(code.a3, code.zero, value->GetName(), Xscratch0);
            code.add_imm(code.a4, code.zero, static_cast<u64>(value->GetType()), Xscratch0);

            code.jirl(code.ra, code.s0, 0);
        }
    };

    for (size_t i = 0; i < gprs.size(); i++) {
        do_location(gprs[i], HostLocType::X, i);
    }
    for (size_t i = 0; i < fprs.size(); i++) {
        do_location(fprs[i], HostLocType::Q, i);
    }
    do_location(flags, HostLocType::Nzcv, 0);
    for (size_t i = 0; i < spills.size(); i++) {
        do_location(spills[i], HostLocType::Spill, i);
    }
}

template<HostLoc::Kind kind>
unsigned int RegAlloc::GenerateImmediate(const IR::Value& value) {
    //FIXME this could delete
//    ASSERT(value.GetType() != IR::Type::U1);
    if constexpr (kind == HostLoc::Kind::Gpr) {
        const unsigned int new_location_index = AllocateRegister(gprs, gpr_order);
        SpillGpr(new_location_index);
        gprs[new_location_index].SetupScratchLocation();

        code.add_imm(Xbyak_loongarch64::XReg{new_location_index}, code.zero, value.GetImmediateAsU64(), Xscratch0);

        return new_location_index;
    } else if constexpr (kind == HostLoc::Kind::Fpr) {
        const unsigned int new_location_index = AllocateRegister(fprs, fpr_order);
        SpillFpr(new_location_index);
        fprs[new_location_index].SetupScratchLocation();

        code.add_imm(Xscratch0, code.zero, value.GetImmediateAsU64(), Xscratch1);
        code.vinsgr2vr_d(Xbyak_loongarch64::VReg{new_location_index}, Xscratch0, 0);

        return new_location_index;
    } else if constexpr (kind == HostLoc::Kind::Flags) {
        SpillFlags();
        flags.SetupScratchLocation();

        code.add_imm(Xscratch0, code.zero, value.GetImmediateAsU64(), Xscratch1);
        // FIXME
//        code.MSR(Xbyak_loongarch64::SystemReg::NZCV, Xscratch0);

        return 0;
    } else {
        static_assert(Common::always_false_v<mcl::mp::lift_value<kind>>);
    }
}

template<HostLoc::Kind required_kind>
unsigned int RegAlloc::RealizeReadImpl(const IR::Value& value) {
    if (value.IsImmediate()) {
        return GenerateImmediate<required_kind>(value);
    }

    const auto current_location = ValueLocation(value.GetInst());
    ASSERT(current_location);

    if (current_location->kind == required_kind) {
        ValueInfo(*current_location).realized = true;
        return current_location->index;
    }

    ASSERT(!ValueInfo(*current_location).realized);
    ASSERT(ValueInfo(*current_location).locked);

    if constexpr (required_kind == HostLoc::Kind::Gpr) {
        const unsigned int new_location_index = AllocateRegister(gprs, gpr_order);
        SpillGpr(new_location_index);

        switch (current_location->kind) {
        case HostLoc::Kind::Gpr:
            ASSERT_FALSE("Logic error");
            break;
        case HostLoc::Kind::Fpr:
            code.vpickve2gr_d(Xbyak_loongarch64::XReg{new_location_index}, Xbyak_loongarch64::VReg{current_location->index}, 0);
            // ASSERT size fits
            break;
        case HostLoc::Kind::Spill:
            code.ld_d(Xbyak_loongarch64::XReg{new_location_index}, code.sp, spill_offset + current_location->index * spill_slot_size);
            break;
        case HostLoc::Kind::Flags:
            // FIXME
//            code.MRS(Xbyak_loongarch64::XReg{new_location_index}, Xbyak_loongarch64::SystemReg::NZCV);
            break;
        }

        gprs[new_location_index] = std::exchange(ValueInfo(*current_location), {});
        gprs[new_location_index].realized = true;
        return new_location_index;
    } else if constexpr (required_kind == HostLoc::Kind::Fpr) {
        const unsigned int new_location_index = AllocateRegister(fprs, fpr_order);
        SpillFpr(new_location_index);

        switch (current_location->kind) {
        case HostLoc::Kind::Gpr:
            code.vinsgr2vr_d(Xbyak_loongarch64::VReg{new_location_index}, Xbyak_loongarch64::XReg{current_location->index}, 0);
            break;
        case HostLoc::Kind::Fpr:
            ASSERT_FALSE("Logic error");
            break;
        case HostLoc::Kind::Spill:
            code.ld_d(Xscratch0, code.sp, spill_offset + current_location->index * spill_slot_size);
            code.ld_d(Xscratch1, code.sp, spill_offset + current_location->index * spill_slot_size + 64);
            code.vinsgr2vr_d(Xbyak_loongarch64::VReg{new_location_index}, Xscratch0, 0);
            code.vinsgr2vr_d(Xbyak_loongarch64::VReg{new_location_index}, Xscratch1, 1);
            break;
        case HostLoc::Kind::Flags:
            ASSERT_FALSE("Moving from flags into fprs is not currently supported");
            break;
        }

        fprs[new_location_index] = std::exchange(ValueInfo(*current_location), {});
        fprs[new_location_index].realized = true;
        return new_location_index;
    } else if constexpr (required_kind == HostLoc::Kind::Flags) {
        ASSERT_FALSE("A simple read from flags is likely a logic error.");
    } else {
        static_assert(Common::always_false_v<mcl::mp::lift_value<required_kind>>);
    }
}

template<HostLoc::Kind kind>
unsigned int RegAlloc::RealizeWriteImpl(const IR::Inst* value) {
    defined_insts.emplace(value);
    auto op = value->GetType();
    ASSERT_MSG(!ValueLocation(value), Dynarmic::IR::GetNameOf((IR::Type)op) + " " + std::to_string( value->GetName()) + "\r\n");

    if constexpr (kind == HostLoc::Kind::Gpr) {
        const int new_location_index = AllocateRegister(gprs, gpr_order);
        SpillGpr(new_location_index);
        gprs[new_location_index].SetupLocation(value);
        return new_location_index;
    } else if constexpr (kind == HostLoc::Kind::Fpr) {
        const int new_location_index = AllocateRegister(fprs, fpr_order);
        SpillFpr(new_location_index);
        fprs[new_location_index].SetupLocation(value);
        return new_location_index;
    } else if constexpr (kind == HostLoc::Kind::Flags) {
        SpillFlags();
        flags.SetupLocation(value);
        return 0;
    } else {
        static_assert(Common::always_false_v<mcl::mp::lift_value<kind>>);
    }
}

template<HostLoc::Kind kind>
unsigned int RegAlloc::RealizeReadWriteImpl(const IR::Value& read_value, const IR::Inst* write_value) {
    defined_insts.emplace(write_value);

    // TODO: Move elimination

    const unsigned int write_loc = RealizeWriteImpl<kind>(write_value);

    if constexpr (kind == HostLoc::Kind::Gpr) {
        LoadCopyInto(read_value, Xbyak_loongarch64::XReg{write_loc});
        return write_loc;
    } else if constexpr (kind == HostLoc::Kind::Fpr) {
        LoadCopyInto(read_value, Xbyak_loongarch64::VReg{write_loc});
        return write_loc;
    } else if constexpr (kind == HostLoc::Kind::Flags) {
        ASSERT_FALSE("Incorrect function for ReadWrite of flags");
    } else {
        static_assert(Common::always_false_v<mcl::mp::lift_value<kind>>);
    }
}

template unsigned int RegAlloc::RealizeReadImpl<HostLoc::Kind::Gpr>(const IR::Value& value);
template unsigned int RegAlloc::RealizeReadImpl<HostLoc::Kind::Fpr>(const IR::Value& value);
template unsigned int RegAlloc::RealizeReadImpl<HostLoc::Kind::Flags>(const IR::Value& value);
template unsigned int RegAlloc::RealizeWriteImpl<HostLoc::Kind::Gpr>(const IR::Inst* value);
template unsigned int RegAlloc::RealizeWriteImpl<HostLoc::Kind::Fpr>(const IR::Inst* value);
template unsigned int RegAlloc::RealizeWriteImpl<HostLoc::Kind::Flags>(const IR::Inst* value);
template unsigned int RegAlloc::RealizeReadWriteImpl<HostLoc::Kind::Gpr>(const IR::Value& read_value, const IR::Inst* write_value);
template unsigned int RegAlloc::RealizeReadWriteImpl<HostLoc::Kind::Fpr>(const IR::Value& read_value, const IR::Inst* write_value);
template unsigned int RegAlloc::RealizeReadWriteImpl<HostLoc::Kind::Flags>(const IR::Value& read_value, const IR::Inst* write_value);

int RegAlloc::AllocateRegister(const std::array<HostLocInfo, 32>& regs, const std::vector<int>& order) const {
    const auto empty = std::find_if(order.begin(), order.end(), [&](int i) { return regs[i].IsCompletelyEmpty(); });
    if (empty != order.end()) {
        return *empty;
    }

    std::vector<int> candidates;
    std::copy_if(order.begin(), order.end(), std::back_inserter(candidates), [&](int i) { return regs[i].MaybeAllocatable(); });

    // TODO: LRU
    std::uniform_int_distribution<size_t> dis{0, candidates.size() - 1};
    return candidates[dis(rand_gen)];
}

void RegAlloc::SpillGpr(unsigned int index) {
    ASSERT(!gprs[index].locked && !gprs[index].realized);
    if (gprs[index].values.empty()) {
        return;
    }
    const int new_location_index = FindFreeSpill();
    code.st_d(Xbyak_loongarch64::XReg{index}, code.sp, spill_offset + new_location_index * spill_slot_size);
    spills[new_location_index] = std::exchange(gprs[index], {});
}

void RegAlloc::SpillFpr(unsigned int index) {
    ASSERT(!fprs[index].locked && !fprs[index].realized);
    if (fprs[index].values.empty()) {
        return;
    }
    const int new_location_index = FindFreeSpill();
    code.vst(Xbyak_loongarch64::VReg{index}, code.sp, spill_offset + new_location_index * spill_slot_size);
    spills[new_location_index] = std::exchange(fprs[index], {});
}

void RegAlloc::ReadWriteFlags(Argument& read, IR::Inst* write) {
    defined_insts.emplace(write);

    const auto current_location = ValueLocation(read.value.GetInst());
    ASSERT(current_location);

    if (current_location->kind == HostLoc::Kind::Flags) {
        if (!flags.IsOneRemainingUse()) {
            SpillFlags();
        }
    } else if (current_location->kind == HostLoc::Kind::Gpr) {
        if (!flags.values.empty()) {
            SpillFlags();
        }
        // FIXME
//        code.MSR(Xbyak_loongarch64::SystemReg::NZCV, Xbyak_loongarch64::XReg{current_location->index});
    } else if (current_location->kind == HostLoc::Kind::Spill) {
        if (!flags.values.empty()) {
            SpillFlags();
        }
        code.ld_d(Wscratch0, code.sp, spill_offset + current_location->index * spill_slot_size);
        // FIXME
//        code.MSR(Xbyak_loongarch64::SystemReg::NZCV, Xscratch0);
    } else {
        ASSERT_FALSE("Invalid current location for flags");
    }

    if (write) {
        flags.SetupLocation(write);
        flags.realized = false;
    }
}
// no need probably
void RegAlloc::SpillFlags() {
    ASSERT(!flags.locked && !flags.realized);
    if (flags.values.empty()) {
        return;
    }
    const unsigned int new_location_index = AllocateRegister(gprs, gpr_order);
    SpillGpr(new_location_index);
    // FIXME
//    code.MRS(Xbyak_loongarch64::XReg{new_location_index}, Xbyak_loongarch64::SystemReg::NZCV);
    gprs[new_location_index] = std::exchange(flags, {});
}

int RegAlloc::FindFreeSpill() const {
    const auto iter = std::find_if(spills.begin(), spills.end(), [](const HostLocInfo& info) { return info.values.empty(); });
    ASSERT_MSG(iter != spills.end(), "All spill locations are full");
    return static_cast<int>(iter - spills.begin());
}

void RegAlloc::LoadCopyInto(const IR::Value& value, Xbyak_loongarch64::XReg reg) {
    if (value.IsImmediate()) {
        code.add_imm(reg, code.zero, value.GetImmediateAsU64(), Xscratch0);
        return;
    }

    const auto current_location = ValueLocation(value.GetInst());
    ASSERT(current_location);
    switch (current_location->kind) {
    case HostLoc::Kind::Gpr:
        code.add_d(reg, Xbyak_loongarch64::XReg{current_location->index}, code.zero);
        break;
    case HostLoc::Kind::Fpr:
        code.vpickve2gr_d(reg, Xbyak_loongarch64::VReg{current_location->index}, 0);
        // ASSERT size fits
        break;
    case HostLoc::Kind::Spill:
        code.ld_d(reg, code.sp, spill_offset + current_location->index * spill_slot_size);
        break;
    case HostLoc::Kind::Flags:
        // FIXME
//        code.MRS(reg, Xbyak_loongarch64::SystemReg::NZCV);
        break;
    }
}

void RegAlloc::LoadCopyInto(const IR::Value& value, Xbyak_loongarch64::VReg reg) {
    if (value.IsImmediate()) {
        code.add_imm(Xscratch0, code.zero, value.GetImmediateAsU64(), Xscratch1);
        code.movgr2fr_d(Xbyak_loongarch64::XReg{reg.getIdx()}, Xscratch0);
        return;
    }

    const auto current_location = ValueLocation(value.GetInst());
    ASSERT(current_location);
    switch (current_location->kind) {
    case HostLoc::Kind::Gpr:
        // TODO hi bit to zero?
        code.vinsgr2vr_d(reg, Xbyak_loongarch64::XReg{current_location->index}, 0);

        break;
    case HostLoc::Kind::Fpr:
        code.vor_v(reg, Xbyak_loongarch64::VReg{current_location->index}, Xbyak_loongarch64::VReg{current_location->index});
        break;
    case HostLoc::Kind::Spill:
        // TODO: Minimize move size to max value width
        code.vld(reg, code.sp, spill_offset + current_location->index * spill_slot_size);
        break;
    case HostLoc::Kind::Flags:
        ASSERT_FALSE("Moving from flags into fprs is not currently supported");
        break;
    }
}

std::optional<HostLoc> RegAlloc::ValueLocation(const IR::Inst* value) const {
    const auto contains_value = [value](const HostLocInfo& info) { return info.Contains(value); };

    if (const auto iter = std::find_if(gprs.begin(), gprs.end(), contains_value); iter != gprs.end()) {
        return HostLoc{HostLoc::Kind::Gpr, static_cast<unsigned int>(iter - gprs.begin())};
    }
    if (const auto iter = std::find_if(fprs.begin(), fprs.end(), contains_value); iter != fprs.end()) {
        return HostLoc{HostLoc::Kind::Fpr, static_cast<unsigned int>(iter - fprs.begin())};
    }
    if (contains_value(flags)) {
        return HostLoc{HostLoc::Kind::Flags, 0};
    }
    if (const auto iter = std::find_if(spills.begin(), spills.end(), contains_value); iter != spills.end()) {
        return HostLoc{HostLoc::Kind::Spill, static_cast<unsigned int>(iter - spills.begin())};
    }
    return std::nullopt;
}

HostLocInfo& RegAlloc::ValueInfo(HostLoc host_loc) {
    switch (host_loc.kind) {
    case HostLoc::Kind::Gpr:
        return gprs[static_cast<size_t>(host_loc.index)];
    case HostLoc::Kind::Fpr:
        return fprs[static_cast<size_t>(host_loc.index)];
    case HostLoc::Kind::Flags:
        return flags;
    case HostLoc::Kind::Spill:
        return spills[static_cast<size_t>(host_loc.index)];
    }
    ASSERT_FALSE("RegAlloc::ValueInfo: Invalid HostLoc::Kind");
}

HostLocInfo& RegAlloc::ValueInfo(const IR::Inst* value) {
    const auto contains_value = [value](const HostLocInfo& info) { return info.Contains(value); };

    if (const auto iter = std::find_if(gprs.begin(), gprs.end(), contains_value); iter != gprs.end()) {
        return *iter;
    }
    if (const auto iter = std::find_if(fprs.begin(), fprs.end(), contains_value); iter != fprs.end()) {
        return *iter;
    }
    if (contains_value(flags)) {
        return flags;
    }
    if (const auto iter = std::find_if(spills.begin(), spills.end(), contains_value); iter != spills.end()) {
        return *iter;
    }
    ASSERT_FALSE("RegAlloc::ValueInfo: Value not found");
}

}  // namespace Dynarmic::Backend::LoongArch64
