/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include "dynarmic/backend/block_range_information.h"
#include "dynarmic/backend/loongarch64/address_space.h"
#include "dynarmic/interface/A64/config.h"

namespace Dynarmic::Backend::LoongArch64 {

struct EmittedBlockInfo;

class A64AddressSpace final : public AddressSpace {
public:
    explicit A64AddressSpace(const A64::UserConfig& conf, JitStateInfo jsi);

    IR::Block GenerateIR(IR::LocationDescriptor) const override;

    void InvalidateCacheRanges(const boost::icl::interval_set<u64>& ranges);

protected:
    friend class A64Core;

    void EmitPrelude();
    EmitConfig GetEmitConfig() override;
    void RegisterNewBasicBlock(const IR::Block& block, const EmittedBlockInfo& block_info) override;

    const A64::UserConfig conf;
    BlockRangeInformation<u64> block_ranges;

    void SwitchFcsrOnEntry();

    void SwitchFcsrOnExit();
};

}  // namespace Dynarmic::Backend::LoongArch64
