/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <Tensile/AMDGPU.hpp>
#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipUtils.hpp>

#ifdef TENSILELITE_HAS_AMD_SMI
#include <amd_smi/amdsmi.h>
#endif

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace TensileLite
{
    namespace hip
    {
        namespace
        {
            // A copy of `prop` naming the base architecture instead of the
            // silicon revision the runtime reported.
            //
            // Origami matches gcnArchName exactly against a table of base
            // architectures, so a v0 part reporting gfx1250-strict matches
            // nothing and leaves analyticalHardware null, which throws on the
            // assert guarding every query that consults the analytical model.
            // The base entry's machine constants are the closest model a
            // revision has -- a deliberate approximation, since Origami
            // consumes CU counts and clocks rather than the ISA.
            //
            // Rebuilt through toProcessor rather than trimmed as a string so
            // the revisions folded here stay the ones Tensile recognises.
            // Feature suffixes such as `:xnack-` need no handling; Origami
            // drops everything past the first colon itself.
            hipDeviceProp_t propNamingBaseArchitecture(hipDeviceProp_t const& prop)
            {
                hipDeviceProp_t   base = prop;
                std::string const name = AMDGPU::toString(AMDGPU::toProcessor(prop.gcnArchName));
                std::snprintf(base.gcnArchName, sizeof(base.gcnArchName), "%s", name.c_str());
                return base;
            }

            // AIDs of the whole chip; compute partitions split them evenly.
            int chipAIDs(AMDGPU::Processor processor)
            {
                return processor == AMDGPU::Processor::gfx1250 ? 2 : 1;
            }

#ifdef TENSILELITE_HAS_AMD_SMI
            int queryPartitionAIDs(int deviceId, int aidsPerChip)
            {
                char busId[32] = {};
                if(hipDeviceGetPCIBusId(busId, sizeof(busId), deviceId) != hipSuccess)
                    return 0;
                unsigned domain = 0, bus = 0, device = 0, function = 0;
                if(std::sscanf(busId, "%x:%x:%x.%x", &domain, &bus, &device, &function) != 4)
                    return 0;
                amdsmi_bdf_t bdf    = {};
                bdf.domain_number   = domain;
                bdf.bus_number      = bus;
                bdf.device_number   = device;
                bdf.function_number = function;

                // amdsmi_init/amdsmi_shut_down are reference counted, so this does not
                // tear down an amd_smi session the application holds.
                if(amdsmi_init(AMDSMI_INIT_AMD_GPUS) != AMDSMI_STATUS_SUCCESS)
                    return 0;
                int                                    aids    = 0;
                amdsmi_processor_handle                handle  = nullptr;
                amdsmi_accelerator_partition_profile_t profile = {};
                uint32_t partitionIds[AMDSMI_MAX_ACCELERATOR_PARTITIONS] = {};
                if(amdsmi_get_processor_handle_from_bdf(bdf, &handle) == AMDSMI_STATUS_SUCCESS
                   && amdsmi_get_gpu_accelerator_partition_profile(handle, &profile, partitionIds)
                          == AMDSMI_STATUS_SUCCESS
                   && profile.num_partitions > 0)
                    aids = std::max(1, aidsPerChip / static_cast<int>(profile.num_partitions));
                amdsmi_shut_down();
                return aids;
            }
#endif

            // AIDs in deviceId's current compute partition, 0 if unknown. HipAMDGPU is
            // rebuilt on every GetCurrentDevice() (e.g. per ext-op call), so the amd_smi
            // query is cached per device; the partition cannot change while a process
            // holds the GPU.
            int partitionAIDs(int deviceId, AMDGPU::Processor processor)
            {
                int const aidsPerChip = chipAIDs(processor);
                if(aidsPerChip <= 1)
                    return aidsPerChip;
#ifdef TENSILELITE_HAS_AMD_SMI
                static std::mutex                   cacheMutex;
                static std::unordered_map<int, int> cache;
                std::lock_guard<std::mutex>         lock(cacheMutex);
                auto                                it = cache.find(deviceId);
                if(it == cache.end())
                    it = cache.emplace(deviceId, queryPartitionAIDs(deviceId, aidsPerChip)).first;
                return it->second;
#else
                return 0;
#endif
            }
        } // namespace

        HipAMDGPU::HipAMDGPU(hipDeviceProp_t const& prop,
                             int                    deviceId,
                             std::optional<int>     pciChipId)
            : AMDGPU(AMDGPU::toProcessor(prop.gcnArchName),
                     prop.multiProcessorCount,
                     std::string(prop.name),
                     pciChipId)
            , properties(prop)
            , deviceId(deviceId)
        {
            // Both entry points read gcnArchName out of the properties they are
            // handed, so the same renamed copy has to reach both: renaming only
            // the support check would trade the null for a throw out of
            // get_default_num_xcds when the lookup misses.
            hipDeviceProp_t const analyticalProp = propNamingBaseArchitecture(prop);
            if(origami::hardware_t::is_hardware_supported(analyticalProp))
            {
                // Route analyticalHardware construction through Origami's device-id
                // entry point so selection uses the runtime-queried XCC count
                // (hipDeviceAttributeNumberOfXccs on HIP 7+) instead of the hardcoded
                // per-architecture default in get_default_num_xcds().
                //
                // Pass a copy of the caller-provided `prop` so any adjustments made upstream
                // (e.g. overriding multiProcessorCount with
                // hipDeviceAttributePhysicalMultiProcessorCount on multi-XCC
                // architectures) are preserved. The int-only overload re-queries
                // `hipGetDeviceProperties` and would discard those adjustments.
                //
                // Pass PCI chip ID when known (e.g. from hipDeviceAttributePciChipId upstream);
                // Origami does not query PCI chip ID itself—nullopt selects gfx950 id75a0.
                analyticalHardware
                    = std::make_shared<origami::hardware_t>(origami::hardware_t::
                                                                get_hardware_for_device(
                                                                    deviceId,
                                                                    analyticalProp,
                                                                    pciChipId));
            }
            numAIDs = partitionAIDs(deviceId, processor);
        }

        std::string HipAMDGPU::archName() const
        {
            return properties.gcnArchName;
        }

        std::shared_ptr<Hardware> GetCurrentDevice()
        {
            int deviceId = 0;
            HIP_CHECK_EXC(hipGetDevice(&deviceId));
            return GetDevice(deviceId);
        }

        std::shared_ptr<Hardware> GetDevice(int deviceId)
        {
            hipDeviceProp_t prop;
            HIP_CHECK_EXC(hipGetDeviceProperties(&prop, deviceId));
            int hip_version;
            HIP_CHECK_EXC(hipRuntimeGetVersion(&hip_version));
            if(hip_version >= 50220730)
            {
                HIP_CHECK_EXC(hipDeviceGetAttribute(&prop.multiProcessorCount,
                                                    hipDeviceAttributePhysicalMultiProcessorCount,
                                                    deviceId));
            }
            return GetDevice(prop, deviceId);
        }

        std::shared_ptr<Hardware> GetDevice(hipDeviceProp_t const& prop, int deviceId)
        {
            const auto processor = AMDGPU::toProcessor(prop.gcnArchName);
            if(!ChipIdRegistry::supportsChipIdPredicate(processor))
            {
                return std::make_shared<HipAMDGPU>(prop, deviceId);
            }

            int        pciChipId = 0;
            hipError_t chipIdResult
                = hipDeviceGetAttribute(&pciChipId, hipDeviceAttributePciChipId, deviceId);

            // Check hip runtime support for PCI Chip ID attribute
            if(chipIdResult == hipErrorInvalidValue)
                throw std::runtime_error(pciChipIdUnsupportedErrorMessage(prop));

            // For any other error, use standard error checking
            HIP_CHECK_EXC(chipIdResult);

            if(!ChipIdRegistry::isKnownChipId(pciChipId))
                logUnregisteredPciChipIdWarningOnce(prop, pciChipId);

            return std::make_shared<HipAMDGPU>(prop, deviceId, std::make_optional(pciChipId));
        }
    } // namespace hip
} // namespace TensileLite
