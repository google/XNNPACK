#include <cstddef>
#include <cstdint>
#include <fstream>
#include <set>
#include <string>

size_t detect_num_smcus() {
    // Determines the number of SMCUs using SMIDR_EL1 register exposed by the kernel.
    size_t cpu_idx = 0;

    size_t num_private_smcus = 0;
    std::set<uint32_t> shared_smcu_ids;

    while (true) {
        // Reads SMIDR_EL1 register.
        const std::string smidr_el1_path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu_idx) +
            "/regs/identification/smidr_el1";

        std::ifstream file(smidr_el1_path);

        if (!file.is_open()) {
            break;
        }

        uint64_t smidr_el1 = 0;
        file >> std::hex >> smidr_el1;

        // Checks whether the SMCU is shared and what is the affinity.
        // See also:
        // https://developer.arm.com/documentation/ddi0601/2025-09/AArch64-Registers/SMIDR-EL1--Streaming-Mode-Identification-Register

        const uint32_t sh = (smidr_el1 >> 13) & 0b11;
        const uint32_t affinity = (smidr_el1 & 0xFFF) | ((smidr_el1 >> 20) & 0xFFFFF000);

        switch (sh) {
            case 0b00:
                if (affinity == 0) {
                    ++num_private_smcus;
                } else {
                    shared_smcu_ids.emplace(affinity);
                }
                break;

            case 0b10:
                ++num_private_smcus;
                break;

            case 0b11:
                shared_smcu_ids.emplace(affinity);
                break;

            default:
                // SH = 0b01 is reserved.
                break;
        }

        ++cpu_idx;
    }

    const size_t num_smcus = num_private_smcus + shared_smcu_ids.size();

    return num_smcus;
}
