// Packs a GPT-Neo checkpoint into the blob the firmware runs:
//   pack_model --model DIR [--precision fp32|w8a8] OUT
// DIR holds a Hugging Face config.json and model.safetensors. The host
// writes OUT to memory the core can read and points the mailbox at it
// (coral/include/reef_kernels/host_if.hpp).
#include "pack_model.hpp"
#include "inference_engine/weights.hpp"
#include "reef_kernels/model_blob.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr const char* kUsage =
    "usage: pack_model --model DIR [--precision fp32|w8a8] OUT\n";

} // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    std::optional<std::string> model_dir;
    std::optional<std::string> out;
    std::string precision = "fp32";
    for (std::size_t index = 0; index < args.size(); ++index) {
        const std::string& arg = args[index];
        const bool has_value = index + 1 < args.size();
        if (arg == "--model" && has_value) {
            model_dir = args[++index];
        } else if (arg == "--precision" && has_value) {
            precision = args[++index];
        } else if (!out) {
            out = arg;
        } else {
            std::cerr << kUsage;
            return 1;
        }
    }
    if (!model_dir || !out || (precision != "fp32" && precision != "w8a8")) {
        std::cerr << kUsage;
        return 1;
    }

    try {
        inference_engine::ModelWeights<float> weights =
            inference_engine::load_gpt_neo(*model_dir);
        auto packed_precision = reef_kernels::Precision::kFloat32;
        if (precision == "w8a8") {
            inference_engine::quantize_w8a8(weights);
            packed_precision = reef_kernels::Precision::kW8A8;
        }
        const std::vector<std::uint8_t> blob =
            coral_host::pack_gpt_neo(weights, packed_precision);
        std::ofstream file(*out, std::ios::binary);
        file.write(reinterpret_cast<const char*>(blob.data()),
                   static_cast<std::streamsize>(blob.size()));
        if (!file) {
            std::cerr << "cannot write " << *out << '\n';
            return 1;
        }
        std::cout << "wrote " << blob.size() << " bytes to " << *out << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
