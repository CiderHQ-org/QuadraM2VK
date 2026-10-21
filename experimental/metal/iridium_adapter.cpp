// SPDX-License-Identifier: MIT
/* Independent adapter between validated metal2vulkan cache packets and
 * licensed Indium. Shader translation runs in the offline cache builder. */
#include <cstdint>
#include <iridium/iridium.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <set>
#include <vector>

extern "C" unsigned char *CC_SHA256(const void *, unsigned int, unsigned char *);
bool Iridium::init() { return true; }
void Iridium::finit() {}

namespace {
struct PacketReader {
    const std::vector<unsigned char> &packet;
    size_t position = 0;
    const unsigned char *take(size_t length) {
        if (position > packet.size() || length > packet.size() - position)
            throw std::runtime_error("Truncated native Metal shader packet");
        auto data = packet.data() + position;
        position += length;
        return data;
    }
    uint32_t integer() {
        const auto p = take(4);
        return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
    }
};

void checkEntryPoint(const unsigned char *spirv, size_t length, uint32_t stage,
                     const std::string& name) {
    auto word = [&](size_t index) {
        auto p = spirv + index * 4;
        return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
    };
    size_t count = length / 4, position = 5;
    uint32_t entryId = 0;
    std::set<uint32_t> functions;
    while (position < count) {
        auto instruction = word(position);
        size_t size = instruction >> 16;
        auto opcode = instruction & 0xffff;
        if (!size || size > count - position)
            throw std::runtime_error("Malformed native Metal SPIR-V instruction");
        if (opcode == 15) { // OpEntryPoint
            if (size < 4 || entryId || word(position + 1) != (stage == 1 ? 0u : 4u))
                throw std::runtime_error("Invalid native Metal SPIR-V entry point");
            auto raw = spirv + (position + 3) * 4;
            size_t available = (size - 3) * 4;
            auto end = static_cast<const unsigned char *>(std::memchr(raw, 0, available));
            if (!end || size_t(end - raw) != name.size() || std::memcmp(raw, name.data(), name.size()))
                throw std::runtime_error("Native Metal SPIR-V entry-point name mismatch");
            entryId = word(position + 2);
        } else if (opcode == 54 && size == 5) { // OpFunction
            functions.insert(word(position + 2));
        }
        position += size;
    }
    if (!entryId || entryId >= word(3) || !functions.count(entryId))
        throw std::runtime_error("Native Metal SPIR-V has no matching shader function");
}
}

void *Iridium::translate(const void *source, size_t length, size_t &size, OutputInfo &output) {
    size = 0;
    output.functionInfos.clear();
    const char *cache = std::getenv("MACOBLOX_METAL_SHADER_CACHE");
    if (!cache || !source || length > std::numeric_limits<unsigned int>::max())
        throw std::runtime_error("Native Metal requires a validated shader cache");
    unsigned char digest[32];
    CC_SHA256(source, static_cast<unsigned int>(length), digest);
    char hex[65];
    for (size_t i = 0; i < 32; ++i)
        std::snprintf(hex + i * 2, 3, "%02x", digest[i]);
    auto path = std::string(cache) + "/" + hex + ".mvk";
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
        throw std::runtime_error("Native Metal shader cache miss: " + std::string(hex));
    auto packetSize = stream.tellg();
    if (packetSize < 24 || packetSize > 128 * 1024 * 1024)
        throw std::runtime_error("Invalid native Metal shader packet size");
    stream.seekg(0);
    std::vector<unsigned char> packet(static_cast<size_t>(packetSize));
    if (!stream.read(reinterpret_cast<char *>(packet.data()), packet.size()))
        throw std::runtime_error("Cannot read native Metal shader packet");
    PacketReader reader {packet};
    if (std::memcmp(reader.take(4), "MVK1", 4) || reader.integer() != 1)
        throw std::runtime_error("Unsupported native Metal shader packet version");
    auto stage = reader.integer(), nameSize = reader.integer();
    auto bindingCount = reader.integer(), spirvSize = reader.integer();
    if (stage < 1 || stage > 2 || !nameSize || nameSize > 1024 ||
        bindingCount > 1024 || spirvSize < 20 || spirvSize % 4)
        throw std::runtime_error("Invalid native Metal shader metadata");
    auto rawName = reader.take(nameSize);
    std::string name(reinterpret_cast<const char *>(rawName), nameSize);
    if (name.find('\0') != std::string::npos)
        throw std::runtime_error("Invalid native Metal entry-point name");
    FunctionInfo info {};
    info.type = static_cast<FunctionType>(stage);
    std::set<uint32_t> occupied;
    for (uint32_t i = 0; i < bindingCount; ++i) {
        auto type = reader.integer(), index = reader.integer();
        auto internal = reader.integer(), access = reader.integer();
        if ((type != 1 && type != 2 && type != 4) || access > 3 ||
            index > 127 || internal > 4095 || !occupied.insert(internal).second)
            throw std::runtime_error("Unsupported native Metal resource binding");
        BindingInfo binding {};
        binding.type = static_cast<BindingType>(type);
        binding.index = index;
        binding.internalIndex = internal;
        binding.textureAccessType = static_cast<TextureAccessType>(access);
        info.bindings.push_back(binding);
    }
    auto spirv = reader.take(spirvSize);
    if (reader.position != packet.size() ||
        (uint32_t(spirv[0]) | uint32_t(spirv[1]) << 8 | uint32_t(spirv[2]) << 16 | uint32_t(spirv[3]) << 24) != 0x07230203)
        throw std::runtime_error("Invalid native Metal SPIR-V payload");
    checkEntryPoint(spirv, spirvSize, stage, name);
    auto result = std::malloc(spirvSize);
    if (!result)
        throw std::bad_alloc();
    std::memcpy(result, spirv, spirvSize);
    output.functionInfos.emplace(name, std::move(info));
    size = spirvSize;
    return result;
}
