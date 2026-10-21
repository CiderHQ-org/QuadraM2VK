// SPDX-License-Identifier: MIT
#import <Metal/MTLDeviceInternal.h>
#include <indium/indium.hpp>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

static void step(const char *message) { std::fprintf(stderr, "%s\n", message); }
static void leave(int code) {
    __asm__ volatile("syscall" : : "a"(231L), "D"((long)code) : "rcx", "r11", "memory");
}
static std::vector<char> read(const std::string &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if(!file) throw std::runtime_error("Cannot read shader " + path);
    std::vector<char> data(static_cast<size_t>(file.tellg()));
    file.seekg(0); file.read(data.data(), data.size());
    if(!file) throw std::runtime_error("Cannot load shader " + path);
    return data;
}
int main(int argc, char **argv) {
    if(argc != 3) return 2;
    @autoreleasepool {
        try {
            auto metal = (MTLDeviceInternal *)MTLCreateSystemDefaultDevice();
            auto device = metal.device;
            if(!device) throw std::runtime_error("No native Metal device");
            std::fprintf(stderr, "Native pipeline: Vulkan device %s\n", device->name().c_str());
            auto vertexBytes = read(argv[1]), fragmentBytes = read(argv[2]);
            auto vertexLibrary = device->newLibrary(vertexBytes.data(), vertexBytes.size());
            auto fragmentLibrary = device->newLibrary(fragmentBytes.data(), fragmentBytes.size());
            step("Native pipeline: loaded validated current Roblox Metal shaders");
            Indium::RenderPipelineDescriptor descriptor {};
            descriptor.vertexFunction = vertexLibrary->newFunction("main0");
            descriptor.fragmentFunction = fragmentLibrary->newFunction("main0");
            descriptor.rasterSampleCount = 1;
            descriptor.inputPrimitiveTopology = Indium::PrimitiveTopologyClass::Triangle;
            descriptor.colorAttachments.resize(1);
            descriptor.colorAttachments[0].pixelFormat = Indium::PixelFormat::RGBA8Unorm;
            Indium::VertexDescriptor vertex {};
            vertex.layouts[3].stride = 8;
            vertex.attributes[0].format = Indium::VertexFormat::Int2;
            vertex.attributes[0].bufferIndex = 3;
            descriptor.vertexDescriptor = vertex;
            auto pipeline = device->newRenderPipelineState(descriptor);

            Indium::TextureDescriptor image {};
            image.width = 16; image.height = 16;
            image.pixelFormat = Indium::PixelFormat::RGBA8Unorm;
            image.resourceOptions = Indium::ResourceOptions::StorageModePrivate;
            image.usage = Indium::TextureUsage::RenderTarget;
            auto target = device->newTexture(image);
            auto queue = device->newCommandQueue();
            auto commands = queue->commandBuffer();
            Indium::RenderPassDescriptor pass {};
            pass.colorAttachments.resize(1);
            pass.colorAttachments[0].texture = target;
            pass.colorAttachments[0].loadAction = Indium::LoadAction::Clear;
            pass.colorAttachments[0].storeAction = Indium::StoreAction::Store;
            pass.colorAttachments[0].clearColor = {0, 0, 1, 1};
            auto encoder = commands->renderCommandEncoder(pass);
            encoder->setRenderPipelineState(pipeline);
            float cb0[4] = {1, 1, 0, 1};
            unsigned int zeros[1024] = {};
            float cb13[64] = {}; cb13[0] = 1;
            int indices[8] = {0, 0, 1, 0, 3, 0, 2, 0};
            encoder->setVertexBytes(cb0, sizeof(cb0), 0);
            encoder->setVertexBytes(zeros, sizeof(zeros), 1);
            encoder->setVertexBytes(zeros, sizeof(zeros), 2);
            encoder->setVertexBytes(indices, sizeof(indices), 3);
            encoder->setFragmentBytes(cb13, sizeof(cb13), 13);
            encoder->drawPrimitives(Indium::PrimitiveType::TriangleStrip, 0, 4);
            encoder->endEncoding();
            step("Native pipeline: graphics pipeline created and real draw encoded");
            auto readback = device->newBuffer(16 * 16 * 4, Indium::ResourceOptions::StorageModeShared);
            auto blit = commands->blitCommandEncoder();
            blit->copy(target, 0, 0, {0, 0, 0}, {16, 16, 1}, readback, 0, 16 * 4, 16 * 16 * 4);
            blit->endEncoding();
            commands->commit();
            commands->waitUntilCompleted();
            step("Native pipeline: Vulkan draw and image readback completed");
            auto pixels = static_cast<const unsigned char *>(readback->contents());
            unsigned int red = 0;
            for(unsigned int i = 0; i < 256; ++i)
                red += pixels[i * 4] == 255 && pixels[i * 4 + 1] == 0 &&
                       pixels[i * 4 + 2] == 0 && pixels[i * 4 + 3] == 255;
            std::fprintf(stderr, "Native pipeline: %u / 256 expected red pixels\n", red);
            if(red != 256) throw std::runtime_error("Client shaders did not produce expected GPU pixels");
            step("Native pipeline: PASS real client shaders rendered the expected red quad");
            leave(0);
        } catch(const std::exception &error) {
            std::fprintf(stderr, "Native pipeline: FAIL %s\n", error.what());
            leave(1);
        }
    }
    return 0;
}
