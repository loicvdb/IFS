#pragma once

#include <cstdint>

struct Handle
{
    uint64_t handle;
};

// this sucks, can't == with a handle and it's just generally ugly
#define NULL_HANDLE { 0 }

struct Image : Handle {};
struct Buffer : Handle {};
struct PipelineLayout : Handle {};
struct DescriptorPool : Handle {};
struct Pipeline : Handle {};
struct Fence : Handle {};
struct Swapchain : Handle {};
struct Surface : Handle {};

enum struct DescriptorType
{
    UniformBuffer,
    StorageImage
};

struct Descriptor
{
    DescriptorType descriptorType;
    Image image;
    Buffer buffer;
};

struct CommandBuffer : Handle
{
    void begin();
    void end();
    void bindDescriptorSet(PipelineLayout pipelineLayout, DescriptorPool descriptorPool, uint32_t descriptorIndex);
    void bindPipeline(Pipeline pipeline);
    void dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ);
    void barrier();
    void memoryBarrier();
};

enum struct Format
{
    r32_uint,
    r8g8b8a8_unorm
};

enum struct SwapchainResult
{
    Success,
    OutOfDate,
    Suboptimal
};

struct Device : Handle
{
    [[nodiscard]]
    PipelineLayout createPipelineLayout(uint32_t descriptorCount, const DescriptorType* descriptorTypes);
    void destroyPipelineLayout(PipelineLayout pipelineLayout);

    [[nodiscard]]
    DescriptorPool createDescriptorPool(PipelineLayout pipelineLayout, uint32_t descriptorSetCout);
    void destroyDescriptorPool(DescriptorPool descriptorPool);

    [[nodiscard]]
    Pipeline createPipeline(PipelineLayout pipelineLayout, uint64_t codeSize, const uint32_t* code);
    void destroyPipeline(Pipeline pipeline);

    // maybe check if the descriptor is in use
    void updateDescriptorSet(DescriptorPool descriptorPool, uint32_t descriptorSetIndex, uint32_t descriptorCount, const Descriptor* descriptors);

    // maybe replace with a raw pointer ?
    // maybe check if the buffer is in use
    void udpateBuffer(Buffer buffer, uint64_t dataOffset, uint64_t dataSize, const void* data);

    [[nodiscard]]
    Swapchain createSwapchain(uint32_t width, uint32_t height, Surface surface);
    void destroySwapchain(Swapchain swapchain);

    [[nodiscard]]
    Image createImage(uint32_t width, uint32_t height, Format format);
    void destroyImage(Image image);

    [[nodiscard]]
    Buffer createBuffer(uint64_t size);
    void destroyBuffer(Buffer buffer);

    [[nodiscard]]
    CommandBuffer createCommandBuffer();
    void destroyCommandBuffer(CommandBuffer commandBuffer);

    [[nodiscard]]
    Fence createFence();
    void destroyFence(Fence fence);

    void submit(CommandBuffer commandBuffer, Fence fence);

    void waitForFence(Fence fence);

    SwapchainResult acquireSwapchainImage(Swapchain swapchain, Image* image);
    SwapchainResult presentSwapchainImage(Swapchain swapchain, Image image);
};

[[nodiscard]]
Device createDevice();
void destroyDevice(Device device);
