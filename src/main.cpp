#include "abstraction_glfw.h"
#include "glm/glm.hpp"

#define GLM_FORCE_LEFT_HANDED
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#include "glm/gtc/matrix_transform.hpp"

#include <fstream>
#include <vector>
#include <chrono>
#include <thread>

#define SHADER_PATH "./shaders/"

// uses fp units, a bit hacky
#define UINT_POW(x, y) static_cast<uint32_t>(round(pow(float(x), float(y))))

// 3^17 => 129m points
#define FRACTAL_ITERATIONS 17

#define WORK_IN_FLIGHT_COUNT 2

uint32_t fractSeed = 0;

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_LEFT && action == GLFW_PRESS)
    {
        fractSeed--;
    }
    if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS)
    {
        fractSeed++;
    }
}

glm::mat4 randomTransform(uint32_t seed)
{
    float scaleX      = glm::mix(0.5f, 0.8f, float(0xa265c5cfu * seed) / float(0xFFFFFFFFu));
    float scaleY      = glm::mix(0.5f, 0.8f, float(0x531680cbu * seed) / float(0xFFFFFFFFu));
    float scaleZ      = glm::mix(0.5f, 0.8f, float(0x69f2cf94u * seed) / float(0xFFFFFFFFu));
    float rotateAngle = glm::mix(-3.f, 3.0f, float(0x5153c709u * seed) / float(0xFFFFFFFFu));
    float rotateVecX  = glm::mix(-1.f, 1.0f, float(0x288ef917u * seed) / float(0xFFFFFFFFu));
    float rotateVecY  = glm::mix(-1.f, 1.0f, float(0xbe62e7f9u * seed) / float(0xFFFFFFFFu));
    float rotateVecZ  = glm::mix(-1.f, 1.0f, float(0xb0260e61u * seed) / float(0xFFFFFFFFu));
    float translateX  = glm::mix(-1.f, 1.0f, float(0x73eddacbu * seed) / float(0xFFFFFFFFu));
    float translateY  = glm::mix(-1.f, 1.0f, float(0x936f427eu * seed) / float(0xFFFFFFFFu));
    float translateZ  = glm::mix(-1.f, 1.0f, float(0xf34f753au * seed) / float(0xFFFFFFFFu));

    glm::mat4 translate = glm::translate(glm::mat4(1.0f), glm::vec3(translateX, translateY, translateZ));
    glm::mat4 rotate = glm::rotate(glm::mat4(1.0f), rotateAngle, glm::normalize(glm::vec3(rotateVecX, rotateVecY, rotateVecZ)));
    glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(scaleX, scaleY, scaleZ));

    return scale * rotate * translate;
}

Pipeline loadPipeline(Device device, PipelineLayout pipelineLayout, const char* path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);

    assert(file.is_open());

    size_t fileSize = static_cast<size_t>(file.tellg());

    assert(fileSize % sizeof(uint32_t) == 0);

    std::vector<uint32_t> spv(fileSize / sizeof(uint32_t));

    file.seekg(0);
    file.read(reinterpret_cast<char*>(spv.data()), fileSize);
    file.close();

    return device.createPipeline(pipelineLayout, spv.size() * sizeof(uint32_t), spv.data());
}

struct FrameData
{
    glm::mat4 matrices[3];
    glm::mat4 view;
    glm::vec4 color0;
    glm::vec4 color1;
    glm::float32 fov;
    glm::float32 focalPlane;
    glm::float32 aperture;
    glm::float32 exposure;
    glm::uint iterationCount;
};

int main()
{
    glfwInit();

    Device device = createDevice();

    const uint32_t descriptorCount = 3;
    DescriptorType descriptorTypes[descriptorCount]
    {
        DescriptorType::StorageImage,
        DescriptorType::StorageImage,
        DescriptorType::UniformBuffer
    };

    PipelineLayout pipelineLayout = device.createPipelineLayout(descriptorCount, descriptorTypes);

    Pipeline clearPipeline = loadPipeline(device, pipelineLayout, SHADER_PATH "clear.comp.spv");
    Pipeline splatPipeline = loadPipeline(device, pipelineLayout, SHADER_PATH "splat.comp.spv");
    Pipeline displayPipeline = loadPipeline(device, pipelineLayout, SHADER_PATH "display.comp.spv");

    uint32_t width = 1280;
    uint32_t height = 720;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(width, height, "Vulkan window", nullptr, nullptr);

    glfwSetKeyCallback(window, key_callback);
    
    Surface surface = createGLFWSurface(device, window);
    Swapchain swapchain = device.createSwapchain(width, height, surface);

    DescriptorPool descriptorPool = device.createDescriptorPool(pipelineLayout, WORK_IN_FLIGHT_COUNT);

    uint32_t workInFlightIndex = 0;

    Image atomicImages[WORK_IN_FLIGHT_COUNT];
    Buffer frameDataBuffers[WORK_IN_FLIGHT_COUNT];
    CommandBuffer commandBuffers[WORK_IN_FLIGHT_COUNT];
    Fence fences[WORK_IN_FLIGHT_COUNT];

    for (uint32_t i = 0; i < WORK_IN_FLIGHT_COUNT; i++)
    {
        atomicImages[i] = device.createImage(width, height, Format::r32_uint);
        frameDataBuffers[i] = device.createBuffer(sizeof(FrameData));
        commandBuffers[i] = NULL_HANDLE;
        fences[i] = NULL_HANDLE;
    }

    std::chrono::steady_clock::time_point startTime = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        Image swapchainImage;
        device.acquireSwapchainImage(swapchain, &swapchainImage);

        if (fences[workInFlightIndex].handle != 0)
        {
            device.waitForFence(fences[workInFlightIndex]);
            device.destroyCommandBuffer(commandBuffers[workInFlightIndex]);
            device.destroyFence(fences[workInFlightIndex]);
        }

        float angle = 6.2831855f * static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - startTime).count() & 0xFFFFFFFF) / float(0xFFFFFFFF);

        glm::mat4 view = glm::rotate(glm::translate(glm::mat4(1.0f), glm::vec3(0.0, 0.0, -15.0)), angle, glm::vec3(0.0f, 1.0f, 0.0f));

        FrameData frameData{};
        frameData.matrices[0] = randomTransform(1 + 3 * fractSeed);
        frameData.matrices[1] = randomTransform(2 + 3 * fractSeed);
        frameData.matrices[2] = randomTransform(3 + 3 * fractSeed);
        frameData.color0 = glm::vec4(0.9, 0.0, 0.7, 0.0);
        frameData.color1 = glm::vec4(0.0, 0.7, 0.9, 0.0);
        frameData.view = view;
        frameData.fov = 0.1f;
        frameData.focalPlane = 14.75f;
        frameData.aperture = 0.03f;
        frameData.exposure = 500.0f / pow(3.0f, static_cast<float>(FRACTAL_ITERATIONS));
        frameData.iterationCount = FRACTAL_ITERATIONS - 3;

        device.udpateBuffer(frameDataBuffers[workInFlightIndex], 0, sizeof(FrameData), &frameData);

        Descriptor atomicImageDescriptor{};
        atomicImageDescriptor.descriptorType = DescriptorType::StorageImage;
        atomicImageDescriptor.image = atomicImages[workInFlightIndex];
        atomicImageDescriptor.buffer = NULL_HANDLE;

        Descriptor swapchainImageDescriptor{};
        swapchainImageDescriptor.descriptorType = DescriptorType::StorageImage;
        swapchainImageDescriptor.image = swapchainImage;
        swapchainImageDescriptor.buffer = NULL_HANDLE;

        Descriptor frameDataBufferDescriptor{};
        frameDataBufferDescriptor.descriptorType = DescriptorType::UniformBuffer;
        frameDataBufferDescriptor.image = NULL_HANDLE;
        frameDataBufferDescriptor.buffer = frameDataBuffers[workInFlightIndex];

        const uint32_t descriptorCount = 3;
        Descriptor descriptors[descriptorCount]
        {
            atomicImageDescriptor,
            swapchainImageDescriptor,
            frameDataBufferDescriptor
        };

        device.updateDescriptorSet(descriptorPool, workInFlightIndex, descriptorCount, descriptors);

        CommandBuffer commandBuffer = device.createCommandBuffer();

        commandBuffer.begin();

        commandBuffer.bindDescriptorSet(pipelineLayout, descriptorPool, workInFlightIndex);
        commandBuffer.bindPipeline(clearPipeline);
        commandBuffer.dispatch((width + 15) / 16, (height + 15) / 16, 1);
        commandBuffer.memoryBarrier();
        commandBuffer.bindPipeline(splatPipeline);
        commandBuffer.dispatch((UINT_POW(3, FRACTAL_ITERATIONS - 3) + 255u) / 256u, 1, 1);
        commandBuffer.memoryBarrier();
        commandBuffer.bindPipeline(displayPipeline);
        commandBuffer.dispatch((width + 15) / 16, (height + 15) / 16, 1);

        commandBuffer.end();

        Fence fence = device.createFence();
        device.submit(commandBuffer, fence);

        SwapchainResult presentResult = device.presentSwapchainImage(swapchain, swapchainImage);

        fences[workInFlightIndex] = fence;
        commandBuffers[workInFlightIndex] = commandBuffer;
        workInFlightIndex = (workInFlightIndex + 1) % WORK_IN_FLIGHT_COUNT;

        if (presentResult != SwapchainResult::Success)
        {
            for (uint32_t i = 0; i < WORK_IN_FLIGHT_COUNT; i++)
            {
                if (fences[i].handle != 0)
                {
                    device.waitForFence(fences[i]);
                }

                device.destroyImage(atomicImages[i]);
                device.destroyBuffer(frameDataBuffers[i]);
            }

            device.destroySwapchain(swapchain);

            int newWidth = 0;
            int newHeight = 0;

            while ((newWidth == 0 || newHeight == 0) && !glfwWindowShouldClose(window))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                glfwPollEvents();
                glfwGetWindowSize(window, &newWidth, &newHeight);
            }

            width = static_cast<uint32_t>(newWidth);
            height = static_cast<uint32_t>(newHeight);

            swapchain = device.createSwapchain(width, height, surface);

            for (uint32_t i = 0; i < WORK_IN_FLIGHT_COUNT; i++)
            {
                atomicImages[i] = device.createImage(width, height, Format::r32_uint);
                frameDataBuffers[i] = device.createBuffer(sizeof(FrameData));
            }
        }
    }

    for (uint32_t i = 0; i < WORK_IN_FLIGHT_COUNT; i++)
    {
        if (fences[i].handle != 0)
        {
            device.waitForFence(fences[i]);
            device.destroyCommandBuffer(commandBuffers[i]);
            device.destroyFence(fences[i]);
        }

        device.destroyImage(atomicImages[i]);
        device.destroyBuffer(frameDataBuffers[i]);
    }

    device.destroyDescriptorPool(descriptorPool);

    device.destroySwapchain(swapchain);

    destroyGLFWSurface(device, surface);

    device.destroyPipeline(displayPipeline);
    device.destroyPipeline(splatPipeline);
    device.destroyPipeline(clearPipeline);

    device.destroyPipelineLayout(pipelineLayout);

    destroyDevice(device);

    glfwTerminate();

    return 0;
}
