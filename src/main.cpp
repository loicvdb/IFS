#define VMA_IMPLEMENTATION
#include "vma/vk_mem_alloc.h"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"

#include <iostream>
#include <fstream>
#include <vector>

void checkResult(VkResult result)
{
    assert(result == VK_SUCCESS);
}

#define ASSERT_SUCCESS(res) checkResult(res)
#define SWAPCHAIN_BUFFER_COUNT 3

#define SHADER_PATH "../shaders/spv/"

VkPipeline loadPipeline(VkDevice device, VkPipelineLayout pipelineLayout, const char* path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);

    assert(file.is_open());

    size_t fileSize = (size_t) file.tellg();

    assert(fileSize % sizeof(uint32_t) == 0);

    std::vector<uint32_t> spv(fileSize / sizeof(uint32_t));

    file.seekg(0);
    file.read(reinterpret_cast<char*>(spv.data()), fileSize);
    file.close();

    VkShaderModuleCreateInfo shaderModuleCreateInfo{};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.pNext = nullptr;
    shaderModuleCreateInfo.flags = 0;
    shaderModuleCreateInfo.codeSize = static_cast<uint32_t>(spv.size() * sizeof(uint32_t));
    shaderModuleCreateInfo.pCode = spv.data();

    VkShaderModule shaderModule;
    ASSERT_SUCCESS(vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &shaderModule));

    VkPipelineShaderStageCreateInfo shaderStageCreateInfo{};
    shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.pNext = nullptr;
    shaderStageCreateInfo.flags = 0;
    shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCreateInfo.module = shaderModule;
    shaderStageCreateInfo.pName = "main";
    shaderStageCreateInfo.pSpecializationInfo = nullptr;

    VkComputePipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.pNext = nullptr;
    pipelineCreateInfo.flags = 0;
    pipelineCreateInfo.stage = shaderStageCreateInfo;
    pipelineCreateInfo.layout = pipelineLayout;
    pipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineCreateInfo.basePipelineIndex = 0;

    VkPipeline pipeline;
    ASSERT_SUCCESS(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline));

    vkDestroyShaderModule(device, shaderModule, nullptr);

    return pipeline;
}

struct Swapchain
{
    VkSwapchainKHR swapchain;
    VkImage colorImages[SWAPCHAIN_BUFFER_COUNT];
    VkImageView colorImageViews[SWAPCHAIN_BUFFER_COUNT];
    VkImage depthImage;
    VkImageView depthImageView;
    VmaAllocation depthImageAllocation;
};

Swapchain createSwapchain(VkDevice device, VmaAllocator allocator, VkSurfaceKHR surface, uint32_t width, uint32_t height)
{
    Swapchain swapchain{};

    VkExtent2D swapchainExtent{};
    swapchainExtent.width = width;
    swapchainExtent.height = height;

    VkSwapchainCreateInfoKHR swapchainCreateInfo{};
    swapchainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchainCreateInfo.pNext = nullptr;
    swapchainCreateInfo.flags = 0;
    swapchainCreateInfo.surface = surface;
    swapchainCreateInfo.minImageCount = SWAPCHAIN_BUFFER_COUNT;
    swapchainCreateInfo.imageFormat = VK_FORMAT_B8G8R8A8_UNORM;
    swapchainCreateInfo.imageColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
    swapchainCreateInfo.imageExtent = swapchainExtent;
    swapchainCreateInfo.imageArrayLayers = 1;
    swapchainCreateInfo.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT;
    swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchainCreateInfo.queueFamilyIndexCount = 0;
    swapchainCreateInfo.pQueueFamilyIndices = nullptr;
    swapchainCreateInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    swapchainCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchainCreateInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    swapchainCreateInfo.clipped = VK_TRUE;
    swapchainCreateInfo.oldSwapchain = VK_NULL_HANDLE;

    ASSERT_SUCCESS(vkCreateSwapchainKHR(device, &swapchainCreateInfo, nullptr, &swapchain.swapchain));

    uint32_t swapchainImageCount = SWAPCHAIN_BUFFER_COUNT;
    ASSERT_SUCCESS(vkGetSwapchainImagesKHR(device, swapchain.swapchain, &swapchainImageCount, swapchain.colorImages));

    assert(swapchainImageCount == SWAPCHAIN_BUFFER_COUNT);

    for (uint32_t i = 0; i < SWAPCHAIN_BUFFER_COUNT; i++)
    {
        VkImageSubresourceRange subresourceRange{};
        subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresourceRange.baseMipLevel = 0;
        subresourceRange.levelCount = 1;
        subresourceRange.baseArrayLayer = 0;
        subresourceRange.layerCount = 1;

        VkImageViewCreateInfo imageViewCreateInfo{};
        imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        imageViewCreateInfo.pNext = nullptr;
        imageViewCreateInfo.flags = 0;
        imageViewCreateInfo.image = swapchain.colorImages[i];
        imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        imageViewCreateInfo.format = VK_FORMAT_B8G8R8A8_UNORM;
        imageViewCreateInfo.components = {};
        imageViewCreateInfo.subresourceRange = subresourceRange;

        ASSERT_SUCCESS(vkCreateImageView(device, &imageViewCreateInfo, nullptr, &swapchain.colorImageViews[i]));
    }

    VkExtent3D depthImageExtent{};
    depthImageExtent.width = width;
    depthImageExtent.height = height;
    depthImageExtent.depth = 1;

    VkImageCreateInfo depthImageCreateInfo{};
    depthImageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    depthImageCreateInfo.pNext = nullptr;
    depthImageCreateInfo.flags = 0;
    depthImageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    depthImageCreateInfo.format = VK_FORMAT_R32_UINT;
    depthImageCreateInfo.extent = depthImageExtent;
    depthImageCreateInfo.mipLevels = 1;
    depthImageCreateInfo.arrayLayers = 1;
    depthImageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    depthImageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    depthImageCreateInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT;
    depthImageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    depthImageCreateInfo.queueFamilyIndexCount = 0;
    depthImageCreateInfo.pQueueFamilyIndices = nullptr;
    depthImageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocationCreateInfo{};
    allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    ASSERT_SUCCESS(vmaCreateImage(allocator, &depthImageCreateInfo, &allocationCreateInfo, &swapchain.depthImage, &swapchain.depthImageAllocation, nullptr));

    VkImageSubresourceRange subresourceRange{};
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = 1;
    subresourceRange.baseArrayLayer = 0;
    subresourceRange.layerCount = 1;

    VkImageViewCreateInfo depthImageViewCreateInfo{};
    depthImageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    depthImageViewCreateInfo.pNext = nullptr;
    depthImageViewCreateInfo.flags = 0;
    depthImageViewCreateInfo.image = swapchain.depthImage;
    depthImageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depthImageViewCreateInfo.format = VK_FORMAT_R32_UINT;
    depthImageViewCreateInfo.components = {};
    depthImageViewCreateInfo.subresourceRange = subresourceRange;

    ASSERT_SUCCESS(vkCreateImageView(device, &depthImageViewCreateInfo, nullptr, &swapchain.depthImageView));

    return swapchain;
}

void destroySwapchain(VkDevice device, VmaAllocator allocator, const Swapchain& swapchain)
{
    vkDestroyImageView(device, swapchain.depthImageView, nullptr);

    vmaDestroyImage(allocator, swapchain.depthImage, swapchain.depthImageAllocation);

    for (uint32_t i = 0; i < SWAPCHAIN_BUFFER_COUNT; i++)
    {
        vkDestroyImageView(device, swapchain.colorImageViews[i], nullptr);
    }

    vkDestroySwapchainKHR(device, swapchain.swapchain, nullptr);
}

int main()
{
    glfwInit();

    uint32_t instanceExtensionCount;
    const char** instanceExtensionNames = glfwGetRequiredInstanceExtensions(&instanceExtensionCount);

    VkApplicationInfo applicationInfo{};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pNext = nullptr;
    applicationInfo.pApplicationName = "IFS";
    applicationInfo.applicationVersion = VK_MAKE_API_VERSION(0, 0, 0, 0);
    applicationInfo.pEngineName = "";
    applicationInfo.engineVersion = VK_MAKE_API_VERSION(0, 0, 0, 0);
    applicationInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo instanceCreateInfo{};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pNext = nullptr;
    instanceCreateInfo.flags = 0;
    instanceCreateInfo.pApplicationInfo = &applicationInfo;
    instanceCreateInfo.enabledLayerCount = 0;
    instanceCreateInfo.ppEnabledLayerNames = nullptr;
    instanceCreateInfo.enabledExtensionCount = instanceExtensionCount;
    instanceCreateInfo.ppEnabledExtensionNames = instanceExtensionNames;

    VkInstance instance;
    ASSERT_SUCCESS(vkCreateInstance(&instanceCreateInfo, nullptr, &instance));

    VkPhysicalDevice physicalDevice;
    uint32_t physicalDeviceCount = 1;
    VkResult enumeratePhysicalDevicesRes = vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, &physicalDevice);

    assert(enumeratePhysicalDevicesRes == VK_SUCCESS || enumeratePhysicalDevicesRes == VK_INCOMPLETE);
    assert(physicalDeviceCount > 0);

    float queuePriority = 1.0f;

    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.pNext = nullptr;
    queueCreateInfo.flags = 0;
    queueCreateInfo.queueFamilyIndex = 0;   // should get the graphics/present queue on any GPU
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    const uint32_t deviceExtensionCount = 1;
    const char* deviceExtensionNames[deviceExtensionCount]
    {
        "VK_KHR_swapchain",
    };

    VkPhysicalDeviceFeatures enabledFeatures{};

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pNext = nullptr;
    deviceCreateInfo.flags = 0;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.enabledLayerCount = 0;
    deviceCreateInfo.ppEnabledLayerNames = nullptr;
    deviceCreateInfo.enabledExtensionCount = deviceExtensionCount;
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensionNames;
    deviceCreateInfo.pEnabledFeatures = &enabledFeatures;

    VkDevice device;
    ASSERT_SUCCESS(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));

    VkQueue queue;
    vkGetDeviceQueue(device, 0, 0, &queue);

    VmaAllocatorCreateInfo allocatorCreateInfo{};
    allocatorCreateInfo.instance = instance;
    allocatorCreateInfo.physicalDevice = physicalDevice;
    allocatorCreateInfo.device = device;
    allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_0;

    VmaAllocator allocator;
    ASSERT_SUCCESS(vmaCreateAllocator(&allocatorCreateInfo, &allocator));

    VkCommandPoolCreateInfo commandPoolCreateInfo{};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.pNext = nullptr;
    commandPoolCreateInfo.flags = 0;
    commandPoolCreateInfo.queueFamilyIndex = 0;

    VkCommandPool commandPool;
    ASSERT_SUCCESS(vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &commandPool));

    VkDescriptorSetLayoutBinding depthBinding{};
    depthBinding.binding = 0;
    depthBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    depthBinding.descriptorCount = 1;
    depthBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    depthBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutBinding colorBinding{};
    colorBinding.binding = 1;
    colorBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    colorBinding.descriptorCount = 1;
    colorBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    colorBinding.pImmutableSamplers = nullptr;

    const uint32_t bindingCount = 2;
    VkDescriptorSetLayoutBinding bindings[bindingCount]
    {
        depthBinding,
        colorBinding
    };

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{};
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.pNext = nullptr;
    descriptorSetLayoutCreateInfo.flags = 0;
    descriptorSetLayoutCreateInfo.bindingCount = bindingCount;
    descriptorSetLayoutCreateInfo.pBindings = bindings;
    
    VkDescriptorSetLayout descriptorSetLayout;
    ASSERT_SUCCESS(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayout));

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.pNext = nullptr;
    pipelineLayoutCreateInfo.flags = 0;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
    pipelineLayoutCreateInfo.pPushConstantRanges = nullptr;

    VkPipelineLayout pipelineLayout;
    ASSERT_SUCCESS(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

    VkPipeline clearPipeline = loadPipeline(device, pipelineLayout, SHADER_PATH "clear.comp.spv");
    VkPipeline splatPipeline = loadPipeline(device, pipelineLayout, SHADER_PATH "splat.comp.spv");
    VkPipeline displayPipeline = loadPipeline(device, pipelineLayout, SHADER_PATH "display.comp.spv");

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSize.descriptorCount = 2;

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo{};
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.pNext = nullptr;
    descriptorPoolCreateInfo.flags = 0;
    descriptorPoolCreateInfo.maxSets = 1;
    descriptorPoolCreateInfo.poolSizeCount = 1;
    descriptorPoolCreateInfo.pPoolSizes = &poolSize;

    VkDescriptorPool descriptorPool;
    vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &descriptorPool);

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.pNext = nullptr;
    descriptorSetAllocateInfo.descriptorPool = descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = 1;
    descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;

    VkDescriptorSet descriptorSet;
    vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet);

    uint32_t width = 800;
    uint32_t height = 600;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(width, height, "Vulkan window", nullptr, nullptr);

    VkSurfaceKHR surface;
    ASSERT_SUCCESS(glfwCreateWindowSurface(instance, window, nullptr, &surface));

    Swapchain swapchain = createSwapchain(device, allocator, surface, width, height);
    
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        VkSemaphoreCreateInfo semaphoreCreateInfo{};
        semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        semaphoreCreateInfo.pNext = nullptr;
        semaphoreCreateInfo.flags = 0;

        VkSemaphore acquireSemaphore;
        ASSERT_SUCCESS(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &acquireSemaphore));

        VkSemaphore presentSemaphore;
        ASSERT_SUCCESS(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &presentSemaphore));

        uint32_t imageIndex;
        VkResult acquireResult = vkAcquireNextImageKHR(device, swapchain.swapchain, UINT64_MAX, acquireSemaphore, nullptr, &imageIndex);

        assert(acquireResult == VK_SUCCESS || acquireResult == VK_ERROR_OUT_OF_DATE_KHR || acquireResult == VK_SUBOPTIMAL_KHR);

        VkDescriptorImageInfo depthInfo{};
        depthInfo.sampler = VK_NULL_HANDLE;
        depthInfo.imageView = swapchain.depthImageView;
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet depthWrite{};
        depthWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        depthWrite.pNext = nullptr;
        depthWrite.dstSet = descriptorSet;
        depthWrite.dstBinding = 0;
        depthWrite.dstArrayElement = 0;
        depthWrite.descriptorCount = 1;
        depthWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        depthWrite.pImageInfo = &depthInfo;
        depthWrite.pBufferInfo = nullptr;
        depthWrite.pTexelBufferView = nullptr;

        VkDescriptorImageInfo colorInfo{};
        colorInfo.sampler = VK_NULL_HANDLE;
        colorInfo.imageView = swapchain.colorImageViews[imageIndex];
        colorInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet colorWrite{};
        colorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        colorWrite.pNext = nullptr;
        colorWrite.dstSet = descriptorSet;
        colorWrite.dstBinding = 1;
        colorWrite.dstArrayElement = 0;
        colorWrite.descriptorCount = 1;
        colorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        colorWrite.pImageInfo = &colorInfo;
        colorWrite.pBufferInfo = nullptr;
        colorWrite.pTexelBufferView = nullptr;

        const uint32_t descriptorWriteCount = 2;
        VkWriteDescriptorSet descriptorWrites[descriptorWriteCount]
        {
            depthWrite,
            colorWrite
        };

        vkUpdateDescriptorSets(device, descriptorWriteCount, descriptorWrites, 0, nullptr);

        VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
        commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.pNext = nullptr;
        commandBufferAllocateInfo.commandPool = commandPool;
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        ASSERT_SUCCESS(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer));

        VkCommandBufferBeginInfo commandBufferBeginInfo{};
        commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        commandBufferBeginInfo.pNext = nullptr;
        commandBufferBeginInfo.flags = 0;
        commandBufferBeginInfo.pInheritanceInfo = nullptr;

        ASSERT_SUCCESS(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo));

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

        VkImageSubresourceRange wholeImageRange{};
        wholeImageRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        wholeImageRange.baseMipLevel = 0;
        wholeImageRange.levelCount = 1;
        wholeImageRange.baseArrayLayer = 0;
        wholeImageRange.layerCount = 1;

        VkImageMemoryBarrier depthToComputeBarrier{};
        depthToComputeBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        depthToComputeBarrier.pNext = nullptr;
        depthToComputeBarrier.srcAccessMask = 0;
        depthToComputeBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        depthToComputeBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthToComputeBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        depthToComputeBarrier.srcQueueFamilyIndex = 0;
        depthToComputeBarrier.dstQueueFamilyIndex = 0;
        depthToComputeBarrier.image = swapchain.depthImage;
        depthToComputeBarrier.subresourceRange = wholeImageRange;

        VkImageMemoryBarrier colorToComputeBarrier{};
        colorToComputeBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        colorToComputeBarrier.pNext = nullptr;
        colorToComputeBarrier.srcAccessMask = 0;
        colorToComputeBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        colorToComputeBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorToComputeBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        colorToComputeBarrier.srcQueueFamilyIndex = 0;
        colorToComputeBarrier.dstQueueFamilyIndex = 0;
        colorToComputeBarrier.image = swapchain.colorImages[imageIndex];
        colorToComputeBarrier.subresourceRange = wholeImageRange;

        VkImageMemoryBarrier depthComputeToComputeBarrier{};
        depthComputeToComputeBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        depthComputeToComputeBarrier.pNext = nullptr;
        depthComputeToComputeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        depthComputeToComputeBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        depthComputeToComputeBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        depthComputeToComputeBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        depthComputeToComputeBarrier.srcQueueFamilyIndex = 0;
        depthComputeToComputeBarrier.dstQueueFamilyIndex = 0;
        depthComputeToComputeBarrier.image = swapchain.depthImage;
        depthComputeToComputeBarrier.subresourceRange = wholeImageRange;

        VkImageMemoryBarrier colorComputeToPresentBarrier{};
        colorComputeToPresentBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        colorComputeToPresentBarrier.pNext = nullptr;
        colorComputeToPresentBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        colorComputeToPresentBarrier.dstAccessMask = 0;
        colorComputeToPresentBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        colorComputeToPresentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        colorComputeToPresentBarrier.srcQueueFamilyIndex = 0;
        colorComputeToPresentBarrier.dstQueueFamilyIndex = 0;
        colorComputeToPresentBarrier.image = swapchain.colorImages[imageIndex];
        colorComputeToPresentBarrier.subresourceRange = wholeImageRange;

        const uint32_t imageMemoryBarrierCount = 2;
        VkImageMemoryBarrier imageMemoryBarriers[imageMemoryBarrierCount]
        {
            depthToComputeBarrier,
            colorToComputeBarrier
        };

        // transition depth/color image
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, imageMemoryBarrierCount, imageMemoryBarriers);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, clearPipeline);
        vkCmdDispatch(commandBuffer, (width + 15u) / 16u, (height + 15u) / 16u, 1);

        // guard depth writes against reads/writes
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &depthComputeToComputeBarrier);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, splatPipeline);
        vkCmdDispatch(commandBuffer, (1162261467u + 255u) / 256u, 1, 1);

        // guard depth writes against reads
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &depthComputeToComputeBarrier);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, displayPipeline);
        vkCmdDispatch(commandBuffer, (width + 15u) / 16u, (height + 15u) / 16u, 1);

        // transition color image
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &colorComputeToPresentBarrier);

        ASSERT_SUCCESS(vkEndCommandBuffer(commandBuffer));

        VkPipelineStageFlags waitDstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.pNext = nullptr;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &acquireSemaphore;
        submitInfo.pWaitDstStageMask = &waitDstStageMask;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &presentSemaphore;

        ASSERT_SUCCESS(vkQueueSubmit(queue, 1, &submitInfo, nullptr));

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.pNext = nullptr;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &presentSemaphore;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain.swapchain;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr;

        VkResult presentResult = vkQueuePresentKHR(queue, &presentInfo);

        ASSERT_SUCCESS(vkQueueWaitIdle(queue));

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        vkDestroySemaphore(device, presentSemaphore, nullptr);
        vkDestroySemaphore(device, acquireSemaphore, nullptr);

        if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR)
        {
            destroySwapchain(device, allocator, swapchain);

            int newWidth, newHeight;
            glfwGetWindowSize(window, &newWidth, &newHeight);

            width = static_cast<uint32_t>(newWidth);
            height = static_cast<uint32_t>(newHeight);

            swapchain = createSwapchain(device, allocator, surface, width, height);
        }
        else
        {
            ASSERT_SUCCESS(presentResult);
        }
    }

    destroySwapchain(device, allocator, swapchain);

    vkDestroySurfaceKHR(instance, surface, nullptr);

    vkDestroyDescriptorPool(device, descriptorPool, nullptr);

    vkDestroyPipeline(device, displayPipeline, nullptr);
    vkDestroyPipeline(device, splatPipeline, nullptr);
    vkDestroyPipeline(device, clearPipeline, nullptr);

    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

    vkDestroyCommandPool(device, commandPool, nullptr);

    vmaDestroyAllocator(allocator);

    vkDestroyDevice(device, nullptr);

    vkDestroyInstance(instance, nullptr);

    glfwDestroyWindow(window);

    glfwTerminate();

	return 0;
}
