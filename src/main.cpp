#define VMA_IMPLEMENTATION
#include "vma/vk_mem_alloc.h"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"

#define GLM_FORCE_LEFT_HANDED
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#include "glm/gtc/matrix_transform.hpp"

#include <iostream>
#include <fstream>
#include <vector>

void checkResult(VkResult result)
{
    assert(result == VK_SUCCESS);
}

#define ASSERT_SUCCESS(res) checkResult(res)
#define SWAPCHAIN_BUFFER_COUNT 3

#define SHADER_PATH "./shaders/"

// uses fp units, a bit hacky
#define UINT_POW(x, y) static_cast<uint32_t>(round(pow(float(x), float(y))))

// 3^17 => 129m points
#define FRACTAL_ITERATIONS 17

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
    float scaleX      = glm::mix(0.6f, 0.8f, float(0xa265c5cfu * seed) / float(0xFFFFFFFFu));
    float scaleY      = glm::mix(0.6f, 0.8f, float(0x531680cbu * seed) / float(0xFFFFFFFFu));
    float scaleZ      = glm::mix(0.6f, 0.8f, float(0x69f2cf94u * seed) / float(0xFFFFFFFFu));
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
    VkImage depthImages[SWAPCHAIN_BUFFER_COUNT];
    VkImageView depthImageViews[SWAPCHAIN_BUFFER_COUNT];
    VmaAllocation depthImageAllocations[SWAPCHAIN_BUFFER_COUNT];
};

Swapchain createSwapchain(VkDevice device, VmaAllocator allocator, VkSurfaceKHR surface, VkPresentModeKHR presentMode, uint32_t width, uint32_t height)
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
    swapchainCreateInfo.presentMode = presentMode;
    swapchainCreateInfo.clipped = VK_TRUE;
    swapchainCreateInfo.oldSwapchain = VK_NULL_HANDLE;

    ASSERT_SUCCESS(vkCreateSwapchainKHR(device, &swapchainCreateInfo, nullptr, &swapchain.swapchain));

    uint32_t swapchainImageCount = SWAPCHAIN_BUFFER_COUNT;
    ASSERT_SUCCESS(vkGetSwapchainImagesKHR(device, swapchain.swapchain, &swapchainImageCount, swapchain.colorImages));

    assert(swapchainImageCount == SWAPCHAIN_BUFFER_COUNT);

    for (uint32_t i = 0; i < SWAPCHAIN_BUFFER_COUNT; i++)
    {
        VkImageSubresourceRange wholeSubresourceRange{};
        wholeSubresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        wholeSubresourceRange.baseMipLevel = 0;
        wholeSubresourceRange.levelCount = 1;
        wholeSubresourceRange.baseArrayLayer = 0;
        wholeSubresourceRange.layerCount = 1;

        VkImageViewCreateInfo imageViewCreateInfo{};
        imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        imageViewCreateInfo.pNext = nullptr;
        imageViewCreateInfo.flags = 0;
        imageViewCreateInfo.image = swapchain.colorImages[i];
        imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        imageViewCreateInfo.format = VK_FORMAT_B8G8R8A8_UNORM;
        imageViewCreateInfo.components = {};
        imageViewCreateInfo.subresourceRange = wholeSubresourceRange;

        ASSERT_SUCCESS(vkCreateImageView(device, &imageViewCreateInfo, nullptr, &swapchain.colorImageViews[i]));

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
        allocationCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        ASSERT_SUCCESS(vmaCreateImage(allocator, &depthImageCreateInfo, &allocationCreateInfo, &swapchain.depthImages[i], &swapchain.depthImageAllocations[i], nullptr));

        VkImageViewCreateInfo depthImageViewCreateInfo{};
        depthImageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        depthImageViewCreateInfo.pNext = nullptr;
        depthImageViewCreateInfo.flags = 0;
        depthImageViewCreateInfo.image = swapchain.depthImages[i];
        depthImageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        depthImageViewCreateInfo.format = VK_FORMAT_R32_UINT;
        depthImageViewCreateInfo.components = {};
        depthImageViewCreateInfo.subresourceRange = wholeSubresourceRange;

        ASSERT_SUCCESS(vkCreateImageView(device, &depthImageViewCreateInfo, nullptr, &swapchain.depthImageViews[i]));
    }

    return swapchain;
}

void destroySwapchain(VkDevice device, VmaAllocator allocator, const Swapchain& swapchain)
{
    for (uint32_t i = 0; i < SWAPCHAIN_BUFFER_COUNT; i++)
    {
        vkDestroyImageView(device, swapchain.depthImageViews[i], nullptr);
        vmaDestroyImage(allocator, swapchain.depthImages[i], swapchain.depthImageAllocations[i]);
        vkDestroyImageView(device, swapchain.colorImageViews[i], nullptr);
    }

    vkDestroySwapchainKHR(device, swapchain.swapchain, nullptr);
}

struct alignas(64) FrameData
{
    glm::mat4 matrices[3];
    glm::mat4 viewProj;
    glm::mat4 inverseViewProj;
    glm::uint iterationCount;
};

struct WorkInFlight
{
    VkCommandBuffer graphicsCommandBuffer;
    VkCommandBuffer presentCommandBuffer;
    VkSemaphore acquireSemaphore;
    VkSemaphore queueTransferSemaphore;
    VkFence renderingFence;
};

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

    VkQueueFamilyProperties queueFamilyProperties[16];
    uint32_t queueFamilyPropertyCount = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertyCount, queueFamilyProperties);

    // graphics queue is always the first
    uint32_t graphicsQueueFamilyIndex = 0;

    assert(queueFamilyProperties[graphicsQueueFamilyIndex].queueFlags & VK_QUEUE_GRAPHICS_BIT);
    assert(queueFamilyProperties[graphicsQueueFamilyIndex].queueFlags & VK_QUEUE_COMPUTE_BIT);

    uint32_t presentQueueFamilyIndex = 0;

    // check if other present queues exist
    for (uint32_t i = 1; i < queueFamilyPropertyCount; i++)
    {
        if (glfwGetPhysicalDevicePresentationSupport(instance, physicalDevice, i))
        {
            presentQueueFamilyIndex = i;
            break;
        }
    }

    bool multiQueue = presentQueueFamilyIndex != graphicsQueueFamilyIndex;

    assert(glfwGetPhysicalDevicePresentationSupport(instance, physicalDevice, presentQueueFamilyIndex));

    float queuePriority = 1.0f;

    VkDeviceQueueCreateInfo graphicsQueueCreateInfo{};
    graphicsQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    graphicsQueueCreateInfo.pNext = nullptr;
    graphicsQueueCreateInfo.flags = 0;
    graphicsQueueCreateInfo.queueFamilyIndex = graphicsQueueFamilyIndex;
    graphicsQueueCreateInfo.queueCount = 1;
    graphicsQueueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceQueueCreateInfo presentQueueCreateInfo{};
    presentQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    presentQueueCreateInfo.pNext = nullptr;
    presentQueueCreateInfo.flags = 0;
    presentQueueCreateInfo.queueFamilyIndex = presentQueueFamilyIndex;
    presentQueueCreateInfo.queueCount = 1;
    presentQueueCreateInfo.pQueuePriorities = &queuePriority;

    const uint32_t multiQueueCreateInfoCount = 2;
    VkDeviceQueueCreateInfo multiQueueCreateInfos[multiQueueCreateInfoCount]
    {
        graphicsQueueCreateInfo,
        presentQueueCreateInfo
    };

    const uint32_t singleQueueCreateInfoCount = 1;
    VkDeviceQueueCreateInfo singleQueueCreateInfos[singleQueueCreateInfoCount]
    {
        graphicsQueueCreateInfo
    };

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
    deviceCreateInfo.queueCreateInfoCount = multiQueue ? multiQueueCreateInfoCount : singleQueueCreateInfoCount;
    deviceCreateInfo.pQueueCreateInfos = multiQueue ? multiQueueCreateInfos : singleQueueCreateInfos;
    deviceCreateInfo.enabledLayerCount = 0;
    deviceCreateInfo.ppEnabledLayerNames = nullptr;
    deviceCreateInfo.enabledExtensionCount = deviceExtensionCount;
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensionNames;
    deviceCreateInfo.pEnabledFeatures = &enabledFeatures;

    VkDevice device;
    ASSERT_SUCCESS(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));

    VkQueue graphicsQueue;
    vkGetDeviceQueue(device, graphicsQueueFamilyIndex, 0, &graphicsQueue);

    VkQueue presentQueue;
    vkGetDeviceQueue(device, presentQueueFamilyIndex, 0, &presentQueue);

    VmaAllocatorCreateInfo allocatorCreateInfo{};
    allocatorCreateInfo.instance = instance;
    allocatorCreateInfo.physicalDevice = physicalDevice;
    allocatorCreateInfo.device = device;
    allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_0;

    VmaAllocator allocator;
    ASSERT_SUCCESS(vmaCreateAllocator(&allocatorCreateInfo, &allocator));

    VkCommandPoolCreateInfo graphicsCommandPoolCreateInfo{};
    graphicsCommandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    graphicsCommandPoolCreateInfo.pNext = nullptr;
    graphicsCommandPoolCreateInfo.flags = 0;
    graphicsCommandPoolCreateInfo.queueFamilyIndex = graphicsQueueFamilyIndex;

    VkCommandPool graphicsCommandPool;
    ASSERT_SUCCESS(vkCreateCommandPool(device, &graphicsCommandPoolCreateInfo, nullptr, &graphicsCommandPool));

    VkCommandPoolCreateInfo presentCommandPoolCreateInfo{};
    presentCommandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    presentCommandPoolCreateInfo.pNext = nullptr;
    presentCommandPoolCreateInfo.flags = 0;
    presentCommandPoolCreateInfo.queueFamilyIndex = presentQueueFamilyIndex;

    VkCommandPool presentCommandPool;
    ASSERT_SUCCESS(vkCreateCommandPool(device, &presentCommandPoolCreateInfo, nullptr, &presentCommandPool));

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

    VkDescriptorSetLayoutBinding matricesBinding{};
    matricesBinding.binding = 2;
    matricesBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    matricesBinding.descriptorCount = 1;
    matricesBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    matricesBinding.pImmutableSamplers = nullptr;

    const uint32_t bindingCount = 3;
    VkDescriptorSetLayoutBinding bindings[bindingCount]
    {
        depthBinding,
        colorBinding,
        matricesBinding
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

    VkDescriptorPoolSize imagePoolSize{};
    imagePoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    imagePoolSize.descriptorCount = 2 * SWAPCHAIN_BUFFER_COUNT;

    VkDescriptorPoolSize bufferPoolSize{};
    bufferPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bufferPoolSize.descriptorCount = 1 * SWAPCHAIN_BUFFER_COUNT;

    const uint32_t poolSizeCount = 2;
    VkDescriptorPoolSize poolSizes[poolSizeCount]
    {
        imagePoolSize,
        bufferPoolSize
    };

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo{};
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.pNext = nullptr;
    descriptorPoolCreateInfo.flags = 0;
    descriptorPoolCreateInfo.maxSets = SWAPCHAIN_BUFFER_COUNT;
    descriptorPoolCreateInfo.poolSizeCount = poolSizeCount;
    descriptorPoolCreateInfo.pPoolSizes = poolSizes;

    VkDescriptorPool descriptorPool;
    vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &descriptorPool);

    VkDescriptorSetLayout descriptorSetLayouts[SWAPCHAIN_BUFFER_COUNT];

    for (uint32_t i = 0; i < SWAPCHAIN_BUFFER_COUNT; i++)
    {
        descriptorSetLayouts[i] = descriptorSetLayout;
    }

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.pNext = nullptr;
    descriptorSetAllocateInfo.descriptorPool = descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = SWAPCHAIN_BUFFER_COUNT;
    descriptorSetAllocateInfo.pSetLayouts = descriptorSetLayouts;

    VkDescriptorSet descriptorSets[SWAPCHAIN_BUFFER_COUNT];
    vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, descriptorSets);

    uint32_t width = 1280;
    uint32_t height = 720;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(width, height, "Vulkan window", nullptr, nullptr);

    glfwSetKeyCallback(window, key_callback);

    VkSurfaceKHR surface;
    ASSERT_SUCCESS(glfwCreateWindowSurface(instance, window, nullptr, &surface));

    VkPresentModeKHR presentModes[16];
    uint32_t presentModeCount = 16;
    ASSERT_SUCCESS(vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, presentModes));

    assert(presentModeCount > 0);

    bool supportsMailbox = false;
    bool supportsFifoRelaxed = false;
    bool supportsImmediate = false;

    for (uint32_t i = 0; i < presentModeCount; i++)
    {
        supportsMailbox = supportsMailbox || presentModes[i] == VK_PRESENT_MODE_MAILBOX_KHR;
        supportsFifoRelaxed = supportsFifoRelaxed || presentModes[i] == VK_PRESENT_MODE_FIFO_RELAXED_KHR;
        supportsImmediate = supportsImmediate || presentModes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR;
    }

    VkPresentModeKHR presentMode =
        supportsMailbox ? VK_PRESENT_MODE_MAILBOX_KHR :
        supportsFifoRelaxed ? VK_PRESENT_MODE_FIFO_RELAXED_KHR :
        supportsImmediate ? VK_PRESENT_MODE_IMMEDIATE_KHR :
        VK_PRESENT_MODE_FIFO_KHR;

    Swapchain swapchain = createSwapchain(device, allocator, surface, presentMode, width, height);

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.pNext = nullptr;
    bufferCreateInfo.flags = 0;
    bufferCreateInfo.size = SWAPCHAIN_BUFFER_COUNT * sizeof(FrameData);
    bufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferCreateInfo.queueFamilyIndexCount = 0;
    bufferCreateInfo.pQueueFamilyIndices = nullptr;

    VmaAllocationCreateInfo allocationCreateInfo{};
    allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocationCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    allocationCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkBuffer buffer;
    VmaAllocation bufferAllocation;
    VmaAllocationInfo bufferAllocationInfo;
    ASSERT_SUCCESS(vmaCreateBuffer(allocator, &bufferCreateInfo, &allocationCreateInfo, &buffer, &bufferAllocation, &bufferAllocationInfo));

    FrameData* pMappedFrameData = reinterpret_cast<FrameData*>(bufferAllocationInfo.pMappedData);

    float angle = 0.0f;

    WorkInFlight workInFlights[SWAPCHAIN_BUFFER_COUNT];

    for (uint32_t i = 0; i < SWAPCHAIN_BUFFER_COUNT; i++)
    {
        workInFlights[i] = {};  // sets all to null
    }

    // we need a very large pool of semaphores because it improves our odds of not deleting a semaphore that's in use
    // see https://github.com/KhronosGroup/Vulkan-Docs/issues/2007
    const uint32_t presentSemaphoreCount = 64;
    VkSemaphore presentSemaphores[presentSemaphoreCount];
    for (uint32_t i = 0; i < presentSemaphoreCount; i++)
    {
        presentSemaphores[i] = VK_NULL_HANDLE;
    }

    uint32_t presentSemaphoreIndex = 0;

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        VkSemaphoreCreateInfo semaphoreCreateInfo{};
        semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        semaphoreCreateInfo.pNext = nullptr;
        semaphoreCreateInfo.flags = 0;

        VkSemaphore acquireSemaphore;
        ASSERT_SUCCESS(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &acquireSemaphore));

        VkSemaphore queueTransferSemaphore;
        ASSERT_SUCCESS(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &queueTransferSemaphore));

        VkFenceCreateInfo fenceCreateInfo{};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.pNext = nullptr;
        fenceCreateInfo.flags = 0;

        VkFence acquireFence;
        ASSERT_SUCCESS(vkCreateFence(device, &fenceCreateInfo, nullptr, &acquireFence));

        VkFence renderingFence;
        ASSERT_SUCCESS(vkCreateFence(device, &fenceCreateInfo, nullptr, &renderingFence));

        uint32_t imageIndex;
        VkResult acquireResult = vkAcquireNextImageKHR(device, swapchain.swapchain, UINT64_MAX, acquireSemaphore, acquireFence, &imageIndex);

        assert(acquireResult == VK_SUCCESS || acquireResult == VK_ERROR_OUT_OF_DATE_KHR || acquireResult == VK_SUBOPTIMAL_KHR);

        if (workInFlights[imageIndex].renderingFence != VK_NULL_HANDLE)
        {
            ASSERT_SUCCESS(vkWaitForFences(device, 1, &workInFlights[imageIndex].renderingFence, VK_TRUE, UINT64_MAX));

            vkFreeCommandBuffers(device, graphicsCommandPool, 1, &workInFlights[imageIndex].graphicsCommandBuffer);
            vkFreeCommandBuffers(device, presentCommandPool, 1, &workInFlights[imageIndex].presentCommandBuffer);
            vkDestroySemaphore(device, workInFlights[imageIndex].acquireSemaphore, nullptr);
            vkDestroySemaphore(device, workInFlights[imageIndex].queueTransferSemaphore, nullptr);
            vkDestroyFence(device, workInFlights[imageIndex].renderingFence, nullptr);
        }

        ASSERT_SUCCESS(vkWaitForFences(device, 1, &acquireFence, VK_TRUE, UINT64_MAX));

        vkDestroyFence(device, acquireFence, nullptr);

        if (presentSemaphores[presentSemaphoreIndex] != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(device, presentSemaphores[presentSemaphoreIndex], nullptr);
        }

        VkSemaphore presentSemaphore;
        ASSERT_SUCCESS(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &presentSemaphore));

        angle += 0.01f;

        glm::mat4 view = glm::rotate(glm::translate(glm::mat4(1.0f), glm::vec3(0.0, 0.0, -16.0)), angle, glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 proj = glm::perspectiveFov(0.3f, static_cast<float>(width), static_cast<float>(height), 9.0f, 21.0f);

        FrameData frameData{};
        frameData.matrices[0] = randomTransform(1 + 3 * fractSeed);
        frameData.matrices[1] = randomTransform(2 + 3 * fractSeed);
        frameData.matrices[2] = randomTransform(3 + 3 * fractSeed);
        frameData.viewProj = proj * view;
        frameData.inverseViewProj = glm::inverse(proj * view);
        frameData.iterationCount = FRACTAL_ITERATIONS - 3;

        pMappedFrameData[imageIndex] = frameData;

        vmaFlushAllocation(allocator, bufferAllocation, imageIndex * sizeof(FrameData), sizeof(FrameData));

        VkDescriptorImageInfo depthInfo{};
        depthInfo.sampler = VK_NULL_HANDLE;
        depthInfo.imageView = swapchain.depthImageViews[imageIndex];
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet depthWrite{};
        depthWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        depthWrite.pNext = nullptr;
        depthWrite.dstSet = descriptorSets[imageIndex];
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
        colorWrite.dstSet = descriptorSets[imageIndex];
        colorWrite.dstBinding = 1;
        colorWrite.dstArrayElement = 0;
        colorWrite.descriptorCount = 1;
        colorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        colorWrite.pImageInfo = &colorInfo;
        colorWrite.pBufferInfo = nullptr;
        colorWrite.pTexelBufferView = nullptr;

        VkDescriptorBufferInfo matricesInfo{};
        matricesInfo.buffer = buffer;
        matricesInfo.offset = imageIndex * sizeof(FrameData);
        matricesInfo.range = sizeof(FrameData);

        VkWriteDescriptorSet matricesWrite{};
        matricesWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        matricesWrite.pNext = nullptr;
        matricesWrite.dstSet = descriptorSets[imageIndex];
        matricesWrite.dstBinding = 2;
        matricesWrite.dstArrayElement = 0;
        matricesWrite.descriptorCount = 1;
        matricesWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        matricesWrite.pImageInfo = nullptr;
        matricesWrite.pBufferInfo = &matricesInfo;
        matricesWrite.pTexelBufferView = nullptr;

        const uint32_t descriptorWriteCount = 3;
        VkWriteDescriptorSet descriptorWrites[descriptorWriteCount]
        {
            depthWrite,
            colorWrite,
            matricesWrite
        };

        vkUpdateDescriptorSets(device, descriptorWriteCount, descriptorWrites, 0, nullptr);

        VkCommandBufferAllocateInfo graphicsCommandBufferAllocateInfo{};
        graphicsCommandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        graphicsCommandBufferAllocateInfo.pNext = nullptr;
        graphicsCommandBufferAllocateInfo.commandPool = graphicsCommandPool;
        graphicsCommandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        graphicsCommandBufferAllocateInfo.commandBufferCount = 1;

        VkCommandBuffer graphicsCommandBuffer;
        ASSERT_SUCCESS(vkAllocateCommandBuffers(device, &graphicsCommandBufferAllocateInfo, &graphicsCommandBuffer));

        VkCommandBufferAllocateInfo presentCommandBufferAllocateInfo{};
        presentCommandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        presentCommandBufferAllocateInfo.pNext = nullptr;
        presentCommandBufferAllocateInfo.commandPool = presentCommandPool;
        presentCommandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        presentCommandBufferAllocateInfo.commandBufferCount = 1;

        VkCommandBuffer presentCommandBuffer;
        ASSERT_SUCCESS(vkAllocateCommandBuffers(device, &presentCommandBufferAllocateInfo, &presentCommandBuffer));

        VkCommandBufferBeginInfo graphicsCommandBufferBeginInfo{};
        graphicsCommandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        graphicsCommandBufferBeginInfo.pNext = nullptr;
        graphicsCommandBufferBeginInfo.flags = 0;
        graphicsCommandBufferBeginInfo.pInheritanceInfo = nullptr;

        ASSERT_SUCCESS(vkBeginCommandBuffer(graphicsCommandBuffer, &graphicsCommandBufferBeginInfo));

        vkCmdBindDescriptorSets(graphicsCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSets[imageIndex], 0, nullptr);

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
        depthToComputeBarrier.srcQueueFamilyIndex = graphicsQueueFamilyIndex;
        depthToComputeBarrier.dstQueueFamilyIndex = graphicsQueueFamilyIndex;
        depthToComputeBarrier.image = swapchain.depthImages[imageIndex];
        depthToComputeBarrier.subresourceRange = wholeImageRange;

        VkImageMemoryBarrier colorToComputeBarrier{};
        colorToComputeBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        colorToComputeBarrier.pNext = nullptr;
        colorToComputeBarrier.srcAccessMask = 0;
        colorToComputeBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        colorToComputeBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorToComputeBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        colorToComputeBarrier.srcQueueFamilyIndex = graphicsQueueFamilyIndex;
        colorToComputeBarrier.dstQueueFamilyIndex = graphicsQueueFamilyIndex;
        colorToComputeBarrier.image = swapchain.colorImages[imageIndex];
        colorToComputeBarrier.subresourceRange = wholeImageRange;

        VkImageMemoryBarrier depthComputeToComputeBarrier{};
        depthComputeToComputeBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        depthComputeToComputeBarrier.pNext = nullptr;
        depthComputeToComputeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        depthComputeToComputeBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        depthComputeToComputeBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        depthComputeToComputeBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        depthComputeToComputeBarrier.srcQueueFamilyIndex = graphicsQueueFamilyIndex;
        depthComputeToComputeBarrier.dstQueueFamilyIndex = graphicsQueueFamilyIndex;
        depthComputeToComputeBarrier.image = swapchain.depthImages[imageIndex];
        depthComputeToComputeBarrier.subresourceRange = wholeImageRange;

        VkImageMemoryBarrier colorComputeToPresentBarrier{};
        colorComputeToPresentBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        colorComputeToPresentBarrier.pNext = nullptr;
        colorComputeToPresentBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        colorComputeToPresentBarrier.dstAccessMask = 0;
        colorComputeToPresentBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        colorComputeToPresentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        colorComputeToPresentBarrier.srcQueueFamilyIndex = graphicsQueueFamilyIndex;
        colorComputeToPresentBarrier.dstQueueFamilyIndex = graphicsQueueFamilyIndex;
        colorComputeToPresentBarrier.image = swapchain.colorImages[imageIndex];
        colorComputeToPresentBarrier.subresourceRange = wholeImageRange;

        VkImageMemoryBarrier colorComputeToOwnershipReleaseBarrier{};
        colorComputeToOwnershipReleaseBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        colorComputeToOwnershipReleaseBarrier.pNext = nullptr;
        colorComputeToOwnershipReleaseBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        colorComputeToOwnershipReleaseBarrier.dstAccessMask = 0;
        colorComputeToOwnershipReleaseBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        colorComputeToOwnershipReleaseBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        colorComputeToOwnershipReleaseBarrier.srcQueueFamilyIndex = graphicsQueueFamilyIndex;
        colorComputeToOwnershipReleaseBarrier.dstQueueFamilyIndex = presentQueueFamilyIndex;
        colorComputeToOwnershipReleaseBarrier.image = swapchain.colorImages[imageIndex];
        colorComputeToOwnershipReleaseBarrier.subresourceRange = wholeImageRange;

        VkImageMemoryBarrier colorOwnershipAcquireToPresentBarrier{};
        colorOwnershipAcquireToPresentBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        colorOwnershipAcquireToPresentBarrier.pNext = nullptr;
        colorOwnershipAcquireToPresentBarrier.srcAccessMask = 0;  // handled by semaphore
        colorOwnershipAcquireToPresentBarrier.dstAccessMask = 0;
        colorOwnershipAcquireToPresentBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        colorOwnershipAcquireToPresentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        colorOwnershipAcquireToPresentBarrier.srcQueueFamilyIndex = graphicsQueueFamilyIndex;
        colorOwnershipAcquireToPresentBarrier.dstQueueFamilyIndex = presentQueueFamilyIndex;
        colorOwnershipAcquireToPresentBarrier.image = swapchain.colorImages[imageIndex];
        colorOwnershipAcquireToPresentBarrier.subresourceRange = wholeImageRange;

        const uint32_t imageMemoryBarrierCount = 2;
        VkImageMemoryBarrier imageMemoryBarriers[imageMemoryBarrierCount]
        {
            depthToComputeBarrier,
            colorToComputeBarrier
        };

        // transition depth/color image
        vkCmdPipelineBarrier(graphicsCommandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, imageMemoryBarrierCount, imageMemoryBarriers);

        vkCmdBindPipeline(graphicsCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, clearPipeline);
        vkCmdDispatch(graphicsCommandBuffer, (width + 15u) / 16u, (height + 15u) / 16u, 1);

        // guard depth writes against reads/writes
        vkCmdPipelineBarrier(graphicsCommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &depthComputeToComputeBarrier);

        vkCmdBindPipeline(graphicsCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, splatPipeline);
        vkCmdDispatch(graphicsCommandBuffer, (UINT_POW(3, FRACTAL_ITERATIONS - 3) + 255u) / 256u, 1, 1);

        // guard depth writes against reads
        vkCmdPipelineBarrier(graphicsCommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &depthComputeToComputeBarrier);

        vkCmdBindPipeline(graphicsCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, displayPipeline);
        vkCmdDispatch(graphicsCommandBuffer, (width + 15u) / 16u, (height + 15u) / 16u, 1);

        VkResult presentResult;

        if (multiQueue)
        {
            // release ownership
            vkCmdPipelineBarrier(graphicsCommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &colorComputeToOwnershipReleaseBarrier);

            ASSERT_SUCCESS(vkEndCommandBuffer(graphicsCommandBuffer));

            VkPipelineStageFlags graphicsWaitDstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

            VkSubmitInfo graphicsSubmitInfo{};
            graphicsSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            graphicsSubmitInfo.pNext = nullptr;
            graphicsSubmitInfo.waitSemaphoreCount = 1;
            graphicsSubmitInfo.pWaitSemaphores = &acquireSemaphore;
            graphicsSubmitInfo.pWaitDstStageMask = &graphicsWaitDstStageMask;
            graphicsSubmitInfo.commandBufferCount = 1;
            graphicsSubmitInfo.pCommandBuffers = &graphicsCommandBuffer;
            graphicsSubmitInfo.signalSemaphoreCount = 1;
            graphicsSubmitInfo.pSignalSemaphores = &queueTransferSemaphore;

            ASSERT_SUCCESS(vkQueueSubmit(graphicsQueue, 1, &graphicsSubmitInfo, nullptr));

            VkCommandBufferBeginInfo presentCommandBufferBeginInfo{};
            presentCommandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            presentCommandBufferBeginInfo.pNext = nullptr;
            presentCommandBufferBeginInfo.flags = 0;
            presentCommandBufferBeginInfo.pInheritanceInfo = nullptr;

            ASSERT_SUCCESS(vkBeginCommandBuffer(presentCommandBuffer, &presentCommandBufferBeginInfo));

            // transfer image from graphics to present queue
            vkCmdPipelineBarrier(presentCommandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &colorOwnershipAcquireToPresentBarrier);

            ASSERT_SUCCESS(vkEndCommandBuffer(presentCommandBuffer));

            VkPipelineStageFlags presentWaitDstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

            VkSubmitInfo presentSubmitInfo{};
            presentSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            presentSubmitInfo.pNext = nullptr;
            presentSubmitInfo.waitSemaphoreCount = 1;
            presentSubmitInfo.pWaitSemaphores = &queueTransferSemaphore;
            presentSubmitInfo.pWaitDstStageMask = &presentWaitDstStageMask;
            presentSubmitInfo.commandBufferCount = 1;
            presentSubmitInfo.pCommandBuffers = &presentCommandBuffer;
            presentSubmitInfo.signalSemaphoreCount = 1;
            presentSubmitInfo.pSignalSemaphores = &presentSemaphore;

            ASSERT_SUCCESS(vkQueueSubmit(presentQueue, 1, &presentSubmitInfo, renderingFence));

            VkPresentInfoKHR presentInfo{};
            presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            presentInfo.pNext = nullptr;
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = &presentSemaphore;
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = &swapchain.swapchain;
            presentInfo.pImageIndices = &imageIndex;
            presentInfo.pResults = nullptr;

            presentResult = vkQueuePresentKHR(presentQueue, &presentInfo);

        }
        else
        {    // transfer image from graphics to present queue
            vkCmdPipelineBarrier(graphicsCommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &colorComputeToPresentBarrier);

            ASSERT_SUCCESS(vkEndCommandBuffer(graphicsCommandBuffer));

            VkPipelineStageFlags graphicsWaitDstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

            VkSubmitInfo graphicsSubmitInfo{};
            graphicsSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            graphicsSubmitInfo.pNext = nullptr;
            graphicsSubmitInfo.waitSemaphoreCount = 1;
            graphicsSubmitInfo.pWaitSemaphores = &acquireSemaphore;
            graphicsSubmitInfo.pWaitDstStageMask = &graphicsWaitDstStageMask;
            graphicsSubmitInfo.commandBufferCount = 1;
            graphicsSubmitInfo.pCommandBuffers = &graphicsCommandBuffer;
            graphicsSubmitInfo.signalSemaphoreCount = 1;
            graphicsSubmitInfo.pSignalSemaphores = &presentSemaphore;

            ASSERT_SUCCESS(vkQueueSubmit(graphicsQueue, 1, &graphicsSubmitInfo, renderingFence));

            VkPresentInfoKHR presentInfo{};
            presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            presentInfo.pNext = nullptr;
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = &presentSemaphore;
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = &swapchain.swapchain;
            presentInfo.pImageIndices = &imageIndex;
            presentInfo.pResults = nullptr;

            presentResult = vkQueuePresentKHR(graphicsQueue, &presentInfo);
        }

        WorkInFlight newWorkInFlight{};
        newWorkInFlight.graphicsCommandBuffer = graphicsCommandBuffer;
        newWorkInFlight.presentCommandBuffer = presentCommandBuffer;
        newWorkInFlight.acquireSemaphore = acquireSemaphore;
        newWorkInFlight.queueTransferSemaphore = queueTransferSemaphore;
        newWorkInFlight.renderingFence = renderingFence;

        workInFlights[imageIndex] = newWorkInFlight;

        presentSemaphores[presentSemaphoreIndex] = presentSemaphore;

        presentSemaphoreIndex = (presentSemaphoreIndex + 1) % presentSemaphoreCount;

        if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR)
        {
            int newWidth = 0;
            int newHeight = 0;

            while ((newWidth == 0 || newHeight == 0) && !glfwWindowShouldClose(window))
            {
                glfwPollEvents();
                glfwGetWindowSize(window, &newWidth, &newHeight);
            }

            if (newWidth > 0 && newHeight > 0)
            {
                // doesn't make the deletion of the present semaphore safe, see https://github.com/KhronosGroup/Vulkan-Docs/issues/2007
                // I hate this
                vkDeviceWaitIdle(device);

                destroySwapchain(device, allocator, swapchain);

                width = static_cast<uint32_t>(newWidth);
                height = static_cast<uint32_t>(newHeight);

                swapchain = createSwapchain(device, allocator, surface, presentMode, width, height);
            }
        }
        else
        {
            ASSERT_SUCCESS(presentResult);
        }
    }

    vkDeviceWaitIdle(device);

    for (uint32_t i = 0; i < SWAPCHAIN_BUFFER_COUNT; i++)
    {
        if (workInFlights[i].renderingFence != VK_NULL_HANDLE)
        {
            vkFreeCommandBuffers(device, graphicsCommandPool, 1, &workInFlights[i].graphicsCommandBuffer);
            vkFreeCommandBuffers(device, presentCommandPool, 1, &workInFlights[i].presentCommandBuffer);
            vkDestroySemaphore(device, workInFlights[i].acquireSemaphore, nullptr);
            vkDestroySemaphore(device, workInFlights[i].queueTransferSemaphore, nullptr);
            vkDestroyFence(device, workInFlights[i].renderingFence, nullptr);
        }
    }

    for (uint32_t i = 0; i < presentSemaphoreCount; i++)
    {
        if (presentSemaphores[i] != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(device, presentSemaphores[i], nullptr);
        }
    }

    vmaDestroyBuffer(allocator, buffer, bufferAllocation);

    destroySwapchain(device, allocator, swapchain);

    vkDestroySurfaceKHR(instance, surface, nullptr);

    vkDestroyDescriptorPool(device, descriptorPool, nullptr);

    vkDestroyPipeline(device, displayPipeline, nullptr);
    vkDestroyPipeline(device, splatPipeline, nullptr);
    vkDestroyPipeline(device, clearPipeline, nullptr);

    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

    vkDestroyCommandPool(device, presentCommandPool, nullptr);

    vkDestroyCommandPool(device, graphicsCommandPool, nullptr);

    vmaDestroyAllocator(allocator);

    vkDestroyDevice(device, nullptr);

    vkDestroyInstance(instance, nullptr);

    glfwDestroyWindow(window);

    glfwTerminate();

	return 0;
}
