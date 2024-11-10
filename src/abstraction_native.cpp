#include "abstraction_native.h"

#define VMA_IMPLEMENTATION
#include "vma/vk_mem_alloc.h"

#include <vector>
#include <assert.h>

void checkResult(VkResult result)
{
    assert(result == VK_SUCCESS);
}

#define ASSERT_SUCCESS(res) checkResult(res)

#define SWAPCHAIN_IMAGE_COUNT 3u

#define PRESENT_SEMAPHORE_COUNT 256u

template<class T, class P>
struct HandleManager
{
    T& operator[](P handle)
    {
        uint32_t index   = static_cast<uint32_t>((handle.handle      ) & 0xFFFFFFFF);
        uint32_t version = static_cast<uint32_t>((handle.handle >> 32) & 0xFFFFFFFF);

        assert(versionedObjects[index].version == version);

        return versionedObjects[index].object;
    }

    [[nodiscard]]
    P create(const T& object)
    {
        if (freeHandles.empty())
        {
            uint64_t handle = 0x100000000 | static_cast<uint64_t>(versionedObjects.size());

            VersionedObjects versionedObject{};
            versionedObject.object = object;
            versionedObject.version = 1;

            versionedObjects.push_back(versionedObject);

            return P{ handle };
        }
        else
        {
            uint64_t handle = freeHandles.back();

            uint32_t index   = static_cast<uint32_t>((handle      ) & 0xFFFFFFFF);
            uint32_t version = static_cast<uint32_t>((handle >> 32) & 0xFFFFFFFF);

            assert(versionedObjects[index].version == version);

            versionedObjects[index].object = object;

            freeHandles.pop_back();

            return P{ handle };
        }
    }

    void destroy(P handle)
    {
        uint32_t index   = static_cast<uint32_t>((handle.handle) & 0xFFFFFFFF);
        uint32_t version = static_cast<uint32_t>((handle.handle >> 32) & 0xFFFFFFFF);

        assert(versionedObjects[index].version == version);

        versionedObjects[index].object = {};
        versionedObjects[index].version++;

        freeHandles.push_back(handle.handle + 0x100000000);
    }

private:

    struct VersionedObjects
    {
        T object;
        uint32_t version;
    };

    std::vector<VersionedObjects> versionedObjects{};
    std::vector<uint64_t> freeHandles{};
};

struct _Device
{
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VmaAllocator allocator;
    uint32_t graphicsQueueFamilyIndex;
    uint32_t computeQueueFamilyIndex;
    VkCommandPool graphicsCommandPool;
    VkCommandPool computeCommandPool;
    VkSemaphore presentSemaphores[PRESENT_SEMAPHORE_COUNT];
    uint32_t presentSemaphoreIndex;
};

struct _Buffer
{
    VkBuffer buffer;
    VmaAllocation allocation;
    void* pMappedData;
};

struct _Image
{
    VkImage image;
    VmaAllocation allocation;
    VkImageView imageView;
    VkCommandBuffer commandBuffer;
    VkFence fence;
};

struct _PipelineLayout
{
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    uint32_t imageCount;
    uint32_t bufferCount;
};

struct _DescriptorPool
{
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;
};

struct _Pipeline
{
    VkPipeline pipeline;
};

struct _CommandBuffer
{
    VkCommandBuffer commandBuffer;
};

struct _Fence
{
    VkFence fence;
};

struct _Swapchain
{
    VkSwapchainKHR swapchain;
    Image images[SWAPCHAIN_IMAGE_COUNT];
    VkFence acquireFences[SWAPCHAIN_IMAGE_COUNT];
    VkSemaphore acquireSemaphores[SWAPCHAIN_IMAGE_COUNT];
    VkFence presentFences[SWAPCHAIN_IMAGE_COUNT];
    VkCommandBuffer graphicsCommandBuffers[SWAPCHAIN_IMAGE_COUNT];
    VkCommandBuffer computeCommandBuffers[SWAPCHAIN_IMAGE_COUNT];
    VkSemaphore transferSemaphores[SWAPCHAIN_IMAGE_COUNT];
};

struct _Surface
{
    VkSurfaceKHR surface;
};

static HandleManager<_Device, Device> _devices;
static HandleManager<_Buffer, Buffer> _buffers;
static HandleManager<_Image, Image> _images;
static HandleManager<_Pipeline, Pipeline> _pipelines;
static HandleManager<_PipelineLayout, PipelineLayout> _pipelineLayouts;
static HandleManager<_DescriptorPool, DescriptorPool> _descriptorPools;
static HandleManager<_CommandBuffer, CommandBuffer> _commandBuffers;
static HandleManager<_Fence, Fence> _fences;
static HandleManager<_Swapchain, Swapchain> _swapchains;
static HandleManager<_Surface, Surface> _surfaces;

Device createDevice()
{
    uint32_t totalInstanceExtensionCount;
    ASSERT_SUCCESS(vkEnumerateInstanceExtensionProperties(nullptr, &totalInstanceExtensionCount, nullptr));

    std::vector<VkExtensionProperties> totalInstanceExtensions(totalInstanceExtensionCount);
    ASSERT_SUCCESS(vkEnumerateInstanceExtensionProperties(nullptr, &totalInstanceExtensionCount, totalInstanceExtensions.data()));

    std::vector<const char*> instanceExtensionNames{};

    for (const char* extensionName : {
        "VK_KHR_surface",
        "VK_KHR_win32_surface",
        "VK_MVK_macos_surface",
        "VK_EXT_metal_surface",
        "VK_KHR_xlib_surface",
        "VK_KHR_xcb_surface",
        "VK_KHR_wayland_surface",
        "VK_EXT_headless_surface"
        })
    {
        for (uint32_t i = 0; i < totalInstanceExtensionCount; i++)
        {
            if (strcmp(totalInstanceExtensions[i].extensionName, extensionName) == 0)
            {
                instanceExtensionNames.push_back(extensionName);
                break;
            }
        }

        // congrats I didn't even know that was possible
        assert(extensionName != "VK_KHR_surface");
    }

    // should have VK_KHR_surface and the OS specific extension e.g. VK_KHR_win32_surface
    assert(instanceExtensionNames.size() == 2);

    VkApplicationInfo applicationInfo{};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pNext = nullptr;
    applicationInfo.pApplicationName = "";
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
    instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensionNames.size());
    instanceCreateInfo.ppEnabledExtensionNames = instanceExtensionNames.data();

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

    // we'll use the graphics queue for submits and compute queue for present
    uint32_t graphicsQueueFamilyIndex = queueFamilyPropertyCount;
    for (uint32_t i = 0; i < queueFamilyPropertyCount; i++)
    {
        if (queueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            graphicsQueueFamilyIndex = i;
            break;
        }
    }

    // no graphics queue found?? is it even a GPU?
    assert(graphicsQueueFamilyIndex != queueFamilyPropertyCount);

    uint32_t computeQueueFamilyIndex = queueFamilyPropertyCount;
    for (uint32_t i = 0; i < queueFamilyPropertyCount; i++)
    {
        if ((queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && (graphicsQueueFamilyIndex != i))
        {
            computeQueueFamilyIndex = i;
            break;
        }
    }

    if (computeQueueFamilyIndex == queueFamilyPropertyCount)
    {
        // no async compute queue found, set it to the graphics queue
        computeQueueFamilyIndex = graphicsQueueFamilyIndex;
    }

    bool hasAsync = computeQueueFamilyIndex != graphicsQueueFamilyIndex;

    float queuePriority = 1.0f;

    VkDeviceQueueCreateInfo graphicsQueueCreateInfo{};
    graphicsQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    graphicsQueueCreateInfo.pNext = nullptr;
    graphicsQueueCreateInfo.flags = 0;
    graphicsQueueCreateInfo.queueFamilyIndex = graphicsQueueFamilyIndex;
    graphicsQueueCreateInfo.queueCount = 1;
    graphicsQueueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceQueueCreateInfo computeQueueCreateInfo{};
    computeQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    computeQueueCreateInfo.pNext = nullptr;
    computeQueueCreateInfo.flags = 0;
    computeQueueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
    computeQueueCreateInfo.queueCount = 1;
    computeQueueCreateInfo.pQueuePriorities = &queuePriority;

    const uint32_t queueCreateInfoCount = 2;
    VkDeviceQueueCreateInfo queueCreateInfos[queueCreateInfoCount]
    {
        graphicsQueueCreateInfo,
        computeQueueCreateInfo
    };

    const uint32_t noAsyncQueueCreateInfoCount = 1;
    VkDeviceQueueCreateInfo noAsyncQueueCreateInfos[noAsyncQueueCreateInfoCount]
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
    deviceCreateInfo.queueCreateInfoCount = hasAsync ? queueCreateInfoCount : noAsyncQueueCreateInfoCount;
    deviceCreateInfo.pQueueCreateInfos = hasAsync ? queueCreateInfos : noAsyncQueueCreateInfos;
    deviceCreateInfo.enabledLayerCount = 0;
    deviceCreateInfo.ppEnabledLayerNames = nullptr;
    deviceCreateInfo.enabledExtensionCount = deviceExtensionCount;
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensionNames;
    deviceCreateInfo.pEnabledFeatures = &enabledFeatures;

    VkDevice device;
    ASSERT_SUCCESS(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));

    VkQueue graphicsQueue;
    vkGetDeviceQueue(device, graphicsQueueFamilyIndex, 0, &graphicsQueue);

    VkQueue computeQueue;
    vkGetDeviceQueue(device, graphicsQueueFamilyIndex, 0, &computeQueue);

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

    VkCommandPoolCreateInfo computeCommandPoolCreateInfo{};
    computeCommandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    computeCommandPoolCreateInfo.pNext = nullptr;
    computeCommandPoolCreateInfo.flags = 0;
    computeCommandPoolCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;

    VkCommandPool computeCommandPool;
    ASSERT_SUCCESS(vkCreateCommandPool(device, &computeCommandPoolCreateInfo, nullptr, &computeCommandPool));

    _Device _device{};
    _device.instance = instance;
    _device.physicalDevice = physicalDevice;
    _device.device = device;
    _device.allocator = allocator;
    _device.graphicsQueueFamilyIndex = graphicsQueueFamilyIndex;
    _device.computeQueueFamilyIndex = computeQueueFamilyIndex;
    _device.graphicsCommandPool = graphicsCommandPool;
    _device.computeCommandPool = computeCommandPool;

    for (uint32_t i = 0; i < PRESENT_SEMAPHORE_COUNT; i++)
    {
        _device.presentSemaphores[i] = VK_NULL_HANDLE;
    }

    _device.presentSemaphoreIndex = 0;
    
    return _devices.create(_device);
}

void destroyDevice(Device device)
{
    _Device& _device = _devices[device];

    for (uint32_t i = 0; i < PRESENT_SEMAPHORE_COUNT; i++)
    {
        vkDestroySemaphore(_device.device, _device.presentSemaphores[i], nullptr);
    }

    vkDestroyCommandPool(_device.device, _device.graphicsCommandPool, nullptr);
    vkDestroyCommandPool(_device.device, _device.computeCommandPool, nullptr);
    vmaDestroyAllocator(_device.allocator);
    vkDestroyDevice(_device.device, nullptr);
    vkDestroyInstance(_device.instance, nullptr);

    _devices.destroy(device);
}

Buffer Device::createBuffer(uint64_t size)
{
    _Device& _device = _devices[*this];

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.pNext = nullptr;
    bufferCreateInfo.flags = 0;
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferCreateInfo.queueFamilyIndexCount = 0;
    bufferCreateInfo.pQueueFamilyIndices = nullptr;

    VmaAllocationCreateInfo allocationCreateInfo{};
    allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocationCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    allocationCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo allocationInfo;
    ASSERT_SUCCESS(vmaCreateBuffer(_device.allocator, &bufferCreateInfo, &allocationCreateInfo, &buffer, &allocation, &allocationInfo));

    void* pMappedData = allocationInfo.pMappedData;

    _Buffer _buffer{};
    _buffer.buffer = buffer;
    _buffer.allocation = allocation;
    _buffer.pMappedData = pMappedData;

    return _buffers.create(_buffer);
}

void Device::destroyBuffer(Buffer buffer)
{
    _Device& _device = _devices[*this];
    _Buffer& _buffer = _buffers[buffer];

    vmaDestroyBuffer(_device.allocator, _buffer.buffer, _buffer.allocation);

    _buffers.destroy(buffer);
}

CommandBuffer Device::createCommandBuffer()
{
    _Device& _device = _devices[*this];

    VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.pNext = nullptr;
    commandBufferAllocateInfo.commandPool = _device.graphicsCommandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    ASSERT_SUCCESS(vkAllocateCommandBuffers(_device.device, &commandBufferAllocateInfo, &commandBuffer));

    _CommandBuffer _commandBuffer{};
    _commandBuffer.commandBuffer = commandBuffer;

    return _commandBuffers.create(_commandBuffer);
}

void Device::destroyCommandBuffer(CommandBuffer commandBuffer)
{
    _Device& _device = _devices[*this];
    _CommandBuffer& _commandBuffer = _commandBuffers[commandBuffer];

    vkFreeCommandBuffers(_device.device, _device.graphicsCommandPool, 1, &_commandBuffer.commandBuffer);

    _commandBuffers.destroy(commandBuffer);
}

Fence Device::createFence()
{
    _Device& _device = _devices[*this];

    VkFenceCreateInfo fenceCreateInfo{};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.pNext = nullptr;
    fenceCreateInfo.flags = 0;

    VkFence fence;
    ASSERT_SUCCESS(vkCreateFence(_device.device, &fenceCreateInfo, nullptr, &fence));

    _Fence _fence{};
    _fence.fence = fence;

    return _fences.create(_fence);
}

void Device::destroyFence(Fence fence)
{
    _Device& _device = _devices[*this];
    _Fence& _fence = _fences[fence];

    vkDestroyFence(_device.device, _fence.fence, nullptr);

    _fences.destroy(fence);
}

void Device::submit(CommandBuffer commandBuffer, Fence fence)
{
    _Device& _device = _devices[*this];
    _CommandBuffer& _commandBuffer = _commandBuffers[commandBuffer];

    VkFence signalFence = fence.handle == 0 ? VK_NULL_HANDLE : _fences[fence].fence;

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = nullptr;
    submitInfo.waitSemaphoreCount = 0;
    submitInfo.pWaitSemaphores = nullptr;
    submitInfo.pWaitDstStageMask = nullptr;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &_commandBuffer.commandBuffer;
    submitInfo.signalSemaphoreCount = 0;
    submitInfo.pSignalSemaphores = nullptr;

    VkQueue graphicsQueue;
    vkGetDeviceQueue(_device.device, _device.graphicsQueueFamilyIndex, 0, &graphicsQueue);

    ASSERT_SUCCESS(vkQueueSubmit(graphicsQueue, 1, &submitInfo, signalFence));
}

void Device::waitForFence(Fence fence)
{
    _Device& _device = _devices[*this];
    _Fence& _fence = _fences[fence];

    ASSERT_SUCCESS(vkWaitForFences(_device.device, 1, &_fence.fence, VK_TRUE, UINT64_MAX));
}

SwapchainResult Device::acquireSwapchainImage(Swapchain swapchain, Image* image)
{
    _Device& _device = _devices[*this];
    _Swapchain& _swapchain = _swapchains[swapchain];

    VkFenceCreateInfo fenceCreateInfo{};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.pNext = nullptr;
    fenceCreateInfo.flags = 0;

    VkFence acquireFence;
    ASSERT_SUCCESS(vkCreateFence(_device.device, &fenceCreateInfo, nullptr, &acquireFence));

    VkSemaphoreCreateInfo semaphoreCreateInfo{};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreCreateInfo.pNext = nullptr;
    semaphoreCreateInfo.flags = 0;

    VkSemaphore acquireSemaphore;
    ASSERT_SUCCESS(vkCreateSemaphore(_device.device, &semaphoreCreateInfo, nullptr, &acquireSemaphore));

    uint32_t imageIndex;
    VkResult acquireResult = vkAcquireNextImageKHR(_device.device, _swapchain.swapchain, UINT64_MAX, acquireSemaphore, VK_NULL_HANDLE, &imageIndex);

    assert(acquireResult == VK_SUCCESS || acquireResult == VK_ERROR_OUT_OF_DATE_KHR || acquireResult == VK_SUBOPTIMAL_KHR);

    *image = _swapchain.images[imageIndex];

    _Image& _image = _images[*image];

    VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.pNext = nullptr;
    commandBufferAllocateInfo.commandPool = _device.graphicsCommandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    ASSERT_SUCCESS(vkAllocateCommandBuffers(_device.device, &commandBufferAllocateInfo, &commandBuffer));

    VkCommandBufferBeginInfo commandBufferBeginInfo{};
    commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    commandBufferBeginInfo.pNext = nullptr;
    commandBufferBeginInfo.flags = 0;
    commandBufferBeginInfo.pInheritanceInfo = nullptr;

    ASSERT_SUCCESS(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo));

    VkImageSubresourceRange subresourceRange{};
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = 1;
    subresourceRange.baseArrayLayer = 0;
    subresourceRange.layerCount = 1;

    VkImageMemoryBarrier imageMemoryBarrier{};
    imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageMemoryBarrier.pNext = nullptr;
    imageMemoryBarrier.srcAccessMask = 0;
    imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageMemoryBarrier.srcQueueFamilyIndex = _device.graphicsQueueFamilyIndex;
    imageMemoryBarrier.dstQueueFamilyIndex = _device.graphicsQueueFamilyIndex;
    imageMemoryBarrier.image = _image.image;
    imageMemoryBarrier.subresourceRange = subresourceRange;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);

    ASSERT_SUCCESS(vkEndCommandBuffer(commandBuffer));

    VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = nullptr;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &acquireSemaphore;
    submitInfo.pWaitDstStageMask = &dstStageMask;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount = 0;
    submitInfo.pSignalSemaphores = nullptr;

    VkQueue graphicsQueue;
    vkGetDeviceQueue(_device.device, _device.graphicsQueueFamilyIndex, 0, &graphicsQueue);

    ASSERT_SUCCESS(vkQueueSubmit(graphicsQueue, 1, &submitInfo, acquireFence));

    if (_swapchain.acquireFences[imageIndex] != VK_NULL_HANDLE)
    {
        ASSERT_SUCCESS(vkWaitForFences(_device.device, 1, &_swapchain.acquireFences[imageIndex], VK_TRUE, UINT64_MAX));

        vkDestroyFence(_device.device, _swapchain.acquireFences[imageIndex], nullptr);
        vkDestroySemaphore(_device.device, _swapchain.acquireSemaphores[imageIndex], nullptr);
    }

    _swapchain.acquireFences[imageIndex] = acquireFence;
    _swapchain.acquireSemaphores[imageIndex] = acquireSemaphore;

    SwapchainResult result = SwapchainResult::Success;

    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR)
    {
        result = SwapchainResult::OutOfDate;
    }

    if (acquireResult == VK_SUBOPTIMAL_KHR)
    {
        result = SwapchainResult::Suboptimal;
    }

    return result;
}

SwapchainResult Device::presentSwapchainImage(Swapchain swapchain, Image image)
{
    _Device& _device = _devices[*this];
    _Swapchain& _swapchain = _swapchains[swapchain];
    _Image& _image = _images[image];

    uint32_t imageIndex = SWAPCHAIN_IMAGE_COUNT;

    for (uint32_t i = 0; i < SWAPCHAIN_IMAGE_COUNT; i++)
    {
        if (_swapchain.images[i].handle == image.handle)
        {
            imageIndex = i;
        }
    }

    // image not part of swapchain
    assert(imageIndex != SWAPCHAIN_IMAGE_COUNT);

    VkQueue graphicsQueue;
    vkGetDeviceQueue(_device.device, _device.graphicsQueueFamilyIndex, 0, &graphicsQueue);

    VkQueue computeQueue;
    vkGetDeviceQueue(_device.device, _device.computeQueueFamilyIndex, 0, &computeQueue);

    VkFenceCreateInfo fenceCreateInfo{};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.pNext = nullptr;
    fenceCreateInfo.flags = 0;

    VkFence presentFence;
    ASSERT_SUCCESS(vkCreateFence(_device.device, &fenceCreateInfo, nullptr, &presentFence));

    VkSemaphoreCreateInfo semaphoreCreateInfo{};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreCreateInfo.pNext = nullptr;
    semaphoreCreateInfo.flags = 0;

    VkSemaphore presentSemaphore;
    ASSERT_SUCCESS(vkCreateSemaphore(_device.device, &semaphoreCreateInfo, nullptr, &presentSemaphore));

    VkSemaphore transferSemaphore = VK_NULL_HANDLE;
    VkCommandBuffer graphicsCommandBuffer = VK_NULL_HANDLE;
    VkCommandBuffer computeCommandBuffer = VK_NULL_HANDLE;

    VkResult presentResult{};
    if (_device.graphicsQueueFamilyIndex == _device.computeQueueFamilyIndex)
    {
        VkCommandBufferAllocateInfo graphicsCommandBufferAllocateInfo{};
        graphicsCommandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        graphicsCommandBufferAllocateInfo.pNext = nullptr;
        graphicsCommandBufferAllocateInfo.commandPool = _device.graphicsCommandPool;
        graphicsCommandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        graphicsCommandBufferAllocateInfo.commandBufferCount = 1;

        ASSERT_SUCCESS(vkAllocateCommandBuffers(_device.device, &graphicsCommandBufferAllocateInfo, &graphicsCommandBuffer));

        VkCommandBufferBeginInfo graphicsCommandBufferBeginInfo{};
        graphicsCommandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        graphicsCommandBufferBeginInfo.pNext = nullptr;
        graphicsCommandBufferBeginInfo.flags = 0;
        graphicsCommandBufferBeginInfo.pInheritanceInfo = nullptr;

        ASSERT_SUCCESS(vkBeginCommandBuffer(graphicsCommandBuffer, &graphicsCommandBufferBeginInfo));

        VkImageSubresourceRange subresourceRange{};
        subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresourceRange.baseMipLevel = 0;
        subresourceRange.levelCount = 1;
        subresourceRange.baseArrayLayer = 0;
        subresourceRange.layerCount = 1;

        VkImageMemoryBarrier imageMemoryBarrier{};
        imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imageMemoryBarrier.pNext = nullptr;
        imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        imageMemoryBarrier.dstAccessMask = 0;
        imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        imageMemoryBarrier.srcQueueFamilyIndex = _device.graphicsQueueFamilyIndex;
        imageMemoryBarrier.dstQueueFamilyIndex = _device.graphicsQueueFamilyIndex;
        imageMemoryBarrier.image = _image.image;
        imageMemoryBarrier.subresourceRange = subresourceRange;

        vkCmdPipelineBarrier(graphicsCommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);

        ASSERT_SUCCESS(vkEndCommandBuffer(graphicsCommandBuffer));

        VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

        VkSubmitInfo graphicsSubmitInfo{};
        graphicsSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        graphicsSubmitInfo.pNext = nullptr;
        graphicsSubmitInfo.waitSemaphoreCount = 0;
        graphicsSubmitInfo.pWaitSemaphores = nullptr;
        graphicsSubmitInfo.pWaitDstStageMask = nullptr;
        graphicsSubmitInfo.commandBufferCount = 1;
        graphicsSubmitInfo.pCommandBuffers = &graphicsCommandBuffer;
        graphicsSubmitInfo.signalSemaphoreCount = 1;
        graphicsSubmitInfo.pSignalSemaphores = &presentSemaphore;

        ASSERT_SUCCESS(vkQueueSubmit(graphicsQueue, 1, &graphicsSubmitInfo, presentFence));

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.pNext = nullptr;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &presentSemaphore;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &_swapchain.swapchain;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr;

        presentResult = vkQueuePresentKHR(graphicsQueue, &presentInfo);

        assert(presentResult == VK_SUCCESS || presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR);
    }
    else
    {
        ASSERT_SUCCESS(vkCreateSemaphore(_device.device, &semaphoreCreateInfo, nullptr, &transferSemaphore));

        VkCommandBufferAllocateInfo graphicsCommandBufferAllocateInfo{};
        graphicsCommandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        graphicsCommandBufferAllocateInfo.pNext = nullptr;
        graphicsCommandBufferAllocateInfo.commandPool = _device.graphicsCommandPool;
        graphicsCommandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        graphicsCommandBufferAllocateInfo.commandBufferCount = 1;

        VkCommandBuffer graphicsCommandBuffer;
        ASSERT_SUCCESS(vkAllocateCommandBuffers(_device.device, &graphicsCommandBufferAllocateInfo, &graphicsCommandBuffer));

        VkCommandBufferBeginInfo graphicsCommandBufferBeginInfo{};
        graphicsCommandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        graphicsCommandBufferBeginInfo.pNext = nullptr;
        graphicsCommandBufferBeginInfo.flags = 0;
        graphicsCommandBufferBeginInfo.pInheritanceInfo = nullptr;

        ASSERT_SUCCESS(vkBeginCommandBuffer(graphicsCommandBuffer, &graphicsCommandBufferBeginInfo));

        VkImageSubresourceRange subresourceRange{};
        subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresourceRange.baseMipLevel = 0;
        subresourceRange.levelCount = 1;
        subresourceRange.baseArrayLayer = 0;
        subresourceRange.layerCount = 1;

        // transfer ownership to compute queue

        VkImageMemoryBarrier releaseImageMemoryBarrier{};
        releaseImageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        releaseImageMemoryBarrier.pNext = nullptr;
        releaseImageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        releaseImageMemoryBarrier.dstAccessMask = 0;
        releaseImageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        releaseImageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        releaseImageMemoryBarrier.srcQueueFamilyIndex = _device.graphicsQueueFamilyIndex;
        releaseImageMemoryBarrier.dstQueueFamilyIndex = _device.computeQueueFamilyIndex;
        releaseImageMemoryBarrier.image = _image.image;
        releaseImageMemoryBarrier.subresourceRange = subresourceRange;

        vkCmdPipelineBarrier(graphicsCommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &releaseImageMemoryBarrier);

        ASSERT_SUCCESS(vkEndCommandBuffer(graphicsCommandBuffer));

        VkSubmitInfo graphicsSubmitInfo{};
        graphicsSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        graphicsSubmitInfo.pNext = nullptr;
        graphicsSubmitInfo.waitSemaphoreCount = 0;
        graphicsSubmitInfo.pWaitSemaphores = nullptr;
        graphicsSubmitInfo.pWaitDstStageMask = nullptr;
        graphicsSubmitInfo.commandBufferCount = 1;
        graphicsSubmitInfo.pCommandBuffers = &graphicsCommandBuffer;
        graphicsSubmitInfo.signalSemaphoreCount = 1;
        graphicsSubmitInfo.pSignalSemaphores = &transferSemaphore;

        ASSERT_SUCCESS(vkQueueSubmit(graphicsQueue, 1, &graphicsSubmitInfo, VK_NULL_HANDLE));

        VkCommandBufferAllocateInfo computeCommandBufferAllocateInfo{};
        computeCommandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        computeCommandBufferAllocateInfo.pNext = nullptr;
        computeCommandBufferAllocateInfo.commandPool = _device.computeCommandPool;
        computeCommandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        computeCommandBufferAllocateInfo.commandBufferCount = 1;

        VkCommandBuffer computeCommandBuffer;
        ASSERT_SUCCESS(vkAllocateCommandBuffers(_device.device, &computeCommandBufferAllocateInfo, &computeCommandBuffer));

        VkCommandBufferBeginInfo computeCommandBufferBeginInfo{};
        computeCommandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        computeCommandBufferBeginInfo.pNext = nullptr;
        computeCommandBufferBeginInfo.flags = 0;
        computeCommandBufferBeginInfo.pInheritanceInfo = nullptr;

        ASSERT_SUCCESS(vkBeginCommandBuffer(computeCommandBuffer, &computeCommandBufferBeginInfo));

        // acquire ownership from graphics queue

        VkImageMemoryBarrier acquireImageMemoryBarrier{};
        acquireImageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        acquireImageMemoryBarrier.pNext = nullptr;
        acquireImageMemoryBarrier.srcAccessMask = 0;
        acquireImageMemoryBarrier.dstAccessMask = 0;
        acquireImageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        acquireImageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        acquireImageMemoryBarrier.srcQueueFamilyIndex = _device.graphicsQueueFamilyIndex;
        acquireImageMemoryBarrier.dstQueueFamilyIndex = _device.computeQueueFamilyIndex;
        acquireImageMemoryBarrier.image = _image.image;
        acquireImageMemoryBarrier.subresourceRange = subresourceRange;

        vkCmdPipelineBarrier(computeCommandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &acquireImageMemoryBarrier);

        ASSERT_SUCCESS(vkEndCommandBuffer(computeCommandBuffer));

        // is this optimal?
        VkPipelineStageFlags transferWaitDstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

        VkSubmitInfo computeSubmitInfo{};
        computeSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        computeSubmitInfo.pNext = nullptr;
        computeSubmitInfo.waitSemaphoreCount = 1;
        computeSubmitInfo.pWaitSemaphores = &transferSemaphore;
        computeSubmitInfo.pWaitDstStageMask = &transferWaitDstStageMask;
        computeSubmitInfo.commandBufferCount = 1;
        computeSubmitInfo.pCommandBuffers = &computeCommandBuffer;
        computeSubmitInfo.signalSemaphoreCount = 1;
        computeSubmitInfo.pSignalSemaphores = &presentSemaphore;

        ASSERT_SUCCESS(vkQueueSubmit(computeQueue, 1, &computeSubmitInfo, presentFence));

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.pNext = nullptr;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &presentSemaphore;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &_swapchain.swapchain;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr;

        presentResult = vkQueuePresentKHR(computeQueue, &presentInfo);

        assert(presentResult == VK_SUCCESS || presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR);
    }

    if (_swapchain.presentFences[imageIndex] != VK_NULL_HANDLE)
    {
        ASSERT_SUCCESS(vkWaitForFences(_device.device, 1, &_swapchain.presentFences[imageIndex], VK_TRUE, UINT64_MAX));

        vkDestroyFence(_device.device, _swapchain.presentFences[imageIndex], nullptr);
        vkDestroySemaphore(_device.device, _swapchain.transferSemaphores[imageIndex], nullptr);
        vkFreeCommandBuffers(_device.device, _device.graphicsCommandPool, 1, &_swapchain.graphicsCommandBuffers[imageIndex]);
        if (_swapchain.computeCommandBuffers[imageIndex] != VK_NULL_HANDLE)
        {
            vkFreeCommandBuffers(_device.device, _device.computeCommandPool, 1, &_swapchain.computeCommandBuffers[imageIndex]);
        }
    }

    _swapchain.presentFences[imageIndex] = presentFence;
    _swapchain.transferSemaphores[imageIndex] = transferSemaphore;
    _swapchain.graphicsCommandBuffers[imageIndex] = graphicsCommandBuffer;
    _swapchain.computeCommandBuffers[imageIndex] = computeCommandBuffer;

    // there is no way to check if this semaphore is good or not https://github.com/KhronosGroup/Vulkan-Docs/issues/2007
    // could cause issues if multiple swapchains are used concurrently and PRESENT_SEMAPHORE_COUNT isn't big enough
    // something rendered PRESENT_SEMAPHORE_COUNT frames ago is likely done so it should be fine
    vkDestroySemaphore(_device.device, _device.presentSemaphores[_device.presentSemaphoreIndex], nullptr);

    _device.presentSemaphores[_device.presentSemaphoreIndex] = presentSemaphore;
    _device.presentSemaphoreIndex = (_device.presentSemaphoreIndex + 1) % PRESENT_SEMAPHORE_COUNT;

    SwapchainResult result = SwapchainResult::Success;

    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR)
    {
        result = SwapchainResult::OutOfDate;
    }

    if (presentResult == VK_SUBOPTIMAL_KHR)
    {
        result = SwapchainResult::Suboptimal;
    }

    return result;
}

PipelineLayout Device::createPipelineLayout(uint32_t descriptorCount, const DescriptorType* descriptorTypes)
{
    _Device& _device = _devices[*this];

    uint32_t imageCount = 0;
    uint32_t bufferCount = 0;

    std::vector<VkDescriptorSetLayoutBinding> bindings(descriptorCount);

    for (uint32_t i = 0; i < descriptorCount; i++)
    {
        VkDescriptorType descriptorType{};

        if (descriptorTypes[i] == DescriptorType::UniformBuffer)
        {
            bufferCount++;
            descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        }

        if (descriptorTypes[i] == DescriptorType::StorageImage)
        {
            imageCount++;
            descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        }

        VkDescriptorSetLayoutBinding binding{};
        binding.binding = i;
        binding.descriptorType = descriptorType;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        binding.pImmutableSamplers = nullptr;

        bindings[i] = binding;
    }

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{};
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.pNext = nullptr;
    descriptorSetLayoutCreateInfo.flags = 0;
    descriptorSetLayoutCreateInfo.bindingCount = descriptorCount;
    descriptorSetLayoutCreateInfo.pBindings = bindings.data();

    VkDescriptorSetLayout descriptorSetLayout;
    ASSERT_SUCCESS(vkCreateDescriptorSetLayout(_device.device, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayout));

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.pNext = nullptr;
    pipelineLayoutCreateInfo.flags = 0;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
    pipelineLayoutCreateInfo.pPushConstantRanges = nullptr;

    VkPipelineLayout pipelineLayout;
    ASSERT_SUCCESS(vkCreatePipelineLayout(_device.device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

    _PipelineLayout _pipelinelayout{};
    _pipelinelayout.descriptorSetLayout = descriptorSetLayout;
    _pipelinelayout.pipelineLayout = pipelineLayout;
    _pipelinelayout.imageCount = imageCount;
    _pipelinelayout.bufferCount = bufferCount;

    return _pipelineLayouts.create(_pipelinelayout);
}

void Device::destroyPipelineLayout(PipelineLayout pipelineLayout)
{
    _Device& _device = _devices[*this];
    _PipelineLayout& _pipelinelayout = _pipelineLayouts[pipelineLayout];

    vkDestroyPipelineLayout(_device.device, _pipelinelayout.pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(_device.device, _pipelinelayout.descriptorSetLayout, nullptr);

    _pipelineLayouts.destroy(pipelineLayout);
}

DescriptorPool Device::createDescriptorPool(PipelineLayout pipelineLayout, uint32_t descriptorSetCout)
{
    _Device& _device = _devices[*this];
    _PipelineLayout& _pipelinelayout = _pipelineLayouts[pipelineLayout];

    VkDescriptorPoolSize poolSizes[2];
    uint32_t poolSizeCount = 0;

    if (_pipelinelayout.imageCount > 0)
    {
        VkDescriptorPoolSize imagePoolSize{};
        imagePoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        imagePoolSize.descriptorCount = descriptorSetCout * _pipelinelayout.imageCount;

        poolSizes[poolSizeCount++] = imagePoolSize;
    }

    if (_pipelinelayout.bufferCount > 0)
    {
        VkDescriptorPoolSize bufferPoolSize{};
        bufferPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bufferPoolSize.descriptorCount = descriptorSetCout * _pipelinelayout.bufferCount;

        poolSizes[poolSizeCount++] = bufferPoolSize;
    }

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo{};
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.pNext = nullptr;
    descriptorPoolCreateInfo.flags = 0;
    descriptorPoolCreateInfo.maxSets = descriptorSetCout;
    descriptorPoolCreateInfo.poolSizeCount = poolSizeCount;
    descriptorPoolCreateInfo.pPoolSizes = poolSizes;

    VkDescriptorPool descriptorPool;
    ASSERT_SUCCESS(vkCreateDescriptorPool(_device.device, &descriptorPoolCreateInfo, nullptr, &descriptorPool));

    std::vector<VkDescriptorSetLayout> descriptorSetLayouts(descriptorSetCout);
    for (uint32_t i = 0; i < descriptorSetCout; i++)
    {
        descriptorSetLayouts[i] = _pipelinelayout.descriptorSetLayout;
    }

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.pNext = nullptr;
    descriptorSetAllocateInfo.descriptorPool = descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = descriptorSetCout;
    descriptorSetAllocateInfo.pSetLayouts = descriptorSetLayouts.data();

    std::vector<VkDescriptorSet> descriptorSets(descriptorSetCout);
    ASSERT_SUCCESS(vkAllocateDescriptorSets(_device.device, &descriptorSetAllocateInfo, descriptorSets.data()));

    _DescriptorPool _descriptorPool{};
    _descriptorPool.descriptorPool = descriptorPool;
    _descriptorPool.descriptorSets = descriptorSets;

    return _descriptorPools.create(_descriptorPool);
}

void Device::destroyDescriptorPool(DescriptorPool descriptorPool)
{
    _Device& _device = _devices[*this];
    _DescriptorPool& _descriptorPool = _descriptorPools[descriptorPool];

    vkDestroyDescriptorPool(_device.device, _descriptorPool.descriptorPool, nullptr);

    _descriptorPools.destroy(descriptorPool);
}

Pipeline Device::createPipeline(PipelineLayout pipelineLayout, uint64_t codeSize, const uint32_t* code)
{
    _Device& _device = _devices[*this];
    _PipelineLayout& _pipelinelayout = _pipelineLayouts[pipelineLayout];

    VkShaderModuleCreateInfo shaderModuleCreateInfo{};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.pNext = nullptr;
    shaderModuleCreateInfo.flags = 0;
    shaderModuleCreateInfo.codeSize = codeSize;
    shaderModuleCreateInfo.pCode = code;

    VkShaderModule shaderModule;
    ASSERT_SUCCESS(vkCreateShaderModule(_device.device, &shaderModuleCreateInfo, nullptr, &shaderModule));

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
    pipelineCreateInfo.layout = _pipelinelayout.pipelineLayout;
    pipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineCreateInfo.basePipelineIndex = 0;

    VkPipeline pipeline;
    ASSERT_SUCCESS(vkCreateComputePipelines(_device.device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline));

    vkDestroyShaderModule(_device.device, shaderModule, nullptr);

    _Pipeline _pipeline{};
    _pipeline.pipeline = pipeline;

    return _pipelines.create(_pipeline);
}

void Device::destroyPipeline(Pipeline pipeline)
{
    _Device& _device = _devices[*this];
    _Pipeline& _pipeline = _pipelines[pipeline];

    vkDestroyPipeline(_device.device, _pipeline.pipeline, nullptr);

    _pipelines.destroy(pipeline);
}

void Device::updateDescriptorSet(DescriptorPool descriptorPool, uint32_t descriptorSetIndex, uint32_t descriptorCount, const Descriptor* descriptors)
{
    _Device& _device = _devices[*this];
    _DescriptorPool& _descriptorPool = _descriptorPools[descriptorPool];

    VkDescriptorSet descriptorSet = _descriptorPool.descriptorSets[descriptorSetIndex];

    std::vector<VkDescriptorImageInfo> imageInfos{};
    std::vector<VkDescriptorBufferInfo> bufferInfos{};
    for (uint32_t i = 0; i < descriptorCount; i++)
    {
        Descriptor descriptor = descriptors[i];

        if (descriptor.descriptorType == DescriptorType::StorageImage)
        {
            _Image image = _images[descriptor.image];

            VkDescriptorImageInfo imageInfo{};
            imageInfo.sampler = VK_NULL_HANDLE;
            imageInfo.imageView = image.imageView;
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            imageInfos.push_back(imageInfo);
        }

        if (descriptor.descriptorType == DescriptorType::UniformBuffer)
        {
            _Buffer _buffer = _buffers[descriptor.buffer];

            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = _buffer.buffer;
            bufferInfo.offset = 0;
            bufferInfo.range = VK_WHOLE_SIZE;

            bufferInfos.push_back(bufferInfo);
        }
    }

    uint32_t nextImageInfoIndex = 0;
    uint32_t nextBufferInfoIndex = 0;
    std::vector<VkWriteDescriptorSet> descriptorWrites(descriptorCount);
    for (uint32_t i = 0; i < descriptorCount; i++)
    {
        Descriptor descriptor = descriptors[i];

        VkDescriptorType descriptorType{};
        VkDescriptorImageInfo* pImageInfo = nullptr;
        VkDescriptorBufferInfo* pBufferInfo = nullptr;

        if (descriptor.descriptorType == DescriptorType::StorageImage)
        {
            descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            pImageInfo = &imageInfos[nextImageInfoIndex++];
        }

        if (descriptor.descriptorType == DescriptorType::UniformBuffer)
        {
            descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            pBufferInfo = &bufferInfos[nextBufferInfoIndex++];
        }

        VkWriteDescriptorSet nativeDescriptorWrite{};
        nativeDescriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        nativeDescriptorWrite.pNext = nullptr;
        nativeDescriptorWrite.dstSet = descriptorSet;
        nativeDescriptorWrite.dstBinding = i;
        nativeDescriptorWrite.dstArrayElement = 0;
        nativeDescriptorWrite.descriptorCount = 1;
        nativeDescriptorWrite.descriptorType = descriptorType;
        nativeDescriptorWrite.pImageInfo = pImageInfo;
        nativeDescriptorWrite.pBufferInfo = pBufferInfo;
        nativeDescriptorWrite.pTexelBufferView = nullptr;

        descriptorWrites[i] = nativeDescriptorWrite;
    }

    vkUpdateDescriptorSets(_device.device, descriptorCount, descriptorWrites.data(), 0, nullptr);
}

void Device::udpateBuffer(Buffer buffer, uint64_t dataOffset, uint64_t dataSize, const void* data)
{
    _Device& _device = _devices[*this];
    _Buffer& _buffer = _buffers[buffer];

    memcpy(reinterpret_cast<char*>(_buffer.pMappedData) + dataOffset, data, dataSize);

    vmaFlushAllocation(_device.allocator, _buffer.allocation, dataOffset, dataSize);
}

Swapchain Device::createSwapchain(uint32_t width, uint32_t height, Surface surface)
{
    _Device& _device = _devices[*this];
    _Surface& _surface = _surfaces[surface];

    VkExtent2D extent{};
    extent.width = width;
    extent.height = height;

    VkSwapchainCreateInfoKHR swapchainCreateInfo{};
    swapchainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchainCreateInfo.pNext = nullptr;
    swapchainCreateInfo.flags = 0;
    swapchainCreateInfo.surface = _surface.surface;
    swapchainCreateInfo.minImageCount = SWAPCHAIN_IMAGE_COUNT;
    swapchainCreateInfo.imageFormat = VK_FORMAT_B8G8R8A8_UNORM;
    swapchainCreateInfo.imageColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
    swapchainCreateInfo.imageExtent = extent;
    swapchainCreateInfo.imageArrayLayers = 1;
    swapchainCreateInfo.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT;
    swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchainCreateInfo.queueFamilyIndexCount = 0;
    swapchainCreateInfo.pQueueFamilyIndices = nullptr;
    swapchainCreateInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    swapchainCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchainCreateInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;             // TODO
    swapchainCreateInfo.clipped = VK_TRUE;
    swapchainCreateInfo.oldSwapchain = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain;
    ASSERT_SUCCESS(vkCreateSwapchainKHR(_device.device, &swapchainCreateInfo, nullptr, &swapchain));

    VkImage nativeImages[SWAPCHAIN_IMAGE_COUNT];
    uint32_t swapchainImageCount = SWAPCHAIN_IMAGE_COUNT;
    ASSERT_SUCCESS(vkGetSwapchainImagesKHR(_device.device, swapchain, &swapchainImageCount, nativeImages));

    assert(swapchainImageCount == SWAPCHAIN_IMAGE_COUNT);

    _Swapchain _swapchain{};
    _swapchain.swapchain = swapchain;

    for (uint32_t i = 0; i < SWAPCHAIN_IMAGE_COUNT; i++)
    {
        VkImageSubresourceRange subresourceRange{};
        subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresourceRange.baseMipLevel = 0;
        subresourceRange.levelCount = 1;
        subresourceRange.baseArrayLayer = 0;
        subresourceRange.layerCount = 1;

        VkComponentMapping componentMapping{};
        componentMapping.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        componentMapping.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        componentMapping.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        componentMapping.a = VK_COMPONENT_SWIZZLE_IDENTITY;

        VkImageViewCreateInfo imageViewCreateInfo{};
        imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        imageViewCreateInfo.pNext = nullptr;
        imageViewCreateInfo.flags = 0;
        imageViewCreateInfo.image = nativeImages[i];
        imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        imageViewCreateInfo.format = VK_FORMAT_B8G8R8A8_UNORM;
        imageViewCreateInfo.components = componentMapping;
        imageViewCreateInfo.subresourceRange = subresourceRange;

        VkImageView imageView;
        ASSERT_SUCCESS(vkCreateImageView(_device.device, &imageViewCreateInfo, nullptr, &imageView));

        _Image _image{};
        _image.image = nativeImages[i];
        _image.allocation = VMA_NULL;
        _image.imageView = imageView;
        _image.commandBuffer = VK_NULL_HANDLE;
        _image.fence = VK_NULL_HANDLE;

        Image image = _images.create(_image);

        _swapchain.images[i] = image;
        _swapchain.acquireFences[i] = VK_NULL_HANDLE;
        _swapchain.acquireSemaphores[i] = VK_NULL_HANDLE;
        _swapchain.presentFences[i] = VK_NULL_HANDLE;
        _swapchain.graphicsCommandBuffers[i] = VK_NULL_HANDLE;
        _swapchain.computeCommandBuffers[i] = VK_NULL_HANDLE;
        _swapchain.transferSemaphores[i] = VK_NULL_HANDLE;
    }

    return _swapchains.create(_swapchain);
}

void Device::destroySwapchain(Swapchain swapchain)
{
    _Device& _device = _devices[*this];
    _Swapchain& _swapchain = _swapchains[swapchain];

    for (uint32_t i = 0; i < SWAPCHAIN_IMAGE_COUNT; i++)
    {
        if (_swapchain.acquireFences[i] != VK_NULL_HANDLE)
        {
            ASSERT_SUCCESS(vkWaitForFences(_device.device, 1, &_swapchain.acquireFences[i], VK_TRUE, UINT64_MAX));

            vkDestroyFence(_device.device, _swapchain.acquireFences[i], nullptr);
            vkDestroySemaphore(_device.device, _swapchain.acquireSemaphores[i], nullptr);
        }

        if (_swapchain.presentFences[i] != VK_NULL_HANDLE)
        {
            ASSERT_SUCCESS(vkWaitForFences(_device.device, 1, &_swapchain.presentFences[i], VK_TRUE, UINT64_MAX));

            vkDestroyFence(_device.device, _swapchain.presentFences[i], nullptr);
            vkDestroySemaphore(_device.device, _swapchain.transferSemaphores[i], nullptr);
            vkFreeCommandBuffers(_device.device, _device.graphicsCommandPool, 1, &_swapchain.graphicsCommandBuffers[i]);
            if (_swapchain.computeCommandBuffers[i] != VK_NULL_HANDLE)
            {
                vkFreeCommandBuffers(_device.device, _device.computeCommandPool, 1, &_swapchain.computeCommandBuffers[i]);
            }
        }

        _Image& _image = _images[_swapchain.images[i]];

        vkDestroyImageView(_device.device, _image.imageView, nullptr);

        _images.destroy(_swapchain.images[i]);
    }

    vkDestroySwapchainKHR(_device.device, _swapchain.swapchain, nullptr);

    _swapchains.destroy(swapchain);
}

Image Device::createImage(uint32_t width, uint32_t height, Format format)
{
    _Device& _device = _devices[*this];

    VkFormat nativeFormat{};

    if (format == Format::r32_uint)
    {
        nativeFormat = VK_FORMAT_R32_UINT;
    }
    if (format == Format::r8g8b8a8_unorm)
    {
        nativeFormat = VK_FORMAT_R8G8B8A8_UNORM;
    }

    VkImageViewType nativeImageType{};

    VkExtent3D extent{};
    extent.width = width;
    extent.height = height;
    extent.depth = 1;

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.pNext = nullptr;
    imageCreateInfo.flags = 0;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = nativeFormat;
    imageCreateInfo.extent = extent;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.queueFamilyIndexCount = 0;
    imageCreateInfo.pQueueFamilyIndices = nullptr;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocationCreateInfo{};
    allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocationCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkImage image;
    VmaAllocation allocation;
    ASSERT_SUCCESS(vmaCreateImage(_device.allocator, &imageCreateInfo, &allocationCreateInfo, &image, &allocation, nullptr));

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
    imageViewCreateInfo.image = image;
    imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageViewCreateInfo.format = nativeFormat;
    imageViewCreateInfo.components = {};
    imageViewCreateInfo.subresourceRange = subresourceRange;

    VkImageView imageView;
    ASSERT_SUCCESS(vkCreateImageView(_device.device, &imageViewCreateInfo, nullptr, &imageView));

    VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.pNext = nullptr;
    commandBufferAllocateInfo.commandPool = _device.graphicsCommandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    ASSERT_SUCCESS(vkAllocateCommandBuffers(_device.device, &commandBufferAllocateInfo, &commandBuffer));

    VkCommandBufferBeginInfo commandBufferBeginInfo{};
    commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    commandBufferBeginInfo.pNext = nullptr;
    commandBufferBeginInfo.flags = 0;
    commandBufferBeginInfo.pInheritanceInfo = nullptr;

    ASSERT_SUCCESS(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo));

    VkImageMemoryBarrier imageMemoryBarrier{};
    imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageMemoryBarrier.pNext = nullptr;
    imageMemoryBarrier.srcAccessMask = 0;
    imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageMemoryBarrier.srcQueueFamilyIndex = _device.graphicsQueueFamilyIndex;
    imageMemoryBarrier.dstQueueFamilyIndex = _device.graphicsQueueFamilyIndex;
    imageMemoryBarrier.image = image;
    imageMemoryBarrier.subresourceRange = subresourceRange;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);

    ASSERT_SUCCESS(vkEndCommandBuffer(commandBuffer));

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = nullptr;
    submitInfo.waitSemaphoreCount = 0;
    submitInfo.pWaitSemaphores = nullptr;
    submitInfo.pWaitDstStageMask = nullptr;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount = 0;
    submitInfo.pSignalSemaphores = nullptr;

    VkQueue graphicsQueue;
    vkGetDeviceQueue(_device.device, _device.graphicsQueueFamilyIndex, 0, &graphicsQueue);

    VkFenceCreateInfo fenceCreateInfo{};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.pNext = nullptr;
    fenceCreateInfo.flags = 0;

    VkFence fence;
    ASSERT_SUCCESS(vkCreateFence(_device.device, &fenceCreateInfo, nullptr, &fence));

    ASSERT_SUCCESS(vkQueueSubmit(graphicsQueue, 1, &submitInfo, fence));

    _Image _image{};
    _image.image = image;
    _image.allocation = allocation;
    _image.imageView = imageView;
    _image.commandBuffer = commandBuffer;
    _image.fence = fence;

    return _images.create(_image);
}

void Device::destroyImage(Image image)
{
    _Device& _device = _devices[*this];
    _Image& _image = _images[image];

    // tried to delete a swapchain image
    assert(_image.fence != VK_NULL_HANDLE);

    ASSERT_SUCCESS(vkWaitForFences(_device.device, 1, &_image.fence, VK_TRUE, UINT64_MAX));

    vkFreeCommandBuffers(_device.device, _device.graphicsCommandPool, 1, &_image.commandBuffer);
    vkDestroyFence(_device.device, _image.fence, nullptr);
    vkDestroyImageView(_device.device, _image.imageView, nullptr);
    vmaDestroyImage(_device.allocator, _image.image, _image.allocation);

    _images.destroy(image);
}

void CommandBuffer::begin()
{
    _CommandBuffer& _commandBuffer = _commandBuffers[*this];

    VkCommandBufferBeginInfo commandBufferBeginInfo{};
    commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    commandBufferBeginInfo.pNext = nullptr;
    commandBufferBeginInfo.flags = 0;
    commandBufferBeginInfo.pInheritanceInfo = nullptr;

    ASSERT_SUCCESS(vkBeginCommandBuffer(_commandBuffer.commandBuffer, &commandBufferBeginInfo));
}

void CommandBuffer::end()
{
    _CommandBuffer& _commandBuffer = _commandBuffers[*this];

    ASSERT_SUCCESS(vkEndCommandBuffer(_commandBuffer.commandBuffer));
}

void CommandBuffer::bindDescriptorSet(PipelineLayout pipelineLayout, DescriptorPool descriptorPool, uint32_t descriptorIndex)
{
    _CommandBuffer& _commandBuffer = _commandBuffers[*this];
    _PipelineLayout& _pipelineLayout = _pipelineLayouts[pipelineLayout];
    _DescriptorPool& _descriptorPool = _descriptorPools[descriptorPool];

    vkCmdBindDescriptorSets(_commandBuffer.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineLayout.pipelineLayout, 0, 1, &_descriptorPool.descriptorSets[descriptorIndex], 0, nullptr);
}

void CommandBuffer::bindPipeline(Pipeline pipeline)
{
    _CommandBuffer& _commandBuffer = _commandBuffers[*this];
    _Pipeline& _pipeline = _pipelines[pipeline];

    vkCmdBindPipeline(_commandBuffer.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline.pipeline);
}

void CommandBuffer::dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
    _CommandBuffer& _commandBuffer = _commandBuffers[*this];

    vkCmdDispatch(_commandBuffer.commandBuffer, groupCountX, groupCountY, groupCountZ);
}

void CommandBuffer::barrier()
{
    _CommandBuffer& _commandBuffer = _commandBuffers[*this];

    vkCmdPipelineBarrier(_commandBuffer.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 0, nullptr);
}

void CommandBuffer::memoryBarrier()
{
    _CommandBuffer& _commandBuffer = _commandBuffers[*this];

    VkMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.pNext = nullptr;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(_commandBuffer.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
}

VkInstance getVkInstance(Device device)
{
    _Device& _device = _devices[device];

    return _device.instance;
}

VkSurfaceKHR getVkSurfaceKHR(Surface surface)
{
    _Surface& _surface = _surfaces[surface];

    return _surface.surface;
}

Surface createSurface(VkSurfaceKHR surface)
{
    _Surface _surface{};
    _surface.surface = surface;

    return _surfaces.create(_surface);
}

void destroySurface(Surface surface)
{
    _surfaces.destroy(surface);
}
