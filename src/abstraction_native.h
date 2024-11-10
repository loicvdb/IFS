#pragma once

#include "vulkan/vulkan.h"
#include "abstraction.h"

VkInstance getVkInstance(Device device);
VkSurfaceKHR getVkSurfaceKHR(Surface surface);

Surface createSurface(VkSurfaceKHR surface);
void destroySurface(Surface surface);
