#include "abstraction_native.h"
#include "abstraction_glfw.h"

#include <assert.h>

Surface createGLFWSurface(Device device, GLFWwindow* window)
{
	VkSurfaceKHR surface;
	VkResult result = glfwCreateWindowSurface(getVkInstance(device), window, nullptr, &surface);

	assert(result == VK_SUCCESS);
	
	return createSurface(surface);
}

void destroyGLFWSurface(Device device, Surface surface)
{
	vkDestroySurfaceKHR(getVkInstance(device), getVkSurfaceKHR(surface), nullptr);

	destroySurface(surface);
}
