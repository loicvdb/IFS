#pragma once

#include "abstraction.h"
#include "GLFW/glfw3.h"

Surface createGLFWSurface(Device device, GLFWwindow* window);
void destroyGLFWSurface(Device device, Surface surface);
