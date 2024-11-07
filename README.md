# IFS
 Displaying Iterated Function Systems as point clouds
# Building
 You'll need the Vulkan SDK, you can find it over at https://vulkan.lunarg.com/sdk/home. The Vulkan SDK should include these optional components:
  - GLM Headers
  - VMA header

 If you have the SDK but you don't have these headers you can just use the maintenancetool program in the SDK to add them.

 You'll also need GLFW, download the precompiled binaries (not sources) at https://www.glfw.org/download.html.

 Then download CMake at https://cmake.org/download/ and run the CMakeLists.txt file with it. You will need to put the path to the GLFW binaries in the GLFW3_LIB_PATH variable, for instance C:/Users/Me/Documents/glfw-3.4.bin.WIN64/lib-vc2022
