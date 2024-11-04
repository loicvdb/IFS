cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(GLFW3_LIB_PATH CACHE PATH "glfw3 lib folder (e.g. glfw-3.3.7.bin.WIN64/lib-vc2019)")
find_package(Vulkan REQUIRED)

project(IFS)
add_executable(IFS)

target_include_directories(IFS PRIVATE
	src
	"${GLFW3_LIB_PATH}/../include"
)

target_sources(IFS PRIVATE
	src/main.cpp
)

target_link_libraries(IFS PRIVATE
	Vulkan::Vulkan
	"${GLFW3_LIB_PATH}/glfw3.lib"
)
