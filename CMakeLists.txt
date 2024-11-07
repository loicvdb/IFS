cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

set( GLFW_BUILD_DOCS OFF CACHE BOOL  "GLFW lib only" )
set( GLFW_INSTALL OFF CACHE BOOL  "GLFW lib only" )

add_subdirectory(glfw)

project(IFS)
add_executable(IFS)

target_include_directories(IFS PRIVATE
	src
)

target_sources(IFS PRIVATE
	src/main.cpp
)

target_link_libraries(IFS PRIVATE
	Vulkan::Vulkan
	glfw
)

## https://thatonegamedev.com/cpp/cmake/how-to-compile-shaders-with-cmake/
function(add_shaders TARGET_NAME)
	set(SHADER_SOURCE_FILES ${ARGN}) # the rest of arguments to this function will be assigned as shader source files
	
	# Validate that source files have been passed
	list(LENGTH SHADER_SOURCE_FILES FILE_COUNT)
	if(FILE_COUNT EQUAL 0)
		message(FATAL_ERROR "Cannot create a shaders target without any source files")
	endif()
	
	set(SHADER_COMMANDS)
	set(SHADER_PRODUCTS)
	
	foreach(SHADER_SOURCE IN LISTS SHADER_SOURCE_FILES)
		cmake_path(ABSOLUTE_PATH SHADER_SOURCE NORMALIZE)
		cmake_path(GET SHADER_SOURCE FILENAME SHADER_NAME)
		
		# Build command
		list(APPEND SHADER_COMMAND COMMAND)
		list(APPEND SHADER_COMMAND Vulkan::glslc)
		list(APPEND SHADER_COMMAND "${SHADER_SOURCE}")
		list(APPEND SHADER_COMMAND "-o")
		list(APPEND SHADER_COMMAND "${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_NAME}.spv")
		list(APPEND SHADER_COMMAND "-g")
	
		# Add product
		list(APPEND SHADER_PRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_NAME}.spv")
	
	endforeach()
	
	add_custom_target(${TARGET_NAME} ALL
		${SHADER_COMMAND}
		COMMENT "Compiling Shaders [${TARGET_NAME}]"
		SOURCES ${SHADER_SOURCE_FILES}
		BYPRODUCTS ${SHADER_PRODUCTS}
	)
endfunction()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shaders)

add_shaders(CompileShaders
	shaders/clear.comp
	shaders/display.comp
	shaders/splat.comp
)

add_dependencies(IFS CompileShaders)
