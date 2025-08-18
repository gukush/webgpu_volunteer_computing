cmake_minimum_required(VERSION 3.20)
project(MultiFrameworkClient LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable vcpkg toolchain integration
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "")
endif()

# Find required packages via vcpkg
find_package(Boost REQUIRED COMPONENTS system url)
find_package(OpenSSL REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Threads REQUIRED)

# Optional: CUDA support
option(ENABLE_CUDA "Enable CUDA support" ON)
if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA found - enabling CUDA executor")
        set(HAVE_CUDA ON)
    else()
        message(WARNING "CUDA not found - disabling CUDA executor")
        set(HAVE_CUDA OFF)
    endif()
endif()

# Fix CUDA::cudart target if missing
if(HAVE_CUDA AND NOT TARGET CUDA::cudart)
    find_library(CUDART_LIBRARY NAMES cudart libcudart PATHS
        /usr/local/cuda/lib64
        /usr/local/cuda-*/targets/x86_64-linux/lib
        /usr/local/cuda-*/lib64
        NO_DEFAULT_PATH
    )

    find_path(CUDART_INCLUDE_DIR cuda_runtime.h PATHS
        /usr/local/cuda/include
        /usr/local/cuda-*/targets/x86_64-linux/include
        NO_DEFAULT_PATH
    )

    if(CUDART_LIBRARY)
        message(STATUS "Creating imported target CUDA::cudart -> ${CUDART_LIBRARY}")
        add_library(CUDA::cudart UNKNOWN IMPORTED)
        set_target_properties(CUDA::cudart PROPERTIES
            IMPORTED_LOCATION "${CUDART_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${CUDART_INCLUDE_DIR}"
        )
    else()
        message(WARNING "Could not find libcudart; CUDA runtime functions may link-fail")
        set(HAVE_CUDA OFF)
    endif()
endif()

# Fix CUDA::nvrtc target if missing
if(HAVE_CUDA AND NOT TARGET CUDA::nvrtc)
    message(STATUS "CUDA::nvrtc target not found — attempting manual nvrtc detection")

    set(_cuda_hint_paths
        $ENV{CUDA_HOME}
        /usr/local/cuda-13
        /usr/local/cuda
        /usr
        /usr/local
    )

    find_path(NVRTC_INCLUDE_DIR nvrtc.h
        HINTS ${_cuda_hint_paths}
        PATH_SUFFIXES include include/cuda include/x86_64-linux-gnu/include
    )

    find_library(NVRTC_LIBRARY NAMES nvrtc nvrtc.so nvrtc.so.13 nvrtc.so.12
        HINTS ${_cuda_hint_paths}
        PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib lib/x86_64-linux-gnu
    )

    if(NVRTC_LIBRARY AND NVRTC_INCLUDE_DIR)
        message(STATUS "Found nvrtc: ${NVRTC_LIBRARY}; include: ${NVRTC_INCLUDE_DIR}")
        add_library(CUDA::nvrtc SHARED IMPORTED)
        set_target_properties(CUDA::nvrtc PROPERTIES
            IMPORTED_LOCATION "${NVRTC_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NVRTC_INCLUDE_DIR}"
        )
    else()
        message(WARNING "nvrtc library or header not found; CUDA NVRTC functionality will be unavailable.")
        set(HAVE_CUDA OFF)
    endif()
endif()

# Optional: OpenCL support - WORKING VERSION
option(ENABLE_OPENCL "Enable OpenCL support" ON)
if(ENABLE_OPENCL)
    # Look for OpenCL headers in common locations
    find_path(OPENCL_INCLUDE_DIR CL/cl.h
        PATHS
        /usr/local/cuda-11.8/targets/x86_64-linux/include
        /usr/local/cuda/include
        /usr/include
        /usr/local/include
        /opt/AMDAPPSDK-3.0/include
    )

    # Find the OpenCL library that -lOpenCL would link to
    find_library(OPENCL_LIBRARY_PATH NAMES OpenCL
        PATHS
        /usr/lib/x86_64-linux-gnu
        /usr/lib
        /usr/local/lib
        /lib/x86_64-linux-gnu
        /opt/AMDAPPSDK-3.0/lib/x86_64
    )

    message(STATUS "OpenCL detection:")
    message(STATUS "  Headers: ${OPENCL_INCLUDE_DIR}")
    message(STATUS "  Library: ${OPENCL_LIBRARY_PATH}")

    if(OPENCL_INCLUDE_DIR AND OPENCL_LIBRARY_PATH)
        message(STATUS "OpenCL found - enabling OpenCL executor")
        set(HAVE_OPENCL ON)

        # Create a simple OpenCL target that mimics successful compilation
        add_library(SimpleOpenCL INTERFACE)
        target_include_directories(SimpleOpenCL INTERFACE ${OPENCL_INCLUDE_DIR})
        target_link_libraries(SimpleOpenCL INTERFACE ${OPENCL_LIBRARY_PATH})

    else()
        message(WARNING "OpenCL not found - disabling OpenCL executor")
        message(WARNING "  Looked for CL/cl.h in standard locations")
        message(WARNING "  Looked for libOpenCL in standard locations")
        set(HAVE_OPENCL OFF)
    endif()
else()
    message(STATUS "OpenCL support disabled by user")
    set(HAVE_OPENCL OFF)
endif()

# Optional: Vulkan support
option(ENABLE_VULKAN "Enable Vulkan support" ON)
if(ENABLE_VULKAN)
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        message(STATUS "Vulkan found - enabling Vulkan executor")
        set(HAVE_VULKAN ON)
    else()
        message(WARNING "Vulkan not found - disabling Vulkan executor")
        set(HAVE_VULKAN OFF)
    endif()
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add CUDA include directories if available
if(HAVE_CUDA)
    foreach(_p
        $ENV{CUDA_HOME}
        /usr/local/cuda-13/targets/x86_64-linux/include
        /usr/local/cuda-13/include
        /usr/local/cuda/include
    )
        if(_p AND EXISTS "${_p}")
            message(STATUS "Adding CUDA include dir: ${_p}")
            include_directories(BEFORE "${_p}")
            break()
        endif()
    endforeach()
endif()

# Common source files
set(COMMON_SOURCES
    common/framework_client.cpp
    common/websocket_client.cpp
    common/base64.cpp
)

# Base libraries that all targets need
set(BASE_LIBS
    Boost::system
    Boost::url
    OpenSSL::SSL
    OpenSSL::Crypto
    nlohmann_json::nlohmann_json
    Threads::Threads
)

# CUDA Client
if(HAVE_CUDA)
    add_executable(cuda_client
        main.cpp
        ${COMMON_SOURCES}
        cuda/cuda_executor.cpp
    )

    target_link_libraries(cuda_client PRIVATE
        ${BASE_LIBS}
        CUDA::cuda_driver
        CUDA::nvrtc
        CUDA::cudart
    )

    target_compile_definitions(cuda_client PRIVATE HAVE_CUDA=1)

    set_target_properties(cuda_client PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    message(STATUS "Created target: cuda_client")
endif()

# OpenCL Client - WORKING VERSION
if(HAVE_OPENCL)
    add_executable(opencl_client
        main.cpp
        ${COMMON_SOURCES}
        opencl/opencl_executor.cpp
    )

    target_link_libraries(opencl_client PRIVATE
        ${BASE_LIBS}
        SimpleOpenCL  # Use our working OpenCL target
    )

    target_compile_definitions(opencl_client PRIVATE HAVE_OPENCL=1)

    set_target_properties(opencl_client PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    message(STATUS "Created target: opencl_client")
endif()

# Vulkan Client
if(HAVE_VULKAN)
    add_executable(vulkan_client
        main.cpp
        ${COMMON_SOURCES}
        vulkan/vulkan_executor.cpp
    )

    target_link_libraries(vulkan_client PRIVATE
        ${BASE_LIBS}
        Vulkan::Vulkan
    )

    target_compile_definitions(vulkan_client PRIVATE HAVE_VULKAN=1)

    set_target_properties(vulkan_client PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    message(STATUS "Created target: vulkan_client")
endif()

# Universal client that can be built with all available frameworks
if(HAVE_CUDA OR HAVE_OPENCL OR HAVE_VULKAN)
    set(UNIVERSAL_SOURCES ${COMMON_SOURCES})
    set(UNIVERSAL_LIBS ${BASE_LIBS})
    set(UNIVERSAL_DEFS "")

    if(HAVE_CUDA)
        list(APPEND UNIVERSAL_SOURCES cuda/cuda_executor.cpp)
        list(APPEND UNIVERSAL_LIBS CUDA::cuda_driver CUDA::nvrtc CUDA::cudart)
        list(APPEND UNIVERSAL_DEFS HAVE_CUDA=1)
    endif()

    if(HAVE_OPENCL)
        list(APPEND UNIVERSAL_SOURCES opencl/opencl_executor.cpp)
        list(APPEND UNIVERSAL_LIBS SimpleOpenCL)
        list(APPEND UNIVERSAL_DEFS HAVE_OPENCL=1)
    endif()

    if(HAVE_VULKAN)
        list(APPEND UNIVERSAL_SOURCES vulkan/vulkan_executor.cpp)
        list(APPEND UNIVERSAL_LIBS Vulkan::Vulkan)
        list(APPEND UNIVERSAL_DEFS HAVE_VULKAN=1)
    endif()

    add_executable(universal_client
        main.cpp
        ${UNIVERSAL_SOURCES}
    )

    target_link_libraries(universal_client PRIVATE ${UNIVERSAL_LIBS})
    target_compile_definitions(universal_client PRIVATE ${UNIVERSAL_DEFS})

    set_target_properties(universal_client PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    message(STATUS "Created target: universal_client")
else()
    message(WARNING "No compute frameworks found - no executables will be built")
endif()

# Print build summary
message(STATUS "=== Build Summary ===")
message(STATUS "CUDA Support: ${HAVE_CUDA}")
message(STATUS "OpenCL Support: ${HAVE_OPENCL}")
message(STATUS "Vulkan Support: ${HAVE_VULKAN}")
if(HAVE_OPENCL)
    message(STATUS "OpenCL Include: ${OPENCL_INCLUDE_DIR}")
    message(STATUS "OpenCL Library: ${OPENCL_LIBRARY_PATH}")
endif()
message(STATUS "====================")

# Installation rules
set(INSTALL_TARGETS "")

if(HAVE_CUDA OR HAVE_OPENCL OR HAVE_VULKAN)
    list(APPEND INSTALL_TARGETS universal_client)
endif()

if(HAVE_CUDA)
    list(APPEND INSTALL_TARGETS cuda_client)
endif()

if(HAVE_OPENCL)
    list(APPEND INSTALL_TARGETS opencl_client)
endif()

if(HAVE_VULKAN)
    list(APPEND INSTALL_TARGETS vulkan_client)
endif()

if(INSTALL_TARGETS)
    install(TARGETS ${INSTALL_TARGETS} DESTINATION bin)
    message(STATUS "Will install: ${INSTALL_TARGETS}")
else()
    message(STATUS "No targets to install")
endif()