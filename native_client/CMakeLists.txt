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
find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenSSL REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

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

# Optional: OpenCL support
option(ENABLE_OPENCL "Enable OpenCL support" ON)
if(ENABLE_OPENCL)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        message(STATUS "OpenCL found - enabling OpenCL executor")
        set(HAVE_OPENCL ON)
    else()
        message(WARNING "OpenCL not found - disabling OpenCL executor")
        set(HAVE_OPENCL OFF)
    endif()
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

# Common source files
set(COMMON_SOURCES
    common/framework_client.cpp
    common/websocket_client.cpp
    common/base64.cpp
)

# Function to create executables for each framework
function(create_framework_executable framework_name sources link_libs compile_defs)
    add_executable(${framework_name}_client
        main.cpp
        ${COMMON_SOURCES}
        ${sources}
    )

    target_link_libraries(${framework_name}_client
        PRIVATE
        Boost::system
        OpenSSL::SSL
        OpenSSL::Crypto
        nlohmann_json::nlohmann_json
        ${link_libs}
        ${CMAKE_THREAD_LIBS_INIT}
    )

    target_compile_definitions(${framework_name}_client PRIVATE ${compile_defs})

    # Set output directory
    set_target_properties(${framework_name}_client PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    message(STATUS "Created target: ${framework_name}_client")
endfunction()

# Find threads (required for std::thread)
find_package(Threads REQUIRED)

# CUDA Client
if(HAVE_CUDA)
    create_framework_executable("cuda"
        "cuda/cuda_executor.cpp"
        "CUDA::cuda_driver;CUDA::nvrtc"
        "HAVE_CUDA=1"
    )
endif()

# OpenCL Client
if(HAVE_OPENCL)
    create_framework_executable("opencl"
        "opencl/opencl_executor.cpp"
        "OpenCL::OpenCL"
        "HAVE_OPENCL=1"
    )
endif()

# Vulkan Client
if(HAVE_VULKAN)
    create_framework_executable("vulkan"
        "vulkan/vulkan_executor.cpp"
        "Vulkan::Vulkan"
        "HAVE_VULKAN=1"
    )
endif()

# Universal client that can be built with all available frameworks
set(UNIVERSAL_SOURCES ${COMMON_SOURCES})
set(UNIVERSAL_LIBS Boost::system OpenSSL::SSL OpenSSL::Crypto nlohmann_json::nlohmann_json ${CMAKE_THREAD_LIBS_INIT})
set(UNIVERSAL_DEFS "")

if(HAVE_CUDA)
    list(APPEND UNIVERSAL_SOURCES cuda/cuda_executor.cpp)
    list(APPEND UNIVERSAL_LIBS CUDA::cuda_driver CUDA::nvrtc)
    list(APPEND UNIVERSAL_DEFS HAVE_CUDA=1)
endif()

if(HAVE_OPENCL)
    list(APPEND UNIVERSAL_SOURCES opencl/opencl_executor.cpp)
    list(APPEND UNIVERSAL_LIBS OpenCL::OpenCL)
    list(APPEND UNIVERSAL_DEFS HAVE_OPENCL=1)
endif()

if(HAVE_VULKAN)
    list(APPEND UNIVERSAL_SOURCES vulkan/vulkan_executor.cpp)
    list(APPEND UNIVERSAL_LIBS Vulkan::Vulkan)
    list(APPEND UNIVERSAL_DEFS HAVE_VULKAN=1)
endif()

# Only create universal client if at least one framework is available
if(HAVE_CUDA OR HAVE_OPENCL OR HAVE_VULKAN)
    add_executable(universal_client
        main.cpp
        ${UNIVERSAL_SOURCES}
    )

    target_link_libraries(universal_client
        PRIVATE
        ${UNIVERSAL_LIBS}
    )

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
message(STATUS "====================")

# Installation rules
if(HAVE_CUDA OR HAVE_OPENCL OR HAVE_VULKAN)
    install(TARGETS
        ${HAVE_CUDA AND "cuda_client" OR ""}
        ${HAVE_OPENCL AND "opencl_client" OR ""}
        ${HAVE_VULKAN AND "vulkan_client" OR ""}
        universal_client
        DESTINATION bin
    )
endif()