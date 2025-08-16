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

# Create base64.cpp if it doesn't exist (extracted from the websocket_client.cpp)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/common/base64.cpp")
    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/common/base64.cpp"
"#include \"base64.hpp\"
#include <algorithm>

static const char base64_chars[] = 
    \"ABCDEFGHIJKLMNOPQRSTUVWXYZ\"
    \"abcdefghijklmnopqrstuvwxyz\"
    \"0123456789+/\";

static inline bool is_base64(unsigned char c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

std::string base64_encode(const std::vector<uint8_t>& data) {
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];
    int len = data.size();
    const unsigned char* bytes_to_encode = data.data();

    while (len--) {
        char_array_3[i++] = *(bytes_to_encode++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for(i = 0; (i <4) ; i++)
                ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i) {
        for(j = i; j < 3; j++)
            char_array_3[j] = '\\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (j = 0; (j < i + 1); j++)
            ret += base64_chars[char_array_4[j]];

        while((i++ < 3))
            ret += '=';
    }

    return ret;
}

std::vector<uint8_t> base64_decode(const std::string& encoded_string) {
    int len = encoded_string.size();
    int i = 0;
    int j = 0;
    int in = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::vector<uint8_t> ret;

    while (len-- && ( encoded_string[in] != '=') && is_base64(encoded_string[in])) {
        char_array_4[i++] = encoded_string[in]; in++;
        if (i ==4) {
            for (i = 0; i <4; i++)
                char_array_4[i] = std::find(base64_chars, base64_chars + 64, char_array_4[i]) - base64_chars;

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; (i < 3); i++)
                ret.push_back(char_array_3[i]);
            i = 0;
        }
    }

    if (i) {
        for (j = i; j <4; j++)
            char_array_4[j] = 0;

        for (j = 0; j <4; j++)
            char_array_4[j] = std::find(base64_chars, base64_chars + 64, char_array_4[j]) - base64_chars;

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (j = 0; (j < i - 1); j++) ret.push_back(char_array_3[j]);
    }

    return ret;
}")
endif()

# Create base64.hpp if it doesn't exist
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/common/base64.hpp")
    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/common/base64.hpp"
"#pragma once
#include <string>
#include <vector>
#include <cstdint>

std::string base64_encode(const std::vector<uint8_t>& data);
std::vector<uint8_t> base64_decode(const std::string& encoded);")
endif()

# Create json.hpp if it doesn't exist (nlohmann json include)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/common/json.hpp")
    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/common/json.hpp"
"#pragma once
#include <nlohmann/json.hpp>")
endif()

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
