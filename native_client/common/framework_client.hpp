// native-client/common/framework_client.hpp
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <map>

// Base64 utilities
#include "base64.hpp"
#include <nlohmann/json.hpp>
#include "websocket_client.hpp"

using json = nlohmann::json;

struct TaskData {
    std::string id;
    std::string parentId;
    std::string framework;
    std::string kernel;
    std::string entry;
    std::vector<int> workgroupCount;
    std::string bindLayout;

    // NEW: Multi-input support
    std::vector<std::vector<uint8_t>> inputData; // Multiple inputs
    std::vector<size_t> outputSizes; // Multiple output sizes

    // Legacy single input/output (for backward compatibility)
    std::vector<uint8_t> legacyInputData;
    size_t legacyOutputSize;

    json compilationOptions;
    json chunkUniforms;
    json metadata;  // ADD: Metadata for uniforms/parameters from server

    bool isChunk = false;
    std::string chunkId;
    int chunkOrderIndex = -1;

    // Helper methods
    bool hasMultipleInputs() const { return inputData.size() > 1; }
    bool hasMultipleOutputs() const { return outputSizes.size() > 1; }
    size_t getInputCount() const { return std::max(inputData.size(), legacyInputData.empty() ? 0 : 1); }
    size_t getOutputCount() const { return std::max(outputSizes.size(), legacyOutputSize > 0 ? 1 : 0); }
};

struct TaskResult {
    // NEW: Multi-output support
    std::vector<std::vector<uint8_t>> outputData; // Multiple outputs

    // Legacy single output (for backward compatibility)
    std::vector<uint8_t> legacyOutputData;

    double processingTime;
    bool success;
    std::string errorMessage;

    // Helper methods
    bool hasMultipleOutputs() const { return outputData.size() > 1; }
    size_t getOutputCount() const { return std::max(outputData.size(), legacyOutputData.empty() ? 0 : 1); }
};

class IFrameworkExecutor {
public:
    virtual ~IFrameworkExecutor() = default;
    virtual bool initialize(const json& config = {}) = 0;
    virtual void cleanup() = 0;
    virtual TaskResult executeTask(const TaskData& task) = 0;
    virtual std::string getFrameworkName() const = 0;
    virtual json getCapabilities() const = 0;
};

class FrameworkClient {
private:
    std::unique_ptr<WebSocketClient> wsClient;
    std::unique_ptr<IFrameworkExecutor> executor;
    std::string serverUrl;
    std::string clientId;
    bool connected = false;
    bool busy = false;

    // Task tracking
    std::map<std::string, TaskData> activeTasks;

    // Handlers
    void onConnected();
    void onDisconnected();
    void onMessage(const std::string& message);
    void handleWorkloadAssignment(const json& data);
    void handleChunkAssignment(const json& data);
    void sendResult(const TaskData& task, const TaskResult& result);
    void sendError(const TaskData& task, const std::string& error);

    // NEW: Helper methods for multi-input/output
    TaskData parseTaskData(const json& data, bool isChunk = false);
    std::vector<std::vector<uint8_t>> decodeInputs(const json& data);

public:
    FrameworkClient(std::unique_ptr<IFrameworkExecutor> exec);
    ~FrameworkClient();

    bool connect(const std::string& url);
    void disconnect();
    void run(); // Main event loop

    // Status
    bool isConnected() const { return connected; }
    bool isBusy() const { return busy; }
    std::string getClientId() const { return clientId; }
};


// ============================================================================
// CMakeLists.txt for building
// ============================================================================

/*
cmake_minimum_required(VERSION 3.18)
project(MultiFrameworkClient LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Find required packages
find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenSSL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

# Vulkan
find_package(Vulkan REQUIRED)

# CUDA (optional)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    add_compile_definitions(HAVE_CUDA)
endif()

# OpenCL (optional)
find_package(OpenCL QUIET)
if(OpenCL_FOUND)
    add_compile_definitions(HAVE_OPENCL)
endif()

# shaderc for Vulkan GLSL compilation
find_package(PkgConfig REQUIRED)
pkg_check_modules(SHADERC shaderc)
if(SHADERC_FOUND)
    add_compile_definitions(HAVE_SHADERC)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Create Vulkan client
add_executable(vulkan_client
    main.cpp
    common/framework_client.cpp
    common/websocket_client.cpp
    vulkan/vulkan_executor.cpp
)

target_link_libraries(vulkan_client
    ${Boost_LIBRARIES}
    OpenSSL::SSL
    OpenSSL::Crypto
    Vulkan::Vulkan
    ${SHADERC_LIBRARIES}
    pthread
)

target_include_directories(vulkan_client PRIVATE ${SHADERC_INCLUDE_DIRS})
target_compile_options(vulkan_client PRIVATE ${SHADERC_CFLAGS_OTHER})

# CUDA client (if available)
if(CUDAToolkit_FOUND)
    add_executable(cuda_client
        main.cpp
        common/framework_client.cpp
        common/websocket_client.cpp
        cuda/cuda_executor.cpp
    )

    target_link_libraries(cuda_client
        ${Boost_LIBRARIES}
        OpenSSL::SSL
        OpenSSL::Crypto
        CUDA::cuda_driver
        CUDA::nvrtc
        pthread
    )

    target_compile_definitions(cuda_client PRIVATE HAVE_CUDA CLIENT_CUDA)
endif()

# OpenCL client (if available)
if(OpenCL_FOUND)
    add_executable(opencl_client
        main.cpp
        common/framework_client.cpp
        common/websocket_client.cpp
        opencl/opencl_executor.cpp
    )

    target_link_libraries(opencl_client
        ${Boost_LIBRARIES}
        OpenSSL::SSL
        OpenSSL::Crypto
        OpenCL::OpenCL
        pthread
    )

    target_compile_definitions(opencl_client PRIVATE HAVE_OPENCL CLIENT_OPENCL)
endif()

# Universal client (if all frameworks available)
if(CUDAToolkit_FOUND AND OpenCL_FOUND AND SHADERC_FOUND)
    add_executable(universal_client
        main.cpp
        common/framework_client.cpp
        common/websocket_client.cpp
        vulkan/vulkan_executor.cpp
        cuda/cuda_executor.cpp
        opencl/opencl_executor.cpp
    )

    target_link_libraries(universal_client
        ${Boost_LIBRARIES}
        OpenSSL::SSL
        OpenSSL::Crypto
        Vulkan::Vulkan
        ${SHADERC_LIBRARIES}
        CUDA::cuda_driver
        CUDA::nvrtc
        OpenCL::OpenCL
        pthread
    )

    target_include_directories(universal_client PRIVATE ${SHADERC_INCLUDE_DIRS})
    target_compile_options(universal_client PRIVATE ${SHADERC_CFLAGS_OTHER})
    target_compile_definitions(universal_client PRIVATE
        HAVE_VULKAN HAVE_CUDA HAVE_OPENCL HAVE_SHADERC CLIENT_UNIVERSAL)
endif()
*/