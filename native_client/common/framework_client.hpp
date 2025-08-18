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
    std::vector<uint8_t> inputData;
    size_t outputSize;
    json compilationOptions;
    json chunkUniforms;
    bool isChunk = false;
    std::string chunkId;
    int chunkOrderIndex = -1;
};

struct TaskResult {
    std::vector<uint8_t> outputData;
    double processingTime;
    bool success;
    std::string errorMessage;
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
project(MultiFrameworkClient LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# Find required packages
find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenSSL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

# CUDA
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

# OpenCL
find_package(OpenCL REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Create executables for each framework
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
*/
