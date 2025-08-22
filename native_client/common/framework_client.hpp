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

    // Helper methods - FIXED: Cast to size_t to avoid type mismatch
    bool hasMultipleInputs() const { return inputData.size() > 1; }
    bool hasMultipleOutputs() const { return outputSizes.size() > 1; }
    size_t getInputCount() const { return std::max(inputData.size(), legacyInputData.empty() ? size_t(0) : size_t(1)); }
    size_t getOutputCount() const { return std::max(outputSizes.size(), legacyOutputSize > 0 ? size_t(1) : size_t(0)); }
};

struct TaskResult {
    // NEW: Multi-output support
    std::vector<std::vector<uint8_t>> outputData; // Multiple outputs

    // Legacy single output (for backward compatibility)
    std::vector<uint8_t> legacyOutputData;

    double processingTime;
    bool success;
    std::string errorMessage;

    // Helper methods - FIXED: Cast to size_t to avoid type mismatch
    bool hasMultipleOutputs() const { return outputData.size() > 1; }
    size_t getOutputCount() const { return std::max(outputData.size(), legacyOutputData.empty() ? size_t(0) : size_t(1)); }
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
    bool joinSent = false;
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
