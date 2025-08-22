#include <iostream>
#include <openssl/sha.h>
#include <openssl/evp.h>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <memory>
#include <csignal>
#include <chrono>
#include <thread>
#include <nlohmann/json.hpp>
#include "common/framework_client.hpp"
#include "common/base64.hpp"
#include "common/websocket_client.hpp"

// Framework-specific includes
#if defined(HAVE_VULKAN) && (defined(CLIENT_VULKAN) || defined(CLIENT_UNIVERSAL))
  #include "vulkan/vulkan_executor.hpp"
#endif
#if defined(HAVE_CUDA) && (defined(CLIENT_CUDA) || defined(CLIENT_UNIVERSAL))
  #include "cuda/cuda_executor.hpp"
#endif
#if defined(HAVE_OPENCL) && (defined(CLIENT_OPENCL) || defined(CLIENT_UNIVERSAL))
  #include "opencl/opencl_executor.hpp"
#endif

using nlohmann::json;

// Global state for signal handling
std::unique_ptr<FrameworkClient> globalFrameworkClient;
std::unique_ptr<WebSocketClient> globalWebSocketClient;
std::unique_ptr<IFrameworkExecutor> globalExecutor;
bool shutdownRequested = false;

void signalHandler(int signum) {
    std::cout << "\nReceived signal " << signum << ", shutting down..." << std::endl;
    shutdownRequested = true;

    if (globalWebSocketClient) {
        globalWebSocketClient->disconnect();
    }
    if (globalFrameworkClient) {
        globalFrameworkClient->disconnect();
    }

    std::exit(signum);
}

static void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <framework> [options]\n"
              << "Frameworks: ";

#if defined(HAVE_VULKAN)
    std::cout << "vulkan ";
#endif
#if defined(HAVE_CUDA)
    std::cout << "cuda ";
#endif
#if defined(HAVE_OPENCL)
    std::cout << "opencl ";
#endif

    std::cout << "\nOptions:\n"
              << "  --url <wss://localhost:3000>  Server URL\n"
              << "  --device <0>                  Device ID\n"
              << "  --config <file.json>          Configuration file\n"
              << "  --insecure                    Accept self-signed certificates\n"
              << "  --legacy                      Use legacy FrameworkClient instead of WebSocket\n"
              << std::endl;
}



// WebSocket-based client implementation
class WebSocketFrameworkClient {
private:
    std::unique_ptr<WebSocketClient> wsClient;
    std::unique_ptr<IFrameworkExecutor> executor;
    std::string clientId;
    json capabilities;
    bool isConnected = false;
    bool isProcessing = false;
    std::string framework;

public:
    WebSocketFrameworkClient(std::unique_ptr<IFrameworkExecutor> exec, const std::string& fw)
        : executor(std::move(exec)), framework(fw) {
        wsClient = std::make_unique<WebSocketClient>();
        //setupCapabilities();
        setupEventHandlers();
    }

    void setupCapabilities() {
         if (!executor) {
            std::cerr << "❌ Executor is null in setupCapabilities" << std::endl;
            // Create fallback capabilities
            // ?? capabilities = { ... };
            return;
        }
        try {
            capabilities = executor->getCapabilities();
            // Ensure supportedFrameworks array is present
            if (!capabilities.contains("supportedFrameworks")) {
                capabilities["supportedFrameworks"] = json::array({framework});
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not get executor capabilities: " << e.what() << std::endl;
            // Create minimal capabilities
            capabilities = {
                {"framework", framework},
                {"initialized", true},
                {"supportedFrameworks", {framework}},
                {"device", {{"name", "Unknown Device"}}}
            };
        }
    }
    bool initialize() {
    if (!executor) return false;  // Null check first
        setupCapabilities();          // Safe to call now
        return true;
    }
    void setupEventHandlers() {
        wsClient->setOnConnected([this]() {
            std::cout << "✅ Connected! Starting event loop..." << std::endl;
            isConnected = true;

            // Automatically join computation when connected
            wsClient->joinComputation(capabilities);
        });

        wsClient->setOnDisconnected([this]() {
            std::cout << "❌ Disconnected from server" << std::endl;
            isConnected = false;
            isProcessing = false;
        });

        wsClient->setOnRegister([this](const json& data) {
            clientId = data.value("clientId", "");
            std::cout << "📝 Registered with client ID: " << clientId << std::endl;
        });

        wsClient->setOnTaskAssigned([this](const json& task) {
            std::cout << "📋 Received matrix task: " << task.value("id", "") << std::endl;
            handleMatrixTask(task);
        });

        wsClient->setOnWorkloadAssigned([this](const json& workload) {
            std::cout << "🔧 Received workload: " << workload.value("id", "") << std::endl;
            handleWorkload(workload);
        });

        wsClient->setOnChunkAssigned([this](const json& chunk) {
            std::cout << "🧩 Received chunk: " << chunk.value("chunkId", "") << std::endl;
            handleChunk(chunk);
        });

        wsClient->setOnTaskVerified([this](const json& data) {
            std::cout << "✅ Task verified: " << data.value("taskId", "") << std::endl;
            isProcessing = false;
            requestNextTask();
        });

        wsClient->setOnTaskSubmitted([this](const json& data) {
            std::cout << "📤 Task submitted, waiting for verification: " << data.value("taskId", "") << std::endl;
            isProcessing = false;
            requestNextTask();
        });

        wsClient->setOnWorkloadComplete([this](const json& data) {
            std::cout << "🎉 Workload completed: " << data.value("label", "") << std::endl;
            isProcessing = false;
            requestNextTask();
        });
    }

    bool connect(const std::string& serverUrl) {
        // Parse URL to extract host and port
        std::string host = "localhost";
        std::string port = "3000";

        // Simple URL parsing for wss://host:port format
        if (serverUrl.find("wss://") == 0) {
            std::string hostPort = serverUrl.substr(6); // Remove "wss://"
            size_t colonPos = hostPort.find(':');
            if (colonPos != std::string::npos) {
                host = hostPort.substr(0, colonPos);
                port = hostPort.substr(colonPos + 1);
            } else {
                host = hostPort;
                port = "443"; // Default HTTPS port
            }
        }

        std::cout << "🔌 Connecting to server: " << serverUrl << std::endl;

        if (wsClient->connect(host, port, "/ws-native")) {
            std::cout << "✅ Connected to server" << std::endl;
            return true;
        } else {
            std::cerr << "❌ Failed to connect to server" << std::endl;
            return false;
        }
    }

    void run() {
        std::cout << "🚀 Client ready to receive " << framework << " workloads" << std::endl;
        std::cout << "Press Ctrl+C to shutdown gracefully" << std::endl;

        // Start requesting tasks periodically
        while (isConnected && !shutdownRequested) {
            if (!isProcessing) {
                requestNextTask();
            }
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    }

private:
    // Helper methods for converting between JSON and TaskData structures
    TaskData convertJsonToTaskData(const json& data, bool isChunk) {
        TaskData task;

        if (isChunk) {
            task.id = data.value("chunkId", "");
            task.parentId = data.value("parentId", "");
            task.chunkId = data.value("chunkId", "");
            task.chunkOrderIndex = data.value("chunkOrderIndex", -1);
            task.isChunk = true;
        } else {
            task.id = data.value("id", "");
        }

        task.framework = data.value("framework", framework);

        // Get shader/kernel code
        if (data.contains("kernel")) {
            task.kernel = data["kernel"];
        } else if (data.contains("wgsl")) {
            task.kernel = data["wgsl"];
        } else if (data.contains("vulkanShader")) {
            task.kernel = data["vulkanShader"];
        } else if (data.contains("openclKernel")) {
            task.kernel = data["openclKernel"];
        } else if (data.contains("cudaKernel")) {
            task.kernel = data["cudaKernel"];
        }

        task.entry = data.value("entry", "main");
        task.bindLayout = data.value("bindLayout", "");
        task.compilationOptions = data.value("compilationOptions", json::object());

        // Handle metadata and uniforms
        if (data.contains("metadata") && data["metadata"].is_object()) {
            task.metadata = data["metadata"];
            // Copy metadata into chunkUniforms for executor compatibility
            for (auto it = data["metadata"].begin(); it != data["metadata"].end(); ++it) {
                task.chunkUniforms[it.key()] = it.value();
            }
        }

        if (data.contains("chunkUniforms") && data["chunkUniforms"].is_object()) {
            for (auto it = data["chunkUniforms"].begin(); it != data["chunkUniforms"].end(); ++it) {
                task.chunkUniforms[it.key()] = it.value();
            }
        }

        // Work group sizes
        if (data.contains("globalWorkSize") && data["globalWorkSize"].is_array()) {
            task.workgroupCount = data["globalWorkSize"].get<std::vector<int>>();
        } else if (data.contains("workgroupCount") && data["workgroupCount"].is_array()) {
            task.workgroupCount = data["workgroupCount"].get<std::vector<int>>();
        } else {
            task.workgroupCount = {1, 1, 1};
        }

        // Decode inputs
        task.inputData = decodeInputs(data);

        // Parse output sizes
        if (data.contains("outputs") && data["outputs"].is_array()) {
            for (const auto& output : data["outputs"]) {
                if (output.contains("size")) {
                    task.outputSizes.push_back(output["size"]);
                }
            }
        } else if (data.contains("outputSizes") && data["outputSizes"].is_array()) {
            task.outputSizes = data["outputSizes"].get<std::vector<size_t>>();
        } else if (data.contains("outputSize")) {
            task.outputSizes = {data["outputSize"].get<size_t>()};
        } else {
            task.outputSizes = {1024}; // Default
        }

        // Set legacy fields for backward compatibility
        if (!task.inputData.empty()) {
            task.legacyInputData = task.inputData[0];
        }
        if (!task.outputSizes.empty()) {
            task.legacyOutputSize = task.outputSizes[0];
        }

        return task;
    }

    std::vector<std::vector<uint8_t>> decodeInputs(const json& data) {
        std::vector<std::vector<uint8_t>> inputs;

        // Preferred: inputs as array of { name, data }
        if (data.contains("inputs") && data["inputs"].is_array()) {
            for (const auto& item : data["inputs"]) {
                if (item.is_object() && item.contains("data") && item["data"].is_string()) {
                    auto b64 = item["data"].get<std::string>();
                    if (!b64.empty()) inputs.push_back(base64_decode(b64));
                } else if (item.is_string()) {
                    auto b64 = item.get<std::string>();
                    if (!b64.empty()) inputs.push_back(base64_decode(b64));
                }
            }
        }
        // Legacy single input fallbacks
        else if (data.contains("input") && data["input"].is_string()) {
            auto b64 = data["input"].get<std::string>();
            if (!b64.empty()) inputs.push_back(base64_decode(b64));
        } else if (data.contains("inputData") && data["inputData"].is_string()) {
            auto b64 = data["inputData"].get<std::string>();
            if (!b64.empty()) inputs.push_back(base64_decode(b64));
        }

        return inputs;
    }

    json convertTaskResultToJson(const TaskResult& result) {
        if (result.hasMultipleOutputs()) {
            json results = json::array();
            for (const auto& output : result.outputData) {
                results.push_back(base64_encode(output));
            }
            return results;
        } else {
            if (!result.outputData.empty()) {
                return base64_encode(result.outputData[0]);
            } else if (!result.legacyOutputData.empty()) {
                return base64_encode(result.legacyOutputData);
            }
        }
        return json::array();
    }

    std::string sha256Hex(const std::vector<uint8_t>& data) {
        EVP_MD_CTX* context = EVP_MD_CTX_new();
        const EVP_MD* md = EVP_sha256();
        unsigned char hash[EVP_MAX_MD_SIZE];
        unsigned int lengthOfHash = 0;

        EVP_DigestInit_ex(context, md, nullptr);
        EVP_DigestUpdate(context, data.data(), data.size());
        EVP_DigestFinal_ex(context, hash, &lengthOfHash);
        EVP_MD_CTX_free(context);

        std::stringstream ss;
        for (unsigned int i = 0; i < lengthOfHash; ++i) {
            ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
        }
        return ss.str();
    }


    // Replace the existing generateResultChecksum function
    std::string generateResultChecksum(const TaskResult& result) {
        if (result.hasMultipleOutputs()) {
            // Combine all outputs and hash
            std::vector<uint8_t> combined;
            for (const auto& output : result.outputData) {
                combined.insert(combined.end(), output.begin(), output.end());
            }
            return sha256Hex(combined);
        } else {
            if (!result.outputData.empty()) {
                return sha256Hex(result.outputData[0]);
            } else if (!result.legacyOutputData.empty()) {
                return sha256Hex(result.legacyOutputData);
            }
        }
        return "";
    }

    void requestNextTask() {
        if (isConnected && !isProcessing) {
            wsClient->requestTask();
        }
    }

    void handleMatrixTask(const json& task) {
        isProcessing = true;

        try {
            std::string assignmentId = task.value("assignmentId", "");
            std::string taskId = task.value("id", "");

            // Convert JSON task to TaskData struct
            TaskData taskData = convertJsonToTaskData(task, false);
            taskData.id = taskId;

            auto start = std::chrono::high_resolution_clock::now();
            TaskResult result = executor->executeTask(taskData);
            auto end = std::chrono::high_resolution_clock::now();

            if (result.success) {
                // Convert result back to expected format for matrix task
                json matrixResult = convertTaskResultToJson(result);

                std::string checksum = generateResultChecksum(result);
                wsClient->submitTaskResult(assignmentId, taskId, matrixResult, result.processingTime, checksum);
            } else {
                wsClient->reportError(taskId, result.errorMessage);
            }

        } catch (const std::exception& e) {
            std::cerr << "❌ Error processing matrix task: " << e.what() << std::endl;
            wsClient->reportError(task.value("id", ""), e.what());
        }

        isProcessing = false;
    }

    void handleWorkload(const json& workload) {
        isProcessing = true;

        try {
            std::string workloadId = workload.value("id", "");

            // Convert JSON workload to TaskData struct
            TaskData taskData = convertJsonToTaskData(workload, false);
            taskData.id = workloadId;

            TaskResult result = executor->executeTask(taskData);

            if (result.success) {
                // For single output workloads, send the first output
                std::string resultData = "";
                if (!result.outputData.empty()) {
                    resultData = base64_encode(result.outputData[0]);
                } else if (!result.legacyOutputData.empty()) {
                    resultData = base64_encode(result.legacyOutputData);
                }

                std::string checksum = generateResultChecksum(result);
                wsClient->submitWorkloadResult(workloadId, resultData, result.processingTime, checksum);
            } else {
                wsClient->reportError(workloadId, result.errorMessage);
            }

        } catch (const std::exception& e) {
            std::cerr << "❌ Error processing workload: " << e.what() << std::endl;
            wsClient->reportError(workload.value("id", ""), e.what());
        }

        isProcessing = false;
    }

    void handleChunk(const json& chunk) {
        isProcessing = true;

        try {
            std::string parentId = chunk.value("parentId", "");
            std::string chunkId = chunk.value("chunkId", "");
            std::string strategy = chunk.value("chunkingStrategy", "");
            json metadata = chunk.value("metadata", json::object());

            // Convert JSON chunk to TaskData struct
            TaskData taskData = convertJsonToTaskData(chunk, true);
            taskData.parentId = parentId;
            taskData.chunkId = chunkId;
            taskData.chunkOrderIndex = chunk.value("chunkOrderIndex", -1);

            TaskResult result = executor->executeTask(taskData);

            if (result.success) {
                // Convert multi-output results to JSON array
                json results = json::array();
                if (result.hasMultipleOutputs()) {
                    for (const auto& output : result.outputData) {
                        results.push_back(base64_encode(output));
                    }
                } else {
                    // Single output - add to array for consistency
                    if (!result.outputData.empty()) {
                        results.push_back(base64_encode(result.outputData[0]));
                    } else if (!result.legacyOutputData.empty()) {
                        results.push_back(base64_encode(result.legacyOutputData));
                    }
                }

                std::string checksum = generateResultChecksum(result);

                wsClient->submitChunkResult(parentId, chunkId, results, result.processingTime,
                                           strategy, metadata, checksum);
            } else {
                wsClient->reportChunkError(parentId, chunkId, result.errorMessage);
            }

        } catch (const std::exception& e) {
            std::cerr << "❌ Error processing chunk: " << e.what() << std::endl;
            wsClient->reportChunkError(chunk.value("parentId", ""),
                                      chunk.value("chunkId", ""), e.what());
        }

        isProcessing = false;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string framework = argv[1];
    std::string serverUrl = "wss://localhost:3000";
    int deviceId = 0;
    std::string configFile;
    bool insecure = false;
    bool useLegacy = false;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--url" && i + 1 < argc) {
            serverUrl = argv[++i];
        }
        else if (arg == "--device" && i + 1 < argc) {
            deviceId = std::stoi(argv[++i]);
        }
        else if (arg == "--config" && i + 1 < argc) {
            configFile = argv[++i];
        }
        else if (arg == "--insecure") {
            insecure = true;
        }
        else if (arg == "--legacy") {
            useLegacy = true;
        }
    }

    // Set insecure SSL if requested
    if (insecure) {
        std::cout << "Warning: Accepting self-signed certificates" << std::endl;
    }

    // Create framework executor
    std::unique_ptr<IFrameworkExecutor> executor;

    if (framework == "vulkan") {
    #if defined(HAVE_VULKAN)
        std::cout << "Creating Vulkan executor..." << std::endl;
        executor = std::make_unique<VulkanExecutor>(deviceId);
    #else
        std::cerr << "This binary was built without Vulkan support.\n";
        std::cerr << "Required: Vulkan SDK, shaderc library\n";
        return 1;
    #endif
    }
    else if (framework == "cuda") {
    #if defined(HAVE_CUDA)
        std::cout << "Creating CUDA executor..." << std::endl;
        executor = std::make_unique<CudaExecutor>(deviceId);
    #else
        std::cerr << "This binary was built without CUDA support.\n";
        std::cerr << "Required: CUDA Toolkit, nvrtc library\n";
        return 1;
    #endif
    }
    else if (framework == "opencl") {
    #if defined(HAVE_OPENCL)
        std::cout << "Creating OpenCL executor..." << std::endl;
        executor = std::make_unique<OpenCLExecutor>();
    #else
        std::cerr << "This binary was built without OpenCL support.\n";
        std::cerr << "Required: OpenCL SDK\n";
        return 1;
    #endif
    }
    else {
        std::cerr << "Unknown framework: " << framework << "\n";
        printUsage(argv[0]);
        return 1;
    }

    if (!executor) {
        std::cerr << "No executor created (internal error).\n";
        return 1;
    }

    // Load config
    json config;
    if (!configFile.empty()) {
        std::ifstream configStream(configFile);
        if (configStream.is_open()) {
            try {
                configStream >> config;
                std::cout << "Loaded configuration from " << configFile << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Failed to parse config file: " << e.what() << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Could not open config file: " << configFile << std::endl;
            return 1;
        }
    }
    config["deviceId"] = deviceId;

    std::cout << "Initializing " << framework << " executor..." << std::endl;
    if (!executor->initialize(config)) {
        std::cerr << "Failed to initialize " << framework << " executor" << std::endl;

        // Print capabilities for debugging
        try {
            json caps = executor->getCapabilities();
            std::cout << "Executor capabilities: " << caps.dump(2) << std::endl;
        } catch (...) {
            std::cout << "Could not retrieve executor capabilities" << std::endl;
        }

        return 1;
    }

    // Print successful initialization info
    try {
        json caps = executor->getCapabilities();
        std::cout << "✅ " << framework << " executor initialized successfully" << std::endl;
        std::cout << "Capabilities: " << caps.dump(2) << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✅ " << framework << " executor initialized (capabilities unavailable: "
                  << e.what() << ")" << std::endl;
    }

    // Set up signal handlers
    std::signal(SIGINT,  signalHandler);
    std::signal(SIGTERM, signalHandler);

    // Choose client implementation
    if (useLegacy) {
        std::cout << "Using legacy FrameworkClient..." << std::endl;
        globalFrameworkClient = std::make_unique<FrameworkClient>(std::move(executor));
        globalExecutor = nullptr; // Moved to FrameworkClient

        std::cout << "Connecting to server: " << serverUrl << std::endl;
        if (!globalFrameworkClient->connect(serverUrl)) {
            std::cerr << "Failed to connect to server" << std::endl;
            return 1;
        }

        std::cout << "✅ Connected! Starting event loop..." << std::endl;
        std::cout << "Framework: " << framework << std::endl;
        std::cout << "Client ready to receive " << framework << " workloads" << std::endl;
        std::cout << "Press Ctrl+C to shutdown gracefully" << std::endl;

        globalFrameworkClient->run();
    } else {
        std::cout << "Using WebSocket client..." << std::endl;
        //globalExecutor = std::move(executor);

        auto wsFrameworkClient = std::make_unique<WebSocketFrameworkClient>(std::move(executor), framework);
        if (!wsFrameworkClient->initialize()) {
            std::cerr << "❌ Failed to initialize WebSocketFrameworkClient" << std::endl;
            return 1;
        }
        if (!wsFrameworkClient->connect(serverUrl)) {
            std::cerr << "Failed to connect to server" << std::endl;
            std::cerr << "Troubleshooting:" << std::endl;
            std::cerr << "  - Check server is running on " << serverUrl << std::endl;
            std::cerr << "  - If using self-signed certs, try --insecure flag" << std::endl;
            std::cerr << "  - Verify network connectivity and firewall settings" << std::endl;
            std::cerr << "  - Try --legacy flag to use old client implementation" << std::endl;
            return 1;
        }

        wsFrameworkClient->run();
    }

    std::cout << "👋 Client shutting down" << std::endl;
    return 0;
}