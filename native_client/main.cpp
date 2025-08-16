#include "common/framework_client.hpp"
#include "cuda/cuda_executor.hpp"
#include "opencl/opencl_executor.hpp"
// #include "vulkan/vulkan_executor.hpp"  // Would be implemented similarly

#include <iostream>
#include <memory>
#include <signal.h>

std::unique_ptr<FrameworkClient> globalClient;

void signalHandler(int signum) {
    std::cout << "\nReceived signal " << signum << ", shutting down..." << std::endl;
    if (globalClient) {
        globalClient->disconnect();
    }
    exit(signum);
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <framework> [options]\n"
              << "Frameworks: cuda, opencl, vulkan\n"
              << "Options:\n"
              << "  --url <ws://localhost:3000>  Server URL\n"
              << "  --device <0>                 Device ID\n"
              << "  --config <file.json>         Configuration file\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string framework = argv[1];
    std::string serverUrl = "wss://localhost:3000";
    int deviceId = 0;
    std::string configFile;
    
    // Parse command line arguments
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) break;
        
        std::string arg = argv[i];
        std::string value = argv[i + 1];
        
        if (arg == "--url") {
            serverUrl = value;
        } else if (arg == "--device") {
            deviceId = std::stoi(value);
        } else if (arg == "--config") {
            configFile = value;
        }
    }
    
    // Create framework executor
    std::unique_ptr<IFrameworkExecutor> executor;
    
    if (framework == "cuda") {
        executor = std::make_unique<CudaExecutor>(deviceId);
    } else if (framework == "opencl") {
        executor = std::make_unique<OpenCLExecutor>();
    } else if (framework == "vulkan") {
        // executor = std::make_unique<VulkanExecutor>();
        std::cerr << "Vulkan support not implemented yet" << std::endl;
        return 1;
    } else {
        std::cerr << "Unknown framework: " << framework << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    // Initialize executor
    json config;
    if (!configFile.empty()) {
        // Load config file
        std::ifstream configStream(configFile);
        if (configStream.is_open()) {
            configStream >> config;
        }
    }
    config["deviceId"] = deviceId;
    
    if (!executor->initialize(config)) {
        std::cerr << "Failed to initialize " << framework << " executor" << std::endl;
        return 1;
    }
    
    // Create and start client
    globalClient = std::make_unique<FrameworkClient>(std::move(executor));
    
    // Setup signal handling
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    std::cout << "Connecting to server: " << serverUrl << std::endl;
    
    if (!globalClient->connect(serverUrl)) {
        std::cerr << "Failed to connect to server" << std::endl;
        return 1;
    }
    
    std::cout << "Connected! Starting event loop..." << std::endl;
    globalClient->run();
    
    return 0;
}
