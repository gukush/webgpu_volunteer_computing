#if defined(HAVE_CUDA) && (defined(CLIENT_CUDA) || defined(CLIENT_UNIVERSAL))
  #include "cuda/cuda_executor.hpp"
#endif
#if defined(HAVE_OPENCL) && (defined(CLIENT_OPENCL) || defined(CLIENT_UNIVERSAL))
  #include "opencl/opencl_executor.hpp"
#endif
#if defined(HAVE_VULKAN) && (defined(CLIENT_VULKAN) || defined(CLIENT_UNIVERSAL))
  #include "vulkan/vulkan_executor.hpp"
#endif

#include <iostream>
#include <fstream>
#include <memory>
#include <csignal>
#include <nlohmann/json.hpp>
using nlohmann::json;

std::unique_ptr<FrameworkClient> globalClient;

void signalHandler(int signum) {
    std::cout << "\nReceived signal " << signum << ", shutting down..." << std::endl;
    if (globalClient) globalClient->disconnect();
    std::exit(signum);
}

static void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <framework> [options]\n"
              << "Frameworks: cuda, opencl, vulkan\n"
              << "Options:\n"
              << "  --url <ws://localhost:3000>  Server URL\n"
              << "  --device <0>                 Device ID\n"
              << "  --config <file.json>         Configuration file\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) { printUsage(argv[0]); return 1; }

    std::string framework = argv[1];
    std::string serverUrl = "wss://localhost:3000";
    int deviceId = 0;
    std::string configFile;

    for (int i = 2; i + 1 < argc; i += 2) {
        std::string arg = argv[i], value = argv[i + 1];
        if (arg == "--url") serverUrl = value;
        else if (arg == "--device") deviceId = std::stoi(value);
        else if (arg == "--config") configFile = value;
    }

    // Create framework executor
    std::unique_ptr<IFrameworkExecutor> executor;

    if (framework == "cuda") {
    #if defined(HAVE_CUDA)
        executor = std::make_unique<CudaExecutor>(deviceId);
    #else
        std::cerr << "This binary was built without CUDA support.\n";
        return 1;
    #endif
    }
    else if (framework == "opencl") {
    #if defined(HAVE_OPENCL)
        executor = std::make_unique<OpenCLExecutor>();
    #else
        std::cerr << "This binary was built without OpenCL support.\n";
        return 1;
    #endif
    }
    else if (framework == "vulkan") {
    #if defined(HAVE_VULKAN)
        executor = std::make_unique<VulkanExecutor>();
    #else
        std::cerr << "This binary was built without Vulkan support.\n";
        return 1;
    #endif
    }
    else {
        std::cerr << "Unknown framework: " << framework << "\n";
        printUsage(argv[0]);
        return 1;
    }

    if (!executor) { std::cerr << "No executor created (internal error).\n"; return 1; }

    // Load config
    json config;
    if (!configFile.empty()) {
        std::ifstream configStream(configFile);
        if (configStream.is_open()) configStream >> config;
    }
    config["deviceId"] = deviceId;

    if (!executor->initialize(config)) {
        std::cerr << "Failed to initialize " << framework << " executor\n";
        return 1;
    }

    // Client
    globalClient = std::make_unique<FrameworkClient>(std::move(executor));
    std::signal(SIGINT,  signalHandler);
    std::signal(SIGTERM, signalHandler);

    std::cout << "Connecting to server: " << serverUrl << std::endl;
    if (!globalClient->connect(serverUrl)) { std::cerr << "Failed to connect to server\n"; return 1; }

    std::cout << "Connected! Starting event loop..." << std::endl;
    globalClient->run();
    return 0;
}