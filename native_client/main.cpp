#include <iostream>
#include <fstream>
#include <memory>
#include <csignal>
#include <nlohmann/json.hpp>
#include "common/framework_client.hpp"

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

std::unique_ptr<FrameworkClient> globalClient;

void signalHandler(int signum) {
    std::cout << "\nReceived signal " << signum << ", shutting down..." << std::endl;
    if (globalClient) globalClient->disconnect();
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
    bool insecure = false;

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
    }

    // Set insecure SSL if requested
    if (insecure) {
        // This would typically set an environment variable or config
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

    // Create client
    globalClient = std::make_unique<FrameworkClient>(std::move(executor));
    std::signal(SIGINT,  signalHandler);
    std::signal(SIGTERM, signalHandler);

    std::cout << "Connecting to server: " << serverUrl << std::endl;
    if (!globalClient->connect(serverUrl)) {
        std::cerr << "Failed to connect to server" << std::endl;
        std::cerr << "Troubleshooting:" << std::endl;
        std::cerr << "  - Check server is running on " << serverUrl << std::endl;
        std::cerr << "  - If using self-signed certs, try --insecure flag" << std::endl;
        std::cerr << "  - Verify network connectivity and firewall settings" << std::endl;
        return 1;
    }

    std::cout << "✅ Connected! Starting event loop..." << std::endl;
    std::cout << "Framework: " << framework << std::endl;
    std::cout << "Client ready to receive " << framework << " workloads" << std::endl;
    std::cout << "Press Ctrl+C to shutdown gracefully" << std::endl;

    globalClient->run();
    return 0;
}