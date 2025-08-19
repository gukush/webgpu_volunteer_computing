#include "cuda_executor.hpp"
#include <cuda.h>
#include <iostream>
#include <chrono>

CudaExecutor::CudaExecutor(int devId) : deviceId(devId) {}

CudaExecutor::~CudaExecutor() {
    cleanup();
}

bool CudaExecutor::initialize(const json& config) {
    if (initialized) return true;

    // Initialize CUDA driver API
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to initialize CUDA driver API" << std::endl;
        return false;
    }

    // Get device count
    int deviceCount;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return false;
    }

    if (deviceId >= deviceCount) {
        std::cerr << "Invalid device ID: " << deviceId << std::endl;
        return false;
    }

    // Set device and get properties
    if (cudaSetDevice(deviceId) != cudaSuccess) {
        std::cerr << "Failed to set CUDA device " << deviceId << std::endl;
        return false;
    }

    if (cudaGetDeviceProperties(&deviceProps, deviceId) != cudaSuccess) {
        std::cerr << "Failed to get device properties" << std::endl;
        return false;
    }

    // Create CUDA context
    CUdevice device;
    if (cuDeviceGet(&device, deviceId) != CUDA_SUCCESS) {
        std::cerr << "Failed to get CUDA device" << std::endl;
        return false;
    }

    if (cuCtxCreate(&context, 0, device) != CUDA_SUCCESS) {
        std::cerr << "Failed to create CUDA context" << std::endl;
        return false;
    }

    initialized = true;
    std::cout << "CUDA initialized on device " << deviceId
              << " (" << deviceProps.name << ")" << std::endl;
    return true;
}

void CudaExecutor::cleanup() {
    if (!initialized) return;

    // Cleanup cached kernels
    for (auto& [key, kernel] : kernelCache) {
        if (kernel.module) {
            cuModuleUnload(kernel.module);
        }
    }
    kernelCache.clear();

    // Cleanup context
    if (context) {
        cuCtxDestroy(context);
        context = nullptr;
    }

    initialized = false;
}

bool CudaExecutor::compileKernel(const std::string& source, const std::string& entryPoint,
                                const json& compileOpts, CompiledKernel& result) {
    // Create NVRTC program
    nvrtcProgram prog;
    nvrtcResult res = nvrtcCreateProgram(&prog, source.c_str(), nullptr, 0, nullptr, nullptr);
    if (res != NVRTC_SUCCESS) {
        std::cerr << "Failed to create NVRTC program: " << nvrtcGetErrorString(res) << std::endl;
        return false;
    }

    // Prepare compilation options
    std::vector<const char*> opts;
    std::string computeCapability = "--gpu-architecture=compute_" +
        std::to_string(deviceProps.major) + std::to_string(deviceProps.minor);
    opts.push_back(computeCapability.c_str());

    // Add user-specified options
    std::vector<std::string> optStrings;
    if (compileOpts.contains("optimization")) {
        optStrings.push_back(compileOpts["optimization"].get<std::string>());
        opts.push_back(optStrings.back().c_str());
    }

    // Compile
    res = nvrtcCompileProgram(prog, opts.size(), opts.data());
    if (res != NVRTC_SUCCESS) {
        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        std::vector<char> log(logSize);
        nvrtcGetProgramLog(prog, log.data());
        std::cerr << "CUDA compilation failed:\n" << log.data() << std::endl;
        nvrtcDestroyProgram(&prog);
        return false;
    }

    // Get PTX
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    result.ptx.resize(ptxSize);
    nvrtcGetPTX(prog, result.ptx.data());
    nvrtcDestroyProgram(&prog);

    // Load module
    CUresult cuRes = cuModuleLoadDataEx(&result.module, result.ptx.c_str(), 0, 0, 0);
    if (cuRes != CUDA_SUCCESS) {
        std::cerr << "Failed to load CUDA module" << std::endl;
        return false;
    }

    // Get function
    cuRes = cuModuleGetFunction(&result.function, result.module, entryPoint.c_str());
    if (cuRes != CUDA_SUCCESS) {
        std::cerr << "Failed to get CUDA function: " << entryPoint << std::endl;
        cuModuleUnload(result.module);
        return false;
    }

    return true;
}

TaskResult CudaExecutor::executeTask(const TaskData& task) {
    TaskResult result;
    auto startTime = std::chrono::high_resolution_clock::now();

    if (!initialized) {
        result.success = false;
        result.errorMessage = "CUDA not initialized";
        return result;
    }

    try {
        // Set context
        cuCtxSetCurrent(context);

        // Compile kernel if not cached
        std::string cacheKey = task.kernel + "|" + task.entry;
        auto it = kernelCache.find(cacheKey);
        CompiledKernel* kernel;

        if (it == kernelCache.end()) {
            CompiledKernel newKernel;
            if (!compileKernel(task.kernel, task.entry, task.compilationOptions, newKernel)) {
                result.success = false;
                result.errorMessage = "Kernel compilation failed";
                return result;
            }
            kernelCache[cacheKey] = std::move(newKernel);
            kernel = &kernelCache[cacheKey];
        } else {
            kernel = &it->second;
        }

        // Determine if we're using multi-input/output or legacy single input/output
        bool useMultipleInputs = task.hasMultipleInputs();
        bool useMultipleOutputs = task.hasMultipleOutputs();

        std::vector<CUdeviceptr> d_inputs;
        std::vector<CUdeviceptr> d_outputs;
        std::vector<void*> kernelArgs;

        std::cout << "CUDA Task: " << (useMultipleInputs ? task.inputData.size() : 1) << " inputs, "
                  << (useMultipleOutputs ? task.outputSizes.size() : 1) << " outputs" << std::endl;

        // Allocate and copy input data
        if (useMultipleInputs) {
            // Multi-input mode
            for (size_t i = 0; i < task.inputData.size(); i++) {
                const auto& inputData = task.inputData[i];
                CUdeviceptr d_input = 0;

                if (!inputData.empty()) {
                    if (cuMemAlloc(&d_input, inputData.size()) != CUDA_SUCCESS) {
                        // Cleanup previously allocated inputs
                        for (auto ptr : d_inputs) {
                            if (ptr) cuMemFree(ptr);
                        }
                        result.success = false;
                        result.errorMessage = "Failed to allocate input memory for input " + std::to_string(i);
                        return result;
                    }

                    if (cuMemcpyHtoD(d_input, inputData.data(), inputData.size()) != CUDA_SUCCESS) {
                        cuMemFree(d_input);
                        for (auto ptr : d_inputs) {
                            if (ptr) cuMemFree(ptr);
                        }
                        result.success = false;
                        result.errorMessage = "Failed to copy input data for input " + std::to_string(i);
                        return result;
                    }
                }

                d_inputs.push_back(d_input);
                kernelArgs.push_back(&d_inputs.back());
            }
        } else {
            // Legacy single input mode
            CUdeviceptr d_input = 0;

            if (!task.legacyInputData.empty()) {
                if (cuMemAlloc(&d_input, task.legacyInputData.size()) != CUDA_SUCCESS) {
                    result.success = false;
                    result.errorMessage = "Failed to allocate input memory";
                    return result;
                }

                if (cuMemcpyHtoD(d_input, task.legacyInputData.data(), task.legacyInputData.size()) != CUDA_SUCCESS) {
                    cuMemFree(d_input);
                    result.success = false;
                    result.errorMessage = "Failed to copy input data";
                    return result;
                }
            }

            d_inputs.push_back(d_input);
            kernelArgs.push_back(&d_inputs.back());
        }

        // Allocate output buffers
        if (useMultipleOutputs) {
            // Multi-output mode
            for (size_t i = 0; i < task.outputSizes.size(); i++) {
                CUdeviceptr d_output = 0;

                if (cuMemAlloc(&d_output, task.outputSizes[i]) != CUDA_SUCCESS) {
                    // Cleanup previously allocated memory
                    for (auto ptr : d_inputs) {
                        if (ptr) cuMemFree(ptr);
                    }
                    for (auto ptr : d_outputs) {
                        if (ptr) cuMemFree(ptr);
                    }
                    result.success = false;
                    result.errorMessage = "Failed to allocate output memory for output " + std::to_string(i);
                    return result;
                }

                d_outputs.push_back(d_output);
                kernelArgs.push_back(&d_outputs.back());
            }
        } else {
            // Legacy single output mode
            CUdeviceptr d_output = 0;

            if (cuMemAlloc(&d_output, task.legacyOutputSize) != CUDA_SUCCESS) {
                for (auto ptr : d_inputs) {
                    if (ptr) cuMemFree(ptr);
                }
                result.success = false;
                result.errorMessage = "Failed to allocate output memory";
                return result;
            }

            d_outputs.push_back(d_output);
            kernelArgs.push_back(&d_outputs.back());
        }

        // Launch kernel with all input/output arguments
        CUresult launchResult = cuLaunchKernel(
            kernel->function,
            task.workgroupCount[0], task.workgroupCount[1], task.workgroupCount[2],  // grid
            256, 1, 1,  // block (could be made configurable)
            0, 0,  // shared mem, stream
            kernelArgs.data(), nullptr
        );

        if (launchResult != CUDA_SUCCESS) {
            // Cleanup all allocated memory
            for (auto ptr : d_inputs) {
                if (ptr) cuMemFree(ptr);
            }
            for (auto ptr : d_outputs) {
                if (ptr) cuMemFree(ptr);
            }
            result.success = false;
            result.errorMessage = "Kernel launch failed";
            return result;
        }

        // Wait for completion
        if (cuCtxSynchronize() != CUDA_SUCCESS) {
            for (auto ptr : d_inputs) {
                if (ptr) cuMemFree(ptr);
            }
            for (auto ptr : d_outputs) {
                if (ptr) cuMemFree(ptr);
            }
            result.success = false;
            result.errorMessage = "Kernel execution failed";
            return result;
        }

        // Copy results back
        if (useMultipleOutputs) {
            // Multi-output mode
            result.outputData.resize(task.outputSizes.size());
            for (size_t i = 0; i < task.outputSizes.size(); i++) {
                result.outputData[i].resize(task.outputSizes[i]);
                if (cuMemcpyDtoH(result.outputData[i].data(), d_outputs[i], task.outputSizes[i]) != CUDA_SUCCESS) {
                    // Cleanup memory
                    for (auto ptr : d_inputs) {
                        if (ptr) cuMemFree(ptr);
                    }
                    for (auto ptr : d_outputs) {
                        if (ptr) cuMemFree(ptr);
                    }
                    result.success = false;
                    result.errorMessage = "Failed to copy output data for output " + std::to_string(i);
                    return result;
                }
            }

            // Set legacy output data to first output for backward compatibility
            if (!result.outputData.empty() && !result.outputData[0].empty()) {
                result.legacyOutputData = result.outputData[0];
            }
        } else {
            // Legacy single output mode
            result.legacyOutputData.resize(task.legacyOutputSize);
            if (cuMemcpyDtoH(result.legacyOutputData.data(), d_outputs[0], task.legacyOutputSize) != CUDA_SUCCESS) {
                for (auto ptr : d_inputs) {
                    if (ptr) cuMemFree(ptr);
                }
                for (auto ptr : d_outputs) {
                    if (ptr) cuMemFree(ptr);
                }
                result.success = false;
                result.errorMessage = "Failed to copy output data";
                return result;
            }

            // Also populate new output data structure for consistency
            result.outputData = { result.legacyOutputData };
        }

        // Cleanup device memory
        for (auto ptr : d_inputs) {
            if (ptr) cuMemFree(ptr);
        }
        for (auto ptr : d_outputs) {
            if (ptr) cuMemFree(ptr);
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        result.processingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        result.success = true;

        std::cout << "CUDA Task completed successfully: " << result.outputData.size() << " outputs, "
                  << result.processingTime << "ms" << std::endl;

    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("Exception: ") + e.what();
    }

    return result;
}

json CudaExecutor::getCapabilities() const {
    json caps;
    caps["framework"] = "cuda";
    caps["initialized"] = initialized;
    caps["supportsMultiInput"] = true;   // NEW: Advertise multi-input support
    caps["supportsMultiOutput"] = true;  // NEW: Advertise multi-output support
    caps["maxInputs"] = 4;               // NEW: Maximum number of inputs supported
    caps["maxOutputs"] = 3;              // NEW: Maximum number of outputs supported

    if (initialized) {
        caps["device"] = {
            {"id", deviceId},
            {"name", deviceProps.name},
            {"computeCapability", {deviceProps.major, deviceProps.minor}},
            {"globalMemory", deviceProps.totalGlobalMem},
            {"maxThreadsPerBlock", deviceProps.maxThreadsPerBlock},
            {"multiProcessorCount", deviceProps.multiProcessorCount}
        };
    }

    return caps;
}