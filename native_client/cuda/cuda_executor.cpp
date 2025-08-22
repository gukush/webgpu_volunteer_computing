#include "cuda_executor.hpp"
#include <cuda.h>
#include <iostream>
#include <chrono>
#include <algorithm>

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

// NEW: Task-agnostic metadata processing (matching OpenCL/Vulkan approach)
bool CudaExecutor::processMetadataUniforms(const TaskData& task, std::vector<UniformValue>& uniforms) {
    // NEW: Process metadata first, skip chunkUniforms if metadata exists
    const json* sourceData = nullptr;
    std::string sourceType;

    if (!task.metadata.empty()) {
        sourceData = &task.metadata;
        sourceType = "metadata";
        std::cout << "Processing metadata uniforms..." << std::endl;
    } else if (!task.chunkUniforms.empty()) {
        sourceData = &task.chunkUniforms;
        sourceType = "legacy chunk uniforms";
        std::cout << "Processing legacy chunk uniforms..." << std::endl;
    } else {
        std::cout << "No uniforms to process" << std::endl;
        return true;
    }

    // Extract uniform values in consistent order
    std::vector<std::string> fieldOrder = {
        "block_size", "matrix_size", "matrix_n", "matrixSize",
        "tile_start_row", "tile_start_col", "tile_rows", "tile_cols",
        "tile_size", "tileSize"
    };

    for (const auto& field : fieldOrder) {
        if (sourceData->contains(field)) {
            UniformValue uv;
            uv.name = field;

            if ((*sourceData)[field].is_number_integer()) {
                uv.type = UniformType::INT32;
                uv.intValue = (*sourceData)[field].get<int32_t>();
            } else if ((*sourceData)[field].is_number_unsigned()) {
                uv.type = UniformType::UINT32;
                uv.uintValue = (*sourceData)[field].get<uint32_t>();
            } else {
                uv.type = UniformType::FLOAT;
                uv.floatValue = (*sourceData)[field].get<float>();
            }

            uniforms.push_back(uv);
            std::cout << "  " << field << " = " << (uv.type == UniformType::FLOAT ?
                std::to_string(uv.floatValue) : std::to_string(uv.intValue))
                << " (" << sourceType << ")" << std::endl;
        }
    }

    // Add any remaining numeric values not in fieldOrder
    for (auto it = sourceData->begin(); it != sourceData->end(); ++it) {
        if (it.value().is_number() &&
            std::find(fieldOrder.begin(), fieldOrder.end(), it.key()) == fieldOrder.end()) {

            UniformValue uv;
            uv.name = it.key();

            if (it.value().is_number_integer()) {
                uv.type = UniformType::INT32;
                uv.intValue = it.value().get<int32_t>();
            } else if (it.value().is_number_unsigned()) {
                uv.type = UniformType::UINT32;
                uv.uintValue = it.value().get<uint32_t>();
            } else {
                uv.type = UniformType::FLOAT;
                uv.floatValue = it.value().get<float>();
            }

            uniforms.push_back(uv);
            std::cout << "  " << it.key() << " = " << (uv.type == UniformType::FLOAT ?
                std::to_string(uv.floatValue) : std::to_string(uv.intValue))
                << " (extra " << sourceType << ")" << std::endl;
        }
    }

    return true;
}

// NEW: Add uniforms to kernel arguments with proper type handling
void CudaExecutor::addUniformsToKernelArgs(const std::vector<UniformValue>& uniforms,
                                          std::vector<void*>& kernelArgs,
                                          std::vector<int32_t>& intStorage,
                                          std::vector<uint32_t>& uintStorage,
                                          std::vector<float>& floatStorage) {
    for (const auto& uniform : uniforms) {
        switch (uniform.type) {
            case UniformType::INT32:
                intStorage.push_back(uniform.intValue);
                kernelArgs.push_back(&intStorage.back());
                break;
            case UniformType::UINT32:
                uintStorage.push_back(uniform.uintValue);
                kernelArgs.push_back(&uintStorage.back());
                break;
            case UniformType::FLOAT:
                floatStorage.push_back(uniform.floatValue);
                kernelArgs.push_back(&floatStorage.back());
                break;
        }
    }
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
        bool hasUniforms = !task.metadata.empty() || !task.chunkUniforms.empty();

        const size_t inputCount = useMultipleInputs ? task.inputData.size() :
                                 (task.legacyInputData.empty() ? 0 : 1);
        const size_t outputCount = useMultipleOutputs ? task.outputSizes.size() :
                                  (task.legacyOutputSize ? 1 : 0);

        std::cout << "CUDA Task " << task.id << " - Inputs: " << inputCount
                  << ", Outputs: " << outputCount << ", Has uniforms: " << hasUniforms << std::endl;


        if (useMultipleInputs) {
            for (size_t i = 0; i < task.inputData.size(); i++) {
                const auto& inputData = task.inputData[i];
                if (!inputData.empty() && inputData.size() >= 16) {
                    // Print first 4 floats of each input
                    const float* floats = reinterpret_cast<const float*>(inputData.data());
                    std::cout << "Input " << i << " first 4 values: ";
                    for (int j = 0; j < 4 && j < (int)(inputData.size()/4); j++) {
                        std::cout << floats[j] << " ";
                    }
                    std::cout << std::endl;
                }
            }
        } else if (!task.legacyInputData.empty() && task.legacyInputData.size() >= 16) {
            const float* floats = reinterpret_cast<const float*>(task.legacyInputData.data());
            std::cout << "Legacy input first 4 values: ";
            for (int j = 0; j < 4 && j < (int)(task.legacyInputData.size()/4); j++) {
                std::cout << floats[j] << " ";
            }
            std::cout << std::endl;
        }
        std::vector<CUdeviceptr> d_inputs;
        std::vector<CUdeviceptr> d_outputs;
        std::vector<void*> kernelArgs;

        // NEW: Process uniforms first (task-agnostic approach)
        std::vector<UniformValue> uniforms;
        std::vector<int32_t> intStorage;
        std::vector<uint32_t> uintStorage;
        std::vector<float> floatStorage;

        if (hasUniforms) {
            processMetadataUniforms(task, uniforms);
            addUniformsToKernelArgs(uniforms, kernelArgs, intStorage, uintStorage, floatStorage);
            std::cout << "Set " << uniforms.size() << " uniform arguments" << std::endl;
        }

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
                    std::cout << "Input " << i << ": " << inputData.size() << " bytes" << std::endl;
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
                std::cout << "Legacy input: " << task.legacyInputData.size() << " bytes" << std::endl;
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
                std::cout << "Output " << i << ": " << task.outputSizes[i] << " bytes" << std::endl;
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
            std::cout << "Legacy output: " << task.legacyOutputSize << " bytes" << std::endl;
        }

        if (useMultipleOutputs) {
            for (size_t i = 0; i < result.outputData.size(); i++) {
                if (!result.outputData[i].empty() && result.outputData[i].size() >= 16) {
                    const float* floats = reinterpret_cast<const float*>(result.outputData[i].data());
                    std::cout << "Output " << i << " first 4 values: ";
                    for (int j = 0; j < 4 && j < (int)(result.outputData[i].size()/4); j++) {
                        std::cout << floats[j] << " ";
                    }
                    std::cout << std::endl;
                }
            }
        } else if (!result.legacyOutputData.empty() && result.legacyOutputData.size() >= 16) {
            const float* floats = reinterpret_cast<const float*>(result.legacyOutputData.data());
            std::cout << "Legacy output first 4 values: ";
            for (int j = 0; j < 4 && j < (int)(result.legacyOutputData.size()/4); j++) {
                std::cout << floats[j] << " ";
            }
            std::cout << std::endl;
        }
        // Determine grid and block dimensions
        std::vector<int> gridDim = task.workgroupCount.empty() ? std::vector<int>{1, 1, 1} : task.workgroupCount;
        std::vector<int> blockDim = {16, 16, 1};  // Default block size

        // PRIORITY: Use metadata blockDim/gridDim if specified
        if (task.metadata.contains("blockDim") && task.metadata["blockDim"].is_array()) {
            auto bdim = task.metadata["blockDim"].get<std::vector<int>>();
            if (bdim.size() >= 3) {
                blockDim = bdim;
                std::cout << "Using metadata blockDim: " << blockDim[0] << "," << blockDim[1] << "," << blockDim[2] << std::endl;
            }
        }
        if (task.metadata.contains("gridDim") && task.metadata["gridDim"].is_array()) {
            auto gdim = task.metadata["gridDim"].get<std::vector<int>>();
            if (gdim.size() >= 3) {
                gridDim = gdim;
                std::cout << "Using metadata gridDim: " << gridDim[0] << "," << gridDim[1] << "," << gridDim[2] << std::endl;
            }
        }
        std::cout << "Block size from uniforms: ";
        for (const auto& uniform : uniforms) {
            if (uniform.name == "block_size") {
                std::cout << (uniform.type == UniformType::INT32 ? uniform.intValue : uniform.uintValue);
                break;
            }
        }
        std::cout << std::endl;

        std::cout << "Final kernel launch parameters:" << std::endl;
        std::cout << "  Grid: (" << gridDim[0] << "," << gridDim[1] << "," << gridDim[2] << ")" << std::endl;
        std::cout << "  Block: (" << blockDim[0] << "," << blockDim[1] << "," << blockDim[2] << ")" << std::endl;
        std::cout << "  Total threads: " << (gridDim[0] * blockDim[0]) << "x" << (gridDim[1] * blockDim[1]) << std::endl;
        std::cout << "Kernel launch: grid(" << gridDim[0] << "," << gridDim[1] << "," << gridDim[2]
                  << ") block(" << blockDim[0] << "," << blockDim[1] << "," << blockDim[2] << ")" << std::endl;
        std::cout << "Total kernel arguments: " << kernelArgs.size()
                  << " (uniforms:" << uniforms.size() << " + inputs:" << d_inputs.size()
                  << " + outputs:" << d_outputs.size() << ")" << std::endl;

        // Launch kernel with all arguments (uniforms + inputs + outputs)
        CUresult launchResult = cuLaunchKernel(
            kernel->function,
            gridDim[0], gridDim[1], gridDim[2],        // grid dimensions
            blockDim[0], blockDim[1], blockDim[2],     // block dimensions
            0, 0,                                       // shared mem, stream
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
            result.errorMessage = "Kernel launch failed with error: " + std::to_string(launchResult);
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
                std::cout << "Retrieved output " << i << ": " << result.outputData[i].size() << " bytes" << std::endl;
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
            std::cout << "Retrieved legacy output: " << result.legacyOutputData.size() << " bytes" << std::endl;
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

        std::cout << "CUDA Task " << task.id << " completed successfully: " << result.outputData.size()
                  << " outputs in " << result.processingTime << "ms" << std::endl;

    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("Exception: ") + e.what();
        std::cout << "CUDA Task " << task.id << " failed: " << e.what() << std::endl;
    }

    return result;
}

json CudaExecutor::getCapabilities() const {
    json caps;
    caps["framework"] = "cuda";
    caps["initialized"] = initialized;
    caps["supportsMultiInput"] = true;   // NEW: Advertise multi-input support
    caps["supportsMultiOutput"] = true;  // NEW: Advertise multi-output support
    caps["maxInputs"] = 8;               // NEW: Increased capacity
    caps["maxOutputs"] = 4;              // NEW: Increased capacity
    caps["supportsUniforms"] = true;     // NEW: Advertise uniform support
    caps["taskAgnostic"] = true;         // NEW: Indicate task-agnostic capability

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