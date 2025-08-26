// opencl/opencl_executor.cpp
#include "opencl_executor.hpp"
#include <iostream>
#include <fstream>
#include <chrono>

OpenCLExecutor::OpenCLExecutor() {}

OpenCLExecutor::~OpenCLExecutor() {
    cleanup();
}


bool OpenCLExecutor::initialize(const json& config) {
    if (initialized) return true;

    cl_int err;

    // Get platform
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        std::cerr << "No OpenCL platforms found" << std::endl;
        return false;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get OpenCL platforms" << std::endl;
        return false;
    }

    // Use first platform or user-specified
    int platformId = config.value("platformId", 0);
    if (platformId >= static_cast<int>(numPlatforms)) {
        platformId = 0;
    }
    platform = platforms[platformId];

    // Get platform info
    char platformName[256];
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr);
    std::cout << "Using OpenCL platform: " << platformName << std::endl;

    // ENHANCED: Explicit device type selection
    cl_device_type deviceType = CL_DEVICE_TYPE_GPU; // Default
    std::string requestedDeviceType = "auto";

    if (config.contains("deviceType")) {
        requestedDeviceType = config["deviceType"].get<std::string>();

        if (requestedDeviceType == "cpu" || requestedDeviceType == "CPU") {
            deviceType = CL_DEVICE_TYPE_CPU;
            std::cout << "️  Explicitly requesting CPU devices" << std::endl;
        } else if (requestedDeviceType == "gpu" || requestedDeviceType == "GPU") {
            deviceType = CL_DEVICE_TYPE_GPU;
            std::cout << " Explicitly requesting GPU devices" << std::endl;
        } else if (requestedDeviceType == "all" || requestedDeviceType == "ALL") {
            deviceType = CL_DEVICE_TYPE_ALL;
            std::cout << " Requesting all device types" << std::endl;
        } else if (requestedDeviceType == "auto") {
            deviceType = CL_DEVICE_TYPE_GPU; // Start with GPU
            std::cout << " Auto-selecting devices (GPU preferred)" << std::endl;
        }
    }

    // Get devices with improved selection logic
    cl_uint numDevices = 0;
    std::vector<cl_device_id> devices;
    std::string finalDeviceTypeStr;

    // Try requested device type first
    err = clGetDeviceIDs(platform, deviceType, 0, nullptr, &numDevices);

    if (err == CL_SUCCESS && numDevices > 0) {
        devices.resize(numDevices);
        err = clGetDeviceIDs(platform, deviceType, numDevices, devices.data(), nullptr);

        if (deviceType == CL_DEVICE_TYPE_CPU) finalDeviceTypeStr = "CPU";
        else if (deviceType == CL_DEVICE_TYPE_GPU) finalDeviceTypeStr = "GPU";
        else finalDeviceTypeStr = "Mixed";

        std::cout << " Found " << numDevices << " " << finalDeviceTypeStr << " device(s)" << std::endl;
    }

    // Fallback logic for auto mode
    if ((err != CL_SUCCESS || numDevices == 0) && requestedDeviceType == "auto") {
        std::cout << "No GPU devices found, falling back to CPU devices..." << std::endl;
        deviceType = CL_DEVICE_TYPE_CPU;
        err = clGetDeviceIDs(platform, deviceType, 0, nullptr, &numDevices);

        if (err == CL_SUCCESS && numDevices > 0) {
            devices.resize(numDevices);
            err = clGetDeviceIDs(platform, deviceType, numDevices, devices.data(), nullptr);
            finalDeviceTypeStr = "CPU";
            std::cout << " Found " << numDevices << " CPU device(s)" << std::endl;
        }
    }

    // Final fallback to any device type
    if (err != CL_SUCCESS || numDevices == 0) {
        std::cout << "No devices found for requested type, trying all device types..." << std::endl;
        deviceType = CL_DEVICE_TYPE_ALL;
        err = clGetDeviceIDs(platform, deviceType, 0, nullptr, &numDevices);

        if (err == CL_SUCCESS && numDevices > 0) {
            devices.resize(numDevices);
            err = clGetDeviceIDs(platform, deviceType, numDevices, devices.data(), nullptr);
            finalDeviceTypeStr = "Any";
            std::cout << " Found " << numDevices << " device(s) of any type" << std::endl;
        }
    }

    if (err != CL_SUCCESS || numDevices == 0) {
        std::cerr << " No suitable OpenCL devices found" << std::endl;
        return false;
    }

    // Use first device or user-specified
    int deviceId = config.value("deviceId", 0);
    if (deviceId >= static_cast<int>(numDevices)) {
        deviceId = 0;
    }
    device = devices[deviceId];

    // Get detailed device info
    char deviceName[256];
    char deviceVendor[256];
    cl_device_type actualDeviceType;
    cl_ulong globalMemSize;
    cl_uint maxComputeUnits;
    cl_uint maxWorkGroupSize;

    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(deviceVendor), deviceVendor, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(actualDeviceType), &actualDeviceType, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);

    std::string actualTypeStr;
    if (actualDeviceType == CL_DEVICE_TYPE_CPU) actualTypeStr = "CPU";
    else if (actualDeviceType == CL_DEVICE_TYPE_GPU) actualTypeStr = "GPU";
    else if (actualDeviceType == CL_DEVICE_TYPE_ACCELERATOR) actualTypeStr = "Accelerator";
    else actualTypeStr = "Other";

    std::cout << " Selected OpenCL device:" << std::endl;
    std::cout << "   Name: " << deviceName << std::endl;
    std::cout << "   Vendor: " << deviceVendor << std::endl;
    std::cout << "   Type: " << actualTypeStr << std::endl;
    std::cout << "   Compute Units: " << maxComputeUnits << std::endl;
    std::cout << "   Max Work Group Size: " << maxWorkGroupSize << std::endl;
    std::cout << "   Global Memory: " << (globalMemSize / (1024 * 1024)) << " MB" << std::endl;

    // Store device type info for capabilities reporting
    selectedDeviceType = actualDeviceType;
    selectedDeviceTypeStr = actualTypeStr;

    // Create context
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context" << std::endl;
        return false;
    }

    // Create command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL command queue" << std::endl;
        return false;
    }

    initialized = true;
    std::cout << " OpenCL executor initialized successfully with " << actualTypeStr << " device" << std::endl;

    return true;
}

void OpenCLExecutor::cleanup() {
    if (!initialized) return;

    // Cleanup cached kernels
    for (auto& [key, kernel] : kernelCache) {
        if (kernel.kernel) clReleaseKernel(kernel.kernel);
        if (kernel.program) clReleaseProgram(kernel.program);
    }
    kernelCache.clear();

    if (queue) {
        clReleaseCommandQueue(queue);
        queue = nullptr;
    }

    if (context) {
        clReleaseContext(context);
        context = nullptr;
    }

    initialized = false;
}

bool OpenCLExecutor::compileKernel(const std::string& source, const std::string& entryPoint,
                                  const json& compileOpts, CompiledKernel& result) {
    cl_int err;

    // Create program from source
    const char* sourcePtr = source.c_str();
    size_t sourceSize = source.length();

    result.program = clCreateProgramWithSource(context, 1, &sourcePtr, &sourceSize, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL program" << std::endl;
        return false;
    }

    // Build program
    std::string buildOptions;
    if (compileOpts.contains("buildOptions")) {
        buildOptions = compileOpts["buildOptions"];
    }

    err = clBuildProgram(result.program, 1, &device, buildOptions.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Get build log
        size_t logSize;
        clGetProgramBuildInfo(result.program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

        std::vector<char> buildLog(logSize);
        clGetProgramBuildInfo(result.program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);

        std::cerr << "OpenCL build failed:\n" << buildLog.data() << std::endl;

        clReleaseProgram(result.program);
        return false;
    }

    // Create kernel
    result.kernel = clCreateKernel(result.program, entryPoint.c_str(), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL kernel: " << entryPoint << std::endl;
        clReleaseProgram(result.program);
        return false;
    }

    return true;
}

bool OpenCLExecutor::createInputBuffers(const TaskData& task, BufferSet& buffers) {
    cl_int err;

    for (size_t i = 0; i < task.inputData.size(); i++) {
        const auto& inputData = task.inputData[i];

        if (inputData.empty()) {
            // Empty input - create a small placeholder buffer
            cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 4, nullptr, &err);
            if (err != CL_SUCCESS) {
                std::cerr << "Failed to create placeholder input buffer " << i << std::endl;
                return false;
            }
            buffers.inputBuffers.push_back(buffer);
        } else {
            // Create buffer with data
            cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          inputData.size(), const_cast<uint8_t*>(inputData.data()), &err);
            if (err != CL_SUCCESS) {
                std::cerr << "Failed to create input buffer " << i << std::endl;
                return false;
            }
            buffers.inputBuffers.push_back(buffer);
        }
    }

    // If no inputs provided, create one from legacy data for backward compatibility
    if (buffers.inputBuffers.empty() && !task.legacyInputData.empty()) {
        cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      task.legacyInputData.size(),
                                      const_cast<uint8_t*>(task.legacyInputData.data()), &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to create legacy input buffer" << std::endl;
            return false;
        }
        buffers.inputBuffers.push_back(buffer);
    }

    return true;
}

bool OpenCLExecutor::createOutputBuffers(const TaskData& task, BufferSet& buffers) {
    cl_int err;

    for (size_t i = 0; i < task.outputSizes.size(); i++) {
        size_t outputSize = task.outputSizes[i];

        cl_mem buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, outputSize, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to create output buffer " << i << " (size: " << outputSize << ")" << std::endl;
            return false;
        }
        buffers.outputBuffers.push_back(buffer);
    }

    // If no output sizes provided, create one from legacy size for backward compatibility
    if (buffers.outputBuffers.empty() && task.legacyOutputSize > 0) {
        cl_mem buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, task.legacyOutputSize, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to create legacy output buffer" << std::endl;
            return false;
        }
        buffers.outputBuffers.push_back(buffer);
    }

    return true;
}

bool OpenCLExecutor::processMetadataUniforms(const TaskData& task, std::vector<UniformValue>& uniforms) {
    // Process metadata first (enhanced chunking system)
    if (!task.metadata.empty()) {
        std::cout << "Processing metadata uniforms..." << std::endl;

        // Extract uniform values in consistent order (matching other frameworks)
        std::vector<std::string> fieldOrder = {
            "block_size", "matrix_size", "matrix_n", "matrixSize",
            "tile_start_row", "tile_start_col", "tile_rows", "tile_cols",
            "tile_size", "tileSize"
        };

        for (const auto& field : fieldOrder) {
            if (task.metadata.contains(field)) {
                UniformValue uv;
                uv.name = field;
                uv.type = UniformType::UINT32;
                uv.uintValue = task.metadata[field].get<uint32_t>();
                uniforms.push_back(uv);
                std::cout << "  " << field << " = " << uv.uintValue << std::endl;
            }
        }

        // Add any remaining numeric values not in fieldOrder
        for (auto it = task.metadata.begin(); it != task.metadata.end(); ++it) {
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
                    std::to_string(uv.floatValue) : std::to_string(uv.intValue)) << " (extra)" << std::endl;
            }
        }
    }

    // Process legacy chunkUniforms for backward compatibility
    if (!task.chunkUniforms.empty()) {
        std::cout << "Processing legacy chunk uniforms..." << std::endl;

        for (auto& [key, value] : task.chunkUniforms.items()) {
            // Skip if already processed in metadata
            bool alreadyProcessed = false;
            for (const auto& existing : uniforms) {
                if (existing.name == key) {
                    alreadyProcessed = true;
                    break;
                }
            }
            if (alreadyProcessed) continue;

            UniformValue uv;
            uv.name = key;

            if (value.is_number_integer()) {
                uv.type = UniformType::INT32;
                uv.intValue = value.get<int32_t>();
            } else if (value.is_number_unsigned()) {
                uv.type = UniformType::UINT32;
                uv.uintValue = value.get<uint32_t>();
            } else if (value.is_number_float()) {
                uv.type = UniformType::FLOAT;
                uv.floatValue = value.get<float>();
            } else {
                std::cerr << "Unsupported uniform type for key: " << key << std::endl;
                continue;
            }

            uniforms.push_back(uv);
        }
    }

    return true;
}

// Updated setKernelArguments with proper binding order
bool OpenCLExecutor::setKernelArguments(cl_kernel kernel, const BufferSet& buffers, const TaskData& task) {
    cl_int err;
    int argIndex = 0;

    // NEW: Process uniforms first (matching other frameworks)
    std::vector<UniformValue> uniforms;
    processMetadataUniforms(task, uniforms);

    // Set uniform arguments first
    for (const auto& uniform : uniforms) {
        switch (uniform.type) {
            case UniformType::INT32:
                err = clSetKernelArg(kernel, argIndex++, sizeof(int32_t), &uniform.intValue);
                break;
            case UniformType::UINT32:
                err = clSetKernelArg(kernel, argIndex++, sizeof(uint32_t), &uniform.uintValue);
                break;
            case UniformType::FLOAT:
                err = clSetKernelArg(kernel, argIndex++, sizeof(float), &uniform.floatValue);
                break;
        }

        if (err != CL_SUCCESS) {
            std::cerr << "Failed to set uniform argument: " << uniform.name
                      << " at index " << (argIndex-1) << std::endl;
            return false;
        }
    }

    std::cout << "Set " << uniforms.size() << " uniform arguments (bindings 0-"
              << (uniforms.size()-1) << ")" << std::endl;

    // Set input buffers as arguments (after uniforms)
    for (size_t i = 0; i < buffers.inputBuffers.size(); i++) {
        err = clSetKernelArg(kernel, argIndex++, sizeof(cl_mem), &buffers.inputBuffers[i]);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to set input buffer argument " << i
                      << " at index " << (argIndex-1) << std::endl;
            return false;
        }
    }

    std::cout << "Set " << buffers.inputBuffers.size() << " input buffer arguments (bindings "
              << uniforms.size() << "-" << (uniforms.size() + buffers.inputBuffers.size() - 1) << ")" << std::endl;

    // Set output buffers as arguments (after inputs)
    for (size_t i = 0; i < buffers.outputBuffers.size(); i++) {
        err = clSetKernelArg(kernel, argIndex++, sizeof(cl_mem), &buffers.outputBuffers[i]);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to set output buffer argument " << i
                      << " at index " << (argIndex-1) << std::endl;
            return false;
        }
    }

    std::cout << "Set " << buffers.outputBuffers.size() << " output buffer arguments (bindings "
              << (uniforms.size() + buffers.inputBuffers.size()) << "-"
              << (uniforms.size() + buffers.inputBuffers.size() + buffers.outputBuffers.size() - 1) << ")" << std::endl;

    return true;
}

bool OpenCLExecutor::readOutputBuffers(const BufferSet& buffers, const TaskData& task, TaskResult& result) {
    cl_int err;

    // Read all output buffers
    for (size_t i = 0; i < buffers.outputBuffers.size(); i++) {
        size_t outputSize = (i < task.outputSizes.size()) ? task.outputSizes[i] : task.legacyOutputSize;

        std::vector<uint8_t> outputData(outputSize);
        err = clEnqueueReadBuffer(queue, buffers.outputBuffers[i], CL_TRUE, 0, outputSize,
                                 outputData.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to read output buffer " << i << std::endl;
            return false;
        }

        result.outputData.push_back(std::move(outputData));
    }

    // Set legacy output for backward compatibility
    if (!result.outputData.empty()) {
        result.legacyOutputData = result.outputData[0];
    }

    return true;
}

TaskResult OpenCLExecutor::executeTask(const TaskData& task) {
    TaskResult result;
    auto startTime = std::chrono::high_resolution_clock::now();

    if (!initialized) {
        result.success = false;
        result.errorMessage = "OpenCL not initialized";
        return result;
    }

    // Determine IO counts and metadata presence
    const bool multipleInputs = !task.inputData.empty();
    const bool multipleOutputs = !task.outputSizes.empty();
    const bool hasUniforms = !task.metadata.empty() || !task.chunkUniforms.empty();

    const size_t inputCount = multipleInputs ? task.inputData.size() :
                             (task.legacyInputData.empty() ? 0 : 1);
    const size_t outputCount = multipleOutputs ? task.outputSizes.size() :
                              (task.legacyOutputSize ? 1 : 0);

    std::cout << "Task " << task.id << " - Framework: " << task.framework
              << ", Inputs: " << inputCount << ", Outputs: " << outputCount
              << ", Has uniforms: " << hasUniforms << std::endl;

    if (outputCount == 0) {
        result.success = false;
        result.errorMessage = "No outputs requested";
        return result;
    }

    BufferSet buffers;

    try {
        // Compile kernel if not cached
        std::string cacheKey = task.kernel + "|" + task.entry;
        auto it = kernelCache.find(cacheKey);
        CompiledKernel* kernelPtr;

        if (it == kernelCache.end()) {
            std::cout << "Compiling OpenCL kernel..." << std::endl;
            CompiledKernel newKernel;
            if (!compileKernel(task.kernel, task.entry, task.compilationOptions, newKernel)) {
                result.success = false;
                result.errorMessage = "Kernel compilation failed";
                return result;
            }
            kernelCache[cacheKey] = std::move(newKernel);
            kernelPtr = &kernelCache[cacheKey];
            std::cout << "Kernel compiled and cached successfully" << std::endl;
        } else {
            kernelPtr = &it->second;
            std::cout << "Using cached kernel" << std::endl;
        }

        // Create input buffers
        std::cout << "Creating input buffers..." << std::endl;
        if (!createInputBuffers(task, buffers)) {
            result.success = false;
            result.errorMessage = "Failed to create input buffers";
            return result;
        }

        for (size_t i = 0; i < buffers.inputBuffers.size(); ++i) {
            size_t inputSize = (i < task.inputData.size()) ? task.inputData[i].size() : task.legacyInputData.size();
            std::cout << "Input buffer " << i << ": " << inputSize << " bytes" << std::endl;
        }

        // Create output buffers
        std::cout << "Creating output buffers..." << std::endl;
        if (!createOutputBuffers(task, buffers)) {
            result.success = false;
            result.errorMessage = "Failed to create output buffers";
            return result;
        }

        for (size_t i = 0; i < buffers.outputBuffers.size(); ++i) {
            size_t outputSize = (i < task.outputSizes.size()) ? task.outputSizes[i] : task.legacyOutputSize;
            std::cout << "Output buffer " << i << ": " << outputSize << " bytes" << std::endl;
        }

        // Set kernel arguments (enhanced with proper binding order)
        std::cout << "Setting kernel arguments..." << std::endl;
        if (!setKernelArguments(kernelPtr->kernel, buffers, task)) {
            result.success = false;
            result.errorMessage = "Failed to set kernel arguments";
            return result;
        }

        // Execute kernel
        std::cout << "Executing OpenCL kernel..." << std::endl;

        // Determine work dimensions and sizes
        size_t workDim = 1;
        size_t globalWorkSize[3] = {1, 1, 1};

        if (!task.workgroupCount.empty()) {
            workDim = std::min<size_t>(3, task.workgroupCount.size());
            for (size_t i = 0; i < workDim; ++i) {
                globalWorkSize[i] = static_cast<size_t>(std::max(1, task.workgroupCount[i]));
            }
        }

        std::cout << "Dispatching kernel: " << globalWorkSize[0];
        if (workDim > 1) std::cout << "x" << globalWorkSize[1];
        if (workDim > 2) std::cout << "x" << globalWorkSize[2];
        std::cout << " global work items" << std::endl;

        cl_int err = clEnqueueNDRangeKernel(queue, kernelPtr->kernel, workDim, nullptr,
                                           globalWorkSize, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            result.success = false;
            result.errorMessage = "Kernel execution failed with error: " + std::to_string(err);
            return result;
        }

        // Wait for completion
        err = clFinish(queue);
        if (err != CL_SUCCESS) {
            result.success = false;
            result.errorMessage = "Failed to wait for kernel completion";
            return result;
        }

        std::cout << "Kernel execution completed, reading results..." << std::endl;

        // Read results from all output buffers
        if (!readOutputBuffers(buffers, task, result)) {
            result.success = false;
            result.errorMessage = "Failed to read output buffers";
            return result;
        }

        // Log results
        if (result.hasMultipleOutputs()) {
            for (size_t i = 0; i < result.outputData.size(); ++i) {
                std::cout << "Retrieved output " << i << ": " << result.outputData[i].size() << " bytes" << std::endl;
            }
        } else {
            std::cout << "Retrieved legacy output: " << result.legacyOutputData.size() << " bytes" << std::endl;
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        result.processingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        result.success = true;

        std::cout << "Task " << task.id << " completed successfully in "
                  << result.processingTime << "ms" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Task " << task.id << " failed: " << e.what() << std::endl;
        result.success = false;
        result.errorMessage = std::string("Exception: ") + e.what();
    }

    return result;
}
json OpenCLExecutor::getCapabilities() const {
    json caps;
    caps["framework"] = "opencl";
    caps["initialized"] = initialized;
    caps["supportsMultiInput"] = true;
    caps["supportsMultiOutput"] = true;
    caps["maxInputs"] = 4;
    caps["maxOutputs"] = 3;

    if (initialized && device) {
        char deviceName[256];
        char deviceVendor[256];
        cl_ulong globalMemSize;
        cl_uint maxComputeUnits;
        cl_uint maxWorkGroupSize;

        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
        clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(deviceVendor), deviceVendor, nullptr);
        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, nullptr);
        clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, nullptr);
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);

        caps["device"] = {
            {"name", deviceName},
            {"vendor", deviceVendor},
            {"type", selectedDeviceTypeStr},
            {"isCPU", selectedDeviceType == CL_DEVICE_TYPE_CPU},
            {"isGPU", selectedDeviceType == CL_DEVICE_TYPE_GPU},
            {"globalMemory", globalMemSize},
            {"computeUnits", maxComputeUnits},
            {"maxWorkGroupSize", maxWorkGroupSize}
        };

        // Add performance hints based on device type
        if (selectedDeviceType == CL_DEVICE_TYPE_CPU) {
            caps["performanceHints"] = {
                {"preferredWorkGroupSize", std::min(maxWorkGroupSize, 64u)},
                {"memoryPattern", "sequential_preferred"},
                {"parallelismLevel", "thread_level"},
                {"cacheFriendly", true}
            };
        } else {
            caps["performanceHints"] = {
                {"preferredWorkGroupSize", std::min(maxWorkGroupSize, 256u)},
                {"memoryPattern", "coalesced_preferred"},
                {"parallelismLevel", "massive_parallel"},
                {"cacheFriendly", false}
            };
        }
    }

    return caps;
}