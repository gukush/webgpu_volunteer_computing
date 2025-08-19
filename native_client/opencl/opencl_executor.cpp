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

    // Get devices
    cl_uint numDevices;
    cl_device_type deviceType = CL_DEVICE_TYPE_GPU;

    if (config.contains("deviceType")) {
        std::string devTypeStr = config["deviceType"];
        if (devTypeStr == "CPU") deviceType = CL_DEVICE_TYPE_CPU;
        else if (devTypeStr == "GPU") deviceType = CL_DEVICE_TYPE_GPU;
        else if (devTypeStr == "ALL") deviceType = CL_DEVICE_TYPE_ALL;
    }

    err = clGetDeviceIDs(platform, deviceType, 0, nullptr, &numDevices);
    if (err != CL_SUCCESS || numDevices == 0) {
        std::cerr << "No suitable OpenCL devices found" << std::endl;
        return false;
    }

    std::vector<cl_device_id> devices(numDevices);
    err = clGetDeviceIDs(platform, deviceType, numDevices, devices.data(), nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get OpenCL devices" << std::endl;
        return false;
    }

    // Use first device or user-specified
    int deviceId = config.value("deviceId", 0);
    if (deviceId >= static_cast<int>(numDevices)) {
        deviceId = 0;
    }
    device = devices[deviceId];

    // Get device info
    char deviceName[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    std::cout << "Using OpenCL device: " << deviceName << std::endl;

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

bool OpenCLExecutor::setKernelArguments(cl_kernel kernel, const BufferSet& buffers, const TaskData& task) {
    cl_int err;
    int argIndex = 0;

    // Set input buffers as arguments
    for (size_t i = 0; i < buffers.inputBuffers.size(); i++) {
        err = clSetKernelArg(kernel, argIndex++, sizeof(cl_mem), &buffers.inputBuffers[i]);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to set input buffer argument " << i << " at index " << (argIndex-1) << std::endl;
            return false;
        }
    }

    // Set output buffers as arguments
    for (size_t i = 0; i < buffers.outputBuffers.size(); i++) {
        err = clSetKernelArg(kernel, argIndex++, sizeof(cl_mem), &buffers.outputBuffers[i]);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to set output buffer argument " << i << " at index " << (argIndex-1) << std::endl;
            return false;
        }
    }

    // Set additional arguments from chunk uniforms if present
    if (!task.chunkUniforms.empty()) {
        for (auto& [key, value] : task.chunkUniforms.items()) {
            if (value.is_number_integer()) {
                int intVal = value;
                err = clSetKernelArg(kernel, argIndex++, sizeof(int), &intVal);
            } else if (value.is_number_float()) {
                float floatVal = value;
                err = clSetKernelArg(kernel, argIndex++, sizeof(float), &floatVal);
            } else if (value.is_number_unsigned()) {
                unsigned int uintVal = value;
                err = clSetKernelArg(kernel, argIndex++, sizeof(unsigned int), &uintVal);
            } else {
                std::cerr << "Unsupported uniform type for key: " << key << std::endl;
                continue;
            }

            if (err != CL_SUCCESS) {
                std::cerr << "Failed to set chunk uniform argument: " << key << " at index " << (argIndex-1) << std::endl;
                return false;
            }
        }
    }

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

    BufferSet buffers;

    try {
        // Compile kernel if not cached
        std::string cacheKey = task.kernel + "|" + task.entry;
        auto it = kernelCache.find(cacheKey);
        CompiledKernel* kernelPtr;

        if (it == kernelCache.end()) {
            CompiledKernel newKernel;
            if (!compileKernel(task.kernel, task.entry, task.compilationOptions, newKernel)) {
                result.success = false;
                result.errorMessage = "Kernel compilation failed";
                return result;
            }
            kernelCache[cacheKey] = std::move(newKernel);
            kernelPtr = &kernelCache[cacheKey];
        } else {
            kernelPtr = &it->second;
        }

        std::cout << "Executing OpenCL kernel with " << task.getInputCount()
                  << " inputs and " << task.getOutputCount() << " outputs" << std::endl;

        // Create input buffers
        if (!createInputBuffers(task, buffers)) {
            result.success = false;
            result.errorMessage = "Failed to create input buffers";
            return result;
        }

        // Create output buffers
        if (!createOutputBuffers(task, buffers)) {
            result.success = false;
            result.errorMessage = "Failed to create output buffers";
            return result;
        }

        // Set kernel arguments
        if (!setKernelArguments(kernelPtr->kernel, buffers, task)) {
            result.success = false;
            result.errorMessage = "Failed to set kernel arguments";
            return result;
        }

        // Execute kernel
        size_t globalWorkSize[3] = {
            static_cast<size_t>(task.workgroupCount.size() > 0 ? task.workgroupCount[0] : 1),
            static_cast<size_t>(task.workgroupCount.size() > 1 ? task.workgroupCount[1] : 1),
            static_cast<size_t>(task.workgroupCount.size() > 2 ? task.workgroupCount[2] : 1)
        };

        cl_int err = clEnqueueNDRangeKernel(queue, kernelPtr->kernel, 3, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr);
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

        // Read results from all output buffers
        if (!readOutputBuffers(buffers, task, result)) {
            result.success = false;
            result.errorMessage = "Failed to read output buffers";
            return result;
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        result.processingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        result.success = true;

        std::cout << "OpenCL execution completed: " << result.getOutputCount() << " outputs, "
                  << result.processingTime << "ms" << std::endl;

    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("Exception: ") + e.what();
    }

    return result;
}

json OpenCLExecutor::getCapabilities() const {
    json caps;
    caps["framework"] = "opencl";
    caps["initialized"] = initialized;
    caps["supportsMultiInput"] = true;  // NEW: Advertise multi-input/output support
    caps["supportsMultiOutput"] = true;
    caps["maxInputs"] = 4;              // NEW: Practical limits
    caps["maxOutputs"] = 3;

    if (initialized && device) {
        char deviceName[256];
        char deviceVendor[256];
        cl_device_type deviceType;
        cl_ulong globalMemSize;
        cl_uint maxComputeUnits;

        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
        clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(deviceVendor), deviceVendor, nullptr);
        clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr);
        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, nullptr);
        clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, nullptr);

        caps["device"] = {
            {"name", deviceName},
            {"vendor", deviceVendor},
            {"type", (deviceType == CL_DEVICE_TYPE_GPU) ? "GPU" :
                    (deviceType == CL_DEVICE_TYPE_CPU) ? "CPU" : "Other"},
            {"globalMemory", globalMemSize},
            {"computeUnits", maxComputeUnits}
        };
    }

    return caps;
}