// cuda_executor.cpp
#include "cuda_executor.hpp"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <sstream>
#include <iomanip>

// ================= Error helpers =================
void CudaExecutor::logCuError(const char* where, CUresult r) {
    const char* name = nullptr;
    const char* desc = nullptr;
    cuGetErrorName(r, &name);
    cuGetErrorString(r, &desc);
    std::cerr << where << " failed: "
              << (name ? name : "?") << " - "
              << (desc ? desc : "?") << std::endl;
}
bool CudaExecutor::cuCheck(CUresult r, const char* where) {
    if (r == CUDA_SUCCESS) return true;
    logCuError(where, r);
    return false;
}
bool CudaExecutor::nvrtcCheck(nvrtcResult r, const char* where, const std::string& log) {
    if (r == NVRTC_SUCCESS) return true;
    std::cerr << where << " failed: " << nvrtcGetErrorString(r) << std::endl;
    if (!log.empty()) {
        std::cerr << "---- NVRTC LOG ----\n" << log << "\n-------------------\n";
    }
    return false;
}

// ================= Small helpers =================
static inline int jsonToInt(const json& j, int def = 0) {
    if (j.is_number_integer())  return static_cast<int>(j.get<int64_t>());
    if (j.is_number_unsigned()) return static_cast<int>(j.get<uint64_t>());
    return def;
}
static inline unsigned jsonToUnsigned(const json& j, unsigned def = 0) {
    if (j.is_number_unsigned()) return static_cast<unsigned>(j.get<uint64_t>());
    if (j.is_number_integer())  return static_cast<unsigned>(j.get<int64_t>());
    return def;
}
static inline float jsonToFloat(const json& j, float def = 0.f) {
    if (j.is_number_float())   return static_cast<float>(j.get<double>());
    if (j.is_number_integer()) return static_cast<float>(j.get<int64_t>());
    return def;
}
static inline std::string getUniformValueStr(const CudaExecutor::UniformValue& uv) {
    switch (uv.type) {
        case CudaExecutor::UniformType::FLOAT:  return std::to_string(uv.floatValue);
        case CudaExecutor::UniformType::UINT32: return std::to_string(uv.uintValue);
        case CudaExecutor::UniformType::INT32:  return std::to_string(uv.intValue);
    }
    return "?";
}

std::string CudaExecutor::makeCacheKey(const std::string& source, const std::string& entry) {
    std::hash<std::string> H;
    size_t h = H(source);
    h ^= (H(entry) + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2));
    std::ostringstream os;
    os << std::hex << h;
    return os.str();
}

// ================= Constructor / Destructor =================
CudaExecutor::CudaExecutor(int devId) : deviceId(devId) {}

CudaExecutor::~CudaExecutor() {
    // Unload cached modules
    for (auto& kv : kernelCache) {
        if (kv.second.module) {
            cuModuleUnload(kv.second.module);
        }
    }
    kernelCache.clear();

    // Destroy context on shutdown
    if (context) {
        cuCtxDestroy(context);
        context = nullptr;
    }
    initialized = false;
}

// ================= Initialization / Context =================
bool CudaExecutor::ensureDriverLoaded() {
    static bool inited = false;
    if (inited) return true;
    if (!cuCheck(cuInit(0), "cuInit")) return false;
    inited = true;
    return true;
}

bool CudaExecutor::initialize(const json& /*config*/) {
    if (initialized) return true;
    if (!ensureDriverLoaded()) return false;

    int deviceCount = 0;
    if (!cuCheck(cuDeviceGetCount(&deviceCount), "cuDeviceGetCount")) return false;
    if (deviceCount <= 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return false;
    }
    if (deviceId < 0 || deviceId >= deviceCount) {
        std::cerr << "Invalid device ID: " << deviceId << std::endl;
        return false;
    }

    CUdevice device;
    if (!cuCheck(cuDeviceGet(&device, deviceId), "cuDeviceGet")) return false;

    // Query device info via Driver API
    char nameBuf[256] = {0};
    cuDeviceGetName(nameBuf, sizeof(nameBuf), device);
    deviceName = nameBuf;

    cuDeviceGetAttribute(&computeMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&computeMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    cuDeviceGetAttribute(&maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
    cuDeviceGetAttribute(&multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
    cuDeviceTotalMem(&totalGlobalMem, device);

    if (!cuCheck(cuCtxCreate(&context, 0, device), "cuCtxCreate")) return false;
    if (!cuCheck(cuCtxSetCurrent(context), "cuCtxSetCurrent")) return false;

    initialized = true;
    std::cout << "CUDA initialized on device " << deviceId
              << " (" << deviceName << "), CC " << computeMajor << "." << computeMinor
              << std::endl;
    return true;
}

bool CudaExecutor::ensureContextCurrent() {
    if (!initialized) return false;
    CUcontext cur = nullptr;
    if (!cuCheck(cuCtxGetCurrent(&cur), "cuCtxGetCurrent")) return false;
    if (cur == context) return true;
    return cuCheck(cuCtxSetCurrent(context), "cuCtxSetCurrent");
}

// cleanup() NO LONGER DESTROYS THE CONTEXT (kept across tasks)
void CudaExecutor::cleanup() {
    // Optionally evict compiled modules to save memory
    for (auto& kv : kernelCache) {
        if (kv.second.module) cuModuleUnload(kv.second.module);
    }
    kernelCache.clear();
    // Do NOT destroy context; keep executor initialized for subsequent tasks.
}

// ================= NVRTC compilation =================
bool CudaExecutor::compileKernel(const std::string& source,
                                 const std::string& entryPoint,
                                 const json& compileOpts,
                                 CompiledKernel& result) {
    nvrtcProgram prog = nullptr;
    nvrtcResult nres = nvrtcCreateProgram(&prog, source.c_str(), "kernel.cu", 0, nullptr, nullptr);
    if (!nvrtcCheck(nres, "nvrtcCreateProgram")) return false;

    // Build options
    std::vector<std::string> optStrings;
    // Target current device architecture
    {
        std::ostringstream arch;
        arch << "--gpu-architecture=compute_" << computeMajor << computeMinor;
        optStrings.push_back(arch.str());
    }
    // Standard & (optional) fast math
    optStrings.push_back("--std=c++14");

    // Custom options (if provided)
    if (compileOpts.contains("optimization") && compileOpts["optimization"].is_string()) {
        optStrings.push_back(compileOpts["optimization"].get<std::string>());
    }
    if (compileOpts.contains("options") && compileOpts["options"].is_array()) {
        for (const auto& o : compileOpts["options"]) {
            if (o.is_string()) optStrings.push_back(o.get<std::string>());
        }
    }

    std::vector<const char*> optsC;
    optsC.reserve(optStrings.size());
    for (auto& s : optStrings) optsC.push_back(s.c_str());

    // Compile
    nres = nvrtcCompileProgram(prog, static_cast<int>(optsC.size()), optsC.data());

    // Collect log
    size_t logSize = 0;
    nvrtcGetProgramLogSize(prog, &logSize);
    std::string log;
    if (logSize > 1) {
        log.resize(logSize);
        nvrtcGetProgramLog(prog, log.data());
    }
    if (!nvrtcCheck(nres, "nvrtcCompileProgram", log)) {
        nvrtcDestroyProgram(&prog);
        return false;
    }

    // Get PTX
    size_t ptxSize = 0;
    nvrtcGetPTXSize(prog, &ptxSize);
    result.ptx.resize(ptxSize);
    nvrtcGetPTX(prog, result.ptx.data());
    nvrtcDestroyProgram(&prog);

    // Load module and function
    if (!cuCheck(cuModuleLoadDataEx(&result.module, result.ptx.c_str(), 0, nullptr, nullptr),
                 "cuModuleLoadDataEx")) {
        return false;
    }
    if (!cuCheck(cuModuleGetFunction(&result.function, result.module, entryPoint.c_str()),
                 "cuModuleGetFunction")) {
        cuModuleUnload(result.module);
        result.module = nullptr;
        return false;
    }
    return true;
}

// ================= Uniform handling =================
bool CudaExecutor::processMetadataUniforms(const TaskData& task, std::vector<CudaExecutor::UniformValue>& uniforms) {
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

    // Preferred ordering for common fields
    std::vector<std::string> fieldOrder = {
        "block_size", "matrix_size", "matrix_n", "matrixSize",
        "tile_start_row", "tile_start_col", "tile_rows", "tile_cols",
        "tile_size", "tileSize"
    };

    auto pushUniform = [&](const std::string& key, const json& v, const char* originTag) {
        if (!v.is_number()) return;
        UniformValue uv;
        uv.name = key;
        if (v.is_number_float()) {
            uv.type = UniformType::FLOAT;   uv.floatValue = jsonToFloat(v);
        } else if (v.is_number_unsigned()) {
            uv.type = UniformType::UINT32;  uv.uintValue  = jsonToUnsigned(v);
        } else { // integer
            uv.type = UniformType::INT32;   uv.intValue   = jsonToInt(v);
        }
        uniforms.push_back(uv);
        std::cout << "  " << key << " = " << getUniformValueStr(uv)
                  << " (" << originTag << ")" << std::endl;
    };

    for (const auto& field : fieldOrder) {
        if (sourceData->contains(field)) {
            pushUniform(field, (*sourceData)[field], sourceType.c_str());
        }
    }

    // Append any remaining numeric fields not already added
    for (auto it = sourceData->begin(); it != sourceData->end(); ++it) {
        if (!it.value().is_number()) continue;
        if (std::find(fieldOrder.begin(), fieldOrder.end(), it.key()) != fieldOrder.end()) continue;
        pushUniform(it.key(), it.value(), ("extra " + sourceType).c_str());
    }
    return true;
}

void CudaExecutor::addUniformsToKernelArgs(const std::vector<UniformValue>& uniforms,
                                           std::vector<void*>& kernelArgs,
                                           std::vector<int32_t>& intStorage,
                                           std::vector<uint32_t>& uintStorage,
                                           std::vector<float>& floatStorage) {
    for (const auto& u : uniforms) {
        switch (u.type) {
            case UniformType::INT32:
                intStorage.push_back(u.intValue);
                kernelArgs.push_back(&intStorage.back());
                break;
            case UniformType::UINT32:
                uintStorage.push_back(u.uintValue);
                kernelArgs.push_back(&uintStorage.back());
                break;
            case UniformType::FLOAT:
                floatStorage.push_back(u.floatValue);
                kernelArgs.push_back(&floatStorage.back());
                break;
        }
    }
}

// ================= Task execution =================
TaskResult CudaExecutor::executeTask(const TaskData& task) {
    TaskResult result;
    auto startTime = std::chrono::high_resolution_clock::now();

    if (!initialized) {
        result.success = false;
        result.errorMessage = "CUDA not initialized";
        return result;
    }
    if (!ensureContextCurrent()) {
        result.success = false;
        result.errorMessage = "Failed to make CUDA context current";
        return result;
    }

    try {
        // ---- Compile or retrieve kernel (ASSUMES CUDA C++ IN task.kernel) ----
        const std::string& src = task.kernel;   // authoritative: kernel is CUDA code
        const std::string& entry = task.entry.empty() ? std::string("kernel") : task.entry;

        std::string cacheKey = makeCacheKey(src, entry);
        CompiledKernel* kfun = nullptr;

        auto it = kernelCache.find(cacheKey);
        if (it == kernelCache.end()) {
            CompiledKernel ck;
            if (!compileKernel(src, entry, task.compilationOptions, ck)) {
                result.success = false;
                result.errorMessage = "Kernel compilation failed";
                return result;
            }
            auto [insIt, ok] = kernelCache.emplace(cacheKey, std::move(ck));
            kfun = &insIt->second;
        } else {
            kfun = &it->second;
        }

        // ---- Determine input/output counts ----
        const bool useMultipleInputs  = task.hasMultipleInputs();
        const bool useMultipleOutputs = task.hasMultipleOutputs();
        const bool hasUniforms = !task.metadata.empty() || !task.chunkUniforms.empty();

        const size_t inputCount  = useMultipleInputs  ? task.inputData.size()
                               : (!task.legacyInputData.empty() ? 1 : 0);
        const size_t outputCount = useMultipleOutputs ? task.outputSizes.size()
                               : (task.legacyOutputSize ? 1 : 0);

        std::cout << "CUDA Task " << task.id << " - Inputs: " << inputCount
                  << ", Outputs: " << outputCount
                  << ", Has uniforms: " << (hasUniforms ? 1 : 0) << std::endl;

        // Debug: first 4 floats of inputs (if present)
        if (useMultipleInputs) {
            for (size_t i = 0; i < task.inputData.size(); i++) {
                const auto& inputData = task.inputData[i];
                if (!inputData.empty() && inputData.size() >= 16) {
                    const float* f = reinterpret_cast<const float*>(inputData.data());
                    std::cout << "Input " << i << " first 4 values: ";
                    for (int j = 0; j < 4 && j < static_cast<int>(inputData.size()/4); j++) {
                        std::cout << f[j] << " ";
                    }
                    std::cout << std::endl;
                }
            }
        } else if (!task.legacyInputData.empty() && task.legacyInputData.size() >= 16) {
            const float* f = reinterpret_cast<const float*>(task.legacyInputData.data());
            std::cout << "Legacy input first 4 values: ";
            for (int j = 0; j < 4 && j < static_cast<int>(task.legacyInputData.size()/4); j++) {
                std::cout << f[j] << " ";
            }
            std::cout << std::endl;
        }

        // ---- Build uniforms (first in the kernel arg list) ----
        std::vector<void*>   kernelArgs;
        std::vector<int32_t> intStorage;
        std::vector<uint32_t> uintStorage;
        std::vector<float>   floatStorage;
        std::vector<UniformValue> uniforms;

        if (hasUniforms) {
            processMetadataUniforms(task, uniforms);
            addUniformsToKernelArgs(uniforms, kernelArgs, intStorage, uintStorage, floatStorage);
            std::cout << "Set " << uniforms.size() << " uniform arguments" << std::endl;
        }

        // ---- Allocate/upload inputs (use Driver API; no mixing with runtime) ----
        std::vector<CUdeviceptr> dInputs;
        std::vector<CUdeviceptr> dOutputs;
        dInputs.reserve(inputCount);
        dOutputs.reserve(outputCount);

        auto freeAll = [&](void){
            for (auto d : dInputs)  if (d) cuMemFree(d);
            for (auto d : dOutputs) if (d) cuMemFree(d);
        };

        if (useMultipleInputs) {
            for (size_t i = 0; i < task.inputData.size(); ++i) {
                const auto& host = task.inputData[i];
                CUdeviceptr d = 0;
                size_t bytes = host.size();
                if (bytes > 0) {
                    if (!cuCheck(cuMemAlloc(&d, bytes), "cuMemAlloc(input)")) {
                        freeAll();
                        result.success = false;
                        result.errorMessage = "Failed to allocate input memory for input " + std::to_string(i);
                        return result;
                    }
                    if (!cuCheck(cuMemcpyHtoD(d, host.data(), bytes), "cuMemcpyHtoD(input)")) {
                        cuMemFree(d);
                        freeAll();
                        result.success = false;
                        result.errorMessage = "Failed to copy input data for input " + std::to_string(i);
                        return result;
                    }
                    std::cout << "Input " << i << ": " << bytes << " bytes" << std::endl;
                }
                dInputs.push_back(d); // can be 0 if empty buffer
            }
        } else if (!task.legacyInputData.empty()) {
            CUdeviceptr d = 0;
            size_t bytes = task.legacyInputData.size();
            if (!cuCheck(cuMemAlloc(&d, bytes), "cuMemAlloc(legacy input)")) {
                result.success = false;
                result.errorMessage = "Failed to allocate input memory";
                return result;
            }
            if (!cuCheck(cuMemcpyHtoD(d, task.legacyInputData.data(), bytes), "cuMemcpyHtoD(legacy input)")) {
                cuMemFree(d);
                result.success = false;
                result.errorMessage = "Failed to copy input data";
                return result;
            }
            std::cout << "Legacy input: " << bytes << " bytes" << std::endl;
            dInputs.push_back(d);
        }

        // ---- Allocate outputs ----
        if (useMultipleOutputs) {
            for (size_t i = 0; i < task.outputSizes.size(); ++i) {
                size_t bytes = task.outputSizes[i];
                if (bytes == 0) {
                    freeAll();
                    result.success = false;
                    result.errorMessage = "Output size is zero for output " + std::to_string(i);
                    return result;
                }
                CUdeviceptr d = 0;
                if (!cuCheck(cuMemAlloc(&d, bytes), "cuMemAlloc(output)")) {
                    freeAll();
                    result.success = false;
                    result.errorMessage = "Failed to allocate output memory for output " + std::to_string(i);
                    return result;
                }
                dOutputs.push_back(d);
                std::cout << "Output " << i << ": " << bytes << " bytes" << std::endl;
            }
        } else {
            if (task.legacyOutputSize == 0) {
                freeAll();
                result.success = false;
                result.errorMessage = "Legacy output size is zero";
                return result;
            }
            CUdeviceptr d = 0;
            if (!cuCheck(cuMemAlloc(&d, task.legacyOutputSize), "cuMemAlloc(legacy output)")) {
                freeAll();
                result.success = false;
                result.errorMessage = "Failed to allocate output memory";
                return result;
            }
            dOutputs.push_back(d);
            std::cout << "Legacy output: " << task.legacyOutputSize << " bytes" << std::endl;
        }

        // ---- Push inputs/outputs to kernel args AFTER vectors are fully populated ----
        for (auto& d : dInputs)  kernelArgs.push_back(&d);   // pass ADDRESS of CUdeviceptr
        for (auto& d : dOutputs) kernelArgs.push_back(&d);   // same for outputs

        // ---- Resolve grid/block from metadata (fallbacks preserved) ----
        std::vector<int> gridDim  = task.workgroupCount.empty() ? std::vector<int>{1, 1, 1} : task.workgroupCount;
        std::vector<int> blockDim = {16, 16, 1};  // default

        if (task.metadata.contains("blockDim") && task.metadata["blockDim"].is_array()) {
            auto bdim = task.metadata["blockDim"].get<std::vector<int>>();
            if (bdim.size() >= 3) blockDim = bdim;
            std::cout << "Using metadata blockDim: " << blockDim[0] << "," << blockDim[1] << "," << blockDim[2] << std::endl;
        }
        if (task.metadata.contains("gridDim") && task.metadata["gridDim"].is_array()) {
            auto gdim = task.metadata["gridDim"].get<std::vector<int>>();
            if (gdim.size() >= 3) gridDim = gdim;
            std::cout << "Using metadata gridDim: " << gridDim[0] << "," << gridDim[1] << "," << gridDim[2] << std::endl;
        }

        std::cout << "Block size from uniforms: ";
        bool printed = false;
        for (const auto& u : uniforms) {
            if (u.name == "block_size") {
                std::cout << (u.type == UniformType::FLOAT ? u.floatValue
                                 : (u.type == UniformType::UINT32 ? static_cast<float>(u.uintValue)
                                                                  : static_cast<float>(u.intValue)));
                printed = true;
                break;
            }
        }
        if (!printed) std::cout << "(not provided)";
        std::cout << std::endl;

        std::cout << "Final kernel launch parameters:\n";
        std::cout << "  Grid: ("  << gridDim[0]  << "," << gridDim[1]  << "," << gridDim[2]  << ")\n";
        std::cout << "  Block: (" << blockDim[0] << "," << blockDim[1] << "," << blockDim[2] << ")\n";
        std::cout << "  Total threads: " << (gridDim[0] * blockDim[0]) << "x" << (gridDim[1] * blockDim[1]) << std::endl;
        std::cout << "Total kernel arguments: " << kernelArgs.size()
                  << " (uniforms:" << uniforms.size()
                  << " + inputs:" << dInputs.size()
                  << " + outputs:" << dOutputs.size() << ")" << std::endl;

        // ---- Launch ----
        CUresult launchResult = cuLaunchKernel(
            kfun->function,
            gridDim[0], gridDim[1], gridDim[2],
            blockDim[0], blockDim[1], blockDim[2],
            0, nullptr,
            kernelArgs.data(), nullptr
        );
        if (launchResult != CUDA_SUCCESS) {
            logCuError("cuLaunchKernel", launchResult);
            freeAll();
            result.success = false;
            result.errorMessage = "Kernel launch failed";
            return result;
        }

        // ---- Sync ----
        if (!cuCheck(cuCtxSynchronize(), "cuCtxSynchronize")) {
            freeAll();
            result.success = false;
            result.errorMessage = "Kernel execution failed";
            return result;
        }

        // ---- Download outputs ----
        if (useMultipleOutputs) {
            result.outputData.resize(task.outputSizes.size());
            for (size_t i = 0; i < task.outputSizes.size(); ++i) {
                size_t bytes = task.outputSizes[i];
                result.outputData[i].resize(bytes);
                if (!cuCheck(cuMemcpyDtoH(result.outputData[i].data(), dOutputs[i], bytes),
                             "cuMemcpyDtoH(output)")) {
                    freeAll();
                    result.success = false;
                    result.errorMessage = "Failed to copy output data for output " + std::to_string(i);
                    return result;
                }
                std::cout << "Retrieved output " << i << ": " << bytes << " bytes" << std::endl;
            }
            // Back-compat: set legacyOutputData to first output, if any
            if (!result.outputData.empty()) result.legacyOutputData = result.outputData[0];
        } else {
            result.legacyOutputData.resize(task.legacyOutputSize);
            if (!cuCheck(cuMemcpyDtoH(result.legacyOutputData.data(), dOutputs[0], task.legacyOutputSize),
                         "cuMemcpyDtoH(legacy output)")) {
                freeAll();
                result.success = false;
                result.errorMessage = "Failed to copy legacy output data";
                return result;
            }
            result.outputData = { result.legacyOutputData };
            std::cout << "Retrieved legacy output: " << task.legacyOutputSize << " bytes" << std::endl;
        }

        // ---- Free device memory ----
        freeAll();

        auto endTime = std::chrono::high_resolution_clock::now();
        result.processingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        result.success = true;
        std::cout << "CUDA Task " << task.id << " completed successfully: "
                  << result.outputData.size() << " outputs in "
                  << result.processingTime << " ms" << std::endl;

    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("Exception: ") + e.what();
        std::cerr << "CUDA Task " << task.id << " failed with exception: " << e.what() << std::endl;
    }

    return result;
}

// Capabilities
json CudaExecutor::getCapabilities() const {
    json caps;
    caps["framework"] = "cuda";
    caps["initialized"] = initialized;
    caps["supportsMultiInput"]  = true;
    caps["supportsMultiOutput"] = true;
    caps["maxInputs"]  = 8;
    caps["maxOutputs"] = 4;
    caps["supportsUniforms"] = true;
    caps["taskAgnostic"] = true;
    caps["expectsKernelIn"] = "kernel";   // important: CUDA source is in `kernel`

    if (initialized) {
        caps["device"] = {
            {"id", deviceId},
            {"name", deviceName},
            {"computeCapability", {computeMajor, computeMinor}},
            {"globalMemory", totalGlobalMem},
            {"maxThreadsPerBlock", maxThreadsPerBlock},
            {"multiProcessorCount", multiProcessorCount}
        };
    }
    return caps;
}
