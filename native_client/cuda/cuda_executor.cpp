// cuda_executor.cpp
#include "cuda_executor.hpp"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <sstream>
#include <set>

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
static inline std::string getUniformValueStr(const CudaExecutor::UniformValue& uv) {
    using UT = CudaExecutor::UniformType;
    switch (uv.type) {
        case UT::FLOAT32: return std::to_string(uv.f32);
        case UT::FLOAT64: return std::to_string(uv.f64);
        case UT::UINT32:  return std::to_string(uv.u32);
        case UT::UINT64:  return std::to_string(uv.u64);
        case UT::INT32:   return std::to_string(uv.i32);
        case UT::INT64:   return std::to_string(uv.i64);
    }
    return "?";
}
static inline CudaExecutor::UniformType inferTypeFromJson(const json& v) {
    using UT = CudaExecutor::UniformType;
    if (v.is_number_float()) return UT::FLOAT32; // default f32 unless schema says f64
    if (v.is_number_unsigned()) {
        uint64_t u = v.get<uint64_t>();
        return (u <= 0xFFFFFFFFULL) ? UT::UINT32 : UT::UINT64;
    }
    if (v.is_number_integer()) {
        int64_t i = v.get<int64_t>();
        return (i >= INT32_MIN && i <= INT32_MAX) ? UT::INT32 : UT::INT64;
    }
    // default to int32 if not numeric (shouldn't happen, caller filters)
    return UT::INT32;
}
static inline bool isLikelyNonUniformKey(const std::string& k) {
    static const std::set<std::string> skip = {
        "blockDim","gridDim","workgroupCount",
        "inputs","outputs","outputSizes","outputSize",
        "kernel","wgsl","entry","webglShaderType","webglVertexShader","webglFragmentShader",
        "webglVaryings","webglNumElements","webglInputSpec",
        "globalWorkSize","localWorkSize",
        "openclKernel","vulkanShader","computeShader","cudaKernel",
        "chunkingStrategy","assemblyStrategy","schema"
    };
    return skip.count(k) > 0;
}
static inline CudaExecutor::UniformType parseTypeString(const std::string& s) {
    using UT = CudaExecutor::UniformType;
    if (s == "int32"  || s == "i32") return UT::INT32;
    if (s == "uint32" || s == "u32") return UT::UINT32;
    if (s == "float32"|| s == "f32") return UT::FLOAT32;
    if (s == "int64"  || s == "i64") return UT::INT64;
    if (s == "uint64" || s == "u64") return UT::UINT64;
    if (s == "float64"|| s == "f64" || s == "double") return UT::FLOAT64;
    // default
    return UT::INT32;
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
    for (auto& kv : kernelCache) {
        if (kv.second.module) cuModuleUnload(kv.second.module);
    }
    kernelCache.clear();
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

void CudaExecutor::cleanup() {
    for (auto& kv : kernelCache) {
        if (kv.second.module) cuModuleUnload(kv.second.module);
    }
    kernelCache.clear();
}

// ================= NVRTC compilation =================
bool CudaExecutor::compileKernel(const std::string& source,
                                 const std::string& entryPoint,
                                 const json& compileOpts,
                                 CompiledKernel& result) {
    nvrtcProgram prog = nullptr;
    nvrtcResult nres = nvrtcCreateProgram(&prog, source.c_str(), "kernel.cu", 0, nullptr, nullptr);
    if (!nvrtcCheck(nres, "nvrtcCreateProgram")) return false;

    std::vector<std::string> optStrings;
    {
        std::ostringstream arch;
        arch << "--gpu-architecture=compute_" << computeMajor << computeMinor;
        optStrings.push_back(arch.str());
    }
    optStrings.push_back("--std=c++14");

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

    nres = nvrtcCompileProgram(prog, static_cast<int>(optsC.size()), optsC.data());

    size_t logSize = 0;
    nvrtcGetProgramLogSize(prog, &logSize);
    std::string log;
    if (logSize > 1) { log.resize(logSize); nvrtcGetProgramLog(prog, log.data()); }
    if (!nvrtcCheck(nres, "nvrtcCompileProgram", log)) {
        nvrtcDestroyProgram(&prog);
        return false;
    }

    size_t ptxSize = 0;
    nvrtcGetPTXSize(prog, &ptxSize);
    result.ptx.resize(ptxSize);
    nvrtcGetPTX(prog, result.ptx.data());
    nvrtcDestroyProgram(&prog);

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

// ================= Kernel-agnostic uniform handling =================
//
// Priority:
// 1) metadata.schema.uniforms: array of { name, type }  --> use this order & types
// 2) metadata.uniformOrder:    array of names           --> use this order (infer types)
// 3) Fallback: iterate metadata's numeric fields in insertion order,
//    excluding non-uniform keys.
//
bool CudaExecutor::buildUniformList(const TaskData& task, std::vector<UniformValue>& uniforms) {
    const json& md = task.metadata;
    uniforms.clear();

    auto pushUniform = [&](const std::string& name, UniformType t, const json* val) {
        UniformValue uv; uv.name = name; uv.type = t;
        if (val && val->is_number()) {
            switch (t) {
                case UniformType::INT32:   uv.i32 = static_cast<int32_t>(val->get<int64_t>()); break;
                case UniformType::UINT32:  uv.u32 = static_cast<uint32_t>(val->get<uint64_t>()); break;
                case UniformType::FLOAT32: uv.f32 = static_cast<float>(val->is_number_float() ? val->get<double>() : static_cast<double>(val->get<int64_t>())); break;
                case UniformType::INT64:   uv.i64 = val->get<int64_t>(); break;
                case UniformType::UINT64:  uv.u64 = val->get<uint64_t>(); break;
                case UniformType::FLOAT64: uv.f64 = val->is_number_float() ? val->get<double>() : static_cast<double>(val->get<int64_t>()); break;
            }
        }
        uniforms.push_back(uv);
    };

    // 1) schema.uniforms
    if (md.contains("schema") && md["schema"].is_object()) {
        const json& sch = md["schema"];
        if (sch.contains("uniforms") && sch["uniforms"].is_array()) {
            for (const auto& u : sch["uniforms"]) {
                if (!u.is_object()) continue;
                if (!u.contains("name")) continue;
                std::string name = u["name"].get<std::string>();
                UniformType t = UniformType::INT32;
                if (u.contains("type") && u["type"].is_string()) {
                    t = parseTypeString(u["type"].get<std::string>());
                } else if (md.contains(name)) {
                    t = inferTypeFromJson(md[name]);
                }
                const json* vptr = md.contains(name) ? &md[name] : nullptr;
                pushUniform(name, t, vptr);
                std::cout << "  uniform (schema) " << name << " = "
                          << (vptr ? getUniformValueStr(uniforms.back()) : std::string("<unset>"))
                          << " (" << (u.contains("type") ? u["type"].get<std::string>() : "inferred") << ")"
                          << std::endl;
            }
            return true;
        }
    }

    // 2) uniformOrder
    if (md.contains("uniformOrder") && md["uniformOrder"].is_array()) {
        for (const auto& n : md["uniformOrder"]) {
            if (!n.is_string()) continue;
            std::string name = n.get<std::string>();
            const json* vptr = md.contains(name) ? &md[name] : nullptr;
            UniformType t = vptr ? inferTypeFromJson(*vptr) : UniformType::INT32;
            pushUniform(name, t, vptr);
            std::cout << "  uniform (order) " << name << " = "
                      << (vptr ? getUniformValueStr(uniforms.back()) : std::string("<unset>"))
                      << " (inferred)" << std::endl;
        }
        return true;
    }

    // 3) Fallback: insertion-order numeric fields, excluding known non-uniform keys
    for (auto it = md.begin(); it != md.end(); ++it) {
        if (isLikelyNonUniformKey(it.key())) continue;
        if (!it.value().is_number()) continue;
        UniformType t = inferTypeFromJson(it.value());
        pushUniform(it.key(), t, &it.value());
        std::cout << "  uniform (fallback) " << it.key() << " = "
                  << getUniformValueStr(uniforms.back()) << " (inferred)" << std::endl;
    }
    return true;
}

// Reallocation-safe: reserve before taking addresses
void CudaExecutor::addUniformsToKernelArgs(const std::vector<UniformValue>& uniforms,
                                           std::vector<void*>& kernelArgs,
                                           std::vector<int32_t>&  i32Store,
                                           std::vector<uint32_t>& u32Store,
                                           std::vector<float>&    f32Store,
                                           std::vector<int64_t>&  i64Store,
                                           std::vector<uint64_t>& u64Store,
                                           std::vector<double>&   f64Store) {
    using UT = UniformType;
    size_t nI32=0,nU32=0,nF32=0,nI64=0,nU64=0,nF64=0;
    for (const auto& u : uniforms) {
        switch (u.type) {
            case UT::INT32:   ++nI32; break;
            case UT::UINT32:  ++nU32; break;
            case UT::FLOAT32: ++nF32; break;
            case UT::INT64:   ++nI64; break;
            case UT::UINT64:  ++nU64; break;
            case UT::FLOAT64: ++nF64; break;
        }
    }
    i32Store.reserve(i32Store.size()+nI32);
    u32Store.reserve(u32Store.size()+nU32);
    f32Store.reserve(f32Store.size()+nF32);
    i64Store.reserve(i64Store.size()+nI64);
    u64Store.reserve(u64Store.size()+nU64);
    f64Store.reserve(f64Store.size()+nF64);

    for (const auto& u : uniforms) {
        switch (u.type) {
            case UT::INT32:   i32Store.push_back(u.i32); kernelArgs.push_back(&i32Store.back()); break;
            case UT::UINT32:  u32Store.push_back(u.u32); kernelArgs.push_back(&u32Store.back()); break;
            case UT::FLOAT32: f32Store.push_back(u.f32); kernelArgs.push_back(&f32Store.back()); break;
            case UT::INT64:   i64Store.push_back(u.i64); kernelArgs.push_back(&i64Store.back()); break;
            case UT::UINT64:  u64Store.push_back(u.u64); kernelArgs.push_back(&u64Store.back()); break;
            case UT::FLOAT64: f64Store.push_back(u.f64); kernelArgs.push_back(&f64Store.back()); break;
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
        // ---- Compile or retrieve kernel (task.kernel = CUDA C++) ----
        const std::string& src = task.kernel;
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

        // ---- Determine IO counts ----
        const bool useMultipleInputs  = task.hasMultipleInputs();
        const bool useMultipleOutputs = task.hasMultipleOutputs();

        const size_t inputCount  = useMultipleInputs  ? task.inputData.size()
                               : (!task.legacyInputData.empty() ? 1 : 0);
        const size_t outputCount = useMultipleOutputs ? task.outputSizes.size()
                               : (task.legacyOutputSize ? 1 : 0);

        // ---- Build uniforms (order according to schema/uniformOrder/fallback) ----
        std::vector<void*> kernelArgs;
        std::vector<int32_t>  i32Store;
        std::vector<uint32_t> u32Store;
        std::vector<float>    f32Store;
        std::vector<int64_t>  i64Store;
        std::vector<uint64_t> u64Store;
        std::vector<double>   f64Store;
        std::vector<UniformValue> uniforms;

        if (!task.metadata.empty() || !task.chunkUniforms.empty()) {
            // Prefer task.metadata if present; otherwise legacy chunkUniforms
            const bool fromMeta = !task.metadata.empty();
            const json& savedMeta = fromMeta ? task.metadata : task.chunkUniforms;
            if (fromMeta) {
                std::cout << "Processing uniforms from metadata..." << std::endl;
            } else {
                std::cout << "Processing uniforms from legacy chunk uniforms..." << std::endl;
            }

            // Build from metadata
            buildUniformList(task, uniforms);
            addUniformsToKernelArgs(uniforms, kernelArgs, i32Store, u32Store, f32Store, i64Store, u64Store, f64Store);
            std::cout << "Set " << uniforms.size() << " uniform arguments" << std::endl;
        } else {
            std::cout << "No uniforms to process" << std::endl;
        }

        // ---- Allocate/upload inputs ----
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
                dInputs.push_back(d);
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

        // ---- Append buffer args (after uniforms) ----
        for (auto& d : dInputs)  kernelArgs.push_back(&d);
        for (auto& d : dOutputs) kernelArgs.push_back(&d);

        // ---- Launch dims ----
        std::vector<int> gridDim  = task.workgroupCount.empty() ? std::vector<int>{1, 1, 1} : task.workgroupCount;
        std::vector<int> blockDim = {16, 16, 1};

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

        // ---- Logs ----
        std::cout << "Uniforms (" << uniforms.size() << "):" << std::endl;
        for (size_t i = 0; i < uniforms.size(); ++i) {
            std::cout << "  [" << i << "] " << uniforms[i].name << " = " << getUniformValueStr(uniforms[i]) << std::endl;
        }
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

// ================= Capabilities =================
json CudaExecutor::getCapabilities() const {
    json caps;
    caps["framework"] = "cuda";
    caps["initialized"] = initialized;
    caps["supportsMultiInput"]  = true;
    caps["supportsMultiOutput"] = true;
    caps["maxInputs"]  = 8;
    caps["maxOutputs"] = 4;
    caps["supportsUniforms"] = true;
    caps["uniformTypes"] = {"int32","uint32","float32","int64","uint64","float64"};
    caps["taskAgnostic"] = true;
    caps["expectsKernelIn"] = "kernel";

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
