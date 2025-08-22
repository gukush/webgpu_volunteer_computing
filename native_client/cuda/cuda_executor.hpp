#pragma once
#include "../common/framework_client.hpp"
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvrtc.h>
#include <vector>
#include <memory>
#include <map>
#include <string>


class CudaExecutor : public IFrameworkExecutor {
private:
    int deviceId = 0;
    CUcontext context = nullptr;
    bool initialized = false;

    // Basic device info (queried via CUDA Driver API)
    std::string deviceName;
    int computeMajor = 0;
    int computeMinor = 0;
    size_t totalGlobalMem = 0;
    int maxThreadsPerBlock = 0;
    int multiProcessorCount = 0;

    struct CompiledKernel {
        CUmodule module = nullptr;
        CUfunction function = nullptr;
        std::string ptx;
    };

    std::map<std::string, CompiledKernel> kernelCache;

    // Enhanced uniform support for task-agnostic operation
    enum class UniformType {
        INT32,
        UINT32,
        FLOAT
    };

    struct UniformValue {
        std::string name;
        UniformType type;
        // store as separate fields (avoid anonymous union pitfalls)
        int32_t  intValue   = 0;
        uint32_t uintValue  = 0;
        float    floatValue = 0.0f;
    };

    // ---- Internal helpers ----
    static void logCuError(const char* where, CUresult r);
    static bool cuCheck(CUresult r, const char* where);
    static bool nvrtcCheck(nvrtcResult r, const char* where, const std::string& log = {});

    bool ensureDriverLoaded();
    bool ensureContextCurrent();

    // Build a cache key from source+entry
    static std::string makeCacheKey(const std::string& source, const std::string& entry);

    // NVRTC compile -> PTX -> module+function
    bool compileKernel(const std::string& source,
                       const std::string& entryPoint,
                       const json& compileOpts,
                       CompiledKernel& result);

    // Task-agnostic metadata processing (uniforms)
    bool processMetadataUniforms(const TaskData& task, std::vector<UniformValue>& uniforms);
    void addUniformsToKernelArgs(const std::vector<UniformValue>& uniforms,
                                 std::vector<void*>& kernelArgs,
                                 std::vector<int32_t>& intStorage,
                                 std::vector<uint32_t>& uintStorage,
                                 std::vector<float>& floatStorage);

public:
    CudaExecutor(int deviceId = 0);
    ~CudaExecutor() override;

    bool initialize(const json& config = {}) override;
    // NOTE: cleanup() no longer destroys the CUDA context (kept across tasks)
    void cleanup() override;

    TaskResult executeTask(const TaskData& task) override;

    std::string getFrameworkName() const override { return "cuda"; }
    json getCapabilities() const override;
};